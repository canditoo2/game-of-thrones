"""Tests for the ONNX Runtime leaf backend.

Skipped when torch, onnx or onnxruntime are missing. Nothing here touches
``agot._core``: the backend is a pure function of its four numpy arguments, so
export, quantization, caching and the spec plumbing are all verifiable without
the engine.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from agot import cli  # noqa: E402
from agot.engine.game import GameConfig  # noqa: E402
from agot.engine.observation import ACTION_DIM, N_HOUSES, STATE_DIM  # noqa: E402
from agot.nn import onnx_backend  # noqa: E402
from agot.nn.model import PolicyValueNet  # noqa: E402
from agot.training import rust_engine  # noqa: E402
from agot.training.gate import GateSpec  # noqa: E402
from agot.training.parallel import SelfPlayPlan  # noqa: E402

#: A handful of uneven candidate sets, because the real driver never pads.
WIDTHS = (4, 7, 2, 15, 11, 3)


def _model(*, hidden: int = 32, value_context: bool = True, seed: int = 0, sharpness=40.0):
    """A small net whose policy logits have a trained checkpoint's spread.

    A randomly initialised scorer emits logits with a standard deviation of
    ~0.06, i.e. a near-uniform policy, where int8 rounding is almost exactly
    harmless and a KL assertion would prove nothing. Multiplying the last
    scorer layer by ``sharpness`` reproduces the ~2.5 spread a trained net has
    at these shapes, which is the regime the 5e-3 nat budget is about.
    """
    torch.manual_seed(seed)
    model = PolicyValueNet(
        hidden=hidden, trunk_layers=1, value_context=value_context
    ).eval()
    if sharpness != 1.0:
        with torch.no_grad():
            model.policy_head[2].weight.mul_(sharpness)
    return model


def _batch(seed: int = 0, widths=WIDTHS):
    """One packed leaf batch: random state rows and uneven action segments."""
    rng = np.random.default_rng(seed)
    leaves = len(widths)
    state = rng.standard_normal((leaves, STATE_DIM)).astype(np.float32)
    index = np.concatenate(
        [np.full(width, row, dtype=np.int64) for row, width in enumerate(widths)]
    )
    actions = rng.standard_normal((index.size, ACTION_DIM)).astype(np.float32)
    mask = np.ones(index.size, dtype=np.float32)
    return state, actions, index, mask


def _reference(model, state, actions, index):
    with torch.inference_mode():
        logits, values = model.forward_packed(
            torch.from_numpy(state), torch.from_numpy(actions), torch.from_numpy(index)
        )
    return logits.numpy(), values.numpy()


def _leaf_kl(reference: np.ndarray, candidate: np.ndarray, widths) -> np.ndarray:
    """KL per leaf, since the packed layout has a different width per segment."""
    out = []
    offset = 0
    for width in widths:
        window = slice(offset, offset + width)
        out.append(float(onnx_backend._policy_kl(reference[window], candidate[window], width)[0]))
        offset += width
    return np.asarray(out)


def _empty_result():
    """The driver's packed buffers for a match with nothing recorded."""
    return (
        np.zeros((0, STATE_DIM), dtype=np.float32),
        np.zeros((0, ACTION_DIM), dtype=np.float32),
        np.array([0]),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.uint8),
        np.zeros((0, N_HOUSES), dtype=np.float32),
        ["stark"],
        [1],
        np.array([0]),
        [1],
    )


def test_export_and_load_round_trip(tmp_path) -> None:
    """The exported graph must be the packed forward, to float32 precision."""
    model = _model(hidden=16, sharpness=1.0)
    backend = onnx_backend.LeafBackend.from_model(model, "onnx-fp32", cache_dir=tmp_path)
    assert backend.rebuilt, "a cold cache must export"
    assert backend.path.is_file()
    assert backend.path.with_name(backend.path.name + ".json").is_file()

    state, actions, index, mask = _batch()
    logits, values = backend(state, actions, index, mask)
    want_logits, want_values = _reference(model, state, actions, index)
    assert logits.shape == (index.size,)
    assert values.shape == (len(WIDTHS), N_HOUSES)
    np.testing.assert_allclose(logits, want_logits, atol=1e-4)
    np.testing.assert_allclose(values, want_values, atol=1e-5)

    # The dynamic axes are real: another leaf count and another width per leaf
    # must run on the same graph without a re-export.
    other = _batch(seed=5, widths=(1, 9, 2))
    other_state, other_actions, other_index, _ = other
    logits, values = backend(*other)
    want_logits, want_values = _reference(model, other_state, other_actions, other_index)
    np.testing.assert_allclose(logits, want_logits, atol=1e-4)
    np.testing.assert_allclose(values, want_values, atol=1e-5)

    # A second visitor reads the cache instead of exporting again.
    reused = onnx_backend.LeafBackend.from_model(model, "onnx-fp32", cache_dir=tmp_path)
    assert reused.rebuilt is False
    assert reused.path == backend.path


def test_int8_output_shape_dtype_and_kernels(tmp_path) -> None:
    """int8 must keep the driver's contract and really hold int8 weights."""
    model = _model(hidden=16)
    fp32 = onnx_backend.LeafBackend.from_model(model, "onnx-fp32", cache_dir=tmp_path)
    int8 = onnx_backend.LeafBackend.from_model(model, "onnx-int8", cache_dir=tmp_path)
    assert int8.path != fp32.path

    logits, values = int8(*_batch())
    assert logits.dtype == np.float32 and values.dtype == np.float32
    assert logits.shape == (sum(WIDTHS),)
    assert values.shape == (len(WIDTHS), N_HOUSES)
    assert np.isfinite(logits).all() and np.isfinite(values).all()
    assert (np.abs(values) <= 1.0).all()

    graph = onnx.load(str(int8.path))
    kinds = {node.op_type for node in graph.graph.node}
    quantized = kinds & {"MatMulInteger", "QLinearMatMul", "MatMulIntegerToFloat"}
    assert quantized, f"the linear layers were not quantized (ops: {sorted(kinds)})"
    # Version-agnostic proof that the weights themselves are int8: the dynamic
    # quantizer writes them as int8 initializers (QOperator or QDQ layout).
    int8_weights = [
        tensor
        for tensor in graph.graph.initializer
        if tensor.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8)
    ]
    assert int8_weights, "the graph has no int8 weight initializer"
    # And the file is genuinely smaller than the float32 graph it came from.
    assert int8.path.stat().st_size < fp32.path.stat().st_size


def test_int8_policy_kl_stays_inside_the_budget(tmp_path) -> None:
    """int8 perturbs the policy by far less than the search's own noise.

    The budget is 5e-3 nats at p95; the measured value on this model is ~1e-3
    (mean ~4e-4), which is two orders of magnitude under the PIMC noise of the
    visit distribution these logits feed.
    """
    model = _model(hidden=32)
    backend = onnx_backend.LeafBackend.from_model(model, "onnx-int8", cache_dir=tmp_path)
    kls = []
    for seed in range(8):
        widths = (2 + seed % 5, 15 - seed % 4, 1 + seed % 7, 6, 9, 3)
        state, actions, index, mask = _batch(seed=seed, widths=widths)
        logits, _ = backend(state, actions, index, mask)
        want_logits, _ = _reference(model, state, actions, index)
        kls.append(_leaf_kl(want_logits, logits, widths))
    kl = np.concatenate(kls)
    assert kl.mean() < 1e-3, f"mean KL {kl.mean():.2e} nats is too high for int8"
    assert np.percentile(kl, 95) < 5e-3, f"p95 KL {np.percentile(kl, 95):.2e} nats"


def test_a_stale_graph_is_refused(tmp_path) -> None:
    """A cache that does not belong to the weights on disk must never be used."""
    model = _model(hidden=16, sharpness=1.0)
    checkpoint = tmp_path / "agot.pt"
    model.save(str(checkpoint))
    assert model.checkpoint == str(checkpoint)

    first = onnx_backend.ensure_graph(model, "onnx-fp32", checkpoint=checkpoint)
    assert first.rebuilt
    assert first.path.parent == tmp_path, "the graph belongs beside its checkpoint"
    assert not onnx_backend.ensure_graph(
        model, "onnx-fp32", checkpoint=checkpoint
    ).rebuilt

    # Different weights, saved over the same file: the cached graph is now a lie.
    with torch.no_grad():
        model.policy_head[2].weight.add_(0.5)
    model.save(str(checkpoint))
    with pytest.raises(onnx_backend.StaleGraphError) as error:
        onnx_backend.ensure_graph(model, "onnx-fp32", checkpoint=checkpoint, rebuild=False)
    assert "weights" in str(error.value) and "checkpoint" in str(error.value)

    rebuilt = onnx_backend.ensure_graph(model, "onnx-fp32", checkpoint=checkpoint)
    assert rebuilt.rebuilt
    backend = onnx_backend.LeafBackend.from_model(model, "onnx-fp32", checkpoint=checkpoint)
    assert backend.rebuilt is False, "the fresh graph must now be reused"
    state, actions, index, _ = _batch()
    logits, _ = backend(state, actions, index)
    want_logits, _ = _reference(model, state, actions, index)
    np.testing.assert_allclose(logits, want_logits, atol=1e-4)


def test_a_missing_cache_is_built_even_without_rebuild(tmp_path) -> None:
    """``rebuild=False`` refuses a *stale* graph, not the first export."""
    cached = onnx_backend.ensure_graph(
        _model(hidden=8), "onnx-int8", cache_dir=tmp_path, rebuild=False
    )
    assert cached.rebuilt and cached.path.is_file()


def test_the_leaf_backend_round_trips_through_the_specs() -> None:
    """Workers, the gate and the CLI must agree on the same three names."""
    assert onnx_backend.BACKENDS == rust_engine.LEAF_BACKENDS == cli.LEAF_BACKENDS

    plan = SelfPlayPlan(games=2, leaf_backend="onnx-int8", leaf_checkpoint="run/agot.pt")
    spec = rust_engine.spec_for(plan)
    assert spec["leaf_backend"] == "onnx-int8"
    assert rust_engine.spec_for(plan, leaf_backend="onnx-fp32")["leaf_backend"] == (
        "onnx-fp32"
    )
    assert rust_engine.spec_for(SelfPlayPlan())["leaf_backend"] == "torch"

    gate = GateSpec(
        iteration=7,
        games=2,
        seed=0,
        simulations=1,
        determinizations=1,
        batch_games=1,
        threshold=0.5,
        best_checkpoint="run/agot-best.pt",
        leaf_backend="onnx-int8",
    )
    assert GateSpec.from_dict(gate.as_dict()).leaf_backend == "onnx-int8"
    # A spec written before the flag existed still loads, as torch.
    legacy = {
        key: value
        for key, value in gate.as_dict().items()
        if key != "leaf_backend"
    }
    assert GateSpec.from_dict(legacy).leaf_backend == "torch"


def test_the_driver_scores_through_the_chosen_backend(monkeypatch) -> None:
    """The spec and the leaf callback must carry the same engine."""
    captured: dict = {}

    class FakeExtension:
        def selfplay(self, spec, evaluator, decider=None, extra=None):
            captured["spec"] = spec
            captured["evaluator"] = evaluator
            return _empty_result()

    class DummyEvaluator:
        model = object()

    monkeypatch.setattr(rust_engine, "core", lambda: FakeExtension())
    monkeypatch.setattr(
        rust_engine,
        "leaf_evaluator",
        lambda model, backend="torch", **kwargs: ("leaf", backend),
    )
    records = rust_engine.try_batched_self_play(
        DummyEvaluator(),
        [GameConfig(seed=1, max_rounds=1)],
        simulations=1,
        leaf_backend="onnx-int8",
    )
    assert records is not None
    assert captured["spec"]["leaf_backend"] == "onnx-int8"
    assert captured["evaluator"] == ("leaf", "onnx-int8")


def test_a_leaf_backend_is_accepted_as_the_evaluator(tmp_path, monkeypatch) -> None:
    """A ``LeafBackend`` must not be replaced by a freshly built torch closure."""
    backend = onnx_backend.LeafBackend.from_model(
        _model(hidden=8), "onnx-fp32", cache_dir=tmp_path
    )
    captured: dict = {}

    class FakeExtension:
        def selfplay(self, spec, evaluator, decider=None, extra=None):
            captured["evaluator"] = evaluator
            return _empty_result()

    monkeypatch.setattr(rust_engine, "core", lambda: FakeExtension())
    records = rust_engine.try_batched_self_play(
        backend, [GameConfig(seed=1, max_rounds=1)], simulations=1
    )
    assert records is not None
    assert captured["evaluator"] is backend


def test_the_torch_backend_is_the_plain_forward() -> None:
    """The default engine must stay bit-identical to the untouched expression."""
    model = _model(hidden=16, sharpness=1.0)
    evaluate = rust_engine.leaf_evaluator(model)
    state, actions, index, mask = _batch()
    logits, values = evaluate(state, actions, index, mask)
    want_logits, want_values = _reference(model, state, actions, index)
    np.testing.assert_array_equal(logits, want_logits)
    np.testing.assert_array_equal(values, want_values)

    with pytest.raises(ValueError, match="unknown leaf backend"):
        rust_engine.leaf_evaluator(model, "onnx-fp16")


def test_auto_resolves_to_a_concrete_engine(monkeypatch) -> None:
    """``auto`` is a CLI alias: it must never reach a spec as-is."""
    monkeypatch.setattr(rust_engine, "onnx_runtime_available", lambda: True)
    assert rust_engine.resolve_leaf_backend("auto") == "onnx-fp32"
    monkeypatch.setattr(rust_engine, "onnx_runtime_available", lambda: False)
    assert rust_engine.resolve_leaf_backend("auto") == "torch"
    # Everything else passes through, and None keeps meaning "inherit".
    assert rust_engine.resolve_leaf_backend("torch") == "torch"
    assert rust_engine.resolve_leaf_backend("onnx-int8") == "onnx-int8"
    assert rust_engine.resolve_leaf_backend(None) is None


def test_the_cli_defaults_to_auto_and_keeps_the_concrete_names() -> None:
    args = cli.build_parser().parse_args(["train", "--checkpoint", "run/agot.pt"])
    assert args.leaf_backend == "auto"
    assert set(cli.LEAF_BACKENDS) == set(rust_engine.LEAF_BACKENDS)
    assert set(cli.LEAF_BACKEND_CHOICES) == {"auto", *cli.LEAF_BACKENDS}
