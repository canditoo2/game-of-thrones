"""ONNX Runtime leaf backend for the packed forward, with dynamic int8.

Why this exists
---------------
Self-play spends most of its Python time in one network call per *batch of
leaves*. At the production shape (22-24 leaves x 15-16 candidates, hidden 256)
the torch forward costs ~1.5 ms even now that ``forward_packed`` scores exactly
the candidates it needs, and the search is CPU-bound on a machine with no GPU.
ONNX Runtime over a dynamically quantized int8 copy of the same graph makes the
same call ~2.6x cheaper. The leaf callback stays a pure function of its inputs,
so the search is unchanged: int8 only perturbs the numbers, by a policy KL of
~5e-4 nats per leaf (p95 ~1.3e-3) on a net whose logits have a trained spread,
two orders of magnitude inside the PIMC noise floor of the visit distribution it
feeds.

What it takes to export the packed forward
------------------------------------------
* The **legacy** TorchScript exporter (``dynamo=False``). The ``torch.export``
  based path cannot infer shapes through the packed forward's
  ``state_part[index]`` gather and fails.
* The value-context pooling is rewritten as a **one-hot matmul**, because
  ``index_add_`` over a dynamically sized row count does not export. The one-hot
  matrix is exact, so the float32 graph agrees with torch to ~1e-8.
* Quantization is ORT's ``quantize_dynamic`` with ``per_channel=True`` and
  ``QInt8`` weights: every Linear becomes an int8 weight matmul, while the
  one-hot pooling and all activations stay float32.

The legacy exporter is deprecated as of torch 2.9 (the ``torch.export`` path is
the default there). ``PackedGraph`` deliberately contains no ``index_add_`` and
no data-dependent control flow, so when the legacy exporter goes away only the
``torch.onnx.export`` call here has to change — the graph does not.

Caching
-------
Exporting costs ~100-200 ms, which is too much to repeat for every worker and
every evaluation, so the graph is cached **beside its checkpoint**::

    <stem>.onnx-int8.onnx          the graph
    <stem>.onnx-int8.onnx.json     the cache key

The sidecar records the checkpoint file hash, a digest of the weights, the model
config, the backend and the graph revision. A cached graph is reused only when
*every* one of those still matches; otherwise it is stale. A stale graph is
never scored with: :func:`ensure_graph` rebuilds it (the default), or raises
:class:`StaleGraphError` when the caller passes ``rebuild=False``.

Compact graphs
--------------
On a trained net most trunk units never fire: on run7's replay buffer the max
pre-activation is <= 0 for 202 of 256 units of ``state_trunk[0]`` and 165-173
of the other three trunk layers. A ReLU unit that never fires contributes an
exact zero downstream, so :func:`calibrate_keep` measures, once per training
iteration, which trunk units fire on the replay buffer, and the exported graph
(``PackedGraph(model, keep)``) carries only those rows and columns. Training's
leaves get ~2.6x cheaper in float32, and the scores do not change on any leaf
where the dropped units stay silent, which was every one of 40k real search
leaves measured (docs/optimization/iterations/008-compact-leaf-graph.md).

Only the trunks are compacted: ``policy_head`` and ``value_head`` stay at full
width, because hidden units of the heads that never fire on the replay buffer
do fire on real leaves. The keep-sets join the cache key and the cached file
name (``<stem>.<backend>.compact.onnx``), so a graph built for other keep-sets,
or the full graph, is never reused for a compact request, and vice versa.
``keep=None`` exports exactly the full graph.

Requires the optional extra::

    pip install -e ".[onnx]"
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

#: Bump when the exported graph changes, so every cache entry is rebuilt.
GRAPH_REVISION = "packed-v2"
#: Opset of the exported graph. 17 carries everything the packed forward needs
#: (Range, Gather, MatMul) and is the newest opset ORT's quantizer handles well.
EXPORT_OPSET = 17
#: Every backend :func:`agot.training.rust_engine.leaf_evaluator` accepts.
BACKENDS = ("torch", "onnx-int8", "onnx-fp32")
#: The subset that runs through ONNX Runtime.
ONNX_BACKENDS = ("onnx-int8", "onnx-fp32")
#: Graph input and output names, in call order.
INPUT_NAMES = ("state", "actions", "index")
OUTPUT_NAMES = ("logits", "values")
#: The fields a cached graph must match to be considered fresh. Everything else
#: in the sidecar is provenance for a human reading it.
KEY_FIELDS = (
    "graph",
    "backend",
    "opset",
    "model_config",
    "weights",
    "checkpoint_sha256",
    "checkpoint_size",
    "keep",
)
#: Fewest replay examples training calibrates on before it uses a compact
#: graph. Measured on run7 (iteration 008): the most recent 33k examples (one
#: production iteration, 64 games) still missed trunk units that fire on real
#: search leaves (value drift up to 0.03-0.45); 45k and more missed none. A
#: production buffer (``--buffer 80000``) holds 60-80k examples from its second
#: iteration on, so this only keeps a fresh run's first iterations, and small
#: buffers, on the full graph.
MIN_CALIBRATION_EXAMPLES = 50_000
#: Rows per calibration matmul. Larger batches measured no faster.
CALIBRATION_CHUNK = 8192
#: Seed of the projection that groups candidate rows before calibration. The
#: grouping is verified bit for bit, so the seed only affects speed.
ROW_KEY_SEED = 0x5EED_CA1B


class StaleGraphError(RuntimeError):
    """A cached graph does not belong to the checkpoint or model asked for.

    Raised instead of scoring with the wrong weights. Callers that may
    regenerate (the training loop and every CLI path) keep the default
    ``rebuild=True`` and never see it.
    """


def _ort():
    try:
        import onnxruntime  # noqa: PLC0415 - optional dependency
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError(
            'the ONNX leaf backend needs onnxruntime: pip install -e ".[onnx]"'
        ) from None
    return onnxruntime


def _onnx():
    try:
        import onnx  # noqa: PLC0415 - optional dependency, quantization only
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError(
            'int8 quantization needs onnx: pip install -e ".[onnx]"'
        ) from None
    return onnx


def trunk_linears(model) -> list[nn.Linear]:
    """The trunks' Linear layers in keep-set order: state trunk, then action trunk.

    Every one of them is followed by a ReLU (``agot.nn.model._mlp``), which is
    what makes dropping a unit that never fires exact.
    """
    layers = []
    for trunk in (model.state_trunk, model.action_trunk):
        modules = list(trunk)
        for position, module in enumerate(modules):
            if isinstance(module, nn.Linear):
                follower = modules[position + 1] if position + 1 < len(modules) else None
                if not isinstance(follower, nn.ReLU):
                    raise ValueError("a trunk Linear is not followed by a ReLU")
                layers.append(module)
    return layers


def normalize_keep(model, keep) -> tuple[tuple[int, ...], ...] | None:
    """Validate keep-sets against ``model``; ``None`` means the full graph.

    ``keep`` holds one sequence of unit indices per entry of
    :func:`trunk_linears`. The result is sorted, de-duplicated tuples, so two
    equal keep-sets always produce the same cache key.
    """
    if keep is None:
        return None
    layers = trunk_linears(model)
    keep = tuple(tuple(sorted({int(unit) for unit in units})) for units in keep)
    if len(keep) != len(layers):
        raise ValueError(
            f"expected {len(layers)} keep-sets (one per trunk layer), got {len(keep)}"
        )
    for units, layer in zip(keep, layers, strict=True):
        if not units:
            raise ValueError("a keep-set must keep at least one unit")
        if units[0] < 0 or units[-1] >= layer.out_features:
            raise ValueError(
                f"keep-set index out of range for a layer of width {layer.out_features}"
            )
    return keep


def keep_widths(keep) -> list[int] | None:
    """How many units each trunk layer keeps (``None`` for the full graph)."""
    return None if keep is None else [len(units) for units in keep]


def _row_keys(rows: np.ndarray) -> np.ndarray:
    """A 64-bit grouping key per row: two random projections, bit-cast.

    Different rows may collide, and equal rows nearly always but not always
    share a key (the matmul may round one row differently depending on where it
    sits in the batch), so callers must verify (see :func:`distinct_rows`);
    this is a speed device only.
    """
    generator = torch.Generator().manual_seed(ROW_KEY_SEED)
    projection = torch.randn(rows.shape[1], 2, generator=generator, dtype=torch.float32)
    with torch.inference_mode():
        keys = torch.from_numpy(rows) @ projection
    return np.ascontiguousarray(keys.numpy()).view(np.uint64).reshape(len(rows))


def distinct_rows(rows: np.ndarray, *, chunk: int = 1 << 16) -> np.ndarray:
    """``rows`` with bitwise-identical repeats (nearly all) removed, in row order.

    Rows are grouped by :func:`_row_keys`; every row is then compared, bit for
    bit, with its group's first row, and any row that differs is kept as well.
    So every input row is either returned or bitwise equal to a returned row:
    a key collision can only add rows, never drop one, and a forward pass over
    the result reaches exactly the row values the full array holds. (A repeat
    whose key came out different is merely scored twice.) A replay
    buffer repeats candidate rows heavily (run7: 54k distinct of 1.29M), which
    is what makes this pay.
    """
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    if len(rows) < 2:
        return rows
    _, first, inverse = np.unique(_row_keys(rows), return_index=True, return_inverse=True)
    representative = torch.from_numpy(first[inverse.reshape(len(rows))])
    bits = torch.from_numpy(rows.view(np.int32))
    extra = []
    with torch.inference_mode():
        for start in range(0, len(rows), chunk):
            stop = min(len(rows), start + chunk)
            differs = torch.ne(
                bits[start:stop], bits.index_select(0, representative[start:stop])
            ).any(dim=1)
            if bool(differs.any()):
                extra.append(torch.nonzero(differs).flatten().numpy() + start)
    selected = np.concatenate([first, *extra]) if extra else first
    return rows[np.sort(selected)]


def calibrate_keep(
    model,
    states: np.ndarray,
    actions: np.ndarray,
    *,
    margin: float = 0.0,
    chunk: int = CALIBRATION_CHUNK,
    dedupe: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Keep-sets of the trunk ReLU units that fire on the calibration rows.

    ``states`` is ``(N, state_dim)`` and ``actions`` ``(T, action_dim)``, the
    replay buffer's packed layout; the trunks see states and candidates
    independently, so the segment index is not needed. A unit is kept iff its
    maximum pre-activation over all rows is ``> -margin`` (``margin >= 0``), so
    a unit that fires on any calibration row is never dropped. Each layer keeps
    at least its most active unit, so a graph never has an empty layer
    (keeping an extra unit is always exact).

    Only the trunk forward runs, with pre-transposed weights and large batches,
    and (``dedupe``, the default) only over the distinct rows: a unit's maximum
    over repeated rows is the maximum over one copy (:func:`distinct_rows`).
    run7 after one iteration (79k examples / 1.29M candidate rows, 54k
    distinct): 1.46 s without dedupe, ~0.45 s with it, on two threads.
    A margin is a poor safety lever on trained nets (iteration 008: the dead
    action-trunk units sit just below zero, so a margin of 0.05 already keeps
    138 instead of 83 units of ``action_trunk[0]``); coverage, i.e. enough
    examples, is what makes the keep-sets hold on search leaves.
    """
    if margin < 0:
        raise ValueError("the calibration margin must be non-negative")
    layers = trunk_linears(model)
    count = len(layers) // 2
    maxima: list[torch.Tensor] = []
    with torch.inference_mode():
        for trunk, rows in ((layers[:count], states), (layers[count:], actions)):
            weights = [
                (
                    layer.weight.detach().to("cpu", torch.float32).t().contiguous(),
                    layer.bias.detach().to("cpu", torch.float32),
                )
                for layer in trunk
            ]
            peaks = [torch.full((weight.shape[1],), -torch.inf) for weight, _ in weights]
            rows = np.asarray(rows, dtype=np.float32)
            if dedupe:
                rows = distinct_rows(rows)
            step = max(1, int(chunk))
            for start in range(0, len(rows), step):
                x = torch.from_numpy(np.ascontiguousarray(rows[start : start + step]))
                for position, (weight, bias) in enumerate(weights):
                    x = torch.addmm(bias, x, weight)
                    peaks[position] = torch.maximum(peaks[position], x.amax(dim=0))
                    x = x.clamp_min_(0.0)
            maxima.extend(peaks)
    keep = []
    for peak in maxima:
        units = torch.nonzero(peak > -float(margin)).flatten().tolist()
        if not units:
            units = [int(torch.argmax(peak))]
        keep.append(tuple(int(unit) for unit in units))
    return tuple(keep)


def _sliced_linear(layer: nn.Linear, rows=None, columns=None) -> nn.Linear:
    """A copy of ``layer`` restricted to output ``rows`` and input ``columns``."""
    weight = layer.weight.detach()
    bias = layer.bias.detach() if layer.bias is not None else None
    if rows is not None:
        weight = weight[rows]
        bias = None if bias is None else bias[rows]
    if columns is not None:
        weight = weight[:, columns]
    out = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        out.weight.copy_(weight)
        if bias is not None:
            out.bias.copy_(bias)
    return out.to(weight.device)


class PackedGraph(nn.Module):
    """``PolicyValueNet.forward_packed`` expressed in exportable ops.

    The arithmetic is identical to the torch method, except that the
    value-context pooling is ``one_hot(index)^T @ a`` instead of
    ``index_add_``, which the exporter cannot represent.
    ``tests/test_onnx_backend.py`` pins the two against each other, so they
    cannot drift apart silently.

    With ``keep`` (see :func:`calibrate_keep`) the trunks carry only the kept
    units: each trunk layer loses the dropped rows (and the columns of the
    units its predecessor dropped), the state and action halves of
    ``policy_head[0]`` and the inputs of ``value_merge`` lose the columns of
    the last trunk layers' dropped units, and both heads keep their full
    width. That is the full arithmetic whenever every dropped unit would have
    output zero, which is what calibration measures.
    """

    def __init__(self, model, keep=None) -> None:
        super().__init__()
        self.model = model
        self.keep = normalize_keep(model, keep)
        if self.keep is None:
            return
        hidden = int(model.hidden)
        layers = trunk_linears(model)
        count = len(layers) // 2
        index = [torch.tensor(units, dtype=torch.long) for units in self.keep]

        def trunk(linears, keeps):
            modules, previous = [], None
            for layer, rows in zip(linears, keeps, strict=True):
                modules += [_sliced_linear(layer, rows, previous), nn.ReLU()]
                previous = rows
            return nn.Sequential(*modules)

        self.state_trunk = trunk(layers[:count], index[:count])
        self.action_trunk = trunk(layers[count:], index[count:])
        last_state, last_action = index[count - 1], index[-1]
        first = model.policy_head[0]
        # The two halves of policy_head[0], as the full graph splits them.
        state_half = nn.Linear(hidden, hidden)
        action_half = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            state_half.weight.copy_(first.weight.detach()[:, :hidden])
            state_half.bias.copy_(first.bias.detach())
            action_half.weight.copy_(first.weight.detach()[:, hidden:])
        self.policy_state = _sliced_linear(state_half, None, last_state).to(
            first.weight.device
        )
        self.policy_action = _sliced_linear(action_half, None, last_action).to(
            first.weight.device
        )
        if model.value_context:
            self.value_merge = _sliced_linear(
                model.value_merge, None, torch.cat([last_state, hidden + last_action])
            )

    def forward(self, state, actions, index):
        model = self.model
        if self.keep is None:
            hidden = model.hidden
            h = model.state_trunk(state)  # (B, H)
            a = model.action_trunk(actions)  # (T, H)
            first = model.policy_head[0]
            weight = first.weight  # (H, 2H)
            state_part = nn.functional.linear(h, weight[:, :hidden], first.bias)
            action_part = nn.functional.linear(a, weight[:, hidden:])
            value_merge = model.value_merge if model.value_context else None
        else:
            h = self.state_trunk(state)  # (B, kept state units)
            a = self.action_trunk(actions)  # (T, kept action units)
            state_part = self.policy_state(h)  # (B, H)
            action_part = self.policy_action(a)  # (T, H)
            value_merge = self.value_merge if model.value_context else None
        logits = model.policy_head[2](
            torch.relu(state_part[index] + action_part)
        ).squeeze(-1)
        if value_merge is not None:
            rows = h.size(0)
            grid = torch.arange(rows, device=index.device)
            one_hot = (index.unsqueeze(1) == grid.unsqueeze(0)).to(a.dtype)
            pooled = one_hot.t().matmul(a)  # (B, width of a)
            counts = one_hot.sum(dim=0).clamp_min(1.0).unsqueeze(1)  # (B, 1)
            h = value_merge(torch.cat([h, pooled / counts], dim=-1))
        return logits, model.value_head(h)


def model_config(model) -> dict:
    """The architecture fields that decide the graph's shapes."""
    return {
        "state_dim": int(model.state_dim),
        "action_dim": int(model.action_dim),
        "hidden": int(model.hidden),
        "trunk_layers": int(model.trunk_layers),
        "n_houses": int(model.n_houses),
        "value_context": bool(model.value_context),
    }


def weights_digest(model) -> str:
    """SHA-256 over the model's parameters, in state-dict order.

    This is what makes a cached graph provably belong to a set of weights: the
    checkpoint hash alone would let an in-memory model (a worker's copy, or a
    freshly initialised net) reuse a graph exported from different numbers.
    """
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().to("cpu", torch.float32).contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_digest(checkpoint) -> tuple[str | None, int | None]:
    """``(sha256, size)`` of a checkpoint file, or ``(None, None)`` if absent."""
    if checkpoint is None:
        return None, None
    path = Path(checkpoint)
    if not path.is_file():
        return None, None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest(), path.stat().st_size


def default_cache_dir() -> Path:
    """Where a graph goes when its model has no checkpoint to sit beside."""
    return Path(tempfile.gettempdir()) / "agot-onnx-cache"


def cache_paths(
    model,
    backend: str,
    *,
    checkpoint=None,
    cache_dir=None,
    keep=None,
) -> tuple[Path, Path]:
    """``(graph, sidecar)`` paths for one model/config/backend combination.

    A compact graph (``keep`` given) has its own file, so the full graph that
    evaluation and the gate open beside the same checkpoint never evicts the
    compact one self-play uses, and vice versa.
    """
    if backend not in ONNX_BACKENDS:
        raise ValueError(
            f"unknown ONNX backend {backend!r}; choose from {ONNX_BACKENDS}"
        )
    if checkpoint is None:
        directory = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        stem = weights_digest(model)[:16]
    else:
        source = Path(checkpoint)
        directory = Path(cache_dir) if cache_dir is not None else source.parent
        stem = source.stem
    kind = backend if keep is None else f"{backend}.compact"
    graph = directory / f"{stem}.{kind}.onnx"
    return graph, graph.with_name(graph.name + ".json")


def graph_key(
    model,
    backend: str,
    *,
    checkpoint=None,
    opset: int = EXPORT_OPSET,
    keep=None,
) -> dict:
    """Everything a cached graph must match to be reused.

    ``keep`` is part of the key as plain lists (``None`` for the full graph),
    because that is how it reads back from the JSON sidecar.
    """
    sha256, size = checkpoint_digest(checkpoint)
    keep = normalize_keep(model, keep)
    return {
        "graph": GRAPH_REVISION,
        "backend": backend,
        "opset": int(opset),
        "model_config": model_config(model),
        "weights": weights_digest(model),
        "checkpoint_sha256": sha256,
        "checkpoint_size": size,
        "keep": None if keep is None else [list(units) for units in keep],
    }


def graph_is_fresh(key: dict, cached: dict | None) -> bool:
    """Whether a sidecar payload matches ``key`` on every field that matters."""
    if not isinstance(cached, dict):
        return False
    return all(cached.get(field) == key.get(field) for field in KEY_FIELDS)


def _stale_reason(key: dict, cached: dict | None) -> str:
    if not isinstance(cached, dict):
        return "its cache metadata is missing or unreadable"
    changed = [
        field for field in KEY_FIELDS if cached.get(field) != key.get(field)
    ]
    return "it was exported from something else (" + ", ".join(changed) + ")"


def _read_sidecar(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_sidecar(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _atomic_replace(temporary, path)


def _atomic_replace(temporary: Path, target: Path, *, attempts: int = 20) -> None:
    """Rename ``temporary`` into place, tolerating another process doing it too.

    On Windows a rename fails with ``PermissionError`` while anyone holds the
    destination open — another worker replacing the same cached graph, or an
    ORT session reading it. Retrying is enough, because every writer computes
    the identical bytes from the identical weights; the last one simply wins.
    """
    for attempt in range(attempts):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.01 * (attempt + 1))


def _temporary(path: Path) -> Path:
    """A per-process temporary beside ``path``, so the rename is atomic."""
    return path.with_name(f"{path.name}.tmp-{os.getpid()}")


def export_onnx(model, path, *, opset: int = EXPORT_OPSET, keep=None) -> Path:
    """Write the float32 ONNX graph of ``model``'s packed forward.

    ``keep`` exports the compact graph (see :class:`PackedGraph`).

    Uses the legacy exporter on purpose (see the module docstring) and never
    leaves a half-written file: the graph is traced into a temporary beside the
    target and renamed into place.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = model_config(model)
    device = next(model.parameters()).device
    leaves, width = 4, 5
    state = torch.zeros(leaves, config["state_dim"], dtype=torch.float32, device=device)
    actions = torch.zeros(
        leaves * width, config["action_dim"], dtype=torch.float32, device=device
    )
    index = torch.repeat_interleave(torch.arange(leaves, device=device), width)

    graph = PackedGraph(model, keep).to(device).eval()
    was_training = model.training
    temporary = _temporary(path)
    try:
        with torch.no_grad(), warnings.catch_warnings():
            # The legacy exporter is chosen on purpose (see the module
            # docstring) and warns about its own deprecation on every call.
            # One export per iteration per worker must not fill the run's log.
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(
                graph,
                (state, actions, index),
                str(temporary),
                input_names=list(INPUT_NAMES),
                output_names=list(OUTPUT_NAMES),
                dynamic_axes={
                    "state": {0: "leaves"},
                    "actions": {0: "candidates"},
                    "index": {0: "candidates"},
                    "logits": {0: "candidates"},
                    "values": {0: "leaves"},
                },
                opset_version=int(opset),
                do_constant_folding=True,
                dynamo=False,
            )
        _atomic_replace(temporary, path)
    finally:
        model.train(was_training)
        temporary.unlink(missing_ok=True)
    return path


def quantize_onnx(source, target) -> Path:
    """Dynamically quantize ``source`` to int8 weights, per output channel."""
    _onnx()
    from onnxruntime.quantization import (  # noqa: PLC0415 - optional dependency
        QuantType,
        quantize_dynamic,
    )

    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(target)
    # quantize_dynamic logs a "consider pre-processing" hint through the root
    # logger on every call. We quantize dynamically on purpose (there is no
    # calibration set, and a leaf batch is not an image pipeline), and with one
    # build per worker per iteration that hint would be the loudest thing in a
    # training log.
    root = logging.getLogger()
    level = root.level
    root.setLevel(max(level, logging.ERROR))
    try:
        quantize_dynamic(
            str(source),
            str(temporary),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        _atomic_replace(temporary, target)
    finally:
        root.setLevel(level)
        temporary.unlink(missing_ok=True)
    return target


@dataclass(frozen=True)
class CachedGraph:
    """A graph on disk plus the key that makes it trustworthy."""

    path: Path
    sidecar: Path
    key: dict
    rebuilt: bool


def ensure_graph(
    model,
    backend: str,
    *,
    checkpoint=None,
    cache_dir=None,
    rebuild: bool = True,
    opset: int = EXPORT_OPSET,
    keep=None,
) -> CachedGraph:
    """Return a graph for ``model``/``backend``, building it if needed.

    ``keep`` asks for the compact graph of those keep-sets (see
    :func:`calibrate_keep`); ``None`` is the full graph. The keep-sets are part
    of the key, so self-play workers that receive the parent's keep-sets get a
    cache hit on the graph the parent built, and anything else is rebuilt.

    ``checkpoint`` decides *where* the graph lives (beside it) and its file hash
    joins the key. A miss builds the graph even with ``rebuild=False`` — a
    missing cache is not a lie. A cache that exists but does not match raises
    :class:`StaleGraphError` when ``rebuild`` is false, and is replaced
    otherwise.
    """
    if backend not in ONNX_BACKENDS:
        raise ValueError(
            f"unknown ONNX backend {backend!r}; choose from {ONNX_BACKENDS}"
        )
    keep = normalize_keep(model, keep)
    key = graph_key(model, backend, checkpoint=checkpoint, opset=opset, keep=keep)
    path, sidecar = cache_paths(
        model, backend, checkpoint=checkpoint, cache_dir=cache_dir, keep=keep
    )
    cached = _read_sidecar(sidecar)
    if path.is_file() and graph_is_fresh(key, cached):
        return CachedGraph(path, sidecar, key, rebuilt=False)
    if (path.exists() or sidecar.exists()) and not rebuild:
        raise StaleGraphError(
            f"{path} cannot be used for this model: {_stale_reason(key, cached)}. "
            "Pass rebuild=True (the default) to replace it."
        )

    try:
        if backend == "onnx-fp32":
            export_onnx(model, path, opset=opset, keep=keep)
        else:
            # quantize_dynamic reads a model file, so the float32 graph goes to
            # a temporary that never joins the cache.
            base = path.with_name(f"{path.stem}.fp32.tmp-{os.getpid()}.onnx")
            try:
                export_onnx(model, base, opset=opset, keep=keep)
                quantize_onnx(base, path)
            finally:
                base.unlink(missing_ok=True)
        _write_sidecar(
            sidecar,
            {
                **key,
                "written": {
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "checkpoint": None if checkpoint is None else str(checkpoint),
                    "torch": torch.__version__,
                    "onnxruntime": _ort().__version__,
                    "keep_widths": keep_widths(keep),
                },
            },
        )
    except OSError:
        # Another process may have produced the same graph while this one was
        # writing it (a pool of workers importing the same checkpoint). If what
        # ended up on disk is fresh, that is the same artefact; anything else is
        # a real failure and must surface.
        if path.is_file() and graph_is_fresh(key, _read_sidecar(sidecar)):
            return CachedGraph(path, sidecar, key, rebuilt=False)
        raise
    return CachedGraph(path, sidecar, key, rebuilt=True)


def create_session(path, *, threads: int = 0, providers=None):
    """An ORT session over ``path``.

    ``threads=0`` keeps ORT's own intra-op default (one thread per core).
    Self-play workers pass 1: a pool of single-threaded workers beats the same
    number of sessions that each open one thread per core, which is why those
    workers also call ``torch.set_num_threads(1)``.

    Two settings are not ORT defaults on purpose. Inter-op parallelism is pinned
    to one: the packed forward is a chain, so a second op-level thread pool only
    adds hand-offs. Spinning is off: ORT's worker threads otherwise busy-wait
    between nodes, which on a box that also runs 8-12 self-play processes burns
    the cores it is meant to be using (measured: ORT's defaults were ~3x slower
    than a 1-thread session for a leaf batch this small).
    """
    ort = _ort()
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    if threads and threads > 0:
        options.intra_op_num_threads = int(threads)
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(path),
        options,
        providers=list(providers) if providers else ["CPUExecutionProvider"],
    )


class LeafBackend:
    """A leaf scorer backed by an ONNX Runtime session.

    The call signature is exactly the driver's leaf protocol — numpy in, numpy
    out — so it stands in for the closure
    :func:`agot.training.rust_engine.leaf_evaluator` returns::

        logits, values = backend(state, actions, index, mask)

    ``state`` is ``(B, state_dim)`` float32, ``actions`` ``(T, action_dim)``
    float32 and ``index`` ``(T,)`` int64; ``logits`` comes back ``(T,)`` float32
    and ``values`` ``(B, n_houses)`` float32. ``mask`` is accepted and ignored,
    like the torch closure: the driver only ever sends legal candidates.

    A session belongs to the process that opened it, so workers build their own
    with :meth:`from_model`; nothing here is picklable.
    """

    def __init__(self, session, cached: CachedGraph, *, threads: int = 0) -> None:
        self.session = session
        self.cached = cached
        self.threads = int(threads)
        self._output_names = list(OUTPUT_NAMES)
        self._native_run = self._find_native_run(session)

    @classmethod
    def from_model(
        cls,
        model,
        backend: str = "onnx-int8",
        *,
        checkpoint=None,
        cache_dir=None,
        rebuild: bool = True,
        threads: int = 0,
        providers=None,
        keep=None,
    ) -> LeafBackend:
        """Build (or reuse) the graph for ``model`` and open a session on it.

        ``checkpoint`` defaults to the file ``model`` was loaded from, when it
        knows one, so ``leaf_evaluator(model, "onnx-int8")`` caches beside the
        checkpoint without the caller threading a path through. ``keep`` opens
        the compact graph of those keep-sets (``None``: the full graph).
        """
        if checkpoint is None:
            checkpoint = getattr(model, "checkpoint", None)
        cached = ensure_graph(
            model,
            backend,
            checkpoint=checkpoint,
            cache_dir=cache_dir,
            rebuild=rebuild,
            keep=keep,
        )
        session = create_session(cached.path, threads=threads, providers=providers)
        return cls(session, cached, threads=threads)

    @property
    def path(self) -> Path:
        return self.cached.path

    @property
    def backend(self) -> str:
        return self.cached.key["backend"]

    @property
    def model_config(self) -> dict:
        return self.cached.key["model_config"]

    @property
    def keep_widths(self) -> list[int] | None:
        """Units kept per trunk layer, or ``None`` for the full graph."""
        keep = self.cached.key.get("keep")
        return None if keep is None else [len(units) for units in keep]

    @property
    def rebuilt(self) -> bool:
        """True when this call had to export (or re-export) the graph."""
        return self.cached.rebuilt

    def as_leaf_evaluator(self) -> LeafBackend:
        """Protocol hook: the driver may score leaves through this object."""
        return self

    def __call__(self, state, actions, index, mask=None):
        feed = {
            "state": np.ascontiguousarray(state, dtype=np.float32),
            "actions": np.ascontiguousarray(actions, dtype=np.float32),
            "index": np.ascontiguousarray(index, dtype=np.int64),
        }
        # The self-play driver calls this tens of thousands of times per chunk
        # with ~3 leaves each, so the Python wrapper's per-call checks (feed
        # names, OrtValue ownership, WebGPU capture) are a visible share. The
        # feed is always these three numpy arrays on a CPU session, which is
        # exactly what those checks accept, so call the native run directly
        # when the session exposes it; the result is the same C call.
        native = self._native_run
        if native is not None:
            logits, values = native(self._output_names, feed, None)
        else:
            logits, values = self.session.run(self._output_names, feed)
        return logits, values

    @staticmethod
    def _find_native_run(session):
        """The session's native ``run``, or ``None`` to use the public one."""
        inner = getattr(session, "_sess", None)
        run = getattr(inner, "run", None)
        get_providers = getattr(session, "get_providers", None)
        providers = list(get_providers()) if callable(get_providers) else []
        return run if callable(run) and providers == ["CPUExecutionProvider"] else None


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _policy_kl(reference: np.ndarray, candidate: np.ndarray, width: int) -> np.ndarray:
    """Per-leaf KL(torch || candidate) over contiguous segments of ``width``."""
    rows = reference.reshape(-1, width).astype(np.float64)
    other = candidate.reshape(-1, width).astype(np.float64)
    rows = rows - rows.max(axis=1, keepdims=True)
    rows = np.exp(rows)
    rows /= rows.sum(axis=1, keepdims=True)
    other = other - other.max(axis=1, keepdims=True)
    other = np.exp(other)
    other /= other.sum(axis=1, keepdims=True)
    return np.sum(rows * (np.log(rows) - np.log(other)), axis=1)


def _measure(calls: dict, samples: list, repeat: int) -> dict:
    """Time several leaf callbacks round-robin, ``repeat`` calls each.

    The box these runs happen on is shared, so a single long burst is a
    lottery: one external process can double any given measurement. Rotating the
    backends inside every round makes all of them see the same load, and the
    minimum is the closest estimate of what the call itself costs. The median is
    kept alongside it to show how noisy the machine was.
    """
    timings = {name: [] for name in calls}
    for position in range(repeat):
        sample = samples[position % len(samples)]
        for name, call in calls.items():
            started = time.perf_counter()
            call(*sample)
            timings[name].append((time.perf_counter() - started) * 1e6)
    return {
        name: {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
        }
        for name, values in timings.items()
    }


def benchmark(
    *,
    checkpoint=None,
    leaves: int = 24,
    width: int = 15,
    hidden: int = 256,
    repeat: int = 120,
    threads: int = 1,
    batches: int = 8,
    seed: int = 0,
) -> dict:
    """Time torch, onnx-fp32 and onnx-int8 on a synthetic production batch.

    ``width`` candidates per leaf is the measured mean of the search's leaves
    (15-16) and ``leaves`` is one worker's batch, so the shapes match what
    self-play actually asks for. With ``checkpoint`` the model is the real saved
    one (and its ``value_context`` decides the graph); without it a random net is
    used, whose near-uniform logits make the int8 KL unrealistically small.

    Every backend is timed through the same numpy-in/numpy-out callback the
    driver calls, so the numbers include the tensor conversion and the
    ``.numpy()`` copy. Returns the table plus the int8 policy KL against torch;
    ``python -m agot.nn.onnx_backend [checkpoint]`` prints it.
    """
    from agot.nn.model import PolicyValueNet  # noqa: PLC0415 - avoid a cycle
    from agot.training.rust_engine import leaf_evaluator  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if checkpoint is None:
        model = PolicyValueNet(
            hidden=hidden, trunk_layers=2, value_context=True
        ).eval()
    else:
        model = PolicyValueNet.load(str(checkpoint))
    state_dim, action_dim = int(model.state_dim), int(model.action_dim)
    index = np.repeat(np.arange(leaves, dtype=np.int64), width)
    mask = np.ones(int(index.size), dtype=np.float32)
    samples = [
        (
            rng.standard_normal((leaves, state_dim)).astype(np.float32),
            rng.standard_normal((leaves * width, action_dim)).astype(np.float32),
            index,
            mask,
        )
        for _ in range(max(1, batches))
    ]

    directory = Path(tempfile.mkdtemp(prefix="agot-onnx-bench-"))
    previous = torch.get_num_threads()
    torch.set_num_threads(max(1, threads))
    try:
        calls = {
            "torch": leaf_evaluator(model, "torch"),
            "onnx-fp32": LeafBackend.from_model(
                model, "onnx-fp32", cache_dir=directory, threads=threads
            ),
            "onnx-int8": LeafBackend.from_model(
                model, "onnx-int8", cache_dir=directory, threads=threads
            ),
        }
        for call in calls.values():
            call(*samples[0])
        timings = _measure(calls, samples, repeat)


        references = [
            _torch_reference(model, state, actions, index_rows)
            for state, actions, index_rows, _ in samples
        ]
        drift = {
            name: _drift(
                references,
                [call(state, actions, rows) for state, actions, rows, _ in samples],
                [rows for _, _, rows, _ in samples],
            )
            for name, call in calls.items()
            if name != "torch"
        }
    finally:
        torch.set_num_threads(previous)

    int8 = drift["onnx-int8"]
    results: dict = {
        "shape": {"leaves": leaves, "width": width, "hidden": int(model.hidden)},
        "threads": int(threads),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "us": {name: entry["min"] for name, entry in timings.items()},
        "median_us": {name: entry["median"] for name, entry in timings.items()},
        # Kept for older readers: the int8 numbers, as before.
        "kl": {
            "mean": int8["kl_mean"],
            "p95": int8["kl_p95"],
            "max": int8["kl_max"],
            "value_max_abs": int8["value_max"],
        },
        "drift": drift,
    }
    results["speedup"] = {
        name: results["us"]["torch"] / us
        for name, us in results["us"].items()
        if name != "torch"
    }
    return results


def _torch_reference(model, state, actions, index) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        logits, values = model.forward_packed(
            torch.from_numpy(state), torch.from_numpy(actions), torch.from_numpy(index)
        )
    return logits.numpy(), values.numpy()


def _segment_kl(reference: np.ndarray, candidate: np.ndarray, index: np.ndarray) -> np.ndarray:
    """KL(reference || candidate) per leaf, over the packed segments of ``index``."""
    leaves = int(index.max()) + 1 if index.size else 0

    def log_softmax(logits):
        logits = logits.astype(np.float64)
        peak = np.full(leaves, -np.inf)
        np.maximum.at(peak, index, logits)
        shifted = logits - peak[index]
        total = np.zeros(leaves)
        np.add.at(total, index, np.exp(shifted))
        return shifted - np.log(total)[index]

    p, q = log_softmax(reference), log_softmax(candidate)
    out = np.zeros(leaves)
    np.add.at(out, index, np.exp(p) * (p - q))
    return out


def _drift(references, outputs, indices) -> dict:
    """Policy KL and value |difference| of ``outputs`` against ``references``.

    The value difference is per leaf, the max over the houses; both are
    summarised over every leaf of every call.
    """
    kls, values = [], []
    for (want_logits, want_values), (logits, got_values), index in zip(
        references, outputs, indices, strict=True
    ):
        kls.append(_segment_kl(want_logits, logits, index))
        values.append(np.abs(np.asarray(got_values) - want_values).max(axis=1))
    kl = np.concatenate(kls)
    value = np.concatenate(values)
    return {
        "kl_mean": float(kl.mean()),
        "kl_p95": float(np.percentile(kl, 95)),
        "kl_max": float(kl.max()),
        "value_mean": float(value.mean()),
        "value_max": float(value.max()),
    }


def _format_drift(name: str, entry: dict) -> str:
    return (
        f"{name:<14}KL mean {entry['kl_mean']:.2e} p95 {entry['kl_p95']:.2e} "
        f"max {entry['kl_max']:.2e} nats; value |diff| mean {entry['value_mean']:.2e} "
        f"max {entry['value_max']:.2e}"
    )


def format_benchmark(results: dict) -> str:
    """The benchmark as a small text table, for the CLI and the docs."""
    shape = results["shape"]
    lines = [
        f"{shape['leaves']} leaves x {shape['width']} candidates, "
        f"hidden {shape['hidden']}, {results['threads']} thread(s), "
        f"model {'random init' if results['checkpoint'] is None else results['checkpoint']}",
        f"{'backend':<12}{'us/call':>10}{'median':>10}{'vs torch':>10}",
    ]
    for name, microseconds in results["us"].items():
        speedup = results["speedup"].get(name, 1.0)
        lines.append(
            f"{name:<12}{microseconds:>10.1f}"
            f"{results['median_us'][name]:>10.1f}{speedup:>9.2f}x"
        )
    lines.append("drift vs torch (synthetic inputs):")
    for name, entry in results["drift"].items():
        lines.append("  " + _format_drift(name, entry))
    return "\n".join(lines)


def load_leaf_calls(path, *, limit: int | None = None) -> list[tuple]:
    """Real leaf batches captured from self-play, one tuple per driver call.

    The file holds the concatenated calls (``states``, ``actions`` and a global
    ``index``) plus ``call_leaves``/``call_rows``, the size of each call, which
    is the layout ``checkpoints/perf-2x/capture_leaves.py`` writes. ``limit``
    keeps an evenly spaced subset of the calls.
    """
    data = np.load(path)
    states, actions, index = data["states"], data["actions"], data["index"]
    leaves, rows = data["call_leaves"], data["call_rows"]
    state_at = np.concatenate([[0], np.cumsum(leaves)])
    row_at = np.concatenate([[0], np.cumsum(rows)])
    picks = range(len(leaves))
    if limit is not None and limit < len(leaves):
        picks = range(0, len(leaves), max(1, len(leaves) // limit))
    calls = []
    for call in picks:
        lo, hi = row_at[call], row_at[call + 1]
        calls.append(
            (
                np.ascontiguousarray(states[state_at[call] : state_at[call + 1]]),
                np.ascontiguousarray(actions[lo:hi]),
                np.ascontiguousarray(index[lo:hi] - state_at[call], dtype=np.int64),
            )
        )
    return calls


def measure_leaves(
    checkpoint,
    leaves_path,
    replay_path,
    *,
    calls: int | None = 1500,
    repeat: int = 5,
    threads: int = 1,
    calibration_threads: int = 2,
) -> dict:
    """Full vs compact, fp32 vs int8, on real leaf calls of ``checkpoint``.

    Calibrates the compact keep-sets on the replay buffer at ``replay_path``
    (its ``states``/``actions`` arrays, the buffer's own npz), builds the four
    graphs in a temporary directory, then times every engine over the same
    calls, rotating engines inside each repetition (the minimum and median of
    the per-call means over ``repeat`` passes), and reports each engine's
    policy KL and value |difference| against the torch forward, plus the
    compact-vs-full drift at equal precision.

        python -m agot.nn.onnx_backend <ckpt> --leaves leaves.npz --replay agot.replay.npz
    """
    from agot.nn.model import PolicyValueNet  # noqa: PLC0415 - avoid a cycle

    model = PolicyValueNet.load(str(checkpoint)).eval()
    with np.load(replay_path) as replay:
        # Decompress before the clock starts: training calibrates on the
        # buffer already in memory.
        replay_states, replay_actions = replay["states"], replay["actions"]
    previous = torch.get_num_threads()
    torch.set_num_threads(max(1, calibration_threads))
    try:
        started = time.perf_counter()
        keep = calibrate_keep(model, replay_states, replay_actions)
        seconds_calibration = time.perf_counter() - started
    finally:
        torch.set_num_threads(previous)
    examples, rows = int(len(replay_states)), int(len(replay_actions))
    del replay_states, replay_actions

    leaf_calls = load_leaf_calls(leaves_path, limit=calls)
    directory = Path(tempfile.mkdtemp(prefix="agot-onnx-leaves-"))
    previous = torch.get_num_threads()
    torch.set_num_threads(max(1, threads))
    try:
        engines = {}
        export_seconds = {}
        for label, kept in (("full", None), ("compact", keep)):
            for backend in ONNX_BACKENDS:
                started = time.perf_counter()
                engines[f"{label}-{backend[5:]}"] = LeafBackend.from_model(
                    model, backend, cache_dir=directory, threads=threads, keep=kept
                )
                export_seconds[f"{label}-{backend[5:]}"] = time.perf_counter() - started
        references = [_torch_reference(model, *call) for call in leaf_calls]
        outputs = {name: [engine(*call) for call in leaf_calls] for name, engine in engines.items()}
        timings = {name: [] for name in engines}
        for _ in range(max(1, repeat)):
            for name, engine in engines.items():
                started = time.perf_counter()
                for call in leaf_calls:
                    engine(*call)
                timings[name].append((time.perf_counter() - started) / len(leaf_calls) * 1e6)
    finally:
        torch.set_num_threads(previous)

    indices = [call[2] for call in leaf_calls]
    drift = {name: _drift(references, outputs[name], indices) for name in engines}
    compaction = {
        precision: _drift(outputs[f"full-{precision}"], outputs[f"compact-{precision}"], indices)
        for precision in ("fp32", "int8")
    }
    return {
        "checkpoint": str(checkpoint),
        "calls": len(leaf_calls),
        "leaves": int(sum(len(call[0]) for call in leaf_calls)),
        "rows": int(sum(len(call[1]) for call in leaf_calls)),
        "threads": int(threads),
        "calibration": {
            "examples": examples,
            "rows": rows,
            "threads": int(calibration_threads),
            "seconds": seconds_calibration,
            "widths": keep_widths(keep),
        },
        "export_seconds": export_seconds,
        "us": {
            name: {"min": float(np.min(values)), "median": float(np.median(values))}
            for name, values in timings.items()
        },
        "drift": drift,
        "compaction": compaction,
    }


def format_leaf_measurement(results: dict) -> str:
    calibration = results["calibration"]
    lines = [
        f"{results['checkpoint']}: {results['calls']} real calls, {results['leaves']} "
        f"leaves, {results['rows']} candidate rows, {results['threads']} thread(s)",
        f"calibration: {calibration['examples']} examples / {calibration['rows']} rows "
        f"in {calibration['seconds']:.2f}s on {calibration['threads']} thread(s); "
        f"kept units per trunk layer {calibration['widths']}",
        f"{'engine':<14}{'us/call min':>12}{'median':>10}{'export s':>10}",
    ]
    for name, entry in results["us"].items():
        lines.append(
            f"{name:<14}{entry['min']:>12.1f}{entry['median']:>10.1f}"
            f"{results['export_seconds'][name]:>10.2f}"
        )
    lines.append("drift vs torch:")
    for name, entry in results["drift"].items():
        lines.append("  " + _format_drift(name, entry))
    lines.append("compaction alone (compact vs full, same precision):")
    for precision, entry in results["compaction"].items():
        lines.append("  " + _format_drift(precision, entry))
    return "\n".join(lines)


def _main(argv=None) -> None:  # pragma: no cover - manual measurement
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m agot.nn.onnx_backend",
        description=(
            "Leaf-engine cost and drift. Without --leaves: torch / fp32 / int8 on a "
            "synthetic production batch. With --leaves and --replay: full vs compact "
            "graphs, fp32 vs int8, on real captured leaf calls."
        ),
    )
    parser.add_argument("checkpoint", nargs="?")
    parser.add_argument("--leaves", help="captured leaf calls (capture_leaves.py layout)")
    parser.add_argument("--replay", help="replay buffer npz to calibrate the compact graph")
    parser.add_argument("--calls", type=int, default=1500)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--calibration-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.leaves:
        if not (args.checkpoint and args.replay):
            parser.error("--leaves needs a checkpoint and --replay")
        print(
            format_leaf_measurement(
                measure_leaves(
                    args.checkpoint,
                    args.leaves,
                    args.replay,
                    calls=args.calls,
                    repeat=args.repeat,
                    threads=args.threads,
                    calibration_threads=args.calibration_threads,
                )
            )
        )
    else:
        print(format_benchmark(benchmark(checkpoint=args.checkpoint, threads=args.threads)))


if __name__ == "__main__":  # pragma: no cover - manual measurement
    _main()
