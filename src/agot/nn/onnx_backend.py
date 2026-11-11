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
GRAPH_REVISION = "packed-v1"
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
)


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


class PackedGraph(nn.Module):
    """``PolicyValueNet.forward_packed`` expressed in exportable ops.

    The arithmetic is identical to the torch method, except that the
    value-context pooling is ``one_hot(index)^T @ a`` instead of
    ``index_add_``, which the exporter cannot represent.
    ``tests/test_onnx_backend.py`` pins the two against each other, so they
    cannot drift apart silently.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, state, actions, index):
        model = self.model
        hidden = model.hidden
        h = model.state_trunk(state)  # (B, H)
        a = model.action_trunk(actions)  # (T, H)
        first = model.policy_head[0]
        weight = first.weight  # (H, 2H)
        state_part = nn.functional.linear(h, weight[:, :hidden], first.bias)
        action_part = nn.functional.linear(a, weight[:, hidden:])
        logits = model.policy_head[2](
            torch.relu(state_part[index] + action_part)
        ).squeeze(-1)
        if model.value_context:
            rows = h.size(0)
            grid = torch.arange(rows, device=index.device)
            one_hot = (index.unsqueeze(1) == grid.unsqueeze(0)).to(a.dtype)
            pooled = one_hot.t().matmul(a)  # (B, H)
            counts = one_hot.sum(dim=0).clamp_min(1.0).unsqueeze(1)  # (B, 1)
            h = model.value_merge(torch.cat([h, pooled / counts], dim=-1))
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
) -> tuple[Path, Path]:
    """``(graph, sidecar)`` paths for one model/config/backend combination."""
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
    graph = directory / f"{stem}.{backend}.onnx"
    return graph, graph.with_name(graph.name + ".json")


def graph_key(
    model,
    backend: str,
    *,
    checkpoint=None,
    opset: int = EXPORT_OPSET,
) -> dict:
    """Everything a cached graph must match to be reused."""
    sha256, size = checkpoint_digest(checkpoint)
    return {
        "graph": GRAPH_REVISION,
        "backend": backend,
        "opset": int(opset),
        "model_config": model_config(model),
        "weights": weights_digest(model),
        "checkpoint_sha256": sha256,
        "checkpoint_size": size,
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


def export_onnx(model, path, *, opset: int = EXPORT_OPSET) -> Path:
    """Write the float32 ONNX graph of ``model``'s packed forward.

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

    graph = PackedGraph(model).to(device).eval()
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
) -> CachedGraph:
    """Return a graph for ``model``/``backend``, building it if needed.

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
    key = graph_key(model, backend, checkpoint=checkpoint, opset=opset)
    path, sidecar = cache_paths(
        model, backend, checkpoint=checkpoint, cache_dir=cache_dir
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
            export_onnx(model, path, opset=opset)
        else:
            # quantize_dynamic reads a model file, so the float32 graph goes to
            # a temporary that never joins the cache.
            base = path.with_name(f"{path.stem}.fp32.tmp-{os.getpid()}.onnx")
            try:
                export_onnx(model, base, opset=opset)
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
    ) -> LeafBackend:
        """Build (or reuse) the graph for ``model`` and open a session on it.

        ``checkpoint`` defaults to the file ``model`` was loaded from, when it
        knows one, so ``leaf_evaluator(model, "onnx-int8")`` caches beside the
        checkpoint without the caller threading a path through.
        """
        if checkpoint is None:
            checkpoint = getattr(model, "checkpoint", None)
        cached = ensure_graph(
            model,
            backend,
            checkpoint=checkpoint,
            cache_dir=cache_dir,
            rebuild=rebuild,
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
    def rebuilt(self) -> bool:
        """True when this call had to export (or re-export) the graph."""
        return self.cached.rebuilt

    def as_leaf_evaluator(self) -> LeafBackend:
        """Protocol hook: the driver may score leaves through this object."""
        return self

    def __call__(self, state, actions, index, mask=None):
        logits, values = self.session.run(
            list(OUTPUT_NAMES),
            {
                "state": np.ascontiguousarray(state, dtype=np.float32),
                "actions": np.ascontiguousarray(actions, dtype=np.float32),
                "index": np.ascontiguousarray(index, dtype=np.int64),
            },
        )
        return logits, values


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

        kls, value_errors = [], []
        for state, actions, index_rows, _ in samples:
            with torch.inference_mode():
                reference_logits, reference_values = model.forward_packed(
                    torch.from_numpy(state),
                    torch.from_numpy(actions),
                    torch.from_numpy(index_rows),
                )
            logits, values = calls["onnx-int8"](state, actions, index_rows)
            kls.append(_policy_kl(reference_logits.numpy(), logits, width))
            value_errors.append(float(np.abs(values - reference_values.numpy()).max()))
    finally:
        torch.set_num_threads(previous)

    kl = np.concatenate(kls)
    results: dict = {
        "shape": {"leaves": leaves, "width": width, "hidden": int(model.hidden)},
        "threads": int(threads),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "us": {name: entry["min"] for name, entry in timings.items()},
        "median_us": {name: entry["median"] for name, entry in timings.items()},
        "kl": {
            "mean": float(kl.mean()),
            "p95": float(np.percentile(kl, 95)),
            "max": float(kl.max()),
            "value_max_abs": float(max(value_errors)),
        },
    }
    results["speedup"] = {
        name: results["us"]["torch"] / us
        for name, us in results["us"].items()
        if name != "torch"
    }
    return results


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
    kl = results["kl"]
    lines.append(
        f"int8 policy KL vs torch: mean {kl['mean']:.2e} p95 {kl['p95']:.2e} "
        f"max {kl['max']:.2e} nats; value max |err| {kl['value_max_abs']:.2e}"
    )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual measurement
    import sys

    print(
        format_benchmark(
            benchmark(checkpoint=sys.argv[1] if len(sys.argv) > 1 else None)
        )
    )
