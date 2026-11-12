"""Self-play driven by the Rust engine.

The training loop's hot path is here: whole games are played by
``agot._core.selfplay`` and the network is called once per *batch of leaves*
through :func:`leaf_evaluator`, so the interpreter only sees one call per batch
instead of one per decision. Scripted opponent seats are the exception: the
driver asks Python for one decision at a time, never one per simulation.

:func:`agot.training.selfplay.batched_self_play` is the entry point. It uses the
Rust driver whenever the evaluator is model-backed and the games share one
configuration, and falls back to the Python reference driver for hand-written
evaluators or mixed configurations.

Shapes match the Python driver exactly, so ``replay.py`` and ``trainer.py`` need
no changes: a :class:`GameRecord` holds :class:`Example` objects with
``state_features``, ``action_features``, ``policy``, ``player`` and ``outcome``.
"""

from __future__ import annotations

import numpy as np

from agot.engine.game import GameConfig
from agot.engine.observation import HOUSE_ORDER
from agot.training.parallel import SelfPlayPlan
from agot.training.replay import Example
from agot.training.selfplay import GameRecord

#: The extension's module name. Imported lazily so a torch-free environment can
#: import this module without failing.
MODULE = "agot._core"

#: Per-phase simulation caps for the stepwise driver, tuned on the batched tier.
#:
#: A fixed budget spends most of its leaves on Planning and Wildling-attack
#: decisions, whose candidate Q values barely differ and whose target sits
#: ~0.03 nats from the prior, while the rare board-changing phases keep the
#: fixed depth. ``off`` (``None``) reproduces the fixed budget exactly, so a
#: run that does not ask for the feature is unchanged. The caps are per
#: *decision*: the driver divides them by ``determinizations`` exactly as it
#: divides ``simulations``, and never goes below one simulation per candidate.
#:
#: Keep these numbers in step with ``PhaseBudgets::DEFAULT`` in
#: ``rust/crates/agot-core/src/budget.rs``: a table is an override of that
#: preset, so a phase missing here (``game_over``, which never opens a root)
#: keeps the Rust value.
PHASE_BUDGETS: dict[str, dict[str, int] | None] = {
    "off": None,
    "default": {
        "planning": 32,
        "wildling_attack": 16,
        "mustering": 32,
        "bidding": 32,
        "action": 32,
        "raven": 64,
        "combat": 64,
        "westeros": 64,
        "westeros_choice": 64,
        "cleanup": 64,
    },
}

#: Values the ``--phase-budgets`` switch accepts, in CLI order.
PHASE_BUDGET_CHOICES = ("off", "default")


def resolve_phase_budgets(name: str | None) -> dict[str, int] | None:
    """The ``phase_budgets`` spec table for a preset name (``None`` is ``off``).

    A copy is returned so a caller cannot mutate the shared preset. Unknown
    names raise instead of silently falling back to the fixed budget.
    """
    if name is None:
        return None
    try:
        preset = PHASE_BUDGETS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown phase budget preset {name!r}; "
            f"choose from {', '.join(PHASE_BUDGET_CHOICES)}"
        ) from error
    return None if preset is None else dict(preset)
#: Inference engines :func:`leaf_evaluator` accepts. ``"torch"`` is the default
#: and the only one that needs nothing beyond PyTorch; the ONNX backends need the
#: optional ``onnx`` extra and cache their graph beside the checkpoint.
LEAF_BACKENDS = ("torch", "onnx-int8", "onnx-fp32")

#: What ``--leaf-backend auto`` means when the optional extra is installed.
#: Training gets the int8 graph, which is what the run at the top of AGENTS.md
#: (`run6`) already selects and the only engine that halves self-play: 2.35x the
#: torch path end to end at batch 8, at a measured 4e-4 nats mean policy KL on a
#: trained net, inside the budget `tests/test_onnx_backend.py` enforces.
#: Evaluation, gauntlet and play get the float32 graph instead: it is a
#: round-off away from torch (7e-8 nats), so numbers stay comparable with past
#: results, and it is still 1.5-2.2x the torch path on the 1-2 leaf batches
#: those commands use.
AUTO_TRAIN_LEAF_BACKEND = "onnx-int8"
AUTO_LEAF_BACKEND = "onnx-fp32"


def onnx_runtime_available() -> bool:
    """Whether the optional ONNX stack (runtime *and* exporter) is importable."""
    try:
        import onnx  # noqa: F401, PLC0415 - optional dependency probe
        import onnxruntime  # noqa: F401, PLC0415 - optional dependency probe
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


def resolve_leaf_backend(backend: str | None, *, training: bool = False) -> str | None:
    """Turn ``auto`` into a concrete engine, leaving every other value alone.

    ``training`` picks the int8 graph: a training run is the one place where the
    leaf batch is big and long-lived enough to pay for quantization, and the
    learning signal already tolerates that noise (run6 trains on it). ``None``
    stays ``None``: for ``agot gate`` it means "use whatever the run being
    judged used", which is not a decision this helper can make.
    """
    if backend != "auto":
        return backend
    if not onnx_runtime_available():
        return "torch"
    return AUTO_TRAIN_LEAF_BACKEND if training else AUTO_LEAF_BACKEND


def core():
    """The extension module, or ``None`` when it is not installed."""
    try:
        from agot import _core  # noqa: PLC0415 - deliberately lazy
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    return _core


def leaf_evaluator(
    model,
    backend: str = "torch",
    *,
    checkpoint: str | None = None,
    threads: int = 0,
):
    """Adapt a :class:`PolicyValueNet` to the driver's callback protocol.

    The driver hands over preallocated float32 arrays and expects raw logits and
    values back; it applies the per-leaf softmax itself, so this stays a single
    torch expression — the cheapest possible thing to run per batch.

    ``backend`` picks the inference engine:

    * ``"torch"`` (default) — the torch expression above, byte for byte.
    * ``"onnx-fp32"`` / ``"onnx-int8"`` — an ONNX Runtime session over an
      exported graph, with dynamic per-channel int8 weights for the second.
      The graph is cached beside ``checkpoint`` (or beside the file the model
      was loaded from) and is rebuilt when the weights or the config no longer
      match it.

    ``threads`` is the ORT intra-op thread count for the ONNX backends
    (``0`` = ORT's default). Self-play workers pass 1 to match the one thread
    they give torch.
    """
    if backend not in LEAF_BACKENDS:
        raise ValueError(
            f"unknown leaf backend {backend!r}; choose from {LEAF_BACKENDS}"
        )
    if backend == "torch":
        import torch

        def evaluate(state, actions, index, mask):
            with torch.inference_mode():
                logits, values = model.forward_packed(
                    torch.from_numpy(state),
                    torch.from_numpy(actions),
                    torch.from_numpy(index),
                )
                return logits.numpy(), values.numpy()

        return evaluate

    from agot.nn.onnx_backend import LeafBackend  # noqa: PLC0415 - optional dependency

    return LeafBackend.from_model(
        model, backend, checkpoint=checkpoint, threads=threads
    )


def game_seeds(configs, fallback: int) -> list[int]:
    """The explicit seed for every game in a request.

    The driver derives ``seed + index`` when it has no list, which is wrong as
    soon as a caller splits games across workers: each worker only sees its
    first config's seed and replays neighbours. Passing the seeds explicitly
    keeps every game distinct regardless of how the chunk was formed.
    """
    return [
        config.seed if config.seed is not None else fallback + index
        for index, config in enumerate(configs)
    ]


def spec_for(
    plan: SelfPlayPlan,
    base_config: GameConfig | None = None,
    *,
    dedupe_leaves: bool = True,
    phase_budgets: dict[str, int] | None = None,
    leaf_backend: str | None = None,
) -> dict:
    """Translate a :class:`SelfPlayPlan` into the driver's spec mapping.

    ``dedupe_leaves`` is the A/B switch for exact leaf deduplication: with it
    on (the default) the driver scores bit-identical leaves once per batch and
    scatters the results back, which is exact for a pure evaluator.

    ``phase_budgets`` is the per-phase cap table (see :data:`PHASE_BUDGETS`);
    ``None`` keeps the fixed simulation budget.
    ``leaf_backend`` is which engine scores the leaves (see
    :func:`leaf_evaluator`). The driver ignores it; it travels in the spec so a
    recorded plan says how its games were searched, and so a gate reproduces the
    run it is judging.
    """
    base = base_config or GameConfig()
    return {
        "games": plan.games,
        "simulations": plan.simulations,
        "determinizations": plan.determinizations,
        "batch_games": max(1, plan.batch_games),
        "dedupe_leaves": dedupe_leaves,
        "phase_budgets": phase_budgets,
        "leaf_backend": plan.leaf_backend if leaf_backend is None else leaf_backend,
        "seed": plan.seed,
        "seeds": [plan.seed + index for index in range(plan.games)],
        "temperature": plan.temperature,
        "temperature_rounds": plan.temperature_rounds,
        "final_temperature": plan.final_temperature,
        "max_rounds": plan.max_rounds,
        "victory_castles": base.victory_castles,
        "use_westeros_decks": base.use_westeros_decks,
        "deferred_battles": base.deferred_battles,
        "sequential_westeros": base.sequential_westeros,
        "doran_choice": base.doran_choice,
        "wildling_tie_choice": base.wildling_tie_choice,
        "bid_tie_choice": base.bid_tie_choice,
        "choice_choices": base.choice_choices,
        "houses": list(base.houses),
    }


def _unpack(result, games: int) -> list[GameRecord]:
    """Unpack the driver's packed buffers into :class:`GameRecord` objects."""
    (
        states,
        actions,
        offsets,
        policy,
        players,
        outcomes,
        winners,
        steps,
        examples_per_game,
        rounds,
    ) = result

    # `steps` is the number of decisions; `examples_per_game` is how many of
    # them were recorded. They differ when a scripted seat acts without
    # `record_opponents`, so the example buffers are sliced with the latter.
    total = int(sum(examples_per_game))
    if total != int(states.shape[0]):
        raise RuntimeError(
            f"the driver recorded {states.shape[0]} examples for {total} recorded decisions"
        )
    if len(steps) != games:
        raise RuntimeError(f"the driver played {len(steps)} of {games} games")

    records: list[GameRecord] = []
    cursor = 0
    for index, taken in enumerate(examples_per_game):
        examples: list[Example] = []
        for offset in range(cursor, cursor + int(taken)):
            start, end = int(offsets[offset]), int(offsets[offset + 1])
            examples.append(
                Example(
                    state=states[offset],
                    actions=actions[start:end],
                    # Already float32: a view avoids a copy per example and the
                    # packed arrays are kept alive by the other views anyway.
                    policy=np.asarray(policy[start:end], dtype=np.float32),
                    player=int(players[offset]),
                    outcome=np.asarray(outcomes[offset], dtype=np.float32),
                )
            )
        cursor += int(taken)
        records.append(
            GameRecord(
                examples=examples,
                winner=winners[index],
                steps=int(steps[index]),
                rounds=int(rounds[index]),
            )
        )
    return records


def self_play_records(
    model,
    plan: SelfPlayPlan,
    *,
    base_config: GameConfig | None = None,
    dedupe_leaves: bool = True,
    phase_budgets: dict[str, int] | None = None,
    leaf_backend: str | None = None,
) -> list[GameRecord]:
    """Play ``plan.games`` games with the Rust driver and return the records.

    ``dedupe_leaves`` is forwarded to :func:`spec_for`; turn it off to compare
    a run against the same leaves scored individually. ``phase_budgets`` is the
    per-phase cap table; ``None`` keeps the fixed budget.
    a run against the same leaves scored individually. ``leaf_backend``
    overrides ``plan.leaf_backend`` for this call.
    """
    extension = core()
    if extension is None:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            f"{MODULE} is not installed; build it with maturin or use "
            "agot.training.selfplay.batched_self_play"
        )
    spec = spec_for(
        plan,
        base_config,
        dedupe_leaves=dedupe_leaves,
        phase_budgets=phase_budgets,
    )
    result = extension.selfplay(spec, leaf_evaluator(model))
    backend = plan.leaf_backend if leaf_backend is None else leaf_backend
    spec = spec_for(plan, base_config, dedupe_leaves=dedupe_leaves, leaf_backend=backend)
    evaluate = leaf_evaluator(
        model, backend, checkpoint=plan.leaf_checkpoint, threads=1
    )
    result = extension.selfplay(spec, evaluate)
    return _unpack(result, plan.games)


def try_batched_self_play(
    evaluator,
    configs,
    *,
    simulations: int,
    temperature: float = 1.0,
    temperature_schedule=None,
    determinizations: int = 1,
    seed: int = 0,
    opponents=None,
    record_opponents: bool = False,
    step_limit: int = 20_000,
    batch_games: int = 4,
    dedupe_leaves: bool = True,
    phase_budgets: dict[str, int] | None = None,
    leaf_backend: str = "torch",
    leaf_checkpoint: str | None = None,
    leaf_threads: int = 1,
) -> list[GameRecord] | None:
    """Run a ``batched_self_play`` request on the Rust driver, or return ``None``.

    Scripted opponent seats are supported too: the driver asks Python for one
    decision at a time (never one per simulation), so a scripted seat costs a
    Python policy call rather than a Python search. ``record_opponents`` decides
    whether those decisions are imitated.

    ``dedupe_leaves`` (default true) scores bit-identical leaves once per batch;
    turn it off to A/B the exactness-preserving shortcut.

    ``phase_budgets`` caps the simulations a decision may spend by phase (see
    :data:`PHASE_BUDGETS`). ``None`` is the fixed budget; the driver divides a
    cap by ``determinizations`` exactly as it divides ``simulations``.
    ``leaf_backend`` selects the inference engine for the search seats (see
    :func:`leaf_evaluator`); the choice is also written into the spec, so the
    record of a plan says how its leaves were scored. ``leaf_checkpoint`` is
    where an ONNX graph is cached and ``leaf_threads`` is its intra-op thread
    count (1 by default, because this driver is used one thread per process).

    Returns ``None`` — leaving the caller to use the Python driver — when the
    games do not share one configuration or when the evaluator is not a
    model-backed one. Falling back is cheap and silent by design: the two paths
    produce the same records, so a caller never has to care which ran.
    """
    extension = core()
    if extension is None:
        return None
    model = getattr(evaluator, "model", None)
    # An evaluator that already knows how to score a leaf batch (a
    # ``LeafBackend``) says so through this hook; using it avoids quietly
    # rebuilding a torch closure and throwing the caller's backend away.
    hook = getattr(evaluator, "as_leaf_evaluator", None)
    if model is None and hook is None:  # a hand-written evaluator
        return None
    if not configs:
        return []

    first = configs[0]
    shared = (
        first.max_rounds,
        first.victory_castles,
        first.use_westeros_decks,
        first.deferred_battles,
        first.sequential_westeros,
        first.doran_choice,
        first.wildling_tie_choice,
        first.bid_tie_choice,
        first.choice_choices,
        tuple(first.houses),
    )
    for config in configs:
        if (
            config.max_rounds,
            config.victory_castles,
            config.use_westeros_decks,
            config.deferred_battles,
            config.sequential_westeros,
            config.doran_choice,
            config.wildling_tie_choice,
            config.bid_tie_choice,
            config.choice_choices,
            tuple(config.houses),
        ) != shared:
            return None

    # The driver derives each game's seed as `seed + index`, which is how
    # `parallel_self_play` builds its configs, so the seeds line up.
    schedule = tuple(temperature_schedule or ())
    scripted: list[list[str]] = [
        list(opponents[index] if opponents is not None else {})
        for index in range(len(configs))
    ]
    decider = _scripted_decider(opponents, record_opponents) if any(scripted) else None
    spec = {
        "games": len(configs),
        "simulations": simulations,
        "determinizations": determinizations,
        "batch_games": max(1, batch_games),
        "dedupe_leaves": dedupe_leaves,
        "phase_budgets": phase_budgets,
        "leaf_backend": leaf_backend,
        "seed": first.seed if first.seed is not None else seed,
        "seeds": game_seeds(configs, seed),
        "temperature": temperature,
        "temperature_rounds": schedule[0][0] if schedule else 3,
        "final_temperature": schedule[-1][1] if schedule else temperature,
        "max_rounds": first.max_rounds,
        "victory_castles": first.victory_castles,
        "use_westeros_decks": first.use_westeros_decks,
        "deferred_battles": first.deferred_battles,
        "sequential_westeros": first.sequential_westeros,
        "doran_choice": first.doran_choice,
        "wildling_tie_choice": first.wildling_tie_choice,
        "bid_tie_choice": first.bid_tie_choice,
        "choice_choices": first.choice_choices,
        "houses": list(first.houses),
        "step_limit": step_limit,
        "scripted": scripted,
        "record_opponents": record_opponents,
    }
    if hook is not None:
        evaluate = hook()
    else:
        evaluate = leaf_evaluator(
            model, leaf_backend, checkpoint=leaf_checkpoint, threads=leaf_threads
        )
    result = extension.selfplay(spec, evaluate, decider)
    return _unpack(result, len(configs))


def _scripted_decider(opponents, record_opponents: bool):
    """Adapt scripted ``Policy`` objects to the driver's per-decision callback.

    The driver hands over its own ``Game`` view; the policy is run through the
    Python-shaped ``RustGame`` wrapper it already understands. Only one policy
    call crosses the boundary per scripted decision — the search seats never do.
    """
    import numpy as np

    from agot.engine.rust import RustGame

    def decide(game_index: int, house: str, game) -> tuple[int, list[float] | None]:
        policy = opponents[game_index][house]
        view = RustGame(game)
        action, target = policy.decide(view)
        actions = view.legal_actions()
        try:
            index = actions.index(action)
        except ValueError as error:  # pragma: no cover - defensive
            raise RuntimeError(
                f"scripted policy {policy.name!r} chose an action the engine did not offer"
            ) from error
        if record_opponents:
            return index, np.asarray(target, dtype=np.float32).tolist()
        return index, None

    return decide


def house_order() -> tuple[str, ...]:
    """The canonical house order the driver indexes `players` with."""
    return HOUSE_ORDER
