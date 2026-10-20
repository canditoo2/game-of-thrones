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


def core():
    """The extension module, or ``None`` when it is not installed."""
    try:
        from agot import _core  # noqa: PLC0415 - deliberately lazy
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    return _core


def available() -> bool:
    """Whether the Rust engine can be used here."""
    return core() is not None


def leaf_evaluator(model):
    """Adapt a :class:`PolicyValueNet` to the driver's callback protocol.

    The driver hands over preallocated float32 arrays and expects raw logits and
    values back; it applies the per-leaf softmax itself, so this stays a single
    torch expression — the cheapest possible thing to run per batch.
    """
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


def spec_for(plan: SelfPlayPlan, base_config: GameConfig | None = None) -> dict:
    """Translate a :class:`SelfPlayPlan` into the driver's spec mapping."""
    base = base_config or GameConfig()
    return {
        "games": plan.games,
        "simulations": plan.simulations,
        "determinizations": plan.determinizations,
        "batch_games": max(1, plan.batch_games),
        "seed": plan.seed,
        "seeds": [plan.seed + index for index in range(plan.games)],
        "temperature": plan.temperature,
        "temperature_rounds": plan.temperature_rounds,
        "final_temperature": plan.final_temperature,
        "max_rounds": plan.max_rounds,
        "victory_castles": base.victory_castles,
        "use_westeros_decks": base.use_westeros_decks,
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
) -> list[GameRecord]:
    """Play ``plan.games`` games with the Rust driver and return the records."""
    extension = core()
    if extension is None:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            f"{MODULE} is not installed; build it with maturin or use "
            "agot.training.selfplay.batched_self_play"
        )
    spec = spec_for(plan, base_config)
    result = extension.selfplay(spec, leaf_evaluator(model))
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
) -> list[GameRecord] | None:
    """Run a ``batched_self_play`` request on the Rust driver, or return ``None``.

    Scripted opponent seats are supported too: the driver asks Python for one
    decision at a time (never one per simulation), so a scripted seat costs a
    Python policy call rather than a Python search. ``record_opponents`` decides
    whether those decisions are imitated.

    Returns ``None`` — leaving the caller to use the Python driver — when the
    games do not share one configuration or when the evaluator is not a
    model-backed one. Falling back is cheap and silent by design: the two paths
    produce the same records, so a caller never has to care which ran.
    """
    extension = core()
    if extension is None:
        return None
    model = getattr(evaluator, "model", None)
    if model is None:  # a hand-written evaluator: the Python driver handles it
        return None
    if not configs:
        return []

    first = configs[0]
    shared = (
        first.max_rounds,
        first.victory_castles,
        first.use_westeros_decks,
        tuple(first.houses),
    )
    for config in configs:
        if (
            config.max_rounds,
            config.victory_castles,
            config.use_westeros_decks,
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
        "seed": first.seed if first.seed is not None else seed,
        "seeds": game_seeds(configs, seed),
        "temperature": temperature,
        "temperature_rounds": schedule[0][0] if schedule else 3,
        "final_temperature": schedule[-1][1] if schedule else temperature,
        "max_rounds": first.max_rounds,
        "victory_castles": first.victory_castles,
        "use_westeros_decks": first.use_westeros_decks,
        "houses": list(first.houses),
        "step_limit": step_limit,
        "scripted": scripted,
        "record_opponents": record_opponents,
    }
    result = extension.selfplay(spec, leaf_evaluator(model), decider)
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
