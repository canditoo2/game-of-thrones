"""Self-play driven by the Rust engine, when the extension is available.

The training loop's hot path is here: whole games are played by
``agot_core_ext.selfplay`` and the network is called once per *batch of leaves*
through :func:`leaf_evaluator`, so the interpreter only sees one call per batch
instead of one per decision.

This module is additive. :func:`agot.training.selfplay.batched_self_play` stays
the reference implementation and keeps working without the extension; once the
port is cut over, this becomes the only path and the two can be compared on the
same plan.

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

#: The extension's module name. Imported lazily so a torch-free or
#: extension-free environment can import this module without failing.
MODULE = "agot_core_ext"


def core():
    """The extension module, or ``None`` when it is not installed."""
    try:
        import agot_core_ext  # noqa: PLC0415 - deliberately lazy
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    return agot_core_ext


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


def spec_for(plan: SelfPlayPlan, base_config: GameConfig | None = None) -> dict:
    """Translate a :class:`SelfPlayPlan` into the driver's spec mapping."""
    base = base_config or GameConfig()
    return {
        "games": plan.games,
        "simulations": plan.simulations,
        "determinizations": plan.determinizations,
        "batch_games": max(1, plan.batch_games),
        "seed": plan.seed,
        "temperature": plan.temperature,
        "temperature_rounds": plan.temperature_rounds,
        "final_temperature": plan.final_temperature,
        "max_rounds": plan.max_rounds,
        "victory_castles": base.victory_castles,
        "use_westeros_decks": base.use_westeros_decks,
        "houses": list(base.houses),
    }


def self_play_records(
    model,
    plan: SelfPlayPlan,
    *,
    base_config: GameConfig | None = None,
) -> list[GameRecord]:
    """Play ``plan.games`` games with the Rust driver and return the records.

    The examples come back packed — one buffer per field, with an offset per
    decision — because that is what lets a whole game cross the boundary in one
    go. This function unpacks them into the objects the rest of the training code
    already understands.
    """
    extension = core()
    if extension is None:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            f"{MODULE} is not installed; build it with maturin or use "
            "agot.training.selfplay.batched_self_play"
        )

    spec = spec_for(plan, base_config)
    (
        states,
        actions,
        offsets,
        policy,
        players,
        outcomes,
        winners,
        steps,
        rounds,
    ) = extension.selfplay(spec, leaf_evaluator(model))

    # Each game records exactly one example per decision it took, so the per-game
    # step counts are also the example counts. Asserted rather than assumed: a
    # mismatch here would silently mis-attribute outcomes to positions.
    total = int(sum(steps))
    if total != int(states.shape[0]):
        raise RuntimeError(
            f"the driver recorded {states.shape[0]} examples for {total} decisions"
        )

    records: list[GameRecord] = []
    cursor = 0
    for index, taken in enumerate(steps):
        examples: list[Example] = []
        for offset in range(cursor, cursor + int(taken)):
            start, end = int(offsets[offset]), int(offsets[offset + 1])
            examples.append(
                Example(
                    state=states[offset],
                    actions=actions[start:end],
                    policy=policy[start:end].astype(np.float32),
                    player=int(players[offset]),
                    outcome=outcomes[offset].astype(np.float32),
                )
            )
        cursor += int(taken)
        records.append(
            GameRecord(
                examples=examples,
                winner=winners[index],
                steps=int(taken),
                rounds=int(rounds[index]),
            )
        )
    return records


def house_order() -> tuple[str, ...]:
    """The canonical house order the driver indexes `players` with."""
    return HOUSE_ORDER
