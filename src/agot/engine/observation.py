"""Observation encoding: the Python face of the Rust encoder.

The encoder moved to Rust with the rest of the engine. This module keeps the
layout constants the Python layer (and its checkpoints) depend on, and exposes
the same functions as before; each call forwards to the extension. There is one
encoder now, so a checkpoint can no longer disagree with the engine about how a
position or a candidate action is represented.

The engine's action space is *combinatorial*: at any moment the number of legal
moves varies wildly (a planning decision can offer dozens of region/order pairs,
a bid offers a handful of integers). There is no fixed move list to score, so the
network cannot emit one logit per move like in chess. :func:`encode_state`
turns the position into one vector and :func:`encode_action` turns a candidate
into another, using only relational information (terrain, who controls the
region, unit counts, order type) rather than one-hot region ids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agot.engine.actions import Action, ActionKind
from agot.engine.enums import ORDERS_BY_KEY, PHASES, ActionStep, UnitType
from agot.engine.game import Game

# ---------------------------------------------------------------------------
# Canonical layout
# ---------------------------------------------------------------------------

#: The order in which houses appear in every encoded vector. Fixed so that a
#: checkpoint trained on one table can be loaded for another.
HOUSE_ORDER: tuple[str, ...] = (
    "baratheon",
    "lannister",
    "stark",
    "greyjoy",
    "tyrell",
    "martell",
)
N_HOUSES: int = len(HOUSE_ORDER)

ORDER_KEYS: tuple[str, ...] = tuple(ORDERS_BY_KEY)
N_ORDERS: int = len(ORDER_KEYS)

UNIT_ORDER: tuple[UnitType, ...] = (
    UnitType.FOOTMAN,
    UnitType.KNIGHT,
    UnitType.SIEGE,
    UnitType.SHIP,
)
N_UNITS: int = len(UNIT_ORDER)

N_ACTION_KINDS: int = len(ActionKind)
N_PHASES: int = len(PHASES)
N_STEPS: int = len(ActionStep)

STEPS: tuple[ActionStep, ...] = tuple(ActionStep)

#: House-card slots: every house brings exactly seven cards.
CARD_SLOTS: int = 7

#: Soft normalisation constants. They only need to keep features in a sane range.
MAX_POWER = 20.0
MAX_THREAT = 12.0

#: Feature counts of the sub-blocks, kept for tools and shape assertions.
_HOUSE_BLOCK = 13
HOUSE_BLOCK = _HOUSE_BLOCK
_GLOBAL_BLOCK = 1 + N_PHASES + N_STEPS + 4
# terrain(3) + power/supply/mustering(3) + unit counts + flags(6)
# + controller relation(4) + order one-hot + order flags(4)
REGION_BLOCK = 3 + 3 + N_UNITS + 6 + 4 + N_ORDERS + 4
CARD_BLOCK = 6
ORDER_BLOCK = N_ORDERS + 1
AMOUNT_BLOCK = 2
#: March detail: a normalised count per unit type, a "subset given" flag and an
#: "establish control" flag.
MOVE_BLOCK = N_UNITS + 2

#: Sides a Support order can be declared for, plus the option to refuse
#: (rulebook pages 17-18).
SUPPORT_SIDES: tuple[str, ...] = ("attacker", "defender", "refuse")
N_SIDES: int = len(SUPPORT_SIDES)

STATE_DIM: int = N_HOUSES * _HOUSE_BLOCK + _GLOBAL_BLOCK
ACTION_DIM: int = (
    N_ACTION_KINDS
    + N_HOUSES
    + 1  # is_active
    + REGION_BLOCK
    + REGION_BLOCK
    + CARD_BLOCK
    + ORDER_BLOCK
    + AMOUNT_BLOCK
    + N_UNITS
    + MOVE_BLOCK
    + N_SIDES
)


# ---------------------------------------------------------------------------
# The extension-backed encoder
# ---------------------------------------------------------------------------


def _inner(game: Game):
    """The extension's session object behind a Python-shaped game."""
    return getattr(game, "_inner", game)


def _rust_action(game: Game, action: Action):
    """The same action, in the extension's own representation.

    This deliberately round-trips through the engine's action builder: an action
    the engine would not recognise cannot be encoded by accident.
    """
    return _inner(game).action(
        action.kind.value,
        house=action.house,
        region=action.region,
        target=action.target,
        order=action.order,
        card=action.card,
        unit=action.unit,
        side=action.side,
        amount=action.amount,
        units=action.units,
        establish=action.establish,
    )


def encode_state(game: Game, actor: str | None = None) -> np.ndarray:
    """Encode the position as one fixed-length float32 vector."""
    return _inner(game).encode_state(actor)


def encode_actions(
    game: Game, actions: list[Action] | tuple[Action, ...], actor: str | None = None
) -> np.ndarray:
    """Encode every candidate action, one row per action."""
    built = [_rust_action(game, action) for action in actions]
    return _inner(game).encode_actions(built, actor)


def encode_action(game: Game, action: Action, actor: str | None = None) -> np.ndarray:
    """Encode one candidate action as a fixed-length float32 vector."""
    return encode_actions(game, [action], actor)[0]


def encode_action_into(
    game: Game,
    action: Action,
    out: np.ndarray,
    actor: str | None = None,
    region_cache: dict | None = None,
) -> np.ndarray:
    """Write one action's features into ``out``.

    ``region_cache`` is accepted for call-site compatibility; the Rust encoder
    owns its own region memo, so the argument is unused.
    """
    del region_cache
    out[:] = encode_action(game, action, actor)
    return out


@dataclass
class Observation:
    """Everything the policy/value network needs at one decision point."""

    actor: str
    actions: tuple[Action, ...]
    state_features: np.ndarray
    action_features: np.ndarray  # shape (len(actions), ACTION_DIM)

    def __len__(self) -> int:
        return len(self.actions)


def observe(game: Game, actions: list[Action] | None = None) -> Observation:
    """Bundle the current state with its legal actions.

    Raises :class:`ValueError` if there are no legal actions, because the
    network cannot be asked to choose from an empty set.
    """
    if actions is None:
        return game.observe()
    actions = tuple(actions)
    if not actions:
        raise ValueError("observe() called with no legal actions")
    actor = game.current_player() or ""
    return Observation(
        actor=actor,
        actions=actions,
        state_features=encode_state(game, actor),
        action_features=encode_actions(game, actions, actor),
    )
