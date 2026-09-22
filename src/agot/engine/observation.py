"""Observation encoding: how a game state and its candidate actions become tensors.

The engine's action space is *combinatorial*: at any moment the number of legal
moves varies wildly (a planning decision can offer dozens of region/order pairs,
a bid offers a handful of integers). There is no fixed move list to score, so the
network cannot emit one logit per move like in chess.

Instead this module follows the pointer-network pattern used by AlphaZero-style
systems for large or open-ended move sets:

* :func:`encode_state` turns the whole position into one fixed-length vector.
* :func:`encode_action` turns a single candidate :class:`~agot.engine.actions.Action`
  into another fixed-length vector, using only *relational* information
  (terrain, who controls the region, unit counts, order type) rather than
  one-hot region ids. That keeps the encoding valid for any board layout.
* :func:`observe` bundles the state vector with the matrix of candidate-action
  vectors, so the policy head can score every legal move at once.

Everything is expressed over a *canonical* six-house order
(:data:`HOUSE_ORDER`). Games with fewer houses leave the unused slots zeroed, so
the network input size never changes with the player count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agot.engine.actions import Action, ActionKind
from agot.engine.enums import (
    ORDERS_BY_KEY,
    PHASES,
    SUPPLY_LIMITS,
    ActionStep,
    Terrain,
    Track,
    UnitType,
)
from agot.engine.game import Game
from agot.engine.map import load_map
from agot.engine.state import GameState

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

#: Feature counts of the two sub-blocks, derived once for shape assertions.
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
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _one_hot(value: object, choices: tuple) -> np.ndarray:
    vector = np.zeros(len(choices), dtype=np.float32)
    try:
        index = choices.index(value)
    except ValueError:
        return vector
    vector[index] = 1.0
    return vector


def _controller_relation(controller: str | None, actor: str | None) -> np.ndarray:
    """Four-way relation between a region's controller and the acting house."""
    vector = np.zeros(4, dtype=np.float32)
    if controller is None:
        vector[0] = 1.0  # unowned
    elif controller == actor:
        vector[2] = 1.0
    else:
        vector[3] = 1.0
    return vector


def _region_block(region, actor: str | None) -> np.ndarray:
    """Static plus dynamic description of one region, from ``actor``'s view."""
    game_map = load_map()
    static = game_map.regions[region.region_id]

    parts: list[np.ndarray] = [
        _one_hot(static.terrain, (Terrain.LAND, Terrain.SEA, Terrain.PORT)),
        np.array(
            [
                static.power / 3.0,
                static.supply / 3.0,
                static.mustering / 2.0,
            ],
            dtype=np.float32,
        ),
    ]

    # Navy first (4 dims): counts of each unit type, normalised.
    unit_counts = np.array(
        [region.units.get(unit, 0) for unit in UNIT_ORDER], dtype=np.float32
    )
    parts.append(np.clip(unit_counts / 4.0, 0.0, 1.5))
    parts.append(
        np.array(
            [
                min(1.0, region.total(include_routed=False) / 8.0),
                min(1.0, sum(region.routed.values()) / 8.0),
                1.0 if region.influence_token is not None else 0.0,
                1.0 if region.order is not None else 0.0,
                region.garrison / 4.0,
                region.neutral_force / 6.0,
            ],
            dtype=np.float32,
        )
    )
    parts.append(_controller_relation(region.controller, actor))
    parts.append(_one_hot(region.order, ORDER_KEYS))

    order_strength = 0.0
    if region.order is not None:
        order_strength = ORDERS_BY_KEY[region.order].strength / 2.0
    parts.append(
        np.array(
            [
                order_strength,
                1.0 if region.order_house == actor else 0.0,
                static.home == actor if static.home else False,
                1.0 if static.host is not None or static.sea is not None else 0.0,
            ],
            dtype=np.float32,
        )
    )
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------


def encode_state(game: Game, actor: str | None = None) -> np.ndarray:
    """Encode ``game`` as a single fixed-length feature vector.

    ``actor`` defaults to the house the engine is currently asking to act; it
    only affects a couple of "*is this mine*" style flags.
    """
    state = game.state
    actor = actor if actor is not None else state.active_house
    game_map = load_map()

    features: list[np.ndarray] = []
    for house_id in HOUSE_ORDER:
        if house_id in state.houses:
            features.append(_house_features(state, house_id, actor))
        else:
            features.append(np.zeros(_HOUSE_BLOCK, dtype=np.float32))

    # Global block.
    global_parts: list[np.ndarray] = [
        np.array(
            [state.round / max(1, state.max_rounds)],
            dtype=np.float32,
        ),
        _one_hot(state.phase, PHASES),
        _one_hot(state.action_step, STEPS),
        np.array(
            [
                min(1.0, state.wildling_threat / MAX_THREAT),
                len([h for h in state.houses.values() if not h.eliminated]) / N_HOUSES,
                len(game_map.regions) / 64.0,
                sum(
                    1
                    for region in state.regions.values()
                    if region.mustering_points > 0
                )
                / N_HOUSES,
            ],
            dtype=np.float32,
        ),
    ]
    features.append(np.concatenate(global_parts).astype(np.float32))

    vector = np.concatenate(features).astype(np.float32)
    if vector.shape != (STATE_DIM,):  # pragma: no cover - shape guard
        raise AssertionError(f"state feature size {vector.shape} != {STATE_DIM}")
    return vector


def _house_features(state: GameState, house_id: str, actor: str | None) -> np.ndarray:
    house = state.houses[house_id]
    castle_share = state.castle_count(house_id) / max(1, state.victory_castles)
    tracks = [
        state.track_position(house_id, Track.IRON_THRONE),
        state.track_position(house_id, Track.FIEFDOMS),
        state.track_position(house_id, Track.KINGS_COURT),
    ]
    land = 0
    ships = 0
    for region in state.regions.values():
        if region.controller != house_id:
            continue
        for unit, count in region.units.items():
            if unit is UnitType.SHIP:
                ships += count
            else:
                land += count
    return np.array(
        [
            1.0,
            1.0 if house.eliminated else 0.0,
            castle_share,
            min(1.0, house.power / MAX_POWER),
            house.supply_level / max(1, len(SUPPLY_LIMITS) - 1),
            *[1.0 - (position - 1) / (N_HOUSES - 1) for position in tracks],
            min(1.0, land / 16.0),
            min(1.0, ships / 12.0),
            min(1.0, len(house.orders_placed) / 12.0),
            len(house.house_cards_available) / CARD_SLOTS,
            1.0 if house_id == actor else 0.0,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------


def encode_action(game: Game, action: Action, actor: str | None = None) -> np.ndarray:
    """Encode one candidate action as a fixed-length feature vector."""
    state = game.state
    actor = actor if actor is not None else state.active_house

    card_vector = np.zeros(CARD_BLOCK, dtype=np.float32)
    if action.card is not None and action.house is not None:
        card_vector = _card_features(action.house, action.card)

    order_vector = np.zeros(ORDER_BLOCK, dtype=np.float32)
    order_vector[:N_ORDERS] = _one_hot(action.order, ORDER_KEYS)
    if action.order is not None:
        order_vector[N_ORDERS] = ORDERS_BY_KEY[action.order].strength / 2.0

    amount_vector = np.zeros(AMOUNT_BLOCK, dtype=np.float32)
    if action.amount is not None:
        amount_vector[0] = min(1.0, action.amount / MAX_POWER)
        amount_vector[1] = 1.0

    region_vector = (
        _region_block(state.regions[action.region], actor)
        if action.region is not None and action.region in state.regions
        else np.zeros(REGION_BLOCK, dtype=np.float32)
    )
    target_vector = (
        _region_block(state.regions[action.target], actor)
        if action.target is not None and action.target in state.regions
        else np.zeros(REGION_BLOCK, dtype=np.float32)
    )

    move_vector = np.zeros(MOVE_BLOCK, dtype=np.float32)
    if action.units is not None:
        for index, count in enumerate(action.units[:N_UNITS]):
            move_vector[index] = min(1.0, count / 4.0)
        move_vector[N_UNITS] = 1.0
    move_vector[N_UNITS + 1] = 1.0 if action.establish else 0.0

    vector = np.concatenate(
        [
            _one_hot(action.kind, tuple(ActionKind)),
            _one_hot(action.house, HOUSE_ORDER),
            np.array([1.0 if action.house == actor else 0.0], dtype=np.float32),
            region_vector,
            target_vector,
            card_vector,
            order_vector,
            amount_vector,
            _one_hot(action.unit, tuple(unit.value for unit in UNIT_ORDER)),
            move_vector,
        ]
    ).astype(np.float32)

    if vector.shape != (ACTION_DIM,):  # pragma: no cover - shape guard
        raise AssertionError(f"action feature size {vector.shape} != {ACTION_DIM}")
    return vector


def _card_features(house_id: str, card_id: str) -> np.ndarray:
    from agot.engine.map import load_house_cards

    deck = load_house_cards().deck(house_id)
    slot = next(
        (index for index, card in enumerate(deck) if card["id"] == card_id), -1
    )
    data = next((card for card in deck if card["id"] == card_id), None)
    vector = np.zeros(CARD_BLOCK, dtype=np.float32)
    vector[0] = 0.0 if data is None else 1.0
    if data is not None:
        vector[1] = data["strength"] / 5.0
        vector[2] = data["swords"] / 3.0
        vector[3] = data["forts"] / 3.0
    vector[4] = (slot + 1) / CARD_SLOTS if slot >= 0 else 0.0
    vector[5] = 1.0
    return vector


# ---------------------------------------------------------------------------
# Bundled observation
# ---------------------------------------------------------------------------


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
        actions = game.legal_actions()
    actor = game.current_player() or ""
    if not actions:
        raise ValueError("observe() called with no legal actions")
    matrix = np.stack([encode_action(game, action, actor) for action in actions])
    return Observation(
        actor=actor,
        actions=tuple(actions),
        state_features=encode_state(game, actor),
        action_features=matrix,
    )
