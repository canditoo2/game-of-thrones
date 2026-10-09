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
from functools import lru_cache

import numpy as np

from agot.engine.actions import Action, ActionKind
from agot.engine.enums import (
    ORDERS_BY_KEY,
    PHASES,
    SUPPLY_LIMITS,
    ActionStep,
    Phase,
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

#: Sides a Support order can be declared for, plus the option to refuse
#: (rulebook pages 17-18).
SUPPORT_SIDES: tuple[str, ...] = ("attacker", "defender", "refuse")
N_SIDES: int = len(SUPPORT_SIDES)

#: Lookup tables built once, so the encoders index instead of scanning tuples.
_ORDER_INDEX: dict[str, int] = {key: index for index, key in enumerate(ORDER_KEYS)}
_TERRAIN_INDEX: dict[Terrain, int] = {
    Terrain.LAND: 0,
    Terrain.SEA: 1,
    Terrain.PORT: 2,
}
_ACTION_KIND_CHOICES: tuple[ActionKind, ...] = tuple(ActionKind)
_UNIT_VALUE_CHOICES: tuple[str, ...] = tuple(unit.value for unit in UNIT_ORDER)
_ACTION_KIND_INDEX: dict[ActionKind, int] = {
    kind: index for index, kind in enumerate(_ACTION_KIND_CHOICES)
}
_HOUSE_INDEX: dict[str, int] = {
    house: index for index, house in enumerate(HOUSE_ORDER)
}
_UNIT_VALUE_INDEX: dict[str, int] = {
    value: index for index, value in enumerate(_UNIT_VALUE_CHOICES)
}
_SIDE_INDEX: dict[str, int] = {
    side: index for index, side in enumerate(SUPPORT_SIDES)
}
_PHASE_INDEX: dict[Phase, int] = {
    phase: index for index, phase in enumerate(PHASES)
}
_STEP_INDEX: dict[ActionStep, int] = {
    step: index for index, step in enumerate(STEPS)
}

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
# Small helpers
# ---------------------------------------------------------------------------


def _set_one_hot(
    value: object, index_map: dict, out: np.ndarray, offset: int
) -> None:
    """Set the slot matching ``value`` to 1.0 (``out`` is assumed zeroed)."""
    index = index_map.get(value)
    if index is not None:
        out[offset + index] = 1.0


@lru_cache(maxsize=1)
def _static_region_tables():
    """Per-region constants, computed once: ``(heads, homes, linked)``.

    ``heads`` holds the six static leading features (terrain one-hot plus the
    power/supply/mustering fractions). Reusing them avoids rebuilding tiny
    numpy arrays for every region of every candidate action.
    """
    game_map = load_map()
    heads: dict[str, np.ndarray] = {}
    homes: dict[str, str | None] = {}
    linked: dict[str, bool] = {}
    for rid, static in game_map.regions.items():
        head = np.zeros(6, dtype=np.float32)
        head[_TERRAIN_INDEX[static.terrain]] = 1.0
        head[3] = static.power / 3.0
        head[4] = static.supply / 3.0
        head[5] = static.mustering / 2.0
        heads[rid] = head
        homes[rid] = static.home
        linked[rid] = static.host is not None or static.sea is not None
    return heads, homes, linked


def _region_block(region, actor: str | None, state: GameState) -> np.ndarray:
    """Static plus dynamic description of one region, from ``actor``'s view.

    Written as an in-place fill of a single array: this runs twice per candidate
    action, so it deliberately avoids the many small temporary arrays the
    previous version allocated.
    """
    heads, homes, linked = _static_region_tables()
    rid = region.region_id
    out = np.empty(REGION_BLOCK, dtype=np.float32)
    out[:6] = heads[rid]

    index = 6
    units = region.units
    for offset, unit in enumerate(UNIT_ORDER):
        count = units.get(unit, 0) / 4.0
        out[index + offset] = 1.5 if count > 1.5 else count
    index += N_UNITS

    out[index] = min(1.0, region.total(include_routed=False) / 8.0)
    out[index + 1] = min(1.0, sum(region.routed.values()) / 8.0)
    out[index + 2] = 1.0 if region.influence_token is not None else 0.0
    out[index + 3] = 1.0 if region.order is not None else 0.0
    out[index + 4] = region.garrison / 4.0
    out[index + 5] = region.neutral_force / 6.0
    index += 6

    controller = region.controller
    # ``out`` comes from ``np.empty``: zero the whole controller block first so
    # no slot can leak uninitialised memory, then set the one flag that applies.
    out[index : index + 4] = 0.0
    if controller is None:
        out[index] = 1.0  # unowned
    elif controller == actor:
        out[index + 2] = 1.0
    else:
        out[index + 3] = 1.0
    index += 4

    visible = actor is None or state.orders_revealed or region.order_house == actor
    order_key = region.order if visible else None
    out[index : index + N_ORDERS] = 0.0
    if order_key is not None:
        slot = _ORDER_INDEX.get(order_key)
        if slot is not None:
            out[index + slot] = 1.0
    index += N_ORDERS

    if order_key is not None:
        out[index] = ORDERS_BY_KEY[order_key].strength / 2.0
    else:
        out[index] = 0.0  # hidden or absent order: no strength to show
    out[index + 1] = 1.0 if region.order_house == actor else 0.0
    out[index + 2] = 1.0 if homes[rid] is not None and homes[rid] == actor else 0.0
    out[index + 3] = 1.0 if linked[rid] else 0.0
    return out


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------


def encode_state(game: Game, actor: str | None = None) -> np.ndarray:
    """Encode ``game`` as a single fixed-length feature vector.

    ``actor`` defaults to the house the engine is currently asking to act; it
    only affects a couple of "*is this mine*" style flags.

    The vector is filled in place into one ``np.zeros`` rather than by allocating
    and concatenating a dozen small arrays: this runs once per search leaf.
    """
    state = game.state
    actor = actor if actor is not None else state.active_house

    # One pass over the board instead of one pass per house: accumulate each
    # house's army sizes, fleet size and castle count together.
    land = {house_id: 0 for house_id in state.houses}
    ships = {house_id: 0 for house_id in state.houses}
    castles = {house_id: 0 for house_id in state.houses}
    mustering_areas = 0
    for rid, region in state.regions.items():
        if region.mustering_points > 0:
            mustering_areas += 1
        controller = region.controller
        if controller is None or controller not in land:
            continue
        if rid in state.fortifications:
            castles[controller] += 1
        for unit, count in region.units.items():
            if unit is UnitType.SHIP:
                ships[controller] += count
            else:
                land[controller] += count

    vector = np.zeros(STATE_DIM, dtype=np.float32)
    offset = 0
    for house_id in HOUSE_ORDER:
        if house_id in state.houses:
            _write_house_block(
                state,
                house_id,
                actor,
                land[house_id],
                ships[house_id],
                castles[house_id],
                vector,
                offset,
            )
        # Houses absent from the game keep their zeroed block.
        offset += _HOUSE_BLOCK

    # Global block.
    vector[offset] = state.round / max(1, state.max_rounds)
    offset += 1
    _set_one_hot(state.phase, _PHASE_INDEX, vector, offset)
    offset += N_PHASES
    _set_one_hot(state.action_step, _STEP_INDEX, vector, offset)
    offset += N_STEPS
    vector[offset] = min(1.0, state.wildling_threat / MAX_THREAT)
    vector[offset + 1] = (
        sum(1 for h in state.houses.values() if not h.eliminated) / N_HOUSES
    )
    vector[offset + 2] = len(state.regions) / 64.0
    vector[offset + 3] = mustering_areas / N_HOUSES

    if vector.shape != (STATE_DIM,):  # pragma: no cover - shape guard
        raise AssertionError(f"state feature size {vector.shape} != {STATE_DIM}")
    return vector


def _write_house_block(
    state: GameState,
    house_id: str,
    actor: str | None,
    land: int,
    ships: int,
    castles: int,
    out: np.ndarray,
    offset: int,
) -> None:
    """Write one house's 13-slot block from stats gathered in a board pass."""
    house = state.houses[house_id]
    out[offset] = 1.0
    out[offset + 1] = 1.0 if house.eliminated else 0.0
    out[offset + 2] = castles / max(1, state.victory_castles)
    out[offset + 3] = min(1.0, house.power / MAX_POWER)
    out[offset + 4] = house.supply_level / max(1, len(SUPPLY_LIMITS) - 1)
    out[offset + 5] = 1.0 - (
        state.track_position(house_id, Track.IRON_THRONE) - 1
    ) / (N_HOUSES - 1)
    out[offset + 6] = 1.0 - (
        state.track_position(house_id, Track.FIEFDOMS) - 1
    ) / (N_HOUSES - 1)
    out[offset + 7] = 1.0 - (
        state.track_position(house_id, Track.KINGS_COURT) - 1
    ) / (N_HOUSES - 1)
    out[offset + 8] = min(1.0, land / 16.0)
    out[offset + 9] = min(1.0, ships / 12.0)
    out[offset + 10] = min(1.0, len(house.orders_placed) / 12.0)
    out[offset + 11] = len(house.house_cards_available) / CARD_SLOTS
    out[offset + 12] = 1.0 if house_id == actor else 0.0


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------


#: Memo of :func:`_region_block` results, keyed by ``id(region)``.
#:
#: Keying on the *object* rather than the region id is what makes the memo
#: shareable across the leaves of one inference batch: two leaves can hold
#: different objects under the same region id, because a determinization
#: swaps in a copy for the regions it resamples. Region objects are immutable
#: in place -- :meth:`agot.engine.state.GameState.mut` is the only write path,
#: and it replaces rather than edits -- so an object's identity pins its
#: contents. The cache must not outlive the objects it keys on (a freed
#: region's address could be reused), so it lives for one encode call, over
#: which the caller keeps the game(s) alive. A single cache is valid for one
#: ``(actor, orders_revealed)`` pair only; batched callers keep one per pair.
RegionCache = dict[int, np.ndarray]


def encode_action(
    game: Game,
    action: Action,
    actor: str | None = None,
    _region_cache: RegionCache | None = None,
) -> np.ndarray:
    """Encode one candidate action as a fixed-length feature vector.

    Thin wrapper over :func:`encode_action_into` that allocates the result. The
    batched paths (pointer-net inference, :func:`observe`) call
    ``encode_action_into`` directly to fill preallocated rows instead.
    """
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    return encode_action_into(game, action, out, actor, _region_cache)


def encode_action_into(
    game: Game,
    action: Action,
    out: np.ndarray,
    actor: str | None = None,
    region_cache: RegionCache | None = None,
) -> np.ndarray:
    """Write one action's features into ``out`` (length ``ACTION_DIM``).

    ``out`` must already be zeroed: the writer sets only the slots that apply,
    which is what lets the batched evaluator fill a whole action tensor
    without allocating a temporary array per candidate. ``region_cache`` is an
    optional memo of :func:`_region_block` results (see :data:`RegionCache`);
    many candidates in one position share a region (every order for an area,
    every march out of it) and the block depends only on the region object, the
    actor and the reveal flag.
    """
    state = game.state
    actor = actor if actor is not None else state.active_house

    offset = 0
    index = _ACTION_KIND_INDEX.get(action.kind)
    if index is not None:
        out[offset + index] = 1.0
    offset += N_ACTION_KINDS
    index = _HOUSE_INDEX.get(action.house)
    if index is not None:
        out[offset + index] = 1.0
    offset += N_HOUSES
    if action.house == actor:
        out[offset] = 1.0
    offset += 1

    # Region and target blocks. The row is already zeroed, so a field with no
    # region (or one outside the map) is simply skipped rather than copied from
    # an all-zero block, and the memo lookup is inlined to keep this hot path
    # free of per-candidate function calls.
    region_id = action.region
    if region_id is not None:
        region = state.regions.get(region_id)
        if region is not None:
            key = id(region)
            block = None if region_cache is None else region_cache.get(key)
            if block is None:
                block = _region_block(region, actor, state)
                if region_cache is not None:
                    region_cache[key] = block
            out[offset : offset + REGION_BLOCK] = block
    offset += REGION_BLOCK
    region_id = action.target
    if region_id is not None:
        region = state.regions.get(region_id)
        if region is not None:
            key = id(region)
            block = None if region_cache is None else region_cache.get(key)
            if block is None:
                block = _region_block(region, actor, state)
                if region_cache is not None:
                    region_cache[key] = block
            out[offset : offset + REGION_BLOCK] = block
    offset += REGION_BLOCK

    if action.card is not None and action.house is not None:
        out[offset : offset + CARD_BLOCK] = _card_features(action.house, action.card)
    offset += CARD_BLOCK

    index = _ORDER_INDEX.get(action.order)
    if index is not None:
        out[offset + index] = 1.0
    if action.order is not None:
        out[offset + N_ORDERS] = ORDERS_BY_KEY[action.order].strength / 2.0
    offset += ORDER_BLOCK

    if action.amount is not None:
        out[offset] = min(1.0, action.amount / MAX_POWER)
        out[offset + 1] = 1.0
    offset += AMOUNT_BLOCK

    index = _UNIT_VALUE_INDEX.get(action.unit)
    if index is not None:
        out[offset + index] = 1.0
    offset += N_UNITS

    if action.units is not None:
        for unit_index, count in enumerate(action.units[:N_UNITS]):
            out[offset + unit_index] = min(1.0, count / 4.0)
        out[offset + N_UNITS] = 1.0
    if action.establish:
        out[offset + N_UNITS + 1] = 1.0
    offset += MOVE_BLOCK

    index = _SIDE_INDEX.get(action.side)
    if index is not None:
        out[offset + index] = 1.0
    offset += N_SIDES

    if offset != ACTION_DIM:  # pragma: no cover - layout guard
        raise AssertionError(f"action feature size {offset} != {ACTION_DIM}")
    return out


@lru_cache(maxsize=1)
def _card_table() -> dict[tuple[str, str], np.ndarray]:
    """Precompute one feature vector per House card, keyed by (house, card id).

    The old version rebuilt the vector (and rescanned the deck for the card's
    slot) on every encoded action; a battle-heavy decision encodes several
    card actions at once, so the lookup is worth caching.
    """
    from agot.engine.map import load_house_cards

    table: dict[tuple[str, str], np.ndarray] = {}
    for house_id, deck in load_house_cards().cards.items():
        for slot, data in enumerate(deck):
            vector = np.zeros(CARD_BLOCK, dtype=np.float32)
            vector[0] = 1.0
            vector[1] = data["strength"] / 5.0
            vector[2] = data["swords"] / 3.0
            vector[3] = data["forts"] / 3.0
            vector[4] = (slot + 1) / CARD_SLOTS
            vector[5] = 1.0
            table[(house_id, data["id"])] = vector
    return table


def _card_features(house_id: str, card_id: str) -> np.ndarray:
    vector = _card_table().get((house_id, card_id))
    if vector is None:
        # Unknown card: keep the "a card field is present" marker only.
        vector = np.zeros(CARD_BLOCK, dtype=np.float32)
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
    cache: RegionCache = {}
    matrix = np.zeros((len(actions), ACTION_DIM), dtype=np.float32)
    for row, action in enumerate(actions):
        encode_action_into(game, action, matrix[row], actor, cache)
    return Observation(
        actor=actor,
        actions=tuple(actions),
        state_features=encode_state(game, actor),
        action_features=matrix,
    )
