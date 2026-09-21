"""Battle resolution.

The model follows the Second Edition rulebook:

* Combat Strength is the sum of each participating unit's printed Combat
  Strength, plus the March/Defend order modifier, plus support, plus the
  revealed House card's strength, plus a possible Valyrian Steel Blade bonus.
* The higher total wins; ties go to the house higher on the Fiefdoms track.
* Only the loser takes casualties: one unit per Sword icon on the *winner's*
  House card, minus one per Fortification icon on the *loser's* card, floored
  at zero.
* The loser then retreats, or its remaining units are destroyed.

A battle is stored with the two forces kept *separate* (``attacker_units`` and
``defender_units``). They briefly share a region on the board, but merging them
into one pool would make it impossible to tell whose units are whose, which is
what casualty removal needs.

Two deliberate simplifications are made here and called out in docs/roadmap.md:

* Support is resolved automatically in favour of each side's own Support
  orders. The real game lets any house support either side, which is where a
  lot of the table politics lives.
* Casualty and retreat selection is automatic (cheapest unit first, then the
  first legal retreat destination) rather than driven by a player decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agot.engine.enums import (
    ORDERS_BY_KEY,
    UNIT_STATS,
    OrderType,
    Track,
    UnitType,
)
from agot.engine.map import load_map
from agot.engine.state import GameState

#: Removal order when a player takes casualties automatically. Cheapest combat
#: value first, because a casualty destroys a unit regardless of its strength.
_CASUALTY_ORDER: tuple[UnitType, ...] = (
    UnitType.FOOTMAN,
    UnitType.SIEGE,
    UnitType.KNIGHT,
    UnitType.SHIP,
)


@dataclass
class BattleSide:
    """One participant in a battle."""

    house: str
    units_strength: int = 0
    order_strength: int = 0
    support_strength: int = 0
    card_strength: int = 0
    swords: int = 0
    forts: int = 0
    blade: bool = False

    @property
    def total(self) -> int:
        return (
            self.units_strength
            + self.order_strength
            + self.support_strength
            + self.card_strength
            + (1 if self.blade else 0)
        )


@dataclass
class BattleResult:
    """Outcome of a battle, ready to be applied to the state."""

    region: str
    attacker: BattleSide
    defender: BattleSide
    winner: str
    loser: str
    casualties: int
    tiebreak: bool = False
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strength building blocks (also used by the heuristic agent)
# ---------------------------------------------------------------------------


def units_strength(units: dict[UnitType, int]) -> int:
    """Combat strength of an explicit collection of units."""
    return sum(
        UNIT_STATS[unit].combat_strength * count
        for unit, count in units.items()
        if count > 0
    )


def unit_strength(state: GameState, region_id: str, house: str) -> int:
    """Combat strength of the units ``house`` currently controls in a region."""
    region = state.regions[region_id]
    if region.controller != house:
        return 0
    return units_strength(region.units)


def support_strength(state: GameState, region_id: str, house: str) -> int:
    """Support a house receives from its own adjacent Support orders.

    Only adjacent regions of the same *kind* contribute: land units support
    land battles, ships support sea battles.
    """
    game_map = load_map()
    battle_is_sea = game_map.regions[region_id].is_sea
    total = 0
    for neighbour in game_map.neighbours_of(region_id):
        region = state.regions[neighbour]
        if region.order is None or region.order_house != house:
            continue
        spec = ORDERS_BY_KEY[region.order]
        if spec.order_type is not OrderType.SUPPORT:
            continue
        if game_map.regions[neighbour].is_sea != battle_is_sea:
            continue
        for unit, count in region.units.items():
            if count <= 0:
                continue
            is_ship = unit is UnitType.SHIP
            if is_ship == battle_is_sea:
                total += count
        total += spec.strength
    return total


def order_modifier(state: GameState, region_id: str, house: str, side: str) -> int:
    """March bonus for an attacker or Defend bonus for a defender."""
    region = state.regions[region_id]
    if region.order is None or region.order_house != house:
        return 0
    spec = ORDERS_BY_KEY[region.order]
    if side == "attacker" and spec.order_type is OrderType.MARCH:
        return spec.strength
    if side == "defender" and spec.order_type is OrderType.DEFEND:
        return spec.strength
    return 0


def card_stats(house: str, card_id: str) -> dict:
    """Look up a house card's printed statistics."""
    from agot.engine.map import load_house_cards

    for card in load_house_cards().deck(house):
        if card["id"] == card_id:
            return card
    raise KeyError(f"{house} has no house card {card_id!r}")


def build_side(
    house: str,
    units: dict[UnitType, int],
    order_strength: int,
    support: int,
    card_id: str | None,
    use_blade: bool = False,
) -> BattleSide:
    """Assemble a :class:`BattleSide` from explicit pieces."""
    side = BattleSide(
        house=house,
        units_strength=units_strength(units),
        order_strength=order_strength,
        support_strength=support,
    )
    if card_id is not None:
        card = card_stats(house, card_id)
        side.card_strength = card["strength"]
        side.swords = card["swords"]
        side.forts = card["forts"]
    side.blade = use_blade
    return side


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_battle(
    state: GameState,
    region_id: str,
    attacker: str,
    defender: str,
    attacker_units: dict[UnitType, int],
    defender_units: dict[UnitType, int],
    attacker_card: str | None,
    defender_card: str | None,
    *,
    attacker_order_strength: int = 0,
    defender_order_strength: int = 0,
    attacker_blade: bool = False,
    defender_blade: bool = False,
) -> BattleResult:
    """Compute the outcome of a battle without mutating ``state``."""
    attacker_side = build_side(
        attacker,
        attacker_units,
        attacker_order_strength,
        support_strength(state, region_id, attacker),
        attacker_card,
        attacker_blade,
    )
    defender_side = build_side(
        defender,
        defender_units,
        defender_order_strength,
        support_strength(state, region_id, defender),
        defender_card,
        defender_blade,
    )

    notes: list[str] = []
    tiebreak = False
    if attacker_side.total > defender_side.total:
        winner, loser = attacker, defender
        winner_side, loser_side = attacker_side, defender_side
    elif defender_side.total > attacker_side.total:
        winner, loser = defender, attacker
        winner_side, loser_side = defender_side, attacker_side
    else:
        tiebreak = True
        attacker_position = state.track_position(attacker, Track.FIEFDOMS)
        defender_position = state.track_position(defender, Track.FIEFDOMS)
        if attacker_position < defender_position:
            winner, loser = attacker, defender
            winner_side, loser_side = attacker_side, defender_side
        else:
            winner, loser = defender, attacker
            winner_side, loser_side = defender_side, attacker_side
        notes.append("strength tied; decided on the Fiefdoms track")

    casualties = max(0, winner_side.swords - loser_side.forts)
    if casualties:
        notes.append(
            f"{loser} takes {casualties} casualt"
            f"{'y' if casualties == 1 else 'ies'}"
        )
    elif winner_side.swords:
        notes.append("all casualties cancelled by fortifications")

    return BattleResult(
        region=region_id,
        attacker=attacker_side,
        defender=defender_side,
        winner=winner,
        loser=loser,
        casualties=casualties,
        tiebreak=tiebreak,
        notes=notes,
    )


def _remove_units(units: dict[UnitType, int], count: int) -> None:
    """Remove ``count`` units in place, cheapest first."""
    for unit in _CASUALTY_ORDER:
        while count > 0 and units.get(unit, 0) > 0:
            units[unit] -= 1
            count -= 1
        if count == 0:
            break


def _prune(units: dict[UnitType, int]) -> dict[UnitType, int]:
    return {unit: count for unit, count in units.items() if count > 0}


def resolve_and_apply(
    state: GameState,
    region_id: str,
    attacker: str,
    defender: str,
    attacker_units: dict[UnitType, int],
    defender_units: dict[UnitType, int],
    attacker_card: str | None,
    defender_card: str | None,
    *,
    attacker_origin: str | None = None,
    attacker_order_strength: int = 0,
    defender_order_strength: int = 0,
    attacker_blade: bool = False,
    defender_blade: bool = False,
) -> BattleResult:
    """Resolve a battle and mutate ``state`` accordingly."""
    attacker_units = dict(attacker_units)
    defender_units = dict(defender_units)

    result = resolve_battle(
        state,
        region_id,
        attacker,
        defender,
        attacker_units,
        defender_units,
        attacker_card,
        defender_card,
        attacker_order_strength=attacker_order_strength,
        defender_order_strength=defender_order_strength,
        attacker_blade=attacker_blade,
        defender_blade=defender_blade,
    )

    region = state.regions[region_id]
    pre_routed = dict(region.routed)

    if result.loser == attacker:
        _remove_units(attacker_units, result.casualties)
        surviving_attacker = _prune(attacker_units)
        region.units = _prune(defender_units)
        region.controller = defender
        region.routed = pre_routed
        total_before = len(_prune(dict(attacker_units))) + result.casualties
        if surviving_attacker and total_before > result.casualties:
            _retreat_survivors(
                state, region_id, attacker, surviving_attacker, attacker_origin
            )
        else:
            result.notes.append(f"{attacker} was destroyed")
    else:
        _remove_units(defender_units, result.casualties)
        surviving_defender = _prune(defender_units)
        region.units = _prune(attacker_units)
        region.controller = attacker
        region.routed = {}
        if surviving_defender or pre_routed:
            retreating = dict(surviving_defender)
            for unit, count in pre_routed.items():
                retreating[unit] = retreating.get(unit, 0) + count
            _retreat_survivors(state, region_id, defender, retreating, None)
        else:
            result.notes.append(f"{defender} was destroyed")

    region.order = None
    region.order_house = None
    if region.is_empty() and not region.routed:
        _release_region(state, region_id)
    return result


def _retreat_survivors(
    state: GameState,
    region_id: str,
    house: str,
    units: dict[UnitType, int],
    preferred: str | None,
) -> None:
    """Route a defeated army to a legal destination, or destroy it."""
    sea = load_map().regions[region_id].is_sea
    options = legal_retreats(state, region_id, house, sea=sea)
    if preferred is not None and preferred in options:
        destination = preferred
    elif options:
        destination = options[0]
    else:
        state.note(f"{house} has no retreat from {region_id}; army destroyed")
        return
    target = state.regions[destination]
    for unit, count in units.items():
        if count > 0:
            target.routed[unit] = target.routed.get(unit, 0) + count
    target.controller = house


# ---------------------------------------------------------------------------
# Region-level helpers (also used for supply enforcement and cleaning up)
# ---------------------------------------------------------------------------


def apply_casualties(
    state: GameState, region_id: str, house: str, count: int
) -> list[UnitType]:
    """Remove ``count`` units controlled by ``house`` in a region, cheapest first."""
    region = state.regions[region_id]
    if region.controller != house:
        return []
    removed: list[UnitType] = []
    for unit in _CASUALTY_ORDER:
        while count > 0 and region.units.get(unit, 0) > 0:
            region.units[unit] -= 1
            if region.units[unit] == 0:
                del region.units[unit]
            count -= 1
            removed.append(unit)
        if count == 0:
            break
    if region.is_empty() and not region.routed:
        _release_region(state, region_id)
    return removed


def legal_retreats(
    state: GameState, region_id: str, house: str, *, sea: bool = False
) -> list[str]:
    """Adjacent regions a defeated force may retreat into."""
    game_map = load_map()
    options: list[str] = []
    for neighbour in game_map.neighbours_of(region_id):
        static = game_map.regions[neighbour]
        if sea and not static.is_sea:
            continue
        if not sea and not static.is_land:
            continue
        target = state.regions[neighbour]
        if target.total(include_routed=False) > 0:
            continue
        if target.controller is not None and target.controller != house:
            continue
        options.append(neighbour)
    return sorted(options)


def retreat(
    state: GameState, region_id: str, house: str, target: str, *, sea: bool = False
) -> None:
    """Move a controlled army out of ``region_id`` and rout it."""
    source = state.regions[region_id]
    units = {unit: count for unit, count in source.units.items() if count > 0}
    source.units = {}
    _retreat_survivors(state, region_id, house, units, target)


def destroy_army(state: GameState, region_id: str) -> None:
    """Wipe out everything in a region (used when no retreat is possible)."""
    region = state.regions[region_id]
    region.units = {}
    region.routed = {}
    _release_region(state, region_id)


def _release_region(state: GameState, region_id: str) -> None:
    """Drop control and any order token from a now-empty region."""
    region = state.regions[region_id]
    if region.influence_token is None:
        region.controller = None
    else:
        region.controller = region.influence_token
    region.order = None
    region.order_house = None
