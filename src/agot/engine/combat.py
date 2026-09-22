"""Battle resolution.

The model follows the Second Edition rulebook:

* Combat Strength is the sum of each participating unit's printed Combat
  Strength, plus the March/Defend order modifier, plus support, plus the
  defending Garrison's strength, plus the revealed House card's strength, plus
  a possible Valyrian Steel Blade bonus.
* A Siege Engine contributes 4 Combat Strength only when attacking (or
  supporting an attack against) an area containing a Castle or Stronghold;
  otherwise it contributes 0.
* The higher total wins; ties go to the house higher on the Fiefdoms track.
* Only the loser takes casualties: one unit per Sword icon on the *winner's*
  House card, minus one per Fortification icon on the *loser's* card, floored
  at zero.
* The loser then retreats (or its remaining units are destroyed). Routed units
  may never be chosen as casualties, and are destroyed if forced to retreat.

A battle is stored with the two forces kept *separate* (``attacker_units`` and
``defender_units``). They briefly share a region on the board, but merging them
into one pool would make it impossible to tell whose units are whose, which is
what casualty removal needs.

Two deliberate simplifications are still made here and called out in
docs/roadmap.md:

* Support is resolved automatically in favour of each side's own Support
  orders. The real game lets any house support either side.
* Casualty and retreat selection is automatic (cheapest unit first, then the
  first legal retreat destination) rather than driven by a player decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agot.engine import house_cards
from agot.engine.enums import (
    ORDERS_BY_KEY,
    UNIT_STATS,
    OrderType,
    Terrain,
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
    garrison_strength: int = 0
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
            + self.garrison_strength
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
    #: The abilities that shaped this battle; :func:`resolve_and_apply` commits
    #: the deferred board changes and the post-combat effects through it.
    context: house_cards.BattleContext | None = None


# ---------------------------------------------------------------------------
# Strength building blocks (also used by the heuristic agent)
# ---------------------------------------------------------------------------


def units_strength(
    units: dict[UnitType, int], *, siege_active: bool = False
) -> int:
    """Combat strength of an explicit collection of units.

    ``siege_active`` must be true when the units are attacking (or supporting
    an attack against) an area containing a Castle or Stronghold, which is the
    only time Siege Engines contribute their printed strength.
    """
    total = 0
    for unit, count in units.items():
        if count <= 0:
            continue
        if unit is UnitType.SIEGE and not siege_active:
            continue
        total += UNIT_STATS[unit].combat_strength * count
    return total


def unit_strength(
    state: GameState, region_id: str, house: str, *, siege_active: bool = False
) -> int:
    """Combat strength of the units ``house`` currently controls in a region."""
    region = state.regions[region_id]
    if region.controller != house:
        return 0
    return units_strength(region.units, siege_active=siege_active)


def _contributing_units(
    state: GameState, region_id: str, battle_is_sea: bool, supporter_is_attacker: bool
) -> dict[UnitType, int]:
    """Which units in a supporting region can actually add strength."""
    game_map = load_map()
    static = game_map.regions[region_id]
    region = state.regions[region_id]
    contributing: dict[UnitType, int] = {}

    # Ships in an adjacent sea area may support a land battle; ships in a Port
    # may support the adjacent sea, but never a land area.
    ships_ok = static.terrain is Terrain.SEA
    if battle_is_sea:
        ships_ok = static.terrain in (Terrain.SEA, Terrain.PORT)

    for unit, count in region.units.items():
        if count <= 0:
            continue
        if unit is UnitType.SHIP:
            if ships_ok:
                contributing[unit] = contributing.get(unit, 0) + count
        elif not battle_is_sea:
            # Footmen/Knights/Siege may never support a battle in a sea area.
            contributing[unit] = contributing.get(unit, 0) + count
    if not contributing:
        return {}
    # A supporting Siege Engine only contributes when supporting the attacker
    # against a Castle or Stronghold.
    if not (supporter_is_attacker and game_map.regions[region_id].is_fortification):
        contributing.pop(UnitType.SIEGE, None)
    return contributing


def support_strength(
    state: GameState,
    region_id: str,
    house: str,
    *,
    supporting_attacker: bool,
) -> int:
    """Support a house receives from its own adjacent Support orders.

    ``region_id`` is the embattled area. Support is only ever granted to a
    house by its own orders here (cross-house support is out of scope).
    """
    total, _ = support_detail(
        state, region_id, house, supporting_attacker=supporting_attacker
    )
    return total


def support_detail(
    state: GameState,
    region_id: str,
    house: str,
    *,
    supporting_attacker: bool,
) -> tuple[int, dict[UnitType, int]]:
    """Support strength *and* the units providing it.

    The House-card abilities of Kevan Lannister, Victarion Greyjoy and
    Salladhor Saan care about *which* units are supporting, so the aggregate is
    returned alongside the strength.
    """
    game_map = load_map()
    battle_is_sea = game_map.regions[region_id].is_sea
    total = 0
    units: dict[UnitType, int] = {}
    for neighbour in game_map.neighbours_of(region_id):
        region = state.regions[neighbour]
        if region.order is None or region.order_house != house:
            continue
        spec = ORDERS_BY_KEY[region.order]
        if spec.order_type is not OrderType.SUPPORT:
            continue
        contributing = _contributing_units(
            state, neighbour, battle_is_sea, supporting_attacker
        )
        for unit, count in contributing.items():
            units[unit] = units.get(unit, 0) + count
        total += units_strength(
            contributing,
            siege_active=supporting_attacker
            and game_map.regions[region_id].is_fortification,
        )
        total += spec.strength
    return total, units


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
    *,
    garrison_strength: int = 0,
    siege_active: bool = False,
) -> BattleSide:
    """Assemble a :class:`BattleSide` from explicit pieces."""
    side = BattleSide(
        house=house,
        units_strength=units_strength(units, siege_active=siege_active),
        order_strength=order_strength,
        support_strength=support,
        garrison_strength=garrison_strength,
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
    attacker_order: str | None = None,
    attacker_origin: str | None = None,
) -> BattleResult:
    """Compute the outcome of a battle without mutating ``state``.

    House-card abilities are applied here too, but only the pure ones: an
    ability that moves an Order token, an Influence track or a card between hand
    and discard is *recorded* on the returned :class:`~agot.engine.house_cards.BattleContext`
    and committed by :func:`resolve_and_apply`.
    """
    attacker_units = dict(attacker_units)
    defender_units = dict(defender_units)

    game_map = load_map()
    is_fortification = game_map.regions[region_id].is_fortification
    region = state.regions[region_id]

    attacker_support, attacker_support_units = support_detail(
        state, region_id, attacker, supporting_attacker=True
    )
    defender_support, defender_support_units = support_detail(
        state, region_id, defender, supporting_attacker=False
    )

    attacker_side = build_side(
        attacker,
        attacker_units,
        attacker_order_strength,
        attacker_support,
        attacker_card,
        attacker_blade,
        siege_active=is_fortification,
    )
    defender_side = build_side(
        defender,
        defender_units,
        defender_order_strength,
        defender_support,
        defender_card,
        defender_blade,
        garrison_strength=region.garrison,
        # A defending Siege Engine never contributes its printed strength.
        siege_active=False,
    )

    ctx = house_cards.BattleContext(
        state=state,
        region_id=region_id,
        attacker=attacker,
        defender=defender,
        attacker_units=attacker_units,
        defender_units=defender_units,
        attacker_card=attacker_card,
        defender_card=defender_card,
        attacker_supported=bool(attacker_support_units) or attacker_support > 0,
        defender_supported=bool(defender_support_units) or defender_support > 0,
        attacker_support_units=attacker_support_units,
        defender_support_units=defender_support_units,
        attacker_order=attacker_order,
        attacker_origin=attacker_origin,
        attacker_side=attacker_side,
        defender_side=defender_side,
    )
    house_cards.apply_pre_battle(ctx)

    notes: list[str] = list(ctx.notes)
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
    if loser in ctx.casualty_immunity:
        casualties = 0
        if winner_side.swords:
            notes.append(f"{loser} ignores casualties (The Blackfish)")
    elif casualties:
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
        context=ctx,
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
    attacker_order: str | None = None,
    attacker_order_strength: int = 0,
    defender_order_strength: int = 0,
    attacker_blade: bool = False,
    defender_blade: bool = False,
) -> BattleResult:
    """Resolve a battle and mutate ``state`` accordingly."""
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
        attacker_order=attacker_order,
        attacker_origin=attacker_origin,
    )
    ctx = result.context
    assert ctx is not None  # resolve_battle always builds one
    # Card swaps, Order removals and Doran's influence move are deferred until
    # now so that resolve_battle stays free of state mutation.
    house_cards.commit_deferred(ctx)
    attacker_units = ctx.attacker_units
    defender_units = ctx.defender_units

    region = state.regions[region_id]
    pre_routed = dict(region.routed)

    robb = house_cards.robb_chose_retreat(ctx, result.winner)
    arianne = house_cards.arianne_holds(ctx, result.loser)

    if result.loser == attacker:
        _remove_units(attacker_units, result.casualties)
        surviving = _prune(attacker_units)
        region.units = _prune(defender_units)
        region.controller = defender
        # The defender keeps its own routed units, order token and Power token.
        region.routed = pre_routed
        if surviving:
            _retreat_attacker(state, region_id, attacker, surviving, attacker_origin)
        elif result.casualties:
            result.notes.append(f"{attacker} was destroyed")
    else:
        _remove_units(defender_units, result.casualties)
        surviving_defender = _prune(defender_units)
        region.garrison = 0
        region.order = None
        region.order_house = None
        if arianne:
            # Arianne Martell: the attacker may not move in; it returns to the
            # area it marched from. The defender still retreats.
            _return_attacker(
                state, attacker, _prune(attacker_units), attacker_origin
            )
            region.units = {}
            region.routed = {}
        else:
            region.units = _prune(attacker_units)
            region.controller = attacker
            # The defender's routed units are destroyed (routed units forced to
            # retreat are destroyed). The Garrison is permanently removed.
            region.routed = {}
            # Winning the attacker removes the defender's Order and Power tokens.
            region.influence_token = None
        if surviving_defender:
            _retreat_defender(
                state,
                region_id,
                defender,
                surviving_defender,
                attacker_origin,
                prefer_min_loss=robb,
            )
        elif result.casualties:
            result.notes.append(f"{defender} was destroyed")

    if region.is_empty() and not region.routed and region.garrison == 0:
        _release_region(state, region_id)

    house_cards.apply_post_battle(ctx, result)
    _try_loras(ctx, result)
    state.log.extend(result.notes)
    return result


def _return_attacker(
    state: GameState,
    house: str,
    units: dict[UnitType, int],
    origin: str | None,
) -> None:
    """Send a blocked attacker (Arianne Martell) back where it marched from."""
    if not units:
        return
    if origin is None:  # pragma: no cover - defensive
        return
    target = state.regions[origin]
    for unit, count in units.items():
        target.units[unit] = target.units.get(unit, 0) + count
    if target.controller in (None, house):
        target.controller = house
    _enforce_supply_at(state, origin, house)


def _try_loras(ctx: house_cards.BattleContext, result: BattleResult) -> None:
    """Loras Tyrell re-places the March order in the conquered area."""
    if result.winner != ctx.attacker:
        return
    if house_cards.ability_of(ctx.attacker_card) != "loras_tyrell":
        return
    if ctx.attacker_order is None:
        return
    region = ctx.state.regions[ctx.region_id]
    region.order = ctx.attacker_order
    region.order_house = ctx.attacker
    ctx.notes.append("Loras Tyrell re-places the March order to resolve again")


def _retreat_attacker(
    state: GameState,
    region_id: str,
    house: str,
    units: dict[UnitType, int],
    origin: str | None,
) -> None:
    """A losing attacker must retreat back to the area it marched from."""
    if origin is None:
        state.note(f"{house} has no retreat from {region_id}; army destroyed")
        return
    retreaters, destroyed = _split_non_retreaters(units, naval=False)
    if destroyed:
        state.note(f"{house} Siege Engine(s) destroyed retreating from {region_id}")
    if not retreaters:
        return
    if not _can_retreat_to(state, origin, house, naval=False):
        state.note(f"{house} cannot retreat to {origin}; army destroyed")
        return
    _move_routed(state, origin, house, retreaters)


def _retreat_defender(
    state: GameState,
    region_id: str,
    house: str,
    units: dict[UnitType, int],
    attacker_origin: str | None,
    *,
    prefer_min_loss: bool = False,
) -> None:
    """A losing defender retreats to a legal adjacent area, or is destroyed."""
    naval = all(unit is UnitType.SHIP for unit in units)
    retreaters, destroyed = _split_non_retreaters(units, naval=naval)
    if destroyed:
        state.note(f"{house} Siege Engine(s) destroyed retreating from {region_id}")
    if not retreaters:
        return
    options = legal_retreats(
        state, region_id, house, naval=naval, exclude_origin=attacker_origin
    )
    if not options:
        state.note(f"{house} has no retreat from {region_id}; army destroyed")
        return
    destination = options[0]
    if prefer_min_loss:
        # Robb Stark chooses the retreat where the loser loses the fewest units:
        # prefer a destination that keeps the house within Supply.
        destination = next(
            (
                option
                for option in options
                if _retreat_keeps_supply(state, option, house, retreaters)
            ),
            options[0],
        )
    _move_routed(state, destination, house, retreaters)


def _retreat_keeps_supply(
    state: GameState, destination: str, house: str, units: dict[UnitType, int]
) -> bool:
    """Whether routing ``units`` into ``destination`` keeps the house legal."""
    target = state.regions[destination]
    saved = dict(target.routed)
    for unit, count in units.items():
        if count > 0:
            target.routed[unit] = target.routed.get(unit, 0) + count
    within = state.within_supply(house)
    target.routed = saved
    return within


def _split_non_retreaters(
    units: dict[UnitType, int], *, naval: bool
) -> tuple[dict[UnitType, int], dict[UnitType, int]]:
    """Separate units that can retreat from those that are destroyed instead."""
    retreaters: dict[UnitType, int] = {}
    destroyed: dict[UnitType, int] = {}
    for unit, count in units.items():
        if count <= 0:
            continue
        if unit is UnitType.SIEGE:
            destroyed[unit] = destroyed.get(unit, 0) + count
        else:
            retreaters[unit] = retreaters.get(unit, 0) + count
    return retreaters, destroyed


def _move_routed(
    state: GameState, destination: str, house: str, units: dict[UnitType, int]
) -> None:
    target = state.regions[destination]
    for unit, count in units.items():
        if count > 0:
            target.routed[unit] = target.routed.get(unit, 0) + count
    if target.controller in (None, house):
        target.controller = house
    _enforce_supply_at(state, destination, house)


def _enforce_supply_at(state: GameState, region_id: str, house: str) -> None:
    """Destroy routed/ready units in a region until Supply is satisfied."""
    region = state.regions[region_id]
    guard = 0
    while not state.within_supply(house) and guard < 100:
        guard += 1
        removed = False
        for unit in _CASUALTY_ORDER:
            if region.routed.get(unit, 0) > 0:
                region.routed[unit] -= 1
                if region.routed[unit] == 0:
                    del region.routed[unit]
                removed = True
                break
            if region.units.get(unit, 0) > 0:
                region.units[unit] -= 1
                if region.units[unit] == 0:
                    del region.units[unit]
                removed = True
                break
        if not removed:
            break


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
    if region.is_empty() and not region.routed and region.garrison == 0:
        _release_region(state, region_id)
    return removed


def _enemy_presence(state: GameState, region_id: str, house: str) -> bool:
    region = state.regions[region_id]
    if region.controller not in (None, house):
        return True
    if region.controller is None and region.total(include_routed=False) > 0:
        return True
    # A Neutral Force token blocks retreats just like a defending army.
    if region.neutral_force > 0:
        return True
    if region.influence_token not in (None, house):
        return True
    return False


def _can_retreat_to(
    state: GameState, destination: str, house: str, *, naval: bool
) -> bool:
    game_map = load_map()
    static = game_map.regions[destination]
    if naval:
        if static.is_land:
            return False
    elif not static.is_land:
        return False
    if _enemy_presence(state, destination, house):
        return False
    return True


def _ship_transport_targets(state: GameState, house: str, origin: str) -> set[str]:
    """Land areas reachable from ``origin`` through a friendly Ship chain.

    Routed Ships may still transport (rulebook page 23).
    """
    game_map = load_map()
    naval = {
        rid
        for rid, region in state.regions.items()
        if region.controller == house
        and game_map.regions[rid].is_sea
        and (
            region.units.get(UnitType.SHIP, 0)
            + region.routed.get(UnitType.SHIP, 0)
            > 0
        )
    }
    frontier = [n for n in game_map.sea_neighbours_of(origin) if n in naval]
    seen = set(frontier)
    while frontier:
        current = frontier.pop()
        for neighbour in game_map.sea_neighbours_of(current):
            if neighbour in naval and neighbour not in seen:
                seen.add(neighbour)
                frontier.append(neighbour)
    targets: set[str] = set()
    for sea in seen:
        targets |= set(game_map.land_neighbours_of(sea))
    return targets


def legal_retreats(
    state: GameState,
    region_id: str,
    house: str,
    *,
    naval: bool = False,
    exclude_origin: str | None = None,
) -> list[str]:
    """Adjacent regions a defeated force may retreat into."""
    game_map = load_map()
    options: list[str] = []
    for neighbour in game_map.neighbours_of(region_id):
        if neighbour == exclude_origin:
            continue
        static = game_map.regions[neighbour]
        if not _can_retreat_to(state, neighbour, house, naval=naval):
            continue
        if static.is_port:
            # A Port can hold at most three ships, and only its owner may use it.
            if not naval or state.regions[neighbour].controller != house:
                continue
            if state.regions[neighbour].total(include_routed=True) >= 3:
                continue
        options.append(neighbour)
    if not naval:
        for neighbour in _ship_transport_targets(state, house, region_id):
            if neighbour == exclude_origin or neighbour in options:
                continue
            if _can_retreat_to(state, neighbour, house, naval=False):
                options.append(neighbour)
    return sorted(options)


def retreat(
    state: GameState, region_id: str, house: str, target: str, *, naval: bool = False
) -> None:
    """Move a controlled army out of ``region_id`` and rout it."""
    source = state.regions[region_id]
    units = {unit: count for unit, count in source.units.items() if count > 0}
    source.units = {}
    retreaters, _ = _split_non_retreaters(units, naval=naval)
    _move_routed(state, target, house, retreaters)


def destroy_army(state: GameState, region_id: str) -> None:
    """Wipe out everything in a region (used when no retreat is possible)."""
    region = state.regions[region_id]
    region.units = {}
    region.routed = {}
    region.garrison = 0
    _release_region(state, region_id)


def _release_region(state: GameState, region_id: str) -> None:
    """Drop control and any order token from a now-empty region."""
    region = state.regions[region_id]
    if region.influence_token is not None:
        region.controller = region.influence_token
    else:
        region.controller = region.owner
    region.order = None
    region.order_house = None
