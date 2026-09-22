"""House-card special abilities.

Every card in ``agot/data/house_cards.json`` carries an ``ability`` id. This
module turns those ids into concrete effects during a battle. It is split out of
:mod:`agot.engine.combat` so that the pure strength arithmetic stays readable.

Three groups of effects exist:

* **Passive modifiers** change a side's strength, swords or forts before the
  totals are compared (Stannis, Davos, Theon, Asha, Nymeria, Balon, Kevan,
  Victarion, Salladhor Saan, Catelyn).
* **Immediate effects** happen at reveal and touch the board (Mace destroys a
  Footman, the Queen of Thorns removes an adjacent Order, Doran moves an
  opponent down an Influence track, Tyrion and Aeron swap cards).
* **Post-combat effects** apply once the winner is known (Renly, Tywin, Cersei,
  Robb, Loras, Roose, Patchface, Arianne).

The rulebook lets the card's owner make several sub-choices (which Footman, which
Order, which alternate card, which retreat area). The engine resolves those with
deterministic defaults: strongest House card, cheapest Footman, first legal
Order, and a retreat that keeps Supply. See ``docs/roadmap.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from agot.engine.enums import ORDERS_BY_KEY, OrderType, Track, UnitType
from agot.engine.map import load_house_cards, load_map
from agot.engine.state import GameState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agot.engine.combat import BattleResult, BattleSide


# ---------------------------------------------------------------------------
# Ability lookup
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _cards_by_id() -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for cards in load_house_cards().cards.values():
        for card in cards:
            table[card["id"]] = card
    return table


def card_data(card_id: str | None) -> dict[str, Any] | None:
    if card_id is None:
        return None
    return _cards_by_id().get(card_id)


def ability_of(card_id: str | None) -> str:
    """The machine-readable ability id of a House card (``"none"`` if any)."""
    card = card_data(card_id)
    return (card or {}).get("ability", "none")


def card_score(card_id: str | None) -> float:
    """A crude value for choosing between cards (strength plus icons)."""
    card = card_data(card_id)
    if card is None:
        return 0.0
    return card["strength"] + 0.8 * card["swords"] + 0.5 * card["forts"]


#: Every ability id the engine actually implements. A data-integrity test checks
#: that the JSON never names an ability outside this set.
IMPLEMENTED_ABILITIES: frozenset[str] = frozenset(
    {
        "stannis_baratheon",
        "davos_seaworth",
        "theon_greyjoy",
        "asha_greyjoy",
        "nymeria_sand",
        "balon_greyjoy",
        "kevan_lannister",
        "victarion_greyjoy",
        "salladhor_saan",
        "catelyn_stark",
        "blackfish",
        "mace_tyrell",
        "queen_of_thorns",
        "doran_martell",
        "tyrion_lannister",
        "aeron_damphair",
        "tywin_lannister",
        "renly_baratheon",
        "cersei_lannister",
        "robb_stark",
        "roose_bolton",
        "patchface",
        "loras_tyrell",
        "arianne_martell",
    }
)


# ---------------------------------------------------------------------------
# Battle context
# ---------------------------------------------------------------------------


@dataclass
class BattleContext:
    """Everything the abilities need about one battle.

    ``attacker_units`` / ``defender_units`` are *effective* unit dictionaries:
    an immediate ability such as Mace Tyrell's may already have removed a unit.
    ``card_swaps``, ``power_costs``, ``orders_to_remove`` and ``influence_move``
    are decided here (purely) and committed by
    :func:`agot.engine.combat.resolve_and_apply`.
    """

    state: GameState
    region_id: str
    attacker: str
    defender: str
    attacker_units: dict[UnitType, int]
    defender_units: dict[UnitType, int]
    attacker_card: str | None
    defender_card: str | None
    attacker_supported: bool = False
    defender_supported: bool = False
    attacker_support_units: dict[UnitType, int] = field(default_factory=dict)
    defender_support_units: dict[UnitType, int] = field(default_factory=dict)
    attacker_order: str | None = None
    attacker_origin: str | None = None
    attacker_side: BattleSide | None = None
    defender_side: BattleSide | None = None
    card_swaps: list[dict[str, Any]] = field(default_factory=list)
    power_costs: list[tuple[str, int]] = field(default_factory=list)
    orders_to_remove: list[str] = field(default_factory=list)
    influence_move: tuple[str, str] | None = None
    #: Houses whose card (The Blackfish) ignores all casualties this battle.
    casualty_immunity: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    # -- small accessors -------------------------------------------------

    def side_of(self, house: str) -> BattleSide:
        return self.attacker_side if house == self.attacker else self.defender_side

    def card_of(self, house: str) -> str | None:
        return self.attacker_card if house == self.attacker else self.defender_card

    def units_of(self, house: str) -> dict[UnitType, int]:
        return self.attacker_units if house == self.attacker else self.defender_units

    def support_of(self, house: str) -> dict[UnitType, int]:
        if house == self.attacker:
            return self.attacker_support_units
        return self.defender_support_units

    def is_attacker(self, house: str) -> bool:
        return house == self.attacker

    def supported(self, house: str) -> bool:
        if house == self.attacker:
            return self.attacker_supported
        return self.defender_supported


def _count(units: dict[UnitType, int], unit: UnitType) -> int:
    return units.get(unit, 0)


# ---------------------------------------------------------------------------
# Pre-battle abilities
# ---------------------------------------------------------------------------


def apply_pre_battle(ctx: BattleContext) -> None:
    """Apply every ability that can change the battle before it is decided."""
    # Card swaps first: they change the printed stats the other rules read.
    _resolve_card_swaps(ctx, ctx.attacker)
    _resolve_card_swaps(ctx, ctx.defender)
    # Immediate destruction of a Footman.
    _mace(ctx, ctx.attacker)
    _mace(ctx, ctx.defender)
    # Passive stat modifiers. Balon zeroes the opponent's card, so resolve it
    # before the opponent's own modifiers would matter (order is irrelevant to
    # its own side's stats).
    _passive(ctx, ctx.attacker)
    _passive(ctx, ctx.defender)
    # Salladhor Saan inspects support, so it runs once both sides are built.
    _salladhor(ctx, ctx.attacker)
    _salladhor(ctx, ctx.defender)
    # Board-touching immediate effects are recorded for the caller to commit.
    _queen_of_thorns(ctx, ctx.attacker)
    _queen_of_thorns(ctx, ctx.defender)
    _doran(ctx, ctx.attacker)
    _doran(ctx, ctx.defender)


def _resolve_card_swaps(ctx: BattleContext, house: str) -> None:
    """Tyrion and Aeron may replace a card; auto-resolve the choice."""
    ability = ability_of(ctx.card_of(house))
    if ability not in ("tyrion_lannister", "aeron_damphair"):
        return
    state = ctx.state

    if ability == "tyrion_lannister":
        # Tyrion belongs to Lannister and targets the *opponent's* card.
        target = ctx.defender if house == ctx.attacker else ctx.attacker
        target_available = list(state.houses[target].house_cards_available)
        old = ctx.card_of(target)
        if not target_available:
            # No other card in hand: the opponent fights without a House card.
            _apply_swap(ctx, target, old, None, "return_swap")
            return
        best_target = max(target_available, key=card_score)
        if card_score(best_target) < card_score(old):
            _apply_swap(ctx, target, old, best_target, "return_swap")
        return

    # Aeron Damphair: pay two Power to trade this card for a stronger one.
    available = list(state.houses[house].house_cards_available)
    current = ctx.card_of(house)
    if not available:
        return
    if state.houses[house].power < 2:
        return
    best = max(available, key=card_score)
    if card_score(best) <= card_score(current):
        return
    _apply_swap(ctx, house, current, best, "discard_swap")
    ctx.power_costs.append((house, 2))
    ctx.notes.append(f"{house} discards 2 Power to swap Aeron Damphair")


def _apply_swap(
    ctx: BattleContext, house: str, old: str | None, new: str | None, kind: str
) -> None:
    """Record a swap and update the side's printed stats immediately."""
    side = ctx.side_of(house)
    if side is not None:
        card = card_data(new)
        side.card_strength = card["strength"] if card else 0
        side.swords = card["swords"] if card else 0
        side.forts = card["forts"] if card else 0
    ctx.card_swaps.append(
        {"house": house, "old": old, "new": new, "kind": kind}
    )
    ctx.notes.append(f"{house} plays {new or 'no House card'} instead of {old}")


def _mace(ctx: BattleContext, owner: str) -> None:
    """Mace Tyrell destroys one of the opponent's participating Footmen."""
    if ability_of(ctx.card_of(owner)) != "mace_tyrell":
        return
    victim = ctx.defender if owner == ctx.attacker else ctx.attacker
    units = ctx.units_of(victim)
    if _count(units, UnitType.FOOTMAN) <= 0:
        return
    units[UnitType.FOOTMAN] = units[UnitType.FOOTMAN] - 1
    side = ctx.side_of(victim)
    if side is not None:
        side.units_strength -= 1
    ctx.notes.append(f"Mace Tyrell destroys a {victim} Footman")


def _passive(ctx: BattleContext, house: str) -> None:
    ability = ability_of(ctx.card_of(house))
    if ability == "none":
        return
    side = ctx.side_of(house)
    opponent = ctx.side_of(ctx.defender if house == ctx.attacker else ctx.attacker)
    if side is None or opponent is None:  # pragma: no cover - defensive
        return
    state = ctx.state
    attacking = ctx.is_attacker(house)

    if ability == "stannis_baratheon":
        mine = state.track_position(house, Track.IRON_THRONE)
        theirs = state.track_position(opponent.house, Track.IRON_THRONE)
        # Position 1 is best, so a lower number means a higher position.
        if theirs < mine:
            side.card_strength += 1
    elif ability == "davos_seaworth":
        if "stannis_baratheon" in state.houses[house].house_cards_used:
            side.card_strength += 1
            side.swords += 1
    elif ability == "theon_greyjoy":
        if not attacking and load_map().regions[ctx.region_id].is_fortification:
            side.card_strength += 1
            side.swords += 1
    elif ability == "asha_greyjoy":
        if not ctx.supported(house):
            side.swords += 2
            side.forts += 1
    elif ability == "nymeria_sand":
        if attacking:
            side.swords += 1
        else:
            side.forts += 1
    elif ability == "balon_greyjoy":
        opponent.card_strength = 0
    elif ability == "kevan_lannister":
        if attacking:
            footmen = _count(ctx.units_of(house), UnitType.FOOTMAN)
            footmen += _count(ctx.support_of(house), UnitType.FOOTMAN)
            side.units_strength += footmen
    elif ability == "victarion_greyjoy":
        if attacking:
            ships = _count(ctx.units_of(house), UnitType.SHIP)
            ships += _count(ctx.support_of(house), UnitType.SHIP)
            side.units_strength += ships
    elif ability == "catelyn_stark":
        region = state.regions[ctx.region_id]
        if (
            not attacking
            and region.order is not None
            and region.order_house == house
            and ORDERS_BY_KEY[region.order].order_type is OrderType.DEFEND
        ):
            side.order_strength *= 2
    elif ability == "blackfish":
        ctx.casualty_immunity.add(house)


def _salladhor(ctx: BattleContext, house: str) -> None:
    """Salladhor Saan zeroes every non-Baratheon Ship in a supported battle."""
    if ability_of(ctx.card_of(house)) != "salladhor_saan":
        return
    if not ctx.supported(house):
        return
    for target in (ctx.attacker, ctx.defender):
        side = ctx.side_of(target)
        if side is None or side.house == "baratheon":
            continue
        ships = _count(ctx.units_of(target), UnitType.SHIP)
        support_ships = _count(ctx.support_of(target), UnitType.SHIP)
        side.units_strength -= ships
        side.support_strength -= support_ships
    ctx.notes.append("Salladhor Saan reduces non-Baratheon Ships to 0")


def _queen_of_thorns(ctx: BattleContext, house: str) -> None:
    """The Queen of Thorns removes an adjacent opponent Order."""
    if ability_of(ctx.card_of(house)) != "queen_of_thorns":
        return
    opponent = ctx.defender if house == ctx.attacker else ctx.attacker
    game_map = load_map()
    for neighbour in sorted(game_map.neighbours_of(ctx.region_id)):
        region = ctx.state.regions[neighbour]
        if region.order is None or region.order_house != opponent:
            continue
        if neighbour == ctx.attacker_origin:
            continue
        ctx.orders_to_remove.append(neighbour)
        ctx.notes.append(
            f"Queen of Thorns removes the {opponent} Order in {neighbour}"
        )
        return


def _doran(ctx: BattleContext, house: str) -> None:
    """Doran Martell moves the opponent to the bottom of an Influence track."""
    if ability_of(ctx.card_of(house)) != "doran_martell":
        return
    opponent = ctx.defender if house == ctx.attacker else ctx.attacker
    state = ctx.state
    track = min(
        (Track.IRON_THRONE, Track.FIEFDOMS, Track.KINGS_COURT),
        key=lambda name: state.track_position(opponent, name),
    )
    ctx.influence_move = (track.value, opponent)
    ctx.notes.append(f"Doran Martell sends {opponent} to the bottom of {track.value}")


# ---------------------------------------------------------------------------
# Committing deferred effects
# ---------------------------------------------------------------------------


def commit_deferred(ctx: BattleContext) -> None:
    """Apply the board changes the pure resolution only recorded."""
    state = ctx.state
    for entry in ctx.card_swaps:
        _commit_swap(state, entry)
    for house, cost in ctx.power_costs:
        state.houses[house].power = max(0, state.houses[house].power - cost)
    for region_id in ctx.orders_to_remove:
        region = state.regions[region_id]
        order = region.order
        holder = region.order_house
        region.order = None
        region.order_house = None
        if order is not None and holder is not None:
            placed = state.houses[holder].orders_placed
            if order in placed:
                placed.remove(order)
    if ctx.influence_move is not None:
        track_name, house = ctx.influence_move
        track = Track(track_name)
        order = state.tracks[track]
        if house in order:
            order.remove(house)
            order.append(house)


def _commit_swap(state: GameState, entry: dict[str, Any]) -> None:
    house = state.houses[entry["house"]]
    old = entry["old"]
    new = entry["new"]
    if entry["kind"] == "return_swap" and old is not None:
        if old in house.house_cards_used:
            house.house_cards_used.remove(old)
        if old not in house.house_cards_available:
            house.house_cards_available.append(old)
    if new is not None:
        if new in house.house_cards_available:
            house.house_cards_available.remove(new)
        house.house_cards_used.append(new)


# ---------------------------------------------------------------------------
# Post-battle abilities
# ---------------------------------------------------------------------------


def apply_post_battle(ctx: BattleContext, result: BattleResult) -> None:
    """Apply winner/loser abilities once the battle has been placed."""
    state = ctx.state
    winner_card = ctx.card_of(result.winner)
    loser_card = ctx.card_of(result.loser)
    winner_ability = ability_of(winner_card)
    loser_ability = ability_of(loser_card)

    if winner_ability == "tywin_lannister":
        state.houses[result.winner].power += 2
        ctx.notes.append("Tywin Lannister gains 2 Power")
    elif winner_ability == "renly_baratheon":
        _renly_upgrade(ctx, result.winner)
    elif winner_ability == "cersei_lannister":
        _remove_any_order(ctx, result.loser)

    if loser_ability == "roose_bolton":
        house = state.houses[result.loser]
        house.house_cards_available.extend(house.house_cards_used)
        house.house_cards_used = []
        ctx.notes.append("Roose Bolton returns the discard pile to hand")

    # Patchface acts after combat whichever side won.
    for house in (ctx.attacker, ctx.defender):
        if ability_of(ctx.card_of(house)) == "patchface":
            _patchface(ctx, house)


def _renly_upgrade(ctx: BattleContext, house: str) -> None:
    region = ctx.state.regions[ctx.region_id]
    if _count(region.units, UnitType.FOOTMAN) > 0:
        region.units[UnitType.FOOTMAN] -= 1
        region.units[UnitType.KNIGHT] = region.units.get(UnitType.KNIGHT, 0) + 1
        if region.units[UnitType.FOOTMAN] == 0:
            del region.units[UnitType.FOOTMAN]
        ctx.notes.append("Renly Baratheon upgrades a Footman to a Knight")
        return
    # Otherwise a supporting Baratheon Footman may be upgraded.
    game_map = load_map()
    for neighbour in sorted(game_map.neighbours_of(ctx.region_id)):
        region = ctx.state.regions[neighbour]
        if region.order_house != house:
            continue
        spec = ORDERS_BY_KEY.get(region.order or "")
        if spec is None or spec.order_type is not OrderType.SUPPORT:
            continue
        if _count(region.units, UnitType.FOOTMAN) <= 0:
            continue
        region.units[UnitType.FOOTMAN] -= 1
        region.units[UnitType.KNIGHT] = region.units.get(UnitType.KNIGHT, 0) + 1
        if region.units[UnitType.FOOTMAN] == 0:
            del region.units[UnitType.FOOTMAN]
        ctx.notes.append("Renly Baratheon upgrades a supporting Footman")
        return


def _remove_any_order(ctx: BattleContext, house: str) -> None:
    for region_id in sorted(ctx.state.regions):
        region = ctx.state.regions[region_id]
        if region.order is None or region.order_house != house:
            continue
        order = region.order
        region.order = None
        region.order_house = None
        placed = ctx.state.houses[house].orders_placed
        if order in placed:
            placed.remove(order)
        ctx.notes.append(f"Cersei Lannister removes the {house} Order in {region_id}")
        return


def _patchface(ctx: BattleContext, house: str) -> None:
    opponent = ctx.defender if house == ctx.attacker else ctx.attacker
    available = ctx.state.houses[opponent].house_cards_available
    if not available:
        return
    card = max(available, key=card_score)
    available.remove(card)
    ctx.state.houses[opponent].house_cards_used.append(card)
    ctx.notes.append(f"Patchface discards {opponent}'s {card}")


def robb_chose_retreat(ctx: BattleContext, winner: str) -> bool:
    """True when Robb Stark won and gets to pick the retreat of the loser."""
    return ability_of(ctx.card_of(winner)) == "robb_stark"


def arianne_holds(ctx: BattleContext, loser: str) -> bool:
    """True when Arianne Martell defended and lost, blocking the conquest."""
    return (
        loser == ctx.defender
        and ability_of(ctx.card_of(loser)) == "arianne_martell"
    )
