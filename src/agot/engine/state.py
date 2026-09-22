"""Mutable game state.

The state is a plain dataclass tree so that it can be deep-copied cheaply for
Monte Carlo tree search and serialised for debugging. Nothing in this module
depends on numpy or torch.

Two simplifications are worth stating up front, because they shape the state:

1.  Order tokens are stored face up. The real game hides them until the reveal.
    Perfect-information search on hidden state is therefore an approximation;
    see docs/roadmap.md for the plan to move to belief states.
2.  A region is controlled by at most one house at a time, which holds in the
    real game once battles have been resolved.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field

from agot.engine.enums import (
    STAR_ALLOWANCE_BY_COURT_POSITION,
    SUPPLY_LIMITS,
    ActionStep,
    OrderType,
    Phase,
    Track,
    UnitType,
)


@dataclass
class RegionState:
    """What is currently in one region."""

    region_id: str
    controller: str | None = None
    owner: str | None = None
    units: dict[UnitType, int] = field(default_factory=dict)
    routed: dict[UnitType, int] = field(default_factory=dict)
    neutral_force: int = 0
    #: Printed strength of a Garrison token still present here (0 = none).
    #: Garrisons are not units: they add to a defender's Combat Strength only,
    #: never count towards Supply, and can never be assigned an order.
    garrison: int = 0
    #: Mustering points still available to spend *in this area* after a
    #: "Mustering" card. Unspent points are lost (they never pool).
    mustering_points: int = 0
    influence_token: str | None = None
    order: str | None = None
    order_house: str | None = None

    # -- helpers ---------------------------------------------------------

    def total(self, *, include_routed: bool = True) -> int:
        units = sum(self.units.values())
        if include_routed:
            units += sum(self.routed.values())
        return units

    def ready_units(self) -> dict[UnitType, int]:
        return {unit: count for unit, count in self.units.items() if count > 0}

    def is_empty(self) -> bool:
        return self.total(include_routed=False) == 0

    def contains_house(self, house_id: str) -> bool:
        return self.controller == house_id and not self.is_empty()

    def clear_units(self) -> None:
        self.units = {}
        self.routed = {}


@dataclass
class HouseState:
    """Per-house player state."""

    house_id: str
    power: int = 5
    supply_level: int = 2
    orders_available: list[str] = field(default_factory=list)
    orders_placed: list[str] = field(default_factory=list)
    house_cards_available: list[str] = field(default_factory=list)
    house_cards_used: list[str] = field(default_factory=list)
    blade_used: bool = False
    passed: bool = False
    eliminated: bool = False

    def stars_placed(self) -> int:
        from agot.engine.enums import ORDERS_BY_KEY

        return sum(1 for key in self.orders_placed if ORDERS_BY_KEY[key].starred)

    def supply_limit(self) -> tuple[int, ...]:
        level = max(0, min(self.supply_level, len(SUPPLY_LIMITS) - 1))
        return SUPPLY_LIMITS[level]


@dataclass
class GameState:
    """Complete mutable state of one game."""

    max_rounds: int = 10
    victory_castles: int = 7

    round: int = 1
    phase: Phase = Phase.WESTEROS
    action_step: ActionStep = ActionStep.RAID
    active_house: str | None = None
    winner: str | None = None
    #: House that resolved the previous order in the current Action Phase step.
    #: The next house is the following one on the Iron Throne track, so each
    #: house resolves one order at a time instead of resolving every order
    #: before passing the turn (rulebook pages 13-16).
    action_house_last: str | None = None
    #: Set once the Messenger Raven has been used (or declined) in a round.
    raven_used: bool = False
    #: Phase to return to once the current Mustering sub-phase finishes. Set
    #: only by a Special Consolidate Power Order that musters in its own area.
    return_phase: Phase | None = None

    tracks: dict[Track, list[str]] = field(default_factory=dict)
    regions: dict[str, RegionState] = field(default_factory=dict)
    houses: dict[str, HouseState] = field(default_factory=dict)

    wildling_threat: int = 0
    #: Wildling attack in progress (only while ``Phase.WILDLING_ATTACK``).
    #: Holds the attack strength, the secret bids so far, the excluded houses
    #: (from a "Preemptive Raid" second attack) and the Wildling card drawn.
    wildling_attack: dict | None = None
    westeros_decks: dict[str, list[str]] = field(default_factory=dict)
    westeros_deck_pos: dict[str, int] = field(default_factory=dict)
    westeros_drawn: dict[str, str | None] = field(default_factory=dict)
    #: Westeros card effects still to resolve this round. Non-``None`` while a
    #: round's Westeros phase is mid-flight (it can be interrupted by a bidding
    #: step or a Wildling attack), because all three cards must still resolve.
    pending_westeros: list[str] | None = None
    #: Set while a holders'-choice Westeros card waits for its holder to pick.
    #: ``{"card": card_id, "holder": house_id, "options": [card_id, ...]}``.
    westeros_choice: dict | None = None
    wildling_deck: list[str] = field(default_factory=list)
    wildling_deck_pos: int = 0
    #: Wildling cards already resolved; recycled when the deck runs out.
    wildling_discard: list[str] = field(default_factory=list)

    #: Order keys or order types that may not be placed this Planning Phase
    #: (from Rains of Autumn, Sea of Storms, Storm of Swords, Web of Lies and
    #: Feast for Crows). Reset at the start of every round.
    order_restrictions: list[str] = field(default_factory=list)
    #: Houses that still have to place order tokens this planning phase.
    pending_planning: list[str] = field(default_factory=list)
    #: Pending bids for the current bidding round, keyed by house.
    pending_bids: dict[str, int] = field(default_factory=dict)
    #: Set while a battle is awaiting House card choices.
    pending_battle: dict | None = None
    #: Which influence tracks have still to be bid on, in resolution order.
    bid_tracks: list[str] = field(default_factory=list)

    log: list[str] = field(default_factory=list)

    #: Region ids that count towards the victory condition (castles and
    #: strongholds). Populated by :func:`new_state`.
    fortifications: frozenset[str] = frozenset()

    # -- convenience -----------------------------------------------------

    def clone(self) -> GameState:
        return copy.deepcopy(self)

    def note(self, message: str) -> None:
        self.log.append(f"R{self.round} {self.phase.value}: {message}")

    def track_position(self, house_id: str, track: Track) -> int:
        """1-based position of a house on a track (1 is best)."""
        return self.tracks[track].index(house_id) + 1

    def track_order(self, track: Track) -> list[str]:
        return list(self.tracks[track])

    def first_player(self) -> str:
        return self.tracks[Track.IRON_THRONE][0]

    def blade_holder(self) -> str:
        return self.tracks[Track.FIEFDOMS][0]

    def raven_holder(self) -> str:
        return self.tracks[Track.KINGS_COURT][0]

    def star_allowance(self, house_id: str) -> int:
        position = self.track_position(house_id, Track.KINGS_COURT)
        index = min(position - 1, len(STAR_ALLOWANCE_BY_COURT_POSITION) - 1)
        return STAR_ALLOWANCE_BY_COURT_POSITION[index]

    def controlled_regions(self, house_id: str) -> list[str]:
        return [
            rid for rid, region in self.regions.items() if region.controller == house_id
        ]

    def castle_count(self, house_id: str) -> int:
        """Castles and strongholds controlled - the victory condition."""
        return sum(
            1 for rid in self.controlled_regions(house_id) if rid in self.fortifications
        )

    def regions_of(self, house_id: str) -> Iterator[RegionState]:
        for rid in self.controlled_regions(house_id):
            yield self.regions[rid]

    def sea_supply_lines(self, house_id: str) -> set[str]:
        """Sea regions containing at least one of a house's ships."""
        from agot.engine.enums import UnitType as _UnitType

        return {
            rid
            for rid, region in self.regions.items()
            if region.controller == house_id and region.units.get(_UnitType.SHIP, 0) > 0
        }

    def army_sizes(self, house_id: str) -> list[int]:
        """Sizes of a house's armies, largest first.

        Per the rulebook a lone unit is not an army and is exempt from Supply,
        while routed units still count towards Supply. Armies can be land
        groups or fleets (two or more Ships in the same sea/port).
        """
        sizes: list[int] = []
        for region in self.regions_of(house_id):
            size = region.total(include_routed=True)
            if size >= 2:
                sizes.append(size)
        return sorted(sizes, reverse=True)

    def has_mustering_points(self, house_id: str) -> bool:
        return any(
            region.mustering_points > 0 for region in self.regions_of(house_id)
        )

    def land_area_count(self, house_id: str) -> int:
        """Controlled land areas (kept for reporting; not a tiebreaker)."""
        from agot.engine.map import load_map

        game_map = load_map()
        return sum(
            1
            for rid in self.controlled_regions(house_id)
            if game_map.regions[rid].is_land
        )

    def stronghold_count(self, house_id: str) -> int:
        """Controlled Strongholds - the second end-of-game tiebreaker."""
        from agot.engine.map import load_map

        game_map = load_map()
        return sum(
            1
            for rid in self.controlled_regions(house_id)
            if game_map.regions[rid].is_stronghold
        )

    def within_supply(self, house_id: str) -> bool:
        limits = self.houses[house_id].supply_limit()
        for index, size in enumerate(self.army_sizes(house_id)):
            if index >= len(limits) or size > limits[index]:
                return False
        return True

    def supply_from_regions(self, house_id: str) -> int:
        from agot.engine.map import load_map

        game_map = load_map()
        return sum(game_map.regions[rid].supply for rid in self.controlled_regions(house_id))

    def orders_on_board(self, house_id: str) -> list[tuple[str, str]]:
        """(region_id, order_key) pairs placed by a house."""
        return [
            (rid, region.order)
            for rid, region in self.regions.items()
            if region.order is not None and region.order_house == house_id
        ]

    def has_order_type(self, house_id: str, order_type: OrderType) -> bool:
        return any(
            region.order_house == house_id
            and region.order is not None
            and _order_type(region.order) is order_type
            for region in self.regions.values()
        )


def _order_type(key: str) -> OrderType:
    from agot.engine.enums import ORDERS_BY_KEY

    return ORDERS_BY_KEY[key].order_type


def new_state(
    houses: dict[str, HouseState],
    tracks: dict[Track, list[str]],
    max_rounds: int,
    victory_castles: int,
) -> GameState:
    """Build an empty state with the given houses and tracks."""
    from agot.engine.map import load_map

    game_map = load_map()
    regions = {
        region.id: RegionState(region_id=region.id, owner=region.home)
        for region in game_map.regions.values()
    }
    state = GameState(
        max_rounds=max_rounds,
        victory_castles=victory_castles,
        tracks=tracks,
        regions=regions,
        houses=houses,
    )
    state.fortifications = frozenset(
        region.id for region in game_map.regions.values() if region.is_fortification
    )
    return state
