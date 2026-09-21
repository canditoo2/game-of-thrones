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

    tracks: dict[Track, list[str]] = field(default_factory=dict)
    regions: dict[str, RegionState] = field(default_factory=dict)
    houses: dict[str, HouseState] = field(default_factory=dict)

    wildling_threat: int = 0
    westeros_decks: dict[str, list[str]] = field(default_factory=dict)
    westeros_deck_pos: dict[str, int] = field(default_factory=dict)
    westeros_drawn: dict[str, str | None] = field(default_factory=dict)
    wildling_deck: list[str] = field(default_factory=list)
    wildling_deck_pos: int = 0

    #: Pending mustering points per house, consumed by mustering actions.
    mustering_points: dict[str, int] = field(default_factory=dict)
    #: Houses that still have to place order tokens this planning phase.
    pending_planning: list[str] = field(default_factory=list)
    #: Pending bids for the current bidding round, keyed by house.
    pending_bids: dict[str, int] = field(default_factory=dict)
    #: Set while a battle is awaiting House card choices.
    pending_battle: dict | None = None
    #: Which influence tracks the current bid will reorder.
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
        """Sizes of a house's land armies, largest first."""
        sizes = [
            region.total(include_routed=False)
            for region in self.regions_of(house_id)
            if region.units and any(
                unit is not _SHIP for unit in region.units
            )
        ]
        return sorted((size for size in sizes if size > 0), reverse=True)

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


_SHIP = UnitType.SHIP


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
