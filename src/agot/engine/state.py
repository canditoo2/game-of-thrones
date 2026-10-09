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
from enum import Enum

from agot.engine.enums import (
    ORDERS_BY_KEY,
    STAR_ALLOWANCE_BY_COURT_POSITION,
    SUPPLY_LIMITS,
    ActionStep,
    OrderType,
    Phase,
    Track,
    UnitType,
)


@dataclass(slots=True)
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

    def clone(self) -> RegionState:
        """Structural copy, far cheaper than ``copy.deepcopy``.

        Every field is an immutable scalar or a flat ``dict`` of scalars, so the
        copy is a plain reconstruction. Tree search calls this millions of times,
        so it must stay allocation-only (see :meth:`GameState.clone`).
        """
        new = RegionState.__new__(RegionState)
        new.region_id = self.region_id
        new.controller = self.controller
        new.owner = self.owner
        new.units = dict(self.units)
        new.routed = dict(self.routed)
        new.neutral_force = self.neutral_force
        new.garrison = self.garrison
        new.mustering_points = self.mustering_points
        new.influence_token = self.influence_token
        new.order = self.order
        new.order_house = self.order_house
        return new


@dataclass(slots=True)
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

    def clone(self) -> HouseState:
        """Structural copy; see :meth:`RegionState.clone`."""
        new = HouseState.__new__(HouseState)
        new.house_id = self.house_id
        new.power = self.power
        new.supply_level = self.supply_level
        new.orders_available = list(self.orders_available)
        new.orders_placed = list(self.orders_placed)
        new.house_cards_available = list(self.house_cards_available)
        new.house_cards_used = list(self.house_cards_used)
        new.blade_used = self.blade_used
        new.passed = self.passed
        new.eliminated = self.eliminated
        return new


def _copy_pending(value, memo: dict | None = None):
    """Copy a pending-effect value, reproducing nested ``dict``/``list`` values.

    ``GameState`` keeps four pending dicts (``wildling_attack``,
    ``westeros_choice``, ``pending_battle``, ``pending_march``). They are small
    and are ``None`` for most of the game, but ``copy.deepcopy`` paid its dispatch
    and memo bookkeeping on every single clone, and a ``GameState.clone`` is the
    hottest operation in tree search. Containers are therefore copied
    structurally and only genuinely opaque values fall through to
    ``copy.deepcopy``, which receives ``memo`` so the caller can rebind
    back-references (see :meth:`GameState.clone`). Immutable leaves (strings,
    numbers, enums) are shared, which is what ``deepcopy`` does with them too.

    Cloning is exactly where sharing has to stop: the engine writes into
    ``pending_battle["support_choices"]`` and ``wildling_attack["bids"]`` in
    place, so those inner dicts must not be shared between a state and its clone.
    """
    if isinstance(value, dict):
        return {key: _copy_pending(item, memo) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_pending(item, memo) for item in value]
    if isinstance(value, (str, int, float, bool, type(None), Enum)):
        return value
    return copy.deepcopy(value, memo)


@dataclass(slots=True)
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
    #: Set while a March Order is being resolved across several destinations.
    #: ``{"house": house_id, "origin": region_id}``. The order token stays on
    #: the board until the march ends, so its bonus still applies to a battle
    #: that happens after the non-combat moves (rulebook page 15).
    pending_march: dict | None = None
    #: Which influence tracks have still to be bid on, in resolution order.
    bid_tracks: list[str] = field(default_factory=list)

    log: list[str] = field(default_factory=list)

    #: Region ids that count towards the victory condition (castles and
    #: strongholds). Populated by :func:`new_state`.
    fortifications: frozenset[str] = frozenset()

    #: Region ids this state has already taken private *since the last clone*.
    #: A clone shares every :class:`RegionState` with its source until one of
    #: them writes to it (see :meth:`mut`). Both sides therefore drop their
    #: bookkeeping when a clone is made: after the split this state's older
    #: private objects are shared again, so it has to re-copy them too.
    #: ``None`` means a brand new state whose regions no one else can see yet,
    #: so writes need no copy. Excluded from equality and ``repr``: it is
    #: bookkeeping, not game state.
    _owned: set[str] | None = field(
        default=None, init=False, compare=False, repr=False
    )

    # -- convenience -----------------------------------------------------

    def clone(self) -> GameState:
        """Structural copy used by tree search.

        This is deliberately hand-written rather than ``copy.deepcopy``: the
        generic copier walks the whole dataclass tree with a memo dict and was
        measured at ~86% of MCTS time. Every field below is copied explicitly.
        The only fields that still need a generic deepcopy are the small opaque
        "pending" dicts, which are ``None`` for most of the game; their copier
        also rebinds any back-reference to the state, see :func:`_copy_pending`.
        """
        new = GameState.__new__(GameState)
        new.max_rounds = self.max_rounds
        new.victory_castles = self.victory_castles
        new.round = self.round
        new.phase = self.phase
        new.action_step = self.action_step
        new.active_house = self.active_house
        new.winner = self.winner
        new.action_house_last = self.action_house_last
        new.raven_used = self.raven_used
        new.return_phase = self.return_phase
        new.tracks = {track: list(order) for track, order in self.tracks.items()}
        new.regions = dict(self.regions)
        new._owned = set()
        # ``new`` shares every region object with this state from here on --
        # including the ones this state had already taken private, which its
        # ``_owned`` set can no longer vouch for. Clearing it makes this state
        # re-copy on its next write to *any* region, so isolation holds in both
        # directions. It costs nothing in tree search: a node is cloned and then
        # stepped, and its parent is never stepped again.
        self._owned = set()
        new.houses = {hid: house.clone() for hid, house in self.houses.items()}
        new.wildling_threat = self.wildling_threat
        # A copied value may hold a back-reference to the state it was parked in:
        # a resolved battle's ``BattleResult`` carries a context whose ``state`` is
        # read *and written* by ``house_cards.apply_post_battle`` (Tywin's Power,
        # Renly's upgrade, Cersei's Order removal, Roose's discard, Patchface).
        # On the live game that back-reference is the state itself, so it has to
        # become ``new`` here; otherwise the clone applies those abilities to a
        # detached ghost board and search silently plays by different rules than
        # the game being played. Seeding `deepcopy`'s memo with that one
        # substitution rebinds every occurrence and skips copying the state.
        pending_memo = {id(self): new}
        new.wildling_attack = _copy_pending(self.wildling_attack, pending_memo)
        new.westeros_decks = {
            name: list(cards) for name, cards in self.westeros_decks.items()
        }
        new.westeros_deck_pos = dict(self.westeros_deck_pos)
        new.westeros_drawn = dict(self.westeros_drawn)
        new.pending_westeros = (
            None if self.pending_westeros is None else list(self.pending_westeros)
        )
        new.westeros_choice = _copy_pending(self.westeros_choice, pending_memo)
        new.wildling_deck = list(self.wildling_deck)
        new.wildling_deck_pos = self.wildling_deck_pos
        new.wildling_discard = list(self.wildling_discard)
        new.order_restrictions = list(self.order_restrictions)
        new.pending_planning = list(self.pending_planning)
        new.pending_bids = dict(self.pending_bids)
        new.pending_battle = _copy_pending(self.pending_battle, pending_memo)
        new.pending_march = _copy_pending(self.pending_march, pending_memo)
        new.bid_tracks = list(self.bid_tracks)
        new.log = list(self.log)
        new.fortifications = self.fortifications
        return new

    def mut(self, region_id: str) -> RegionState:
        """Return ``region_id``'s region, taking a private copy before a write.

        A clone shares its source's :class:`RegionState` objects; this hands the
        caller a region only it can see, copying it the first time. The engine
        mutates a clone right after building it (one ``step``), and a step
        touches on average ~1 of the 58 regions, so this replaces a full
        board copy per MCTS node with a shallow dict plus one region copy.

        Callers must fetch through here (not ``regions[...]``) *before* the
        first write to a region, and must keep using the returned object for
        every later write in the same call.
        """
        owned = self._owned
        if owned is None or region_id in owned:
            return self.regions[region_id]
        region = self.regions[region_id].clone()
        self.regions[region_id] = region
        owned.add(region_id)
        return region

    def note(self, message: str) -> None:
        self.log.append(f"R{self.round} {self.phase.value}: {message}")

    @property
    def orders_revealed(self) -> bool:
        """Whether the Order tokens on the board are face up.

        Orders are assigned face down during Planning and the Messenger Raven
        step, then revealed simultaneously when the Action Phase begins
        (rulebook pages 11-12). A house may always see its own orders, so this
        only tells you that *other* houses' orders have become public.
        """
        return self.phase not in (Phase.PLANNING, Phase.RAVEN)

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
        """Castles and strongholds controlled - the victory condition.

        Iterates only the fortifications rather than every region, because this
        is checked on every phase transition.
        """
        regions = self.regions
        return sum(
            1 for rid in self.fortifications if regions[rid].controller == house_id
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
            order_type_of(region.order) is order_type
            for region in self.regions.values()
            if region.order_house == house_id
        )

    def houses_with_orders(self, order_type: OrderType) -> set[str]:
        """Houses with at least one placed order of ``order_type`` on the board.

        One pass over the board, used by the Action Phase to skip houses with no
        order of the current step instead of scanning once per house.
        """
        result: set[str] = set()
        for region in self.regions.values():
            if region.order_house is not None and order_type_of(region.order) is order_type:
                result.add(region.order_house)
        return result

    def orders_of_type(
        self, house_id: str, order_type: OrderType
    ) -> list[tuple[str, str]]:
        """``(region_id, order_key)`` pairs of ``house_id`` for one order type."""
        matches: list[tuple[str, str]] = []
        for rid, region in self.regions.items():
            order = region.order
            if (
                region.order_house == house_id
                and order is not None
                and order_type_of(order) is order_type
            ):
                matches.append((rid, order))
        return matches


def order_type_of(key: str | None) -> OrderType | None:
    """Order type of a placed order key (``None`` for an empty slot)."""
    if key is None:
        return None
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
