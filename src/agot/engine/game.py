"""The game object: setup, legal actions, phase progression and victory.

``Game`` is the single entry point used by the agents, the self-play loop and
the tests. It owns a :class:`~agot.engine.state.GameState` and exposes a
Gymnasium-flavoured interface::

    game = Game.new(GameConfig(seed=0))
    while not game.is_terminal():
        actions = game.legal_actions()
        game.step(actions[0])

Design notes and known simplifications live in docs/roadmap.md. The most
important ones are repeated here because they affect how strong a trained agent
can become:

* Order tokens are face up. The real game hides them until the reveal.
* Support is automatic and only self-directed.
* Casualty removal and retreat destinations are chosen automatically.
* Wildling attacks resolve without bidding and have no card-specific effects.
* House cards are chosen by the players (that decision *is* modelled), but
  special card abilities are not implemented, so only printed strengths and
  icons matter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from agot.engine.actions import Action, ActionKind
from agot.engine.combat import (
    apply_casualties,
    order_modifier,
    resolve_and_apply,
    resolve_battle,
)
from agot.engine.enums import (
    ORDERS_BY_KEY,
    SUPPLY_LIMITS,
    ActionStep,
    OrderType,
    Phase,
    Terrain,
    Track,
    UnitType,
)
from agot.engine.map import load_house_cards, load_houses, load_map, load_westeros, load_wildlings
from agot.engine.state import GameState, HouseState, new_state

#: Maps the plural names used in the JSON data onto unit types.
_UNITS_FROM_JSON: dict[str, UnitType] = {
    "footmen": UnitType.FOOTMAN,
    "knights": UnitType.KNIGHT,
    "siege": UnitType.SIEGE,
    "ships": UnitType.SHIP,
}

DEFAULT_HOUSES: tuple[str, ...] = (
    "baratheon",
    "lannister",
    "stark",
    "greyjoy",
    "tyrell",
    "martell",
)

_TRACK_BY_VALUE = {
    "iron_throne": Track.IRON_THRONE,
    "fiefdoms": Track.FIEFDOMS,
    "kings_court": Track.KINGS_COURT,
}
_TRACK_CARD_VALUE = {
    Track.IRON_THRONE: "iron_throne",
    Track.FIEFDOMS: "fiefdoms",
    Track.KINGS_COURT: "kings_court",
}


@dataclass
class GameConfig:
    """Options that control a single game."""

    houses: tuple[str, ...] = DEFAULT_HOUSES
    max_rounds: int = 10
    victory_castles: int = 7
    seed: int | None = None
    #: Draw and apply the three Westeros cards each round from round two.
    use_westeros_decks: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.houses) - set(load_houses().houses)
        if unknown:
            raise ValueError(f"unknown houses: {sorted(unknown)}")
        if not self.houses:
            raise ValueError("at least one house is required")


class Game:
    """A single game of A Game of Thrones: The Board Game (2nd Edition)."""

    def __init__(self, state: GameState, config: GameConfig, rng: random.Random) -> None:
        self.state = state
        self.config = config
        self.rng = rng
        self.map = load_map()
        self._houses = load_houses()
        self._house_cards = load_house_cards()
        self._westeros = load_westeros()
        self._wildlings = load_wildlings()
        self._advance()

    # -- construction ----------------------------------------------------

    @classmethod
    def new(cls, config: GameConfig | None = None) -> Game:
        config = config or GameConfig()
        rng = random.Random(config.seed)
        state = cls._initial_state(config, rng)
        return cls(state, config, rng)

    @staticmethod
    def _initial_state(config: GameConfig, rng: random.Random) -> GameState:
        houses_data = load_houses()
        game_map = load_map()

        houses: dict[str, HouseState] = {}
        for house_id in config.houses:
            payload = houses_data.houses[house_id]
            houses[house_id] = HouseState(
                house_id=house_id,
                power=payload.get("initial_power", 5),
                supply_level=payload.get("initial_supply", 2),
                orders_available=[spec.key for spec in ORDERS_BY_KEY.values()],
                house_cards_available=[
                    card["id"] for card in load_house_cards().deck(house_id)
                ],
            )

        tracks: dict[Track, list[str]] = {}
        for track in Track:
            key = _TRACK_CARD_VALUE[track]
            order = [h for h in houses_data.initial_tracks[key] if h in config.houses]
            # Any house missing from the reference ordering goes last.
            order += [h for h in config.houses if h not in order]
            tracks[track] = order

        state = new_state(
            houses=houses,
            tracks=tracks,
            max_rounds=config.max_rounds,
            victory_castles=config.victory_castles,
        )

        # Place starting units.
        for house_id in config.houses:
            for region_id, counts in houses_data.starting_units(house_id).items():
                region = state.regions[region_id]
                region.controller = house_id
                for unit_name, count in counts.items():
                    if count:
                        unit = _UNITS_FROM_JSON[unit_name]
                        region.units[unit] = region.units.get(unit, 0) + count

        # Home areas start under their owner's control.
        for region in game_map.regions.values():
            if region.home:
                state.regions[region.id].owner = region.home
                state.regions[region.id].controller = region.home

        # Shuffle the Westeros decks and the Wildling deck.
        for deck_name, cards in load_westeros().composition.items():
            shuffled = list(cards)
            rng.shuffle(shuffled)
            state.westeros_decks[deck_name] = shuffled
            state.westeros_deck_pos[deck_name] = 0
            state.westeros_drawn[deck_name] = None
        state.wildling_deck = list(load_wildlings().deck)
        rng.shuffle(state.wildling_deck)
        state.wildling_threat = load_wildlings().start_threat

        state.pending_planning = list(state.tracks[Track.IRON_THRONE])
        state.note(f"setup complete with {len(config.houses)} houses")
        return state

    # -- basic queries ---------------------------------------------------

    def clone(self) -> Game:
        """Deep copy, used by tree search."""
        return Game(self.state.clone(), self.config, random.Random(self.rng.random()))

    def is_terminal(self) -> bool:
        return self.state.phase is Phase.GAME_OVER

    def current_player(self) -> str | None:
        """The house that must act, or ``None`` when the engine auto-advances."""
        return self.state.active_house

    def winner(self) -> str | None:
        return self.state.winner

    # -- legal actions ---------------------------------------------------

    def legal_actions(self) -> list[Action]:
        """All actions the current player may take."""
        phase = self.state.phase
        house = self.state.active_house
        if phase is Phase.GAME_OVER or house is None:
            return []
        if phase is Phase.PLANNING:
            return self._planning_actions(house)
        if phase is Phase.MUSTERING:
            return self._mustering_actions(house)
        if phase is Phase.BIDDING:
            return self._bidding_actions(house)
        if phase is Phase.COMBAT:
            return self._combat_actions(house)
        if phase is Phase.ACTION:
            return self._action_actions(house)
        return []

    # -- stepping --------------------------------------------------------

    def step(self, action: Action) -> None:
        """Apply an action, then let the engine auto-advance."""
        if self.is_terminal():
            raise RuntimeError("game is already over")

        kind = action.kind
        if kind is ActionKind.PASS:
            self._apply_pass(action)
        elif kind is ActionKind.PLACE_ORDER:
            self._apply_place_order(action)
        elif kind is ActionKind.MUSTER:
            self._apply_muster(action)
        elif kind is ActionKind.BID:
            self._apply_bid(action)
        elif kind is ActionKind.PLAY_HOUSE_CARD:
            self._apply_house_card(action)
        elif kind is ActionKind.RESOLVE_RAID:
            self._apply_raid(action)
        elif kind is ActionKind.MARCH:
            self._apply_march(action)
        elif kind is ActionKind.RESOLVE_CONSOLIDATE:
            self._apply_consolidate(action)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unsupported action kind: {kind}")

        self._advance()

    # -- setup helpers ---------------------------------------------------

    def _fortification_owner(self, region_id: str) -> str | None:
        return self.state.regions[region_id].controller

    # ==================================================================
    # Phase progression
    # ==================================================================

    def _advance(self) -> None:
        """Resolve every phase that needs no player input."""
        for _ in range(200):
            if self.state.phase is Phase.GAME_OVER:
                self.state.active_house = None
                return

            if self.state.phase is Phase.WESTEROS:
                self._run_westeros_phase()
            elif self.state.phase is Phase.CLEANUP:
                self._run_cleanup_phase()
            elif self.state.phase is Phase.BIDDING:
                self.state.active_house = self._next_bidder()
                if self.state.active_house is not None:
                    return
                self._resolve_bids()
            elif self.state.phase is Phase.MUSTERING:
                self.state.active_house = self._next_musterer()
                if self.state.active_house is not None:
                    return
                self.state.mustering_points = {}
                self.state.phase = Phase.PLANNING
            elif self.state.phase is Phase.PLANNING:
                self.state.active_house = (
                    self.state.pending_planning[0] if self.state.pending_planning else None
                )
                if self.state.active_house is not None:
                    return
                self.state.phase = Phase.ACTION
                self.state.action_step = ActionStep.RAID
            elif self.state.phase is Phase.COMBAT:
                self.state.active_house = self._next_combat_chooser()
                if self.state.active_house is not None:
                    return
                self._resolve_pending_battle()
            elif self.state.phase is Phase.ACTION:
                self.state.active_house = self._next_action_house()
                if self.state.active_house is not None:
                    return
                self._next_action_step()
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"unhandled phase {self.state.phase}")
        raise RuntimeError("phase progression did not settle")  # pragma: no cover

    # -- Westeros phase --------------------------------------------------

    def _run_westeros_phase(self) -> None:
        if self.state.round == 1 or not self.config.use_westeros_decks:
            self.state.phase = Phase.PLANNING
            return

        needed_bidding: list[str] = []
        for deck_name in ("deck_i", "deck_ii", "deck_iii"):
            card_id = self._draw_westeros(deck_name)
            state = self.state
            state.westeros_drawn[deck_name] = card_id
            if card_id is None:
                continue
            card = self._westeros.cards[card_id]
            effect = card["effect"]

            if effect == "mustering":
                self._grant_mustering_points()
            elif effect == "supply":
                self._recalculate_supply()
            elif effect == "gain_power_per_crown":
                self._gain_power_per_crown()
            elif effect == "bid_all_tracks":
                needed_bidding = ["iron_throne", "fiefdoms", "kings_court"]
            elif effect == "bid_kings_court":
                needed_bidding = needed_bidding or ["kings_court"]
            elif effect == "wildlings_attack":
                self._resolve_wildlings()
            elif effect == "wildlings_calm":
                self.state.wildling_threat = max(0, self.state.wildling_threat - 2)
            elif effect == "order_restriction":
                state.note(f"{card['name']}: {card.get('restricted_order')} orders banned")
            elif effect == "unimplemented":
                state.note(f"{card['name']}: effect not implemented")
            else:  # pragma: no cover - defensive
                state.note(f"{card['name']}: unknown effect {effect!r}")

        self.state.bid_tracks = needed_bidding
        if needed_bidding:
            self.state.pending_bids = {}
            self.state.phase = Phase.BIDDING
        elif self.state.mustering_points:
            self.state.phase = Phase.MUSTERING
        else:
            self.state.phase = Phase.PLANNING

    def _draw_westeros(self, deck_name: str) -> str | None:
        deck = self.state.westeros_decks.get(deck_name) or []
        position = self.state.westeros_deck_pos.get(deck_name, 0)
        if position >= len(deck):
            return None
        self.state.westeros_deck_pos[deck_name] = position + 1
        return deck[position]

    def _grant_mustering_points(self) -> None:
        for house_id, house in self.state.houses.items():
            if house.eliminated:
                continue
            points = sum(
                self.map.regions[rid].mustering
                for rid in self.state.controlled_regions(house_id)
            )
            if points:
                self.state.mustering_points[house_id] = (
                    self.state.mustering_points.get(house_id, 0) + points
                )
        self.state.note("mustering points granted")

    def _recalculate_supply(self) -> None:
        for house_id, house in self.state.houses.items():
            if house.eliminated:
                continue
            supply = self.state.supply_from_regions(house_id)
            house.supply_level = max(0, min(supply, len(SUPPLY_LIMITS) - 1))
            self._enforce_supply(house_id)
        self.state.note("supply recalculated")

    def _enforce_supply(self, house_id: str) -> None:
        """Disband units until a house is within its supply limits.

        The real game asks the player which units to remove. Here the engine
        removes from the largest armies, cheapest unit first.
        """
        guard = 0
        while not self.state.within_supply(house_id) and guard < 100:
            guard += 1
            armies = [
                (self.state.regions[rid].total(include_routed=False), rid)
                for rid in self.state.controlled_regions(house_id)
                if any(
                    unit is not UnitType.SHIP
                    for unit in self.state.regions[rid].units
                )
            ]
            if not armies:
                break
            _, region_id = max(armies)
            for unit in (UnitType.FOOTMAN, UnitType.SIEGE, UnitType.KNIGHT):
                if self.state.regions[region_id].units.get(unit, 0) > 0:
                    apply_casualties(self.state, region_id, house_id, 1)
                    break
            else:  # pragma: no cover - defensive
                break

    def _gain_power_per_crown(self) -> None:
        for house_id, house in self.state.houses.items():
            if house.eliminated:
                continue
            gained = sum(
                self.map.regions[rid].power
                for rid in self.state.controlled_regions(house_id)
            )
            house.power += gained
        self.state.note("power gained per crown icon")

    def _resolve_wildlings(self) -> None:
        """Resolve a Wildling attack without bidding (documented stub)."""
        threat = self.state.wildling_threat
        if threat <= 0:
            self.state.note("wildlings threaten but the track is empty")
            return
        # Nobody bids in the current model, so the Wildlings always win if the
        # track is non-empty. Consequences are not modelled yet.
        self.state.note(f"wildlings attack with strength {threat} (no bid modelled)")
        self.state.wildling_threat = 0

    # -- Bidding phase ---------------------------------------------------

    def _next_bidder(self) -> str | None:
        for house_id in self.state.tracks[Track.IRON_THRONE]:
            house = self.state.houses[house_id]
            if house.eliminated:
                continue
            if house_id not in self.state.pending_bids:
                return house_id
        return None

    def _bidding_actions(self, house: str) -> list[Action]:
        house_state = self.state.houses[house]
        return [
            Action.bid(house, amount) for amount in range(house_state.power + 1)
        ]

    def _apply_bid(self, action: Action) -> None:
        house_id = action.house or ""
        amount = max(0, min(action.amount or 0, self.state.houses[house_id].power))
        self.state.pending_bids[house_id] = amount

    def _resolve_bids(self) -> None:
        bids = self.state.pending_bids
        tracks = [_TRACK_BY_VALUE[name] for name in self.state.bid_tracks]
        for house_id, amount in bids.items():
            self.state.houses[house_id].power -= amount
        for track in tracks:
            current = self.state.tracks[track]
            previous = {house: index for index, house in enumerate(current)}
            ordered = sorted(
                (h for h in current if not self.state.houses[h].eliminated),
                key=lambda h: (-bids.get(h, 0), previous[h]),
            )
            eliminated = [h for h in current if self.state.houses[h].eliminated]
            self.state.tracks[track] = ordered + eliminated
        self.state.note(f"bids resolved for {self.state.bid_tracks}")
        self.state.pending_bids = {}
        self.state.bid_tracks = []
        self.state.phase = (
            Phase.MUSTERING if self.state.mustering_points else Phase.PLANNING
        )

    # -- Mustering phase -------------------------------------------------

    def _next_musterer(self) -> str | None:
        for house_id in self.state.tracks[Track.IRON_THRONE]:
            house = self.state.houses[house_id]
            if house.eliminated:
                continue
            if self.state.mustering_points.get(house_id, 0) > 0:
                return house_id
        return None

    def _mustering_actions(self, house: str) -> list[Action]:
        actions: list[Action] = []
        for region_id in self.state.controlled_regions(house):
            terrain = self.map.regions[region_id].terrain
            if terrain is Terrain.PORT:
                actions.append(Action.muster(house, region_id, UnitType.SHIP.value))
                continue
            if terrain is Terrain.SEA:
                actions.append(Action.muster(house, region_id, UnitType.SHIP.value))
                continue
            for unit in (UnitType.FOOTMAN, UnitType.KNIGHT, UnitType.SIEGE):
                if self._muster_is_legal(house, region_id, unit):
                    actions.append(Action.muster(house, region_id, unit.value))
        actions.append(Action.pass_(house))
        return actions

    def _muster_is_legal(self, house: str, region_id: str, unit: UnitType) -> bool:
        """Would mustering one more unit here keep the house within supply?"""
        if self.state.mustering_points.get(house, 0) <= 0:
            return False
        if unit is UnitType.SHIP:
            return True

        controlled = self.state.controlled_regions(house)
        limits = self.state.houses[house].supply_limit()
        sizes: list[int] = []
        for rid in controlled:
            region = self.state.regions[rid]
            size = sum(
                count for u, count in region.units.items() if u is not UnitType.SHIP
            )
            if rid == region_id:
                size += 1
            if size > 0:
                sizes.append(size)
        if region_id not in controlled:
            sizes.append(1)
        sizes.sort(reverse=True)
        return all(
            index < len(limits) and size <= limits[index]
            for index, size in enumerate(sizes)
        )

    def _apply_muster(self, action: Action) -> None:
        house = action.house or ""
        region_id = action.region or ""
        unit = UnitType(action.unit)
        region = self.state.regions[region_id]
        region.units[unit] = region.units.get(unit, 0) + 1
        region.controller = house
        self.state.mustering_points[house] = (
            self.state.mustering_points.get(house, 0) - 1
        )
        self.state.note(f"{house} musters {unit.value} in {region_id}")

    # -- Planning phase --------------------------------------------------

    def _planning_actions(self, house: str) -> list[Action]:
        actions: list[Action] = []
        house_state = self.state.houses[house]
        stars_left = self.state.star_allowance(house) - house_state.stars_placed()
        for region_id, region in self.state.regions.items():
            if region.controller != house or region.is_empty():
                continue
            # A region holds at most one order token, and once placed it stays
            # put. Offering the space again would let an agent churn its orders
            # forever (and is not legal in the real game).
            if region.order is not None:
                continue
            for order_key in house_state.orders_available:
                spec = ORDERS_BY_KEY[order_key]
                if spec.starred and stars_left <= 0:
                    continue
                if spec.order_type is OrderType.MARCH and not region.units:
                    continue
                actions.append(Action.place_order(house, region_id, order_key))
        actions.append(Action.pass_(house))
        return actions

    def _apply_place_order(self, action: Action) -> None:
        house = action.house or ""
        region_id = action.region or ""
        order_key = action.order or ""
        house_state = self.state.houses[house]
        region = self.state.regions[region_id]

        # Return any order already in the region to its owner's pool.
        if region.order is not None and region.order_house is not None:
            previous_owner = self.state.houses[region.order_house]
            if region.order in previous_owner.orders_placed:
                previous_owner.orders_placed.remove(region.order)
                previous_owner.orders_available.append(region.order)

        house_state.orders_available.remove(order_key)
        house_state.orders_placed.append(order_key)
        region.order = order_key
        region.order_house = house

    def _apply_pass(self, action: Action) -> None:
        house = action.house or ""
        phase = self.state.phase
        if phase is Phase.PLANNING and house in self.state.pending_planning:
            self.state.pending_planning.remove(house)
        elif phase is Phase.ACTION:
            # Passing on an action step forfeits the rest of this house's
            # orders of that type.
            for region in self.state.regions.values():
                if region.order_house != house or region.order is None:
                    continue
                if ORDERS_BY_KEY[region.order].order_type is self._step_order_type():
                    region.order = None
                    region.order_house = None
        elif phase is Phase.MUSTERING:
            self.state.mustering_points[house] = 0

    # -- Action phase ----------------------------------------------------

    def _step_order_type(self) -> OrderType:
        return {
            ActionStep.RAID: OrderType.RAID,
            ActionStep.MARCH: OrderType.MARCH,
            ActionStep.CONSOLIDATE_POWER: OrderType.CONSOLIDATE_POWER,
        }[self.state.action_step]

    def _next_action_house(self) -> str | None:
        order_type = self._step_order_type()
        for house_id in self.state.tracks[Track.IRON_THRONE]:
            if self.state.houses[house_id].eliminated:
                continue
            if self._has_resolvable_order(house_id, order_type):
                return house_id
        return None

    def _has_resolvable_order(self, house: str, order_type: OrderType) -> bool:
        for region_id, order_key in self.state.orders_on_board(house):
            if ORDERS_BY_KEY[order_key].order_type is not order_type:
                continue
            if self._order_actions(house, region_id, order_type):
                return True
        return False

    def _action_actions(self, house: str) -> list[Action]:
        order_type = self._step_order_type()
        matches = [
            (region_id, order_key)
            for region_id, order_key in self.state.orders_on_board(house)
            if ORDERS_BY_KEY[order_key].order_type is order_type
        ]
        if not matches:
            return []
        actions: list[Action] = []
        for region_id, _ in matches:
            actions.extend(self._order_actions(house, region_id, order_type))
        # The active house may always decline and forfeit this order.
        actions.append(Action.pass_(house))
        return actions

    def _order_actions(self, house: str, region_id: str, order_type: OrderType) -> list[Action]:
        if order_type is OrderType.RAID:
            return self._raid_actions(house, region_id)
        if order_type is OrderType.MARCH:
            return self._march_actions(house, region_id)
        if order_type is OrderType.CONSOLIDATE_POWER:
            return [Action.resolve_consolidate(house, region_id)]
        return []

    def _raid_actions(self, house: str, region_id: str) -> list[Action]:
        spec = ORDERS_BY_KEY[self.state.regions[region_id].order or ""]
        allowed = {OrderType.SUPPORT, OrderType.CONSOLIDATE_POWER}
        if spec.starred:
            allowed.add(OrderType.DEFEND)
        actions: list[Action] = []
        for neighbour in self.map.neighbours_of(region_id):
            target = self.state.regions[neighbour]
            if target.order is None or target.order_house == house:
                continue
            if ORDERS_BY_KEY[target.order].order_type in allowed:
                actions.append(Action.resolve_raid(house, region_id, neighbour))
        return actions

    def _march_actions(self, house: str, region_id: str) -> list[Action]:
        region = self.state.regions[region_id]
        if region.is_empty():
            return []
        actions: list[Action] = []
        for target in sorted(self._march_targets(house, region_id)):
            actions.append(Action.march(house, region_id, target))
        return actions

    def _march_targets(self, house: str, origin: str) -> set[str]:
        region = self.state.regions[origin]
        origin_terrain = self.map.regions[origin].terrain
        has_land = any(unit is not UnitType.SHIP for unit in region.units)
        has_ships = region.units.get(UnitType.SHIP, 0) > 0
        targets: set[str] = set()

        if has_ships:
            for neighbour in self.map.neighbours_of(origin):
                if self.map.regions[neighbour].terrain in (Terrain.SEA, Terrain.PORT):
                    targets.add(neighbour)

        if has_land and origin_terrain is Terrain.LAND:
            targets |= set(self.map.land_neighbours_of(origin))
            targets |= self._transport_targets(house, origin)

        targets.discard(origin)
        return targets

    def _naval_regions(self, house: str) -> set[str]:
        return {
            rid
            for rid, region in self.state.regions.items()
            if region.controller == house
            and self.map.regions[rid].terrain in (Terrain.SEA, Terrain.PORT)
            and region.units.get(UnitType.SHIP, 0) > 0
        }

    def _transport_targets(self, house: str, origin: str) -> set[str]:
        """Land regions reachable by moving across a chain of owned ships."""
        naval = self._naval_regions(house)
        frontier = [
            rid
            for rid in self.map.neighbours_of(origin)
            if rid in naval
        ]
        seen = set(frontier)
        while frontier:
            current = frontier.pop()
            for neighbour in self.map.neighbours_of(current):
                if neighbour in naval and neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append(neighbour)
        targets: set[str] = set()
        for sea in seen:
            for land in self.map.land_neighbours_of(sea):
                targets.add(land)
        return targets

    def _apply_raid(self, action: Action) -> None:
        house = action.house or ""
        region_id = action.region or ""
        target_id = action.target or ""
        target = self.state.regions[target_id]
        removed = target.order
        owner = target.order_house
        target.order = None
        target.order_house = None
        source = self.state.regions[region_id]
        source.order = None
        source.order_house = None
        if removed and owner and removed in self.state.houses[owner].orders_placed:
            self.state.houses[owner].orders_placed.remove(removed)
            self.state.houses[owner].orders_available.append(removed)
        self.state.note(f"{house} raids {removed} from {target_id}")

    def _apply_march(self, action: Action) -> None:
        house = action.house or ""
        origin = action.region or ""
        target_id = action.target or ""
        source = self.state.regions[origin]
        target = self.state.regions[target_id]
        target_terrain = self.map.regions[target_id].terrain

        # Capture the attacker's own March order bonus before clearing it.
        attacker_order_strength = order_modifier(self.state, origin, house, "attacker")
        source.order = None
        source.order_house = None

        if target_terrain in (Terrain.SEA, Terrain.PORT):
            moving = {UnitType.SHIP: source.units.get(UnitType.SHIP, 0)}
            source.units.pop(UnitType.SHIP, None)
        else:
            moving = {
                unit: count
                for unit, count in source.units.items()
                if unit is not UnitType.SHIP
            }
            for unit in moving:
                source.units.pop(unit, None)

        enemy_here = (
            not target.is_empty() and target.controller not in (None, house)
        )
        if enemy_here:
            defender = target.controller or ""
            # The two forces stay separate until the battle is resolved; see
            # agot.engine.combat for why merging them would be wrong.
            self.state.pending_battle = {
                "region": target_id,
                "attacker": house,
                "defender": defender,
                "origin": origin,
                "attacker_card": None,
                "defender_card": None,
                "attacker_units": moving,
                "defender_units": dict(target.units),
                "attacker_order_strength": attacker_order_strength,
                "defender_order_strength": order_modifier(
                    self.state, target_id, defender, "defender"
                ),
            }
            self.state.phase = Phase.COMBAT
            self.state.note(f"{house} marches into {target_id} and attacks {defender}")
        else:
            for unit, count in moving.items():
                if count:
                    target.units[unit] = target.units.get(unit, 0) + count
            target.controller = house
            self.state.note(f"{house} marches {origin} -> {target_id}")
        if source.is_empty() and not source.routed:
            source.controller = (
                source.influence_token
                if source.influence_token is not None
                else (source.owner if source.owner else None)
            )

    def _apply_consolidate(self, action: Action) -> None:
        house = action.house or ""
        region_id = action.region or ""
        region = self.state.regions[region_id]
        spec = ORDERS_BY_KEY[region.order or ""]
        gained = self.map.regions[region_id].power
        self.state.houses[house].power += gained
        if spec.starred and region.influence_token is None:
            region.influence_token = house
        region.order = None
        region.order_house = None
        self.state.note(f"{house} consolidates in {region_id} for {gained} power")

    def _next_action_step(self) -> None:
        if self.state.action_step is ActionStep.RAID:
            self.state.action_step = ActionStep.MARCH
        elif self.state.action_step is ActionStep.MARCH:
            self.state.action_step = ActionStep.CONSOLIDATE_POWER
        else:
            self._clear_orders()
            self.state.phase = Phase.CLEANUP

    def _clear_orders(self) -> None:
        for region in self.state.regions.values():
            if region.order is None:
                continue
            owner = region.order_house
            if owner and region.order in self.state.houses[owner].orders_placed:
                self.state.houses[owner].orders_placed.remove(region.order)
                self.state.houses[owner].orders_available.append(region.order)
            region.order = None
            region.order_house = None

    # -- Combat phase ----------------------------------------------------

    def _next_combat_chooser(self) -> str | None:
        battle = self.state.pending_battle
        if battle is None:
            return None
        if battle["attacker_card"] is None:
            return battle["attacker"]
        if battle["defender_card"] is None:
            return battle["defender"]
        return None

    def _combat_actions(self, house: str) -> list[Action]:
        house_state = self.state.houses[house]
        if not house_state.house_cards_available:
            # Every card has been played; the house fights with no card.
            return [Action.play_house_card(house, None)]
        return [
            Action.play_house_card(house, card)
            for card in house_state.house_cards_available
        ]

    def _apply_house_card(self, action: Action) -> None:
        battle = self.state.pending_battle
        if battle is None:  # pragma: no cover - defensive
            return
        house = action.house or ""
        card = action.card
        if card is not None:
            house_state = self.state.houses[house]
            house_state.house_cards_available.remove(card)
            house_state.house_cards_used.append(card)
        if house == battle["attacker"]:
            battle["attacker_card"] = card
        else:
            battle["defender_card"] = card

    def _resolve_pending_battle(self) -> None:
        battle = self.state.pending_battle
        if battle is None:  # pragma: no cover - defensive
            self.state.phase = Phase.ACTION
            return
        self.state.pending_battle = None
        self.state.phase = Phase.ACTION

        attacker = battle["attacker"]
        defender = battle["defender"]
        region_id = battle["region"]

        # A defeated defender's order token is discarded by the battle; return
        # it to its owner's pool so it can be placed again next round.
        defender_order = self.state.regions[region_id].order
        defender_order_house = self.state.regions[region_id].order_house

        attacker_blade = self._should_use_blade(attacker, region_id, battle)
        defender_blade = self._should_use_blade(defender, region_id, battle)

        result = resolve_and_apply(
            self.state,
            region_id,
            attacker,
            defender,
            battle["attacker_units"],
            battle["defender_units"],
            battle["attacker_card"],
            battle["defender_card"],
            attacker_origin=battle["origin"],
            attacker_order_strength=battle["attacker_order_strength"],
            defender_order_strength=battle["defender_order_strength"],
            attacker_blade=attacker_blade,
            defender_blade=defender_blade,
        )
        if (
            defender_order is not None
            and defender_order_house is not None
            and defender_order in self.state.houses[defender_order_house].orders_placed
        ):
            house_state = self.state.houses[defender_order_house]
            house_state.orders_placed.remove(defender_order)
            house_state.orders_available.append(defender_order)
        if attacker_blade:
            self.state.houses[attacker].blade_used = True
        if defender_blade:
            self.state.houses[defender].blade_used = True
        self.state.note(
            f"battle in {region_id}: {result.winner} defeats {result.loser} "
            f"({result.attacker.total} vs {result.defender.total})"
        )
        self._recycle_house_cards_for(attacker)
        self._recycle_house_cards_for(defender)

    def _should_use_blade(
        self, house: str, region_id: str, battle: dict
    ) -> bool:
        """Use the Valyrian Steel Blade when the holder is in the fight.

        The blade is worth +1 and is once per round. The engine spends it when
        the holder is a participant and their side is not already clearly ahead.
        """
        if house not in (battle["attacker"], battle["defender"]):
            return False
        if self.state.blade_holder() != house:
            return False
        house_state = self.state.houses[house]
        if house_state.blade_used:
            return False
        # Only spend it if the battle is close enough for +1 to matter.
        preview = resolve_battle(
            self.state,
            region_id,
            battle["attacker"],
            battle["defender"],
            battle["attacker_units"],
            battle["defender_units"],
            battle["attacker_card"],
            battle["defender_card"],
            attacker_order_strength=battle["attacker_order_strength"],
            defender_order_strength=battle["defender_order_strength"],
        )
        if house == battle["attacker"]:
            mine, theirs = preview.attacker.total, preview.defender.total
        else:
            mine, theirs = preview.defender.total, preview.attacker.total
        return mine <= theirs

    def _recycle_house_cards_for(self, house: str) -> None:
        """Return a house's cards once it has none left in hand."""
        house_state = self.state.houses[house]
        if house_state.house_cards_available:
            return
        house_state.house_cards_available = list(house_state.house_cards_used)
        house_state.house_cards_used = []

    # -- Cleanup phase ---------------------------------------------------

    def _run_cleanup_phase(self) -> None:
        state = self.state
        # Routed units recover.
        for region in state.regions.values():
            for unit, count in list(region.routed.items()):
                region.units[unit] = region.units.get(unit, 0) + count
            region.routed = {}

        self._clear_orders()

        for house in state.houses.values():
            house.blade_used = False
            house.orders_placed = []
            house.orders_available = [spec.key for spec in ORDERS_BY_KEY.values()]

        for region in state.regions.values():
            if region.is_empty() and not region.routed:
                if region.influence_token is not None:
                    region.controller = region.influence_token
                elif region.owner is not None:
                    region.controller = region.owner
                else:
                    region.controller = None

        # Victory check.
        for house_id in state.tracks[Track.IRON_THRONE]:
            if state.castle_count(house_id) >= state.victory_castles:
                self._declare_winner(house_id, "controls enough castles")
                return

        if state.round >= state.max_rounds:
            best = max(
                state.tracks[Track.IRON_THRONE],
                key=lambda h: (state.castle_count(h), -state.track_position(h, Track.IRON_THRONE)),
            )
            self._declare_winner(best, "most castles after the final round")
            return

        state.round += 1
        state.pending_planning = list(state.tracks[Track.IRON_THRONE])
        state.mustering_points = {}
        state.bid_tracks = []
        state.pending_bids = {}
        state.westeros_drawn = {name: None for name in state.westeros_drawn}
        state.note(f"round {state.round} begins")
        state.phase = Phase.WESTEROS

    def _declare_winner(self, house: str, reason: str) -> None:
        self.state.winner = house
        self.state.phase = Phase.GAME_OVER
        self.state.active_house = None
        self.state.note(f"{house} wins: {reason}")

    # -- misc ------------------------------------------------------------

    def region_controller(self, region_id: str) -> str | None:
        return self.state.regions[region_id].controller

    def summary(self) -> dict:
        """Compact description of the state, used by the CLI and logging."""
        return {
            "round": self.state.round,
            "phase": self.state.phase.value,
            "active_house": self.state.active_house,
            "winner": self.state.winner,
            "castles": {
                house: self.state.castle_count(house) for house in self.state.houses
            },
            "power": {
                house: self.state.houses[house].power for house in self.state.houses
            },
        }
