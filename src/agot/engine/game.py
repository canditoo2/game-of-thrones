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
* Wildling attacks model the secret Power bid, the Night's Watch strength and
  the per-card consequences, but sub-choices *inside* a consequence (exactly
  which units are destroyed, which Influence track is moved to, which House
  card is discarded) are resolved by a deterministic default.
* Neutral Force tokens (King's Landing 5, The Eyrie 6) defend their area with
  their printed strength; the attacker wins ties, and neither House cards nor
  the Valyrian Steel Blade may be used against them (rulebook page 26).
* House cards are chosen by the players (that decision *is* modelled), but
  special card abilities are not implemented, so only printed strengths and
  icons matter.

The rule mechanics that *are* modelled as explicit player decisions include
order resolution one house at a time (not one house at a time exhausting its
orders), declining an individual order, choosing which units march, establishing
control with a Power token, handling a Special Consolidate Power Order's
mustering, the Messenger Raven, the Valyrian Steel Blade, and the secret bid at
the heart of a Wildling attack.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product

from agot.engine.actions import Action, ActionKind
from agot.engine.combat import (
    order_modifier,
    resolve_and_apply,
    support_strength,
    units_strength,
)
from agot.engine.enums import (
    GARRISON_STRENGTH,
    MUSTER_COST,
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

    def __init__(
        self,
        state: GameState,
        config: GameConfig,
        rng: random.Random,
        *,
        advance: bool = True,
    ) -> None:
        self.state = state
        self.config = config
        self.rng = rng
        self.map = load_map()
        self._houses = load_houses()
        self._house_cards = load_house_cards()
        self._westeros = load_westeros()
        self._wildlings = load_wildlings()
        # ``clone`` disables this: a cloned state is already settled, and
        # re-running the phase machine would advance the turn-order cursor.
        if advance:
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

        # Home areas start under their owner's control and hold a Garrison.
        for region in game_map.regions.values():
            if region.home:
                state.regions[region.id].owner = region.home
                state.regions[region.id].controller = region.home
                state.regions[region.id].garrison = GARRISON_STRENGTH

        # Neutral Force tokens defend the areas printed on them (rulebook page 4).
        for region_id, strength in game_map.neutral_forces.items():
            state.regions[region_id].neutral_force = strength
            state.note(f"neutral force {strength} placed in {region_id}")

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
        """Deep copy, used by tree search.

        The clone does not re-run the phase machine: the source state is
        already settled, and advancing would move the Action Phase cursor.
        """
        return Game(
            self.state.clone(),
            self.config,
            random.Random(self.rng.random()),
            advance=False,
        )

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
        if phase is Phase.RAVEN:
            return self._raven_actions(house)
        if phase is Phase.MUSTERING:
            return self._mustering_actions(house)
        if phase is Phase.BIDDING:
            return self._bidding_actions(house)
        if phase is Phase.WESTEROS_CHOICE:
            return self._westeros_choice_actions(house)
        if phase is Phase.WILDLING_ATTACK:
            return self._wildling_bid_actions(house)
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
        elif kind is ActionKind.UPGRADE:
            self._apply_upgrade(action)
        elif kind is ActionKind.BID:
            self._apply_bid(action)
        elif kind is ActionKind.WESTEROS_CHOICE:
            self._apply_westeros_choice(action)
        elif kind is ActionKind.PLAY_HOUSE_CARD:
            self._apply_house_card(action)
        elif kind is ActionKind.RESOLVE_RAID:
            self._apply_raid(action)
        elif kind is ActionKind.MARCH:
            self._apply_march(action)
        elif kind is ActionKind.RESOLVE_CONSOLIDATE:
            self._apply_consolidate(action)
        elif kind is ActionKind.CONSOLIDATE_MUSTER:
            self._apply_consolidate_muster(action)
        elif kind is ActionKind.USE_BLADE:
            self._apply_use_blade(action)
        elif kind is ActionKind.RAVEN_SWAP:
            self._apply_raven_swap(action)
        elif kind is ActionKind.RAVEN_DECK_TOP:
            self._apply_raven_deck(action, to_bottom=False)
        elif kind is ActionKind.RAVEN_DECK_BOTTOM:
            self._apply_raven_deck(action, to_bottom=True)
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
        for _ in range(500):
            if self.state.phase is Phase.GAME_OVER:
                self.state.active_house = None
                return

            self._check_victory()
            if self.state.phase is Phase.GAME_OVER:
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
            elif self.state.phase is Phase.WESTEROS_CHOICE:
                choice = self.state.westeros_choice
                if choice is None:
                    self.state.phase = Phase.WESTEROS
                    continue
                self.state.active_house = choice["holder"]
                return
            elif self.state.phase is Phase.WILDLING_ATTACK:
                self.state.active_house = self._next_wildling_bidder()
                if self.state.active_house is not None:
                    return
                self._resolve_wildling_attack()
            elif self.state.phase is Phase.MUSTERING:
                self.state.active_house = self._next_musterer()
                if self.state.active_house is not None:
                    return
                if self.state.return_phase is Phase.ACTION:
                    # A Special Consolidate Power Order mustered in its own
                    # area; resume the Action Phase where it left off.
                    self.state.return_phase = None
                    self.state.phase = Phase.ACTION
                else:
                    self.state.phase = Phase.PLANNING
            elif self.state.phase is Phase.PLANNING:
                house = self._next_planner()
                if house is not None:
                    self.state.active_house = house
                    return
                # Assigning orders is done; the Messenger Raven holder acts
                # before orders are revealed (rulebook pages 11-12).
                self.state.phase = Phase.RAVEN
            elif self.state.phase is Phase.RAVEN:
                self.state.active_house = self._next_raven_actor()
                if self.state.active_house is not None:
                    return
                self.state.phase = Phase.ACTION
                self.state.action_step = ActionStep.RAID
                self.state.action_house_last = None
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

    def _check_victory(self) -> None:
        """The game ends immediately when a house controls enough castles."""
        if self.state.winner is not None:
            return
        for house_id in self.state.tracks[Track.IRON_THRONE]:
            if self.state.houses[house_id].eliminated:
                continue
            if self.state.castle_count(house_id) >= self.state.victory_castles:
                self._declare_winner(house_id, "controls enough castles")
                return

    # -- Westeros phase --------------------------------------------------

    def _run_westeros_phase(self) -> None:
        if self.state.pending_westeros is None:
            if self.state.round == 1 or not self.config.use_westeros_decks:
                self.state.phase = Phase.PLANNING
                return

            self.state.order_restrictions = []
            drawn: list[str] = []
            for deck_name in ("deck_i", "deck_ii", "deck_iii"):
                card_id = self._draw_westeros(deck_name)
                self.state.westeros_drawn[deck_name] = card_id
                if card_id is not None:
                    drawn.append(card_id)

            # Every Wildling icon shown this round advances the threat token,
            # and an attack triggered at 12 resolves before the cards' own
            # effects (rulebook page 7).
            icons = sum(
                1
                for card_id in drawn
                if self._westeros.cards[card_id].get("advance_wildlings")
            )
            self.state.pending_westeros = drawn
            if icons:
                self._advance_wildlings(icons)
                if self.state.phase is Phase.WILDLING_ATTACK:
                    return

        self._drain_westeros()

    def _drain_westeros(self) -> None:
        """Resolve the Westeros cards still waiting this round.

        A card may hand control to the bidding step (Clash of Kings), to a
        Wildling attack, or to an Influence token's holder (a holders'-choice
        card). The unresolved cards stay in ``state.pending_westeros`` and are
        drained once that step finishes, because all three cards must still
        resolve (rulebook page 7).
        """
        state = self.state
        queue = state.pending_westeros or []
        while queue:
            card_id = queue.pop(0)
            card = self._westeros.cards[card_id]
            if card["effect"] == "influence_decision":
                holder = self._westeros_choice_holder(card)
                options = list(card.get("options") or [])
                if holder is None or state.houses[holder].eliminated or not options:
                    state.note(f"{card['name']}: holder cannot choose; no effect")
                    continue
                state.pending_westeros = queue
                state.westeros_choice = {
                    "card": card_id,
                    "holder": holder,
                    "options": options,
                }
                state.phase = Phase.WESTEROS_CHOICE
                return
            state.pending_westeros = queue
            if self._apply_westeros_effect(card_id):
                return

        state.pending_westeros = None
        state.bid_tracks = []
        if self._any_mustering_points():
            state.phase = Phase.MUSTERING
        else:
            state.phase = Phase.PLANNING

    def _apply_westeros_effect(self, card_id: str) -> bool:
        """Resolve one Westeros card's effect.

        Returns ``True`` when control passes to another phase; the caller must
        then stop draining. Unresolved cards stay in ``pending_westeros``.
        """
        state = self.state
        card = self._westeros.cards[card_id]
        effect = card["effect"]
        if effect == "mustering":
            self._grant_mustering_points()
        elif effect == "supply":
            self._recalculate_supply()
        elif effect == "gain_power_per_crown":
            self._gain_power_per_crown()
        elif effect == "bid_all_tracks":
            state.bid_tracks = ["iron_throne", "fiefdoms", "kings_court"]
            state.pending_bids = {}
            state.phase = Phase.BIDDING
            return True
        elif effect == "wildlings_attack":
            self._resolve_wildlings()
            if state.phase is Phase.WILDLING_ATTACK:
                return True
        elif effect == "order_restriction":
            restricted = card.get("restricted_order")
            if restricted:
                state.order_restrictions.append(restricted)
            state.note(f"{card['name']}: {restricted} orders banned this round")
        elif effect == "none":
            state.note(f"{card['name']}: nothing happens")
        else:  # pragma: no cover - defensive
            state.note(f"{card['name']}: unknown effect {effect!r}")
        return False

    def _westeros_choice_holder(self, card: dict) -> str | None:
        """The house that owns the Influence token named by a choice card."""
        holder_kind = card.get("holder")
        if holder_kind == "iron_throne":
            return self.state.tracks[Track.IRON_THRONE][0]
        if holder_kind == "raven":
            return self.state.raven_holder()
        if holder_kind == "blade":
            return self.state.blade_holder()
        return None  # pragma: no cover - defensive

    def _westeros_choice_actions(self, house: str) -> list[Action]:
        choice = self.state.westeros_choice
        if choice is None or house != choice["holder"]:
            return []
        return [
            Action.westeros_choice(house, option) for option in choice["options"]
        ]

    def _apply_westeros_choice(self, action: Action) -> None:
        """Apply the holder's pick, then hand the queue back to the drain."""
        state = self.state
        choice = state.westeros_choice
        if choice is None:  # pragma: no cover - defensive
            return
        picked = action.card if action.card in choice["options"] else "none"
        state.westeros_choice = None
        card = self._westeros.cards[choice["card"]]
        if picked == "none":
            state.note(f"{card['name']}: holder chooses no effect")
        else:
            state.note(
                f"{card['name']}: holder chooses {self._westeros.cards[picked]['name']}"
            )
            self._apply_westeros_effect(picked)
        if state.phase is Phase.WESTEROS_CHOICE:  # pragma: no cover - defensive
            state.phase = Phase.WESTEROS

    def _any_mustering_points(self) -> bool:
        return any(
            region.mustering_points > 0 for region in self.state.regions.values()
        )

    def _advance_wildlings(self, amount: int = 1) -> None:
        """Advance the Wildling threat token and resolve an attack at the cap."""
        track_max = self._wildlings.track_max
        self.state.wildling_threat += amount
        if self.state.wildling_threat >= track_max:
            self.state.wildling_threat = track_max
            self.state.note(
                f"wildling threat reached {track_max}; the wildlings attack"
            )
            self._resolve_wildlings()

    def _draw_westeros(self, deck_name: str) -> str | None:
        state = self.state
        for _ in range(20):
            deck = state.westeros_decks.get(deck_name) or []
            position = state.westeros_deck_pos.get(deck_name, 0)
            if position >= len(deck):
                return None
            card_id = deck[position]
            state.westeros_deck_pos[deck_name] = position + 1
            if self._westeros.cards.get(card_id, {}).get("effect") == "winter_is_coming":
                # "Immediately shuffle this deck. Then draw and resolve a new card."
                remaining = list(deck[position + 1 :]) + [card_id]
                self.rng.shuffle(remaining)
                state.westeros_decks[deck_name] = remaining
                state.westeros_deck_pos[deck_name] = 0
                state.note("Winter is Coming: deck reshuffled and redrawn")
                continue
            return card_id
        return None  # pragma: no cover - defensive

    def _grant_mustering_points(self) -> None:
        for house_id, house in self.state.houses.items():
            if house.eliminated:
                continue
            for rid in self.state.controlled_regions(house_id):
                points = self.map.regions[rid].mustering
                if points:
                    self.state.regions[rid].mustering_points += points
        self.state.note("mustering points granted per area")

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
        removes from the largest army, cheapest unit first, and will remove
        routed units as well (they still count towards Supply).
        """
        guard = 0
        while not self.state.within_supply(house_id) and guard < 200:
            guard += 1
            armies = [
                (self.state.regions[rid].total(include_routed=True), rid)
                for rid in self.state.controlled_regions(house_id)
                if self.state.regions[rid].total(include_routed=True) >= 2
            ]
            if not armies:
                break
            _, region_id = max(armies)
            if not self._remove_one_unit(region_id):
                break  # pragma: no cover - defensive

    def _remove_one_unit(self, region_id: str) -> bool:
        region = self.state.regions[region_id]
        for unit in (
            UnitType.FOOTMAN,
            UnitType.SIEGE,
            UnitType.KNIGHT,
            UnitType.SHIP,
        ):
            if region.routed.get(unit, 0) > 0:
                region.routed[unit] -= 1
                if region.routed[unit] == 0:
                    del region.routed[unit]
                return True
            if region.units.get(unit, 0) > 0:
                region.units[unit] -= 1
                if region.units[unit] == 0:
                    del region.units[unit]
                break
        else:
            return False
        if region.is_empty() and not region.routed:
            if region.influence_token is not None:
                region.controller = region.influence_token
            else:
                region.controller = region.owner
        return True

    def _gain_power_per_crown(self) -> None:
        for house_id, house in self.state.houses.items():
            if house.eliminated:
                continue
            gained = sum(
                self.map.regions[rid].power
                for rid in self.state.controlled_regions(house_id)
            )
            gained += self._port_power_bonus(house_id)
            house.power += gained
        self.state.note("power gained per crown icon")

    def _port_power_bonus(self, house: str) -> int:
        """Extra Power from the "Game of Thrones" card (rulebook page 24).

        Each controlled Port that contains at least one of the house's Ships
        grants one Power token, as long as the connected sea area holds no
        enemy Ships.
        """
        bonus = 0
        for port_id in self.map.port_hosts:
            port = self.state.regions[port_id]
            own_ships = port.units.get(UnitType.SHIP, 0) + port.routed.get(
                UnitType.SHIP, 0
            )
            if port.controller != house or own_ships <= 0:
                continue
            sea_id = self.map.regions[port_id].sea
            if sea_id is not None:
                sea = self.state.regions[sea_id]
                enemy_ships = sea.units.get(UnitType.SHIP, 0) + sea.routed.get(
                    UnitType.SHIP, 0
                )
                if sea.controller not in (None, house) and enemy_ships > 0:
                    continue
            bonus += 1
        return bonus

    # -- Wildling attacks ------------------------------------------------

    def _resolve_wildlings(self) -> None:
        """Start a Wildling attack whose strength is the current threat.

        The attack runs through :attr:`Phase.WILDLING_ATTACK`, which asks every
        house to bid Power in Iron Throne order before the outcome and the
        Wildling card's consequences are applied (rulebook page 22).
        """
        self._setup_wildling_attack(strength=self.state.wildling_threat)

    def _draw_wildling(self) -> str | None:
        state = self.state
        if not state.wildling_deck:
            if not state.wildling_discard:
                return None
            state.wildling_deck = list(state.wildling_discard)
            state.wildling_discard = []
            self.rng.shuffle(state.wildling_deck)
        return state.wildling_deck.pop(0)

    def _setup_wildling_attack(
        self, *, strength: int, excluded: tuple[str, ...] = ()
    ) -> None:
        state = self.state
        state.wildling_attack = {
            "strength": strength,
            "bids": {},
            "excluded": list(excluded),
            "card": self._draw_wildling(),
            "return_phase": Phase.WESTEROS,
            "second_attack": None,
            "second_strength": 6,
        }
        state.note(f"wildlings attack with strength {strength}")
        state.phase = Phase.WILDLING_ATTACK

    def _next_wildling_bidder(self) -> str | None:
        attack = self.state.wildling_attack
        if attack is None:
            return None
        for house_id in self.state.tracks[Track.IRON_THRONE]:
            house = self.state.houses[house_id]
            if house.eliminated or house_id in attack["excluded"]:
                continue
            if house_id not in attack["bids"]:
                return house_id
        return None

    def _wildling_bid_actions(self, house: str) -> list[Action]:
        power = self.state.houses[house].power
        return [Action.bid(house, amount) for amount in range(power + 1)]

    def _apply_wildling_bid(self, house: str, amount: int) -> None:
        attack = self.state.wildling_attack
        if attack is None:  # pragma: no cover - defensive
            return
        attack["bids"][house] = max(0, min(amount, self.state.houses[house].power))

    def _resolve_wildling_attack(self) -> None:
        """Compare the Night's Watch and Wildling strengths, then the card."""
        state = self.state
        attack = state.wildling_attack
        if attack is None:  # pragma: no cover - defensive
            state.phase = Phase.WESTEROS
            return
        state.wildling_attack = None

        bids: dict[str, int] = attack["bids"]
        strength: int = attack["strength"]
        card_id: str | None = attack["card"]

        # All bid Power is discarded, win or lose (rulebook page 22).
        for house_id, amount in bids.items():
            state.houses[house_id].power -= amount
        nights_watch = sum(bids.values())

        wildlings_win = nights_watch < strength
        if wildlings_win:
            state.wildling_threat = max(0, strength - 2)
        else:
            state.wildling_threat = 0
        state.note(
            f"wildling attack: Night's Watch {nights_watch} vs wildlings "
            f"{strength}; {'wildlings win' if wildlings_win else 'Night watch holds'}"
        )

        if card_id is not None:
            participants = [h for h in state.tracks[Track.IRON_THRONE] if h in bids]
            card = self._wildlings.cards[card_id]
            branch = (
                "wildling_victory" if wildlings_win else "nights_watch_victory"
            )
            spec = card.get(branch) or {"effect": "nothing"}
            chosen = self._extreme_bidder(
                participants, bids, highest=not wildlings_win
            )
            others = [h for h in participants if h != chosen]
            self._apply_wildling_effect(spec, chosen, others, bids, attack)
            state.wildling_discard.append(card_id)
            state.note(
                f"{card['name']} ({branch}): {spec.get('effect')} for {chosen}"
            )

        second = attack.get("second_attack")
        if second:
            # "Preemptive Raid": the wildlings attack again, and the highest
            # bidder sits the second attack out.
            self._setup_wildling_attack(
                strength=attack.get("second_strength", 6),
                excluded=tuple(second),
            )
            return
        state.phase = attack.get("return_phase") or Phase.WESTEROS

    def _extreme_bidder(
        self, participants: list[str], bids: dict[str, int], *, highest: bool
    ) -> str | None:
        """The highest (or lowest) bidder, breaking ties by Iron Throne seat.

        The rulebook lets the Iron Throne holder decide a tie; the engine
        approximates that with the earliest tied house on the Iron Throne track.
        """
        if not participants:
            return None
        values = [bids[house] for house in participants]
        target = max(values) if highest else min(values)
        tied = [house for house in participants if bids[house] == target]
        order = self.state.tracks[Track.IRON_THRONE]
        return min(tied, key=order.index)

    def _apply_wildling_effect(
        self,
        spec: dict,
        chosen: str | None,
        others: list[str],
        bids: dict[str, int],
        attack: dict,
    ) -> None:
        """Apply one Wildling card consequence.

        The printed cards let the rewarded player make several sub-choices; the
        engine resolves them with a deterministic default (largest army first,
        strongest House card, best Iron Throne seat). See docs/roadmap.md.
        """
        effect = spec.get("effect", "nothing")
        if effect == "nothing" or chosen is None:
            return

        if effect == "replace_knights":
            self._replace_knights(chosen, spec.get("chosen"))
            for house in others:
                self._replace_knights(house, spec.get("others", 0))
        elif effect == "upgrade_footmen":
            self._upgrade_footmen(chosen, spec.get("chosen", 0))
        elif effect == "destroy_units":
            self._destroy_units_anywhere(chosen, spec.get("chosen", 0))
            for house in others:
                self._destroy_units_anywhere(house, spec.get("others", 0))
        elif effect == "destroy_units_at_fort":
            count = spec.get("chosen", 0)
            if not self._destroy_units_at_fortification(chosen, count):
                self._destroy_units_anywhere(chosen, count)
            for house in others:
                self._destroy_units_anywhere(house, spec.get("others", 0))
        elif effect == "muster_at_fort":
            self._auto_muster_at_best_fort(chosen)
        elif effect == "influence_to_bottom":
            for name in self._influence_targets(spec.get("chosen")):
                self._move_influence(_TRACK_BY_VALUE[name], chosen, to_top=False)
            for house in others:
                for name in self._influence_targets(spec.get("others")):
                    self._move_influence(
                        _TRACK_BY_VALUE[name], house, to_top=False
                    )
        elif effect == "influence_to_top":
            for name in self._influence_targets(spec.get("chosen")):
                self._move_influence(_TRACK_BY_VALUE[name], chosen, to_top=True)
        elif effect == "retrieve_house_card":
            self._retrieve_house_card(chosen)
        elif effect == "return_discard_pile":
            self._return_discard_pile(chosen)
        elif effect == "discard_house_cards":
            self._discard_house_cards(chosen, spec.get("chosen", "one"))
            for house in others:
                self._discard_house_cards(house, spec.get("others", "one"))
        elif effect == "adjust_supply":
            self._adjust_supply(chosen, spec.get("chosen", 0))
            for house in others:
                self._adjust_supply(house, spec.get("others", 0))
        elif effect == "discard_power":
            self._discard_power(chosen, spec.get("chosen", 0))
            for house in others:
                self._discard_power(house, spec.get("others", 0))
        elif effect == "second_attack":
            attack["second_attack"] = [chosen]
            attack["second_strength"] = spec.get("strength", 6)
        elif effect == "refund_bid":
            self.state.houses[chosen].power += bids.get(chosen, 0)

    @staticmethod
    def _influence_targets(value: object) -> list[str]:
        if value in (None, "none"):
            return []
        if value == "all":
            return ["iron_throne", "fiefdoms", "kings_court"]
        return [str(value)]

    def _move_influence(self, track: Track, house: str, *, to_top: bool) -> None:
        """Move a house to the top or bottom of an Influence track."""
        rest = [h for h in self.state.tracks[track] if h != house]
        alive = [h for h in rest if not self.state.houses[h].eliminated]
        dead = [h for h in rest if self.state.houses[h].eliminated]
        if to_top:
            self.state.tracks[track] = [house] + alive + dead
        else:
            self.state.tracks[track] = alive + [house] + dead
        self.state.note(
            f"{house} moves to the {'top' if to_top else 'bottom'} of {track.value}"
        )

    def _card_strength(self, house: str, card_id: str) -> int:
        for card in self._house_cards.deck(house):
            if card["id"] == card_id:
                return card["strength"]
        return 0

    def _destroy_units_anywhere(self, house: str, count: int) -> int:
        """Destroy up to ``count`` units, taking from the largest force first."""
        destroyed = 0
        while destroyed < count:
            candidates = [
                (self.state.regions[rid].total(include_routed=True), rid)
                for rid in self.state.controlled_regions(house)
                if self.state.regions[rid].total(include_routed=True) > 0
            ]
            if not candidates:
                break
            _, region_id = max(candidates)
            if not self._remove_one_unit(region_id):
                break
            destroyed += 1
        return destroyed

    def _destroy_units_at_fortification(self, house: str, count: int) -> bool:
        forts = [
            rid
            for rid in self.state.controlled_regions(house)
            if self.map.regions[rid].is_fortification
            and self.state.regions[rid].total(include_routed=True) > 0
        ]
        if not forts:
            return False
        region_id = max(
            forts,
            key=lambda rid: self.state.regions[rid].total(include_routed=True),
        )
        for _ in range(count):
            if not self._remove_one_unit(region_id):
                break
        return True

    def _replace_knights(self, house: str, count: int | None) -> None:
        """Swap Knights for Footmen in place; unit-pool limits are not modelled."""
        remaining = count
        for region_id in sorted(
            self.state.controlled_regions(house),
            key=lambda rid: -self.state.regions[rid].total(include_routed=True),
        ):
            region = self.state.regions[region_id]
            knights = region.units.get(UnitType.KNIGHT, 0)
            if knights <= 0:
                continue
            take = knights if remaining is None else min(knights, remaining)
            if take <= 0:
                continue
            region.units[UnitType.KNIGHT] = knights - take
            if region.units[UnitType.KNIGHT] <= 0:
                del region.units[UnitType.KNIGHT]
            region.units[UnitType.FOOTMAN] = (
                region.units.get(UnitType.FOOTMAN, 0) + take
            )
            if remaining is not None:
                remaining -= take
                if remaining <= 0:
                    return

    def _upgrade_footmen(self, house: str, count: int) -> None:
        """Turn up to ``count`` Footmen into Knights (no Supply change)."""
        remaining = count
        for region_id in sorted(
            self.state.controlled_regions(house),
            key=lambda rid: -self.state.regions[rid].total(include_routed=True),
        ):
            region = self.state.regions[region_id]
            while remaining > 0 and region.units.get(UnitType.FOOTMAN, 0) > 0:
                region.units[UnitType.FOOTMAN] -= 1
                if region.units[UnitType.FOOTMAN] <= 0:
                    del region.units[UnitType.FOOTMAN]
                region.units[UnitType.KNIGHT] = (
                    region.units.get(UnitType.KNIGHT, 0) + 1
                )
                remaining -= 1
            if remaining <= 0:
                return

    def _auto_muster_at_best_fort(self, house: str) -> None:
        """Muster automatically in the house's richest Castle or Stronghold."""
        forts = [
            rid
            for rid in self.state.controlled_regions(house)
            if self.map.regions[rid].is_fortification
        ]
        if not forts:
            return
        region_id = max(forts, key=lambda rid: self.map.regions[rid].mustering)
        region = self.state.regions[region_id]
        points = self.map.regions[region_id].mustering
        while points > 0:
            if points >= MUSTER_COST[UnitType.KNIGHT] and self._muster_fits(
                house, region_id, UnitType.KNIGHT
            ):
                region.units[UnitType.KNIGHT] = (
                    region.units.get(UnitType.KNIGHT, 0) + 1
                )
                points -= MUSTER_COST[UnitType.KNIGHT]
            elif self._muster_fits(house, region_id, UnitType.FOOTMAN):
                region.units[UnitType.FOOTMAN] = (
                    region.units.get(UnitType.FOOTMAN, 0) + 1
                )
                points -= MUSTER_COST[UnitType.FOOTMAN]
            else:
                break
        self.state.note(f"{house} musters at {region_id} from a Wildling reward")

    def _retrieve_house_card(self, house: str) -> None:
        house_state = self.state.houses[house]
        if not house_state.house_cards_used:
            return
        card = house_state.house_cards_used.pop()
        house_state.house_cards_available.append(card)
        self.state.note(f"{house} retrieves House card {card} from the discard")

    def _return_discard_pile(self, house: str) -> None:
        house_state = self.state.houses[house]
        house_state.house_cards_available.extend(house_state.house_cards_used)
        house_state.house_cards_used = []
        self.state.note(f"{house} returns its whole House card discard to hand")

    def _discard_house_cards(self, house: str, mode: str) -> None:
        house_state = self.state.houses[house]
        hand = house_state.house_cards_available
        if len(hand) <= 1:
            return
        if mode == "all_max":
            best = max(self._card_strength(house, card) for card in hand)
            discarded = [
                card for card in hand if self._card_strength(house, card) == best
            ]
        else:
            discarded = [max(hand, key=lambda card: self._card_strength(house, card))]
        for card in discarded:
            hand.remove(card)
            house_state.house_cards_used.append(card)
        self.state.note(f"{house} discards House card(s) {discarded}")

    def _adjust_supply(self, house: str, delta: int) -> None:
        house_state = self.state.houses[house]
        ceiling = len(SUPPLY_LIMITS) - 1
        house_state.supply_level = max(
            0, min(house_state.supply_level + delta, ceiling)
        )
        if delta < 0:
            self._enforce_supply(house)

    def _discard_power(self, house: str, amount: int) -> None:
        house_state = self.state.houses[house]
        house_state.power = max(0, house_state.power - amount)

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
        if self.state.phase is Phase.WILDLING_ATTACK:
            self._apply_wildling_bid(house_id, amount)
        else:
            self.state.pending_bids[house_id] = amount

    def _resolve_bids(self) -> None:
        """Resolve bidding for the next Influence track only.

        Bidding is sequential (Iron Throne, then Fiefdoms, then King's Court);
        all power bid is discarded, so later tracks are bid with what remains.
        Ties are broken automatically by the current Iron Throne position.
        """
        if not self.state.bid_tracks:
            self._resume_after_extra_step()
            return

        track_name = self.state.bid_tracks.pop(0)
        track = _TRACK_BY_VALUE[track_name]
        bids = self.state.pending_bids
        for house_id, amount in bids.items():
            self.state.houses[house_id].power -= amount

        iron_throne = self.state.tracks[Track.IRON_THRONE]
        current = self.state.tracks[track]
        previous = {house: index for index, house in enumerate(current)}
        throne_rank = {house: index for index, house in enumerate(iron_throne)}
        ordered = sorted(
            (h for h in current if not self.state.houses[h].eliminated),
            key=lambda h: (-bids.get(h, 0), throne_rank.get(h, 99), previous[h]),
        )
        eliminated = [h for h in current if self.state.houses[h].eliminated]
        self.state.tracks[track] = ordered + eliminated
        self.state.note(f"bids resolved for {track_name} ({bids})")
        self.state.pending_bids = {}

        if not self.state.bid_tracks:
            self._resume_after_extra_step()

    def _resume_after_extra_step(self) -> None:
        """Return to an interrupted Westeros phase, or move on if it is done."""
        state = self.state
        if state.pending_westeros:
            state.phase = Phase.WESTEROS
            return
        state.pending_westeros = None
        if self._any_mustering_points():
            state.phase = Phase.MUSTERING
        else:
            state.phase = Phase.PLANNING

    # -- Mustering phase -------------------------------------------------

    def _next_musterer(self) -> str | None:
        for house_id in self.state.tracks[Track.IRON_THRONE]:
            house = self.state.houses[house_id]
            if house.eliminated:
                continue
            if self.state.has_mustering_points(house_id):
                return house_id
        return None

    def _mustering_actions(self, house: str) -> list[Action]:
        actions: list[Action] = []
        for region_id in self.state.controlled_regions(house):
            static = self.map.regions[region_id]
            region = self.state.regions[region_id]
            points = region.mustering_points
            if points <= 0 or not static.is_fortification:
                continue
            if points >= MUSTER_COST[UnitType.FOOTMAN] and self._muster_fits(
                house, region_id, UnitType.FOOTMAN
            ):
                actions.append(Action.muster(house, region_id, UnitType.FOOTMAN.value))
            if points >= MUSTER_COST[UnitType.KNIGHT] and self._muster_fits(
                house, region_id, UnitType.KNIGHT
            ):
                actions.append(Action.muster(house, region_id, UnitType.KNIGHT.value))
            if points >= MUSTER_COST[UnitType.SIEGE] and self._muster_fits(
                house, region_id, UnitType.SIEGE
            ):
                actions.append(Action.muster(house, region_id, UnitType.SIEGE.value))
            if points >= 1 and region.units.get(UnitType.FOOTMAN, 0) > 0:
                actions.append(Action.upgrade(house, region_id, UnitType.KNIGHT.value))
                actions.append(Action.upgrade(house, region_id, UnitType.SIEGE.value))
            if points >= MUSTER_COST[UnitType.SHIP]:
                for destination in self._ship_muster_targets(house, region_id):
                    if self._muster_fits(house, destination, UnitType.SHIP):
                        actions.append(
                            Action.muster(
                                house,
                                region_id,
                                UnitType.SHIP.value,
                                target=destination,
                            )
                        )
        actions.append(Action.pass_(house))
        return actions

    def _ship_muster_targets(self, house: str, source: str) -> list[str]:
        """Port or adjacent sea areas where a Ship may be mustered."""
        targets: list[str] = []
        port_id = self.map.port_of(source)
        if port_id is not None:
            port = self.state.regions[port_id]
            if port.controller in (None, house) and port.total(include_routed=True) < 3:
                targets.append(port_id)
        for sea in sorted(self.map.sea_neighbours_of(source)):
            region = self.state.regions[sea]
            # Friendly and empty sea areas may receive a new Ship; only enemy
            # Ships block mustering there (rulebook page 9).
            if region.controller not in (None, house):
                continue
            targets.append(sea)
        return targets

    def _muster_fits(
        self, house: str, destination: str, unit: UnitType, count: int = 1
    ) -> bool:
        """Would adding ``count`` of ``unit`` in ``destination`` stay in Supply?"""
        controlled = self.state.controlled_regions(house)
        limits = self.state.houses[house].supply_limit()
        sizes: list[int] = []
        found = False
        for rid in controlled:
            region = self.state.regions[rid]
            size = region.total(include_routed=True)
            if rid == destination:
                size += count
                found = True
            if size >= 2:
                sizes.append(size)
        if not found and count >= 2:
            sizes.append(count)
        sizes.sort(reverse=True)
        return all(
            index < len(limits) and size <= limits[index]
            for index, size in enumerate(sizes)
        )

    def _apply_muster(self, action: Action) -> None:
        house = action.house or ""
        source = action.region or ""
        unit = UnitType(action.unit)
        cost = MUSTER_COST[unit]
        if unit is UnitType.SHIP:
            destination = action.target or source
            placement = self.state.regions[destination]
        else:
            placement = self.state.regions[source]
        placement.units[unit] = placement.units.get(unit, 0) + 1
        placement.controller = house
        self.state.regions[source].mustering_points -= cost
        self.state.note(f"{house} musters {unit.value} in {placement.region_id}")

    def _apply_upgrade(self, action: Action) -> None:
        house = action.house or ""
        region_id = action.region or ""
        unit = UnitType(action.unit)
        region = self.state.regions[region_id]
        region.units[UnitType.FOOTMAN] = region.units.get(UnitType.FOOTMAN, 0) - 1
        if region.units[UnitType.FOOTMAN] <= 0:
            del region.units[UnitType.FOOTMAN]
        region.units[unit] = region.units.get(unit, 0) + 1
        region.mustering_points -= 1
        self.state.note(f"{house} upgrades a footman to {unit.value} in {region_id}")

    # -- Planning phase --------------------------------------------------

    def _planning_required(self, house: str) -> list[str]:
        """Areas that still must receive an order token this Planning Phase.

        Every area containing units (including routed units and Ships in a
        Port) must be assigned exactly one order. Areas containing only a
        Garrison are exempt.
        """
        required = []
        for region_id, region in self.state.regions.items():
            if region.controller != house:
                continue
            if region.order is not None:
                continue
            if region.total(include_routed=True) > 0:
                required.append(region_id)
        return required

    def _next_planner(self) -> str | None:
        while self.state.pending_planning:
            house = self.state.pending_planning[0]
            if self.state.houses[house].eliminated or not self._planning_required(house):
                self.state.pending_planning.pop(0)
                continue
            return house
        return None

    def _order_is_restricted(self, order_key: str) -> bool:
        spec = ORDERS_BY_KEY[order_key]
        restrictions = self.state.order_restrictions
        return order_key in restrictions or spec.order_type.value in restrictions

    def _planning_actions(self, house: str) -> list[Action]:
        required = self._planning_required(house)
        if not required:
            return []
        actions: list[Action] = []
        house_state = self.state.houses[house]
        stars_left = self.state.star_allowance(house) - house_state.stars_placed()
        for region_id in required:
            region = self.state.regions[region_id]
            for order_key in house_state.orders_available:
                spec = ORDERS_BY_KEY[order_key]
                if spec.starred and stars_left <= 0:
                    continue
                if spec.order_type is OrderType.MARCH and not region.units:
                    continue
                if self._order_is_restricted(order_key):
                    continue
                actions.append(Action.place_order(house, region_id, order_key))
        # Placement is mandatory, but if the pool/restrictions leave no legal
        # order the player must still be able to finish.
        if not actions:
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
        elif phase is Phase.RAVEN:
            self.state.raven_used = True
        elif phase is Phase.ACTION:
            region_id = action.region
            if region_id is None:
                # Backwards-compatible fallback: forfeit every order of the
                # step's type.
                self._decline_orders_of_step(house)
            else:
                self._decline_order(house, region_id)
        elif phase is Phase.MUSTERING:
            for rid in self.state.controlled_regions(house):
                self.state.regions[rid].mustering_points = 0
        elif phase is Phase.COMBAT:
            battle = self.state.pending_battle
            if battle is not None:
                # Declining the Valyrian Steel Blade.
                battle["blade_done"] = True

    def _decline_order(self, house: str, region_id: str) -> None:
        """Resolve one order with no effect, returning its token to the pool."""
        region = self.state.regions[region_id]
        if region.order is None or region.order_house != house:
            return
        order_key = region.order
        region.order = None
        region.order_house = None
        house_state = self.state.houses[house]
        if order_key in house_state.orders_placed:
            house_state.orders_placed.remove(order_key)
            house_state.orders_available.append(order_key)

    def _decline_orders_of_step(self, house: str) -> None:
        order_type = self._step_order_type()
        for region_id, order_key in list(self.state.orders_on_board(house)):
            if ORDERS_BY_KEY[order_key].order_type is order_type:
                self._decline_order(house, region_id)

    # -- Action phase ----------------------------------------------------

    def _step_order_type(self) -> OrderType:
        return {
            ActionStep.RAID: OrderType.RAID,
            ActionStep.MARCH: OrderType.MARCH,
            ActionStep.CONSOLIDATE_POWER: OrderType.CONSOLIDATE_POWER,
        }[self.state.action_step]

    def _next_action_house(self) -> str | None:
        """The next house to resolve one order, cycling by Iron Throne order.

        Each house resolves a single order before the turn passes to the next
        house on the Iron Throne track; a house with no orders of the current
        type is skipped. The step ends once no house can act.
        """
        order_type = self._step_order_type()
        order = [
            house_id
            for house_id in self.state.tracks[Track.IRON_THRONE]
            if not self.state.houses[house_id].eliminated
        ]
        if not order:
            return None
        last = self.state.action_house_last
        start = order.index(last) + 1 if last in order else 0
        for offset in range(len(order)):
            house_id = order[(start + offset) % len(order)]
            if self.state.has_order_type(house_id, order_type):
                self.state.action_house_last = house_id
                return house_id
        self.state.action_house_last = None
        return None

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
            # A player may always resolve an order with no effect; only that
            # one token is removed (rulebook pages 13-16).
            actions.append(Action.pass_(house, region=region_id))
        return actions

    def _order_actions(self, house: str, region_id: str, order_type: OrderType) -> list[Action]:
        if order_type is OrderType.RAID:
            return self._raid_actions(house, region_id)
        if order_type is OrderType.MARCH:
            return self._march_actions(house, region_id)
        if order_type is OrderType.CONSOLIDATE_POWER:
            actions = [Action.resolve_consolidate(house, region_id)]
            spec = ORDERS_BY_KEY[self.state.regions[region_id].order or ""]
            # A Special Consolidate Power Order in a Castle or Stronghold may
            # muster there instead of collecting Power (rulebook page 22).
            if spec.starred and self.map.regions[region_id].is_fortification:
                actions.append(Action.consolidate_muster(house, region_id))
            return actions
        return []

    def _raid_actions(self, house: str, region_id: str) -> list[Action]:
        spec = ORDERS_BY_KEY[self.state.regions[region_id].order or ""]
        allowed = {OrderType.SUPPORT, OrderType.RAID, OrderType.CONSOLIDATE_POWER}
        if spec.starred:
            allowed.add(OrderType.DEFEND)
        source_terrain = self.map.regions[region_id].terrain
        actions: list[Action] = []
        for neighbour in self.map.neighbours_of(region_id):
            target = self.state.regions[neighbour]
            if target.order is None or target.order_house == house:
                continue
            target_terrain = self.map.regions[neighbour].terrain
            # A Raid on land can never reach a sea area. A Raid on a sea area
            # may reach land or sea. A Raid in a Port may only reach the
            # adjacent sea, and land areas may not raid a Port.
            if source_terrain is Terrain.LAND and target_terrain is not Terrain.LAND:
                continue
            if source_terrain is Terrain.PORT and target_terrain is not Terrain.SEA:
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
            if self.map.regions[target].terrain in (Terrain.SEA, Terrain.PORT):
                actions.extend(self._naval_march_actions(house, region_id, target))
            else:
                actions.extend(self._land_march_actions(house, region_id, target))
        return actions

    def _naval_march_actions(
        self, house: str, origin: str, target: str
    ) -> list[Action]:
        ships = self.state.regions[origin].units.get(UnitType.SHIP, 0)
        if ships <= 0:
            return []
        limit = ships
        if self.map.regions[target].terrain is Terrain.PORT:
            # A Port holds at most three Ships (rulebook page 25).
            occupied = self.state.regions[target].total(include_routed=False)
            limit = min(limit, max(0, 3 - occupied))
        actions: list[Action] = []
        for count in range(1, limit + 1):
            if not self._march_supply_ok(house, origin, target, count):
                continue
            actions.append(
                Action.march(house, origin, target, units=(0, 0, 0, count))
            )
        return actions

    def _land_march_actions(
        self, house: str, origin: str, target: str
    ) -> list[Action]:
        region = self.state.regions[origin]
        counts = [
            region.units.get(UnitType.FOOTMAN, 0),
            region.units.get(UnitType.KNIGHT, 0),
            region.units.get(UnitType.SIEGE, 0),
        ]
        if sum(counts) == 0:
            return []
        actions: list[Action] = []
        for combo in product(*(range(count + 1) for count in counts)):
            moving = sum(combo)
            if moving == 0:
                continue
            if not self._march_supply_ok(house, origin, target, moving):
                continue
            units = (combo[0], combo[1], combo[2], 0)
            if not self._beats_neutral(house, origin, target, units):
                # A Neutral Force may only be attacked with enough strength to
                # destroy it (rulebook page 26).
                continue
            actions.append(Action.march(house, origin, target, units=units))
            if self._can_establish_control(house, origin, moving):
                actions.append(
                    Action.march(
                        house, origin, target, units=units, establish=True
                    )
                )
        return actions

    def _neutral_attack_strength(
        self, house: str, origin: str, target: str, units: tuple[int, ...]
    ) -> int:
        """Combat Strength of a march against a Neutral Force (rulebook page 26).

        A Neutral Force may not use a House card or the Valyrian Steel Blade, but
        the attacker may still receive support, and the attacker wins ties.
        """
        static = self.map.regions[target]
        moving = {
            unit: count
            for unit, count in zip(
                (UnitType.FOOTMAN, UnitType.KNIGHT, UnitType.SIEGE, UnitType.SHIP),
                units,
                strict=True,
            )
            if count
        }
        return (
            units_strength(moving, siege_active=static.is_fortification)
            + order_modifier(self.state, origin, house, "attacker")
            + support_strength(self.state, target, house, supporting_attacker=True)
        )

    def _beats_neutral(
        self, house: str, origin: str, target: str, units: tuple[int, ...]
    ) -> bool:
        """True when this march can destroy the Neutral Force in ``target``."""
        neutral = self.state.regions[target].neutral_force
        if neutral <= 0:
            return True
        return (
            self._neutral_attack_strength(house, origin, target, units) >= neutral
        )

    def _march_supply_ok(
        self, house: str, origin: str, target: str, moving: int
    ) -> bool:
        """Would this move leave the house within its Supply limits?

        A player may never take an action that would exceed Supply, which can
        happen when a march merges two forces or splits one into several
        (rulebook page 8).
        """
        state = self.state
        limits = state.houses[house].supply_limit()
        controlled = set(state.controlled_regions(house))
        sizes: list[int] = []
        for rid in controlled:
            size = state.regions[rid].total(include_routed=True)
            if rid == origin:
                size -= moving
            if rid == target:
                size += moving
            if size >= 2:
                sizes.append(size)
        if target not in controlled and moving >= 2:
            # Taking an area with a new army of our own.
            sizes.append(moving)
        sizes.sort(reverse=True)
        return all(
            index < len(limits) and size <= limits[index]
            for index, size in enumerate(sizes)
        )

    def _can_establish_control(
        self, house: str, origin: str, moving: int
    ) -> bool:
        """Can this march leave a Power token in the area it vacates?"""
        region = self.state.regions[origin]
        if self.map.regions[origin].terrain is not Terrain.LAND:
            return False
        if region.owner == house:
            return False  # the printed Home-area shield is enough
        if region.influence_token is not None:
            return False  # only one Power token per area
        if self.state.houses[house].power <= 0:
            return False
        return region.total(include_routed=True) - moving == 0

    def _march_targets(self, house: str, origin: str) -> set[str]:
        region = self.state.regions[origin]
        origin_terrain = self.map.regions[origin].terrain
        has_land = any(unit is not UnitType.SHIP for unit in region.units)
        has_ships = region.units.get(UnitType.SHIP, 0) > 0
        targets: set[str] = set()

        if has_ships:
            for neighbour in self.map.neighbours_of(origin):
                static = self.map.regions[neighbour]
                if static.terrain is Terrain.SEA:
                    targets.add(neighbour)
                elif (
                    static.terrain is Terrain.PORT
                    and self.state.regions[neighbour].controller == house
                ):
                    targets.add(neighbour)

        if has_land and origin_terrain is Terrain.LAND:
            targets |= set(self.map.land_neighbours_of(origin))
            targets |= self._transport_targets(house, origin)

        targets.discard(origin)
        return targets

    def _transport_targets(self, house: str, origin: str) -> set[str]:
        """Land regions reachable by moving across a chain of owned ships.

        Routed Ships may still transport (rulebook page 23).
        """
        naval = {
            rid
            for rid, region in self.state.regions.items()
            if region.controller == house
            and self.map.regions[rid].is_sea
            and (
                region.units.get(UnitType.SHIP, 0)
                + region.routed.get(UnitType.SHIP, 0)
                > 0
            )
        }
        frontier = [
            rid for rid in self.map.sea_neighbours_of(origin) if rid in naval
        ]
        seen = set(frontier)
        while frontier:
            current = frontier.pop()
            for neighbour in self.map.sea_neighbours_of(current):
                if neighbour in naval and neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append(neighbour)
        targets: set[str] = set()
        for sea in seen:
            targets |= set(self.map.land_neighbours_of(sea))
        return targets

    def _apply_raid(self, action: Action) -> None:
        house = action.house or ""
        region_id = action.region or ""
        target_id = action.target or ""
        target = self.state.regions[target_id]
        removed = target.order
        owner = target.order_house
        pillaging = (
            removed is not None
            and ORDERS_BY_KEY[removed].order_type is OrderType.CONSOLIDATE_POWER
        )
        target.order = None
        target.order_house = None
        source = self.state.regions[region_id]
        source.order = None
        source.order_house = None
        if removed and owner and removed in self.state.houses[owner].orders_placed:
            self.state.houses[owner].orders_placed.remove(removed)
            self.state.houses[owner].orders_available.append(removed)
        if pillaging and owner:
            self.state.houses[house].power += 1
            victim = self.state.houses[owner]
            if victim.power > 0:
                victim.power -= 1
            self.state.note(
                f"{house} pillages {target_id}: +1 power, {owner} loses 1"
            )
        else:
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
        attacker_order_key = source.order
        source.order = None
        source.order_house = None

        moving = self._moving_units(source, target_terrain, action.units)
        moving_total = sum(moving.values())
        vacates = source.total(include_routed=True) - moving_total == 0
        for unit, count in moving.items():
            source.units[unit] = source.units.get(unit, 0) - count
            if source.units[unit] <= 0:
                del source.units[unit]

        # The decision to establish control is made before any battle begins
        # (rulebook page 24).
        if (
            action.establish
            and vacates
            and self.map.regions[origin].is_land
            and source.owner != house
        ):
            source.influence_token = house
            self.state.houses[house].power -= 1
            self.state.note(f"{house} establishes control of {origin}")

        enemy_here = target.controller not in (None, house) and (
            not target.is_empty() or target.garrison > 0
        )
        if target.neutral_force > 0:
            self._apply_neutral_attack(
                house, origin, target_id, moving, attacker_order_strength
            )
        elif enemy_here:
            defender = target.controller or ""
            # The two forces stay separate until the battle is resolved; see
            # agot.engine.combat for why merging them would be wrong.
            self.state.pending_battle = {
                "region": target_id,
                "attacker": house,
                "defender": defender,
                "origin": origin,
                "attacker_order": attacker_order_key,
                "attacker_card": None,
                "defender_card": None,
                "attacker_card_chosen": False,
                "defender_card_chosen": False,
                "attacker_units": moving,
                "defender_units": dict(target.units),
                "attacker_order_strength": attacker_order_strength,
                "defender_order_strength": order_modifier(
                    self.state, target_id, defender, "defender"
                ),
                "blade_done": False,
                "attacker_blade": False,
                "defender_blade": False,
            }
            self.state.phase = Phase.COMBAT
            self.state.note(f"{house} marches into {target_id} and attacks {defender}")
        else:
            # Marching onto a lone enemy Power token discards it; it is not a
            # battle and the Power token does not defend.
            if target.influence_token not in (None, house):
                target.influence_token = None
            for unit, count in moving.items():
                if count:
                    target.units[unit] = target.units.get(unit, 0) + count
            target.controller = house
            if self.map.regions[target_id].is_land:
                self._capture_port(house, target_id)
            self.state.note(f"{house} marches {origin} -> {target_id}")
        if source.is_empty() and not source.routed:
            source.controller = (
                source.influence_token
                if source.influence_token is not None
                else (source.owner if source.owner else None)
            )

    def _apply_neutral_attack(
        self,
        house: str,
        origin: str,
        target_id: str,
        moving: dict[UnitType, int],
        attacker_order_strength: int,
    ) -> None:
        """Resolve a march against a Neutral Force token (rulebook page 26).

        The attacker must equal or exceed the token's printed strength; there is
        no House card, no Valyrian Steel Blade and no casualty step. On success
        the token is removed and the area (with its fortification) is taken.
        """
        state = self.state
        target = state.regions[target_id]
        static = self.map.regions[target_id]
        strength = (
            units_strength(moving, siege_active=static.is_fortification)
            + attacker_order_strength
            + support_strength(state, target_id, house, supporting_attacker=True)
        )
        if strength < target.neutral_force:
            # Legal actions are filtered before this point; if an illegal march
            # does get here, leave the army where it was rather than let the
            # attack through.
            source = state.regions[origin]
            for unit, count in moving.items():
                source.units[unit] = source.units.get(unit, 0) + count
            state.note(
                f"{house} cannot destroy the Neutral Force in {target_id} "
                f"({strength} < {target.neutral_force}); march cancelled"
            )
            return
        defeated = target.neutral_force
        target.neutral_force = 0
        target.influence_token = None
        for unit, count in moving.items():
            if count:
                target.units[unit] = target.units.get(unit, 0) + count
        target.controller = house
        if static.is_land:
            self._capture_port(house, target_id)
        state.note(
            f"{house} destroys the Neutral Force ({defeated}) in {target_id} "
            f"with strength {strength}"
        )

    @staticmethod
    def _moving_units(
        source, target_terrain: Terrain, units: tuple[int, ...] | None
    ) -> dict[UnitType, int]:
        """Units that leave a marching area.

        ``units`` is a count per entry of ``(Footman, Knight, Siege, Ship)``;
        ``None`` means every eligible unit, which keeps hand-written calls and
        older recordings working.
        """
        if target_terrain in (Terrain.SEA, Terrain.PORT):
            eligible = (UnitType.SHIP,)
        else:
            eligible = (UnitType.FOOTMAN, UnitType.KNIGHT, UnitType.SIEGE)
        if units is None:
            return {
                unit: source.units[unit]
                for unit in eligible
                if source.units.get(unit, 0) > 0
            }
        moving: dict[UnitType, int] = {}
        for unit, count in zip(
            (UnitType.FOOTMAN, UnitType.KNIGHT, UnitType.SIEGE, UnitType.SHIP),
            units,
            strict=True,
        ):
            if count and unit in eligible:
                moving[unit] = count
        return moving

    def _capture_port(self, house: str, land_region_id: str) -> None:
        """Take the Port of a captured land area (rulebook page 25).

        The new land controller may replace the enemy Ships in the Port with
        an equal number of his own, limited by the Port's three-Ship capacity
        and by Supply.
        """
        port_id = self.map.port_of(land_region_id)
        if port_id is None:
            return
        port = self.state.regions[port_id]
        enemy_ships = port.units.get(UnitType.SHIP, 0) + port.routed.get(
            UnitType.SHIP, 0
        )
        port.units.pop(UnitType.SHIP, None)
        port.routed.pop(UnitType.SHIP, None)
        port.controller = house
        placed = 0
        for _ in range(enemy_ships):
            if placed >= 3 or not self._muster_fits(house, port_id, UnitType.SHIP):
                break
            port.units[UnitType.SHIP] = port.units.get(UnitType.SHIP, 0) + 1
            placed += 1
        if enemy_ships:
            self.state.note(
                f"{house} captures {port_id}, replacing {placed}/{enemy_ships} Ships"
            )

    def _consume_order(self, house: str, region_id: str) -> str | None:
        """Clear an area's order token and return it to its owner's pool."""
        region = self.state.regions[region_id]
        order_key = region.order
        region.order = None
        region.order_house = None
        if order_key is None:
            return None
        house_state = self.state.houses[house]
        if order_key in house_state.orders_placed:
            house_state.orders_placed.remove(order_key)
            house_state.orders_available.append(order_key)
        return order_key

    def _apply_consolidate(self, action: Action) -> None:
        house = action.house or ""
        region_id = action.region or ""
        static = self.map.regions[region_id]
        self._consume_order(house, region_id)
        # A Consolidate Power Order in a Sea area has no effect (page 13).
        if static.is_sea:
            self.state.note(
                f"{house}'s consolidate in {region_id} on open water: no effect"
            )
            return
        # A Consolidate Power Order in a Port is removed without effect while
        # enemy Ships occupy the connected sea area.
        if static.is_port and static.sea is not None:
            sea = self.state.regions[static.sea]
            enemy_ships = sea.units.get(UnitType.SHIP, 0) + sea.routed.get(
                UnitType.SHIP, 0
            )
            if sea.controller not in (None, house) and enemy_ships > 0:
                self.state.note(
                    f"{house}'s consolidate in {region_id} cancelled by enemy ships"
                )
                return
        gained = 1 + static.power
        self.state.houses[house].power += gained
        self.state.note(f"{house} consolidates in {region_id} for {gained} power")

    def _apply_consolidate_muster(self, action: Action) -> None:
        """Resolve a Special Consolidate Power Order as a local mustering."""
        house = action.house or ""
        region_id = action.region or ""
        static = self.map.regions[region_id]
        self._consume_order(house, region_id)
        # The area is mustered as if a "Mustering" card had been played there,
        # using its printed mustering value (rulebook page 22).
        self.state.regions[region_id].mustering_points += static.mustering
        self.state.return_phase = Phase.ACTION
        self.state.phase = Phase.MUSTERING
        self.state.note(
            f"{house} musters in {region_id} from a Special Consolidate Power Order"
        )

    def _next_action_step(self) -> None:
        self.state.action_house_last = None
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

    # -- Messenger Raven -------------------------------------------------

    def _next_raven_actor(self) -> str | None:
        if self.state.raven_used:
            return None
        holder = self.state.raven_holder()
        if holder is None or self.state.houses[holder].eliminated:
            return None
        return holder

    def _raven_actions(self, house: str) -> list[Action]:
        """The Messenger Raven may replace an order or inspect the Wildling deck.

        Only one of the two abilities may be used each round (page 11).
        """
        state = self.state
        actions: list[Action] = []
        allowance = state.star_allowance(house)
        stars = state.houses[house].stars_placed()
        for region_id in state.controlled_regions(house):
            region = state.regions[region_id]
            if region.order is None or region.order_house != house:
                continue
            old_spec = ORDERS_BY_KEY[region.order]
            for order_key in state.houses[house].orders_available:
                spec = ORDERS_BY_KEY[order_key]
                new_stars = (
                    stars
                    - (1 if old_spec.starred else 0)
                    + (1 if spec.starred else 0)
                )
                if new_stars > allowance:
                    continue
                if self._order_is_restricted(order_key):
                    continue
                if spec.order_type is OrderType.MARCH and not region.units:
                    continue
                actions.append(Action.raven_swap(house, region_id, order_key))
        actions.append(Action.raven_deck(house, to_bottom=False))
        actions.append(Action.raven_deck(house, to_bottom=True))
        actions.append(Action.pass_(house))
        return actions

    def _apply_raven_swap(self, action: Action) -> None:
        house = action.house or ""
        region_id = action.region or ""
        order_key = action.order or ""
        region = self.state.regions[region_id]
        old_key = region.order
        house_state = self.state.houses[house]
        if old_key is not None and old_key in house_state.orders_placed:
            house_state.orders_placed.remove(old_key)
            house_state.orders_available.append(old_key)
        house_state.orders_available.remove(order_key)
        house_state.orders_placed.append(order_key)
        region.order = order_key
        region.order_house = house
        self.state.raven_used = True
        self.state.note(
            f"{house} swaps {old_key} for {order_key} in {region_id} "
            "with the Messenger Raven"
        )

    def _apply_raven_deck(self, action: Action, *, to_bottom: bool) -> None:
        house = action.house or ""
        deck = self.state.wildling_deck
        if to_bottom and deck:
            card = deck.pop(0)
            deck.append(card)
        self.state.raven_used = True
        direction = "bottom" if to_bottom else "top"
        self.state.note(
            f"{house} inspects the top Wildling card and returns it to the {direction}"
        )

    # -- Combat phase ----------------------------------------------------

    def _next_combat_chooser(self) -> str | None:
        battle = self.state.pending_battle
        if battle is None:
            return None
        if not battle.get("attacker_card_chosen", False):
            return battle["attacker"]
        if not battle.get("defender_card_chosen", False):
            return battle["defender"]
        if not battle.get("blade_done", False):
            holder = self.state.blade_holder()
            if (
                holder in (battle["attacker"], battle["defender"])
                and not self.state.houses[holder].blade_used
            ):
                return holder
            battle["blade_done"] = True
        return None

    def _combat_actions(self, house: str) -> list[Action]:
        battle = self.state.pending_battle
        if battle is None:  # pragma: no cover - defensive
            return []
        if not battle.get("attacker_card_chosen", False) or not battle.get(
            "defender_card_chosen", False
        ):
            house_state = self.state.houses[house]
            if not house_state.house_cards_available:
                # Every card has been played; the house fights with no card.
                return [Action.play_house_card(house, None)]
            return [
                Action.play_house_card(house, card)
                for card in house_state.house_cards_available
            ]
        # The Valyrian Steel Blade holder decides whether to spend it.
        return [Action.use_blade(house), Action.pass_(house)]

    def _apply_use_blade(self, action: Action) -> None:
        battle = self.state.pending_battle
        if battle is None:  # pragma: no cover - defensive
            return
        house = action.house or ""
        if house == battle["attacker"]:
            battle["attacker_blade"] = True
        else:
            battle["defender_blade"] = True
        battle["blade_done"] = True
        self.state.note(f"{house} uses the Valyrian Steel Blade")

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
            battle["attacker_card_chosen"] = True
        else:
            battle["defender_card"] = card
            battle["defender_card_chosen"] = True

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

        attacker_blade = bool(battle.get("attacker_blade", False))
        defender_blade = bool(battle.get("defender_blade", False))

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
            attacker_order=battle.get("attacker_order"),
            attacker_order_strength=battle["attacker_order_strength"],
            defender_order_strength=battle["defender_order_strength"],
            attacker_blade=attacker_blade,
            defender_blade=defender_blade,
        )
        # If the attacker won, the defender's Order token was removed by the
        # battle; return it to its owner's pool. If the defender won, its Order
        # and Power tokens are unaffected, so the pools stay untouched.
        if result.winner == attacker and defender_order is not None:
            if (
                defender_order_house is not None
                and defender_order
                in self.state.houses[defender_order_house].orders_placed
            ):
                house_state = self.state.houses[defender_order_house]
                house_state.orders_placed.remove(defender_order)
                house_state.orders_available.append(defender_order)
        if attacker_blade:
            self.state.houses[attacker].blade_used = True
        if defender_blade:
            self.state.houses[defender].blade_used = True
        if result.winner == attacker and self.map.regions[region_id].is_land:
            # Capturing the area also takes its Port (rulebook page 25).
            self._capture_port(attacker, region_id)
        self.state.note(
            f"battle in {region_id}: {result.winner} defeats {result.loser} "
            f"({result.attacker.total} vs {result.defender.total})"
        )
        self._recycle_house_cards_for(attacker)
        self._recycle_house_cards_for(defender)

    def _recycle_house_cards_for(self, house: str) -> None:
        """Return six House cards after a house plays its last (seventh).

        The rulebook leaves the just-played card in the discard pile and
        returns the other six to hand.
        """
        house_state = self.state.houses[house]
        if house_state.house_cards_available:
            return
        if not house_state.house_cards_used:
            return
        last_played = house_state.house_cards_used[-1]
        house_state.house_cards_available = [
            card for card in house_state.house_cards_used if card != last_played
        ]
        house_state.house_cards_used = [last_played]

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

        state.raven_used = False
        state.action_house_last = None
        state.return_phase = None

        for region in state.regions.values():
            if region.is_empty() and not region.routed:
                if region.influence_token is not None:
                    region.controller = region.influence_token
                elif region.owner is not None:
                    region.controller = region.owner
                else:
                    region.controller = None
            # Unspent mustering points are lost, and order restrictions last
            # only for the round in which their card was drawn.
            region.mustering_points = 0
        state.order_restrictions = []

        # Victory check: immediate at seven castles, otherwise after round 10
        # with the FAQ v2.0 tiebreakers.
        self._check_victory()
        if state.phase is Phase.GAME_OVER:
            return

        if state.round >= state.max_rounds:
            best = max(
                state.tracks[Track.IRON_THRONE],
                key=self._final_ranking_key,
            )
            self._declare_winner(best, "won the end-of-game tiebreakers")
            return

        state.round += 1
        state.pending_planning = list(state.tracks[Track.IRON_THRONE])
        state.bid_tracks = []
        state.pending_bids = {}
        state.pending_westeros = None
        state.wildling_attack = None
        state.westeros_choice = None
        state.westeros_drawn = {name: None for name in state.westeros_drawn}
        state.note(f"round {state.round} begins")
        state.phase = Phase.WESTEROS

    def _final_ranking_key(self, house: str) -> tuple[int, int, int, int, int]:
        """FAQ v2.0 end-of-game tiebreaker ordering (highest wins).

        Victory-track areas, then Strongholds, then Supply, then available
        Power, then Iron Throne position (rulebook page 16).
        """
        state = self.state
        return (
            state.castle_count(house),
            state.stronghold_count(house),
            state.houses[house].supply_level,
            state.houses[house].power,
            -state.track_position(house, Track.IRON_THRONE),
        )

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
