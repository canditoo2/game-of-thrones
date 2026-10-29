"""The Python-shaped view over the Rust engine.

This module drives the extension's session tier and presents the objects the
rest of the package expects: ``Action`` dataclasses, a ``state`` whose regions
and houses have attribute access, and the static ``map``. Everything from the
CLI to ``GreedyPolicy`` reads a position as ``game.state.regions[rid].controller``
and friends, so the view is what keeps those call sites unchanged.

The view reads through the engine's own ``snapshot()``. It caches the snapshot
and its derived regions/houses/tracks per step and invalidates them in ``step``,
which is safe because the wrapper is the only session mutation path. Building a
position goes through the extension's named builder (``set_region`` /
``set_house`` / ``set_track``), used by the engine tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agot.engine.actions import Action, ActionKind
from agot.engine.enums import ActionStep, OrderType, Phase, Track, UnitType
from agot.engine.map import load_map

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agot.engine.game import GameConfig

MODULE = "agot._core"


def core():
    """The extension module, or ``None`` when it is not installed."""
    try:
        from agot import _core  # noqa: PLC0415 - deliberately lazy
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    return _core


def _units(raw: dict[str, int]) -> dict[UnitType, int]:
    """The snapshot's name-keyed counts as the enum-keyed mapping Python uses."""
    return {
        UnitType(name): count
        for name, count in raw.items()
        if count
    }


@dataclass
class RegionView:
    """One region, with the fields the rest of the package reads."""

    region_id: str
    controller: str | None = None
    owner: str | None = None
    units: dict[UnitType, int] = field(default_factory=dict)
    routed: dict[UnitType, int] = field(default_factory=dict)
    neutral_force: int = 0
    garrison: int = 0
    mustering_points: int = 0
    influence_token: str | None = None
    order: str | None = None
    order_house: str | None = None

    def total(self, *, include_routed: bool = True) -> int:
        total = sum(self.units.values())
        if include_routed:
            total += sum(self.routed.values())
        return total

    def is_empty(self) -> bool:
        return self.total(include_routed=False) == 0

    def contains_house(self, house_id: str) -> bool:
        return self.controller == house_id and not self.is_empty()

    @property
    def order_type(self) -> OrderType | None:
        if self.order is None:
            return None
        from agot.engine.enums import ORDERS_BY_KEY

        return ORDERS_BY_KEY[self.order].order_type


@dataclass
class HouseView:
    """One house, with the fields the rest of the package reads."""

    house_id: str
    power: int = 0
    supply_level: int = 0
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
        from agot.engine.enums import SUPPLY_LIMITS

        return SUPPLY_LIMITS[max(0, min(self.supply_level, len(SUPPLY_LIMITS) - 1))]


class StateView:
    """A position, read from the Rust engine's ``snapshot()``.

    The snapshot is cached on the game and invalidated by every ``step``. The
    cache lives here because the view used to rebuild the whole JSON snapshot on
    *every attribute read*, which made a greedy game ~800x slower than the
    retired Python engine. Invalidation is centralised in ``RustGame.step``, and
    a clone/determinization gets a fresh view, so the cache cannot outlive the
    position it describes.
    """

    def __init__(self, game: RustGame) -> None:
        self._game = game
        self._version = -1
        self._regions: dict[str, RegionView] | None = None
        self._houses: dict[str, HouseView] | None = None
        self._tracks: dict[Track, list[str]] | None = None

    def _data(self) -> dict:
        return self._game.snapshot()

    def _sync(self) -> None:
        """Drop memoised sub-views when the underlying position changed."""
        if self._version != self._game._version:
            self._version = self._game._version
            self._regions = None
            self._houses = None
            self._tracks = None

    @property
    def round(self) -> int:
        return self._data()["round"]

    @property
    def phase(self) -> Phase:
        return Phase(self._data()["phase"])

    @property
    def action_step(self) -> ActionStep:
        return ActionStep(self._data()["action_step"])

    @property
    def active_house(self) -> str | None:
        return self._data()["active_house"]

    @property
    def action_house_last(self) -> str | None:
        return self._data()["action_house_last"]

    @property
    def raven_used(self) -> bool:
        return self._data()["raven_used"]

    @property
    def return_phase(self) -> Phase | None:
        raw = self._data()["return_phase"]
        return Phase(raw) if raw is not None else None

    @property
    def order_restrictions(self) -> list[str]:
        return list(self._data()["order_restrictions"])

    @property
    def bid_tracks(self) -> list[Track]:
        return [Track(name) for name in self._data()["bid_tracks"]]

    @property
    def pending_bids(self) -> dict[str, int]:
        return dict(self._data()["pending_bids"])

    @property
    def pending_planning(self) -> list[str]:
        return list(self._data()["pending_planning"])

    @property
    def pending_westeros(self) -> list[str] | None:
        raw = self._data()["pending_westeros"]
        return None if raw is None else list(raw)

    @property
    def pending_march(self) -> dict | None:
        return self._data()["pending_march"]

    @property
    def pending_battle(self) -> dict | None:
        return self._game._inner.pending_battle()

    @property
    def pending_choice(self) -> dict | None:
        """The generic (card/Wildling) choice the engine is waiting for.

        Older extensions do not expose it; the helper in
        :mod:`agot.choice_context` then falls back to action context tags and
        phase inference.
        """
        getter = getattr(self._game._inner, "pending_choice", None)
        if getter is None:
            return None
        value = getter()
        return dict(value) if isinstance(value, dict) else None

    @property
    def westeros_waiting(self) -> bool:
        return bool(self._game._inner.westeros_waiting())

    @property
    def westeros_choice(self) -> dict | None:
        return self._data()["westeros_choice"]

    @property
    def wildling_threat(self) -> int:
        return self._data()["wildling_threat"]

    @property
    def wildling_attack(self) -> dict | None:
        return self._game._inner.wildling_attack()

    def raven_holder(self) -> str | None:
        order = self.tracks[Track.KINGS_COURT]
        return order[0] if order else None

    def blade_holder(self) -> str | None:
        order = self.tracks[Track.FIEFDOMS]
        return order[0] if order else None

    @property
    def winner(self) -> str | None:
        return self._data()["winner"]

    @winner.setter
    def winner(self, house: str | None) -> None:
        self._game._inner.set_winner(house)

    @property
    def fortifications(self) -> frozenset[str]:
        return frozenset(self._data()["fortifications"])

    @property
    def max_rounds(self) -> int:
        return self._game._inner.max_rounds

    @property
    def victory_castles(self) -> int:
        return self._game._inner.victory_castles

    @property
    def orders_revealed(self) -> bool:
        # Orders are assigned face down and revealed before the Messenger Raven
        # acts, so only the assignment step hides them.
        return self.phase is not Phase.PLANNING

    @property
    def log(self) -> list[str]:
        # The Rust log is off during search; a session that was asked for one
        # (the CLI's demo) reads it here.
        return list(self._game._inner.log())

    @property
    def regions(self) -> dict[str, RegionView]:
        self._sync()
        if self._regions is None:
            raw = self._data()["regions"]
            # The snapshot arrives as a JSON object, whose keys are sorted; the
            # view must iterate regions in *map* order, and call sites depend on
            # it (`controlled_regions`, the planning order, the encoder's own
            # order). Rebuilding in map order keeps it faithful.
            map_order = [
                region_id for region_id in self._game.map.regions if region_id in raw
            ]
            self._regions = {
                region_id: RegionView(
                    region_id=region_id,
                    controller=raw[region_id]["controller"],
                    owner=raw[region_id]["owner"],
                    units=_units(raw[region_id]["units"]),
                    routed=_units(raw[region_id]["routed"]),
                    neutral_force=raw[region_id]["neutral_force"],
                    garrison=raw[region_id]["garrison"],
                    mustering_points=raw[region_id]["mustering_points"],
                    influence_token=raw[region_id]["influence_token"],
                    order=raw[region_id]["order"],
                    order_house=raw[region_id]["order_house"],
                )
                for region_id in map_order
            }
        return self._regions

    @property
    def houses(self) -> dict[str, HouseView]:
        self._sync()
        if self._houses is None:
            raw = self._data()["houses"]
            self._houses = {
                house_id: HouseView(
                    house_id=house_id,
                    power=entry["power"],
                    supply_level=entry["supply_level"],
                    orders_available=list(entry["orders_available"]),
                    orders_placed=list(entry["orders_placed"]),
                    house_cards_available=list(entry["cards_available"]),
                    house_cards_used=list(entry["cards_used"]),
                    blade_used=entry["blade_used"],
                    passed=entry["passed"],
                    eliminated=entry["eliminated"],
                )
                for house_id, entry in raw.items()
            }
        return self._houses

    @property
    def tracks(self) -> dict[Track, list[str]]:
        self._sync()
        if self._tracks is None:
            raw = self._data()["tracks"]
            self._tracks = {Track(name): list(order) for name, order in raw.items()}
        return self._tracks

    def track_position(self, house_id: str, track: Track) -> int:
        order = self.tracks[track]
        return order.index(house_id) + 1 if house_id in order else 0

    def controlled_regions(self, house_id: str) -> list[str]:
        return [
            region_id
            for region_id, region in self.regions.items()
            if region.controller == house_id
        ]

    def castle_count(self, house_id: str) -> int:
        fortifications = set(self._data()["fortifications"])
        return sum(
            1
            for region_id in fortifications
            if self.regions[region_id].controller == house_id
        )

    def star_allowance(self, house_id: str) -> int:
        from agot.engine.enums import STAR_ALLOWANCE_BY_COURT_POSITION

        position = self.track_position(house_id, Track.KINGS_COURT)
        if position == 0:
            return 0
        index = min(position - 1, len(STAR_ALLOWANCE_BY_COURT_POSITION) - 1)
        return STAR_ALLOWANCE_BY_COURT_POSITION[index]

    def within_supply(self, house_id: str) -> bool:
        """Whether the house's armies fit its Supply track position.

        A lone unit is not an army and stays exempt; routed units still count.
        """
        house = self.houses.get(house_id)
        if house is None:
            return True
        sizes = sorted(
            (
                self.regions[region_id].total(include_routed=True)
                for region_id in self.controlled_regions(house_id)
            ),
            reverse=True,
        )
        sizes = [size for size in sizes if size >= 2]
        limits = house.supply_limit()
        return all(
            index < len(limits) and size <= limits[index]
            for index, size in enumerate(sizes)
        )


class RustGame:
    """The Rust engine behind the package's Python-shaped interface.

    Only the session surface is wrapped: enough for the CLI, the demo and a
    policy that reads the board. Building positions and running self-play go
    through the extension's own APIs (`set_region`, `selfplay`).
    """

    def __init__(self, inner, config: GameConfig | None = None) -> None:
        from agot.engine.game import GameConfig  # noqa: PLC0415 - avoids a cycle

        self._inner = inner
        self._config = config or GameConfig()
        self.map = load_map()
        #: Bumped by every step; `StateView` memoises against it.
        self._version = 0
        self._snapshot: dict | None = None
        self._state_view = StateView(self)

    @classmethod
    def new(cls, config: GameConfig | None = None) -> RustGame:
        from agot.engine.game import GameConfig  # noqa: PLC0415 - avoids a cycle

        config = config or GameConfig()
        extension = core()
        if extension is None:  # pragma: no cover - depends on the environment
            raise RuntimeError(f"{MODULE} is not installed")
        inner = extension.Game.new(
            houses=list(config.houses),
            seed=config.seed or 0,
            max_rounds=config.max_rounds,
            victory_castles=config.victory_castles,
            use_westeros_decks=config.use_westeros_decks,
            deferred_battles=config.deferred_battles,
            sequential_westeros=config.sequential_westeros,
            doran_choice=config.doran_choice,
            wildling_tie_choice=config.wildling_tie_choice,
            bid_tie_choice=config.bid_tie_choice,
            choice_choices=config.choice_choices,
        )
        return cls(inner, config)

    @property
    def config(self) -> GameConfig:
        return self._config

    @property
    def state(self) -> StateView:
        return self._state_view

    def snapshot(self) -> dict:
        if self._snapshot is None:
            self._snapshot = self._inner.snapshot()
        return self._snapshot

    def clone(self) -> RustGame:
        return RustGame(self._inner.clone(), self._config)

    def determinize(self, observer: str, seed: int) -> RustGame:
        """A copy with hidden orders and bids resampled, run in Rust.

        ``hidden.determinize`` delegates here; the seed is drawn from the
        caller's own generator so a seeded PIMC run stays reproducible.
        """
        return RustGame(self._inner.determinize(observer, seed), self._config)

    def is_terminal(self) -> bool:
        return self._inner.is_terminal()

    def current_player(self) -> str | None:
        return self._inner.current_player()

    def winner(self) -> str | None:
        return self._inner.winner()

    def legal_actions(self) -> list[Action]:
        """The engine's actions, as the Python ``Action`` dataclass.

        Converting rather than wrapping keeps `action in game.legal_actions()`,
        `str(action)` and the policy code working unchanged.
        """
        return [
            Action(
                kind=ActionKind(action.kind),
                house=action.house,
                region=action.region,
                target=action.target,
                order=action.order,
                card=action.card,
                unit=action.unit,
                side=action.side,
                amount=action.amount,
                units=action.units,
                establish=action.establish,
                card_house=getattr(action, "card_house", None),
                choice_tag=getattr(action, "choice_tag", None),
                _handle=action,
            )
            for action in self._inner.legal_actions()
        ]

    def continue_westeros(self) -> None:
        """Resolve the next reveal/resolution step of a desktop Westeros phase."""
        self._inner.continue_westeros()
        self._version += 1
        self._snapshot = None

    def choose_doran_track(self, track: str) -> None:
        """Apply Doran Martell's Influence-track choice."""
        self._inner.choose_doran_track(track)
        self._version += 1
        self._snapshot = None

    def pending_bid_tie(self) -> dict | None:
        """The Clash of Kings tie the Iron Throne holder must order, if any."""
        return self._inner.pending_bid_tie()

    def pending_wildling_tie(self) -> dict | None:
        """The Wildling bid tie the Iron Throne holder must break, if any."""
        return self._inner.pending_wildling_tie()

    def peek_wildling_card(self) -> dict | None:
        """The top Wildling card, for the Messenger Raven's look."""
        return self._inner.peek_wildling_card()

    def step(self, action: Action) -> None:
        """Apply a Python ``Action`` by matching it against the engine's own.

        Matching on the canonical fields rather than rebuilding a Rust action
        keeps this honest: an action the engine did not offer cannot be played.
        """
        for candidate in self._inner.legal_actions():
            candidate_card_house = getattr(candidate, "card_house", None)
            action_card_house = getattr(action, "card_house", None)
            candidate_choice_tag = getattr(candidate, "choice_tag", None)
            action_choice_tag = getattr(action, "choice_tag", None)
            order_matches = candidate.order == action.order or (
                candidate_choice_tag is not None
                and candidate_choice_tag == action_choice_tag
            )
            if (
                candidate.kind == action.kind.value
                and candidate.house == action.house
                and candidate.region == action.region
                and candidate.target == action.target
                and order_matches
                and candidate.card == action.card
                and (action_card_house is None or candidate_card_house == action_card_house)
                and (action_choice_tag is None or candidate_choice_tag == action_choice_tag)
                and candidate.unit == action.unit
                and candidate.side == action.side
                and candidate.amount == action.amount
                and (None if candidate.units is None else list(candidate.units))
                == (None if action.units is None else list(action.units))
                and candidate.establish == action.establish
            ):
                self._inner.step(candidate)
                self._version += 1
                self._snapshot = None
                return
        raise ValueError(f"{action} is not a legal action in this position")

    def observe(self):
        """The network input, as the ``Observation`` dataclass the trainer uses."""
        from agot.engine.observation import Observation

        actor, state, matrix = self._inner.observe()
        return Observation(
            actor=actor or "",
            actions=tuple(self.legal_actions()),
            state_features=state,
            action_features=matrix,
        )
