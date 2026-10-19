"""Loader and validator for the static board data.

The map lives in ``agot/data/*.json`` so that it can be corrected against the
physical board without touching engine code. :func:`validate` enforces the
structural invariants that the engine relies on, and the test-suite runs it on
every commit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

from agot.engine.enums import Terrain

DATA_PACKAGE = "agot.data"


class MapDataError(ValueError):
    """Raised when the bundled board data is structurally inconsistent."""


def load_json(name: str) -> dict[str, Any]:
    """Read a JSON document bundled in :mod:`agot.data`."""
    with resources.files(DATA_PACKAGE).joinpath(name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class Region:
    """A single space on the board."""

    id: str
    name: str
    terrain: Terrain
    power: int = 0
    supply: int = 0
    mustering: int = 0
    home: str | None = None
    host: str | None = None
    sea: str | None = None

    @property
    def is_land(self) -> bool:
        return self.terrain is Terrain.LAND

    @property
    def is_sea(self) -> bool:
        return self.terrain is Terrain.SEA

    @property
    def is_port(self) -> bool:
        return self.terrain is Terrain.PORT

    @property
    def is_castle(self) -> bool:
        """True for a castle (1 mustering point in the reference data)."""
        return self.mustering == 1

    @property
    def is_stronghold(self) -> bool:
        """True for a stronghold (2 mustering points in the reference data)."""
        return self.mustering == 2

    @property
    def is_fortification(self) -> bool:
        """Castles and strongholds both count towards the victory condition."""
        return self.mustering > 0


class GameMap:
    """Immutable board topology for the base 6-player game."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.name: str = raw["name"]
        self.max_rounds: int = raw.get("max_rounds", 10)
        self.victory_castles: int = raw.get("victory_castles", 7)

        self.regions: dict[str, Region] = {}
        for entry in raw["regions"]:
            region = Region(
                id=entry["id"],
                name=entry["name"],
                terrain=Terrain(entry["terrain"]),
                power=entry.get("power", 0),
                supply=entry.get("supply", 0),
                mustering=entry.get("mustering", 0),
                home=entry.get("home"),
                host=entry.get("host"),
                sea=entry.get("sea"),
            )
            if region.id in self.regions:
                raise MapDataError(f"duplicate region id: {region.id}")
            self.regions[region.id] = region

        neighbours: dict[str, set[str]] = {rid: set() for rid in self.regions}
        for a, b in raw["adjacency"]:
            neighbours[a].add(b)
            neighbours[b].add(a)
        self.neighbours: dict[str, frozenset[str]] = {
            rid: frozenset(n) for rid, n in neighbours.items()
        }

        self.land_region_ids: tuple[str, ...] = tuple(
            rid for rid, r in self.regions.items() if r.is_land
        )
        self.sea_region_ids: tuple[str, ...] = tuple(
            rid for rid, r in self.regions.items() if r.is_sea
        )
        self.port_region_ids: tuple[str, ...] = tuple(
            rid for rid, r in self.regions.items() if r.is_port
        )
        self.port_hosts: dict[str, str] = {
            rid: r.host for rid, r in self.regions.items() if r.is_port and r.host
        }
        #: Regions that start with a Neutral Force token, keyed by strength.
        #: Six-player games use King's Landing (5) and The Eyrie (6).
        self.neutral_forces: dict[str, int] = {
            rid: int(strength)
            for rid, strength in (raw.get("neutral_forces") or {}).items()
        }

        self.validate()

    # -- queries ---------------------------------------------------------

    def region(self, region_id: str) -> Region:
        try:
            return self.regions[region_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown region: {region_id}") from exc

    def neighbours_of(self, region_id: str) -> tuple[str, ...]:
        """All regions orthogonally adjacent to ``region_id``, in a stable order.

        Sorted on purpose. A ``frozenset``'s iteration order depends on the
        interpreter's string-hash seed, so iterating the raw set made the *order*
        of the legal actions a Raid offers (and of the greedy baseline's
        tie-breaks) change from process to process. That is invisible to a single
        run but breaks reproducibility of recorded games, self-play targets and
        the Rust port's differential tests.
        """
        return tuple(sorted(self.neighbours[region_id]))

    def land_neighbours_of(self, region_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(n for n in self.neighbours[region_id] if self.regions[n].is_land)
        )

    def sea_neighbours_of(self, region_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(n for n in self.neighbours[region_id] if self.regions[n].is_sea)
        )

    def port_of(self, land_region_id: str) -> str | None:
        """The port that belongs to a coastal land region, if any."""
        for port_id, host in self.port_hosts.items():
            if host == land_region_id:
                return port_id
        return None

    def controlled_castle_count(self, region_ids) -> int:
        """Number of castle/stronghold regions in ``region_ids``."""
        return sum(1 for rid in region_ids if self.regions[rid].is_fortification)

    # -- validation ------------------------------------------------------

    def validate(self) -> None:
        """Check the invariants the engine depends on.

        Raises :class:`MapDataError` on the first problem found.
        """
        for rid, region in self.regions.items():
            if region.host and region.host not in self.regions:
                raise MapDataError(f"{rid}: unknown port host {region.host!r}")
            if region.sea and region.sea not in self.regions:
                raise MapDataError(f"{rid}: unknown port sea {region.sea!r}")

        for rid, neighbours in self.neighbours.items():
            if rid in neighbours:
                raise MapDataError(f"{rid} is adjacent to itself")
            for other in neighbours:
                if other not in self.regions:
                    raise MapDataError(f"{rid}: unknown neighbour {other!r}")
                if rid not in self.neighbours[other]:
                    raise MapDataError(f"adjacency {rid}-{other} is not symmetric")
            if not neighbours:
                raise MapDataError(f"{rid} has no neighbours")

        for port_id, host in self.port_hosts.items():
            port = self.regions[port_id]
            if host not in self.neighbours[port_id]:
                raise MapDataError(f"{port_id} is not adjacent to its host {host!r}")
            if port.sea not in self.neighbours[port_id]:
                raise MapDataError(f"{port_id} is not adjacent to its sea {port.sea!r}")

        for region in self.regions.values():
            if region.is_land and region.mustering and not region.is_fortification:
                raise MapDataError(f"{region.id}: inconsistent mustering value")

        homes = [r.home for r in self.regions.values() if r.home]
        if len(homes) != len(set(homes)):
            raise MapDataError("two regions share the same home house")

        for rid, strength in self.neutral_forces.items():
            if rid not in self.regions:
                raise MapDataError(f"neutral force on unknown region {rid!r}")
            if not self.regions[rid].is_land:
                raise MapDataError(f"neutral force on non-land region {rid!r}")
            if strength <= 0:
                raise MapDataError(f"neutral force {rid!r} has non-positive strength")
            if self.regions[rid].home:
                raise MapDataError(f"neutral force {rid!r} is a home area")

        if not self.sea_region_ids:
            raise MapDataError("no sea regions defined")
        if not self.land_region_ids:
            raise MapDataError("no land regions defined")


@lru_cache(maxsize=1)
def load_map(name: str = "map_base_6p.json") -> GameMap:
    """Load and cache the bundled map."""
    return GameMap(load_json(name))


class HouseData:
    """Static per-house data: home area, start units and starting tracks."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.player_count: int = raw["player_count"]
        self.houses: dict[str, dict[str, Any]] = {
            house["id"]: house for house in raw["houses"]
        }
        self.initial_tracks: dict[str, list[str]] = raw["initial_tracks"]
        self.order: tuple[str, ...] = tuple(self.houses)

    def starting_units(self, house_id: str) -> dict[str, dict[str, int]]:
        return self.houses[house_id]["starting_units"]

    def home(self, house_id: str) -> str:
        return self.houses[house_id]["home"]


@lru_cache(maxsize=1)
def load_houses() -> HouseData:
    return HouseData(load_json("houses.json"))


class HouseCardData:
    """The seven house cards each house brings to a battle."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.cards: dict[str, list[dict[str, Any]]] = {
            house: payload["cards"] for house, payload in raw["houses"].items()
        }

    def deck(self, house_id: str) -> list[dict[str, Any]]:
        return list(self.cards[house_id])


@lru_cache(maxsize=1)
def load_house_cards() -> HouseCardData:
    return HouseCardData(load_json("house_cards.json"))


class WesterosData:
    """The three Westeros decks and their effects."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.cards: dict[str, dict[str, Any]] = raw["cards"]
        self.composition: dict[str, list[str]] = raw["composition"]

    def deck_names(self) -> tuple[str, ...]:
        return tuple(self.composition)


@lru_cache(maxsize=1)
def load_westeros() -> WesterosData:
    return WesterosData(load_json("westeros.json"))


class WildlingData:
    """The Wildling deck and track configuration."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.track_max: int = raw["track_max"]
        self.start_threat: int = raw["start_threat"]
        self.cards: dict[str, dict[str, Any]] = raw["cards"]
        self.deck: list[str] = raw["deck"]


@lru_cache(maxsize=1)
def load_wildlings() -> WildlingData:
    return WildlingData(load_json("wildlings.json"))
