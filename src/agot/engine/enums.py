"""Enumerations and small immutable value types shared by the engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Terrain(str, Enum):
    """What kind of space a region is."""

    LAND = "land"
    SEA = "sea"
    PORT = "port"


class UnitType(str, Enum):
    """The four unit kinds."""

    FOOTMAN = "footman"
    KNIGHT = "knight"
    SIEGE = "siege"
    SHIP = "ship"


class OrderType(str, Enum):
    """The five order kinds."""

    MARCH = "march"
    DEFEND = "defend"
    SUPPORT = "support"
    RAID = "raid"
    CONSOLIDATE_POWER = "consolidate_power"


class Phase(str, Enum):
    """High-level round phases, in resolution order."""

    WESTEROS = "westeros"
    #: The holder of an Influence token chooses how a "holders' choice" Westeros
    #: card (A Throne of Blades, Dark Wings Dark Words, Put to the Sword) resolves.
    WESTEROS_CHOICE = "westeros_choice"
    BIDDING = "bidding"
    #: Secretly bidding Power to fight off a Wildling attack (rulebook page 22).
    WILDLING_ATTACK = "wildling_attack"
    MUSTERING = "mustering"
    PLANNING = "planning"
    #: The Messenger Raven step, between assigning and revealing orders.
    RAVEN = "raven"
    ACTION = "action"
    COMBAT = "combat"
    CLEANUP = "cleanup"
    GAME_OVER = "game_over"


class ActionStep(str, Enum):
    """Sub-steps of the action phase, in resolution order."""

    RAID = "raid"
    MARCH = "march"
    CONSOLIDATE_POWER = "consolidate_power"


#: The phases in canonical order. Used to build one-hot encodings.
PHASES: tuple[Phase, ...] = tuple(Phase)


class Track(str, Enum):
    """The three influence tracks."""

    IRON_THRONE = "iron_throne"
    FIEFDOMS = "fiefdoms"
    KINGS_COURT = "kings_court"


@dataclass(frozen=True)
class OrderSpec:
    """A single order token type.

    ``strength`` is the combat modifier for march/defend orders, or the
    additional support strength for a starred support order. ``starred``
    tokens are limited by a house's position on the King's Court track.
    """

    key: str
    order_type: OrderType
    strength: int
    starred: bool


#: The fifteen order tokens every house owns.
ORDER_SPECS: tuple[OrderSpec, ...] = (
    OrderSpec("march", OrderType.MARCH, 0, False),
    OrderSpec("march_minus", OrderType.MARCH, -1, False),
    OrderSpec("march_plus", OrderType.MARCH, 1, True),
    OrderSpec("defend_1a", OrderType.DEFEND, 1, False),
    OrderSpec("defend_1b", OrderType.DEFEND, 1, False),
    OrderSpec("defend_2", OrderType.DEFEND, 2, True),
    OrderSpec("support_a", OrderType.SUPPORT, 0, False),
    OrderSpec("support_b", OrderType.SUPPORT, 0, False),
    OrderSpec("support_plus", OrderType.SUPPORT, 1, True),
    OrderSpec("raid_a", OrderType.RAID, 0, False),
    OrderSpec("raid_b", OrderType.RAID, 0, False),
    OrderSpec("raid_star", OrderType.RAID, 0, True),
    OrderSpec("consolidate_a", OrderType.CONSOLIDATE_POWER, 0, False),
    OrderSpec("consolidate_b", OrderType.CONSOLIDATE_POWER, 0, False),
    OrderSpec("consolidate_star", OrderType.CONSOLIDATE_POWER, 0, True),
)

ORDERS_BY_KEY: dict[str, OrderSpec] = {spec.key: spec for spec in ORDER_SPECS}

#: How many starred orders a house may place, by its position (1-based) on the
#: King's Court track. Verified against chibimagic's ``KingsCourtTrack``
#: ``FIVE_SIX_PLAYER_SPECIAL_ORDERS = [3, 3, 2, 1, 0, 0]``.
STAR_ALLOWANCE_BY_COURT_POSITION: tuple[int, ...] = (3, 3, 2, 1, 0, 0)

#: Maximum army sizes a house may support, indexed by supply level. Each row
#: lists the largest army it may field, then the next largest, and so on.
#: Cross-verified against two independent community references.
SUPPLY_LIMITS: tuple[tuple[int, ...], ...] = (
    (2, 2),
    (3, 2),
    (3, 2, 2),
    (3, 2, 2, 2),
    (3, 3, 2, 2),
    (4, 3, 2, 2),
    (4, 3, 2, 2, 2),
)
