"""The action space.

Actions are small immutable records. The action space is combinatorial (there is
no fixed enumeration of legal moves), so the neural network scores a *variable
number* of candidate actions per decision point rather than emitting a fixed
distribution. That is the same trick AlphaZero-style systems use for chess-like
games with large move spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    """Every kind of decision the engine can ask a player to make."""

    PASS = "pass"
    PLACE_ORDER = "place_order"
    RESOLVE_RAID = "resolve_raid"
    MARCH = "march"
    RESOLVE_CONSOLIDATE = "resolve_consolidate"
    PLAY_HOUSE_CARD = "play_house_card"
    RETREAT = "retreat"
    DISBAND = "disband"
    MUSTER = "muster"
    BID = "bid"


@dataclass(frozen=True, slots=True)
class Action:
    """A single decision.

    Only the fields relevant to ``kind`` are populated; the rest stay ``None``.
    Keeping one flat record (rather than a class hierarchy) makes actions cheap
    to copy and easy to encode as network input.
    """

    kind: ActionKind
    house: str | None = None
    region: str | None = None
    target: str | None = None
    order: str | None = None
    card: str | None = None
    unit: str | None = None
    amount: int | None = None

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        parts = [self.kind.value]
        for name in ("house", "region", "target", "order", "card", "unit", "amount"):
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={value}")
        return "(" + " ".join(parts) + ")"

    # -- constructors ----------------------------------------------------

    @staticmethod
    def pass_(house: str) -> Action:
        return Action(ActionKind.PASS, house=house)

    @staticmethod
    def place_order(house: str, region: str, order: str) -> Action:
        return Action(ActionKind.PLACE_ORDER, house=house, region=region, order=order)

    @staticmethod
    def resolve_raid(house: str, region: str, target: str) -> Action:
        return Action(ActionKind.RESOLVE_RAID, house=house, region=region, target=target)

    @staticmethod
    def march(house: str, region: str, target: str) -> Action:
        return Action(ActionKind.MARCH, house=house, region=region, target=target)

    @staticmethod
    def resolve_consolidate(house: str, region: str) -> Action:
        return Action(ActionKind.RESOLVE_CONSOLIDATE, house=house, region=region)

    @staticmethod
    def play_house_card(house: str, card: str | None) -> Action:
        return Action(ActionKind.PLAY_HOUSE_CARD, house=house, card=card)

    @staticmethod
    def retreat(house: str, region: str, target: str) -> Action:
        return Action(ActionKind.RETREAT, house=house, region=region, target=target)

    @staticmethod
    def disband(house: str, region: str, unit: str) -> Action:
        return Action(ActionKind.DISBAND, house=house, region=region, unit=unit)

    @staticmethod
    def muster(house: str, region: str, unit: str) -> Action:
        return Action(ActionKind.MUSTER, house=house, region=region, unit=unit)

    @staticmethod
    def bid(house: str, amount: int) -> Action:
        return Action(ActionKind.BID, house=house, amount=amount)
