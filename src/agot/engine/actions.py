"""The action space.

Actions are small immutable records. The action space is combinatorial (there is
no fixed enumeration of legal moves), so the neural network scores a *variable
number* of candidate actions per decision point rather than emitting a fixed
distribution. That is the same trick AlphaZero-style systems use for chess-like
games with large move spaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActionKind(str, Enum):
    """Every kind of decision the engine can ask a player to make."""

    PASS = "pass"
    PLACE_ORDER = "place_order"
    RESOLVE_RAID = "resolve_raid"
    MARCH = "march"
    RESOLVE_CONSOLIDATE = "resolve_consolidate"
    CONSOLIDATE_MUSTER = "consolidate_muster"
    PLAY_HOUSE_CARD = "play_house_card"
    USE_BLADE = "use_blade"
    RAVEN_SWAP = "raven_swap"
    RAVEN_DECK_TOP = "raven_deck_top"
    RAVEN_DECK_BOTTOM = "raven_deck_bottom"
    RETREAT = "retreat"
    DISBAND = "disband"
    #: The defeated player chooses which unit is destroyed for one casualty
    #: (``region`` is the embattled area, ``unit`` the unit type to remove).
    CHOOSE_CASUALTY = "choose_casualty"
    #: A player declares which side an adjacent Support order helps. ``region``
    #: is the supporting area and ``side`` is ``"attacker"``, ``"defender"`` or
    #: ``"refuse"`` (rulebook pages 17-18).
    DECLARE_SUPPORT = "declare_support"
    MUSTER = "muster"
    UPGRADE = "upgrade"
    BID = "bid"
    #: An Influence-token holder picks how a holders'-choice Westeros card
    #: resolves (``card`` is the chosen card id, or ``"none"``).
    WESTEROS_CHOICE = "westeros_choice"


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
    #: Which side a Support order helps: ``"attacker"``, ``"defender"`` or
    #: ``"refuse"``. Only meaningful for :attr:`ActionKind.DECLARE_SUPPORT`.
    side: str | None = None
    amount: int | None = None
    #: Units that move, as a count per entry of
    #: :data:`agot.engine.observation.UNIT_ORDER` (Footman, Knight, Siege,
    #: Ship). Only meaningful for :attr:`ActionKind.MARCH`; ``None`` means
    #: "every eligible unit" for backwards compatibility.
    units: tuple[int, ...] | None = None
    #: Whether a March action uses a Power token to keep control of the area
    #: it vacates.
    establish: bool = False
    #: For a card choice, which house's deck the card belongs to. It differs
    #: from :attr:`house` when one player chooses a card from the opponent's
    #: hand (Patchface, Tyrion) and is ``None`` for battle card selection where
    #: both players pick from their own deck.
    card_house: str | None = None
    #: The extension's synthetic generic-choice context tag (0..14), when this
    #: action answers a pending card/Wildling choice. The Rust action stores it
    #: in the order slot for the frozen encoder layout; Python keeps it separate
    #: so ``order`` still reads as a real order token for ``PLACE_ORDER``.
    choice_tag: int | None = None
    #: The extension's own action record when this object came from
    #: ``Game.legal_actions``. Not part of equality or the hash: it is a
    #: transport handle that lets the encoder skip rebuilding the action.
    _handle: object | None = field(default=None, compare=False, repr=False)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        parts = [self.kind.value]
        for name in (
            "house",
            "region",
            "target",
            "order",
            "card",
            "card_house",
            "choice_tag",
            "unit",
            "side",
            "amount",
            "units",
        ):
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={value}")
        if self.establish:
            parts.append("establish=True")
        return "(" + " ".join(parts) + ")"

    # -- constructors ----------------------------------------------------

    @staticmethod
    def pass_(house: str, region: str | None = None) -> Action:
        return Action(ActionKind.PASS, house=house, region=region)

    @staticmethod
    def place_order(house: str, region: str, order: str) -> Action:
        return Action(ActionKind.PLACE_ORDER, house=house, region=region, order=order)

    @staticmethod
    def resolve_raid(house: str, region: str, target: str) -> Action:
        return Action(ActionKind.RESOLVE_RAID, house=house, region=region, target=target)

    @staticmethod
    def march(
        house: str,
        region: str,
        target: str,
        units: tuple[int, ...] | None = None,
        establish: bool = False,
    ) -> Action:
        return Action(
            ActionKind.MARCH,
            house=house,
            region=region,
            target=target,
            units=units,
            establish=establish,
        )

    @staticmethod
    def resolve_consolidate(house: str, region: str) -> Action:
        return Action(ActionKind.RESOLVE_CONSOLIDATE, house=house, region=region)

    @staticmethod
    def consolidate_muster(house: str, region: str) -> Action:
        return Action(ActionKind.CONSOLIDATE_MUSTER, house=house, region=region)

    @staticmethod
    def play_house_card(house: str, card: str | None) -> Action:
        return Action(ActionKind.PLAY_HOUSE_CARD, house=house, card=card)

    @staticmethod
    def use_blade(house: str) -> Action:
        return Action(ActionKind.USE_BLADE, house=house)

    @staticmethod
    def raven_swap(house: str, region: str, order: str) -> Action:
        return Action(
            ActionKind.RAVEN_SWAP, house=house, region=region, order=order
        )

    @staticmethod
    def raven_deck(house: str, to_bottom: bool) -> Action:
        kind = ActionKind.RAVEN_DECK_BOTTOM if to_bottom else ActionKind.RAVEN_DECK_TOP
        return Action(kind, house=house)

    @staticmethod
    def retreat(house: str, region: str, target: str) -> Action:
        return Action(ActionKind.RETREAT, house=house, region=region, target=target)

    @staticmethod
    def disband(house: str, region: str, unit: str) -> Action:
        return Action(ActionKind.DISBAND, house=house, region=region, unit=unit)

    @staticmethod
    def choose_casualty(house: str, region: str, unit: str) -> Action:
        """Destroy one named unit for a battle casualty."""
        return Action(
            ActionKind.CHOOSE_CASUALTY, house=house, region=region, unit=unit
        )

    @staticmethod
    def declare_support(house: str, region: str, side: str) -> Action:
        """Declare which side a Support order in ``region`` helps."""
        return Action(
            ActionKind.DECLARE_SUPPORT, house=house, region=region, side=side
        )

    @staticmethod
    def muster(
        house: str, region: str, unit: str, target: str | None = None
    ) -> Action:
        return Action(
            ActionKind.MUSTER, house=house, region=region, unit=unit, target=target
        )

    @staticmethod
    def upgrade(house: str, region: str, unit: str) -> Action:
        return Action(ActionKind.UPGRADE, house=house, region=region, unit=unit)

    @staticmethod
    def bid(house: str, amount: int) -> Action:
        return Action(ActionKind.BID, house=house, amount=amount)

    @staticmethod
    def westeros_choice(house: str, card: str) -> Action:
        """The holder's pick for a holders'-choice Westeros card.

        ``card`` is the chosen card id to resolve, or ``"none"`` for no effect.
        """
        return Action(ActionKind.WESTEROS_CHOICE, house=house, card=card)
