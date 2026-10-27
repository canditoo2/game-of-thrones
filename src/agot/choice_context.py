"""Context for the engine's generic (card/Wildling) choices.

The action-feature layout is frozen, so the engine reuses ordinary action
kinds for card and Wildling sub-choices: a discard is a ``PLAY_HOUSE_CARD``
action, a Crow replacement is a ``CHOOSE_CASUALTY`` action, and so on. Without
the *pending choice* those actions all look alike, which used to make the GUI
print "Play House card" for a discard and let the greedy heuristic score a
forced replacement as if it were a lost unit.

This module turns the engine's pending-choice payload (``PyGame.pending_choice()``,
exposed as ``game.state.pending_choice``) and the synthetic action context tags
(``Action.order`` values ``0..14``) into one small :class:`ChoiceContext`. Both
sources are optional: when neither is available the context is inferred from
the phase, the pending battle stage and the wildling-attack card, so a stale
extension still gets useful labels.

The tag map is the cross-agent contract:

===== ====================== ============================================
Tag   Purpose                Choice
===== ====================== ============================================
0     ``queen_order``        Queen of Thorns: remove an opponent Order
1     ``cersei_order``       Cersei Lannister: remove an opponent Order
2     ``renly_footman``      Renly: upgrade a Footman to a Knight
3     ``horde_fort``         The Horde Descends: area for unit losses
4     ``horde_muster``       The Horde Descends: area to muster in
5     ``king_beyond_other``  King Beyond the Wall: choose a track (bottom)
6     ``king_beyond_highest`` King Beyond the Wall: choose a track (top)
7     ``preemptive_raid``    Preemptive Raid: option A or B
8     ``crow_replace_knight`` Crow Killers: Knight -> Footman
9     ``crow_upgrade_footman`` Crow Killers: Footman -> Knight
10    ``generic_destroy``    Wildling/Horde: destroy a unit
11    ``tyrion_use``         Tyrion Lannister: use the replacement
12    ``tyrion_card``        Tyrion Lannister: the replacement card
13    ``aeron_card``         Aeron Damphair: pay 2 Power for a card
14    ``card_discard_or_retrieve`` Patchface/Massing/Mammoth card choice
===== ====================== ============================================

Callers should treat every field as a useful hint, never as a guarantee: the
Rust schema is allowed to grow extra keys, and older extensions return nothing
at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agot.engine.actions import Action, ActionKind
from agot.engine.enums import ORDERS_BY_KEY, Phase, Track

#: Synthetic action context tags. Kept in one place so the GUI, the greedy
#: baseline and the tests describe the same choice the same way.
CHOICE_TAGS: dict[int, str] = {
    0: "queen_order",
    1: "cersei_order",
    2: "renly_footman",
    3: "horde_fort",
    4: "horde_muster",
    5: "king_beyond_other",
    6: "king_beyond_highest",
    7: "preemptive_raid",
    8: "crow_replace_knight",
    9: "crow_upgrade_footman",
    10: "generic_destroy",
    11: "tyrion_use",
    12: "tyrion_card",
    13: "aeron_card",
    14: "card_discard_or_retrieve",
}

#: Reverse map, useful to tests and to callers that build tagged actions.
CHOICE_TAG_BY_PURPOSE: dict[str, int] = {
    purpose: tag for tag, purpose in CHOICE_TAGS.items()
}

#: The action kinds that can carry a generic-choice context. ``PLACE_ORDER``
#: and ``RAVEN_SWAP`` store a real order key in ``Action.order`` and must never
#: be interpreted as a tag.
CHOICE_KINDS: frozenset[ActionKind] = frozenset(
    {
        ActionKind.PASS,
        ActionKind.WESTEROS_CHOICE,
        ActionKind.PLAY_HOUSE_CARD,
        ActionKind.CHOOSE_CASUALTY,
        ActionKind.UPGRADE,
        ActionKind.MUSTER,
        ActionKind.DISBAND,
    }
)

_KIND_BY_PURPOSE: dict[str, str] = {
    "queen_order": "region",
    "cersei_order": "region",
    "renly_footman": "region",
    "horde_fort": "region",
    "horde_muster": "region",
    "king_beyond_other": "track",
    "king_beyond_highest": "track",
    "preemptive_raid": "option",
    "preemptive_highest": "track",
    "crow_replace_knight": "unit",
    "crow_upgrade_footman": "unit",
    "generic_destroy": "unit",
    "tyrion_use": "option",
    "tyrion_card": "card",
    "aeron_card": "card",
    "robb_retreat": "option",
    "card_discard_or_retrieve": "card",
    # Dedicated paths kept for the labels the GUI already knew.
    "doran": "track",
    "wildling_tie": "option",
}

# Tokens are compared after ``lower`` and with every non-alphanumeric removed,
# so ``TyrionSwap``, ``tyrion_swap`` and ``TYRION-SWAP`` all match.
_PURPOSE_ALIASES: dict[str, str] = {
    "queenorder": "queen_order",
    "queenoftorns": "queen_order",
    "queenoftorn": "queen_order",
    "cerseiorder": "cersei_order",
    "cerseilannister": "cersei_order",
    "renlyfootman": "renly_footman",
    "renlybaratheon": "renly_footman",
    "hordefort": "horde_fort",
    "hordefortification": "horde_fort",
    "hordemuster": "horde_muster",
    "hordemusterfort": "horde_muster",
    "kingbeyondother": "king_beyond_other",
    "kingbeyondwallother": "king_beyond_other",
    "kingbeyondhighest": "king_beyond_highest",
    "kingbeyondwallhighest": "king_beyond_highest",
    "preemptiveraid": "preemptive_raid",
    "preemptive": "preemptive_raid",
    "preemptivehighest": "preemptive_highest",
    "preemptivetrack": "preemptive_highest",
    "crowknight": "crow_replace_knight",
    "crowkillersknight": "crow_replace_knight",
    "replaceknight": "crow_replace_knight",
    "crowupgrade": "crow_upgrade_footman",
    "crowkillersupgrade": "crow_upgrade_footman",
    "upgradefootman": "crow_upgrade_footman",
    "genericdestroy": "generic_destroy",
    "destroyunit": "generic_destroy",
    "destroy": "generic_destroy",
    "tyrionuse": "tyrion_use",
    "tyrionswap": "tyrion_card",
    "tyrionlannister": "tyrion_use",
    "tyrioncard": "tyrion_card",
    "tyrionreplacement": "tyrion_card",
    "aeroncard": "aeron_card",
    "aeron": "aeron_card",
    "aeronswap": "aeron_card",
    "aeronpay": "aeron_card",
    "robbretreat": "robb_retreat",
    "robbstark": "robb_retreat",
    "muster": "horde_muster",
    "patcheface": "card_discard_or_retrieve",
    "patchface": "card_discard_or_retrieve",
    "patchefacediscard": "card_discard_or_retrieve",
    "massingdiscard": "card_discard_or_retrieve",
    "massingatmilkwater": "card_discard_or_retrieve",
    "massing": "card_discard_or_retrieve",
    "mammothretrieve": "card_discard_or_retrieve",
    "mammothriders": "card_discard_or_retrieve",
    "mammoth": "card_discard_or_retrieve",
    "carddiscardorretrieve": "card_discard_or_retrieve",
    "discardorretrieve": "card_discard_or_retrieve",
    "doranmartell": "doran",
    "doran": "doran",
    "wildlingtie": "wildling_tie",
    "tiebreak": "wildling_tie",
}

#: Wildling card that introduces each kind of task. Used only when the engine
#: does not expose a pending-choice payload.
_WILDLING_PURPOSES: dict[str, str] = {
    "crow_killers": "crow_replace_knight",
    "the_horde_descends": "horde_fort",
    "king_beyond_the_wall": "king_beyond_other",
    "mammoth_riders": "card_discard_or_retrieve",
    "massing_milkwater": "card_discard_or_retrieve",
    "preemptive_raid": "preemptive_raid",
}

_TRACK_ORDER: tuple[Track, ...] = (
    Track.IRON_THRONE,
    Track.FIEFDOMS,
    Track.KINGS_COURT,
)

#: King Beyond the Wall only offers the two non-throne tracks.
_KING_BEYOND_TRACKS: tuple[Track, ...] = (
    Track.FIEFDOMS,
    Track.KINGS_COURT,
)

_WILDLING_NAMES: dict[str, str] = {
    "crow_killers": "Crow Killers",
    "the_horde_descends": "The Horde Descends",
    "king_beyond_the_wall": "King Beyond the Wall",
    "mammoth_riders": "Mammoth Riders",
    "massing_milkwater": "Massing at Milkwater",
    "preemptive_raid": "Preemptive Raid",
    "rattleshirt": "Rattleshirt's Raiders",
    "silence": "Silence at the Wall",
    "skinchanger": "The Skinchanger",
}


def _normalise_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "".join(character for character in text.lower() if character.isalnum())


def _purpose_from_token(value: Any) -> str | None:
    token = _normalise_token(value)
    if not token:
        return None
    direct = _PURPOSE_ALIASES.get(token)
    if direct is not None:
        return direct
    # A scalar can be a wrapper such as "ChoiceKind::CardChoice::AeronSwap" or
    # "massing_milkwater_discard". Match the longest known alias it contains.
    matches = [
        (len(alias), purpose)
        for alias, purpose in _PURPOSE_ALIASES.items()
        if alias in token
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], item[1]))
    return matches[0][1]


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
        tail = ""
        for character in reversed(text):
            if not character.isdigit():
                break
            tail = character + tail
        if tail and not any(character.isalpha() for character in text[: -len(tail)]):
            return int(tail)
    return None


def _tag_from_order_key(value: str) -> int | None:
    """Recover a tag from an order key used as the tag's numeric slot.

    The extension's PyAction exposes ``choice_tag`` directly, so this is a
    fallback for a binding that only mirrors the Rust order slot. Real order
    actions never reach here: ``PLACE_ORDER``/``RAVEN_SWAP`` are not choice
    kinds.
    """
    keys = list(ORDERS_BY_KEY)
    if value not in keys:
        return None
    index = keys.index(value)
    return index if index in CHOICE_TAGS else None


def parse_choice_tag(value: Any) -> int | None:
    """Read a synthetic context tag from an ``Action.order`` value.

    Accepts an integer, a numeric string (``"7"``), and prefixed spellings such
    as ``"choice_7"`` or ``"tag:7"``. Returns ``None`` for a real order key
    (``"march_0_a"``), which contains letters before its digits.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in CHOICE_TAGS else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            tag = int(text)
            return tag if tag in CHOICE_TAGS else None
        lowered = text.lower()
        for prefix in ("choice", "tag", "context", "choice_tag", "choice-tag"):
            if lowered.startswith(prefix):
                tail = text[len(prefix) :].strip(" _:-")
                if tail.isdigit():
                    tag = int(tail)
                    return tag if tag in CHOICE_TAGS else None
        # A named purpose is accepted too, which keeps the helper independent
        # of the exact representation chosen by the Rust side.
        purpose = _purpose_from_token(text)
        if purpose is not None:
            return CHOICE_TAG_BY_PURPOSE.get(purpose)
    return None


def _walk(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    """Flatten a nested mapping/list into ``(key_path, scalar)`` entries."""
    entries: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            entries.extend(_walk(item, (*path, str(key))))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            entries.extend(_walk(item, (*path, str(index))))
    else:
        entries.append((path, value))
    return entries


def _entry_key(path: tuple[str, ...]) -> str:
    """The nearest named key, skipping list indices in the path."""
    for part in reversed(path):
        if part.isdigit():
            continue
        return _normalise_token(part)
    return ""


def _scalar_for(entries: list[tuple[tuple[str, ...], Any]], names: set[str]) -> Any:
    """First scalar whose nearest named key matches one of ``names``."""
    for path, value in entries:
        if path and _entry_key(path) in names:
            return value
    return None


def _scalars_for(entries: list[tuple[tuple[str, ...], Any]], names: set[str]) -> list[Any]:
    return [
        value
        for path, value in entries
        if path and _entry_key(path) in names
    ]


def _explicit_purpose(entries: list[tuple[tuple[str, ...], Any]]) -> str | None:
    for path, value in entries:
        if not path:
            continue
        key = _entry_key(path)
        if key in {"purpose", "choice", "task", "variant", "flag", "kind"}:
            purpose = _purpose_from_token(value)
            if purpose is not None:
                return purpose
    # Some payloads encode the variant as a dict key and leave the value
    # nested, e.g. {"kind": {"TyrionSwap": {"target": "lannister"}}}.
    for path, _ in entries:
        for part in path:
            purpose = _purpose_from_token(part)
            if purpose is not None:
                return purpose
    return None


def _kind_from_entries(entries: list[tuple[tuple[str, ...], Any]]) -> str | None:
    for path, value in entries:
        if not path:
            continue
        if _entry_key(path) != "kind":
            continue
        token = _normalise_token(value)
        for kind in ("card", "region", "track", "option", "unit", "muster"):
            if token == kind or token.endswith(kind):
                return kind
    for path, _ in entries:
        for part in path:
            token = _normalise_token(part)
            for kind in ("card", "region", "track", "option", "unit", "muster"):
                if token == kind or token.endswith(kind):
                    return kind
    return None


def _maybe_house(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    from agot.engine.map import load_houses  # noqa: PLC0415 - lazy, avoids a cycle

    return value if value in load_houses().houses else None


def _track(value: Any) -> Track | None:
    if isinstance(value, Track):
        return value
    if isinstance(value, int) and 0 <= value < len(_TRACK_ORDER):
        return _TRACK_ORDER[value]
    if isinstance(value, str):
        try:
            return Track(value)
        except ValueError:
            purpose = _normalise_token(value)
            if purpose in {"ironthrone", "it"}:
                return Track.IRON_THRONE
            if purpose in {"fiefdoms", "fiefdom"}:
                return Track.FIEFDOMS
            if purpose in {"kingscourt", "kc"}:
                return Track.KINGS_COURT
    return None


@dataclass(frozen=True)
class ChoiceContext:
    """One pending generic choice, normalised for labels and heuristics."""

    #: ``"card"``, ``"region"``, ``"track"``, ``"option"``, ``"unit"``,
    #: ``"muster"`` or ``None`` when the payload does not say.
    kind: str | None = None
    #: One of the purposes in :data:`CHOICE_TAGS`, plus the extra purposes the
    #: engine reports for the dedicated/follow-up paths (``preemptive_highest``,
    #: ``robb_retreat``, ``doran`` and ``wildling_tie``).
    purpose: str | None = None
    tag: int | None = None
    #: Where the context came from: ``"pending"``, ``"tag"`` or ``"inferred"``.
    source: str = "pending"
    house: str | None = None
    #: The opponent/target house of a card or Order task.
    target: str | None = None
    cards: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    tracks: tuple[Track, ...] = ()
    options: tuple[int, ...] = ()
    remaining: int | None = None
    count: int | None = None
    points: int | None = None
    allow_none: bool = False
    #: The engine's own card-task name (``patchface_discard``, ``massing_discard``,
    #: ``mammoth_retrieve``, ...), which separates card tasks that share the
    #: generic ``card_discard_or_retrieve`` purpose.
    detail: str | None = None
    #: The Wildling card that created the task, when known.
    wildling_card: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def is_card(self) -> bool:
        return self.kind == "card" or self.purpose in {
            "tyrion_use",
            "tyrion_card",
            "aeron_card",
            "card_discard_or_retrieve",
        }


def _as_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                return item
    return None


def _pending_payload(game: Any) -> dict[str, Any] | None:
    """The engine's pending-choice dict, across the exposed spellings."""
    missing = object()
    state = getattr(game, "state", None)
    if state is not None:
        value = getattr(state, "pending_choice", missing)
        if value is not missing:
            if callable(value):
                try:
                    value = value()
                except Exception:  # pragma: no cover - defensive against stale bindings
                    value = None
            return _as_payload(value)
    value = getattr(game, "pending_choice", missing)
    if value is not missing:
        if callable(value):
            try:
                value = value()
            except Exception:  # pragma: no cover - defensive
                value = None
        return _as_payload(value)
    # A dedicated method on some bindings' game wrapper.
    getter = getattr(game, "pending_choice_dict", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # pragma: no cover - defensive
            value = None
        return _as_payload(value)
    # Last resort: some bindings attach the payload to the snapshot.
    snapshot = getattr(game, "snapshot", None)
    if callable(snapshot):
        try:
            data = snapshot()
        except Exception:  # pragma: no cover - defensive
            return None
        if isinstance(data, dict):
            payload = _as_payload(data.get("pending_choice"))
            if payload is not None:
                return payload
    inner = getattr(game, "_inner", None)
    getter = getattr(inner, "pending_choice", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # pragma: no cover - defensive
            value = None
        return _as_payload(value)
    return None


def _wildling_attack(game: Any) -> dict[str, Any] | None:
    state = getattr(game, "state", None)
    value = getattr(state, "wildling_attack", None)
    return value if isinstance(value, dict) else None


def _pending_battle(game: Any) -> dict[str, Any] | None:
    state = getattr(game, "state", None)
    value = getattr(state, "pending_battle", None)
    return value if isinstance(value, dict) else None


def _phase(game: Any) -> Phase | None:
    state = getattr(game, "state", None)
    value = getattr(state, "phase", None)
    if isinstance(value, Phase):
        return value
    if isinstance(value, str):
        try:
            return Phase(value)
        except ValueError:
            return None
    return None


def _infer_purpose(
    game: Any, action: Action, kind: str | None
) -> tuple[str | None, str | None]:
    """Infer a purpose from the phase when neither engine source is available."""
    phase = _phase(game)
    card = None
    attack = _wildling_attack(game)
    if attack:
        raw_card = attack.get("card")
        if isinstance(raw_card, str):
            card = raw_card
    if (
        phase is Phase.WILDLING_ATTACK
        and attack
        and attack.get("tie_candidates")
    ):
        return "wildling_tie", kind or "option"
    if card in _WILDLING_PURPOSES:
        base = _WILDLING_PURPOSES[card]
        if card == "crow_killers":
            if action.kind is ActionKind.UPGRADE:
                return "crow_upgrade_footman", kind or "unit"
            if action.kind is ActionKind.CHOOSE_CASUALTY:
                return "crow_replace_knight", kind or "unit"
            return None, kind
        if card == "the_horde_descends":
            if action.kind in (
                ActionKind.MUSTER,
                ActionKind.UPGRADE,
                ActionKind.PASS,
            ):
                return "horde_muster", kind or "region"
            if action.kind is ActionKind.WESTEROS_CHOICE:
                return "horde_fort", kind or "region"
            if action.kind is ActionKind.CHOOSE_CASUALTY:
                return "generic_destroy", kind or "unit"
            return None, kind
        if card == "king_beyond_the_wall" and action.kind is ActionKind.WESTEROS_CHOICE:
            return "king_beyond_other", kind or "track"
        if card == "preemptive_raid" and action.kind is ActionKind.WESTEROS_CHOICE:
            return "preemptive_raid", kind or "option"
        if card in ("massing_milkwater", "mammoth_riders"):
            return "card_discard_or_retrieve", kind or "card"
        return base, kind
    if phase is Phase.WILDLING_ATTACK:
        if action.kind is ActionKind.CHOOSE_CASUALTY:
            return "generic_destroy", kind or "unit"
        if action.kind is ActionKind.MUSTER:
            return "horde_muster", kind or "region"
    battle = _pending_battle(game)
    if battle and battle.get("stage") == "Doran":
        return "doran", kind or "track"
    return None, kind


def _payload_context(
    game: Any, payload: dict[str, Any], action: Action | None = None
) -> ChoiceContext:
    entries = _walk(payload)

    purpose = _explicit_purpose(entries)
    kind = _kind_from_entries(entries)

    explicit_house = _scalar_for(entries, {"house", "chooser", "actor"})
    house = _maybe_house(explicit_house)
    if house is None:
        for _path, value in entries:
            candidate = _maybe_house(value)
            if candidate is not None:
                house = candidate
                break

    target = _maybe_house(
        _scalar_for(entries, {"target", "opponent", "victim", "other"})
    )
    if target is None:
        for value in _scalars_for(entries, {"house", "houses"}):
            candidate = _maybe_house(value)
            if candidate is not None and candidate != house:
                target = candidate
                break
    if target is None:
        for _path, value in entries:
            candidate = _maybe_house(value)
            if candidate is not None and candidate != house:
                target = candidate
                break

    cards = tuple(
        str(value)
        for value in _scalars_for(entries, {"card", "cards", "hand"})
        if isinstance(value, str) and value
    )
    regions = tuple(
        str(value)
        for value in _scalars_for(entries, {"region", "regions", "area", "areas"})
        if isinstance(value, str) and value
    )
    tracks: list[Track] = []
    for value in _scalars_for(entries, {"track", "tracks"}):
        track = _track(value)
        if track is not None and track not in tracks:
            tracks.append(track)
    options = tuple(
        value
        for value in (
            _as_int(item) for item in _scalars_for(entries, {"option", "options"})
        )
        if value is not None
    )

    remaining = _as_int(_scalar_for(entries, {"remaining"}))
    count = _as_int(_scalar_for(entries, {"count"}))
    points = _as_int(_scalar_for(entries, {"points"}))
    detail_value = _scalar_for(entries, {"detail"})
    detail = str(detail_value) if isinstance(detail_value, str) else None
    allow_none = bool(_scalar_for(entries, {"allownone"}))
    wildling_card = _wildling_attack(game) or {}

    if purpose is None:
        inferred_action = action or Action(ActionKind.PASS)
        purpose, kind = _infer_purpose(game, inferred_action, kind)
    if purpose == "wildling_tie" and not cards:
        cards = tuple(
            str(value)
            for value in _scalars_for(entries, {"tiecandidates"})
            if isinstance(value, str) and value
        )
    if kind is None and purpose is not None:
        kind = _KIND_BY_PURPOSE.get(purpose)
    return ChoiceContext(
        kind=kind,
        purpose=purpose,
        house=house,
        target=target,
        cards=cards,
        regions=regions,
        tracks=tuple(tracks),
        options=options,
        remaining=remaining,
        count=count,
        points=points,
        allow_none=allow_none,
        detail=detail,
        wildling_card=wildling_card.get("card"),
        raw=payload,
    )


def resolve_choice_context(
    game: Any,
    action: Action,
    *,
    pending: dict[str, Any] | None = None,
) -> ChoiceContext | None:
    """Describe the generic choice ``action`` answers, or ``None``.

    ``pending`` lets a caller that already fetched the payload (the GUI builds
    one list of actions) skip repeated extension calls.
    """
    if action.kind not in CHOICE_KINDS:
        return None
    payload = _pending_payload(game) if pending is None else pending
    if payload:
        return _payload_context(game, payload, action)

    tag: int | None = None
    for name in ("choice_tag", "choice", "order"):
        value = getattr(action, name, None)
        if value is None:
            continue
        tag = parse_choice_tag(value)
        if tag is None and name == "order" and isinstance(value, str):
            tag = _tag_from_order_key(value)
        if tag is not None:
            break
    if tag is not None:
        purpose = CHOICE_TAGS[tag]
        context = ChoiceContext(
            kind=_KIND_BY_PURPOSE.get(purpose),
            purpose=purpose,
            tag=tag,
            source="tag",
        )
    else:
        purpose, kind = _infer_purpose(game, action, None)
        if purpose is None:
            return None
        context = ChoiceContext(
            kind=kind,
            purpose=purpose,
            source="inferred",
            wildling_card=(_wildling_attack(game) or {}).get("card"),
        )
    return context


# -- label helpers -------------------------------------------------------


def _pretty(value: str | None) -> str:
    return value.replace("_", " ").title() if value else "?"


def _unit_label(value: Any) -> str:
    if value is None:
        return "unit"
    if hasattr(value, "value") and isinstance(value.value, str):
        return _pretty(value.value)
    return _pretty(str(value))


_TRACK_LABELS = {
    Track.IRON_THRONE: "Iron Throne",
    Track.FIEFDOMS: "Fiefdoms",
    Track.KINGS_COURT: "King's Court",
}


def choice_track(context: ChoiceContext, action: Action) -> Track | None:
    """The track an action selects, using the task's own track order."""
    tracks = context.tracks
    if not tracks and context.purpose == "king_beyond_other":
        tracks = _KING_BEYOND_TRACKS
    elif not tracks and context.purpose == "king_beyond_highest":
        tracks = _TRACK_ORDER
    amount = action.amount if action.amount is not None else -1
    if 0 <= amount < len(tracks):
        return tracks[amount]
    if tracks:
        return tracks[0]
    if 0 <= amount < len(_TRACK_ORDER):
        return _TRACK_ORDER[amount]
    for name in ("order", "target", "region"):
        value = getattr(action, name, None)
        track = _track(value) if isinstance(value, str) else None
        if track is not None:
            return track
    return None


def track_label(track: Track | None) -> str:
    return _TRACK_LABELS.get(track, "?") if track is not None else "?"


def card_label(card: str | None) -> str:
    return _pretty(card) if card else "no card"


def wildling_card_name(card: str | None) -> str:
    if not card:
        return "Wildling card"
    return _WILDLING_NAMES.get(card, _pretty(card))


def _wildling_prefix(context: ChoiceContext) -> str:
    detail = context.detail or ""
    if "massing" in detail:
        return "Massing at Milkwater"
    if "mammoth" in detail:
        return "Mammoth Riders"
    if "patchface" in detail:
        return "Patchface"
    if context.wildling_card:
        return wildling_card_name(context.wildling_card)
    purpose = context.purpose
    if purpose == "card_discard_or_retrieve":
        if context.target and context.target != context.house:
            return "Patchface"
        return "House-card choice"
    if purpose in ("tyrion_use", "tyrion_card"):
        return "Tyrion Lannister"
    if purpose == "aeron_card":
        return "Aeron Damphair"
    return "Wildling card"


def _optional_suffix(context: ChoiceContext) -> str:
    return " (or decline)" if context.allow_none else ""


def describe_choice(context: ChoiceContext) -> str:
    """One sentence describing what the pending choice is asking for."""
    purpose = context.purpose
    if purpose == "queen_order":
        return "Queen of Thorns: choose an opponent's Order token to remove."
    if purpose == "cersei_order":
        return "Cersei Lannister: choose an opponent's Order token to remove."
    if purpose == "renly_footman":
        return "Renly Baratheon: choose a Footman to upgrade to a Knight."
    if purpose == "horde_fort":
        return (
            "The Horde Descends: choose the Castle/Stronghold where your units "
            "are destroyed."
        )
    if purpose == "horde_muster":
        return "The Horde Descends: choose the Castle/Stronghold to muster in."
    if purpose == "king_beyond_other":
        return "King Beyond the Wall: choose the Influence track to move to the bottom."
    if purpose == "king_beyond_highest":
        return "King Beyond the Wall: choose the Influence track to move to the top."
    if purpose == "preemptive_raid":
        return (
            "Preemptive Raid: A) destroy 2 of your units anywhere, or "
            "B) drop 2 positions on your highest Influence track."
        )
    if purpose == "preemptive_highest":
        return (
            "Preemptive Raid: choose which of your tied highest Influence "
            "tracks drops 2 positions."
        )
    if purpose == "crow_replace_knight":
        return "Crow Killers: choose which Knights to replace with Footmen."
    if purpose == "crow_upgrade_footman":
        return "Crow Killers: choose which Footmen to upgrade to Knights."
    if purpose == "generic_destroy":
        return f"{_wildling_prefix(context)}: choose which unit to destroy."
    if purpose == "tyrion_use":
        target = _pretty(context.target) if context.target else "the opponent"
        return (
            f"Tyrion Lannister: choose whether {target} must replace their "
            "House card."
        )
    if purpose == "tyrion_card":
        target = _pretty(context.target) if context.target else "the opponent"
        return f"Tyrion Lannister: {target} chooses a replacement House card."
    if purpose == "aeron_card":
        return (
            "Aeron Damphair: choose whether to pay 2 Power for a different "
            "House card."
        )
    if purpose == "robb_retreat":
        return (
            "Robb Stark: choose whether to take over the defeated defender's "
            "retreat."
        )
    if purpose == "card_discard_or_retrieve":
        detail = context.detail or ""
        if "massing" in detail or context.wildling_card == "massing_milkwater":
            return "Massing at Milkwater: choose a House card to discard."
        if "mammoth" in detail or context.wildling_card == "mammoth_riders":
            return "Mammoth Riders: choose a discarded House card to retrieve."
        if "patchface" in detail or (
            context.target and context.target != context.house
        ):
            target = _pretty(context.target) if context.target else "the opponent"
            return f"Patchface: choose a House card to discard from {target}'s hand."
        return "Card choice: choose a House card to discard or retrieve."
    if purpose == "doran":
        return "Doran Martell: choose an Influence track to send the opponent to."
    if purpose == "wildling_tie":
        return "Wildling bid tie: choose which tied bidder the card applies to."
    if context.kind == "card":
        return "House-card choice: choose a card."
    if context.kind == "region":
        return "Card effect: choose an area."
    if context.kind == "track":
        return "Card effect: choose an Influence track."
    if context.kind == "unit":
        return "Card effect: choose a unit."
    if context.kind == "muster":
        return "Wildling reward: choose what to muster."
    if context.kind == "option":
        return "Card effect: choose between the printed options."
    return "A card or Wildling choice is pending."


def describe_action_choice(
    context: ChoiceContext,
    action: Action,
    *,
    region_name: str | None = None,
    track: Track | None = None,
    mode: str | None = None,
) -> str | None:
    """A specific label for one action of a pending choice.

    ``region_name`` is supplied by the caller so this module does not need the
    game's map. ``mode`` lets the caller pass the discard/retrieve/replace
    distinction it derived from the house card piles. Returns ``None`` when the
    context cannot name the action, which lets the caller fall back to its older
    kind-based labels.
    """
    purpose = context.purpose
    target = _pretty(context.target) if context.target else ""
    region = region_name or "?"
    card = card_label(action.card)
    routed = " routed" if action.amount == 1 else ""
    prefix = _wildling_prefix(context)

    if action.kind is ActionKind.PASS:
        return _decline_choice(context)

    if purpose == "queen_order":
        return f"Queen of Thorns: remove {target or 'an opponent'}'s Order in {region}"
    if purpose == "cersei_order":
        return (
            f"Cersei Lannister: remove {target or 'an opponent'}'s Order in {region}"
        )
    if purpose == "renly_footman":
        return f"Renly Baratheon: upgrade a Footman to a Knight in {region}"
    if purpose == "horde_fort":
        count = context.count or context.remaining or 1
        return f"The Horde Descends: destroy up to {count} unit(s) in {region}"
    if purpose == "horde_muster":
        if action.kind is ActionKind.MUSTER and action.unit:
            return (
                f"The Horde Descends: muster one {_unit_label(action.unit)} "
                f"in {region}"
            )
        if action.kind is ActionKind.UPGRADE and action.unit:
            return (
                "The Horde Descends: upgrade a Footman to a "
                f"{_unit_label(action.unit)} in {region}"
            )
        return f"The Horde Descends: muster in {region}"
    if purpose == "king_beyond_other":
        return f"King Beyond the Wall: move to the bottom of {track_label(track)}"
    if purpose == "king_beyond_highest":
        return f"King Beyond the Wall: move to the top of {track_label(track)}"
    if purpose == "preemptive_raid":
        option = action.amount if action.amount is not None else -1
        if option == 0:
            return "Preemptive Raid: A) destroy 2 of your units anywhere"
        if option == 1:
            return (
                "Preemptive Raid: B) drop 2 positions on your highest "
                "Influence track"
            )
        return "Preemptive Raid: choose option A or B"
    if purpose == "preemptive_highest":
        return (
            "Preemptive Raid: drop 2 positions on "
            f"{track_label(track)} (tied highest track)"
        )
    if purpose == "robb_retreat":
        option = action.amount if action.amount is not None else 0
        if option == 0:
            return "Robb Stark: decline to choose the defender's retreat"
        return "Robb Stark: choose the defender's retreat yourself"
    if purpose == "crow_replace_knight":
        return f"Crow Killers: replace one{routed} Knight with a Footman in {region}"
    if purpose == "crow_upgrade_footman":
        return f"Crow Killers: upgrade one{routed} Footman to a Knight in {region}"
    if purpose == "generic_destroy":
        unit = _unit_label(action.unit) if action.unit else "unit"
        return f"{prefix}: destroy one{routed} {unit} in {region}"
    if purpose == "tyrion_use":
        option = action.amount if action.amount is not None else 0
        target_text = f"{target}'s" if target else "the opponent's"
        if option == 0:
            return "Tyrion Lannister: do not use the ability"
        return (
            f"Tyrion Lannister: return {target_text} House card and make them "
            "choose a replacement"
        )
    if purpose == "tyrion_card":
        if action.card and action.card != "none":
            if target:
                return (
                    f"Tyrion Lannister: {target} replaces their House card with "
                    f"{card}"
                )
            return (
                "Tyrion Lannister: the opponent replaces their House card with "
                f"{card}"
            )
        return "Tyrion Lannister: the opponent plays no replacement House card"
    if purpose == "aeron_card":
        return f"Aeron Damphair: pay 2 Power and play {card}"
    if purpose == "card_discard_or_retrieve":
        resolved = mode or card_choice_mode(context, action)
        if resolved == "retrieve":
            return f"{prefix}: retrieve {card} from the discard pile"
        if resolved == "discard":
            if prefix == "Patchface" and target:
                return f"Patchface: discard {card} from {target}'s hand"
            return f"{prefix}: discard {card}"
        return f"{prefix}: choose {card}"
    if purpose == "doran":
        return _doran_label(action, context)
    if purpose == "wildling_tie":
        return _tie_label(context, action)
    if context.kind == "region":
        return f"Choose {region} for this card's effect"
    if context.kind == "track":
        return f"Choose {track_label(track)}"
    return None


def _decline_choice(context: ChoiceContext) -> str:
    purpose = context.purpose
    if purpose == "queen_order":
        return "Decline to remove an Order with the Queen of Thorns"
    if purpose == "cersei_order":
        return "Decline to remove an Order with Cersei Lannister"
    if purpose == "renly_footman":
        return "Decline to upgrade a Footman with Renly Baratheon"
    if purpose == "horde_muster":
        return "Decline to muster with The Horde Descends"
    if purpose == "crow_replace_knight":
        return "Decline to replace a Knight with a Footman"
    if purpose == "crow_upgrade_footman":
        return "Decline to upgrade a Footman to a Knight"
    if purpose == "generic_destroy":
        return f"Decline to destroy a unit ({_wildling_prefix(context)})"
    if purpose in ("tyrion_use", "tyrion_card"):
        if purpose == "tyrion_card":
            return "Tyrion Lannister: play no replacement House card"
        return "Decline to use Tyrion Lannister"
    if purpose == "aeron_card":
        return "Decline to pay 2 Power (keep Aeron Damphair)"
    if purpose == "card_discard_or_retrieve":
        detail = context.detail or ""
        if "massing" in detail or context.wildling_card == "massing_milkwater":
            return "Decline to discard a House card (Massing at Milkwater)"
        if "mammoth" in detail or context.wildling_card == "mammoth_riders":
            return "Decline to retrieve a House card (Mammoth Riders)"
        if "patchface" in detail:
            return "Decline to discard a House card (Patchface)"
        return "Decline this House-card choice"
    if purpose == "doran":
        return "Decline Doran Martell's track choice"
    if purpose == "wildling_tie":
        return "Decline to break the Wildling bid tie"
    if context.kind == "card":
        return "Decline this House-card choice"
    if context.kind == "region":
        return "Decline this area choice"
    return "Decline this choice"


def _doran_label(action: Action, context: ChoiceContext) -> str:
    tracks = context.tracks or _TRACK_ORDER
    index = action.amount if action.amount is not None else 0
    track = tracks[index] if 0 <= index < len(tracks) else tracks[0]
    return f"Doran Martell: move the opponent to the bottom of {track_label(track)}"


def _tie_label(context: ChoiceContext, action: Action) -> str:
    candidates = context.cards  # overloaded for tie candidates by the payload
    index = action.amount if action.amount is not None else 0
    if 0 <= index < len(candidates):
        return f"Break the Wildling bid tie: choose {_pretty(candidates[index])}"
    target = _pretty(context.target) if context.target else "a tied bidder"
    return f"Break the Wildling bid tie: choose {target}"


def card_choice_mode(
    context: ChoiceContext, action: Action, game: Any | None = None
) -> str | None:
    """Whether a card action discards, retrieves, replaces or plays a card."""
    detail = context.detail or ""
    if "massing" in detail:
        return "discard"
    if "mammoth" in detail:
        return "retrieve"
    if "patchface" in detail:
        return "discard"
    if "tyrion" in detail:
        return "replace"
    if "aeron" in detail:
        return "aeron"
    if context.wildling_card == "massing_milkwater":
        return "discard"
    if context.wildling_card == "mammoth_riders":
        return "retrieve"
    purpose = context.purpose
    if purpose == "tyrion_card":
        return "replace"
    if purpose == "aeron_card":
        return "aeron"
    if purpose == "generic_destroy":
        return None
    house = getattr(action, "card_house", None) or context.house or action.house
    card = action.card
    if game is not None and house and card and card != "none":
        state = getattr(game, "state", None)
        houses = getattr(state, "houses", {}) if state is not None else {}
        record = houses.get(house) if hasattr(houses, "get") else None
        if record is not None:
            used = getattr(record, "house_cards_used", ()) or ()
            available = getattr(record, "house_cards_available", ()) or ()
            if card in used:
                return "retrieve"
            if card in available:
                return "discard"
    # Patchface discards from the opponent's hand.
    if purpose == "card_discard_or_retrieve" and context.target:
        if context.target != (context.house or action.house):
            return "discard"
    return None


__all__ = [
    "CHOICE_KINDS",
    "CHOICE_TAGS",
    "CHOICE_TAG_BY_PURPOSE",
    "ChoiceContext",
    "card_choice_mode",
    "card_label",
    "choice_track",
    "describe_action_choice",
    "describe_choice",
    "parse_choice_tag",
    "resolve_choice_context",
    "track_label",
    "wildling_card_name",
]
