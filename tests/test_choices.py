"""Tests for generic (card/Wildling) choice labels, scoring and config copies.

Most tests use the pure-Python context helper with a lightweight fake game or
assert the spec/config plumbing; the sequential self-play regression uses the
real Rust-backed game.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agot.agents.greedy_agent import GreedyPolicy
from agot.choice_context import (
    CHOICE_TAGS,
    parse_choice_tag,
    resolve_choice_context,
)
from agot.engine.actions import Action, ActionKind
from agot.engine.enums import Phase, Track, UnitType
from agot.engine.game import GameConfig
from agot.engine.map import load_map
from agot.engine.rust import RustGame
from agot.play import ai_action_summary, format_action
from agot.training import evaluate, rust_engine
from agot.training.parallel import SelfPlayPlan, configs_for
from agot.training.rust_engine import spec_for
from agot.training.selfplay import play_game

HOUSES = ("baratheon", "lannister", "stark", "greyjoy", "tyrell", "martell")


def _region(**fields):
    defaults = {
        "controller": None,
        "owner": None,
        "units": {},
        "routed": {},
        "neutral_force": 0,
        "garrison": 0,
        "mustering_points": 0,
        "influence_token": None,
        "order": None,
        "order_house": None,
    }
    defaults.update(fields)
    return SimpleNamespace(**defaults)


class FakeState:
    def __init__(
        self,
        *,
        phase: Phase = Phase.ACTION,
        pending_choice: dict | None = None,
        pending_battle: dict | None = None,
        wildling_attack: dict | None = None,
        regions: dict | None = None,
        houses: dict | None = None,
        tracks: dict | None = None,
    ) -> None:
        map_data = load_map()
        self.phase = phase
        self.pending_choice = pending_choice
        self.pending_battle = pending_battle
        self.wildling_attack = wildling_attack
        self.order_restrictions = []
        self.pending_march = None
        self.pending_westeros = None
        self.westeros_choice = None
        self.wildling_threat = 0
        self.round = 1
        self.victory_castles = 7
        self.regions = {region_id: _region() for region_id in map_data.regions}
        if regions:
            for region_id, fields in regions.items():
                self.regions[region_id] = _region(**fields)
        self.houses = {
            house: SimpleNamespace(
                power=5,
                supply_level=2,
                house_cards_available=[],
                house_cards_used=[],
                orders_available=[],
                blade_used=False,
                passed=False,
                eliminated=False,
            )
            for house in HOUSES
        }
        if houses:
            for house, fields in houses.items():
                for key, value in fields.items():
                    setattr(self.houses[house], key, value)
        self.tracks = {
            track: list(HOUSES) for track in (Track.IRON_THRONE, Track.FIEFDOMS)
        }
        self.tracks[Track.KINGS_COURT] = list(HOUSES)
        if tracks:
            self.tracks.update(tracks)

    def controlled_regions(self, house: str) -> list[str]:
        return [
            region_id
            for region_id, region in self.regions.items()
            if region.controller == house
        ]

    def track_position(self, house: str, track: Track) -> int:
        order = self.tracks.get(track, [])
        return order.index(house) + 1 if house in order else 0


class FakeGame:
    def __init__(self, state: FakeState | None = None) -> None:
        self.state = state or FakeState()
        self.map = load_map()


def _score(game: FakeGame, policy: GreedyPolicy, action: Action) -> float:
    return policy._score(game, action)


# -- configuration plumbing ------------------------------------------------


def test_game_config_defaults_include_generic_choices() -> None:
    config = GameConfig()

    assert config.choice_choices is True
    assert config.doran_choice is True
    assert config.wildling_tie_choice is True
    assert config.sequential_westeros is True


def test_sequential_westeros_reference_self_play_finishes() -> None:
    """A sequential reveal must not truncate the game at the first pause."""
    config = GameConfig(seed=9, max_rounds=4, sequential_westeros=True)
    policies = {
        house: GreedyPolicy(seed=index)
        for index, house in enumerate(config.houses)
    }

    record = play_game(policies, config, record=False)

    assert record.rounds > 2, "the game must continue past the first reveal"
    assert record.winner in config.houses


def test_selfplay_spec_carries_every_choice_flag() -> None:
    base = GameConfig(
        seed=3,
        deferred_battles=True,
        sequential_westeros=True,
        choice_choices=True,
    )

    spec = spec_for(SelfPlayPlan(games=2, seed=10), base)

    assert spec["choice_choices"] is True
    assert spec["deferred_battles"] is True
    assert spec["sequential_westeros"] is True


def test_parallel_plan_configs_preserve_every_choice_flag() -> None:
    base = GameConfig(
        deferred_battles=True,
        sequential_westeros=True,
        choice_choices=True,
    )

    configs = configs_for(SelfPlayPlan(games=3, seed=1), base)

    assert len(configs) == 3
    for config in configs:
        assert config.choice_choices is True
        assert config.deferred_battles is True
        assert config.sequential_westeros is True


def test_evaluation_config_copies_every_choice_flag() -> None:
    base = GameConfig(
        deferred_battles=True,
        sequential_westeros=True,
        choice_choices=True,
    )

    config = evaluate._game_config(base, seed=7)

    assert config.choice_choices is True
    assert config.deferred_battles is True
    assert config.sequential_westeros is True


def test_batched_selfplay_spec_carries_choice_flags(monkeypatch) -> None:
    captured: dict = {}

    class FakeExtension:
        def selfplay(self, spec, evaluator, decider=None):
            captured["spec"] = spec
            return (
                np.zeros((0, 61), dtype=np.float32),
                np.zeros((0, 61), dtype=np.float32),
                np.array([0]),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 6), dtype=np.float32),
                ["stark"],
                [1],
                np.array([0]),
                [1],
            )

    class DummyEvaluator:
        model = object()

    monkeypatch.setattr(rust_engine, "core", lambda: FakeExtension())
    monkeypatch.setattr(rust_engine, "leaf_evaluator", lambda model: None)
    config = GameConfig(
        seed=1,
        max_rounds=2,
        deferred_battles=True,
        sequential_westeros=True,
        choice_choices=True,
    )

    records = rust_engine.try_batched_self_play(
        DummyEvaluator(), [config], simulations=1
    )

    assert records is not None
    spec = captured["spec"]
    assert spec["choice_choices"] is True
    assert spec["deferred_battles"] is True
    assert spec["sequential_westeros"] is True


def test_batched_selfplay_declines_mixed_choice_configs(monkeypatch) -> None:
    class DummyEvaluator:
        model = object()

    monkeypatch.setattr(rust_engine, "core", lambda: object())
    configs = [
        GameConfig(seed=1, sequential_westeros=True),
        GameConfig(seed=2, sequential_westeros=False),
    ]

    assert rust_engine.try_batched_self_play(DummyEvaluator(), configs, simulations=1) is None


def test_evaluate_native_spec_carries_choice_flags(monkeypatch) -> None:
    captured: dict = {}

    class FakeExtension:
        def selfplay(self, spec, evaluator, decider=None, extra=None):
            captured["spec"] = spec
            return (None, None, None, None, None, None, ["stark"], [1])

    monkeypatch.setattr(rust_engine, "core", lambda: FakeExtension())
    monkeypatch.setattr(rust_engine, "leaf_evaluator", lambda model: None)
    config = GameConfig(
        seed=1,
        max_rounds=2,
        deferred_battles=True,
        sequential_westeros=True,
        choice_choices=True,
    )

    evaluate._play_model_games(
        [object()],
        [[0, 0, 0, 0, 0, 0]],
        [{}],
        simulations=1,
        determinizations=1,
        games=1,
        config=config,
        seed=1,
        step_limit=10,
        batch_games=1,
    )

    spec = captured["spec"]
    assert spec["choice_choices"] is True
    assert spec["deferred_battles"] is True
    assert spec["sequential_westeros"] is True


def test_rust_game_step_matches_amount_and_card_house() -> None:
    class FakeInner:
        def __init__(self, candidates):
            self.candidates = candidates
            self.stepped = []

        def legal_actions(self):
            return list(self.candidates)

        def step(self, action):
            self.stepped.append(action)

    casualty = SimpleNamespace(
        kind="choose_casualty",
        house="stark",
        region="winterfell",
        target=None,
        order=None,
        card=None,
        card_house=None,
        unit="knight",
        side=None,
        amount=1,
        units=None,
        establish=False,
    )
    inner = FakeInner([casualty])
    game = RustGame(inner)
    action = Action(
        ActionKind.CHOOSE_CASUALTY,
        house="stark",
        region="winterfell",
        unit="knight",
        amount=1,
    )

    game.step(action)

    assert inner.stepped == [casualty]
    with pytest.raises(ValueError):
        game.step(
            Action(
                ActionKind.CHOOSE_CASUALTY,
                house="stark",
                region="winterfell",
                unit="knight",
                amount=0,
            )
        )

    card = SimpleNamespace(
        kind="play_house_card",
        house="baratheon",
        region=None,
        target=None,
        order="defend_0_a",
        card="gregor_clegane",
        card_house="lannister",
        choice_tag=14,
        unit=None,
        side=None,
        amount=None,
        units=None,
        establish=False,
    )
    card_inner = FakeInner([card])
    card_game = RustGame(card_inner)
    card_game.step(
        Action(
            ActionKind.PLAY_HOUSE_CARD,
            house="baratheon",
            card="gregor_clegane",
            card_house="lannister",
            choice_tag=14,
        )
    )
    assert card_inner.stepped == [card]
    with pytest.raises(ValueError):
        card_game.step(
            Action(
                ActionKind.PLAY_HOUSE_CARD,
                house="baratheon",
                card="gregor_clegane",
                card_house="stark",
                choice_tag=14,
            )
        )
    with pytest.raises(ValueError):
        card_game.step(
            Action(
                ActionKind.PLAY_HOUSE_CARD,
                house="baratheon",
                card="gregor_clegane",
                card_house="lannister",
                choice_tag=13,
            )
        )


# -- context parsing -------------------------------------------------------


def test_choice_tags_cover_the_agreed_mapping() -> None:
    assert len(CHOICE_TAGS) == 15
    assert CHOICE_TAGS[0] == "queen_order"
    assert CHOICE_TAGS[7] == "preemptive_raid"
    assert CHOICE_TAGS[14] == "card_discard_or_retrieve"
    assert parse_choice_tag("7") == 7
    assert parse_choice_tag("choice_14") == 14
    assert parse_choice_tag("march_0_a") is None


def test_context_prefers_the_pending_payload_over_the_tag() -> None:
    game = FakeGame(
        FakeState(
            pending_choice={
                "kind": "region",
                "purpose": "queen_order",
                "house": "tyrell",
                "opponent": "lannister",
                "regions": ["highgarden"],
            }
        )
    )
    action = Action(
        ActionKind.WESTEROS_CHOICE,
        house="tyrell",
        region="highgarden",
        target="highgarden",
        order="3",
    )

    context = resolve_choice_context(game, action)

    assert context is not None
    assert context.purpose == "queen_order"
    assert context.house == "tyrell"
    assert context.target == "lannister"


def test_payload_lists_keep_their_order() -> None:
    state = FakeState(
        pending_choice={
            "kind": "track",
            "purpose": "king_beyond_highest",
            "house": "stark",
            "tracks": ["kings_court", "fiefdoms"],
        }
    )
    game = FakeGame(state)
    action = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=0)

    context = resolve_choice_context(game, action)

    assert context is not None
    assert context.tracks == (Track.KINGS_COURT, Track.FIEFDOMS)
    assert "top of King's Court" in format_action(game, action)


def test_robb_retreat_option_labels() -> None:
    game = FakeGame(
        FakeState(
            pending_choice={
                "kind": "option",
                "purpose": "robb_retreat",
                "house": "stark",
                "options": [0, 1],
                "queued": 1,
            }
        )
    )
    decline = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=0)
    take_over = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=1)

    assert "decline" in format_action(game, decline).lower()
    assert "choose the defender's retreat yourself" in format_action(game, take_over)


def test_rust_schema_tyrion_use_option_labels_and_scores() -> None:
    state = FakeState(
        pending_choice={
            "kind": "option",
            "purpose": "tyrion_use",
            "house": "lannister",
            "target": "stark",
            "options": [0, 1],
            "queued": 1,
        }
    )
    game = FakeGame(state)
    policy = GreedyPolicy(seed=0, noise=0.0)
    decline = Action(ActionKind.WESTEROS_CHOICE, house="lannister", amount=0)
    use = Action(ActionKind.WESTEROS_CHOICE, house="lannister", amount=1)

    assert "do not use" in format_action(game, decline)
    assert "Stark" in format_action(game, use)
    assert "replacement" in format_action(game, use)
    assert _score(game, policy, use) > _score(game, policy, decline)


def test_rust_schema_preemptive_highest_track_choice() -> None:
    state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={"card": "preemptive_raid", "strength": 8},
        pending_choice={
            "kind": "track",
            "purpose": "preemptive_highest",
            "house": "stark",
            "tracks": ["iron_throne", "kings_court"],
            "queued": 1,
        },
        tracks={
            Track.IRON_THRONE: ["stark", "baratheon", "lannister", "greyjoy", "tyrell", "martell"],
            Track.KINGS_COURT: ["stark", "martell", "greyjoy", "lannister", "baratheon", "tyrell"],
        },
    )
    game = FakeGame(state)
    policy = GreedyPolicy(seed=0, noise=0.0)
    iron_throne = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=0)
    kings_court = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=1)

    assert "drop 2 positions on Iron Throne" in format_action(game, iron_throne)
    assert "King's Court" in format_action(game, kings_court)
    assert np.isfinite(_score(game, policy, iron_throne))
    assert np.isfinite(_score(game, policy, kings_court))


def test_card_detail_separates_massing_from_mammoth_without_wildling_state() -> None:
    massing = FakeGame(
        FakeState(
            pending_choice={
                "kind": "card",
                "purpose": "card_discard_or_retrieve",
                "detail": "massing_discard",
                "house": "lannister",
                "cards": ["gregor_clegane"],
                "allow_none": False,
            },
            houses={"lannister": {"house_cards_available": ["gregor_clegane"]}},
        )
    )
    mammoth = FakeGame(
        FakeState(
            pending_choice={
                "kind": "card",
                "purpose": "card_discard_or_retrieve",
                "detail": "mammoth_retrieve",
                "house": "stark",
                "cards": ["eddard_stark"],
                "allow_none": True,
            },
            houses={"stark": {"house_cards_used": ["eddard_stark"]}},
        )
    )
    discard = Action(
        ActionKind.PLAY_HOUSE_CARD, house="lannister", card="gregor_clegane"
    )
    retrieve = Action(
        ActionKind.PLAY_HOUSE_CARD, house="stark", card="eddard_stark"
    )

    assert "Massing at Milkwater: discard Gregor Clegane" in format_action(
        massing, discard
    )
    assert "Mammoth Riders: retrieve Eddard Stark" in format_action(
        mammoth, retrieve
    )


def test_nested_rust_style_payload_is_understood() -> None:
    state = FakeState(
        pending_choice={
            "kind": {
                "Card": {
                    "purpose": {"TyrionSwap": {"target": "lannister"}},
                    "cards": ["gregor_clegane"],
                    "allow_none": False,
                }
            }
        },
        houses={"lannister": {"house_cards_available": ["gregor_clegane"]}},
    )
    game = FakeGame(state)
    action = Action(
        ActionKind.PLAY_HOUSE_CARD,
        house="lannister",
        card="gregor_clegane",
        card_house="lannister",
    )

    context = resolve_choice_context(game, action)

    assert context is not None
    assert context.purpose in ("tyrion_use", "tyrion_card")
    assert context.kind == "card"
    assert context.target == "lannister"
    assert context.cards == ("gregor_clegane",)
    text = format_action(game, action)
    assert "Tyrion Lannister" in text
    assert "Gregor Clegane" in text


# -- action labels ---------------------------------------------------------


def _pending_region(purpose: str, opponent: str = "lannister") -> dict:
    return {
        "kind": "region",
        "purpose": purpose,
        "house": "tyrell",
        "opponent": opponent,
        "regions": ["highgarden"],
        "allow_none": True,
    }


def test_format_action_names_the_queen_region_and_purpose() -> None:
    game = FakeGame(FakeState(pending_choice=_pending_region("queen_order")))
    action = Action(
        ActionKind.WESTEROS_CHOICE,
        house="tyrell",
        region="highgarden",
        target="highgarden",
    )

    text = format_action(game, action)

    assert "Queen of Thorns" in text
    assert "Lannister" in text
    assert "Highgarden" in text
    assert "Order" in text


def test_format_action_names_the_cersei_region_and_purpose() -> None:
    game = FakeGame(FakeState(pending_choice=_pending_region("cersei_order")))
    action = Action(
        ActionKind.WESTEROS_CHOICE,
        house="lannister",
        region="highgarden",
        target="highgarden",
    )

    text = format_action(game, action)

    assert "Cersei Lannister" in text
    assert "Highgarden" in text


def test_choice_tag_can_be_recovered_from_an_order_slot() -> None:
    from agot.engine.enums import ORDERS_BY_KEY

    game = FakeGame(FakeState(phase=Phase.WILDLING_ATTACK))
    tag_six = list(ORDERS_BY_KEY)[6]
    action = Action(
        ActionKind.WESTEROS_CHOICE, house="stark", amount=0, order=tag_six
    )

    text = format_action(game, action)

    assert "King Beyond the Wall" in text
    assert "top of Iron Throne" in text


def test_format_action_names_tracks_not_option_numbers() -> None:
    game = FakeGame(FakeState(phase=Phase.WILDLING_ATTACK))
    action = Action(
        ActionKind.WESTEROS_CHOICE, house="stark", amount=0, choice_tag=6
    )

    text = format_action(game, action)

    assert "King Beyond the Wall" in text
    assert "top of Iron Throne" in text
    assert "option" not in text.lower()

    lower = Action(
        ActionKind.WESTEROS_CHOICE, house="stark", amount=1, choice_tag=5
    )
    lower_text = format_action(game, lower)
    assert "bottom of King's Court" in lower_text


def test_format_action_explains_preemptive_raid_options() -> None:
    game = FakeGame(FakeState(phase=Phase.WILDLING_ATTACK))
    action_a = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=0, order="7")
    action_b = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=1, order="7")

    text_a = format_action(game, action_a)
    text_b = format_action(game, action_b)

    assert "Preemptive Raid: A)" in text_a
    assert "destroy 2" in text_a
    assert "Preemptive Raid: B)" in text_b
    assert "drop 2 positions" in text_b


def _massing_game() -> FakeGame:
    state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={"card": "massing_milkwater", "strength": 10},
        pending_choice={
            "kind": "card",
            "purpose": "massing_discard",
            "house": "lannister",
            "cards": ["gregor_clegane", "the_hound"],
        },
        houses={
            "lannister": {
                "house_cards_available": ["gregor_clegane", "the_hound"],
            }
        },
    )
    return FakeGame(state)


def test_format_action_distinguishes_massing_discard_from_playing_a_card() -> None:
    game = _massing_game()
    action = Action(
        ActionKind.PLAY_HOUSE_CARD,
        house="lannister",
        card="gregor_clegane",
    )

    text = format_action(game, action)

    assert "Massing at Milkwater" in text
    assert "discard" in text.lower()
    assert "Gregor Clegane" in text
    assert not text.startswith("Play House card")


def test_ai_action_summary_does_not_call_a_discard_a_card_selection() -> None:
    game = _massing_game()
    action = Action(
        ActionKind.PLAY_HOUSE_CARD,
        house="lannister",
        card="the_hound",
    )

    summary = ai_action_summary(game, "lannister", action)

    assert "selects a House card" not in summary
    assert "discard" in summary.lower()


def test_format_action_distinguishes_mammoth_retrieve() -> None:
    state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={"card": "mammoth_riders", "strength": 10},
        pending_choice={
            "kind": "card",
            "purpose": "mammoth_retrieve",
            "house": "stark",
            "cards": ["eddard_stark"],
        },
        houses={"stark": {"house_cards_used": ["eddard_stark"]}},
    )
    game = FakeGame(state)
    action = Action(
        ActionKind.PLAY_HOUSE_CARD, house="stark", card="eddard_stark"
    )

    text = format_action(game, action)

    assert "Mammoth Riders" in text
    assert "retrieve" in text.lower()
    assert "discard pile" in text.lower()


def test_format_action_names_the_patchface_target() -> None:
    state = FakeState(
        pending_choice={
            "kind": "card",
            "purpose": "patchface_discard",
            "house": "baratheon",
            "target": "lannister",
            "cards": ["gregor_clegane"],
            "allow_none": True,
        },
        houses={"lannister": {"house_cards_available": ["gregor_clegane"]}},
    )
    game = FakeGame(state)
    action = Action(
        ActionKind.PLAY_HOUSE_CARD,
        house="baratheon",
        card="gregor_clegane",
        card_house="lannister",
    )

    text = format_action(game, action)

    assert "Patchface" in text
    assert "discard" in text.lower()
    assert "Gregor Clegane" in text


def test_format_action_labels_a_generic_routed_destroy() -> None:
    state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={"card": "the_horde_descends", "strength": 8},
        pending_choice={
            "kind": "unit",
            "purpose": "generic_destroy",
            "house": "stark",
            "areas": ["winterfell"],
            "remaining": 1,
            "allow_none": False,
        },
        regions={
            "winterfell": {
                "controller": "stark",
                "units": {UnitType.FOOTMAN: 1},
                "routed": {UnitType.FOOTMAN: 1},
            }
        },
    )
    game = FakeGame(state)
    action = Action(
        ActionKind.CHOOSE_CASUALTY,
        house="stark",
        region="winterfell",
        unit="footman",
        amount=1,
    )

    text = format_action(game, action)

    assert "The Horde Descends" in text
    assert "destroy one routed Footman" in text


def test_format_action_labels_the_horde_muster_area_and_unit() -> None:
    state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={"card": "the_horde_descends", "strength": 8},
        pending_choice={
            "kind": "region",
            "purpose": "horde_muster_fort",
            "house": "stark",
            "regions": ["winterfell"],
        },
        regions={"winterfell": {"controller": "stark", "mustering_points": 2}},
    )
    game = FakeGame(state)
    area = Action(ActionKind.WESTEROS_CHOICE, house="stark", region="winterfell")
    unit = Action(
        ActionKind.MUSTER, house="stark", region="winterfell", unit="footman"
    )

    assert "The Horde Descends: muster in Winterfell" in format_action(game, area)
    assert (
        "The Horde Descends: muster one Footman in Winterfell"
        in format_action(game, unit)
    )


def test_format_action_labels_aeron_payment_and_card() -> None:
    state = FakeState(
        pending_choice={
            "kind": "card",
            "purpose": "aeron_card",
            "house": "greyjoy",
            "cards": ["victarion_greyjoy"],
        }
    )
    game = FakeGame(state)
    action = Action(
        ActionKind.PLAY_HOUSE_CARD,
        house="greyjoy",
        card="victarion_greyjoy",
    )

    text = format_action(game, action)

    assert "Aeron Damphair" in text
    assert "pay 2 Power" in text
    assert "Victarion Greyjoy" in text


def test_format_action_distinguishes_crow_replace_from_a_casualty() -> None:
    state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={"card": "crow_killers", "strength": 8},
        pending_choice={
            "kind": "unit",
            "purpose": "crow_replace_knight",
            "house": "stark",
            "areas": ["winterfell"],
            "remaining": 1,
            "allow_none": False,
        },
        regions={
            "winterfell": {
                "controller": "stark",
                "units": {UnitType.KNIGHT: 1},
            }
        },
    )
    game = FakeGame(state)
    action = Action(
        ActionKind.CHOOSE_CASUALTY,
        house="stark",
        region="winterfell",
        unit="knight",
        amount=1,
    )

    text = format_action(game, action)

    assert "Crow Killers" in text
    assert "replace" in text.lower()
    assert "routed Knight" in text
    assert "destroy" not in text.lower()


def test_format_action_explains_an_optional_decline() -> None:
    game = FakeGame(FakeState(pending_choice=_pending_region("cersei_order")))
    action = Action(ActionKind.PASS, house="lannister")

    text = format_action(game, action)

    assert "Decline" in text
    assert "Cersei Lannister" in text


def test_format_action_keeps_doran_labels_working() -> None:
    state = FakeState(
        phase=Phase.COMBAT,
        pending_battle={"stage": "Doran", "attacker": "stark", "defender": "lannister"},
    )
    game = FakeGame(state)
    action = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=2)

    text = format_action(game, action)

    assert "Doran Martell" in text
    assert "King's Court" in text


def test_format_action_keeps_wildling_tie_labels_working() -> None:
    state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={
            "card": "crow_killers",
            "tie_candidates": ["stark", "greyjoy"],
        },
    )
    game = FakeGame(state)
    action = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=1)

    text = format_action(game, action)

    assert "Wildling bid tie" in text
    assert "Greyjoy" in text


def test_battle_card_selection_still_reads_as_a_house_card() -> None:
    game = FakeGame()
    action = Action(
        ActionKind.PLAY_HOUSE_CARD, house="stark", card="eddard_stark"
    )

    assert format_action(game, action) == "Play House card Eddard Stark"
    assert "selects a House card" in ai_action_summary(game, "stark", action)


# -- greedy scoring --------------------------------------------------------


def test_massing_prefers_discarding_the_weakest_card() -> None:
    game = _massing_game()
    policy = GreedyPolicy(seed=0, noise=0.0)
    strong = Action(
        ActionKind.PLAY_HOUSE_CARD,
        house="lannister",
        card="gregor_clegane",
    )
    weak = Action(
        ActionKind.PLAY_HOUSE_CARD, house="lannister", card="the_hound"
    )

    assert _score(game, policy, weak) > _score(game, policy, strong)


def test_mammoth_prefers_retrieving_the_strongest_card() -> None:
    state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={"card": "mammoth_riders", "strength": 10},
        pending_choice={
            "kind": "card",
            "purpose": "mammoth_retrieve",
            "house": "lannister",
            "cards": ["gregor_clegane", "the_hound"],
        },
        houses={
            "lannister": {
                "house_cards_used": ["gregor_clegane", "the_hound"],
            }
        },
    )
    game = FakeGame(state)
    policy = GreedyPolicy(seed=0, noise=0.0)
    strong = Action(
        ActionKind.PLAY_HOUSE_CARD,
        house="lannister",
        card="gregor_clegane",
    )
    weak = Action(
        ActionKind.PLAY_HOUSE_CARD, house="lannister", card="the_hound"
    )

    assert _score(game, policy, strong) > _score(game, policy, weak)


def test_patchface_prefers_discarding_the_opponents_strongest_card() -> None:
    state = FakeState(
        pending_choice={
            "kind": "card",
            "purpose": "patchface_discard",
            "house": "baratheon",
            "target": "lannister",
            "cards": ["gregor_clegane", "the_hound"],
        },
        houses={
            "lannister": {
                "house_cards_available": ["gregor_clegane", "the_hound"],
            }
        },
    )
    game = FakeGame(state)
    policy = GreedyPolicy(seed=0, noise=0.0)
    strong = Action(
        ActionKind.PLAY_HOUSE_CARD,
        house="baratheon",
        card="gregor_clegane",
        card_house="lannister",
    )
    weak = Action(
        ActionKind.PLAY_HOUSE_CARD,
        house="baratheon",
        card="the_hound",
        card_house="lannister",
    )

    assert _score(game, policy, strong) > _score(game, policy, weak)


def test_crow_replacement_is_not_scored_as_a_plain_casualty() -> None:
    regions = {
        "winterfell": {
            "controller": "stark",
            "units": {UnitType.KNIGHT: 1},
        }
    }
    crow_state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={"card": "crow_killers", "strength": 8},
        pending_choice={
            "kind": "unit",
            "purpose": "crow_replace_knight",
            "house": "stark",
            "areas": ["winterfell"],
            "remaining": 1,
            "allow_none": False,
        },
        regions=regions,
    )
    destroy_state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={"card": "the_horde_descends", "strength": 8},
        pending_choice={
            "kind": "unit",
            "purpose": "generic_destroy",
            "house": "stark",
            "areas": ["winterfell"],
            "remaining": 1,
            "allow_none": False,
        },
        regions=regions,
    )
    action = Action(
        ActionKind.CHOOSE_CASUALTY,
        house="stark",
        region="winterfell",
        unit="knight",
    )
    policy = GreedyPolicy(seed=0, noise=0.0)

    crow = _score(FakeGame(crow_state), policy, action)
    destroyed = _score(FakeGame(destroy_state), policy, action)

    assert crow > destroyed
    assert crow < 0.0  # still a cost, just not the whole Knight


def test_king_beyond_other_prefers_the_track_that_costs_least() -> None:
    state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={"card": "king_beyond_the_wall", "strength": 10},
        pending_choice={
            "kind": "track",
            "purpose": "king_beyond_other",
            "house": "stark",
            "tracks": ["fiefdoms", "kings_court"],
        },
        tracks={
            Track.FIEFDOMS: ["stark", "baratheon", "lannister", "greyjoy", "tyrell", "martell"],
            Track.KINGS_COURT: [
                "tyrell",
                "martell",
                "greyjoy",
                "lannister",
                "baratheon",
                "stark",
            ],
        },
    )
    game = FakeGame(state)
    policy = GreedyPolicy(seed=0, noise=0.0)
    fiefdoms = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=0)
    kings_court = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=1)

    assert _score(game, policy, kings_court) > _score(game, policy, fiefdoms)


def test_king_beyond_highest_prefers_the_biggest_climb() -> None:
    state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={"card": "king_beyond_the_wall", "strength": 10},
        pending_choice={
            "kind": "track",
            "purpose": "king_beyond_highest",
            "house": "stark",
            "tracks": ["fiefdoms", "kings_court"],
        },
        tracks={
            Track.FIEFDOMS: [
                "tyrell",
                "martell",
                "lannister",
                "baratheon",
                "stark",
                "greyjoy",
            ],
            Track.KINGS_COURT: ["stark", "baratheon", "lannister", "greyjoy", "tyrell", "martell"],
        },
    )
    game = FakeGame(state)
    policy = GreedyPolicy(seed=0, noise=0.0)
    fiefdoms = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=0)
    kings_court = Action(ActionKind.WESTEROS_CHOICE, house="stark", amount=1)

    assert _score(game, policy, fiefdoms) > _score(game, policy, kings_court)


def test_horde_fort_prefers_losing_cheap_units() -> None:
    state = FakeState(
        phase=Phase.WILDLING_ATTACK,
        wildling_attack={"card": "the_horde_descends", "strength": 8},
        pending_choice={
            "kind": "region",
            "purpose": "horde_fort",
            "house": "stark",
            "regions": ["winterfell", "white_harbor"],
            "count": 2,
        },
        regions={
            "winterfell": {
                "controller": "stark",
                "units": {UnitType.FOOTMAN: 2},
            },
            "white_harbor": {
                "controller": "stark",
                "units": {UnitType.KNIGHT: 2},
            },
        },
    )
    game = FakeGame(state)
    policy = GreedyPolicy(seed=0, noise=0.0)
    cheap = Action(ActionKind.WESTEROS_CHOICE, house="stark", region="winterfell")
    costly = Action(ActionKind.WESTEROS_CHOICE, house="stark", region="white_harbor")

    assert _score(game, policy, cheap) > _score(game, policy, costly)


def test_optional_decline_can_beat_a_harmful_removal() -> None:
    state = FakeState(
        pending_choice={
            "kind": "region",
            "purpose": "cersei_order",
            "house": "lannister",
            "opponent": "stark",
            "regions": ["winterfell"],
            "allow_none": True,
        },
        regions={"winterfell": {"controller": "stark"}},
    )
    game = FakeGame(state)
    policy = GreedyPolicy(seed=0, noise=0.0)
    decline = Action(ActionKind.PASS, house="lannister")
    removal = Action(ActionKind.WESTEROS_CHOICE, house="lannister", region="winterfell")

    assert _score(game, policy, decline) >= _score(game, policy, removal)


def test_choice_scores_are_finite_and_deterministic() -> None:
    game = _massing_game()
    policy_a = GreedyPolicy(seed=0, noise=0.0)
    policy_b = GreedyPolicy(seed=0, noise=0.0)
    action = Action(
        ActionKind.PLAY_HOUSE_CARD, house="lannister", card="gregor_clegane"
    )

    first = _score(game, policy_a, action)
    second = _score(game, policy_b, action)

    assert np.isfinite(first)
    assert first == second


@pytest.mark.parametrize("purpose", sorted(CHOICE_TAGS.values()))
def test_every_context_purpose_has_a_finite_score(purpose: str) -> None:
    pending = {
        "kind": (
            "card"
            if "card" in purpose or "tyrion" in purpose or "aeron" in purpose
            else "region"
        ),
        "purpose": purpose,
        "house": "stark",
        "target": "lannister",
        "regions": ["winterfell"],
        "tracks": ["fiefdoms", "kings_court"],
        "areas": ["winterfell"],
        "cards": ["eddard_stark"],
        "remaining": 1,
        "allow_none": True,
    }
    game = FakeGame(
        FakeState(
            phase=Phase.WILDLING_ATTACK,
            wildling_attack={"card": "the_horde_descends", "strength": 8},
            pending_choice=pending,
            regions={
                "winterfell": {
                    "controller": "stark",
                    "units": {UnitType.KNIGHT: 1, UnitType.FOOTMAN: 2},
                }
            },
            houses={
                "stark": {
                    "house_cards_available": ["eddard_stark"],
                    "house_cards_used": [],
                }
            },
        )
    )
    policy = GreedyPolicy(seed=0, noise=0.0)
    actions = [
        Action(ActionKind.PASS, house="stark"),
        Action(
            ActionKind.WESTEROS_CHOICE,
            house="stark",
            region="winterfell",
            amount=0,
            order="7",
        ),
        Action(
            ActionKind.CHOOSE_CASUALTY,
            house="stark",
            region="winterfell",
            unit="knight",
            amount=1,
        ),
        Action(
            ActionKind.PLAY_HOUSE_CARD,
            house="stark",
            card="eddard_stark",
            card_house="stark",
        ),
        Action(ActionKind.MUSTER, house="stark", region="winterfell", unit="footman"),
        Action(ActionKind.UPGRADE, house="stark", region="winterfell", unit="knight"),
    ]

    scores = [_score(game, policy, action) for action in actions]

    assert all(np.isfinite(score) for score in scores)
