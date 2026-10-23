"""Smoke tests for the desktop interface and its pure helpers."""

from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter")

from agot.engine.actions import Action  # noqa: E402
from agot.engine.enums import OrderType, UnitType  # noqa: E402
from agot.engine.map import load_map  # noqa: E402
from agot.gui import (  # noqa: E402
    AREAS,
    BOARD_ANCHORS,
    BOARD_CROP,
    AgotApp,
    GameSettings,
    house_card_line,
    order_badge,
    supply_limit_text,
    unit_badge,
)


def test_area_layout_covers_every_region_exactly_once() -> None:
    placed = [region_id for _title, region_ids in AREAS for region_id in region_ids]

    assert len(placed) == len(set(placed))
    assert set(placed) == set(load_map().regions)


def test_board_image_anchors_cover_every_region() -> None:
    left, top, right, bottom = BOARD_CROP
    assert set(BOARD_ANCHORS) == set(load_map().regions)
    for x, y in BOARD_ANCHORS.values():
        assert left < x < right
        assert top < y < bottom


def test_order_badges_are_compact_and_recognisable() -> None:
    assert order_badge("march") == "M"
    assert order_badge("march_minus") == "M-1"
    assert order_badge("march_plus") == "M+1*"
    assert order_badge("defend_2") == "D2*"
    assert order_badge("support_plus") == "S+1*"
    assert order_badge("raid_star") == "R*"
    assert order_badge("consolidate_star") == "C*"


def test_unit_badges_follow_the_canonical_order() -> None:
    assert (
        unit_badge(
            {
                UnitType.FOOTMAN: 1,
                UnitType.KNIGHT: 2,
                UnitType.SIEGE: 0,
                UnitType.SHIP: 1,
            }
        )
        == "F1 K2 S1"
    )


def test_house_card_line_reports_strength_and_effect() -> None:
    eddard = house_card_line("stark", "eddard_stark")
    catelyn = house_card_line("stark", "catelyn_stark")

    assert eddard is not None and "Strength 4" in eddard
    assert eddard is not None and "2 sword(s)" in eddard
    assert catelyn is not None and "Defence Order" in catelyn


def test_supply_text_lists_every_level() -> None:
    text = supply_limit_text()

    for level in range(7):
        assert f"{level}:" in text
    assert "6: 4, 3, 2, 2, 2" in text


def test_march_establish_control_uses_a_prompt(monkeypatch) -> None:
    from agot.engine.game import Game, GameConfig

    plain = Action.march(
        "stark", "winterfell", "white_harbor", units=(1, 0, 0, 0)
    )
    establish = Action.march(
        "stark",
        "winterfell",
        "white_harbor",
        units=(1, 0, 0, 0),
        establish=True,
    )
    app = object.__new__(AgotApp)
    app.game = Game.new(GameConfig(seed=0))
    app.root = None
    app.march_variants = {
        (plain.region, plain.target, plain.units): [plain, establish]
    }
    monkeypatch.setattr("agot.gui.messagebox.askyesno", lambda *args, **kwargs: True)

    assert app._resolve_march_action(plain) == establish


def test_battle_summary_includes_troops_and_revealed_cards() -> None:
    from agot.engine.game import Game, GameConfig

    app = object.__new__(AgotApp)
    app.game = Game.new(GameConfig(seed=0))
    battle = {
        "region": "riverrun",
        "origin": "harrenhal",
        "attacker": "lannister",
        "defender": "stark",
        "attacker_units": {"footman": 2, "knight": 1, "siege": 0, "ship": 0},
        "defender_units": {"footman": 1, "knight": 1, "siege": 0, "ship": 0},
        "attacker_order_strength": 1,
        "defender_order_strength": 2,
        "attacker_card": "tywin_lannister",
        "defender_card": "eddard_stark",
        "stage": "Blade",
        "casualties_remaining": 1,
    }

    lines = app._battle_summary_lines(battle, reveal_cards=True)
    text = "\n".join(lines)

    assert "Attacker: F2 K1 (unit strength 4)" in text
    assert "Defender: F1 K1 (unit strength 3)" in text
    assert "Tywin Lannister" in text
    assert "Eddard Stark" in text


def test_region_sheet_hides_an_opponents_face_down_order() -> None:
    from agot.engine.game import Game, GameConfig
    from agot.gui import region_details

    game = Game.new(GameConfig(seed=0))
    action = game.legal_actions()[0]
    game.step(action)

    hidden = region_details(game, "stark", action.region)
    own = region_details(game, action.house, action.region)

    assert "face down" in hidden
    assert "face down" not in own


def _root_or_skip() -> tk.Tk:
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no graphical display available")
    root.withdraw()
    return root


def test_setup_screen_constructs() -> None:
    root = _root_or_skip()
    try:
        app = AgotApp(
            root,
            defaults=GameSettings(house="stark", opponents="greedy"),
        )
        root.update_idletasks()
        assert app.setup is not None
        assert app.setup.selected_house == "stark"
    finally:
        root.destroy()


def test_game_screen_starts_with_a_human_turn(tmp_path) -> None:
    root = _root_or_skip()
    try:
        app = AgotApp(root, match_log_dir=tmp_path)
        started = app.start_game(
            GameSettings(
                house="baratheon",
                opponents="greedy",
                simulations=1,
                determinizations=1,
                seed=0,
                max_rounds=1,
            )
        )
        root.update_idletasks()
        assert started
        assert app.game is not None
        assert app.game.current_player() == "baratheon"
        assert app.setup is None
        assert app.game_frame is not None

        # Planning uses the guided order picker: choose the order type, then
        # the specific order, then the region.
        app._select_order_type(OrderType.MARCH)
        app._select_order_family("march")
        app._play_selected()
        assert app.steps == 1
        assert not app.finished

        app._undo()
        assert app.steps == 0
        assert app.undo_stack == []
    finally:
        root.destroy()
