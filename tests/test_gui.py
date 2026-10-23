"""Smoke tests for the desktop interface and its pure helpers."""

from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter")

from agot.engine.enums import UnitType  # noqa: E402
from agot.engine.map import load_map  # noqa: E402
from agot.gui import (  # noqa: E402
    AREAS,
    BOARD_ANCHORS,
    BOARD_CROP,
    AgotApp,
    GameSettings,
    order_badge,
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

        # The first legal action is selected automatically; playing it must
        # advance the session and leave the game screen alive.
        app._play_selected()
        assert app.steps == 1
        assert not app.finished

        app._undo()
        assert app.steps == 0
        assert app.undo_stack == []
    finally:
        root.destroy()
