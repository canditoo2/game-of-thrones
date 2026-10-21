"""Tests for the interactive play renderer and game loop."""

from __future__ import annotations

import pytest

from agot.engine.actions import Action
from agot.engine.game import Game, GameConfig
from agot.play import (
    format_action,
    play_interactive,
    render_board,
    resolve_checkpoint,
)


def test_format_action_describes_a_planning_move() -> None:
    game = Game.new(GameConfig(seed=0))
    action = game.legal_actions()[0]

    text = format_action(game, action)

    assert text.startswith("Place ")
    assert " in " in text
    assert "None" not in text


def test_format_action_describes_units_and_establishing_control() -> None:
    game = Game.new(GameConfig(seed=0))
    action = Action.march(
        "stark",
        "winterfell",
        "white_harbor",
        units=(1, 2, 0, 0),
        establish=True,
    )

    assert (
        format_action(game, action)
        == "March F1 K2 from Winterfell to White Harbor (establish control)"
    )


def test_format_action_describes_support() -> None:
    game = Game.new(GameConfig(seed=0))

    assert (
        format_action(game, Action.declare_support("stark", "winterfell", "refuse"))
        == "Refuse to support from Winterfell"
    )
    assert (
        format_action(game, Action.declare_support("stark", "winterfell", "defender"))
        == "Support the defender from Winterfell"
    )


def test_render_board_hides_other_houses_face_down_orders() -> None:
    game = Game.new(GameConfig(seed=0))
    action = game.legal_actions()[0]
    region_name = game.map.regions[action.region].name

    game.step(action)
    board = render_board(game, human_house="stark")
    line = next(line for line in board.splitlines() if region_name in line)

    assert "[?]" in line
    assert "March" not in line


def test_render_board_shows_the_humans_own_order() -> None:
    game = Game.new(GameConfig(seed=0))
    action = game.legal_actions()[0]
    region_name = game.map.regions[action.region].name

    game.step(action)
    board = render_board(game, human_house=action.house)
    line = next(line for line in board.splitlines() if region_name in line)

    assert "[?]" not in line
    assert "[March" in line


def test_resolve_checkpoint_prefers_the_best_checkpoint(tmp_path) -> None:
    (tmp_path / "agot.pt").write_bytes(b"latest")
    (tmp_path / "agot-best.pt").write_bytes(b"best")

    assert resolve_checkpoint(tmp_path).endswith("agot-best.pt")


def test_play_interactive_plays_a_typed_action() -> None:
    output: list[str] = []
    result = play_interactive(
        human_house="baratheon",
        opponents="greedy",
        simulations=1,
        determinizations=1,
        config=GameConfig(seed=0, max_rounds=1),
        input_fn=lambda _prompt: "1",
        output_fn=output.append,
        max_steps=1,
    )

    assert result.steps == 1
    assert result.aborted
    assert result.human_house == "baratheon"
    assert any("Place " in line for line in output)
    assert any("Stopped after 1 decisions" in line for line in output)


def test_play_interactive_can_quit_before_the_first_move() -> None:
    output: list[str] = []
    result = play_interactive(
        human_house="baratheon",
        opponents="greedy",
        simulations=1,
        determinizations=1,
        config=GameConfig(seed=0, max_rounds=1),
        input_fn=lambda _prompt: "q",
        output_fn=output.append,
    )

    assert result.steps == 0
    assert result.aborted
    assert any("Game ended early" in line for line in output)


def test_play_interactive_can_undo_the_previous_move() -> None:
    answers = iter(["1", "u", "q"])
    output: list[str] = []
    result = play_interactive(
        human_house="baratheon",
        opponents="greedy",
        simulations=1,
        determinizations=1,
        config=GameConfig(seed=0, max_rounds=1),
        input_fn=lambda _prompt: next(answers),
        output_fn=output.append,
        max_steps=50,
    )

    assert result.steps == 0
    assert result.aborted
    assert any("Undid your last decision" in line for line in output)


def test_play_interactive_rejects_an_unknown_house() -> None:
    with pytest.raises(ValueError):
        play_interactive(
            human_house="targaryen",
            opponents="greedy",
            simulations=1,
            determinizations=1,
        )
