"""Tests for tools/reanalyze_human_game.py (the offline human-game reanalysis)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from agot.engine.game import Game, GameConfig
from agot.match_log import _action_record

TOOL = Path(__file__).resolve().parents[1] / "tools" / "reanalyze_human_game.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("reanalyze_human_game", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_undo_drops_the_last_human_decision_and_what_followed() -> None:
    tool = _load_tool()
    records = [
        {"type": "start"},
        {"type": "decision", "actor": "bot", "step": 0},
        {"type": "decision", "actor": "human", "step": 1},
        {"type": "decision", "actor": "bot", "step": 2},
        {"type": "undo", "step": 1},
        {"type": "decision", "actor": "human", "step": 1},
        {"type": "end"},
    ]

    active = tool.active_decisions(records)

    assert [item["step"] for item in active] == [0, 1]
    assert [item["actor"] for item in active] == ["bot", "human"]


def _scripted_log(tmp_path: Path, seed: int = 3) -> Path:
    """Play one deterministic game, logging every decision like the GUI does."""
    config = GameConfig(
        seed=seed,
        max_rounds=10,
        deferred_battles=True,
        sequential_westeros=True,
        doran_choice=True,
        wildling_tie_choice=True,
        choice_choices=True,
    )
    game = Game.new(config)
    decisions: list[dict] = []
    steps = 0
    while not game.is_terminal() and steps < 4000:
        while (
            game.state.phase.value == "westeros"
            and game.state.westeros_waiting
            and not game.is_terminal()
        ):
            game.continue_westeros()
        if game.is_terminal():
            break
        actions = game.legal_actions()
        if not actions:
            break
        action = actions[0]
        player = game.current_player() or action.house
        decisions.append(
            {
                "type": "decision",
                "step": steps,
                "round": game.state.round,
                "phase": game.state.phase.value,
                "action_step": game.state.action_step.value,
                "house": player,
                "actor": "bot",
                "source": "scripted",
                "action": _action_record(action),
            }
        )
        game.step(action)
        steps += 1
    assert game.is_terminal()

    path = tmp_path / "scripted.jsonl"
    records = [
        {
            "type": "start",
            "format": "agot-match-v1",
            "human_house": "stark",
            "opponents": "net",
            "checkpoint": None,
            "simulations": 1,
            "determinizations": 1,
            "seed": seed,
            "max_rounds": 10,
        },
        *decisions,
        {
            "type": "end",
            "winner": game.winner(),
            "rounds": game.state.round,
            "steps": steps,
            "aborted": False,
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


def test_replay_reproduces_a_logged_game(tmp_path: Path) -> None:
    tool = _load_tool()
    log = _scripted_log(tmp_path)
    records = tool.load_records(log)
    start = next(record for record in records if record["type"] == "start")
    end = next(record for record in records if record["type"] == "end")
    decisions = tool.active_decisions(records)

    solution = tool.find_solution(tool.game_config(start), decisions)
    assert solution is not None

    captured, replay = tool.replay_with_choices(
        Game.new(tool.game_config(start)), decisions, solution
    )

    assert replay.is_terminal()
    assert replay.winner() == end["winner"]
    assert replay.state.round == end["rounds"]
    assert len(captured) == len(decisions)


def test_replay_refuses_a_log_that_does_not_match(tmp_path: Path) -> None:
    """A corrupted log must fail instead of producing misaligned training data."""
    tool = _load_tool()
    log = _scripted_log(tmp_path)
    records = tool.load_records(log)
    start = next(record for record in records if record["type"] == "start")
    decisions = tool.active_decisions(records)
    # Point the opening order at a region its house cannot reach.
    opening = decisions[0]["action"]
    opening["region"] = "pyke" if opening["region"] != "pyke" else "winterfell"

    assert tool.find_solution(tool.game_config(start), decisions) is None
