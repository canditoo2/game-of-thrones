"""Tests for the CLI policy factory and the round-robin command."""

from __future__ import annotations

import json

import pytest

from agot.cli import _policy_factory, build_parser, main
from agot.engine.game import Game, GameConfig


def test_importing_the_cli_does_not_load_torch() -> None:
    """Demo/evaluate must start without paying the training stack's import."""
    import subprocess
    import sys

    output = subprocess.check_output(
        [sys.executable, "-c", "import sys, agot.cli; print('torch' in sys.modules)"],
        text=True,
    )
    assert output.strip() == "False"


def test_net_policy_without_a_checkpoint_is_rejected() -> None:
    """A bare ``net`` has no weights to load, so it fails with a message."""
    with pytest.raises(SystemExit):
        _policy_factory("net")


def test_train_gate_mode_defaults_to_inline() -> None:
    args = build_parser().parse_args(["train"])
    assert args.gate_mode == "inline"


def test_train_phase_budgets_defaults_to_off() -> None:
    """The per-phase caps are opt-in: a run without the switch is unchanged."""
    assert build_parser().parse_args(["train"]).phase_budgets == "off"
    args = build_parser().parse_args(["train", "--phase-budgets", "default"])
    assert args.phase_budgets == "default"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--phase-budgets", "bogus"])
def test_train_q_head_and_teacher_flags_are_off_by_default() -> None:
    args = build_parser().parse_args(["train"])
    assert args.q_head is False
    assert args.q_weight == 0.5
    assert args.planning_teacher_rate == 0.0
    assert args.planning_teacher_games == 0


def test_gate_subcommand_parses_the_queue_options() -> None:
    args = build_parser().parse_args(
        [
            "gate",
            "--checkpoint",
            "checkpoints/run5/agot.pt",
            "--max-minutes",
            "90",
            "--shard",
            "2/3",
        ]
    )
    assert args.func.__name__ == "cmd_gate"
    assert args.checkpoint == "checkpoints/run5/agot.pt"
    assert args.max_minutes == 90
    assert args.shard == "2/3"


def test_net_policy_with_a_missing_checkpoint_is_rejected(tmp_path) -> None:
    pytest.importorskip("torch")
    with pytest.raises(SystemExit):
        _policy_factory(f"net:{tmp_path / 'missing.pt'}")


def test_net_policy_plays_a_legal_move(tmp_path) -> None:
    pytest.importorskip("torch")
    from agot.nn.model import PolicyValueNet

    path = tmp_path / "tiny.pt"
    PolicyValueNet(hidden=8).save(str(path))
    factory = _policy_factory(f"net:{path}", simulations=2, determinizations=1)
    game = Game.new(GameConfig(seed=0))
    action, target = factory("stark", 0).decide(game)
    assert len(target) == len(game.legal_actions())
    assert action in game.legal_actions()


def test_play_command_opens_the_gui(monkeypatch) -> None:
    pytest.importorskip("tkinter")
    import agot.gui

    calls: list[dict[str, object]] = []

    def fake_launch(**kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(agot.gui, "launch", fake_launch)
    code = main(
        [
            "play",
            "--house",
            "baratheon",
            "--opponents",
            "greedy",
            "--max-rounds",
            "2",
            "--simulations",
            "8",
            "--determinizations",
            "1",
        ]
    )

    assert code == 0
    assert calls == [
        {
            "checkpoint": None,
            "house": "baratheon",
            "opponents": "greedy",
            "simulations": 8,
            "determinizations": 1,
            "seed": 0,
            "max_rounds": 2,
        }
    ]


def test_round_robin_runs_two_checkpoints(tmp_path, capsys) -> None:
    pytest.importorskip("torch")
    from agot.nn.model import PolicyValueNet

    checkpoints = []
    for index in range(2):
        path = tmp_path / f"tiny{index}.pt"
        PolicyValueNet(hidden=8).save(str(path))
        checkpoints.append(str(path))

    code = main(
        [
            "round-robin",
            "--checkpoints",
            *checkpoints,
            "--games",
            "1",
            "--max-rounds",
            "2",
            "--simulations",
            "2",
            "--determinizations",
            "1",
            "--anchors",
            "",
            "--seed",
            "0",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["results"]) == set(checkpoints)
    assert all(result["games"] == 1 for result in payload["results"].values())
