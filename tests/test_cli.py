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


def test_seed_replay_is_opt_in_and_seeds_an_empty_buffer(tmp_path) -> None:
    """`--seed-replay` is off by default and loads examples into a fresh buffer."""
    import numpy as np

    from agot.cli import seed_replay_buffer
    from agot.training.replay import STATE_DIM, Example, ReplayBuffer

    assert build_parser().parse_args(["train"]).seed_replay is None

    example = Example(
        state=np.zeros(STATE_DIM, dtype=np.float32),
        actions=np.zeros((2, 141), dtype=np.float32),
        policy=np.array([0.5, 0.5], dtype=np.float32),
        player=0,
        outcome=np.ones(6, dtype=np.float32),
    )
    source = ReplayBuffer(capacity=8)
    source.add(example)
    path = tmp_path / "seed.npz"
    source.save(str(path))

    buffer = ReplayBuffer(capacity=8)
    assert seed_replay_buffer(buffer, path) == 1
    assert len(buffer) == 1


def test_train_parses_init_from() -> None:
    """A warm start is weights-only and must be selectable from the CLI."""
    default = build_parser().parse_args(["train"])
    assert default.init_from is None

    args = build_parser().parse_args(["train", "--init-from", "run6/agot.pt"])
    assert args.init_from == "run6/agot.pt"


def test_train_rejects_resume_with_init_from() -> None:
    with pytest.raises(SystemExit):
        main(["train", "--resume", "a.pt", "--init-from", "b.pt"])


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
            "checkpoint_map": None,
            "house": "baratheon",
            "opponents": "greedy",
            "simulations": 8,
            "determinizations": 1,
            "seed": None,
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
