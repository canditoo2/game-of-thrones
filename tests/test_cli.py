"""Tests for the CLI policy factory and the round-robin command."""

from __future__ import annotations

import json

import pytest

from agot.cli import _policy_factory, main
from agot.engine.game import Game, GameConfig


def test_net_policy_without_a_checkpoint_is_rejected() -> None:
    """A bare ``net`` has no weights to load, so it fails with a message."""
    with pytest.raises(SystemExit):
        _policy_factory("net")


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
