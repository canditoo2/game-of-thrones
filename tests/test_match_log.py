"""Tests for the local human-vs-bot match log."""

from __future__ import annotations

import json

import numpy as np

from agot.engine.game import Game, GameConfig
from agot.engine.observation import N_HOUSES
from agot.match_log import MatchLog
from agot.training.replay import ReplayBuffer


def test_match_log_writes_jsonl_and_replay_examples(tmp_path) -> None:
    log = MatchLog(
        tmp_path,
        human_house="stark",
        opponents="net",
        checkpoint="checkpoints/run3/agot-best.pt",
        simulations=8,
        determinizations=2,
        seed=0,
        max_rounds=10,
    )
    game = Game.new(GameConfig(seed=0))
    observation = game.observe()
    actions = list(observation.actions)
    target = np.zeros(len(actions), dtype=np.float64)
    target[0] = 1.0

    log.decision(
        step=0,
        round_number=1,
        phase="planning",
        action_step="raid",
        house=observation.actor,
        actor="bot",
        source="mcts",
        action=actions[0],
        legal_actions=actions,
        chosen_index=0,
        target=target,
        observation=observation,
    )
    outcome = np.zeros(N_HOUSES, dtype=np.float32)
    outcome[0] = 1.0
    log.finish(
        winner="baratheon",
        rounds=1,
        steps=1,
        aborted=False,
        outcome=outcome,
    )

    records = [
        json.loads(line)
        for line in log.jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["type"] for record in records] == ["start", "decision", "end"]
    assert records[1]["source"] == "mcts"
    assert records[1]["target_max"] == 1.0
    assert log.npz_path.exists()

    restored = ReplayBuffer.load(log.npz_path)
    assert len(restored) == 1
    example = restored[0]
    assert example.policy.shape == (len(actions),)
    assert abs(float(example.policy.sum()) - 1.0) < 1e-6
    assert np.array_equal(example.outcome, outcome)


def test_human_decisions_are_logged_but_not_training_examples(tmp_path) -> None:
    log = MatchLog(
        tmp_path,
        human_house="stark",
        opponents="greedy",
        checkpoint=None,
        simulations=1,
        determinizations=1,
        seed=0,
        max_rounds=1,
    )
    game = Game.new(GameConfig(seed=0))
    observation = game.observe()
    actions = list(observation.actions)
    target = np.zeros(len(actions), dtype=np.float64)
    target[0] = 1.0
    log.decision(
        step=0,
        round_number=1,
        phase="planning",
        action_step="raid",
        house=observation.actor,
        actor="human",
        source="human",
        action=actions[0],
        legal_actions=actions,
        chosen_index=0,
        target=target,
        observation=observation,
    )
    log.finish(winner=None, rounds=1, steps=1, aborted=True)

    assert log.example_count == 0
    assert not log.npz_path.exists()
