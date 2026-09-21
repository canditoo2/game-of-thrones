"""Tests for self-play data, the replay buffer and evaluation."""

from __future__ import annotations

import numpy as np

from agot.agents import RandomPolicy
from agot.engine.game import GameConfig
from agot.engine.observation import HOUSE_ORDER
from agot.training.evaluate import evaluate_match
from agot.training.replay import ReplayBuffer
from agot.training.selfplay import play_game, self_play


def _factory(house: str, seed: int) -> RandomPolicy:
    return RandomPolicy(seed=seed)


def test_self_play_labels_every_example_with_the_outcome() -> None:
    record = self_play(_factory, GameConfig(seed=0))
    assert record.examples
    assert record.winner in GameConfig().houses
    for example in record.examples:
        assert example.state.ndim == 1
        assert example.actions.ndim == 2
        assert example.policy.shape == (len(example.actions),)
        np.testing.assert_allclose(example.policy.sum(), 1.0, atol=1e-6)
        assert 0 <= example.player < len(HOUSE_ORDER)
        assert example.outcome[example.player] in (-1.0, 1.0)


def test_play_game_accepts_one_policy_per_house() -> None:
    config = GameConfig(seed=1)
    policies = {house: RandomPolicy(seed=i) for i, house in enumerate(config.houses)}
    record = play_game(policies, config)
    assert record.steps > 0


def test_replay_buffer_round_trips(tmp_path) -> None:
    record = self_play(_factory, GameConfig(seed=2))
    buffer = ReplayBuffer(capacity=10_000)
    buffer.extend(record.examples)
    path = str(tmp_path / "replay.npz")
    buffer.save(path)

    loaded = ReplayBuffer.load(path)
    assert len(loaded) == len(buffer)
    original = buffer.all()[0]
    restored = loaded.all()[0]
    np.testing.assert_allclose(original.state, restored.state)
    np.testing.assert_allclose(original.actions, restored.actions)
    np.testing.assert_allclose(original.policy, restored.policy)
    assert original.player == restored.player
    np.testing.assert_allclose(original.outcome, restored.outcome)


def test_evaluate_match_splits_the_wins() -> None:
    result = evaluate_match(_factory, _factory, games=2, seed=0)
    assert result.games == 2
    assert result.wins_a + result.wins_b == 2
    assert 0.0 <= result.win_rate_a <= 1.0
    assert result.average_steps > 0
