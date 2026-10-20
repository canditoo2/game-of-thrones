"""Tests for self-play data, the replay buffer and evaluation."""

from __future__ import annotations

import numpy as np

from agot.agents import RandomPolicy
from agot.engine.game import GameConfig
from agot.engine.observation import HOUSE_ORDER
from agot.training.evaluate import evaluate_match, evaluate_vs_field, wilson_interval
from agot.training.replay import Example, ReplayBuffer
from agot.training.selfplay import (
    DEFAULT_TEMPERATURE_SCHEDULE,
    play_game,
    scheduled_temperature,
    self_play,
)


def test_replay_buffer_is_structured_arrays_and_drops_the_oldest() -> None:
    from agot.engine.observation import ACTION_DIM, N_HOUSES, STATE_DIM

    buffer = ReplayBuffer(capacity=4)
    for player in range(6):
        count = 2 + player
        buffer.add(
            Example(
                state=np.full(STATE_DIM, player, dtype=np.float32),
                actions=np.full((count, ACTION_DIM), player, dtype=np.float32),
                policy=np.full(count, 1.0 / count, dtype=np.float32),
                player=player,
                outcome=np.zeros(N_HOUSES, dtype=np.float32),
            )
        )

    assert len(buffer) == 4
    examples = buffer.all()
    # capacity 4 with a quarter-drop keeps the last four entries: 2,3,4,5.
    assert [example.player for example in examples] == [2, 3, 4, 5]
    assert all(example.actions.shape[1] == ACTION_DIM for example in examples)
    assert all(example.state.shape == (STATE_DIM,) for example in examples)
    # The examples are views into the buffer, not per-example allocations.
    assert examples[0].actions.base is not None


def test_temperature_schedule_sharpens_after_the_opening() -> None:
    """The recorded self-play target is built from the temperature.

    A flat ``1.0`` for a whole game, with only a handful of visits per candidate,
    trains the network to imitate something close to noise: hence a schedule.
    """
    schedule = ((3, 1.0), (10_000, 0.25))
    assert scheduled_temperature(schedule, 1) == 1.0
    assert scheduled_temperature(schedule, 3) == 1.0
    assert scheduled_temperature(schedule, 4) == 0.25
    assert scheduled_temperature(schedule, 99) == 0.25
    # Past the last band the final temperature holds, rather than raising.
    assert scheduled_temperature(DEFAULT_TEMPERATURE_SCHEDULE, 10_000) == 0.25
    assert scheduled_temperature(((1, 0.5),), 7) == 0.5


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


def test_wilson_interval_brackets_the_mean() -> None:
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_evaluate_vs_field_rotates_the_candidate_seat() -> None:
    result = evaluate_vs_field(_factory, {"random": _factory}, games=6, seed=0)
    assert result.games == 6
    assert 0 <= result.wins <= 6
    assert result.wins + result.draws <= 6
    assert 0.0 <= result.win_rate <= 1.0
    assert result.ci_low <= result.win_rate <= result.ci_high
    assert len(result.per_seat) == 6  # every seat was used across six games
