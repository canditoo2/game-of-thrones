"""Tests for the random, greedy and MCTS policies."""

from __future__ import annotations

import numpy as np

from agot.agents import MCTS, GreedyPolicy, RandomPolicy, outcome_vector
from agot.agents.mcts import SearchResult
from agot.engine.game import Game, GameConfig
from agot.engine.observation import HOUSE_ORDER


def test_random_policy_target_matches_actions() -> None:
    game = Game.new(GameConfig(seed=0))
    policy = RandomPolicy(seed=0)
    action, target = policy.decide(game)
    assert action in game.legal_actions()
    assert target.shape == (len(game.legal_actions()),)
    np.testing.assert_allclose(target.sum(), 1.0)


def test_greedy_policy_target_is_one_hot() -> None:
    game = Game.new(GameConfig(seed=1))
    policy = GreedyPolicy(seed=0)
    actions = game.legal_actions()
    action, target = policy.decide(game)
    assert action in actions
    assert target.sum() == 1.0
    assert target.max() == 1.0


def test_mcts_search_returns_a_visit_distribution() -> None:
    game = Game.new(GameConfig(seed=2))
    mcts = MCTS(simulations=8, seed=0)
    result = mcts.search(game)
    assert len(result.actions) == len(game.legal_actions())
    # The first simulation expands the root without descending, so the children
    # accumulate one visit fewer than the simulation budget.
    assert 0 < result.visits.sum() <= mcts.simulations
    probabilities = result.probabilities(temperature=1.0)
    np.testing.assert_allclose(probabilities.sum(), 1.0)


def test_mcts_decide_returns_a_legal_action() -> None:
    game = Game.new(GameConfig(seed=3))
    mcts = MCTS(simulations=6, seed=0)
    action, target = mcts.decide(game)
    assert action in game.legal_actions()
    assert target.shape == (len(game.legal_actions()),)


def test_outcome_vector_has_one_winner() -> None:
    config = GameConfig(seed=4)
    game = Game.new(config)
    game.state.winner = "stark"
    outcome = outcome_vector(game)
    assert outcome.sum() == 1.0 + (-1.0) * (len(config.houses) - 1)
    assert outcome[HOUSE_ORDER.index("stark")] == 1.0


def test_visit_distribution_sharpens_as_temperature_drops() -> None:
    """Temperature is what the recorded self-play target is built from."""
    game = Game.new(GameConfig(seed=5))
    actions = game.legal_actions()[:3]
    # A skewed visit count: the point is the shape, not how it was produced.
    result = SearchResult(actions, np.array([6.0, 2.0, 1.0]), np.zeros(3))

    hot = result.probabilities(temperature=1.0)
    cold = result.probabilities(temperature=0.25)
    frozen = result.probabilities(temperature=0.0)

    np.testing.assert_allclose(hot.sum(), 1.0)
    np.testing.assert_allclose(cold.sum(), 1.0)
    assert cold[0] > hot[0], "a colder temperature must concentrate the mass"
    assert frozen[0] == 1.0, "temperature 0 is a one-hot on the most visited"
    assert frozen[1] == frozen[2] == 0.0


def test_pick_honours_a_temperature_override() -> None:
    """The schedule is applied by the caller, without rebuilding the policy."""
    game = Game.new(GameConfig(seed=6))
    mcts = MCTS(simulations=8, temperature=1.0, seed=0)
    result = mcts.search(game)

    action, target = mcts.pick(result, temperature=0.25)

    assert action in result.actions
    np.testing.assert_allclose(target, result.probabilities(0.25), atol=1e-6)
    assert mcts.temperature == 1.0, "the instance's own temperature is unchanged"
