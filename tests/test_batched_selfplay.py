"""Tests for stepwise MCTS and batched (lockstep) self-play."""

from __future__ import annotations

import numpy as np
import pytest

from agot.agents import GreedyPolicy
from agot.agents.mcts import MCTS, UniformEvaluator
from agot.engine.game import Game, GameConfig, PythonGame
from agot.engine.observation import N_HOUSES
from agot.training.selfplay import batched_self_play, play_game


def _stepwise_search(mcts: MCTS, game: Game, simulations: int):
    """Drive ``mcts`` one simulation at a time, as the batched driver does."""
    root = mcts.open_search(game)
    assert root is not None
    for _ in range(simulations):
        simulation = mcts.prepare_simulation(root)
        if simulation.leaf is None:
            mcts.finish_simulation(simulation)
        else:
            priors, value = mcts.evaluator(
                simulation.leaf.game, simulation.actions
            )
            mcts.finish_simulation(simulation, priors, value)
    return mcts.result_from_root(root)


def test_stepwise_search_matches_plain_search() -> None:
    game = Game.new(GameConfig(seed=11))
    plain = MCTS(UniformEvaluator(), simulations=16, seed=3).search(game)
    stepwise = _stepwise_search(
        MCTS(UniformEvaluator(), simulations=16, seed=3), game, 16
    )
    assert stepwise.actions == plain.actions
    assert (stepwise.visits == plain.visits).all()


def test_batched_selfplay_produces_one_record_per_game() -> None:
    configs = [GameConfig(seed=seed, max_rounds=2) for seed in (0, 1, 2)]
    records = batched_self_play(
        UniformEvaluator(), configs, simulations=4, seed=0
    )
    assert len(records) == 3
    for record in records:
        assert record.examples, "every game should record decisions"
        for example in record.examples:
            assert example.outcome.shape == (N_HOUSES,)
            assert example.policy.sum() > 0


def test_batched_selfplay_matches_single_game_example_count() -> None:
    """A pure-self-play batched game records roughly what play_game records."""
    config = GameConfig(seed=4, max_rounds=2)
    single = play_game(
        {
            house: MCTS(UniformEvaluator(), simulations=4, seed=i)
            for i, house in enumerate(config.houses)
        },
        config,
        game_class=PythonGame,
    )
    batched = batched_self_play(
        UniformEvaluator(), [config], simulations=4, seed=0
    )[0]
    assert len(batched.examples) == len(single.examples)


def test_scripted_seats_are_not_imitated_by_default() -> None:
    config = GameConfig(seed=7, max_rounds=2)
    opponent_house = config.houses[0]
    opponents = [{opponent_house: GreedyPolicy(seed=0)}]
    records = batched_self_play(
        UniformEvaluator(),
        [config],
        simulations=4,
        seed=0,
        opponents=opponents,
    )
    record = records[0]
    assert record.examples
    assert all(example.player != 0 for example in record.examples), (
        "decisions from the scripted seat should be skipped"
    )


def test_batched_selfplay_applies_the_temperature_schedule() -> None:
    """A cold schedule must reach the recorded targets, not just the moves.

    Wiring a schedule into the driver while leaving ``pick`` on the policy's own
    temperature is a silent no-op: the games change but the targets do not, and
    the target is the whole point of the schedule.
    """
    configs = [GameConfig(seed=seed, max_rounds=3) for seed in (0, 1)]
    hot = batched_self_play(
        UniformEvaluator(),
        configs,
        simulations=6,
        seed=0,
        temperature_schedule=((10_000, 1.0),),
    )
    cold = batched_self_play(
        UniformEvaluator(),
        configs,
        simulations=6,
        seed=0,
        temperature_schedule=((10_000, 0.0),),
    )

    def mean_entropy(records) -> float:
        policies = [
            example.policy for record in records for example in record.examples
        ]
        assert policies
        return float(np.mean([-(p * np.log(p + 1e-12)).sum() for p in policies]))

    hot_entropy = mean_entropy(hot)
    cold_entropy = mean_entropy(cold)
    assert cold_entropy < hot_entropy, (
        f"the schedule never reached the targets: cold={cold_entropy:.3f} "
        f"hot={hot_entropy:.3f}"
    )
    assert cold_entropy < 1e-6, "temperature 0 must record a one-hot target"


def test_batched_selfplay_with_pimc() -> None:
    configs = [GameConfig(seed=seed, max_rounds=2) for seed in (0, 1)]
    records = batched_self_play(
        UniformEvaluator(),
        configs,
        simulations=8,
        determinizations=4,
        seed=0,
    )
    assert len(records) == 2
    assert all(record.examples for record in records)


def test_scripted_seats_can_be_recorded_on_request() -> None:
    config = GameConfig(seed=7, max_rounds=2)
    opponent_house = config.houses[0]
    records = batched_self_play(
        UniformEvaluator(),
        [config],
        simulations=4,
        seed=0,
        opponents=[{opponent_house: GreedyPolicy(seed=0)}],
        record_opponents=True,
    )
    assert any(example.player == 0 for example in records[0].examples)


def test_parallel_self_play_in_process() -> None:
    pytest.importorskip("torch")
    from agot.nn.model import PolicyValueNet
    from agot.training.parallel import SelfPlayPlan, parallel_self_play

    model = PolicyValueNet(hidden=16).eval()
    plan = SelfPlayPlan(
        games=2, max_rounds=2, simulations=4, workers=1, batch_games=2, seed=0
    )
    records = parallel_self_play(model, plan)
    assert len(records) == 2
    assert all(record.examples for record in records)


def test_parallel_self_play_against_scripted_opponents() -> None:
    pytest.importorskip("torch")
    from agot.nn.model import PolicyValueNet
    from agot.training.parallel import SelfPlayPlan, parallel_self_play

    model = PolicyValueNet(hidden=16).eval()
    plan = SelfPlayPlan(
        games=2,
        max_rounds=2,
        simulations=4,
        workers=1,
        batch_games=2,
        opponent_rate=0.5,
        seed=0,
    )
    records = parallel_self_play(model, plan)
    assert len(records) == 2


def test_worker_count_respects_the_plan_and_the_machine() -> None:
    from agot.training.parallel import SelfPlayPlan, worker_count

    assert worker_count(SelfPlayPlan(workers=3), games=10) == 3
    # Never more workers than games, and never zero.
    assert worker_count(SelfPlayPlan(workers=0), games=1) == 1
    assert worker_count(SelfPlayPlan(workers=8), games=5) == 5
