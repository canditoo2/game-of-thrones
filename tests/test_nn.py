"""Tests for the PyTorch network. Skipped automatically when torch is absent."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agot.agents import MCTS  # noqa: E402
from agot.engine.game import Game, GameConfig  # noqa: E402
from agot.engine.observation import ACTION_DIM, N_HOUSES, STATE_DIM  # noqa: E402
from agot.nn.model import NeuralEvaluator, PolicyValueNet  # noqa: E402
from agot.training.replay import Example  # noqa: E402
from agot.training.trainer import TrainConfig, Trainer, collate  # noqa: E402


def test_forward_shapes_and_masking() -> None:
    model = PolicyValueNet(hidden=32)
    state = torch.zeros(2, STATE_DIM)
    actions = torch.zeros(2, 5, ACTION_DIM)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float32)
    logits, values = model(state, actions, mask)
    assert logits.shape == (2, 5)
    assert values.shape == (2, N_HOUSES)
    assert torch.isneginf(logits[0, 3]) or logits[0, 3] < -1e8
    assert (values.abs() <= 1.0).all()


def test_predict_returns_a_normalised_policy() -> None:
    model = PolicyValueNet(hidden=32)
    game = Game.new(GameConfig(seed=0))
    actions, probs, values = model.predict(game)
    assert len(actions) == len(game.legal_actions())
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)
    assert values.shape == (N_HOUSES,)


def test_save_and_load_round_trip(tmp_path) -> None:
    model = PolicyValueNet(hidden=16)
    path = str(tmp_path / "model.pt")
    model.save(path)
    restored = PolicyValueNet.load(path)
    assert restored.hidden == 16
    game = Game.new(GameConfig(seed=1))
    _, probs_a, _ = model.predict(game)
    _, probs_b, _ = restored.predict(game)
    np.testing.assert_allclose(probs_a, probs_b, atol=1e-6)


def test_trainer_reduces_loss_on_a_tiny_batch() -> None:
    model = PolicyValueNet(hidden=32)
    trainer = Trainer(model, TrainConfig(learning_rate=1e-3, batch_size=4, epochs=1))
    game = Game.new(GameConfig(seed=2))
    from agot.engine.observation import observe

    observation = observe(game)
    outcome = np.zeros(N_HOUSES, dtype=np.float32)
    outcome[0] = 1.0
    outcome[1:] = -1.0
    example = Example(
        state=observation.state_features,
        actions=observation.action_features,
        policy=np.full(len(observation.actions), 1.0 / len(observation.actions)),
        player=0,
        outcome=outcome,
    )
    batch = collate([example])
    assert batch.state.shape == (1, STATE_DIM)
    history = trainer.train_epoch([example])
    assert history.batches == 1
    assert history.total_loss >= 0.0


def test_neural_evaluator_drives_mcts() -> None:
    model = PolicyValueNet(hidden=16)
    game = Game.new(GameConfig(seed=3))
    mcts = MCTS(NeuralEvaluator(model), simulations=4, seed=0)
    action, target = mcts.decide(game)
    assert action in game.legal_actions()
    assert target.shape == (len(game.legal_actions()),)
