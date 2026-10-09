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


def test_batched_inference_agrees_with_one_leaf_at_a_time() -> None:
    """``predict_many`` must not let a leaf's distribution depend on its batch.

    The batched path concatenates every leaf's candidates and normalises each
    leaf's slice on its own, so a leaf that shares a batch with a much wider one
    must score exactly as it would alone.
    """
    torch.manual_seed(0)
    model = PolicyValueNet(hidden=32)
    model.eval()

    requests = []
    for seed in (0, 1, 2):
        game = Game.new(GameConfig(seed=seed, max_rounds=10))
        for _ in range(12):
            actions = game.legal_actions()
            if game.is_terminal() or not actions:
                break
            requests.append((game.clone(), actions))
            game.step(actions[0])
    assert len(requests) >= 5
    widths = {len(actions) for _, actions in requests}
    assert len(widths) > 1, "the batch must mix action-set sizes to mean anything"

    batched = model.predict_many(requests)
    assert len(batched) == len(requests)
    for (game, actions), (probs, values) in zip(requests, batched, strict=True):
        _, alone, alone_values = model.predict_actions(game, actions)
        assert probs.shape == (len(actions),)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)
        np.testing.assert_allclose(probs, alone, atol=1e-6)
        np.testing.assert_allclose(values, alone_values, atol=1e-6)


# ---------------------------------------------------------------------------
# Policy-head health
#
# Regression cover for the collapse described in docs/roadmap.md: the old head
# scored an action by the dot product h . a_i, so once either factor was driven
# to zero the gradient through the other vanished and the head emitted a uniform
# distribution forever.
# ---------------------------------------------------------------------------


def _learnable_examples(size: int = 8) -> list[Example]:
    """Real encodings with a synthetic target that the features determine.

    The target is a softmax of a fixed random linear function of the *action*
    features, so it is learnable by construction -- unlike a search target, which
    at low simulation counts is close to noise.
    """
    from agot.engine.observation import observe

    game = Game.new(GameConfig(seed=11))
    observation = observe(game)
    rng = np.random.default_rng(0)
    # The scale makes the softmax clearly non-uniform, so the target carries real
    # signal. A near-uniform target has no headroom and proves nothing.
    weight = (
        rng.standard_normal(ACTION_DIM).astype(np.float32) / np.sqrt(ACTION_DIM) * 12.0
    )
    scores = observation.action_features @ weight
    shifted = scores - scores.max()
    target = np.exp(shifted)
    target /= target.sum()

    outcome = np.zeros(N_HOUSES, dtype=np.float32)
    outcome[0] = 1.0
    outcome[1:] = -1.0
    return [
        Example(
            state=observation.state_features,
            actions=observation.action_features,
            policy=target.astype(np.float32),
            player=0,
            outcome=outcome,
        )
        for _ in range(size)
    ]


def _cross_entropy(model: PolicyValueNet, batch) -> float:
    model.eval()
    with torch.no_grad():
        logits, _ = model(batch.state, batch.actions, batch.mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        return float(-(batch.policy * log_probs).sum(dim=-1).mean())


def test_policy_logits_vary_across_actions() -> None:
    """A fresh head must not score every candidate identically."""
    model = PolicyValueNet(hidden=32)
    batch = collate(_learnable_examples())
    logits, _ = model(batch.state, batch.actions, batch.mask)
    assert logits[batch.mask > 0].std() > 1e-4


def test_every_parameter_receives_gradient() -> None:
    """No parameter may sit at zero gradient, which is how the head died."""
    model = PolicyValueNet(hidden=32)
    batch = collate(_learnable_examples())
    logits, values = model(batch.state, batch.actions, batch.mask)
    log_probs = torch.log_softmax(logits, dim=-1)
    policy_loss = -(batch.policy * log_probs).sum(dim=-1).mean()
    mask = batch.outcome_mask
    value_loss = ((values - batch.outcome) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
    (policy_loss + value_loss).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert parameter.grad.norm() > 0, f"{name} received a zero gradient"


def test_policy_head_fits_a_learnable_target() -> None:
    """The head must approach the best achievable loss for a learnable target.

    The floor is the target's own entropy: no model can do better than reproduce
    the target. Reaching it shows the head and its gradients work end to end.
    """
    examples = _learnable_examples(size=24)
    model = PolicyValueNet(hidden=32)
    trainer = Trainer(model, TrainConfig(learning_rate=1e-2, batch_size=8, epochs=1))
    batch = collate(examples)

    floor = float(
        -(batch.policy * torch.log(batch.policy.clamp(min=1e-12))).sum(dim=1).mean()
    )
    start = _cross_entropy(model, batch)
    for _ in range(300):
        trainer.train_epoch(examples)
    end = _cross_entropy(model, batch)

    assert end < start - 0.1, f"no learning at all: {start:.3f} -> {end:.3f}"
    assert end < floor + 0.05, f"fell short of the floor {floor:.3f}: got {end:.3f}"


def test_load_rejects_the_collapsed_old_checkpoint(tmp_path) -> None:
    path = tmp_path / "old.pt"
    torch.save(
        {"state_dict": {"policy_proj.weight": torch.zeros(4, 4)}, "config": {}},
        path,
    )
    with pytest.raises(ValueError, match="dot-product policy head"):
        PolicyValueNet.load(str(path))
