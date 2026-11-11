"""Tests for the optional per-candidate Q head and the planning teacher.

Torch-only: the rest of the suite still runs on a machine without it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agot.engine.enums import Phase  # noqa: E402
from agot.engine.observation import ACTION_DIM, N_HOUSES, STATE_DIM  # noqa: E402
from agot.nn.model import PolicyValueNet, segment_log_softmax  # noqa: E402
from agot.nn.teacher import (  # noqa: E402
    PLANNING_SMOOTHING,
    planning_target,
    soft_teacher_target,
)
from agot.training.replay import Example  # noqa: E402
from agot.training.trainer import (  # noqa: E402
    TrainConfig,
    Trainer,
    _gather_chosen,
    collate,
    collate_packed,
)


def _examples(
    count: int = 6,
    *,
    actions: int = 4,
    chosen: list[int] | None = None,
    seed: int = 0,
) -> list[Example]:
    """Synthetic examples; no encoder or engine needed."""
    rng = np.random.default_rng(seed)
    outcome = np.zeros(N_HOUSES, dtype=np.float32)
    outcome[0] = 1.0
    outcome[1] = -1.0
    examples = []
    for row in range(count):
        examples.append(
            Example(
                state=rng.standard_normal(STATE_DIM).astype(np.float32),
                actions=rng.standard_normal((actions, ACTION_DIM)).astype(np.float32),
                policy=np.full(actions, 1.0 / actions, dtype=np.float32),
                player=0,
                outcome=outcome.copy(),
                chosen=None if chosen is None else chosen[row % len(chosen)],
            )
        )
    return examples


# ---------------------------------------------------------------------------
# The Q head
# ---------------------------------------------------------------------------


def test_default_model_keeps_the_two_tuple_forward() -> None:
    model = PolicyValueNet(hidden=16)
    state = torch.zeros(2, STATE_DIM)
    actions = torch.zeros(2, 3, ACTION_DIM)
    assert len(model(state, actions)) == 2
    index = torch.tensor([0, 0, 0, 1, 1, 1])
    flat = torch.zeros(6, ACTION_DIM)
    assert len(model.forward_packed(state, flat, index)) == 2


def test_q_head_forward_shapes() -> None:
    model = PolicyValueNet(hidden=32, q_head=True)
    state = torch.randn(2, STATE_DIM)
    actions = torch.randn(2, 5, ACTION_DIM)
    logits, values, q_values = model(state, actions)
    assert logits.shape == (2, 5)
    assert values.shape == (2, N_HOUSES)
    assert q_values.shape == (2, 5, N_HOUSES)
    assert (q_values.abs() <= 1.0).all()

    index = torch.tensor([0, 0, 0, 1, 1])
    flat = torch.cat([actions[0, :3], actions[1, :2]], dim=0)
    flat_logits, flat_values, flat_q = model.forward_packed(state, flat, index)
    assert flat_logits.shape == (5,)
    assert flat_values.shape == (2, N_HOUSES)
    assert flat_q.shape == (5, N_HOUSES)
    # The packed and padded forms must agree per candidate.
    np.testing.assert_allclose(
        flat_q[:3].detach().numpy(),
        q_values[0, :3].detach().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        flat_q[3:].detach().numpy(),
        q_values[1, :2].detach().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_q_head_parameters_and_shared_trunk_receive_gradient() -> None:
    model = PolicyValueNet(hidden=16, q_head=True)
    batch = collate_packed(_examples(chosen=[0, 1, 2, 3, 0, 1]))
    trainer = Trainer(model, TrainConfig(batch_size=4))

    _, _, q_loss, _, _ = trainer._losses(batch)
    assert q_loss is not None
    q_loss.backward()

    for name, parameter in model.q_value_head.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert parameter.grad.norm() > 0, f"{name} received a zero gradient"
    # The Q head reads [h, a], so both shared trunks must learn from it too.
    assert model.state_trunk[0].weight.grad.norm() > 0
    assert model.action_trunk[0].weight.grad.norm() > 0


def test_gather_chosen_reads_one_candidate_per_row() -> None:
    examples = _examples(chosen=[0, 3, 1, 2, 0, 3])
    packed = collate_packed(examples)
    model = PolicyValueNet(hidden=8, q_head=True)
    with torch.no_grad():
        q_values = model.forward_packed(packed.state, packed.actions, packed.index)[2]
    gathered = _gather_chosen(q_values, packed)

    for row, example in enumerate(examples):
        start = int(packed.index.tolist().index(row))
        np.testing.assert_allclose(
            gathered[row].detach().numpy(),
            q_values[start + example.chosen].detach().numpy(),
            rtol=0,
            atol=0,
        )

    padded = collate(examples)
    with torch.no_grad():
        padded_q = model(padded.state, padded.actions, padded.mask)[2]
    padded_gathered = _gather_chosen(padded_q, padded)
    np.testing.assert_allclose(
        padded_gathered.detach().numpy(), gathered.detach().numpy(), atol=1e-6
    )


def test_q_loss_only_reads_the_sampled_action() -> None:
    """The same position with a different played index gets a different loss."""
    base = _examples(count=1, chosen=[0], seed=4)[0]

    def with_chosen(index: int) -> Example:
        return Example(
            state=base.state,
            actions=base.actions,
            policy=base.policy,
            player=base.player,
            outcome=base.outcome,
            chosen=index,
        )

    model = PolicyValueNet(hidden=16, q_head=True)
    model.eval()
    trainer = Trainer(model, TrainConfig(batch_size=1))

    batch0 = collate_packed([with_chosen(0)])
    batch3 = collate_packed([with_chosen(3)])
    q0 = trainer._losses(batch0)[2]
    q3 = trainer._losses(batch3)[2]
    assert q0 is not None and q3 is not None
    assert not torch.isclose(q0, q3)

    with torch.no_grad():
        q_values = model.forward_packed(batch0.state, batch0.actions, batch0.index)[2]
        gathered = _gather_chosen(q_values, batch0)
        mask = batch0.outcome_mask
        expected = ((gathered - batch0.outcome) ** 2 * mask).sum() / mask.sum()
    assert torch.allclose(q0, expected)


def test_q_loss_requires_both_the_head_and_the_indices() -> None:
    plain = PolicyValueNet(hidden=8)
    plain_trainer = Trainer(plain, TrainConfig(batch_size=4))
    with_chosen = collate_packed(_examples(chosen=[0, 1, 2, 3, 0, 1]))
    assert plain_trainer._losses(with_chosen)[2] is None

    headed = PolicyValueNet(hidden=8, q_head=True)
    headed_trainer = Trainer(headed, TrainConfig(batch_size=4))
    without_chosen = collate_packed(_examples())
    assert headed_trainer._losses(without_chosen)[2] is None


def test_default_off_loss_is_bit_identical_to_the_old_formula() -> None:
    """No Q head and no chosen indices must reproduce today's arithmetic."""
    torch.manual_seed(0)
    model = PolicyValueNet(hidden=16)
    model.eval()
    batch = collate_packed(_examples())
    assert batch.chosen is None

    trainer = Trainer(model, TrainConfig(batch_size=4, value_weight=1.0))
    policy_loss, value_loss, q_loss, aux_loss, total = trainer._losses(batch)
    assert q_loss is None
    assert aux_loss is None

    with torch.no_grad():
        logits, values = model.forward_packed(batch.state, batch.actions, batch.index)
        log_probs = segment_log_softmax(logits, batch.index, batch.rows)
        expected_policy = -(batch.policy * log_probs).sum() / batch.rows
        mask = batch.outcome_mask
        expected_value = ((values - batch.outcome) ** 2 * mask).sum() / mask.sum()
        expected_total = expected_policy + 1.0 * expected_value

    assert torch.equal(policy_loss, expected_policy)
    assert torch.equal(value_loss, expected_value)
    assert torch.equal(total, expected_total)


def test_checkpoint_config_round_trips_the_q_head(tmp_path) -> None:
    path = str(tmp_path / "q.pt")
    PolicyValueNet(hidden=8, q_head=True).save(path)
    restored = PolicyValueNet.load(path)
    assert restored.q_head is True

    # A checkpoint written before the option existed loads with the head off.
    legacy = str(tmp_path / "legacy.pt")
    model = PolicyValueNet(hidden=8)
    config = {
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "hidden": model.hidden,
        "trunk_layers": model.trunk_layers,
        "n_houses": model.n_houses,
        "value_context": model.value_context,
    }
    torch.save({"state_dict": model.state_dict(), "config": config}, legacy)
    assert PolicyValueNet.load(legacy).q_head is False


# ---------------------------------------------------------------------------
# The planning teacher
# ---------------------------------------------------------------------------


def test_teacher_target_is_a_smoothed_distribution() -> None:
    scores = np.array([4.0, 2.0, 0.0, -1.0])
    target = soft_teacher_target(scores)

    assert target.shape == scores.shape
    np.testing.assert_allclose(target.sum(), 1.0, atol=1e-6)
    assert target.argmax() == 0
    assert target.min() >= PLANNING_SMOOTHING / len(scores) - 1e-7
    # The greedy best still dominates by a wide margin.
    assert target[0] > 0.8

    shifted = np.exp(scores - scores.max())
    manual = (1.0 - PLANNING_SMOOTHING) * shifted / shifted.sum()
    manual = manual + PLANNING_SMOOTHING / len(scores)
    np.testing.assert_allclose(target, manual, atol=1e-6)


def test_teacher_target_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        soft_teacher_target([])
    with pytest.raises(ValueError):
        soft_teacher_target([1.0, np.nan])
    with pytest.raises(ValueError):
        soft_teacher_target([1.0, 2.0], smoothing=1.0)
    with pytest.raises(ValueError):
        soft_teacher_target([1.0, 2.0], temperature=0.0)


def test_planning_target_only_fires_in_the_planning_phase() -> None:
    class _Policy:
        def scores(self, game, actions):
            return np.array([3.0, 1.0, 0.0])

    game = SimpleNamespace(
        state=SimpleNamespace(phase=Phase.PLANNING),
        legal_actions=lambda: ["a", "b", "c"],
    )
    target = planning_target(game, policy=_Policy())
    assert target is not None
    np.testing.assert_allclose(target.sum(), 1.0, atol=1e-6)

    game.state.phase = Phase.ACTION
    assert planning_target(game, policy=_Policy()) is None


def test_trained_q_head_spreads_candidates_on_random_data() -> None:
    """Sanity, not strength: the head must rank candidates after training.

    The situation this exists for is a planning node whose children encode to
    identical state vectors. A trained Q head must still give them different
    values, otherwise PUCT would rank them by the prior alone.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(7)
    examples = _examples(count=16, actions=5, chosen=[0, 1, 2, 3, 4], seed=3)
    for example in examples:
        outcome = np.zeros(N_HOUSES, dtype=np.float32)
        winner = int(rng.integers(0, N_HOUSES))
        outcome[:] = -1.0
        outcome[winner] = 1.0
        example.outcome = outcome

    model = PolicyValueNet(hidden=32, q_head=True)
    trainer = Trainer(model, TrainConfig(learning_rate=1e-2, batch_size=8))
    batch = collate_packed(examples)
    before = trainer._losses(batch)[2]
    assert before is not None

    for _ in range(40):
        trainer.train_epoch(examples)

    model.eval()
    after = trainer._losses(batch)[2]
    assert after is not None
    assert float(after.detach()) < float(before.detach())

    with torch.no_grad():
        q_values = model.forward_packed(batch.state, batch.actions, batch.index)[2]
    # Within each example the candidates must no longer be interchangeable.
    spread = q_values.reshape(batch.rows, -1, N_HOUSES).std(dim=1).mean()
    assert float(spread) > 1e-3

    # And the two losses stay distinct terms in the stats.
    stats = trainer.train_epoch(examples)
    assert stats.q_loss > 0.0
