"""Speed levers of the optimizer pass that must not change what is learned."""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agot.engine.observation import ACTION_DIM, N_HOUSES, STATE_DIM  # noqa: E402
from agot.nn.model import PolicyValueNet  # noqa: E402
from agot.training.replay import Example, ReplayBuffer  # noqa: E402
from agot.training.trainer import (  # noqa: E402
    PackedBatch,
    TrainConfig,
    Trainer,
    collate_packed,
    flush_denormals,
    scrub_subnormals,
)

TINY = torch.finfo(torch.float32).tiny


def _random_examples(count: int, seed: int = 0, max_actions: int = 20) -> list[Example]:
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(count):
        actions = int(rng.integers(1, max_actions + 1))
        policy = rng.random(actions).astype(np.float32)
        if rng.random() < 0.1:
            policy[:] = 0.0  # exercises the uniform fallback
        outcome = rng.choice([-1.0, 0.0, 0.5, 1.0], size=N_HOUSES).astype(np.float32)
        examples.append(
            Example(
                state=rng.standard_normal(STATE_DIM).astype(np.float32),
                actions=(rng.random((actions, ACTION_DIM)) < 0.1).astype(np.float32),
                policy=policy,
                player=int(rng.integers(0, N_HOUSES)),
                outcome=outcome,
            )
        )
    return examples


def _subnormal_count(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        int(((value != 0) & (value.abs() < TINY)).sum())
        for state in optimizer.state.values()
        for key, value in state.items()
        if key != "step" and torch.is_tensor(value)
    )


def _poison(optimizer: torch.optim.Optimizer, seed: int = 1) -> None:
    """Write subnormals into most of the moments, as a long run leaves them."""
    generator = torch.Generator().manual_seed(seed)
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            mask = torch.rand(value.shape, generator=generator) < 0.7
            sign = 1.0 if key == "exp_avg_sq" else -1.0
            value[mask] = sign * 3e-40


def test_scrub_zeroes_only_subnormals() -> None:
    model = PolicyValueNet(hidden=8)
    trainer = Trainer(model, TrainConfig(batch_size=4))
    trainer.train_epoch(_random_examples(8))
    state = next(iter(trainer.optimizer.state.values()))
    state["exp_avg"].view(-1)[:3] = torch.tensor([1e-40, -2e-39, 1e-30])
    before = {key: value.clone() for key, value in state.items()}
    assert scrub_subnormals(trainer.optimizer) == 2
    assert state["exp_avg"].view(-1)[:3].tolist() == [0.0, 0.0, pytest.approx(1e-30)]
    keep = before["exp_avg"].view(-1)[2:]
    assert torch.equal(state["exp_avg"].view(-1)[2:], keep)
    assert torch.equal(state["exp_avg_sq"], before["exp_avg_sq"])
    assert torch.equal(state["step"], before["step"])


def test_flush_is_scoped_to_the_block() -> None:
    small = torch.tensor([1e-30])
    with flush_denormals():
        pass
    # Outside the block IEEE subnormals are back (inference must not change).
    assert float(small * 1e-9) != 0.0


def test_subnormal_state_trains_like_the_plain_path() -> None:
    """The flushed, scrubbed epoch matches the IEEE epoch to round-off."""
    examples = _random_examples(96, seed=3)
    torch.manual_seed(0)
    model = PolicyValueNet(hidden=16, value_context=True)
    trainer = Trainer(model, TrainConfig(batch_size=8))
    trainer.train_epoch(examples)  # creates the moments
    _poison(trainer.optimizer)
    reference = copy.deepcopy(trainer)
    assert _subnormal_count(reference.optimizer) > 0

    trainer.train_epoch(examples)
    reference._train_epoch(examples)  # IEEE arithmetic, no scrub

    assert _subnormal_count(trainer.optimizer) == 0
    for (name, fast), (_, slow) in zip(
        trainer.model.named_parameters(), reference.model.named_parameters(), strict=True
    ):
        assert torch.allclose(fast, slow, rtol=0, atol=1e-6), name


PACKED_FIELDS = ("state", "actions", "index", "policy", "outcome", "outcome_mask")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_buffer_gather_matches_per_example_collate(seed: int) -> None:
    """`ReplayBuffer.packed_batch` is byte-identical to `collate_packed`."""
    rng = np.random.default_rng(seed)
    # Segments of 1..300 candidates cover numpy's sequential (<8), blocked
    # (8..128) and recursive (>128) summation paths, plus zero-sum targets.
    examples = _random_examples(150, seed=seed, max_actions=300)
    buffer = ReplayBuffer(capacity=1000)
    buffer.extend(examples)
    for _ in range(20):
        indices = rng.choice(len(buffer), size=int(rng.integers(1, 65)), replace=False)
        old = collate_packed([buffer[int(index)] for index in indices])
        new = PackedBatch.from_arrays(*buffer.packed_batch(indices))
        assert new.rows == old.rows
        for name in PACKED_FIELDS:
            a, b = getattr(old, name), getattr(new, name)
            assert a.dtype == b.dtype and a.shape == b.shape, name
            assert a.numpy().tobytes() == b.numpy().tobytes(), name


def test_epoch_over_buffer_matches_epoch_over_list() -> None:
    """Training from the buffer and from a list of its examples is identical."""
    examples = _random_examples(70, seed=5)
    buffer = ReplayBuffer(capacity=1000)
    buffer.extend(examples)
    torch.manual_seed(0)
    model = PolicyValueNet(hidden=16, value_context=True)
    fast = Trainer(model, TrainConfig(batch_size=8))
    slow = copy.deepcopy(fast)
    fast_stats = fast.train_epoch(buffer).as_dict()
    slow_stats = slow.train_epoch([buffer[index] for index in range(len(buffer))]).as_dict()
    assert fast_stats == slow_stats
    for (name, a), (_, b) in zip(
        fast.model.named_parameters(), slow.model.named_parameters(), strict=True
    ):
        assert torch.equal(a, b), name


def _plain_trainer(model: PolicyValueNet, config: TrainConfig) -> Trainer:
    """A trainer on the unfused AdamW the code used before the fused kernel."""
    from agot.training.trainer import _parameter_groups

    trainer = Trainer(model, config)
    trainer._optimizer = torch.optim.AdamW(
        _parameter_groups(model, config.weight_decay), lr=config.learning_rate
    )
    return trainer


def test_optimizer_state_resumes_across_kernels() -> None:
    """Old (unfused) states resume on the new trainer and the reverse."""
    examples = _random_examples(48, seed=7)
    config = TrainConfig(batch_size=8)
    torch.manual_seed(0)
    plain = _plain_trainer(PolicyValueNet(hidden=16, value_context=True), config)
    plain.train_epoch(examples)
    old_state = copy.deepcopy(plain.state_dict())

    current = Trainer(copy.deepcopy(plain.model), config)
    fused = [group.get("fused") for group in current.optimizer.param_groups]
    current.load_state_dict(copy.deepcopy(old_state))
    assert [group.get("fused") for group in current.optimizer.param_groups] == fused

    plain.train_epoch(examples)
    current.train_epoch(examples)
    for (name, a), (_, b) in zip(
        plain.model.named_parameters(), current.model.named_parameters(), strict=True
    ):
        assert torch.allclose(a, b, rtol=0, atol=1e-6), name

    # The saved groups carry no kernel choice, so the unfused code resumes them.
    saved = current.state_dict()
    assert all(group.get("fused") is None for group in saved["optimizer"]["param_groups"])
    again = _plain_trainer(copy.deepcopy(current.model), config)
    again.load_state_dict(copy.deepcopy(saved))
    assert all(not group.get("fused") for group in again.optimizer.param_groups)
    again.train_epoch(examples)
    current.train_epoch(examples)
    for (name, a), (_, b) in zip(
        again.model.named_parameters(), current.model.named_parameters(), strict=True
    ):
        assert torch.allclose(a, b, rtol=0, atol=1e-6), name


def _packed_loss_grads(model: PolicyValueNet, batch: PackedBatch) -> dict[str, torch.Tensor]:
    from agot.nn.model import segment_log_softmax

    model.zero_grad(set_to_none=True)
    logits, values = model.forward_packed(batch.state, batch.actions, batch.index)
    log_probs = segment_log_softmax(logits, batch.index, batch.rows)
    policy = -(batch.policy * log_probs).sum() / batch.rows
    mask = batch.outcome_mask
    value = (((values - batch.outcome) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
    (policy + value).backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    grads["logits"] = logits.detach().clone()
    grads["values"] = values.detach().clone()
    return grads


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_live_column_training_forward_matches_dense(seed: int) -> None:
    """Skipping all-zero ReLU columns changes outputs and gradients by round-off only."""
    torch.manual_seed(seed)
    model = PolicyValueNet(hidden=64, value_context=True)
    with torch.no_grad():
        # Kill most action-trunk units, as a trained net does (70-76 % on run7).
        for layer in (model.action_trunk[0], model.action_trunk[2]):
            layer.bias[: 48] = -1e3
    buffer = ReplayBuffer(capacity=1000)
    buffer.extend(_random_examples(64, seed=seed, max_actions=30))
    batch = PackedBatch.from_arrays(*buffer.packed_batch(np.arange(64)))

    model.live_columns = False
    dense = _packed_loss_grads(model, batch)
    model.live_columns = True
    live = _packed_loss_grads(model, batch)
    for name, reference in dense.items():
        scale = float(reference.abs().max())
        error = float((live[name] - reference).abs().max())
        assert error <= 1e-5 * max(scale, 1e-3), (name, error, scale)
    # Dead columns get exactly-zero weight gradients on both paths.
    dead = slice(0, 48)
    weight = "action_trunk.2.weight"
    assert torch.equal(live[weight][:, dead], dense[weight][:, dead])
    assert not live[weight][:, dead].any()
