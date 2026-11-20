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
