"""The human value-only dataset and trainer step (needs torch)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agot.engine.observation import ACTION_DIM, HOUSE_ORDER, STATE_DIM  # noqa: E402
from agot.nn.model import PolicyValueNet  # noqa: E402
from agot.training.human_value import HumanValueSet  # noqa: E402
from agot.training.trainer import TrainConfig, Trainer  # noqa: E402


def make_set(count: int = 6) -> HumanValueSet:
    rows = 2
    rng = np.random.default_rng(0)
    states = rng.normal(size=(count, STATE_DIM)).astype(np.float32)
    actions = rng.normal(size=(count * rows, ACTION_DIM)).astype(np.float32)
    offsets = np.arange(0, (count + 1) * rows, rows, dtype=np.int64)
    outcomes = np.full((count, len(HOUSE_ORDER)), -1.0, dtype=np.float32)
    for row in range(count):
        outcomes[row, row % len(HOUSE_ORDER)] = 1.0
    return HumanValueSet(
        states=states,
        actions=actions,
        offsets=offsets,
        outcomes=outcomes,
        games=np.array([f"game-{row // 3}" for row in range(count)]),
        rounds=np.arange(count, dtype=np.int16) + 1,
    )


def test_batch_regroups_the_concatenated_actions():
    dataset = make_set()
    state, actions, index, outcome = dataset.batch(np.array([0, 2]))
    assert state.shape == (2, STATE_DIM)
    assert actions.shape == (4, ACTION_DIM)
    assert index.tolist() == [0, 0, 1, 1]
    assert outcome.shape == (2, len(HOUSE_ORDER))
    assert outcome[0, 0] == 1.0 and outcome[0, 1] == -1.0


def test_round_trip(tmp_path):
    dataset = make_set()
    path = tmp_path / "values.npz"
    np.savez_compressed(
        path,
        states=dataset.states,
        actions=dataset.actions,
        offsets=dataset.offsets,
        outcomes=dataset.outcomes,
        games=dataset.games,
        rounds=dataset.rounds,
    )
    loaded = HumanValueSet.load(path)
    assert len(loaded) == len(dataset)
    assert np.array_equal(loaded.offsets, dataset.offsets)
    assert np.array_equal(loaded.games, dataset.games)


def test_value_only_step_restores_the_learning_rate():
    dataset = make_set()
    model = PolicyValueNet()
    trainer = Trainer(model, TrainConfig(batch_size=3, device="cpu"))
    before = [group["lr"] for group in trainer.optimizer.param_groups]
    stats = trainer.train_value_only(dataset, lr_scale=0.5, epochs=2)
    assert stats["batches"] == 4
    assert np.isfinite(stats["value_loss"])
    after = [group["lr"] for group in trainer.optimizer.param_groups]
    assert after == before
