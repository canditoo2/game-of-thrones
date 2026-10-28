"""Tests for resumable training state and the run log.

Skipped automatically when torch is absent, like the rest of the RL suite.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agot.engine.observation import ACTION_DIM, N_HOUSES, STATE_DIM  # noqa: E402
from agot.nn.model import PolicyValueNet  # noqa: E402
from agot.training.checkpoint import (  # noqa: E402
    RunLog,
    RunState,
    log_path,
    state_path,
)
from agot.training.replay import Example  # noqa: E402
from agot.training.trainer import TrainConfig, Trainer  # noqa: E402


def _examples(count: int = 4) -> list[Example]:
    rng = np.random.default_rng(0)
    examples = []
    for _ in range(count):
        actions = 3
        outcome = np.zeros(N_HOUSES, dtype=np.float32)
        outcome[0] = 1.0
        outcome[1:] = -1.0
        examples.append(
            Example(
                state=rng.standard_normal(STATE_DIM).astype(np.float32),
                actions=rng.standard_normal((actions, ACTION_DIM)).astype(np.float32),
                policy=np.full(actions, 1.0 / actions, dtype=np.float32),
                player=0,
                outcome=outcome,
            )
        )
    return examples


def test_state_and_log_paths_are_companions() -> None:
    assert state_path("ckpt/agot.pt").name == "agot.state.pt"
    assert log_path("ckpt/agot.pt").name == "agot.log.jsonl"


def test_run_state_round_trips(tmp_path) -> None:
    state = RunState(iteration=7, best_rate=0.42, trainer={"optimizer": {}, "rng": {}})
    path = tmp_path / "run.state.pt"
    state.save(path)

    loaded = RunState.load(path)
    assert loaded is not None
    assert loaded.iteration == 7
    assert loaded.best_rate == pytest.approx(0.42)
    assert loaded.trainer == {"optimizer": {}, "rng": {}}


def test_run_state_load_missing_returns_none(tmp_path) -> None:
    assert RunState.load(tmp_path / "absent.state.pt") is None


def test_run_state_gate_failures_round_trip(tmp_path) -> None:
    path = tmp_path / "run.state.pt"
    RunState(iteration=3, gate_failures=2).save(path)

    loaded = RunState.load(path)
    assert loaded is not None
    assert loaded.gate_failures == 2


def test_run_state_without_gate_failures_loads_zero(tmp_path) -> None:
    # State files written before the counter existed must still resume.
    path = tmp_path / "old.state.pt"
    torch.save({"iteration": 4, "best_rate": 0.1, "trainer": {}}, path)

    loaded = RunState.load(path)
    assert loaded is not None
    assert loaded.gate_failures == 0


def test_run_log_appends_records(tmp_path) -> None:
    log = RunLog(tmp_path / "run.log.jsonl")
    log.append({"iteration": 1, "eval": 0.1})
    log.append({"iteration": 2, "eval": 0.2})
    assert [record["iteration"] for record in log.records()] == [1, 2]


def test_trainer_state_restores_optimizer_and_rng() -> None:
    examples = _examples()
    model = PolicyValueNet(hidden=8)
    trainer = Trainer(model, TrainConfig(epochs=1, batch_size=2))
    trainer.train_epoch(examples)
    state = trainer.state_dict()

    clone = Trainer(PolicyValueNet(hidden=8), TrainConfig(epochs=1, batch_size=2))
    clone.load_state_dict(state)

    assert clone.rng.bit_generator.state == trainer.rng.bit_generator.state
    assert set(clone.optimizer.state_dict()["state"]) == set(
        trainer.optimizer.state_dict()["state"]
    )
