"""The replay snapshot: background writes, atomic replace, identical arrays."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from agot.engine.observation import ACTION_DIM, N_HOUSES, STATE_DIM
from agot.training import replay as replay_module
from agot.training.replay import Example, ReplayBuffer

NAMES = ("states", "actions", "offsets", "policies", "players", "outcomes")


def _examples(count: int, seed: int, max_actions: int = 9) -> list[Example]:
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(count):
        width = int(rng.integers(1, max_actions + 1))
        examples.append(
            Example(
                state=rng.standard_normal(STATE_DIM).astype(np.float32),
                actions=rng.standard_normal((width, ACTION_DIM)).astype(np.float32),
                policy=rng.random(width).astype(np.float32),
                player=int(rng.integers(0, N_HOUSES)),
                outcome=rng.standard_normal(N_HOUSES).astype(np.float32),
            )
        )
    return examples


def _arrays(path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        assert sorted(data.files) == sorted(NAMES)
        return {name: data[name] for name in NAMES}


def _assert_same(left: dict, right: dict) -> None:
    for name in NAMES:
        assert left[name].dtype == right[name].dtype, name
        assert left[name].shape == right[name].shape, name
        assert left[name].tobytes() == right[name].tobytes(), name


def _leftovers(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if ".tmp" in path.name)


@pytest.mark.parametrize("capacity", [1000, 40])
def test_background_snapshot_writes_the_synchronous_arrays(tmp_path, capacity) -> None:
    buffer = ReplayBuffer(capacity=capacity)
    buffer.extend(_examples(90, seed=1))  # capacity 40 also evicts
    buffer.save(str(tmp_path / "sync.npz"))
    buffer.save_async(str(tmp_path / "async.npz"))
    assert buffer.save_pending
    buffer.wait_save()
    assert not buffer.save_pending
    assert buffer.last_save_seconds is not None
    _assert_same(_arrays(tmp_path / "sync.npz"), _arrays(tmp_path / "async.npz"))
    assert _leftovers(tmp_path) == []
    # And what a resume restores from it is the same buffer.
    restored = ReplayBuffer.load(str(tmp_path / "async.npz"), capacity=capacity)
    reference = ReplayBuffer.load(str(tmp_path / "sync.npz"), capacity=capacity)
    for name in ("_states", "_actions", "_policies", "_players", "_outcomes", "_offsets"):
        assert getattr(restored, name).tobytes() == getattr(reference, name).tobytes()


def test_empty_buffer_snapshot_round_trips(tmp_path) -> None:
    buffer = ReplayBuffer(capacity=10)
    buffer.save(str(tmp_path / "sync.npz"))
    buffer.save_async(str(tmp_path / "async.npz"))
    buffer.wait_save()
    _assert_same(_arrays(tmp_path / "sync.npz"), _arrays(tmp_path / "async.npz"))


def test_snapshot_keeps_the_npz_suffix_rule(tmp_path) -> None:
    # `np.savez_compressed(name)` appended `.npz`; the handle-based writer must too.
    buffer = ReplayBuffer(capacity=10)
    buffer.extend(_examples(3, seed=2))
    buffer.save(str(tmp_path / "replay"))
    assert (tmp_path / "replay.npz").is_file()
    assert not (tmp_path / "replay").exists()


def _gated_savez(monkeypatch):
    """Make the snapshot writer block until the returned event is set."""
    release = threading.Event()
    entered = threading.Event()
    original = np.savez_compressed

    def gated(file, **arrays):
        entered.set()
        assert release.wait(30)
        return original(file, **arrays)

    monkeypatch.setattr(replay_module.np, "savez_compressed", gated)
    return entered, release


@pytest.mark.parametrize("mutation", ["extend", "add", "clear"])
def test_mutations_wait_for_the_background_snapshot(tmp_path, monkeypatch, mutation) -> None:
    buffer = ReplayBuffer(capacity=50)
    buffer.extend(_examples(45, seed=3))
    reference = tmp_path / "reference.npz"
    buffer.save(str(reference))
    entered, release = _gated_savez(monkeypatch)
    buffer.save_async(str(tmp_path / "async.npz"))
    assert entered.wait(30)

    extra = _examples(20, seed=4)  # 45 + 20 > 50: an add would also evict in place
    done = threading.Event()

    def mutate() -> None:
        if mutation == "extend":
            buffer.extend(extra)
        elif mutation == "add":
            buffer.add(extra[0])
        else:
            buffer.clear()
        done.set()

    worker = threading.Thread(target=mutate)
    worker.start()
    # The mutation is parked behind the snapshot while the writer is held.
    assert not done.wait(0.3)
    assert len(buffer) == 45
    release.set()
    worker.join(30)
    assert done.is_set()
    monkeypatch.undo()
    _assert_same(_arrays(reference), _arrays(tmp_path / "async.npz"))


def test_a_new_snapshot_waits_for_the_previous_one(tmp_path, monkeypatch) -> None:
    buffer = ReplayBuffer(capacity=50)
    buffer.extend(_examples(10, seed=5))
    entered, release = _gated_savez(monkeypatch)
    buffer.save_async(str(tmp_path / "first.npz"))
    assert entered.wait(30)
    second = threading.Thread(target=buffer.save, args=(str(tmp_path / "second.npz"),))
    second.start()
    second.join(0.3)
    assert second.is_alive()
    release.set()
    second.join(30)
    monkeypatch.undo()
    _assert_same(_arrays(tmp_path / "first.npz"), _arrays(tmp_path / "second.npz"))


def _failing_savez(monkeypatch, message: str = "disk full"):
    def failing(file, **arrays):
        file.write(b"PK\x03\x04 half a zip")  # a partial write, then the failure
        raise OSError(message)

    monkeypatch.setattr(replay_module.np, "savez_compressed", failing)


def test_a_failed_background_snapshot_surfaces_once(tmp_path, monkeypatch) -> None:
    buffer = ReplayBuffer(capacity=50)
    buffer.extend(_examples(10, seed=6))
    _failing_savez(monkeypatch)
    buffer.save_async(str(tmp_path / "replay.npz"))
    with pytest.raises(OSError, match="disk full"):
        buffer.wait_save()
    buffer.wait_save()  # reported once, then cleared
    assert not buffer.save_pending


def test_a_mutation_raises_an_unwaited_snapshot_failure(tmp_path, monkeypatch) -> None:
    buffer = ReplayBuffer(capacity=50)
    buffer.extend(_examples(10, seed=7))
    _failing_savez(monkeypatch)
    buffer.save_async(str(tmp_path / "replay.npz"))
    with pytest.raises(OSError, match="disk full"):
        buffer.extend(_examples(2, seed=8))
    assert len(buffer) == 10  # the failure surfaced before anything changed
    buffer.extend(_examples(2, seed=8))
    assert len(buffer) == 12


@pytest.mark.parametrize("background", [False, True])
def test_an_interrupted_write_keeps_the_previous_snapshot(
    tmp_path, monkeypatch, background
) -> None:
    path = tmp_path / "agot.replay.npz"
    buffer = ReplayBuffer(capacity=50)
    buffer.extend(_examples(10, seed=9))
    buffer.save(str(path))
    previous = _arrays(path)
    buffer.extend(_examples(5, seed=10))
    _failing_savez(monkeypatch)
    with pytest.raises(OSError, match="disk full"):
        if background:
            buffer.save_async(str(path))
            buffer.wait_save()
        else:
            buffer.save(str(path))
    monkeypatch.undo()
    _assert_same(previous, _arrays(path))
    assert len(ReplayBuffer.load(str(path))) == 10
    assert _leftovers(tmp_path) == []


def test_finish_replay_snapshot_reports_instead_of_raising(tmp_path, monkeypatch, capsys) -> None:
    from agot import cli

    buffer = ReplayBuffer(capacity=50)
    assert cli._finish_replay_snapshot(buffer) == {"wait": None, "snapshot": None}
    buffer.extend(_examples(4, seed=11))
    buffer.save_async(str(tmp_path / "ok.npz"))
    done = cli._finish_replay_snapshot(buffer)
    assert done["wait"] is not None and done["snapshot"] is not None
    _failing_savez(monkeypatch)
    buffer.save_async(str(tmp_path / "bad.npz"))
    failed = cli._finish_replay_snapshot(buffer)
    assert failed["snapshot"] is None
    assert "replay buffer not saved: disk full" in capsys.readouterr().out
    assert not buffer.save_pending


def test_train_leaves_a_complete_snapshot_every_iteration(tmp_path) -> None:
    pytest.importorskip("torch")
    from agot.cli import main

    checkpoint = tmp_path / "run" / "agot.pt"
    common = [
        "train",
        "--games", "2",
        "--simulations", "4",
        "--determinizations", "1",
        "--max-rounds", "2",
        "--workers", "1",
        "--batch-games", "2",
        "--seed", "3",
        "--buffer", "5000",
        "--buffer-save-every", "1",
        "--eval-games", "0",
        "--gate-games", "0",
        "--leaf-backend", "torch",
        "--device", "cpu",
        "--train-threads", "1",
        "--checkpoint", str(checkpoint),
    ]
    assert main([*common, "--iterations", "2"]) == 0
    replay_file = checkpoint.with_name("agot.replay.npz")
    records = [
        json.loads(line)
        for line in checkpoint.with_name("agot.log.jsonl").read_text().splitlines()
    ]
    assert [record["iteration"] for record in records] == [1, 2]
    # The second iteration waited for the first iteration's background snapshot.
    assert records[0]["seconds_replay_wait"] is None
    assert records[1]["seconds_replay_wait"] is not None
    assert records[1]["seconds_replay_snapshot"] is not None
    # The run's last snapshot was finished before `main` returned.
    assert len(ReplayBuffer.load(str(replay_file))) == records[-1]["buffer"]
    assert _leftovers(replay_file.parent) == []
    # A resume restores it.
    assert main([*common, "--iterations", "3", "--resume", str(checkpoint)]) == 0
    records = [
        json.loads(line)
        for line in checkpoint.with_name("agot.log.jsonl").read_text().splitlines()
    ]
    assert records[-1]["buffer_restored"] is True
