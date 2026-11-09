"""Tests for the deferred checkpoint gate (no engine required)."""

from __future__ import annotations

import json
from pathlib import Path

from agot.engine.game import DEFAULT_HOUSES
from agot.training import gate
from agot.training.evaluate import (
    GameOutcome,
    build_match_plan,
    split_houses,
)


def test_chunked_plan_equals_the_single_call_plan() -> None:
    """Splitting the plan must not change a seed or a seat split."""
    whole = build_match_plan(DEFAULT_HOUSES, 10, 123)
    chunks = whole.chunks(3)

    assert [len(chunk) for chunk in chunks] == [3, 3, 3, 1]
    # Chunks are aligned to global index 0, not to the slice start.
    assert [chunk.games[0].index for chunk in chunks] == [0, 3, 6, 9]

    rebuilt = [game for chunk in chunks for game in chunk.games]
    assert rebuilt == list(whole.games)
    assert [game.seed for game in rebuilt] == [123 + index for index in range(10)]
    # The 3v3 split is random per game but reproducible from the match seed,
    # and every game gives A exactly half the houses.
    again = build_match_plan(DEFAULT_HOUSES, 10, 123)
    assert [game.seats_a for game in again.games] == [
        game.seats_a for game in rebuilt
    ]
    for game in rebuilt:
        assert len(game.seats_a) == len(DEFAULT_HOUSES) // 2
        assert len(set(game.seats_a)) == len(game.seats_a)
        assert set(game.seats_a) <= set(DEFAULT_HOUSES)


def _append_verdict(log: Path, *, passed: bool, iteration: int | None = None) -> None:
    record = {"event": "verdict", "passed": passed}
    if iteration is not None:
        record["iteration"] = iteration
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_patience_is_recomputed_from_the_verdict_log(tmp_path) -> None:
    checkpoint = tmp_path / "agot.pt"
    log = gate.gate_dir(checkpoint) / gate.LOG_NAME
    log.parent.mkdir(parents=True)
    assert gate.gate_failures(checkpoint) == 0

    _append_verdict(log, passed=False)
    _append_verdict(log, passed=False)
    assert gate.gate_failures(checkpoint) == 2

    _append_verdict(log, passed=True)
    assert gate.gate_failures(checkpoint) == 0

    _append_verdict(log, passed=False)
    assert gate.gate_failures(checkpoint) == 1

    # A candidate queued but not played has no verdict and is not a strike.
    pending = gate.candidate_dir(gate.gate_dir(checkpoint), 11)
    pending.mkdir(parents=True)
    (pending / gate.SPEC_NAME).write_text("{}", encoding="utf-8")
    assert gate.gate_failures(checkpoint) == 1

    # Unrelated records and a truncated last line are ignored.
    with log.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "queued"}\n{"passed": fal\n')
    assert gate.gate_failures(checkpoint) == 1

    # Two finalizers racing on one candidate must not count as two strikes.
    _append_verdict(log, passed=False, iteration=20)
    _append_verdict(log, passed=False, iteration=20)
    assert gate.gate_failures(checkpoint) == 2


def _candidate(
    tmp_path: Path,
    *,
    games: int = 4,
    batch_games: int = 2,
    threshold: float = 0.5,
    seeded: bool = False,
    iteration: int = 10,
) -> tuple[Path, Path, Path, gate.GateSpec]:
    checkpoint = tmp_path / "agot.pt"
    best = tmp_path / "agot-best.pt"
    directory = gate.candidate_dir(gate.gate_dir(checkpoint), iteration)
    directory.mkdir(parents=True)
    (directory / gate.CANDIDATE_NAME).write_bytes(b"candidate")
    (directory / gate.BEST_SNAPSHOT_NAME).write_bytes(b"best")
    if not seeded:
        best.write_bytes(b"best")
    spec = gate.GateSpec(
        iteration=iteration,
        games=games,
        seed=5000,
        simulations=2,
        determinizations=1,
        batch_games=batch_games,
        threshold=threshold,
        best_checkpoint=str(best),
        seeded=seeded,
    )
    gate._write_json(directory / gate.SPEC_NAME, spec.as_dict())
    return checkpoint, best, directory, spec


def _winners(spec: gate.GateSpec, *, candidate_wins: bool) -> list[str]:
    plan = build_match_plan(spec.houses, spec.games, spec.seed)
    winners = []
    for game in plan.games:
        seats_a = set(game.seats_a) if game.seats_a else split_houses(
            plan.houses, game.swap
        )[0]
        seats_b = set(plan.houses) - seats_a
        winners.append(sorted(seats_a if candidate_wins else seats_b)[0])
    return winners


def _fill_shards(
    directory: Path,
    spec: gate.GateSpec,
    winners: list[str],
    *,
    chunks: set[int] | None = None,
) -> None:
    plan = build_match_plan(spec.houses, spec.games, spec.seed)
    for chunk, chunk_plan in enumerate(plan.chunks(spec.batch_games)):
        if chunks is not None and chunk not in chunks:
            continue
        outcomes = [
            GameOutcome(index=game.index, winner=winners[game.index], steps=42)
            for game in chunk_plan.games
        ]
        gate.write_shard(directory, chunk, chunk_plan, outcomes, seconds=0.01)


def test_spec_and_verdict_round_trip(tmp_path) -> None:
    _checkpoint, _best, directory, spec = _candidate(tmp_path)

    loaded = gate.load_spec(directory)
    assert loaded == spec
    assert loaded is not None
    assert loaded.houses == spec.houses
    assert loaded.game_config().houses == spec.houses
    assert loaded.game_config().deferred_battles is True
    assert loaded.game_config().sequential_westeros is True

    payload = {"event": "verdict", "iteration": 10, "passed": True}
    gate._write_json(directory / gate.VERDICT_NAME, payload)
    assert gate.load_verdict(directory) == payload


def test_promotion_is_idempotent(tmp_path) -> None:
    checkpoint, best, directory, spec = _candidate(tmp_path)
    _fill_shards(directory, spec, _winners(spec, candidate_wins=True))

    first = gate.finalize_candidate(directory)
    assert first is not None
    assert first["passed"] is True
    assert first["promoted"] is True
    assert best.read_bytes() == b"candidate"

    log = gate.gate_dir(checkpoint) / gate.LOG_NAME
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1

    second = gate.finalize_candidate(directory)
    assert second == first
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert best.read_bytes() == b"candidate"


def test_failed_candidate_keeps_the_best(tmp_path) -> None:
    checkpoint, best, directory, spec = _candidate(tmp_path)
    _fill_shards(directory, spec, _winners(spec, candidate_wins=False))

    verdict = gate.finalize_candidate(directory)
    assert verdict is not None
    assert verdict["passed"] is False
    assert verdict["promoted"] is False
    assert best.read_bytes() == b"best"
    assert gate.gate_failures(checkpoint) == 1


def test_missing_chunk_yields_no_verdict(tmp_path) -> None:
    checkpoint, best, directory, spec = _candidate(tmp_path)
    _fill_shards(directory, spec, _winners(spec, candidate_wins=True), chunks={0})

    assert gate.finalize_candidate(directory) is None
    assert gate.load_verdict(directory) is None
    assert not (gate.gate_dir(checkpoint) / gate.LOG_NAME).exists()
    assert best.read_bytes() == b"best"


def test_finalize_repairs_a_verdict_missing_from_the_log(tmp_path) -> None:
    checkpoint, _best, directory, spec = _candidate(tmp_path)
    _fill_shards(directory, spec, _winners(spec, candidate_wins=False))
    verdict = gate.finalize_candidate(directory)
    assert verdict is not None

    log = gate.gate_dir(checkpoint) / gate.LOG_NAME
    log.unlink()
    assert gate.finalize_candidate(directory) == verdict
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_seeded_gate_promotes_the_first_candidate(tmp_path) -> None:
    _checkpoint, best, directory, spec = _candidate(tmp_path, seeded=True)
    _fill_shards(directory, spec, _winners(spec, candidate_wins=False))

    verdict = gate.finalize_candidate(directory)
    assert verdict is not None
    assert verdict["seeded"] is True
    assert verdict["passed"] is True
    assert best.read_bytes() == b"candidate"


def test_only_the_first_queued_gate_seeds_the_best(tmp_path) -> None:
    _checkpoint, best, first_dir, first_spec = _candidate(tmp_path, seeded=True)
    _fill_shards(first_dir, first_spec, _winners(first_spec, candidate_wins=False))
    assert gate.finalize_candidate(first_dir)["seeded"] is True
    assert best.read_bytes() == b"candidate"

    # The second candidate was queued while no best existed, so its spec says
    # seeded too; it must still be compared against the checkpoint the first
    # gate just wrote, not promote itself unconditionally.
    _ckpt, _b, second_dir, second_spec = _candidate(
        tmp_path, seeded=True, iteration=20
    )
    (second_dir / gate.CANDIDATE_NAME).write_bytes(b"candidate-two")
    _fill_shards(second_dir, second_spec, _winners(second_spec, candidate_wins=False))

    second = gate.finalize_candidate(second_dir)
    assert second is not None
    assert second["seeded"] is False
    assert second["passed"] is False
    assert best.read_bytes() == b"candidate"


class _FakeModel:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"weights")


def test_enqueue_writes_candidate_best_and_spec_last(tmp_path) -> None:
    checkpoint = tmp_path / "run" / "agot.pt"
    checkpoint.parent.mkdir()
    best = checkpoint.with_name("agot-best.pt")

    directory = gate.enqueue_candidate(
        checkpoint,
        3,
        _FakeModel(),
        best_checkpoint=best,
        games=4,
        seed=0,
        simulations=2,
        determinizations=1,
        batch_games=2,
        threshold=0.5,
    )
    assert directory == gate.candidate_dir(gate.gate_dir(checkpoint), 3)
    assert (directory / gate.CANDIDATE_NAME).read_bytes() == b"weights"
    assert (directory / gate.BEST_SNAPSHOT_NAME).read_bytes() == b"weights"
    spec = gate.load_spec(directory)
    assert spec is not None
    assert spec.seeded is True
    assert gate.pending_candidates(checkpoint) == [directory]

    # Enqueuing the same iteration again is a no-op.
    assert (
        gate.enqueue_candidate(
            checkpoint,
            3,
            _FakeModel(),
            best_checkpoint=best,
            games=4,
            seed=0,
            simulations=2,
            determinizations=1,
            batch_games=2,
            threshold=0.5,
        )
        == directory
    )


def test_shard_runs_only_its_chunks(tmp_path, monkeypatch) -> None:
    _checkpoint, _best, directory, spec = _candidate(tmp_path, games=4, batch_games=2)

    monkeypatch.setattr(gate, "_load_models", lambda path: [object(), object()])
    calls: list[list[int]] = []

    def fake_play_plan(models, plan, **options) -> list[GameOutcome]:
        calls.append([game.index for game in plan.games])
        return [
            GameOutcome(index=game.index, winner=None, steps=1)
            for game in plan.games
        ]

    monkeypatch.setattr(gate, "play_plan", fake_play_plan)
    status = gate.run_candidate(directory, shard=(1, 2))

    assert calls == [[2, 3]]
    assert status["chunks_done"] == 1
    assert status["chunks_played"] == 1
    assert status["verdict"] is None
