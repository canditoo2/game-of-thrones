"""Deferred, resumable checkpoint gate.

The inline relative gate runs inside a training iteration, so on Colab one
200-game gate (~2 h) can take the whole iteration down with the runtime. The
deferred mode turns the gate into files on disk instead: the training loop
enqueues the candidate after it has committed the iteration, and ``agot gate``
plays it later, in fixed-size chunks.

Every chunk is one atomic JSON shard, so a disconnected gate resumes at the
first missing chunk and at most the chunk in flight is lost. A verdict is
written only once every shard exists; patience is recomputed from the verdict
log rather than incremented in place, because a queued candidate without a
verdict must not count as a failure.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from agot.engine.game import DEFAULT_HOUSES, GameConfig
from agot.training.evaluate import (
    GameOutcome,
    MatchPlan,
    build_match_plan,
    play_plan,
    summarize_plan,
    wilson_interval,
)
from agot.training.selfplay import DEFAULT_STEP_LIMIT

#: Suffix appended to a checkpoint to find its gate directory.
GATE_SUFFIX = ".gate"
CANDIDATES_DIR = "candidates"
SHARDS_DIR = "shards"
SPEC_NAME = "spec.json"
CANDIDATE_NAME = "candidate.pt"
BEST_SNAPSHOT_NAME = "best.pt"
VERDICT_NAME = "verdict.json"
LOG_NAME = "gate.log.jsonl"

Progress = Callable[[str], None]


def gate_dir(checkpoint: str | Path) -> Path:
    """Directory holding the deferred gate queue for a checkpoint."""
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(f"{checkpoint.stem}{GATE_SUFFIX}")


def candidate_dir(gate_root: str | Path, iteration: int) -> Path:
    """Directory of one candidate: ``<stem>.gate/candidates/iter-NNNNNN``."""
    return Path(gate_root) / CANDIDATES_DIR / f"iter-{int(iteration):06d}"


def shard_path(directory: str | Path, chunk: int) -> Path:
    """Path of one chunk shard inside a candidate directory."""
    return Path(directory) / SHARDS_DIR / f"chunk-{int(chunk):02d}.json"


@dataclass
class GateSpec:
    """Everything one deferred gate needs to reproduce the inline match.

    The spec is written last, so a candidate directory without it is an
    interrupted enqueue and is never picked up by ``agot gate``.
    """

    iteration: int
    games: int
    seed: int
    simulations: int
    determinizations: int
    batch_games: int
    threshold: float
    best_checkpoint: str
    step_limit: int = DEFAULT_STEP_LIMIT
    min_house: float = 0.0
    house_tolerance: float = 0.10
    house_margin: float = 0.0
    max_rounds: int = 10
    victory_castles: int = 7
    houses: tuple[str, ...] = DEFAULT_HOUSES
    use_westeros_decks: bool = True
    deferred_battles: bool = True
    sequential_westeros: bool = True
    doran_choice: bool = True
    wildling_tie_choice: bool = True
    bid_tie_choice: bool = True
    choice_choices: bool = True
    #: True when no best checkpoint existed at enqueue time: this gate seeds it.
    seeded: bool = False

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["houses"] = list(self.houses)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> GateSpec:
        known = {item.name for item in fields(cls)}
        values = {key: value for key, value in payload.items() if key in known}
        if "houses" in values:
            values["houses"] = tuple(values["houses"])
        return cls(**values)

    def game_config(self) -> GameConfig:
        return GameConfig(
            houses=self.houses,
            max_rounds=self.max_rounds,
            victory_castles=self.victory_castles,
            seed=self.seed,
            use_westeros_decks=self.use_westeros_decks,
            deferred_battles=self.deferred_battles,
            sequential_westeros=self.sequential_westeros,
            doran_choice=self.doran_choice,
            wildling_tie_choice=self.wildling_tie_choice,
            bid_tie_choice=self.bid_tie_choice,
            choice_choices=self.choice_choices,
        )


def _write_json(path: str | Path, payload: dict) -> None:
    """Write JSON beside the target and rename it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A per-process temporary lets two gate shards finish at once without
    # clobbering each other's half-written file.
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def load_spec(directory: str | Path) -> GateSpec | None:
    path = Path(directory) / SPEC_NAME
    if not path.exists():
        return None
    return GateSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_verdict(directory: str | Path) -> dict | None:
    path = Path(directory) / VERDICT_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pending_candidates(checkpoint: str | Path) -> list[Path]:
    """Candidate directories with a complete spec and no verdict yet."""
    root = gate_dir(checkpoint) / CANDIDATES_DIR
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / SPEC_NAME).exists()
        and not (path / VERDICT_NAME).exists()
    )


def parse_shard(value: str) -> tuple[int, int]:
    """Parse ``i/N`` into a zero-based ``(index, count)`` pair."""
    left, separator, right = str(value).partition("/")
    if not separator:
        raise ValueError("--shard must be i/N")
    try:
        index, count = int(left), int(right)
    except ValueError:
        raise ValueError("--shard must be i/N with integer values") from None
    if count < 1 or not 1 <= index <= count:
        raise ValueError("--shard needs 1 <= i <= N")
    return index - 1, count


def enqueue_candidate(
    checkpoint: str | Path,
    iteration: int,
    model,
    *,
    best_checkpoint: str | Path,
    games: int,
    seed: int,
    simulations: int,
    determinizations: int,
    batch_games: int,
    threshold: float,
    min_house: float = 0.0,
    house_tolerance: float = 0.10,
    house_margin: float = 0.0,
    config: GameConfig | None = None,
    step_limit: int = DEFAULT_STEP_LIMIT,
) -> Path:
    """Copy a candidate and its gate opponent into the queue.

    ``spec.json`` is written last: a disconnect while copying can leave an
    incomplete directory, but never a runnable candidate with a truncated
    checkpoint. Enqueuing the same iteration twice is a no-op.
    """
    base = config or GameConfig()
    if int(games) <= 0:
        raise ValueError("a deferred gate needs at least one game")
    root = gate_dir(checkpoint)
    directory = candidate_dir(root, iteration)
    if (directory / SPEC_NAME).exists():
        return directory
    directory.mkdir(parents=True, exist_ok=True)

    model.save(str(directory / CANDIDATE_NAME))
    best_path = Path(best_checkpoint)
    seeded = not best_path.exists()
    if seeded:
        shutil.copyfile(directory / CANDIDATE_NAME, directory / BEST_SNAPSHOT_NAME)
    else:
        shutil.copyfile(best_path, directory / BEST_SNAPSHOT_NAME)

    spec = GateSpec(
        iteration=int(iteration),
        games=int(games),
        seed=int(seed),
        simulations=int(simulations),
        determinizations=int(determinizations),
        batch_games=max(1, int(batch_games)),
        threshold=float(threshold),
        best_checkpoint=str(best_path.resolve()),
        step_limit=int(step_limit),
        min_house=float(min_house),
        house_tolerance=float(house_tolerance),
        house_margin=float(house_margin),
        max_rounds=base.max_rounds,
        victory_castles=base.victory_castles,
        houses=tuple(base.houses),
        use_westeros_decks=base.use_westeros_decks,
        deferred_battles=base.deferred_battles,
        sequential_westeros=base.sequential_westeros,
        doran_choice=base.doran_choice,
        wildling_tie_choice=base.wildling_tie_choice,
        bid_tie_choice=base.bid_tie_choice,
        choice_choices=base.choice_choices,
        seeded=seeded,
    )
    _write_json(directory / SPEC_NAME, spec.as_dict())
    return directory


def write_shard(
    directory: str | Path,
    chunk: int,
    plan: MatchPlan,
    outcomes: list[GameOutcome],
    *,
    seconds: float = 0.0,
) -> Path:
    """Write one chunk's results atomically and return its path."""
    by_index = {outcome.index: outcome for outcome in outcomes}
    expected = [game.index for game in plan.games]
    missing = [index for index in expected if index not in by_index]
    extra = sorted(set(by_index) - set(expected))
    if missing or extra:
        raise ValueError(
            f"the shard does not cover its plan (missing={missing}, extra={extra})"
        )
    payload = {
        "chunk": int(chunk),
        "start": expected[0] if expected else 0,
        "stop": expected[-1] + 1 if expected else 0,
        "games": [
            {
                "index": game.index,
                "seed": game.seed,
                "swap": game.swap,
                "winner": by_index[game.index].winner,
                "steps": int(by_index[game.index].steps),
            }
            for game in plan.games
        ],
        "seconds": round(float(seconds), 3),
    }
    path = shard_path(directory, chunk)
    _write_json(path, payload)
    return path


def _load_shard_outcomes(directory: Path, chunk: int) -> list[GameOutcome]:
    payload = json.loads(shard_path(directory, chunk).read_text(encoding="utf-8"))
    if int(payload.get("chunk", -1)) != chunk:
        raise ValueError(f"shard {chunk} does not describe chunk {chunk}")
    return [
        GameOutcome(
            index=int(record["index"]),
            winner=record.get("winner"),
            steps=int(record["steps"]),
        )
        for record in payload["games"]
    ]


def gate_decision(spec: GateSpec, match) -> dict:
    """The inline gate rule, shared with the deferred finalizer."""
    low, high = wilson_interval(match.wins_a, match.games)
    house_ok = all(
        match.per_house_a[house] + spec.house_tolerance
        >= match.per_house_b.get(house, 1.0) + spec.house_margin
        for house in match.per_house_a
    )
    absolute_ok = all(
        rate >= spec.min_house for rate in match.per_house_a.values()
    )
    passed = match.win_rate_a > spec.threshold and house_ok and absolute_ok
    worst_house = min(
        (
            rate - match.per_house_b.get(house, 0.0)
            for house, rate in match.per_house_a.items()
        ),
        default=0.0,
    )
    return {
        "passed": bool(passed),
        "house_ok": bool(house_ok),
        "absolute_ok": bool(absolute_ok),
        "win_rate_a": match.win_rate_a,
        "win_rate_b": match.win_rate_b,
        "wins_a": match.wins_a,
        "wins_b": match.wins_b,
        "games": match.games,
        "ci_low": low,
        "ci_high": high,
        "per_house_a": match.per_house_a,
        "per_house_b": match.per_house_b,
        "average_steps": match.average_steps,
        "worst_house": worst_house,
    }


def _promote(source: Path, target: Path) -> None:
    """Copy ``source`` beside ``target`` and rename it into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def finalize_candidate(directory: str | Path) -> dict | None:
    """Aggregate every shard into a verdict, or return ``None`` if incomplete.

    Finalizing is idempotent: once ``verdict.json`` exists it is returned
    unchanged, so a rerun after a promotion-then-disconnect cannot append a
    second verdict line or promote the same candidate twice.
    """
    directory = Path(directory)
    existing = load_verdict(directory)
    if existing is not None:
        # A crash between the verdict write and the log append would otherwise
        # hide a failure from patience forever: repair the missing line.
        if not _logged(directory, existing):
            _append_verdict_log(directory, existing)
        return existing
    spec = load_spec(directory)
    if spec is None:
        return None

    plan = build_match_plan(spec.houses, spec.games, spec.seed)
    chunks = plan.chunks(spec.batch_games)
    if not all(shard_path(directory, chunk).exists() for chunk in range(len(chunks))):
        return None

    outcomes: list[GameOutcome] = []
    for chunk in range(len(chunks)):
        outcomes.extend(_load_shard_outcomes(directory, chunk))
    match = summarize_plan(outcomes, plan, spec.houses)
    decision = gate_decision(spec, match)

    best_path = Path(spec.best_checkpoint)
    # Re-check the file rather than trusting the enqueue-time flag: several
    # candidates can be queued before the first gate runs, and only the first
    # one to finish may seed the best. A later one must compare against it.
    seeded = not best_path.exists()
    if seeded:
        decision["passed"] = True

    promoted = False
    if decision["passed"]:
        _promote(directory / CANDIDATE_NAME, best_path)
        promoted = True

    verdict = {
        "event": "verdict",
        "iteration": spec.iteration,
        "seeded": bool(seeded),
        "promoted": promoted,
        **decision,
        "threshold": spec.threshold,
        "min_house": spec.min_house,
        "house_tolerance": spec.house_tolerance,
        "house_margin": spec.house_margin,
        "best_checkpoint": spec.best_checkpoint,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # The best checkpoint has already been replaced atomically; the verdict is
    # the commit marker. A crash before it simply re-finalizes on the next run.
    _write_json(directory / VERDICT_NAME, verdict)
    if not _logged(directory, verdict):
        _append_verdict_log(directory, verdict)
    return verdict


def _candidate_log(directory: Path) -> Path:
    """The gate log shared by every candidate of one checkpoint."""
    for parent in (directory, *directory.parents):
        if parent.name.endswith(GATE_SUFFIX):
            return parent / LOG_NAME
    return directory.parent.parent / LOG_NAME


def _logged(directory: Path, verdict: dict) -> bool:
    iteration = verdict.get("iteration")
    if iteration is None:
        return False
    for record in _read_log(_candidate_log(directory)):
        if (
            record.get("iteration") == iteration
            and bool(record.get("passed")) == bool(verdict.get("passed"))
        ):
            return True
    return False


def _append_verdict_log(directory: Path, verdict: dict) -> None:
    log = _candidate_log(directory)
    log.parent.mkdir(parents=True, exist_ok=True)
    record = {
        key: verdict[key]
        for key in (
            "event",
            "iteration",
            "seeded",
            "promoted",
            "passed",
            "games",
            "wins_a",
            "wins_b",
            "win_rate_a",
            "ci_low",
            "ci_high",
            "worst_house",
            "house_ok",
            "absolute_ok",
            "completed_at",
        )
        if key in verdict
    }
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _read_log(path: Path) -> list[dict]:
    """Read a JSONL gate log, skipping truncated lines."""
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A disconnect can truncate the last append.
                continue
    return records


def verdict_records(checkpoint: str | Path) -> list[dict]:
    """Read the deferred gate's verdict log, skipping truncated lines."""
    return _read_log(gate_dir(checkpoint) / LOG_NAME)


def gate_failures(checkpoint: str | Path) -> int:
    """Consecutive failed verdicts at the tail of the gate log.

    Recomputed from disk on purpose: a candidate that was queued but never
    played has no verdict and is not a strike. Any pass resets the count.
    Duplicate lines for one iteration (two shard processes finalizing the same
    candidate) collapse to the last one, so patience cannot double-count them.
    """
    seen: dict[object, tuple[int, bool]] = {}
    for position, record in enumerate(verdict_records(checkpoint)):
        if "passed" not in record:
            continue
        key = record.get("iteration")
        if key is None:
            key = ("anonymous", position)
        seen[key] = (position, bool(record["passed"]))
    failures = 0
    for _, passed in sorted(seen.values()):
        failures = 0 if passed else failures + 1
    return failures


def _load_models(directory: Path) -> list:
    from agot.nn.model import PolicyValueNet

    return [
        PolicyValueNet.load(str(directory / CANDIDATE_NAME)),
        PolicyValueNet.load(str(directory / BEST_SNAPSHOT_NAME)),
    ]


def run_candidate(
    directory: str | Path,
    *,
    deadline: float | None = None,
    shard: tuple[int, int] | None = None,
    progress: Progress | None = None,
) -> dict:
    """Play the missing chunks of one candidate, then finalize if complete.

    ``deadline`` is a ``time.monotonic()`` value; no new chunk starts after it.
    ``shard`` selects chunks by ``chunk % count == index`` for parallel runs.
    """
    directory = Path(directory)
    spec = load_spec(directory)
    if spec is None:
        return {
            "candidate": str(directory),
            "error": "missing spec.json",
            "chunks_total": 0,
            "chunks_done": 0,
            "verdict": None,
        }

    plan = build_match_plan(spec.houses, spec.games, spec.seed)
    chunks = plan.chunks(spec.batch_games)
    models = None
    played = 0
    for chunk, chunk_plan in enumerate(chunks):
        if shard is not None and chunk % shard[1] != shard[0]:
            continue
        if shard_path(directory, chunk).exists():
            continue
        if deadline is not None and time.monotonic() >= deadline:
            break
        if models is None:
            models = _load_models(directory)
        started = time.monotonic()
        outcomes = play_plan(
            models,
            chunk_plan,
            simulations=spec.simulations,
            determinizations=spec.determinizations,
            config=spec.game_config(),
            step_limit=spec.step_limit,
            batch_games=spec.batch_games,
        )
        write_shard(
            directory,
            chunk,
            chunk_plan,
            outcomes,
            seconds=time.monotonic() - started,
        )
        played += 1
        if progress is not None:
            progress(
                f"{directory.name}: chunk {chunk + 1}/{len(chunks)} "
                f"({len(chunk_plan)} games)"
            )

    done = sum(
        shard_path(directory, chunk).exists() for chunk in range(len(chunks))
    )
    verdict = finalize_candidate(directory) if done == len(chunks) else None
    if verdict is not None and progress is not None:
        state = "seeded" if verdict["seeded"] else ("passed" if verdict["passed"] else "failed")
        progress(
            f"{directory.name}: gate {state} "
            f"({verdict['wins_a']}/{verdict['games']} for the candidate)"
        )
    return {
        "candidate": str(directory),
        "iteration": spec.iteration,
        "chunks_total": len(chunks),
        "chunks_done": done,
        "chunks_played": played,
        "verdict": verdict,
    }


def run_pending(
    checkpoint: str | Path,
    *,
    max_minutes: float = 0.0,
    shard: tuple[int, int] | None = None,
    progress: Progress | None = None,
) -> list[dict]:
    """Play queued candidates in order, respecting a wall-clock budget."""
    deadline = (
        time.monotonic() + max_minutes * 60.0
        if max_minutes and max_minutes > 0
        else None
    )
    results: list[dict] = []
    for directory in pending_candidates(checkpoint):
        if deadline is not None and time.monotonic() >= deadline:
            break
        results.append(
            run_candidate(directory, deadline=deadline, shard=shard, progress=progress)
        )
    return results
