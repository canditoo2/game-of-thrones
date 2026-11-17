"""Export the human round snapshots to a value-training dataset.

Each full snapshot becomes one row: the engine state encoding, the legal-action
rows at the first decision (the value head's action context), and the ±1
per-House outcome. The output is a compressed NPZ that
``agot.training.human_value`` loads directly.

Usage::

    python tools/export_human_value.py
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from agot.engine.game import Game
from agot.engine.observation import ACTION_DIM, HOUSE_ORDER, STATE_DIM

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROUNDS = ROOT / "checkpoints" / "human-games" / "rounds.ndjson.gz"
DEFAULT_OUT = ROOT / "checkpoints" / "human-games" / "values.npz"


def read_snapshots(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=Path, default=DEFAULT_ROUNDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--keep-partial",
        action="store_true",
        help="Also export board-only snapshots (no tracks, power or Supply).",
    )
    args = parser.parse_args()

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    index: list[np.ndarray] = []
    offsets: list[int] = [0]
    outcomes: list[np.ndarray] = []
    games: list[str] = []
    rounds: list[int] = []
    seen: set[tuple[str, int]] = set()
    skipped = 0

    for snapshot in read_snapshots(args.rounds):
        if not snapshot.get("winner"):
            skipped += 1
            continue
        if not snapshot.get("full") and not args.keep_partial:
            skipped += 1
            continue
        key = (snapshot["game"], int(snapshot["round"]))
        if key in seen:
            continue
        try:
            game = Game.from_snapshot(snapshot)
            observation = game.observe()
        except Exception:  # noqa: BLE001 - one malformed game must not stop the export
            skipped += 1
            continue
        if not observation.actions:
            skipped += 1
            continue
        state = np.asarray(observation.state_features, dtype=np.float32)
        matrix = np.asarray(observation.action_features, dtype=np.float32)
        assert state.shape == (STATE_DIM,), state.shape
        assert matrix.shape[1] == ACTION_DIM, matrix.shape
        winner = snapshot["winner"]
        outcome = np.array(
            [1.0 if house == winner else -1.0 for house in HOUSE_ORDER], dtype=np.float32
        )
        states.append(state)
        actions.append(matrix)
        index.append(np.full(matrix.shape[0], len(states) - 1, dtype=np.int32))
        offsets.append(offsets[-1] + matrix.shape[0])
        outcomes.append(outcome)
        games.append(snapshot["game"])
        rounds.append(int(snapshot["round"]))
        seen.add(key)
        if args.limit and len(states) >= args.limit:
            break

    if not states:
        print("no exportable snapshots")
        return 1

    states_array = np.stack(states)
    actions_array = np.concatenate(actions, axis=0)
    index_array = np.concatenate(index, axis=0)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    outcomes_array = np.stack(outcomes)
    games_array = np.asarray(games, dtype=str)
    rounds_array = np.asarray(rounds, dtype=np.int16)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        states=states_array,
        actions=actions_array,
        index=index_array,
        offsets=offsets_array,
        outcomes=outcomes_array,
        games=games_array,
        rounds=rounds_array,
    )
    unique = len(set(games))
    print(
        f"{len(states)} snapshots from {unique} games, {actions_array.shape[0]} action rows, "
        f"skipped {skipped}; {args.out.stat().st_size / 1e6:.1f} MB"
    )
    wins = (outcomes_array > 0).sum(axis=0)
    for house, count in zip(HOUSE_ORDER, wins, strict=True):
        print(f"  {house:10s} {int(count):4d} wins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
