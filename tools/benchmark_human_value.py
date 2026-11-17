"""Score a checkpoint on the human round snapshots.

Every ``orders-revealed`` position is a complete board with the eventual winner
as the label. The snapshot importer fills the encoder state, and the engine's
own legal-action list at the first decision gives the value head the action
context it was trained with, so this is the checkpoint's real evaluation on
human play rather than a hand-built proxy.

Usage::

    python tools/benchmark_human_value.py --checkpoint checkpoints/run7/agot.pt
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from agot.engine.game import Game
from agot.engine.observation import HOUSE_ORDER
from agot.nn.model import PolicyValueNet

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROUNDS = ROOT / "checkpoints" / "human-games" / "rounds.ndjson.gz"


def read_snapshots(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def castle_counts(snapshot: dict, forts: set[str]) -> dict[str, int]:
    """Controlled fortifications per house, from the board alone."""
    counts = defaultdict(int)
    for region in snapshot["regions"]:
        if region["id"] not in forts:
            continue
        houses = {unit[1] for unit in region["units"]}
        if len(houses) == 1:
            counts[houses.pop()] += 1
    return counts


def fortifications() -> set[str]:
    map_data = json.loads(
        (ROOT / "src" / "agot" / "data" / "map_base_6p.json").read_text(encoding="utf-8")
    )
    return {region["id"] for region in map_data["regions"] if region.get("mustering", 0) > 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rounds", type=Path, default=DEFAULT_ROUNDS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--actor",
        choices=("engine", "none"),
        default="engine",
        help="Encode with the imported position's acting house, or with no actor flag.",
    )
    args = parser.parse_args()

    # Loaded outside inference mode: the packed-inference path reads each
    # parameter's version counter, which inference tensors do not track.
    model = PolicyValueNet.load(str(args.checkpoint))
    model.eval()

    forts = fortifications()
    rows = []
    skipped = 0
    for snapshot in read_snapshots(args.rounds):
        if not snapshot.get("full") or not snapshot.get("winner"):
            continue
        try:
            game = Game.from_snapshot(snapshot)
            observation = game.observe()
        except Exception:  # noqa: BLE001 - one malformed game must not stop the run
            skipped += 1
            continue
        if not observation.actions:
            skipped += 1
            continue
        state = torch.from_numpy(np.asarray(observation.state_features))[None]
        if args.actor == "none":
            state = torch.from_numpy(np.asarray(game.encode_state(None)))[None]
        actions = torch.from_numpy(np.asarray(observation.action_features))
        index = torch.zeros(actions.shape[0], dtype=torch.long)
        with torch.inference_mode():
            _, values = model.forward_packed(state, actions, index)
        values = values[0].tolist()
        rows.append(
            {
                "game": snapshot["game"],
                "round": snapshot["round"],
                "winner": snapshot["winner"],
                "values": values,
                "castles": castle_counts(snapshot, forts),
                "actions": len(observation.actions),
            }
        )
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        print("no usable snapshots")
        return 1

    houses = list(HOUSE_ORDER)

    def metrics(selected: list[dict]) -> dict:
        mse = 0.0
        brier = 0.0
        hits = 0
        castle_hits = 0
        winner_values = []
        loser_values = []
        for row in selected:
            winner = row["winner"]
            target = np.array([1.0 if house == winner else -1.0 for house in houses])
            values = np.array(row["values"])
            mse += float(np.mean((values - target) ** 2))
            brier += float(np.mean(((values + 1.0) / 2.0 - (target > 0)) ** 2))
            hits += int(houses[int(np.argmax(values))] == winner)
            castles = row["castles"]
            if castles:
                castle_hits += int(max(castles, key=castles.get) == winner)
            winner_values.append(float(values[houses.index(winner)]))
            loser_values.extend(
                float(value)
                for house, value in zip(houses, values, strict=True)
                if house != winner
            )
        n = len(selected)
        return {
            "n": n,
            "mse": mse / n,
            "brier": brier / n,
            "accuracy": hits / n,
            "castle_accuracy": castle_hits / n,
            "winner_value": float(np.mean(winner_values)),
            "loser_value": float(np.mean(loser_values)),
        }

    overall = metrics(rows)
    action_counts = np.array([row["actions"] for row in rows])
    print(f"checkpoint {args.checkpoint}")
    print(
        f"snapshots {len(rows)}, skipped {skipped} | actions: "
        f"min {action_counts.min()}, median {int(np.median(action_counts))}, "
        f"max {action_counts.max()}, single {np.mean(action_counts == 1) * 100:.0f}%"
    )
    print(
        f"MSE {overall['mse']:.3f} | Brier {overall['brier']:.3f} | "
        f"winner argmax {overall['accuracy'] * 100:.1f}% "
        f"(chance {100 / len(houses):.1f}%, castles {overall['castle_accuracy'] * 100:.1f}%)"
    )
    print(f"mean value: winner {overall['winner_value']:+.3f}, loser {overall['loser_value']:+.3f}")

    by_round: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_round[row["round"]].append(row)
    print("\nby round:")
    for round_number in sorted(by_round):
        stats = metrics(by_round[round_number])
        print(
            f"  round {round_number:2d}  n={stats['n']:4d}  "
            f"MSE {stats['mse']:.3f}  acc {stats['accuracy'] * 100:5.1f}%  "
            f"winner {stats['winner_value']:+.3f}"
        )

    print("\nby winner house:")
    for house in houses:
        selected = [row for row in rows if row["winner"] == house]
        if not selected:
            continue
        stats = metrics(selected)
        print(
            f"  {house:10s} n={stats['n']:3d}  MSE {stats['mse']:.3f}  "
            f"recall {stats['accuracy'] * 100:5.1f}%  winner {stats['winner_value']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
