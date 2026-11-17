"""Game-level A/B for the human value-only pass.

Splits the archive by game (never by snapshot, or rounds of one game leak
across the split), trains the value head on the training games and measures the
eventual-winner ranking on the held-out games before and after.

Usage::

    python tools/ab_human_value.py --checkpoint checkpoints/run7/agot.pt \
        --values checkpoints/human-games/values.npz --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from agot.nn.model import PolicyValueNet
from agot.training.human_value import HumanValueSet
from agot.training.trainer import TrainConfig, Trainer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VALUES = ROOT / "checkpoints" / "human-games" / "values.npz"


def evaluate(model: PolicyValueNet, dataset: HumanValueSet, batch_size: int = 64) -> dict:
    model.eval()
    predictions = []
    outcomes = []
    with torch.inference_mode():
        for state, actions, index, outcome in dataset.batches(batch_size, shuffle=False):
            _, values = model.forward_packed(state, actions, index)
            predictions.append(values.numpy())
            outcomes.append(outcome.numpy())
    values = np.concatenate(predictions)
    targets = np.concatenate(outcomes)
    winner = targets.argmax(axis=1)
    return {
        "n": int(values.shape[0]),
        "accuracy": float((values.argmax(axis=1) == winner).mean()),
        "mse": float(((values - targets) ** 2).mean()),
        "winner_value": float(values[targets > 0].mean()),
        "loser_value": float(values[targets < 0].mean()),
    }


def show(label: str, stats: dict) -> None:
    print(
        f"{label:12s} n={stats['n']:4d}  argmax {stats['accuracy'] * 100:5.1f}%  "
        f"MSE {stats['mse']:.3f}  winner {stats['winner_value']:+.3f}  "
        f"loser {stats['loser_value']:+.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--values", type=Path, default=DEFAULT_VALUES)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-scale", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    dataset = HumanValueSet.load(args.values)
    games = np.unique(dataset.games)
    order = np.random.default_rng(args.seed).permutation(len(games))
    test_count = max(1, round(len(games) * args.test_fraction))
    test_games = set(games[order[:test_count]].tolist())
    is_test = np.array([game in test_games for game in dataset.games])
    train_set = dataset.select(np.flatnonzero(~is_test))
    test_set = dataset.select(np.flatnonzero(is_test))
    print(
        f"{len(dataset)} snapshots, {len(games)} games: "
        f"{len(train_set)} train snapshots from {len(games) - test_count} games, "
        f"{len(test_set)} test snapshots from {test_count} games"
    )

    model = PolicyValueNet.load(str(args.checkpoint))
    show("before/test", evaluate(model, test_set))
    show("before/train", evaluate(model, train_set))

    trainer = Trainer(model, TrainConfig(batch_size=args.batch_size, device="cpu"))
    stats = trainer.train_value_only(
        train_set,
        lr_scale=args.lr_scale,
        epochs=args.epochs,
        seed=args.seed,
        head_only=args.head_only,
    )
    print(f"trained {stats['batches']} batches, last value loss {stats['value_loss']:.3f}")
    show("after/test", evaluate(model, test_set))
    show("after/train", evaluate(model, train_set))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.save))
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
