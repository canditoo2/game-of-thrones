"""Command-line entry point.

Examples::

    agot demo --policy greedy --seed 1
    agot evaluate --a random --b greedy --games 20
    agot train --games 50 --simulations 32 --epochs 2

The ``train`` subcommand is the only one that needs PyTorch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agot.agents import MCTS, GreedyPolicy, RandomPolicy
from agot.engine.game import Game, GameConfig
from agot.training.selfplay import play_game


def _policy_factory(name: str, **options):
    """Build a ``factory(house, seed) -> Policy`` for a named agent."""

    if name == "random":
        return lambda house, seed: RandomPolicy(seed)
    if name == "greedy":
        return lambda house, seed: GreedyPolicy(
            seed, noise=options.get("noise", 0.05)
        )
    if name in ("mcts", "uniform-mcts"):
        simulations = options.get("simulations", 24)
        return lambda house, seed: MCTS(simulations=simulations, seed=seed)
    raise SystemExit(f"unknown policy: {name!r}")


def cmd_demo(args: argparse.Namespace) -> int:
    config = GameConfig(seed=args.seed, max_rounds=args.max_rounds)
    factory = _policy_factory(args.policy, simulations=args.simulations)
    policies = {house: factory(house, args.seed + i) for i, house in enumerate(config.houses)}
    started = time.time()
    record = play_game(policies, config)
    elapsed = time.time() - started
    print(json.dumps(
        {
            "policy": args.policy,
            "winner": record.winner,
            "rounds": record.rounds,
            "steps": record.steps,
            "seconds": round(elapsed, 3),
        },
        indent=2,
    ))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from agot.training.evaluate import evaluate_match

    factory_a = _policy_factory(args.a, simulations=args.simulations)
    factory_b = _policy_factory(args.b, simulations=args.simulations)
    result = evaluate_match(
        factory_a,
        factory_b,
        games=args.games,
        config=GameConfig(seed=args.seed, max_rounds=args.max_rounds),
        seed=args.seed,
    )
    print(json.dumps({"a": args.a, "b": args.b, **result.as_dict()}, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "Training needs PyTorch. Install it with:\n"
            "    pip install -e \".[rl]\""
        )
        return 1
    return _run_training(args)


def _run_training(args: argparse.Namespace) -> int:
    from agot.nn.model import NeuralEvaluator, PolicyValueNet
    from agot.training.replay import ReplayBuffer
    from agot.training.trainer import TrainConfig, Trainer

    config = GameConfig(seed=args.seed, max_rounds=args.max_rounds)
    model = PolicyValueNet(hidden=args.hidden)
    if args.resume:
        model = PolicyValueNet.load(args.resume)
    trainer = Trainer(model, TrainConfig(learning_rate=args.lr, epochs=args.epochs))
    buffer = ReplayBuffer(capacity=args.buffer)

    checkpoint = Path(args.checkpoint)
    for iteration in range(args.iterations):
        iteration_started = time.time()
        for game_index in range(args.games):
            evaluator = NeuralEvaluator(model)
            policies = {
                house: MCTS(
                    evaluator,
                    simulations=args.simulations,
                    seed=args.seed + iteration * 1000 + game_index * 10 + i,
                )
                for i, house in enumerate(config.houses)
            }
            record = play_game(policies, config)
            buffer.extend(record.examples)
        history = trainer.train(buffer)
        last = history[-1] if history else {}
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(checkpoint))
        print(
            f"iter {iteration + 1}/{args.iterations} "
            f"buffer={len(buffer)} "
            f"policy_loss={last.get('policy_loss', 0):.4f} "
            f"value_loss={last.get('value_loss', 0):.4f} "
            f"({time.time() - iteration_started:.1f}s) -> {checkpoint}"
        )
    return 0


def cmd_legal(args: argparse.Namespace) -> int:
    game = Game.new(GameConfig(seed=args.seed, max_rounds=args.max_rounds))
    actions = game.legal_actions()
    print(f"active house: {game.current_player()} ({len(actions)} actions)")
    for action in actions[: args.limit]:
        print(f"  {action}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--seed", type=int, default=0)
        target.add_argument("--max-rounds", type=int, default=10)
        target.add_argument("--simulations", type=int, default=24)

    demo = sub.add_parser("demo", help="play a single game")
    add_common(demo)
    demo.add_argument("--policy", default="greedy", choices=["random", "greedy", "mcts"])
    demo.set_defaults(func=cmd_demo)

    evaluate = sub.add_parser("evaluate", help="play a head-to-head match")
    add_common(evaluate)
    evaluate.add_argument("--a", default="random")
    evaluate.add_argument("--b", default="greedy")
    evaluate.add_argument("--games", type=int, default=10)
    evaluate.set_defaults(func=cmd_evaluate)

    legal = sub.add_parser("legal", help="show the legal actions of a fresh game")
    add_common(legal)
    legal.add_argument("--limit", type=int, default=20)
    legal.set_defaults(func=cmd_legal)

    train = sub.add_parser("train", help="run the self-play training loop (needs torch)")
    add_common(train)
    train.add_argument("--iterations", type=int, default=1)
    train.add_argument("--games", type=int, default=10)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--hidden", type=int, default=256)
    train.add_argument("--buffer", type=int, default=200_000)
    train.add_argument("--resume", default=None)
    train.add_argument("--checkpoint", default="checkpoints/agot.pt")
    train.set_defaults(func=cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
