"""Command-line entry point.

Examples::

    agot demo --policy greedy --seed 1
    agot evaluate --a random --b greedy --games 20
    agot train --games 50 --simulations 32 --epochs 2
    agot report --checkpoint checkpoints/agot.pt --open

The ``train`` subcommand is the only one that needs PyTorch.
"""

from __future__ import annotations

import argparse
import json
import time
import webbrowser
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
        determinizations = options.get("determinizations", 1)
        return lambda house, seed: MCTS(
            simulations=simulations,
            determinizations=determinizations,
            seed=seed,
        )
    raise SystemExit(f"unknown policy: {name!r}")


def cmd_demo(args: argparse.Namespace) -> int:
    config = GameConfig(seed=args.seed, max_rounds=args.max_rounds)
    factory = _policy_factory(
        args.policy,
        simulations=args.simulations,
        determinizations=args.determinizations,
    )
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

    factory_a = _policy_factory(
        args.a, simulations=args.simulations, determinizations=args.determinizations
    )
    factory_b = _policy_factory(
        args.b, simulations=args.simulations, determinizations=args.determinizations
    )
    result = evaluate_match(
        factory_a,
        factory_b,
        games=args.games,
        config=GameConfig(seed=args.seed, max_rounds=args.max_rounds),
        seed=args.seed,
    )
    print(json.dumps({"a": args.a, "b": args.b, **result.as_dict()}, indent=2))
    return 0


def cmd_gauntlet(args: argparse.Namespace) -> int:
    """Measure one candidate's single-seat win rate against a field."""
    from agot.training.evaluate import evaluate_vs_field

    candidate = _policy_factory(
        args.candidate,
        simulations=args.simulations,
        determinizations=args.determinizations,
    )
    opponents = {
        name: _policy_factory(
            name, simulations=args.simulations, determinizations=args.determinizations
        )
        for name in args.opponents.split(",")
        if name
    }
    result = evaluate_vs_field(
        candidate,
        opponents,
        games=args.games,
        config=GameConfig(seed=args.seed, max_rounds=args.max_rounds),
        seed=args.seed,
    )
    print(json.dumps({"candidate": args.candidate, **result.as_dict()}, indent=2))
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
    from agot.nn.model import PolicyValueNet
    from agot.training.checkpoint import RunLog, RunState, log_path, state_path
    from agot.training.parallel import (
        SelfPlayPlan,
        create_executor,
        parallel_self_play,
        worker_count,
    )
    from agot.training.replay import ReplayBuffer
    from agot.training.trainer import TrainConfig, Trainer

    if args.train_threads > 0:
        # Small mini-batches are faster on a few threads than on every core; the
        # self-play workers are separate processes with one thread each.
        import torch

        torch.set_num_threads(args.train_threads)

    config = GameConfig(seed=args.seed, max_rounds=args.max_rounds)
    model = (
        PolicyValueNet.load(args.resume)
        if args.resume
        else PolicyValueNet(hidden=args.hidden)
    )
    trainer = Trainer(model, TrainConfig(learning_rate=args.lr, epochs=args.epochs))
    buffer = ReplayBuffer(capacity=args.buffer)

    checkpoint = Path(args.checkpoint)
    best_checkpoint = (
        Path(args.best_checkpoint)
        if args.best_checkpoint
        else checkpoint.with_name(f"{checkpoint.stem}-best{checkpoint.suffix}")
    )
    state_file = state_path(checkpoint)
    log = RunLog(log_path(checkpoint))

    run = RunState()
    if args.resume:
        resumed = RunState.load(state_path(args.resume))
        if resumed is not None:
            run = resumed
            trainer.load_state_dict(run.trainer)
            print(
                f"resumed {args.resume} at iter {run.iteration} "
                f"(best={run.best_rate:.3f}); replay buffer starts empty",
                flush=True,
            )
        else:
            print(
                f"resumed {args.resume} weights (no companion state file; "
                "iteration counter and best-eval reset)",
                flush=True,
            )
    best_rate = run.best_rate

    plan = SelfPlayPlan(
        games=args.games,
        max_rounds=args.max_rounds,
        simulations=args.simulations,
        determinizations=args.determinizations,
        seed=args.seed,
        workers=args.workers,
        batch_games=args.batch_games,
        opponent_rate=args.opponent_rate,
        temperature=args.temperature,
        temperature_rounds=args.temperature_rounds,
        final_temperature=args.final_temperature,
    )

    if run.iteration >= args.iterations:
        print(
            f"nothing to do: already at iter {run.iteration} "
            f"(of {args.iterations})",
            flush=True,
        )
        return 0

    # One long-lived pool for the whole run: re-spawning workers and re-importing
    # torch every iteration is pure overhead.
    workers = worker_count(plan, args.games)
    executor = create_executor(workers) if workers > 1 else None
    try:
        for iteration in range(run.iteration, args.iterations):
            iteration_started = time.time()
            plan.seed = args.seed + iteration * 100_000
            records = parallel_self_play(
                model, plan, base_config=config, executor=executor
            )
            for record in records:
                buffer.extend(record.examples)
            history = trainer.train(buffer)
            last = history[-1] if history else {}
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(checkpoint))
            line = (
                f"iter {iteration + 1}/{args.iterations} "
                f"games={len(records)} buffer={len(buffer)} "
                f"policy_loss={last.get('policy_loss', 0):.4f} "
                f"value_loss={last.get('value_loss', 0):.4f}"
            )
            record = {
                "iteration": iteration + 1,
                "games": len(records),
                "buffer": len(buffer),
                "policy_loss": last.get("policy_loss"),
                "value_loss": last.get("value_loss"),
            }
            if args.eval_games > 0 and (iteration + 1) % max(1, args.eval_every) == 0:
                result = _evaluate_model(model, config, args)
                line += (
                    f" eval={result.win_rate:.3f} "
                    f"ci=[{result.ci_low:.2f},{result.ci_high:.2f}]"
                )
                record["eval"] = result.win_rate
                record["eval_low"] = result.ci_low
                record["eval_high"] = result.ci_high
                if result.win_rate > best_rate:
                    best_rate = result.win_rate
                    model.save(str(best_checkpoint))
                    line += f" -> best {best_checkpoint}"
            elapsed = time.time() - iteration_started
            record["seconds"] = round(elapsed, 3)
            log.append(record)
            run.iteration = iteration + 1
            run.best_rate = best_rate
            run.trainer = trainer.state_dict()
            run.save(state_file)
            print(f"{line} ({elapsed:.1f}s) -> {checkpoint}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    return 0


def _evaluate_model(model, config: GameConfig, args: argparse.Namespace):
    """Single-seat win rate of ``model`` against random and greedy opponents."""
    from agot.agents import MCTS, GreedyPolicy, RandomPolicy
    from agot.nn.model import NeuralEvaluator
    from agot.training.evaluate import evaluate_vs_field

    evaluator = NeuralEvaluator(model)

    # The gauntlet must measure the agent that self-play actually searched as,
    # so it defaults to the training search budget rather than a cheaper one.
    # A mismatch silently reports a weaker policy than the one being trained.
    simulations = getattr(args, "eval_simulations", None) or args.simulations

    def candidate(house: str, seed: int):
        return MCTS(
            evaluator,
            simulations=simulations,
            determinizations=args.determinizations,
            temperature=0.0,
            seed=seed,
        )

    opponents = {
        "random": lambda house, seed: RandomPolicy(seed),
        "greedy": lambda house, seed: GreedyPolicy(seed),
    }
    return evaluate_vs_field(
        candidate,
        opponents,
        games=args.eval_games,
        config=config,
        seed=args.seed + 999,
    )


def cmd_report(args: argparse.Namespace) -> int:
    """Render a training run's JSONL log as a self-contained HTML page.

    With ``--watch`` the page is regenerated on an interval, which is what keeps
    an open tab current: the page is a static file and cannot read the log by
    itself, so the browser's own reload only re-reads whatever was last written.
    """
    from agot.training.report import log_path_for, render_file

    log = Path(args.log) if args.log else log_path_for(args.checkpoint)
    refresh = args.refresh
    if args.watch > 0 and refresh > 0:
        # Never let the page wait longer than the file takes to change.
        refresh = min(refresh, args.watch)
    opened = False
    while True:
        out = render_file(
            log,
            out=args.out,
            refresh=refresh,
            target_iterations=args.iterations,
        )
        print(f"[{time.strftime('%H:%M:%S')}] {log} -> {out}", flush=True)
        if args.open and not opened:
            webbrowser.open(out.resolve().as_uri())
            opened = True
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


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

    def add_common(
        target: argparse.ArgumentParser,
        *,
        simulations: int = 24,
        determinizations: int = 1,
    ) -> None:
        target.add_argument("--seed", type=int, default=0)
        target.add_argument("--max-rounds", type=int, default=10)
        target.add_argument("--simulations", type=int, default=simulations)
        target.add_argument(
            "--determinizations",
            type=int,
            default=determinizations,
            help="hidden states sampled per move (PIMC); >1 hides face-down info",
        )

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

    gauntlet = sub.add_parser(
        "gauntlet",
        help="single-seat win rate of one policy against a field",
    )
    add_common(gauntlet)
    gauntlet.add_argument("--candidate", default="greedy")
    gauntlet.add_argument(
        "--opponents",
        default="random",
        help="comma-separated policy names filling the other five seats",
    )
    gauntlet.add_argument("--games", type=int, default=60)
    gauntlet.set_defaults(func=cmd_gauntlet)

    legal = sub.add_parser("legal", help="show the legal actions of a fresh game")
    add_common(legal)
    legal.add_argument("--limit", type=int, default=20)
    legal.set_defaults(func=cmd_legal)

    report = sub.add_parser(
        "report",
        help="render a run's *.log.jsonl as a self-contained HTML page",
    )
    report.add_argument(
        "--checkpoint",
        default="checkpoints/agot.pt",
        help="checkpoint whose companion *.log.jsonl to read",
    )
    report.add_argument(
        "--log",
        default=None,
        help="read this JSONL directly instead of deriving it from --checkpoint",
    )
    report.add_argument(
        "--out",
        default=None,
        help="output HTML path (default: the log with an .html suffix)",
    )
    report.add_argument(
        "--refresh",
        type=int,
        default=30,
        help=(
            "seconds between the page's own reloads (0 disables); clamped to "
            "--watch so the tab never waits longer than the file takes to change"
        ),
    )
    report.add_argument(
        "--watch",
        type=int,
        default=0,
        help=(
            "regenerate the page every N seconds so an open tab keeps up "
            "(0 = render once and exit)"
        ),
    )
    report.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="total planned iterations, used for the ETA readout (0 = unknown)",
    )
    report.add_argument(
        "--open",
        action="store_true",
        help="open the page in the default browser",
    )
    report.set_defaults(func=cmd_report)

    train = sub.add_parser("train", help="run the self-play training loop (needs torch)")
    # Training needs a search deep enough to produce a target worth imitating:
    # at 24 simulations spread over 1 determinization a leaf's candidates get
    # roughly one visit each, so the recorded policy is close to uniform. The
    # demo/evaluate defaults stay cheap because those are for eyeballing.
    add_common(train, simulations=128, determinizations=2)
    train.add_argument("--iterations", type=int, default=1)
    train.add_argument("--games", type=int, default=10)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--hidden", type=int, default=256)
    train.add_argument("--buffer", type=int, default=200_000)
    train.add_argument("--resume", default=None)
    train.add_argument("--checkpoint", default="checkpoints/agot.pt")
    train.add_argument(
        "--best-checkpoint",
        default=None,
        help="where to save the best-evaluated net (default: <checkpoint>-best)",
    )
    train.add_argument(
        "--workers",
        type=int,
        default=0,
        help="self-play processes (0 = one per CPU; 1 = in-process)",
    )
    train.add_argument(
        "--batch-games",
        type=int,
        default=4,
        help="games advanced in lockstep per worker",
    )
    train.add_argument(
        "--opponent-rate",
        type=float,
        default=0.0,
        help="chance a seat is filled by the greedy baseline instead of the net",
    )
    train.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="self-play temperature for the opening rounds",
    )
    train.add_argument(
        "--temperature-rounds",
        type=int,
        default=3,
        help=(
            "rounds played at --temperature before cooling to "
            "--final-temperature"
        ),
    )
    train.add_argument(
        "--final-temperature",
        type=float,
        default=0.25,
        help=(
            "self-play temperature once past --temperature-rounds; the target "
            "is built from it, so 1.0 for a whole game trains on noise"
        ),
    )
    train.add_argument(
        "--train-threads",
        type=int,
        default=4,
        help=(
            "torch threads in the training process; small batches are faster "
            "with a handful of threads than with every core (0 = torch default)"
        ),
    )
    train.add_argument(
        "--eval-games",
        type=int,
        default=0,
        help="gauntlet games per evaluation (0 disables periodic evaluation)",
    )
    train.add_argument("--eval-every", type=int, default=5)
    train.add_argument(
        "--eval-simulations",
        type=int,
        default=None,
        help="simulations for the periodic gauntlet (default: same as --simulations)",
    )
    train.set_defaults(func=cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
