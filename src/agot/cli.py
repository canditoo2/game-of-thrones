"""Command-line entry point.

Examples::

    agot demo --policy greedy --seed 1
    agot evaluate --a random --b greedy --games 20
    agot gauntlet --candidate net:checkpoints/run4/agot-best.pt --opponents random
    agot round-robin --checkpoints checkpoints/run4/agot.pt checkpoints/run4/agot-best.pt
    agot play --checkpoint checkpoints/run4/agot-best.pt --house stark
    agot train --iterations 100 --games 48 --workers 12 --checkpoint checkpoints/agot.pt
    agot report --checkpoint checkpoints/agot.pt --open

Every policy argument also accepts ``net:<checkpoint path>``, which searches
with MCTS over that saved network. The ``train`` subcommand and ``net:``
policies are the only parts that need PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import webbrowser
from pathlib import Path

from agot.agents import MCTS, GreedyPolicy, RandomPolicy
from agot.engine.game import DEFAULT_HOUSES, Game, GameConfig

#: Loaded networks, keyed by checkpoint path. A gauntlet builds a fresh policy
#: per seat per game, so the checkpoint is loaded once and the model shared.
_NET_CACHE: dict[str, object] = {}


def _load_net(path: str):
    """Load (and cache) the network behind a ``net:<path>`` policy."""
    try:
        from agot.nn.model import PolicyValueNet
    except ImportError:
        raise SystemExit(
            "net policies need PyTorch: pip install -e \".[rl]\""
        ) from None
    model = _NET_CACHE.get(path)
    if model is None:
        try:
            model = PolicyValueNet.load(path)
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from None
        _NET_CACHE[path] = model
    return model


def _policy_factory(name: str, **options):
    """Build a ``factory(house, seed) -> Policy`` for a named agent.

    Besides ``random``, ``greedy`` and ``mcts``, a name of the form
    ``net:<checkpoint path>`` searches with MCTS over that saved network.
    """

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
    if name == "net" or name.startswith("net:"):
        path = name.partition(":")[2] or options.get("checkpoint")
        if not path:
            raise SystemExit(
                "the net policy needs a checkpoint: use net:<path> "
                "(e.g. --candidate net:checkpoints/run4/agot-best.pt)"
            )
        model = _load_net(path)
        from agot.nn.model import NeuralEvaluator

        evaluator = NeuralEvaluator(model)
        simulations = options.get("simulations", 24)
        determinizations = options.get("determinizations", 1)
        temperature = options.get("temperature", 0.0)
        return lambda house, seed: MCTS(
            evaluator,
            simulations=simulations,
            determinizations=determinizations,
            temperature=temperature,
            seed=seed,
        )
    raise SystemExit(f"unknown policy: {name!r}")


def cmd_demo(args: argparse.Namespace) -> int:
    # Imported here, not at module scope: `agot.training` eagerly imports the
    # torch trainer, and a demo/gauntlet must not pay a 2 s torch import.
    from agot.training.selfplay import play_game

    config = GameConfig(
        seed=args.seed, max_rounds=args.max_rounds, sequential_westeros=True, deferred_battles=True
    )
    factory = _policy_factory(
        args.policy,
        simulations=args.simulations,
        determinizations=args.determinizations,
    )
    policies = {house: factory(house, args.seed + i) for i, house in enumerate(config.houses)}
    started = time.time()
    record = play_game(policies, config, record=False)
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


def cmd_play(args: argparse.Namespace) -> int:
    """Open the desktop game window."""
    from agot.gui import launch

    try:
        launch(
            checkpoint=args.checkpoint,
            house=args.house,
            opponents=args.opponents,
            simulations=args.simulations,
            determinizations=args.determinizations,
            seed=args.seed,
            max_rounds=args.max_rounds,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from None
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = GameConfig(
        seed=args.seed, max_rounds=args.max_rounds, sequential_westeros=True, deferred_battles=True
    )
    if args.a.startswith("net:") and args.b.startswith("net:"):
        from agot.training.evaluate import evaluate_models_match

        result = evaluate_models_match(
            _load_net(args.a.partition(":")[2]),
            _load_net(args.b.partition(":")[2]),
            simulations=args.simulations,
            determinizations=args.determinizations,
            games=args.games,
            config=config,
            seed=args.seed,
        )
        print(json.dumps({"a": args.a, "b": args.b, **result.as_dict()}, indent=2))
        return 0

    if args.a.startswith("net:") and args.b in ("random", "greedy"):
        from agot.training.evaluate import evaluate_model_vs_factory

        result = evaluate_model_vs_factory(
            _load_net(args.a.partition(":")[2]),
            _policy_factory(
                args.b,
                simulations=args.simulations,
                determinizations=args.determinizations,
            ),
            simulations=args.simulations,
            determinizations=args.determinizations,
            games=args.games,
            config=config,
            seed=args.seed,
        )
        print(json.dumps({"a": args.a, "b": args.b, **result.as_dict()}, indent=2))
        return 0

    from agot.training.evaluate import evaluate_match

    factory_a = _policy_factory(
        args.a, simulations=args.simulations, determinizations=args.determinizations
    )
    factory_b = _policy_factory(
        args.b, simulations=args.simulations, determinizations=args.determinizations
    )
    result = evaluate_match(factory_a, factory_b, games=args.games, config=config, seed=args.seed)
    print(json.dumps({"a": args.a, "b": args.b, **result.as_dict()}, indent=2))
    return 0


def cmd_gauntlet(args: argparse.Namespace) -> int:
    """Measure one candidate's single-seat win rate against a field."""
    names = [name for name in args.opponents.split(",") if name]
    config = GameConfig(
        seed=args.seed, max_rounds=args.max_rounds, sequential_westeros=True, deferred_battles=True
    )

    # A saved network against scripted opponents is the training gauntlet and
    # the common CLI case: run it on the Rust driver instead of Python MCTS.
    if args.candidate.startswith("net:") and all(
        name in ("random", "greedy") for name in names
    ):
        from agot.agents import GreedyPolicy, RandomPolicy
        from agot.training.evaluate import evaluate_net_vs_field

        model = _load_net(args.candidate.partition(":")[2])
        factories = {}
        if "random" in names:
            factories["random"] = lambda house, seed: RandomPolicy(seed)
        if "greedy" in names:
            factories["greedy"] = lambda house, seed: GreedyPolicy(seed)
        result = evaluate_net_vs_field(
            model,
            factories,
            simulations=args.simulations,
            determinizations=args.determinizations,
            games=args.games,
            config=config,
            seed=args.seed,
        )
        print(json.dumps({"candidate": args.candidate, **result.as_dict()}, indent=2))
        return 0

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
        for name in names
    }
    result = evaluate_vs_field(
        candidate,
        opponents,
        games=args.games,
        config=config,
        seed=args.seed,
    )
    print(json.dumps({"candidate": args.candidate, **result.as_dict()}, indent=2))
    return 0


def cmd_round_robin(args: argparse.Namespace) -> int:
    """League table: every checkpoint plays one seat against the others.

    Each checkpoint is evaluated in turn as a single rotating seat; the other
    five seats are filled round-robin from the remaining checkpoints and any
    anchors, and the win rate comes with a Wilson interval. This is the
    robustness check (G2): a checkpoint that only beats one particular past
    version is not a strong agent.
    """
    from agot.training.evaluate import evaluate_models_vs_field

    checkpoints = list(args.checkpoints)
    if len(checkpoints) < 2:
        raise SystemExit("round-robin needs at least two checkpoints")
    anchors = [name for name in args.anchors.split(",") if name]
    models = [_load_net(path) for path in checkpoints]
    anchor_factories = {
        anchor: _policy_factory(
            anchor,
            simulations=args.simulations,
            determinizations=args.determinizations,
        )
        for anchor in anchors
    }
    results = {}
    for index, candidate in enumerate(checkpoints):
        opponents = {
            other: models[other_index]
            for other_index, other in enumerate(checkpoints)
            if other_index != index
        }
        result = evaluate_models_vs_field(
            models[index],
            opponents,
            anchors=anchor_factories,
            simulations=args.simulations,
            determinizations=args.determinizations,
            games=args.games,
            config=GameConfig(
                seed=args.seed,
                max_rounds=args.max_rounds,
                sequential_westeros=True, deferred_battles=True,
            ),
            seed=args.seed,
        )
        results[candidate] = result.as_dict()
    print(
        json.dumps(
            {
                "checkpoints": checkpoints,
                "games_per_candidate": args.games,
                "anchors": anchors,
                "results": results,
            },
            indent=2,
        )
    )
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

    train_threads = args.train_threads
    if train_threads is None:
        try:
            usable = len(os.sched_getaffinity(0))
        except AttributeError:
            usable = os.cpu_count() or 4
        # Small mini-batches are faster on a few threads than on every core; the
        # self-play workers are separate processes with one thread each.
        train_threads = max(1, min(8, usable))
    if train_threads > 0:
        import torch

        torch.set_num_threads(train_threads)

    config = GameConfig(
        seed=args.seed, max_rounds=args.max_rounds, sequential_westeros=True, deferred_battles=True
    )
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
    gate_failures = 0
    stop_training = False
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
            gate_due = (
                (iteration + 1) % max(1, args.eval_every) == 0
                and (args.eval_games > 0 or args.gate_games > 0)
            )
            if gate_due:
                result = None
                if args.eval_games > 0:
                    result = _evaluate_model(model, config, args)
                    line += (
                        f" eval={result.win_rate:.3f} "
                        f"ci=[{result.ci_low:.2f},{result.ci_high:.2f}]"
                    )
                    record["eval"] = result.win_rate
                    record["eval_low"] = result.ci_low
                    record["eval_high"] = result.ci_high
                    # The random/greedy anchor is a sanity check, not the
                    # relative gate: do not let it replace a gate-approved best.
                    if args.gate_games <= 0 and result.win_rate > best_rate:
                        best_rate = result.win_rate
                        model.save(str(best_checkpoint))
                        line += f" -> best {best_checkpoint}"

                if args.gate_games > 0:
                    from agot.training.evaluate import (
                        evaluate_models_match,
                        wilson_interval,
                    )

                    if best_checkpoint.exists():
                        previous_best = PolicyValueNet.load(str(best_checkpoint))
                        match = evaluate_models_match(
                            model,
                            previous_best,
                            simulations=args.simulations,
                            determinizations=args.determinizations,
                            games=args.gate_games,
                            config=config,
                            seed=args.seed + 900_000,
                            batch_games=args.batch_games,
                        )
                        low, high = wilson_interval(match.wins_a, match.games)
                        house_ok = all(
                            match.per_house_a[house]
                            >= match.per_house_b.get(house, 1.0)
                            + args.gate_house_margin
                            for house in match.per_house_a
                        )
                        absolute_ok = all(
                            rate >= args.gate_min_house
                            for rate in match.per_house_a.values()
                        )
                        passed = (
                            match.win_rate_a > args.gate_threshold
                            and house_ok
                            and absolute_ok
                        )
                        record["gate"] = match.win_rate_a
                        record["gate_low"] = low
                        record["gate_high"] = high
                        record["gate_per_house"] = match.per_house_a
                        record["gate_passed"] = passed
                        line += (
                            f" gate={match.win_rate_a:.3f} "
                            f"ci=[{low:.2f},{high:.2f}]"
                        )
                        if passed:
                            best_rate = match.win_rate_a
                            gate_failures = 0
                            model.save(str(best_checkpoint))
                            line += " -> gate best"
                        else:
                            gate_failures += 1
                            line += " -> kept previous best"
                            if (
                                args.gate_patience > 0
                                and gate_failures >= args.gate_patience
                            ):
                                stop_training = True
                                line += " -> stopping after gate failures"
                    else:
                        # Seed the gate once, then only a candidate that beats
                        # it (overall and house by house) may replace it.
                        model.save(str(best_checkpoint))
                        if result is not None:
                            best_rate = result.win_rate
                        record["gate_seeded"] = True
                        gate_failures = 0
                        line += " -> seeded gate best"
            elapsed = time.time() - iteration_started
            record["seconds"] = round(elapsed, 3)
            log.append(record)
            run.iteration = iteration + 1
            run.best_rate = best_rate
            run.trainer = trainer.state_dict()
            run.save(state_file)
            print(f"{line} ({elapsed:.1f}s) -> {checkpoint}", flush=True)
            if stop_training:
                break
    finally:
        if executor is not None:
            executor.shutdown()
    return 0


def _evaluate_model(model, config: GameConfig, args: argparse.Namespace):
    """Single-seat win rate of ``model`` against random and greedy opponents."""
    from agot.agents import GreedyPolicy, RandomPolicy
    from agot.training.evaluate import evaluate_net_vs_field

    # The gauntlet must measure the agent that self-play actually searched as,
    # so it defaults to the training search budget rather than a cheaper one.
    # A mismatch silently reports a weaker policy than the one being trained.
    simulations = getattr(args, "eval_simulations", None) or args.simulations
    opponents = {
        "random": lambda house, seed: RandomPolicy(seed),
        "greedy": lambda house, seed: GreedyPolicy(seed),
    }
    return evaluate_net_vs_field(
        model,
        opponents,
        simulations=simulations,
        determinizations=args.determinizations,
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

    play = sub.add_parser(
        "play",
        help="open the desktop game against a saved network",
    )
    play.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "preselected .pt file or run directory (agot-best.pt preferred); "
            "the setup window can also choose it"
        ),
    )
    play.add_argument(
        "--house",
        default="stark",
        choices=list(DEFAULT_HOUSES),
        help="preselected house; the setup window still asks you to choose",
    )
    play.add_argument(
        "--opponents",
        default="net",
        choices=("net", "greedy", "random"),
        help="how the other five seats are played",
    )
    play.add_argument("--simulations", type=int, default=64)
    play.add_argument(
        "--determinizations",
        type=int,
        default=2,
        help=(
            "hidden worlds per AI move; 2 keeps your face-down orders and bids "
            "hidden, while 1 is faster but lets the search see them"
        ),
    )
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--max-rounds", type=int, default=10)
    play.set_defaults(func=cmd_play)

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

    round_robin = sub.add_parser(
        "round-robin",
        help="league table: each checkpoint plays one seat against the others",
    )
    add_common(round_robin)
    round_robin.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="two or more checkpoints; every one is evaluated against the rest",
    )
    round_robin.add_argument(
        "--games",
        type=int,
        default=30,
        help="games per checkpoint (the league costs games x checkpoints)",
    )
    round_robin.add_argument(
        "--anchors",
        default="random,greedy",
        help="comma-separated extra field policies (empty to disable)",
    )
    round_robin.set_defaults(func=cmd_round_robin)

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
        default=None,
        help=(
            "torch threads in the training process; defaults to min(8, usable "
            "CPUs). Small batches are faster with a handful of threads than "
            "with every core (0 = torch default)"
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
    train.add_argument(
        "--gate-games",
        type=int,
        default=0,
        help=(
            "head-to-head games against the current agot-best.pt; 0 disables the "
            "relative gate and keeps the random/greedy best selection"
        ),
    )
    train.add_argument(
        "--gate-threshold",
        type=float,
        default=0.5,
        help="minimum overall win rate against the previous best to replace it",
    )
    train.add_argument(
        "--gate-min-house",
        type=float,
        default=0.0,
        help=(
            "optional absolute floor for every per-house win rate (0 = only "
            "compare each house with the previous best)"
        ),
    )
    train.add_argument(
        "--gate-house-margin",
        type=float,
        default=0.0,
        help="required advantage over the previous best in every house",
    )
    train.add_argument(
        "--gate-patience",
        type=int,
        default=0,
        help=(
            "stop after this many consecutive gate failures (0 = keep training; "
            "the gate-approved best is preserved either way)"
        ),
    )
    train.set_defaults(func=cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
