"""Parallel self-play across CPU processes.

The rules engine and the observation encoder are pure Python, and the pointer
net is small, so on a many-core CPU the largest throughput win comes from
running whole games in separate processes: each core gets its own interpreter
and its own pending forward passes instead of sharing one.

A worker rebuilds the network from a state dict — once via a pool initializer for
a short-lived pool, or per task for a long-lived one (see :func:`create_executor`,
which lets a training run reuse its workers instead of paying process spawn and
``torch`` import every iteration) — then plays a chunk of games with
:func:`agot.training.selfplay.batched_self_play` so that its own games are still
batched together. ``opponent_rate`` hands a random fraction of the seats to a
scripted greedy baseline, which keeps the self-play from chasing only its own
style.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from agot.agents import GreedyPolicy
from agot.engine.game import GameConfig
from agot.training.selfplay import GameRecord, batched_self_play

#: Populated in each worker process by :func:`_initialise`.
_WORKER: dict = {}


@dataclass
class SelfPlayPlan:
    """How to generate one iteration of self-play data."""

    games: int = 16
    max_rounds: int = 10
    simulations: int = 32
    determinizations: int = 1
    temperature: float = 1.0
    seed: int = 0
    #: 0 picks ``cpus - 1`` worker processes. Each is single-threaded on purpose:
    #: many small nets are faster as N processes than as one multi-threaded run.
    workers: int = 0
    #: Games advanced in lockstep inside a worker (amortises the forward pass).
    batch_games: int = 4
    #: Probability that any single seat is filled by the greedy baseline.
    opponent_rate: float = 0.0
    greedy_noise: float = 0.05
    #: Rounds played at ``temperature`` before dropping to ``final_temperature``.
    #: The opening needs exploration; the rest of the game needs a target sharp
    #: enough to be worth imitating (see ``selfplay.DEFAULT_TEMPERATURE_SCHEDULE``).
    temperature_rounds: int = 3
    final_temperature: float = 0.25

    def temperature_schedule(self) -> tuple[tuple[int, float], ...]:
        """``(last_round, temperature)`` bands handed to the self-play loop."""
        return (
            (self.temperature_rounds, self.temperature),
            (1 << 30, self.final_temperature),
        )


def _options(plan: SelfPlayPlan) -> dict:
    return {
        "simulations": plan.simulations,
        "determinizations": plan.determinizations,
        "temperature": plan.temperature,
        "temperature_schedule": plan.temperature_schedule(),
        "seed": plan.seed,
        "batch_games": max(1, plan.batch_games),
        "opponent_rate": plan.opponent_rate,
        "greedy_noise": plan.greedy_noise,
    }


def _initialise(payload: dict) -> None:
    """Build the worker's network once, then reuse it for every task."""
    import torch

    from agot.nn.model import NeuralEvaluator, PolicyValueNet

    torch.set_num_threads(1)
    model = PolicyValueNet(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    _WORKER["evaluator"] = NeuralEvaluator(model)


def _scripted_seats(
    config: GameConfig, options: dict, rng: random.Random
) -> dict:
    seats = {}
    for house in config.houses:
        if rng.random() < options["opponent_rate"]:
            seats[house] = GreedyPolicy(
                rng.randrange(1 << 30), noise=options["greedy_noise"]
            )
    return seats


def _play_chunk(configs: list[GameConfig], options: dict) -> list[GameRecord]:
    evaluator = _WORKER["evaluator"]
    rng = random.Random(options["seed"])
    batch = options["batch_games"]
    records: list[GameRecord] = []
    for start in range(0, len(configs), batch):
        chunk = configs[start : start + batch]
        opponents = None
        if options["opponent_rate"] > 0:
            opponents = [
                _scripted_seats(config, options, rng) for config in chunk
            ]
        records.extend(
            batched_self_play(
                evaluator,
                chunk,
                simulations=options["simulations"],
                determinizations=options["determinizations"],
                temperature=options["temperature"],
                temperature_schedule=options["temperature_schedule"],
                seed=options["seed"] + start,
                opponents=opponents,
            )
        )
    return records


def _apply_model(state: dict, model_config: dict) -> None:
    """Install (or refresh) the worker's network from a state dict.

    Used with a long-lived pool: the weights change every training iteration, so
    they are shipped with each task instead of being fixed at pool creation. The
    module is rebuilt only when its configuration changes, which keeps the
    per-iteration update cheap.
    """
    import torch

    from agot.nn.model import NeuralEvaluator, PolicyValueNet

    torch.set_num_threads(1)
    model = _WORKER.get("model")
    if model is None or _WORKER.get("model_config") != model_config:
        model = PolicyValueNet(**model_config)
        _WORKER["model"] = model
        _WORKER["model_config"] = model_config
    model.load_state_dict(state)
    model.eval()
    _WORKER["evaluator"] = NeuralEvaluator(model)


def _play_with_state(
    state: dict, model_config: dict, configs: list[GameConfig], options: dict
) -> list[GameRecord]:
    _apply_model(state, model_config)
    return _play_chunk(configs, options)


def worker_count(plan: SelfPlayPlan, games: int | None = None) -> int:
    """How many processes :func:`parallel_self_play` would use."""
    total = plan.games if games is None else games
    workers = plan.workers or max(1, (os.cpu_count() or 1) - 1)
    return max(1, min(workers, max(1, total)))


def create_executor(workers: int) -> ProcessPoolExecutor:
    """A reusable worker pool, so iterations do not pay spawn + torch import each.

    Pass the result as ``executor`` to :func:`parallel_self_play` and shut it
    down once the training run is over.
    """
    return ProcessPoolExecutor(max_workers=max(1, workers))


def _model_config(model) -> dict:
    return {
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "hidden": model.hidden,
        "trunk_layers": model.trunk_layers,
        "n_houses": model.n_houses,
    }


def parallel_self_play(
    model,
    plan: SelfPlayPlan,
    *,
    base_config: GameConfig | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> list[GameRecord]:
    """Generate ``plan.games`` self-play games, using several processes.

    ``executor`` is optional. When given, it is reused across calls (the model
    state is shipped per task), which avoids re-spawning workers and re-importing
    torch on every training iteration. When omitted, a short-lived pool is built
    for this call exactly as before.
    """
    base = base_config or GameConfig()
    configs = [
        GameConfig(
            houses=base.houses,
            max_rounds=plan.max_rounds,
            victory_castles=base.victory_castles,
            seed=plan.seed + index,
            use_westeros_decks=base.use_westeros_decks,
        )
        for index in range(plan.games)
    ]
    if not configs:
        return []

    workers = worker_count(plan, len(configs))
    options = _options(plan)

    if workers == 1:
        from agot.nn.model import NeuralEvaluator

        _WORKER["evaluator"] = NeuralEvaluator(model)
        return _play_chunk(configs, options)

    # Round-robin so every worker sees a spread of game seeds, not a contiguous
    # block that could correlate their data.
    chunks = [configs[index::workers] for index in range(workers)]
    chunks = [chunk for chunk in chunks if chunk]
    model_config = _model_config(model)

    records: list[GameRecord] = []
    if executor is not None:
        state = model.state_dict()
        futures = [
            executor.submit(_play_with_state, state, model_config, chunk, options)
            for chunk in chunks
        ]
        for future in futures:
            records.extend(future.result())
        return records

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialise,
        initargs=({"model_state": model.state_dict(), "model_config": model_config},),
    ) as pool:
        futures = [pool.submit(_play_chunk, chunk, options) for chunk in chunks]
        for future in futures:
            records.extend(future.result())
    return records
