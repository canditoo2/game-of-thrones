"""Generate training data by playing games against yourself.

The loop is the classic AlphaZero one: for every decision, ask each house's
policy for an action *and* the distribution it favours, play the game to the
end, then label every recorded position with the final outcome. The network is
later trained to reproduce the distributions and predict the outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

from agot.agents.base import Policy, outcome_vector
from agot.agents.mcts import MCTS
from agot.engine.game import Game, GameConfig
from agot.engine.observation import HOUSE_ORDER, observe
from agot.training.replay import Example

#: Hard cap so a bug can never hang a training run forever.
DEFAULT_STEP_LIMIT = 20_000

#: Which driver produced the most recent :func:`batched_self_play` records:
#: ``"rust"`` when the extension handled the request, ``"python"`` when the
#: reference driver did. Logged with a run's timings so a regression can be
#: attributed to the right backend.
LAST_BACKEND = "python"

#: Self-play temperature schedule as ``(last_round, temperature)`` bands: the
#: first band whose ``last_round`` is not exceeded wins. The opening plays hot so
#: the search explores, and everything after the opening is sharpened. The
#: temperature is what the recorded target is built from, so leaving it at a flat
#: ``1.0`` for a whole game -- with only a handful of visits per candidate --
#: trains the network to imitate something very close to noise.
DEFAULT_TEMPERATURE_SCHEDULE: tuple[tuple[int, float], ...] = (
    (3, 1.0),
    (10_000, 0.25),
)


def scheduled_temperature(
    schedule: tuple[tuple[int, float], ...], round_number: int
) -> float:
    """Temperature to use in ``round_number`` (1-based) under ``schedule``."""
    for last_round, temperature in schedule:
        if round_number <= last_round:
            return temperature
    return schedule[-1][1] if schedule else 1.0


@dataclass
class GameRecord:
    """The examples from one game plus a little metadata for logging."""

    examples: list[Example]
    winner: str | None
    steps: int
    rounds: int


def play_game(
    policies: dict[str, Policy],
    config: GameConfig | None = None,
    *,
    step_limit: int = DEFAULT_STEP_LIMIT,
    record: bool = True,
) -> GameRecord:
    """Play one game with ``policies`` (one per house).

    ``record=False`` skips the observation/encoding work for callers that only
    need the winner and step count (evaluation, the demo). The decisions
    themselves still run through the same policies.
    """
    game = Game.new(config or GameConfig())
    records: list[tuple[Example, ...]] = []
    steps = 0

    while not game.is_terminal() and steps < step_limit:
        house = game.current_player()
        if house is None:  # pragma: no cover - engine always advances to someone
            break
        policy = policies[house]
        action, target = policy.decide(game)
        if record:
            observation = observe(game)
            if len(observation.actions) != len(target):
                raise RuntimeError(
                    f"policy {policy.name!r} returned {len(target)} targets for "
                    f"{len(observation.actions)} legal actions"
                )
            records.append(
                (
                    Example(
                        state=observation.state_features,
                        actions=observation.action_features,
                        policy=target.astype("float32"),
                        player=HOUSE_ORDER.index(house),
                        outcome=None,  # type: ignore[arg-type]  # filled in below
                    ),
                )
            )
        game.step(action)
        steps += 1

    outcome = outcome_vector(game)
    examples: list[Example] = []
    for (example,) in records:
        example.outcome = outcome
        examples.append(example)
    return GameRecord(
        examples=examples,
        winner=game.winner(),
        steps=steps,
        rounds=game.state.round,
    )


def self_play(
    policy_factory,
    config: GameConfig | None = None,
    *,
    seed: int = 0,
    step_limit: int = DEFAULT_STEP_LIMIT,
) -> GameRecord:
    """Play a self-play game where one policy controls every seat.

    ``policy_factory(house, seed) -> Policy`` is called once per house so that
    each seat gets an independently seeded instance.
    """
    config = config or GameConfig(seed=seed)
    policies = {
        house: policy_factory(house, seed + index)
        for index, house in enumerate(config.houses)
    }
    return play_game(policies, config, step_limit=step_limit)


def _evaluate_leaves(
    evaluator, requests: list[tuple[Game, list]]
) -> list[tuple]:
    """Score leaves in one batch when the evaluator supports it."""
    batch = getattr(evaluator, "evaluate_batch", None)
    if batch is None:
        return [evaluator(game, actions) for game, actions in requests]
    return batch(requests)


def batched_self_play(
    evaluator,
    configs: list[GameConfig],
    *,
    simulations: int,
    temperature: float = 1.0,
    temperature_schedule: tuple[tuple[int, float], ...] | None = None,
    determinizations: int = 1,
    seed: int = 0,
    opponents: list[dict[str, Policy]] | None = None,
    record_opponents: bool = False,
    step_limit: int = DEFAULT_STEP_LIMIT,
) -> list[GameRecord]:
    """Play several games in lockstep, batching every network call.

    All seats default to MCTS over the shared ``evaluator``. Instead of letting
    one game finish before the next starts, the driver advances every game by
    one simulation, gathers the leaves that need a network score, and evaluates
    them in a single forward pass. A batch of ``N`` games therefore scores ``N``
    leaves per call instead of one, which is where most of the CPU speed-up
    comes from.

    ``opponents[i]`` may map a house in game ``i`` to a scripted
    :class:`~agot.agents.base.Policy` (for example greedy). Those seats decide
    immediately and are not searched. Their decisions are recorded only when
    ``record_opponents`` is set, so a learner is not trained to imitate a
    scripted baseline by accident.

    ``determinizations > 1`` gives every seat PIMC: each decision searches one
    resampled hidden world per determinization (batched together with every
    other world and game) and aggregates the visit counts, so the recorded
    target never depends on information the house cannot see.

    ``temperature_schedule`` is a ``(last_round, temperature)`` sequence applied
    by round; ``temperature`` is used for every decision when it is ``None``.
    The moving action *and* the recorded target come from the searched visit
    distribution at that temperature, so a flat 1.0 all game records a
    near-uniform target whenever visits are scarce. See
    :data:`DEFAULT_TEMPERATURE_SCHEDULE`.
    """
    # Fast path: when the Rust extension is installed and the request needs
    # nothing it does not implement, the whole game runs natively and only the
    # leaf batches cross into Python. `try_batched_self_play` returns None when
    # it cannot help, and the reference driver below takes over unchanged.
    from agot.training import rust_engine

    global LAST_BACKEND
    rust_records = rust_engine.try_batched_self_play(
        evaluator,
        configs,
        simulations=simulations,
        temperature=temperature,
        temperature_schedule=temperature_schedule,
        determinizations=determinizations,
        seed=seed,
        opponents=opponents,
        record_opponents=record_opponents,
        step_limit=step_limit,
    )
    if rust_records is not None:
        LAST_BACKEND = "rust"
        return rust_records
    LAST_BACKEND = "python"

    # The reference driver is the fallback for hand-written evaluators and
    # mixed configurations. It drives the same Rust game as everything else.
    games = [Game.new(config) for config in configs]
    seat_policies: list[dict[str, Policy]] = []
    for index, game in enumerate(games):
        table: dict[str, Policy] = {}
        for seat, house in enumerate(game.config.houses):
            opponent = None if opponents is None else opponents[index].get(house)
            if opponent is not None:
                table[house] = opponent
            else:
                table[house] = MCTS(
                    evaluator,
                    simulations=simulations,
                    temperature=temperature,
                    determinizations=determinizations,
                    seed=seed + index * 101 + seat,
                )
        seat_policies.append(table)

    histories: list[list[Example]] = [[] for _ in games]
    steps = [0] * len(games)
    rounds = [0] * len(games)
    winners: list[str | None] = [None] * len(games)
    finished = [False] * len(games)
    worlds: list[list] = [[] for _ in games]
    active_search: list[MCTS | None] = [None] * len(games)
    active_house: list[str | None] = [None] * len(games)
    per_world = max(1, simulations // max(1, determinizations))

    while True:
        active: list[int] = []
        for index, game in enumerate(games):
            if finished[index]:
                continue
            if game.is_terminal() or steps[index] >= step_limit:
                finished[index] = True
                winners[index] = game.winner()
                continue
            house = game.current_player()
            if house is None:  # pragma: no cover - engine always advances
                finished[index] = True
                winners[index] = game.winner()
                continue
            policy = seat_policies[index][house]
            active_house[index] = house
            active_search[index] = None
            worlds[index] = []
            if isinstance(policy, MCTS):
                roots = policy.open_search_worlds(game)
                if not roots:  # pragma: no cover - terminal root
                    finished[index] = True
                    winners[index] = game.winner()
                    continue
                worlds[index] = roots
                active_search[index] = policy
            active.append(index)
        if not active:
            break

        searching = [index for index in active if active_search[index] is not None]
        if searching:
            for _ in range(per_world):
                pending = []
                for index in searching:
                    policy = active_search[index]
                    for root in worlds[index]:
                        pending.append((policy, policy.prepare_simulation(root)))
                requests = [
                    (policy, simulation)
                    for policy, simulation in pending
                    if simulation.leaf is not None
                ]
                if requests:
                    leaves = [
                        (simulation.leaf.game, simulation.actions)
                        for _, simulation in requests
                    ]
                    for (policy, simulation), (priors, value) in zip(
                        requests, _evaluate_leaves(evaluator, leaves), strict=True
                    ):
                        policy.finish_simulation(simulation, priors, value)
                for policy, simulation in pending:
                    if simulation.leaf is None:
                        policy.finish_simulation(simulation)

        for index in active:
            game = games[index]
            house = active_house[index]
            policy = seat_policies[index][house]
            if isinstance(policy, MCTS):
                result = policy.aggregate_roots(worlds[index], house)
                move_temperature = (
                    scheduled_temperature(
                        temperature_schedule, game.state.round
                    )
                    if temperature_schedule
                    else None
                )
                action, target = policy.pick(result, move_temperature)
            else:
                action, target = policy.decide(game)
            if record_opponents or isinstance(policy, MCTS):
                observation = observe(game)
                if len(observation.actions) != len(target):
                    raise RuntimeError(
                        f"policy {policy.name!r} returned {len(target)} targets "
                        f"for {len(observation.actions)} legal actions"
                    )
                histories[index].append(
                    Example(
                        state=observation.state_features,
                        actions=observation.action_features,
                        policy=target.astype("float32"),
                        player=HOUSE_ORDER.index(house),
                        outcome=None,  # type: ignore[arg-type]  # filled below
                    )
                )
            game.step(action)
            steps[index] += 1
            rounds[index] = game.state.round

    records: list[GameRecord] = []
    for index, game in enumerate(games):
        outcome = outcome_vector(game)
        examples: list[Example] = []
        for example in histories[index]:
            example.outcome = outcome
            examples.append(example)
        records.append(
            GameRecord(
                examples=examples,
                winner=winners[index],
                steps=steps[index],
                rounds=rounds[index],
            )
        )
    return records
