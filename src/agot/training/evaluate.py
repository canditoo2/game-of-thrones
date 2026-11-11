"""Head-to-head evaluation between policies.

Both agents play the *same* seats across a match, swapping sides every game, so
that the first-player advantage (and the asymmetric starting positions) cancel
out in expectation.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from agot.agents.base import Policy
from agot.engine.game import GameConfig
from agot.training.selfplay import DEFAULT_STEP_LIMIT, play_game

#: ``(house, seed) -> Policy`` builder, as used by the CLI and self-play.
PolicyFactory = Callable[[str, int], Policy]


@dataclass
class MatchResult:
    """Aggregate outcome of a match."""

    games: int = 0
    wins_a: int = 0
    wins_b: int = 0
    win_rate_a: float = 0.0
    win_rate_b: float = 0.0
    average_steps: float = 0.0
    seats_of_a: list[str] = field(default_factory=list)
    #: Win rate of A when it controls each house (and of B in the mirror).
    per_house_a: dict[str, float] = field(default_factory=dict)
    per_house_b: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "games": self.games,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "win_rate_a": self.win_rate_a,
            "win_rate_b": self.win_rate_b,
            "average_steps": self.average_steps,
            "per_house_a": self.per_house_a,
            "per_house_b": self.per_house_b,
        }


def split_houses(houses: tuple[str, ...], swap: bool) -> tuple[set[str], set[str]]:
    """Give the first half of the seats to A and the rest to B (or vice versa)."""
    midpoint = len(houses) // 2
    first, second = set(houses[:midpoint]), set(houses[midpoint:])
    if swap:
        return second, first
    return first, second


@dataclass(frozen=True)
class MatchGame:
    """One game of a match: its global index, RNG seed and seat split."""

    index: int
    seed: int
    swap: bool
    #: The houses model A plays this game. Empty means the historical
    #: first-half/second-half split, kept for old plans and tests.
    seats_a: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchPlan:
    """The per-game schedule of a head-to-head match.

    ``MatchGame.index`` is the game's position in the *whole* match, not in
    the slice being played. A plan can therefore be split into chunks (one per
    gate shard) and the results merged without renumbering anything.
    """

    houses: tuple[str, ...]
    seed: int
    games: tuple[MatchGame, ...]

    def __len__(self) -> int:
        return len(self.games)

    def __iter__(self):
        return iter(self.games)

    def __getitem__(self, index):
        return self.games[index]

    @property
    def seeds(self) -> list[int]:
        """Per-game RNG seeds, in game order."""
        return [game.seed for game in self.games]

    @property
    def swaps(self) -> list[bool]:
        """Per-game seat split, in game order."""
        return [game.swap for game in self.games]

    def slice(self, start: int, stop: int) -> MatchPlan:
        """A contiguous sub-plan, keeping each game's global index."""
        return MatchPlan(houses=self.houses, seed=self.seed, games=self.games[start:stop])

    def chunks(self, size: int) -> list[MatchPlan]:
        """Split the plan into chunks of at most ``size`` games, from index 0."""
        size = max(1, int(size))
        return [
            self.slice(start, min(len(self), start + size))
            for start in range(0, len(self), size)
        ]

    def per_game_seeds(self) -> list[int]:
        return [game.seed for game in self.games]

    def seat_models(self) -> list[list[int]]:
        """Model index per house per game: 0 for A's seats, 1 for B's."""
        return [
            [
                0
                if house in (game.seats_a or split_houses(self.houses, game.swap)[0])
                else 1
                for house in self.houses
            ]
            for game in self.games
        ]


@dataclass(frozen=True)
class GameOutcome:
    """The result of one planned game, keyed by its global index."""

    index: int
    winner: str | None
    steps: int


def build_match_plan(houses: tuple[str, ...], games: int, seed: int) -> MatchPlan:
    """The deterministic schedule shared by a single call and chunked runs.

    Game ``i`` gets seed ``seed + i`` and A takes the first half of the seats
    when ``i`` is even, exactly as the historical loop did.
    """
    # The 3v3 split is drawn at random for every game: a fixed split would
    # confound model strength with house strength and would always pair the
    # same three houses against the other three. The draw is seeded, so a plan
    # is reproducible and a chunk of it is identical to the whole match.
    rng = random.Random(seed)
    half = len(houses) // 2
    schedule = []
    drawn: tuple[str, ...] = ()
    for index in range(games):
        if index % 2 == 0 or not drawn:
            seats = tuple(rng.sample(list(houses), half))
            drawn = seats
        else:
            # Complement of the previous game: every house plays for A exactly
            # half the time, while which houses are paired stays random.
            seats = tuple(house for house in houses if house not in drawn)
        schedule.append(
            MatchGame(
                index=index,
                seed=seed + index,
                swap=bool(seats[0] in houses[half:]),
                seats_a=seats,
            )
        )
    return MatchPlan(houses=tuple(houses), seed=seed, games=tuple(schedule))


def seat_policies(
    factory_a,
    factory_b,
    houses: tuple[str, ...],
    *,
    seed: int,
    swap: bool,
) -> dict[str, Policy]:
    """Instantiate policies for one game, splitting the seats between A and B."""
    seats_a, seats_b = split_houses(houses, swap)
    policies: dict[str, Policy] = {}
    for index, house in enumerate(houses):
        if house in seats_a:
            policies[house] = factory_a(house, seed + index)
        else:
            policies[house] = factory_b(house, seed + 1000 + index)
    return policies


def evaluate_match(
    factory_a,
    factory_b,
    *,
    games: int = 10,
    config: GameConfig | None = None,
    seed: int = 0,
    step_limit: int = DEFAULT_STEP_LIMIT,
) -> MatchResult:
    """Play ``games`` games between A and B, swapping seats each game.

    A "win" is credited to whichever agent controls the winning house. If the
    winning house was not controlled by A, the point goes to B.
    """
    base = config or GameConfig()
    houses = base.houses
    result = MatchResult()
    total_steps = 0
    for game_index in range(games):
        swap = bool(game_index % 2)
        game_config = GameConfig(
            houses=houses,
            max_rounds=base.max_rounds,
            victory_castles=base.victory_castles,
            seed=seed + game_index,
            use_westeros_decks=base.use_westeros_decks,
            deferred_battles=base.deferred_battles,
            sequential_westeros=base.sequential_westeros,
            doran_choice=base.doran_choice,
            wildling_tie_choice=base.wildling_tie_choice,
            bid_tie_choice=base.bid_tie_choice,
            choice_choices=base.choice_choices,
        )
        policies = seat_policies(
            factory_a, factory_b, houses, seed=seed + game_index * 7, swap=swap
        )
        record = play_game(
            policies, game_config, step_limit=step_limit, record=False
        )
        seats_a, _ = split_houses(houses, swap)
        result.games += 1
        total_steps += record.steps
        if record.winner in seats_a:
            result.wins_a += 1
        else:
            result.wins_b += 1
    result.win_rate_a = result.wins_a / max(1, result.games)
    result.win_rate_b = result.wins_b / max(1, result.games)
    result.average_steps = total_steps / max(1, result.games)
    result.seats_of_a = sorted(split_houses(houses, False)[0])
    return result


def wilson_interval(
    wins: int, games: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because a small gauntlet can sit
    near 0% or 100%, where the naive interval is badly wrong.
    """
    if games <= 0:
        return 0.0, 0.0
    phat = wins / games
    denominator = 1.0 + z * z / games
    centre = (phat + z * z / (2 * games)) / denominator
    half = (
        z
        * math.sqrt(phat * (1.0 - phat) / games + z * z / (4.0 * games * games))
        / denominator
    )
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class FieldResult:
    """Outcome of a candidate playing a whole field of opponents.

    In a six-player free-for-all only one house can win, so this measures the
    candidate's *single-seat win rate*: it holds one of the six seats, the
    field holds the other five, and it must beat all of them.
    """

    games: int = 0
    wins: int = 0
    draws: int = 0
    win_rate: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    average_steps: float = 0.0
    per_seat: dict[str, float] = field(default_factory=dict)
    opponents: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "win_rate": round(self.win_rate, 4),
            "ci95": [round(self.ci_low, 4), round(self.ci_high, 4)],
            "average_steps": round(self.average_steps, 1),
            "per_seat": {k: round(v, 4) for k, v in self.per_seat.items()},
            "opponents": self.opponents,
        }


def _game_config(
    base: GameConfig, *, seed: int
) -> GameConfig:
    return GameConfig(
        houses=base.houses,
        max_rounds=base.max_rounds,
        victory_castles=base.victory_castles,
        seed=seed,
        use_westeros_decks=base.use_westeros_decks,
        deferred_battles=base.deferred_battles,
        sequential_westeros=base.sequential_westeros,
        doran_choice=base.doran_choice,
        wildling_tie_choice=base.wildling_tie_choice,
        bid_tie_choice=base.bid_tie_choice,
        choice_choices=base.choice_choices,
    )


def _play_model_games(
    models: list,
    seat_models: list[list[int | None]],
    scripted: list[dict[str, Policy]],
    *,
    simulations: int,
    determinizations: int,
    games: int,
    config: GameConfig,
    seed: int,
    step_limit: int,
    batch_games: int,
    seeds: list[int] | None = None,
    leaf_backend: str = "torch",
) -> tuple[list[str | None], list[int]]:
    """Play ``games`` with one model per search seat, plus optional scripted seats.

    ``seat_models[game][house_index]`` selects the model for a search seat;
    ``None`` means the seat is scripted and lives in ``scripted[game]``. The
    driver groups each leaf batch by model, so two checkpoints can play each
    other without falling back to Python MCTS.

    ``seeds`` gives every game an explicit RNG seed. Chunked callers (the
    deferred gate) pass the seeds from the whole-match plan so splitting the
    games across processes does not replay neighbouring games.

    ``leaf_backend`` is the inference engine for every search seat (see
    :func:`agot.training.rust_engine.leaf_evaluator`); each model caches its
    graph beside the checkpoint it came from.
    """
    from agot.engine.rust import RustGame
    from agot.training import rust_engine

    extension = rust_engine.core()
    if extension is None:  # pragma: no cover - the extension is mandatory now
        raise RuntimeError("the Rust extension is not installed")

    if seeds is None:
        seeds = [seed + index for index in range(games)]
    elif len(seeds) != games:
        raise ValueError(f"expected {games} seeds, got {len(seeds)}")

    houses = config.houses
    evaluators = [
        rust_engine.leaf_evaluator(model, leaf_backend) for model in models
    ]

    def decide(game_index: int, house: str, game_view):
        policy = scripted[game_index][house]
        view = RustGame(game_view)
        action, _ = policy.decide(view)
        try:
            index = view.legal_actions().index(action)
        except ValueError as error:  # pragma: no cover - defensive
            raise RuntimeError(
                f"scripted policy {policy.name!r} chose an action the engine "
                "did not offer"
            ) from error
        return index, None

    spec = {
        "games": games,
        "simulations": simulations,
        "determinizations": max(1, determinizations),
        "batch_games": max(1, min(batch_games, games)),
        "seed": seed,
        "seeds": list(seeds),
        "temperature": 0.0,
        "temperature_rounds": 0,
        "final_temperature": 0.0,
        "max_rounds": config.max_rounds,
        "victory_castles": config.victory_castles,
        "use_westeros_decks": config.use_westeros_decks,
        "deferred_battles": config.deferred_battles,
        "sequential_westeros": config.sequential_westeros,
        "doran_choice": config.doran_choice,
        "wildling_tie_choice": config.wildling_tie_choice,
        "bid_tie_choice": config.bid_tie_choice,
        "choice_choices": config.choice_choices,
        "houses": list(houses),
        "step_limit": step_limit,
        "scripted": [list(policies) for policies in scripted],
        "seat_models": seat_models,
        "record_opponents": False,
    }
    result = extension.selfplay(
        spec,
        evaluators[0],
        decide if any(scripted) else None,
        evaluators[1:],
    )
    return list(result[6]), [int(step) for step in result[7]]


def evaluate_model_vs_factory(
    model,
    opponent_factory: PolicyFactory,
    *,
    simulations: int = 128,
    determinizations: int = 1,
    games: int = 10,
    config: GameConfig | None = None,
    seed: int = 0,
    step_limit: int = DEFAULT_STEP_LIMIT,
    batch_games: int = 4,
    leaf_backend: str = "torch",
) -> MatchResult:
    """One network (three seats) against a scripted factory, on the Rust driver."""
    base = config or GameConfig()
    houses = base.houses
    seat_models: list[list[int | None]] = []
    scripted: list[dict[str, Policy]] = []
    for game_index in range(games):
        seats_a, _ = split_houses(houses, bool(game_index % 2))
        row: list[int | None] = [None] * len(houses)
        policies: dict[str, Policy] = {}
        for index, house in enumerate(houses):
            if house in seats_a:
                row[index] = 0
            else:
                policies[house] = opponent_factory(
                    house, seed + game_index * 7 + 1000 + index
                )
        seat_models.append(row)
        scripted.append(policies)

    winners, steps = _play_model_games(
        [model],
        seat_models,
        scripted,
        simulations=simulations,
        determinizations=determinizations,
        games=games,
        config=base,
        seed=seed,
        step_limit=step_limit,
        batch_games=batch_games,
        leaf_backend=leaf_backend,
    )

    result = MatchResult()
    total_steps = 0
    for game_index, winner in enumerate(winners):
        seats_a, _ = split_houses(houses, bool(game_index % 2))
        result.games += 1
        total_steps += steps[game_index]
        if winner in seats_a:
            result.wins_a += 1
        else:
            result.wins_b += 1
    result.win_rate_a = result.wins_a / max(1, result.games)
    result.win_rate_b = result.wins_b / max(1, result.games)
    result.average_steps = total_steps / max(1, result.games)
    result.seats_of_a = sorted(split_houses(houses, False)[0])
    return result


def play_plan(
    models: list,
    plan: MatchPlan,
    *,
    simulations: int = 128,
    determinizations: int = 1,
    config: GameConfig | None = None,
    step_limit: int = DEFAULT_STEP_LIMIT,
    batch_games: int = 4,
    leaf_backend: str = "torch",
) -> list[GameOutcome]:
    """Play every game in ``plan`` with one model per search seat.

    The plan carries explicit per-game seeds, so playing a chunk of a match in
    one process produces the same games as playing the whole match at once.
    ``leaf_backend`` is the inference engine every search seat uses.
    """
    base = config or GameConfig()
    if tuple(plan.houses) != tuple(base.houses):
        raise ValueError("the match plan and the game config disagree on the houses")
    if not plan.games:
        return []
    seat_models = plan.seat_models()
    needed = max(max(row, default=0) for row in seat_models) + 1
    if len(models) < needed:
        raise ValueError(f"the plan needs {needed} models, got {len(models)}")
    winners, steps = _play_model_games(
        models,
        seat_models,
        [{} for _ in plan.games],
        simulations=simulations,
        determinizations=determinizations,
        games=len(plan),
        config=base,
        seed=plan.seed,
        seeds=plan.per_game_seeds(),
        step_limit=step_limit,
        batch_games=batch_games,
        leaf_backend=leaf_backend,
    )
    return [
        GameOutcome(index=game.index, winner=winners[position], steps=steps[position])
        for position, game in enumerate(plan.games)
    ]


def summarize_plan(
    results: list[GameOutcome],
    plan: MatchPlan,
    houses: tuple[str, ...],
) -> MatchResult:
    """Aggregate per-game outcomes into the historical :class:`MatchResult`.

    ``results`` may arrive in any order (the gate merges one file per chunk),
    but the plan decides the order and every planned game must be present.
    """
    houses = tuple(houses)
    if houses != plan.houses:
        raise ValueError("the match plan and the summary disagree on the houses")
    by_index = {outcome.index: outcome for outcome in results}
    if len(by_index) != len(results):
        raise ValueError("the match results contain a duplicate game index")
    expected = [game.index for game in plan.games]
    missing = [index for index in expected if index not in by_index]
    extra = sorted(set(by_index) - set(expected))
    if missing or extra:
        raise ValueError(
            f"the match results do not cover the plan (missing={missing}, extra={extra})"
        )

    result = MatchResult()
    total_steps = 0
    a_games = {house: 0 for house in houses}
    a_wins = {house: 0 for house in houses}
    b_games = {house: 0 for house in houses}
    b_wins = {house: 0 for house in houses}
    for game in plan.games:
        outcome = by_index[game.index]
        if game.seats_a:
            seats_a = set(game.seats_a)
            seats_b = set(houses) - seats_a
        else:
            seats_a, seats_b = split_houses(houses, game.swap)
        winner = outcome.winner
        result.games += 1
        total_steps += outcome.steps
        for house in seats_a:
            a_games[house] += 1
            if winner == house:
                a_wins[house] += 1
        for house in seats_b:
            b_games[house] += 1
            if winner == house:
                b_wins[house] += 1
        if winner in seats_a:
            result.wins_a += 1
        else:
            result.wins_b += 1
    result.win_rate_a = result.wins_a / max(1, result.games)
    result.win_rate_b = result.wins_b / max(1, result.games)
    result.average_steps = total_steps / max(1, result.games)
    result.seats_of_a = sorted(split_houses(houses, False)[0])
    result.per_house_a = {
        house: a_wins[house] / a_games[house]
        for house in houses
        if a_games[house]
    }
    result.per_house_b = {
        house: b_wins[house] / b_games[house]
        for house in houses
        if b_games[house]
    }
    return result


def evaluate_models_match(
    model_a,
    model_b,
    *,
    simulations: int = 128,
    determinizations: int = 1,
    games: int = 10,
    config: GameConfig | None = None,
    seed: int = 0,
    step_limit: int = DEFAULT_STEP_LIMIT,
    batch_games: int = 4,
    leaf_backend: str = "torch",
) -> MatchResult:
    """Two networks, three seats each, swapped each game, all on the Rust driver."""
    base = config or GameConfig()
    plan = build_match_plan(base.houses, games, seed)
    results = play_plan(
        [model_a, model_b],
        plan,
        simulations=simulations,
        determinizations=determinizations,
        config=base,
        step_limit=step_limit,
        batch_games=batch_games,
        leaf_backend=leaf_backend,
    )
    return summarize_plan(results, plan, base.houses)


def evaluate_models_vs_field(
    candidate_model,
    opponent_models: dict[str, object],
    *,
    anchors: dict[str, PolicyFactory] | None = None,
    simulations: int = 128,
    determinizations: int = 1,
    games: int = 60,
    config: GameConfig | None = None,
    seed: int = 0,
    step_limit: int = DEFAULT_STEP_LIMIT,
    batch_games: int = 4,
    leaf_backend: str = "torch",
) -> FieldResult:
    """Single-seat win rate against net checkpoints and scripted anchors."""
    base = config or GameConfig()
    houses = base.houses
    anchors = anchors or {}
    field_names = list(opponent_models) + list(anchors)
    if not field_names:
        raise ValueError("evaluate_models_vs_field needs at least one opponent")

    models = [candidate_model] + list(opponent_models.values())
    model_index = {
        name: index + 1 for index, name in enumerate(opponent_models)
    }

    seat_models: list[list[int | None]] = []
    scripted: list[dict[str, Policy]] = []
    for game_index in range(games):
        candidate_seat = houses[game_index % len(houses)]
        row: list[int | None] = [None] * len(houses)
        row[houses.index(candidate_seat)] = 0
        policies: dict[str, Policy] = {}
        cursor = game_index
        for seat, house in enumerate(houses):
            if house == candidate_seat:
                continue
            name = field_names[cursor % len(field_names)]
            cursor += 1
            if name in model_index:
                row[houses.index(house)] = model_index[name]
            else:
                policies[house] = anchors[name](
                    house, seed + game_index * 17 + 500 + seat
                )
        seat_models.append(row)
        scripted.append(policies)

    winners, steps = _play_model_games(
        models,
        seat_models,
        scripted,
        simulations=simulations,
        determinizations=determinizations,
        games=games,
        config=base,
        seed=seed,
        step_limit=step_limit,
        batch_games=batch_games,
        leaf_backend=leaf_backend,
    )

    field = FieldResult(opponents=field_names)
    seat_games = {house: 0 for house in houses}
    seat_wins = {house: 0 for house in houses}
    total_steps = 0
    for game_index, winner in enumerate(winners):
        candidate_seat = houses[game_index % len(houses)]
        field.games += 1
        total_steps += steps[game_index]
        seat_games[candidate_seat] += 1
        if winner == candidate_seat:
            field.wins += 1
            seat_wins[candidate_seat] += 1
        elif winner is None:
            field.draws += 1
    field.win_rate = field.wins / max(1, field.games)
    field.ci_low, field.ci_high = wilson_interval(field.wins, field.games)
    field.average_steps = total_steps / max(1, field.games)
    field.per_seat = {
        house: seat_wins[house] / seat_games[house]
        for house in houses
        if seat_games[house]
    }
    return field


def evaluate_net_vs_field(
    model,
    opponents: dict[str, PolicyFactory],
    *,
    simulations: int = 128,
    determinizations: int = 1,
    games: int = 60,
    config: GameConfig | None = None,
    seed: int = 0,
    step_limit: int = DEFAULT_STEP_LIMIT,
    batch_games: int = 4,
    leaf_backend: str = "torch",
) -> FieldResult:
    """Single-seat win rate of a network against scripted opponents, in Rust.

    The same protocol as :func:`evaluate_vs_field` (rotating candidate seat,
    round-robin opponents, per-seat win rates), but every game is played by the
    Rust driver: the candidate seat is searched and batched by the native
    driver, and the other five seats call the scripted policy once per decision.
    This turns a gauntlet from roughly two minutes per game of Python MCTS into
    the native self-play speed.
    """
    from agot.engine.rust import RustGame
    from agot.training import rust_engine

    extension = rust_engine.core()
    if extension is None:  # pragma: no cover - the extension is mandatory now
        raise RuntimeError("the Rust extension is not installed")

    base = config or GameConfig()
    houses = base.houses
    names = list(opponents)
    if not names:
        raise ValueError("evaluate_net_vs_field needs at least one opponent")

    scripted: list[list[str]] = []
    policies_by_game: list[dict[str, Policy]] = []
    for game_index in range(games):
        candidate_seat = houses[game_index % len(houses)]
        policies: dict[str, Policy] = {}
        opponent_cursor = game_index
        for seat, house in enumerate(houses):
            if house == candidate_seat:
                continue
            name = names[opponent_cursor % len(names)]
            opponent_cursor += 1
            policies[house] = opponents[name](
                house, seed + game_index * 17 + 500 + seat
            )
        policies_by_game.append(policies)
        scripted.append(list(policies))

    def decide(game_index: int, house: str, game_view):
        policy = policies_by_game[game_index][house]
        view = RustGame(game_view)
        action, _ = policy.decide(view)
        try:
            index = view.legal_actions().index(action)
        except ValueError as error:  # pragma: no cover - defensive
            raise RuntimeError(
                f"scripted policy {policy.name!r} chose an action the engine "
                "did not offer"
            ) from error
        return index, None

    spec = {
        "games": games,
        "simulations": simulations,
        "determinizations": max(1, determinizations),
        "batch_games": max(1, min(batch_games, games)),
        "seed": seed,
        "seeds": [seed + index for index in range(games)],
        "temperature": 0.0,
        "temperature_rounds": 0,
        "final_temperature": 0.0,
        "max_rounds": base.max_rounds,
        "victory_castles": base.victory_castles,
        "use_westeros_decks": base.use_westeros_decks,
        "deferred_battles": base.deferred_battles,
        "sequential_westeros": base.sequential_westeros,
        "doran_choice": base.doran_choice,
        "wildling_tie_choice": base.wildling_tie_choice,
        "bid_tie_choice": base.bid_tie_choice,
        "choice_choices": base.choice_choices,
        "houses": list(houses),
        "step_limit": step_limit,
        "scripted": scripted,
        "record_opponents": False,
    }
    result = extension.selfplay(
        spec, rust_engine.leaf_evaluator(model, leaf_backend), decide
    )
    winners = list(result[6])
    steps = list(result[7])

    field = FieldResult(opponents=names)
    seat_games: dict[str, int] = {house: 0 for house in houses}
    seat_wins: dict[str, int] = {house: 0 for house in houses}
    total_steps = 0
    for game_index, winner in enumerate(winners):
        candidate_seat = houses[game_index % len(houses)]
        field.games += 1
        total_steps += int(steps[game_index])
        seat_games[candidate_seat] += 1
        if winner == candidate_seat:
            field.wins += 1
            seat_wins[candidate_seat] += 1
        elif winner is None:
            field.draws += 1

    field.win_rate = field.wins / max(1, field.games)
    field.ci_low, field.ci_high = wilson_interval(field.wins, field.games)
    field.average_steps = total_steps / max(1, field.games)
    field.per_seat = {
        house: seat_wins[house] / seat_games[house]
        for house in houses
        if seat_games[house]
    }
    return field


def evaluate_vs_field(
    candidate_factory: PolicyFactory,
    opponents: dict[str, PolicyFactory],
    *,
    games: int = 60,
    config: GameConfig | None = None,
    seed: int = 0,
    step_limit: int = DEFAULT_STEP_LIMIT,
) -> FieldResult:
    """Play ``games`` free-for-alls with the candidate in a single seat.

    The candidate's seat rotates through the houses from game to game, and the
    five opponent seats are filled round-robin from ``opponents``. A game that
    runs past ``step_limit`` without a winner is a draw, not a loss.
    """
    base = config or GameConfig()
    houses = base.houses
    names = list(opponents)
    if not names:
        raise ValueError("evaluate_vs_field needs at least one opponent")

    result = FieldResult(opponents=names)
    seat_games: dict[str, int] = {house: 0 for house in houses}
    seat_wins: dict[str, int] = {house: 0 for house in houses}
    total_steps = 0

    for game_index in range(games):
        candidate_seat = houses[game_index % len(houses)]
        game_config = _game_config(base, seed=seed + game_index)
        policies: dict[str, Policy] = {}
        opponent_cursor = game_index
        for seat, house in enumerate(houses):
            if house == candidate_seat:
                policies[house] = candidate_factory(
                    house, seed + game_index * 17 + seat
                )
            else:
                name = names[opponent_cursor % len(names)]
                opponent_cursor += 1
                policies[house] = opponents[name](
                    house, seed + game_index * 17 + 500 + seat
                )

        record = play_game(
            policies, game_config, step_limit=step_limit, record=False
        )
        result.games += 1
        total_steps += record.steps
        seat_games[candidate_seat] += 1
        if record.winner == candidate_seat:
            result.wins += 1
            seat_wins[candidate_seat] += 1
        elif record.winner is None:
            result.draws += 1

    result.win_rate = result.wins / max(1, result.games)
    result.ci_low, result.ci_high = wilson_interval(result.wins, result.games)
    result.average_steps = total_steps / max(1, result.games)
    result.per_seat = {
        house: seat_wins[house] / seat_games[house]
        for house in houses
        if seat_games[house]
    }
    return result
