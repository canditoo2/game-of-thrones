"""Head-to-head evaluation between policies.

Both agents play the *same* seats across a match, swapping sides every game, so
that the first-player advantage (and the asymmetric starting positions) cancel
out in expectation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agot.agents.base import Policy
from agot.engine.game import GameConfig
from agot.training.selfplay import DEFAULT_STEP_LIMIT, play_game


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

    def as_dict(self) -> dict:
        return {
            "games": self.games,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "win_rate_a": self.win_rate_a,
            "win_rate_b": self.win_rate_b,
            "average_steps": self.average_steps,
        }


def split_houses(houses: tuple[str, ...], swap: bool) -> tuple[set[str], set[str]]:
    """Give the first half of the seats to A and the rest to B (or vice versa)."""
    midpoint = len(houses) // 2
    first, second = set(houses[:midpoint]), set(houses[midpoint:])
    if swap:
        return second, first
    return first, second


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
        )
        policies = seat_policies(
            factory_a, factory_b, houses, seed=seed + game_index * 7, swap=swap
        )
        record = play_game(policies, game_config, step_limit=step_limit)
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
