"""Generate training data by playing games against yourself.

The loop is the classic AlphaZero one: for every decision, ask each house's
policy for an action *and* the distribution it favours, play the game to the
end, then label every recorded position with the final outcome. The network is
later trained to reproduce the distributions and predict the outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

from agot.agents.base import Policy, outcome_vector
from agot.engine.game import Game, GameConfig
from agot.engine.observation import HOUSE_ORDER, observe
from agot.training.replay import Example

#: Hard cap so a bug can never hang a training run forever.
DEFAULT_STEP_LIMIT = 20_000


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
) -> GameRecord:
    """Play one game with ``policies`` (one per house) and record every decision."""
    game = Game.new(config or GameConfig())
    records: list[tuple[Example, ...]] = []
    steps = 0

    while not game.is_terminal() and steps < step_limit:
        house = game.current_player()
        if house is None:  # pragma: no cover - engine always advances to someone
            break
        policy = policies[house]
        action, target = policy.decide(game)
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
