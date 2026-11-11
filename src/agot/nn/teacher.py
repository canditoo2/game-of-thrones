"""Planning teacher: greedy scores as a soft target for order placement.

The 97-dim state vector does not encode the board, so every child of a planning
node looks identical to a state-only value head; the recorded visit target is
therefore near-uniform (about 2.6 nats, 43% of decisions). The *action* encoder
does see the board, so the scorer can rank the candidate orders -- it only needs
a ranking to imitate. The hand-written greedy heuristic already produces one,
and a network trained on 4k such planning decisions reached 40% held-out top-1
against 10% for the self-play policy.

This module turns the greedy score vector into a distribution and builds a small,
separate dataset of planning examples for the trainer's auxiliary loss. The
defaults match the experiment: softmax at temperature 1 (the scores are additive
and roughly one unit apart) and label smoothing 0.05, so no candidate is ever
exactly zero and the target stays a proper distribution.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from agot.agents.base import outcome_vector
from agot.agents.greedy_agent import GreedyPolicy
from agot.engine.enums import Phase
from agot.engine.game import Game, GameConfig
from agot.engine.observation import HOUSE_ORDER, observe
from agot.training.replay import Example

#: Uniform mass mixed into every teacher target. 0.05 keeps the target honest
#: about the heuristic's uncertainty without washing out the argmax.
PLANNING_SMOOTHING = 0.05
#: Softmax temperature over the greedy scores.
PLANNING_TEMPERATURE = 1.0
#: Safety cap for teacher games, like the self-play drivers.
DEFAULT_STEP_LIMIT = 20_000


def soft_teacher_target(
    scores,
    *,
    smoothing: float = PLANNING_SMOOTHING,
    temperature: float = PLANNING_TEMPERATURE,
) -> np.ndarray:
    """Normalise greedy scores into a smoothed distribution.

    ``softmax(scores / temperature)`` is the soft one-hot; label smoothing then
    mixes in ``smoothing`` uniform mass, so the result sums to one and every
    candidate keeps at least ``smoothing / N``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("teacher scores must be a non-empty 1-D vector")
    if not np.isfinite(scores).all():
        raise ValueError("teacher scores must be finite")
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    shifted = (scores - scores.max()) / temperature
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:  # pragma: no cover - defensive
        weights = np.full(scores.size, 1.0 / scores.size)
    else:
        weights /= total
    target = (1.0 - smoothing) * weights + smoothing / scores.size
    return target.astype(np.float32)


def planning_target(
    game: Game,
    actions=None,
    *,
    policy: GreedyPolicy | None = None,
    smoothing: float = PLANNING_SMOOTHING,
    temperature: float = PLANNING_TEMPERATURE,
) -> np.ndarray | None:
    """Teacher target for a planning decision, or ``None`` outside planning.

    The caller may pass the action list it already holds; otherwise the game's
    legal actions are used. Returned in the same order as those actions.
    """
    if game.state.phase is not Phase.PLANNING:
        return None
    candidates = list(game.legal_actions() if actions is None else actions)
    if not candidates:
        return None
    policy = policy or GreedyPolicy(noise=0.0)
    return soft_teacher_target(
        policy.scores(game, candidates),
        smoothing=smoothing,
        temperature=temperature,
    )


def planning_teacher_examples(
    *,
    games: int = 32,
    seed: int = 0,
    config: GameConfig | None = None,
    smoothing: float = PLANNING_SMOOTHING,
    temperature: float = PLANNING_TEMPERATURE,
    step_limit: int = DEFAULT_STEP_LIMIT,
) -> list[Example]:
    """Play greedy games and keep every planning decision as a teacher example.

    Only planning decisions are recorded: that is the phase whose children a
    state-only value head cannot tell apart. The final outcome is filled in so
    the examples are valid :class:`Example` objects, although the auxiliary
    policy loss does not read it.
    """
    base = config or GameConfig(seed=seed, sequential_westeros=True, deferred_battles=True)
    examples: list[Example] = []
    for game_index in range(max(0, games)):
        game_seed = (base.seed if base.seed is not None else seed) + game_index
        game = Game.new(replace(base, seed=game_seed))
        policies = {
            house: GreedyPolicy(seed=game_seed * 31 + seat)
            for seat, house in enumerate(base.houses)
        }
        local: list[Example] = []
        steps = 0
        while not game.is_terminal() and steps < step_limit:
            if game.state.phase is Phase.WESTEROS and game.state.westeros_waiting:
                game.continue_westeros()
                continue
            house = game.current_player()
            if house is None:  # pragma: no cover - the engine advances to someone
                break
            actions = game.legal_actions()
            if not actions:  # pragma: no cover - terminal handled above
                break
            action, _ = policies[house].decide(game)
            if game.state.phase is Phase.PLANNING:
                target = planning_target(
                    game,
                    actions,
                    policy=policies[house],
                    smoothing=smoothing,
                    temperature=temperature,
                )
                if target is not None:
                    observation = observe(game, actions)
                    local.append(
                        Example(
                            state=observation.state_features,
                            actions=observation.action_features,
                            policy=target,
                            player=HOUSE_ORDER.index(house),
                            outcome=None,  # type: ignore[arg-type]  # filled below
                        )
                    )
            game.step(action)
            steps += 1
        outcome = outcome_vector(game)
        for example in local:
            example.outcome = outcome
        examples.extend(local)
    return examples
