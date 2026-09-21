"""A uniform random baseline.

It is deliberately weak: its only jobs are to prove the engine can run a full
game and to give the learning agents an opponent that cannot be gamed.
"""

from __future__ import annotations

import random

import numpy as np

from agot.agents.base import Policy, uniform_target
from agot.engine.actions import Action
from agot.engine.game import Game


class RandomPolicy(Policy):
    """Chooses uniformly among legal actions."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def decide(self, game: Game) -> tuple[Action, np.ndarray]:
        actions = game.legal_actions()
        if not actions:  # pragma: no cover - defensive
            raise RuntimeError("no legal actions to choose from")
        index = self.rng.randrange(len(actions))
        return actions[index], uniform_target(len(actions))
