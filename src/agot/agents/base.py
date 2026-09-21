"""The policy interface shared by every agent.

A policy answers one question: *at this decision point, which action do I take,
and how much do I favour each candidate?* Returning the distribution alongside
the action lets the self-play loop train the network by imitation of the
search, which is the heart of the AlphaZero recipe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from agot.engine.actions import Action
from agot.engine.game import Game
from agot.engine.observation import HOUSE_ORDER, N_HOUSES


class Policy(ABC):
    """Base class for anything that can act in a :class:`~agot.engine.game.Game`."""

    #: Human-readable identifier, used in logs and evaluation tables.
    name: str = "policy"

    @abstractmethod
    def decide(self, game: Game) -> tuple[Action, np.ndarray]:
        """Return ``(action, target)``.

        ``target`` is a probability vector aligned with ``game.legal_actions()``.
        """
        raise NotImplementedError

    def act(self, game: Game) -> Action:
        """Convenience wrapper that discards the target distribution."""
        return self.decide(game)[0]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"


def uniform_target(size: int) -> np.ndarray:
    return np.full(size, 1.0 / size, dtype=np.float32)


def one_hot_target(index: int, size: int) -> np.ndarray:
    target = np.zeros(size, dtype=np.float32)
    target[index] = 1.0
    return target


def outcome_vector(game: Game) -> np.ndarray:
    """Per-house terminal reward: ``+1`` for the winner, ``-1`` for the rest.

    Houses that are not part of this game (when playing with fewer than six)
    keep a zero in their slot so the vector always has :data:`N_HOUSES` entries.
    """
    result = np.zeros(N_HOUSES, dtype=np.float32)
    winner = game.winner()
    present = set(game.config.houses)
    for index, house in enumerate(HOUSE_ORDER):
        if house not in present:
            continue
        result[index] = 1.0 if house == winner else -1.0
    return result
