"""Monte Carlo tree search with a pluggable evaluator (PUCT).

The engine supports any number of players, so a two-player negamax-style value
(the usual AlphaZero sign flip) is not enough. Instead each node stores a
*value vector* with one entry per canonical house::

    value_sums[house]  ->  summed reward for ``house`` over all visits

When a node chooses among its children it looks up the entry for *the house that
is choosing* (``node.player``). That keeps the search correct for 2-6 players
without any sign bookkeeping.

An :class:`Evaluator` supplies a prior distribution over the legal actions and a
predicted value vector for a position. The default
:class:`UniformEvaluator` needs no network at all, which is what lets the engine
and MCTS be exercised without PyTorch installed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from agot.agents.base import Policy, outcome_vector
from agot.engine.actions import Action
from agot.engine.game import Game
from agot.engine.observation import N_HOUSES


class Evaluator(Protocol):
    """Supplies a policy prior and a value vector for a position."""

    def __call__(
        self, game: Game, actions: list[Action]
    ) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - protocol
        ...


class UniformEvaluator:
    """A network-free evaluator: uniform priors, neutral values."""

    def __call__(
        self, game: Game, actions: list[Action]
    ) -> tuple[np.ndarray, np.ndarray]:
        priors = np.full(len(actions), 1.0 / len(actions), dtype=np.float64)
        return priors, np.zeros(N_HOUSES, dtype=np.float64)


class _Node:
    """One position in the search tree."""

    __slots__ = (
        "_game",
        "_pending",
        "player",
        "actions",
        "priors",
        "children",
        "visits",
        "value_sums",
    )

    def __init__(
        self,
        game: Game | None = None,
        pending: tuple[Game, Action] | None = None,
    ) -> None:
        self._game = game
        self._pending = pending
        self.player: str | None = None
        self.actions: list[Action] | None = None
        self.priors: np.ndarray | None = None
        self.children: list[_Node] | None = None
        self.visits = 0
        self.value_sums = np.zeros(N_HOUSES, dtype=np.float64)

    @property
    def game(self) -> Game:
        if self._game is None:
            assert self._pending is not None
            parent_game, action = self._pending
            clone = parent_game.clone()
            clone.step(action)
            self._game = clone
            self._pending = None
        return self._game

    @property
    def expanded(self) -> bool:
        return self.children is not None


@dataclass
class SearchResult:
    """What a completed search exposes to its caller."""

    actions: list[Action]
    visits: np.ndarray
    q_values: np.ndarray

    def probabilities(self, temperature: float = 1.0) -> np.ndarray:
        visits = self.visits.astype(np.float64)
        if visits.sum() <= 0:
            return np.full(len(self.actions), 1.0 / len(self.actions))
        if temperature <= 1e-3:
            probs = np.zeros_like(visits)
            probs[int(np.argmax(visits))] = 1.0
            return probs
        scaled = np.power(visits, 1.0 / temperature)
        return scaled / scaled.sum()


class MCTS(Policy):
    """PUCT search that turns an evaluator into a policy."""

    name = "mcts"

    def __init__(
        self,
        evaluator: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
        *,
        simulations: int = 32,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        seed: int | None = None,
    ) -> None:
        self.evaluator: Evaluator = evaluator or UniformEvaluator()
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.rng = np.random.default_rng(seed)

    # -- public API ------------------------------------------------------

    def search(self, game: Game) -> SearchResult:
        root = _Node(game=game.clone())
        if root.game.is_terminal():
            return SearchResult([], np.array([]), np.array([]))

        for _ in range(max(1, self.simulations)):
            self._simulate(root)

        assert root.actions is not None and root.children is not None
        visits = np.array([child.visits for child in root.children], dtype=np.float64)
        return SearchResult(root.actions, visits, self._root_q(root))

    def decide(self, game: Game) -> tuple[Action, np.ndarray]:
        result = self.search(game)
        if not result.actions:  # pragma: no cover - defensive
            raise RuntimeError("MCTS was asked to move in a terminal position")
        probs = result.probabilities(self.temperature)
        if self.temperature <= 1e-3:
            index = int(np.argmax(result.visits))
        else:
            index = int(self.rng.choice(len(result.actions), p=probs))
        return result.actions[index], probs.astype(np.float32)

    def act(self, game: Game) -> Action:  # deterministic greedy play
        result = self.search(game)
        index = int(np.argmax(result.visits))
        return result.actions[index]

    # -- simulation ------------------------------------------------------

    def _simulate(self, root: _Node) -> None:
        node = root
        path = [root]

        while True:
            if node.game.is_terminal():
                value = outcome_vector(node.game)
                break
            if not node.expanded:
                value = self._expand(node, apply_noise=node is root)
                break
            node = self._select_child(node)
            path.append(node)

        for visited in path:
            visited.visits += 1
            visited.value_sums += value

    def _expand(self, node: _Node, *, apply_noise: bool) -> np.ndarray:
        actions = node.game.legal_actions()
        if not actions:
            return outcome_vector(node.game)
        priors, value = self.evaluator(node.game, actions)
        priors = np.asarray(priors, dtype=np.float64)
        priors = np.clip(priors, 1e-8, None)
        priors /= priors.sum()
        if apply_noise and len(actions) > 1:
            noise = self.rng.dirichlet(
                np.full(len(actions), self.dirichlet_alpha)
            )
            priors = (1 - self.dirichlet_weight) * priors + self.dirichlet_weight * noise
        node.player = node.game.current_player()
        node.actions = actions
        node.priors = priors
        node.children = [_Node(pending=(node.game, action)) for action in actions]
        return np.asarray(value, dtype=np.float64)

    def _select_child(self, node: _Node) -> _Node:
        assert node.children is not None and node.priors is not None
        assert node.player is not None
        sqrt_total = math.sqrt(max(1, node.visits))
        best_index = 0
        best_score = -math.inf
        for index, child in enumerate(node.children):
            if child.visits:
                q = child.value_sums[node_value_index(node.player)] / child.visits
            else:
                q = 0.0
            u = (
                self.c_puct
                * node.priors[index]
                * sqrt_total
                / (1 + child.visits)
            )
            score = q + u
            if score > best_score:
                best_score = score
                best_index = index
        return node.children[best_index]

    def _root_q(self, root: _Node) -> np.ndarray:
        assert root.children is not None
        assert root.player is not None
        index = node_value_index(root.player)
        q = np.zeros(len(root.children), dtype=np.float64)
        for i, child in enumerate(root.children):
            if child.visits:
                q[i] = child.value_sums[index] / child.visits
        return q


def node_value_index(house: str | None) -> int:
    """Canonical vector index for a house id."""
    from agot.engine.observation import HOUSE_ORDER

    if house in HOUSE_ORDER:
        return HOUSE_ORDER.index(house)
    return 0
