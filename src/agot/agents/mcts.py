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

Hidden information
------------------

The engine stores the true state, so a naive search would read face-down Order
tokens and secret bids. With ``determinizations > 1`` the search runs PIMC
instead: it copies the position several times through
:func:`agot.engine.hidden.determinize`, searching each resampled world and
aggregating the root visit counts. The result reflects only what the acting
house can actually see.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from agot.agents.base import Policy, outcome_vector
from agot.engine.actions import Action
from agot.engine.game import Game
from agot.engine.hidden import determinize
from agot.engine.observation import HOUSE_ORDER, N_HOUSES

#: House id -> canonical value-vector index, resolved once.
_HOUSE_INDEX: dict[str, int] = {house: index for index, house in enumerate(HOUSE_ORDER)}


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
        #: Summed per-house reward, allocated on the first visit. A fresh
        #: expansion creates one node per candidate action, and most of those
        #: are never visited, so the array is created lazily.
        self.value_sums: np.ndarray | None = None

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


@dataclass
class Simulation:
    """One MCTS simulation paused between descent and leaf evaluation.

    ``value`` is set when the descent stopped at a terminal (or actionless)
    position. Otherwise ``leaf`` is the node that still needs a network score and
    ``actions`` are its legal moves; a batching driver collects these across
    games, scores them in one forward pass, then calls
    :meth:`MCTS.finish_simulation`.
    """

    path: list[_Node]
    leaf: _Node | None
    actions: list[Action] | None
    value: np.ndarray | None
    apply_noise: bool


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
        determinizations: int = 1,
        seed: int | None = None,
    ) -> None:
        self.evaluator: Evaluator = evaluator or UniformEvaluator()
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        #: Number of resampled hidden states searched per move. ``1`` means plain
        #: perfect-information search; higher values enable PIMC.
        self.determinizations = max(1, determinizations)
        self.rng = np.random.default_rng(seed)
        self._det_rng = random.Random(seed)

    # -- public API ------------------------------------------------------

    def search(self, game: Game) -> SearchResult:
        observer = game.current_player()
        if self.determinizations > 1 and observer is not None:
            return self._search_determinized(game, observer)

        root = _Node(game=game.clone())
        if root.game.is_terminal():
            return SearchResult([], np.array([]), np.array([]))
        self._run(root, self.simulations)
        return self.result_from_root(root)

    def open_search(self, game: Game) -> _Node | None:
        """Create (but do not run) a search root, for stepwise drivers."""
        root = _Node(game=game.clone())
        if root.game.is_terminal():
            return None
        return root

    def open_search_worlds(self, game: Game) -> list[_Node]:
        """Roots for one stepwise decision: one per determinized world.

        With ``determinizations == 1`` this is a single perfect-information root.
        Otherwise each root is a resampled world the acting house cannot
        distinguish; a stepwise driver runs simulations on all of them and
        aggregates with :meth:`aggregate_roots`.
        """
        observer = game.current_player()
        if self.determinizations <= 1 or observer is None:
            root = self.open_search(game)
            return [] if root is None else [root]
        roots: list[_Node] = []
        for _ in range(self.determinizations):
            world = determinize(game, observer, self._det_rng)
            root = _Node(game=world)
            if not root.game.is_terminal():
                roots.append(root)
        return roots

    def result_from_root(self, root: _Node) -> SearchResult:
        """Package a finished root into a :class:`SearchResult`."""
        assert root.actions is not None and root.children is not None
        visits = np.array([child.visits for child in root.children], dtype=np.float64)
        return SearchResult(root.actions, visits, self._root_q(root))

    def aggregate_roots(
        self, roots: list[_Node], observer: str | None = None
    ) -> SearchResult:
        """Merge finished PIMC roots into one visit distribution.

        The action list is the same in every world (determinization never changes
        the acting house's own legal moves), so visits simply add up.
        """
        if not roots:
            return SearchResult([], np.array([]), np.array([]))
        if len(roots) == 1:
            return self.result_from_root(roots[0])
        assert roots[0].actions is not None
        actions = roots[0].actions
        index_of = {action: index for index, action in enumerate(actions)}
        visits = np.zeros(len(actions), dtype=np.float64)
        value_sums = np.zeros((len(actions), N_HOUSES), dtype=np.float64)
        for root in roots:
            assert root.actions is not None and root.children is not None
            for child_index, action in enumerate(root.actions):
                position = index_of.get(action)
                if position is None:  # pragma: no cover - action sets must agree
                    continue
                child = root.children[child_index]
                if not child.visits:
                    continue
                visits[position] += child.visits
                assert child.value_sums is not None
                value_sums[position] += child.value_sums
        value_index = node_value_index(observer or roots[0].player)
        q = np.zeros(len(actions), dtype=np.float64)
        for index in range(len(actions)):
            if visits[index]:
                q[index] = value_sums[index][value_index] / visits[index]
        return SearchResult(actions, visits, q)

    def _run(self, root: _Node, simulations: int) -> None:
        for _ in range(max(1, simulations)):
            self._simulate(root)

    def _search_determinized(self, game: Game, observer: str) -> SearchResult:
        """PIMC: average the root statistics over resampled hidden states."""
        roots = self.open_search_worlds(game)
        if not roots:
            return SearchResult([], np.array([]), np.array([]))
        per_world = max(1, self.simulations // self.determinizations)
        for root in roots:
            self._run(root, per_world)
        return self.aggregate_roots(roots, observer)

    def decide(self, game: Game) -> tuple[Action, np.ndarray]:
        return self.pick(self.search(game))

    def pick(
        self, result: SearchResult, temperature: float | None = None
    ) -> tuple[Action, np.ndarray]:
        """Sample an action from a finished search (shared by every caller).

        ``temperature`` overrides this policy's own value for the one call, which
        is how a training driver applies a per-round schedule without rebuilding
        the policy. The returned target is the raw visit distribution, so the
        sampling temperature never sharpens the training label.
        """
        if not result.actions:  # pragma: no cover - defensive
            raise RuntimeError("MCTS was asked to move in a terminal position")
        temperature = self.temperature if temperature is None else temperature
        probs = result.probabilities(temperature)
        if temperature <= 1e-3:
            index = int(np.argmax(result.visits))
        else:
            index = int(self.rng.choice(len(result.actions), p=probs))
        # The target is always the visit distribution; sharpening it with 1/T
        # would teach the network certainty the search does not have.
        target = result.probabilities(1.0).astype(np.float32)
        return result.actions[index], target

    def act(self, game: Game) -> Action:  # deterministic greedy play
        result = self.search(game)
        index = int(np.argmax(result.visits))
        return result.actions[index]

    # -- simulation ------------------------------------------------------
    #
    # A simulation splits into ``prepare_simulation`` / ``finish_simulation`` so
    # a driver can pause many games at their leaves, score every leaf in one
    # batched network call, then resume. ``_simulate`` is just the two halves
    # run back to back with the evaluator in the middle.

    def prepare_simulation(self, root: _Node) -> Simulation:
        """Descend to this simulation's leaf without evaluating it yet."""
        node = root
        path = [root]
        while True:
            if node.game.is_terminal():
                return Simulation(path, None, None, outcome_vector(node.game), False)
            if not node.expanded:
                actions = node.game.legal_actions()
                if not actions:
                    return Simulation(
                        path, None, None, outcome_vector(node.game), False
                    )
                return Simulation(path, node, actions, None, node is root)
            node = self._select_child(node)
            path.append(node)

    def finish_simulation(
        self,
        simulation: Simulation,
        priors: np.ndarray | None = None,
        value: np.ndarray | None = None,
    ) -> None:
        """Expand the leaf (if any) and back up the value along the path."""
        if simulation.leaf is not None:
            if priors is None or value is None:
                raise ValueError("an unexpanded leaf needs priors and a value")
            assert simulation.actions is not None
            value = self._expand_with(
                simulation.leaf,
                simulation.actions,
                priors,
                value,
                apply_noise=simulation.apply_noise,
            )
        elif value is None:
            value = simulation.value
        assert value is not None
        for visited in simulation.path:
            visited.visits += 1
            if visited.value_sums is None:
                visited.value_sums = value.copy()
            else:
                visited.value_sums += value

    def _simulate(self, root: _Node) -> None:
        simulation = self.prepare_simulation(root)
        if simulation.leaf is None:
            self.finish_simulation(simulation)
        else:
            assert simulation.actions is not None
            priors, value = self.evaluator(simulation.leaf.game, simulation.actions)
            self.finish_simulation(simulation, priors, value)

    def _expand(self, node: _Node, *, apply_noise: bool) -> np.ndarray:
        actions = node.game.legal_actions()
        if not actions:
            return outcome_vector(node.game)
        priors, value = self.evaluator(node.game, actions)
        return self._expand_with(node, actions, priors, value, apply_noise=apply_noise)

    def _expand_with(
        self,
        node: _Node,
        actions: list[Action],
        priors: np.ndarray,
        value: np.ndarray,
        *,
        apply_noise: bool,
    ) -> np.ndarray:
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
        value_index = node_value_index(node.player)
        # Values here are almost always negative, so a fixed zero first-play
        # value would make every unvisited child look like a win. Use the mean
        # of the children already seen instead.
        seen = [
            child.value_sums[value_index] / child.visits
            for child in node.children
            if child.visits
        ]
        first_play = sum(seen) / len(seen) if seen else 0.0
        best_index = 0
        best_score = -math.inf
        for index, child in enumerate(node.children):
            if child.visits:
                q = child.value_sums[value_index] / child.visits
            else:
                q = first_play
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
    return _HOUSE_INDEX.get(house or "", 0)
