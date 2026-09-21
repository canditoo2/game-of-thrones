"""Policies that choose actions.

Every policy implements :class:`~agot.agents.base.Policy`, which returns both a
chosen action and a *target distribution* over the legal actions. The target is
what the self-play loop distils into the neural policy: a random agent reports
the uniform distribution, a greedy agent a one-hot peak, and MCTS its visit
counts.
"""

from agot.agents.base import Policy, outcome_vector
from agot.agents.greedy_agent import GreedyPolicy
from agot.agents.mcts import MCTS, UniformEvaluator
from agot.agents.random_agent import RandomPolicy

__all__ = [
    "MCTS",
    "GreedyPolicy",
    "Policy",
    "RandomPolicy",
    "UniformEvaluator",
    "outcome_vector",
]
