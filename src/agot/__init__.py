"""AlphaZero-style agent for A Game of Thrones: The Board Game (2nd Edition).

The package is split into three layers:

``agot.engine``
    A pure-Python (numpy-only) rules engine: map data, game state, legal
    actions, combat resolution and the round loop.
``agot.agents``
    Policies that choose actions, including a random baseline and a
    neural-network-guided Monte Carlo tree search.
``agot.nn`` / ``agot.training``
    The policy/value network and the self-play training loop.

The rules engine is deliberately free of any PyTorch dependency so it can be
tested and simulated on machines without a GPU or without torch installed.
"""

from agot.engine.game import Game, GameConfig
from agot.engine.map import GameMap, load_map

__all__ = ["Game", "GameConfig", "GameMap", "load_map"]
__version__ = "0.1.0"
