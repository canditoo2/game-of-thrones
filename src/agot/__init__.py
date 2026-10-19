"""AlphaZero-style agent for A Game of Thrones: The Board Game (2nd Edition).

The package is split into three layers:

``agot.engine``
    A thin Python facade over the Rust engine: the ``Game`` session view,
    actions, static map data and the encoder entry points.
``agot.agents``
    Policies that choose actions, including a random baseline and a
    neural-network-guided Monte Carlo tree search.
``agot.nn`` / ``agot.training``
    The policy/value network and the self-play training loop.

The engine is a Rust extension (`agot._core`); the Python layer depends only on
numpy, and PyTorch remains optional for the network and training loop.
"""

from agot.engine.game import Game, GameConfig
from agot.engine.map import GameMap, load_map

__all__ = ["Game", "GameConfig", "GameMap", "load_map"]
__version__ = "0.1.0"
