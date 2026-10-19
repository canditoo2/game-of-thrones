"""The rules engine: board data, the Rust game view and the action space."""

from agot.engine.actions import Action, ActionKind
from agot.engine.game import Game, GameConfig
from agot.engine.map import GameMap, load_map
from agot.engine.rust import HouseView, RegionView, StateView

#: The Python engine's names, kept as aliases for callers that imported them.
GameState = StateView
HouseState = HouseView
RegionState = RegionView

__all__ = [
    "Action",
    "ActionKind",
    "Game",
    "GameConfig",
    "GameMap",
    "GameState",
    "HouseState",
    "HouseView",
    "RegionState",
    "RegionView",
    "StateView",
    "load_map",
]
