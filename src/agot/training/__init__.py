"""Self-play data generation, replay storage and evaluation.

The neural trainer is imported only if PyTorch is available, so the data and
evaluation tooling works on a torch-free machine.
"""

from agot.training.evaluate import MatchResult, evaluate_match, seat_policies
from agot.training.replay import Example, ReplayBuffer
from agot.training.selfplay import GameRecord, play_game, self_play

__all__ = [
    "Example",
    "GameRecord",
    "MatchResult",
    "ReplayBuffer",
    "evaluate_match",
    "play_game",
    "seat_policies",
    "self_play",
]

try:  # pragma: no cover - depends on the optional torch extra
    from agot.training.trainer import TrainConfig, Trainer, TrainStats

    __all__ += ["TrainConfig", "TrainStats", "Trainer"]
except ImportError:  # torch is not installed
    pass
