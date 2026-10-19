"""Self-play data generation, replay storage and evaluation.

The neural trainer is imported lazily: the data and evaluation tooling (and
the CLI's demo/evaluate commands) must work on a torch-free machine and must
not pay a multi-second torch import when they do not train.
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

_TRAINER_NAMES = {"TrainConfig", "TrainStats", "Trainer"}


def __getattr__(name: str):
    """Load the trainer only when a caller actually asks for it."""
    if name in _TRAINER_NAMES:
        from agot.training.trainer import TrainConfig, Trainer, TrainStats

        return {
            "TrainConfig": TrainConfig,
            "TrainStats": TrainStats,
            "Trainer": Trainer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
