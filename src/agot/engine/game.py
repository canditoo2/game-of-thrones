"""The game object: configuration and the session engine.

The rules engine is the Rust extension (`agot._core`). This module is the
small Python facade the rest of the package imports: :class:`GameConfig` plus
``Game``, the Python-shaped view over the extension's session tier. There is no
second implementation to select between; a missing extension is an install
error, not a fallback.

The engine is driven one decision at a time::

    game = Game.new(GameConfig(seed=0))
    while not game.is_terminal():
        actions = game.legal_actions()
        game.step(actions[0])

Design notes and known simplifications live in docs/roadmap.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from agot.engine.map import load_houses

DEFAULT_HOUSES: tuple[str, ...] = (
    "baratheon",
    "lannister",
    "stark",
    "greyjoy",
    "tyrell",
    "martell",
)

#: Which implementation the package runs. There is one now; kept because runs
#: log it and results should be attributable to an engine.
BACKEND = "rust"


@dataclass
class GameConfig:
    """Options that control a single game."""

    houses: tuple[str, ...] = DEFAULT_HOUSES
    max_rounds: int = 10
    victory_castles: int = 7
    seed: int | None = None
    #: Draw and apply the three Westeros cards each round from round two.
    use_westeros_decks: bool = True
    #: Queue a March order's battle until all its destinations are chosen.
    #: Desktop play enables this; training keeps the legacy immediate battle
    #: so the frozen reference fixtures stay valid.
    deferred_battles: bool = False
    #: Reveal and resolve the three Westeros cards one at a time, resolving
    #: each card's sub-phase (mustering, supply, bidding, a Wildling attack)
    #: before drawing the next. This is the Python default for play, training
    #: and evaluation, so nobody decides with knowledge of cards that have not
    #: been revealed yet. The raw Rust fixture default stays simultaneous.
    sequential_westeros: bool = True
    #: Ask the Doran Martell player which Influence track to move. Python
    #: training/evaluation include the decision; the raw Rust fixture default
    #: stays deterministic.
    doran_choice: bool = True
    #: Ask the Iron Throne holder to break a tied highest/lowest Wildling bid.
    #: Python training/evaluation include the decision; the raw Rust fixture
    #: default stays deterministic.
    wildling_tie_choice: bool = True
    #: Expose House-card and Wildling sub-choices as real decisions. Python
    #: training/evaluation enable them; the low-level Rust helpers keep their
    #: deterministic defaults.
    choice_choices: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.houses) - set(load_houses().houses)
        if unknown:
            raise ValueError(f"unknown houses: {sorted(unknown)}")
        if not self.houses:
            raise ValueError("at least one house is required")


# Imported after GameConfig so `agot.engine.rust` can import this module to
# build a default configuration without a circular import.
from agot.engine import rust  # noqa: E402

Game = rust.RustGame

__all__ = ["BACKEND", "DEFAULT_HOUSES", "Game", "GameConfig"]
