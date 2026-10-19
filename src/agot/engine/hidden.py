"""Root determinization for imperfect-information search (PIMC).

The Rust engine owns the resampling: Order tokens placed face down and secret
Power bids are resampled in the core, and the result is an independent game.
This module keeps the function the search imports and draws a seed from the
caller's generator, so a seeded PIMC run stays reproducible.

The one known remaining leak is unchanged: inside a battle the player who
chooses second can see the first player's House card in the search tree, while
the physical game reveals both simultaneously (see ``docs/methods.md``).
"""

from __future__ import annotations

import random

from agot.engine.game import Game


def determinize(game: Game, observer: str, rng: random.Random) -> Game:
    """Copy ``game`` with the hidden state resampled as ``observer`` sees it.

    The original game is never mutated. With the same ``rng`` sequence the
    result is deterministic, which keeps tests and seeded self-play reproducible.
    """
    return game.determinize(observer, rng.getrandbits(64))
