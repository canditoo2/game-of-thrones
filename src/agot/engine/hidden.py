"""Root determinization for imperfect-information search (PIMC).

The engine tracks the *true* state, including the commitments a real player
cannot see: Order tokens placed face down, secret Power bids, and the first
player's House card inside a battle. Perfect-information MCTS reads all of them,
which is an information leak - the agent would learn to play against knowledge
it will never have at the table.

:func:`determinize` returns an independent copy of a game in which every hidden
variable the ``observer`` cannot see is resampled from a plausible prior.
Running MCTS over several determinizations and aggregating the root visit counts
(Perfect-Information Monte Carlo) yields a policy that reflects the observer's
actual information set.

Resampled here
--------------

* **Order tokens** placed by another house while they are face down (Planning
  and Messenger Raven steps). The sampled keys respect the public Order
  restrictions in force and the house's starred-order allowance, so the
  determinization is consistent with everything the observer does know.
* **Power bids** already cast by other houses, during both a *Clash of Kings*
  and a Wildling attack.

Not resampled here
------------------

House-card hands are already private in the observation encoding: the network
only ever sees the acting house's own cards (:mod:`agot.engine.observation`).
The one remaining leak is that, inside a battle, the player who chooses second
can see the first player's card in the search tree; the physical game reveals
both simultaneously. That is tracked in ``docs/methods.md``.
"""

from __future__ import annotations

import random

from agot.engine.enums import ORDERS_BY_KEY, OrderType, Phase
from agot.engine.game import Game

#: Every Order token a house owns, as a stable list.
_ALL_ORDER_KEYS: tuple[str, ...] = tuple(ORDERS_BY_KEY)


def determinize(
    game: Game, observer: str, rng: random.Random
) -> Game:
    """Copy ``game`` with the hidden state resampled as ``observer`` sees it.

    The original game is never mutated. With the same ``rng`` sequence the
    result is deterministic, which keeps tests and seeded self-play reproducible.
    """
    # The Rust engine has its own determinization (and an owned state, so there
    # is no `state.mut` to write through). Delegate when the game offers it; the
    # seed is drawn from the caller's generator so a seeded run stays
    # reproducible. The Python engine keeps the reference path below.
    rust_determinize = getattr(game, "determinize", None)
    if rust_determinize is not None:
        return rust_determinize(observer, rng.getrandbits(64))

    clone = game.clone()
    state = clone.state
    _resample_orders(state, observer, rng)
    _resample_bids(state, observer, rng)
    return clone


def _order_restricted(key: str, restrictions: list[str]) -> bool:
    spec = ORDERS_BY_KEY[key]
    return key in restrictions or spec.order_type.value in restrictions


def _take_plausible(
    pool: list[str], region, stars_left: int
) -> str | None:
    """Remove and return the first pool key that could legally sit in ``region``."""
    for index, key in enumerate(pool):
        spec = ORDERS_BY_KEY[key]
        if spec.starred and stars_left <= 0:
            continue
        if spec.order_type is OrderType.MARCH and not region.units:
            continue
        del pool[index]
        return key
    return None


def _resample_orders(
    state, observer: str, rng: random.Random
) -> None:
    """Resample every face-down Order token the observer did not place."""
    if state.orders_revealed:
        return

    hidden: dict[str, list[str]] = {}
    for region_id, region in state.regions.items():
        if region.order is None or region.order_house is None:
            continue
        if region.order_house == observer:
            continue
        hidden.setdefault(region.order_house, []).append(region_id)
    if not hidden:
        return

    restrictions = state.order_restrictions
    for house, region_ids in hidden.items():
        house_state = state.houses.get(house)
        if house_state is None:  # pragma: no cover - defensive
            continue
        pool = [key for key in _ALL_ORDER_KEYS if not _order_restricted(key, restrictions)]
        rng.shuffle(pool)
        stars_left = state.star_allowance(house)
        placed: list[str] = []
        for region_id in region_ids:
            region = state.regions[region_id]
            pick = _take_plausible(pool, region, stars_left)
            if pick is None:
                # No plausible token left (should not happen with 15 tokens);
                # keep whatever was there rather than leaving the region empty.
                pick = region.order or _ALL_ORDER_KEYS[0]
            elif ORDERS_BY_KEY[pick].starred:
                stars_left -= 1
            state.mut(region_id).order = pick
            placed.append(pick)
        # Keep the token-pool bookkeeping consistent with the resampled board.
        house_state.orders_placed = placed
        house_state.orders_available = [
            key for key in _ALL_ORDER_KEYS if key not in placed
        ]


def _resample_bids(state, observer: str, rng: random.Random) -> None:
    """Resample other houses' secret Power bids, uniform over what they can pay."""
    if state.phase is Phase.WILDLING_ATTACK:
        attack = state.wildling_attack
        if attack is None:  # pragma: no cover - defensive
            return
        bids: dict[str, int] = attack.get("bids") or {}
        for house in list(bids):
            if house == observer:
                continue
            power = state.houses[house].power
            bids[house] = rng.randint(0, max(0, power))
        return

    if state.phase is Phase.BIDDING:
        for house in list(state.pending_bids):
            if house == observer:
                continue
            power = state.houses[house].power
            state.pending_bids[house] = rng.randint(0, max(0, power))
