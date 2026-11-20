"""Scripted (greedy) seats on the Rust self-play driver.

The driver hands each scripted decision to a Python policy through
``rust_engine._scripted_decider``. These tests pin what that path must keep:
the decision and the random draws of every scripted seat are exactly those of
``GreedyPolicy.decide`` on a full ``RustGame`` view of the same position, and a
training-shaped request with scripted seats records the same games with the
leaf cache on and off.
"""

from __future__ import annotations

import copy
import random

import pytest

from agot import _core as core
from agot.agents import GreedyPolicy
from agot.engine.rust import RustGame
from agot.training import rust_engine
from test_leaf_cache import TRAINING_SPEC, assert_same_results, hashing_evaluator

HOUSES = ("stark", "lannister", "baratheon", "greyjoy", "tyrell", "martell")

#: Six games over a few rounds so every phase reaches a scripted seat.
SCRIPTED_SPEC = dict(TRAINING_SPEC, games=6, batch_games=4, max_rounds=3)


def _opponents(games: int, seed: int, rate: float) -> list[dict]:
    """One independent ``GreedyPolicy`` per scripted seat, as the training plan draws them."""
    opponents = []
    for game in range(games):
        rng = random.Random(seed + game)
        seats = {}
        for house in HOUSES:
            if rng.random() < rate:
                seats[house] = GreedyPolicy(rng.randrange(1 << 30), noise=0.05)
        opponents.append(seats)
    return opponents


def _spec(opponents, record: bool, **extra) -> dict:
    spec = dict(SCRIPTED_SPEC, **extra)
    spec["scripted"] = [list(seats) for seats in opponents]
    spec["record_opponents"] = record
    return spec


@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("determinizations", [1, 2])
def test_scripted_seats_record_identical_games_cache_on_and_off(record, determinizations):
    results = []
    for cache in (False, True):
        opponents = _opponents(SCRIPTED_SPEC["games"], 1000, 0.35)
        assert any(opponents)
        spec = _spec(
            opponents,
            record,
            determinizations=determinizations,
            leaf_cache=cache,
            leaf_cache_verify=cache,
        )
        decider = rust_engine._scripted_decider(opponents, record)
        results.append(core.selfplay(spec, hashing_evaluator(5), decider))
    assert_same_results(results[0], results[1])


@pytest.mark.parametrize("record", [False, True])
def test_scripted_decisions_match_greedy_on_a_full_view(record):
    """Each driver decision equals ``GreedyPolicy.decide`` on a full clone.

    The twin policy starts from a copy of the seat's RNG, so the check also
    pins how many random draws the decision consumed.
    """
    opponents = _opponents(SCRIPTED_SPEC["games"], 2000, 0.4)
    inner = rust_engine._scripted_decider(opponents, record)
    checked = []

    def checking(game_index, house, game):
        policy = opponents[game_index][house]
        twin = copy.deepcopy(policy)
        reference = RustGame(game.clone())
        action, target = twin.decide(reference)
        expected = reference.legal_actions().index(action)
        index, reply_target = inner(game_index, house, game)
        assert index == expected
        assert policy.rng.getstate() == twin.rng.getstate()
        if record:
            assert reply_target == [float(x) for x in target]
        else:
            assert reply_target is None
        checked.append(index)
        return index, reply_target

    spec = _spec(opponents, record, determinizations=1)
    result = core.selfplay(spec, hashing_evaluator(5), checking)
    assert len(checked) > 50
    steps, examples = result[7], result[8]
    if record:
        assert list(steps) == list(examples)
    else:
        assert sum(steps) - sum(examples) == len(checked)
