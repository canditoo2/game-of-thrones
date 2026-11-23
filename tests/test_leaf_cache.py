"""The Rust driver's cross-pass leaf evaluation cache is exact.

With an evaluator that is a pure row-wise function of its input, self-play
with the cache on must return the same arrays, bit for bit, as with it off:
the cache only changes *which* leaves reach the evaluator, never what a leaf is
scored. The fake evaluators below hash each leaf's float bit patterns with
integer arithmetic, so their outputs cannot depend on batch shape or on a
row's position in the batch (a BLAS matmul could).
"""

from __future__ import annotations

import numpy as np
import pytest

from agot import _core as core
from agot.engine.game import GameConfig
from agot.training import rust_engine
from agot.training.parallel import SelfPlayPlan

N_HOUSES = 6
FIELDS = (
    "states",
    "actions",
    "offsets",
    "policy",
    "players",
    "outcomes",
    "winners",
    "steps",
    "examples_per_game",
    "rounds",
)


def hashing_evaluator(salt: int, calls: list | None = None):
    """A deterministic, row-wise pure stand-in for the network.

    Logits depend on the leaf's state row and the candidate's own row; values
    on the state row and all of the leaf's candidates (the real value head
    reads the candidates too). All mixing is exact uint64 arithmetic.
    """
    rng = np.random.default_rng(salt)
    state_mix = rng.integers(1, 1 << 20, size=97, dtype=np.uint64)
    action_mix = rng.integers(1, 1 << 20, size=141, dtype=np.uint64)
    house_mix = rng.integers(1, 1 << 20, size=N_HOUSES, dtype=np.uint64)

    def evaluate(state, actions, index, mask):
        if calls is not None:
            calls.append((int(state.shape[0]), int(actions.shape[0])))
        state_bits = np.ascontiguousarray(state).view(np.uint32).astype(np.uint64)
        action_bits = np.ascontiguousarray(actions).view(np.uint32).astype(np.uint64)
        state_hash = (state_bits * state_mix).sum(axis=1)
        action_hash = (action_bits * action_mix).sum(axis=1)
        index = np.asarray(index, dtype=np.int64)
        slot_hash = action_hash + state_hash[index] * np.uint64(31)
        logits = ((slot_hash % np.uint64(4001)).astype(np.float32) / 1000.0) - 2.0
        leaf_hash = state_hash.copy()
        np.add.at(leaf_hash, index, action_hash)
        values = (
            ((leaf_hash[:, None] * house_mix[None, :]) % np.uint64(2001)).astype(np.float32)
            / 1000.0
        ) - 1.0
        return logits.astype(np.float32), values.astype(np.float32)

    return evaluate


def assert_same_results(left, right):
    assert len(left) == len(right) == len(FIELDS)
    for name, a, b in zip(FIELDS, left, right, strict=True):
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype, name
            assert a.shape == b.shape, name
            assert a.tobytes() == b.tobytes(), f"{name} differs bit for bit"
        else:
            assert list(a) == list(b), name


#: A training-shaped request: temperature schedule, the Python rule options,
#: the FAQ Lannister ship and phase budgets that actually cap something.
TRAINING_SPEC = {
    "games": 3,
    "simulations": 32,
    "batch_games": 3,
    "seed": 11,
    "temperature": 1.0,
    "temperature_rounds": 1,
    "final_temperature": 0.25,
    "max_rounds": 2,
    "use_westeros_decks": True,
    "deferred_battles": True,
    "sequential_westeros": True,
    "doran_choice": True,
    "choice_choices": True,
    "lannister_port_ship": True,
    "phase_budgets": {"planning": 8, "wildling_attack": 4},
}


@pytest.mark.parametrize("determinizations", [1, 2])
def test_cache_on_and_off_record_identical_games(determinizations):
    spec = dict(TRAINING_SPEC, determinizations=determinizations)

    off_calls: list = []
    off = core.selfplay(dict(spec, leaf_cache=False), hashing_evaluator(5, off_calls))
    off_stats = core.selfplay_stats()
    on_calls: list = []
    on = core.selfplay(
        dict(spec, leaf_cache=True, leaf_cache_verify=True), hashing_evaluator(5, on_calls)
    )
    on_stats = core.selfplay_stats()

    assert_same_results(on, off)

    assert off_stats["leaf_cache"] is False and on_stats["leaf_cache"] is True
    # Both runs searched the same leaves ...
    assert on_stats["leaves"] == off_stats["leaves"] > 0
    assert on_stats["slots"] == off_stats["slots"]
    # ... but the cache answered a large share of them without the network,
    # and every hit was checked byte for byte against the stored input.
    assert on_stats["cache_hits"] > 0
    assert on_stats["cache_verified"] == on_stats["cache_hits"]
    assert on_stats["rows_sent"] < off_stats["rows_sent"]
    assert on_stats["slots_sent"] < off_stats["slots_sent"]
    assert (
        on_stats["cache_hits"] + on_stats["pass_duplicates"] + on_stats["rows_sent"]
        == on_stats["leaves"]
    )
    # A pass whose every leaf is a hit makes no callback at all.
    assert on_stats["calls_skipped"] > 0
    assert on_stats["evaluator_calls"] == len(on_calls)
    assert on_stats["evaluator_calls"] + on_stats["calls_skipped"] == len(off_calls)
    assert sum(rows for rows, _ in on_calls) == on_stats["rows_sent"]


def test_cache_matches_the_undeduplicated_path_too():
    """Off with dedupe off is the oldest reference path; the cache matches it."""
    spec = dict(TRAINING_SPEC, determinizations=2, games=2, batch_games=2)
    reference = core.selfplay(
        dict(spec, leaf_cache=False, dedupe_leaves=False), hashing_evaluator(9)
    )
    cached = core.selfplay(dict(spec, leaf_cache_verify=True), hashing_evaluator(9))
    assert_same_results(cached, reference)


def test_a_tiny_cache_still_records_identical_games():
    """Evictions only cost re-evaluations; they never change a result."""
    spec = dict(TRAINING_SPEC, determinizations=1)
    off = core.selfplay(dict(spec, leaf_cache=False), hashing_evaluator(3))
    tiny = core.selfplay(
        dict(spec, leaf_cache_verify=True, leaf_cache_entries=16, leaf_cache_logits=256),
        hashing_evaluator(3),
    )
    stats = core.selfplay_stats()
    assert_same_results(tiny, off)
    assert stats["cache_rotations"] > 0
    assert stats["cache_evicted"] > 0
    assert stats["cache_peak_entries"] <= 16
    # One entry may exceed half the logit cap on its own; the rest is bounded.
    assert stats["cache_peak_logits"] <= 256 + 200


def test_two_models_keep_separate_caches():
    """Evaluate-style request: two networks share leaves, never outputs.

    Both fake networks see many identical inputs (the same opening, the same
    aggregate state rows), so a shared cache would hand one network the other's
    scores and change the games. Equality with the cache off rules that out.
    """
    houses = GameConfig().houses
    games = 4
    seat_models = [
        [(seat + game) % 2 for seat in range(len(houses))] for game in range(games)
    ]
    spec = {
        "games": games,
        "simulations": 16,
        "determinizations": 1,
        "batch_games": 4,
        "seed": 21,
        "seeds": [21 + index for index in range(games)],
        "temperature": 0.0,
        "temperature_rounds": 0,
        "final_temperature": 0.0,
        "max_rounds": 2,
        "use_westeros_decks": True,
        "deferred_battles": True,
        "sequential_westeros": True,
        "choice_choices": True,
        "houses": list(houses),
        "seat_models": seat_models,
        "record_opponents": False,
    }
    off = core.selfplay(
        dict(spec, leaf_cache=False), hashing_evaluator(1), None, [hashing_evaluator(2)]
    )
    on = core.selfplay(
        dict(spec, leaf_cache_verify=True), hashing_evaluator(1), None, [hashing_evaluator(2)]
    )
    stats = core.selfplay_stats()
    assert_same_results(on, off)
    assert stats["cache_hits"] > 0


def test_the_training_plan_reaches_the_driver():
    plan = SelfPlayPlan(games=1)
    assert plan.leaf_cache is True
    assert rust_engine.spec_for(plan)["leaf_cache"] is True
    assert rust_engine.spec_for(SelfPlayPlan(leaf_cache=False))["leaf_cache"] is False
    assert rust_engine.spec_for(plan, leaf_cache=False)["leaf_cache"] is False


def test_try_batched_self_play_forwards_the_switch(monkeypatch):
    seen: list = []

    class Recorder:
        def selfplay(self, spec, evaluator, decider=None, extra=None):
            seen.append(spec["leaf_cache"])
            return core.selfplay(spec, evaluator, decider)

    class Fake:
        def as_leaf_evaluator(self):
            return hashing_evaluator(4)

    monkeypatch.setattr(rust_engine, "core", lambda: Recorder())
    configs = [GameConfig(seed=1, max_rounds=1)]
    for flag in (True, False):
        records = rust_engine.try_batched_self_play(
            Fake(), configs, simulations=4, leaf_cache=flag
        )
        assert records and records[0].examples
    assert seen == [True, False]
