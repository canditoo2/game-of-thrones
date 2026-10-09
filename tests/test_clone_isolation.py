"""Clones must be fully independent of the state they came from.

MCTS materialises a node by cloning its parent and applying one action. If that
clone shared any *mutable* state with the parent, the parent would silently
corrupt and the search would be reading different positions than it thinks.

Since copy-on-write landed, :meth:`GameState.clone` shares every
:class:`RegionState` with its source and relies on the engine writing through
:meth:`GameState.mut`. That is only safe while *every* write does so. These
tests pin the invariant down from the outside, with no knowledge of which call
sites exist: cloning a game and stepping the clone must leave the original
identical, for every legal action, at every decision of a game.

Isolation has to hold in *both* directions, and the two directions exercise
different code. Stepping a clone tests the clone's copy-on-write; stepping the
source tests the source's. The suite originally only checked the first, which
let a real defect through: ``mut`` trusted its ``_owned`` set without noticing
that a later clone had put those very objects back on the shared heap, so the
*next* source write leaked into the clone. See ``test_stepping_the_source_
never_touches_an_existing_clone`` and the "Notable fixes" section of
``docs/roadmap.md``.

The comparison is a structural fingerprint rather than ``state == other``: a
pending battle embeds its :class:`BattleContext`, whose ``state`` back-reference
makes dataclass equality recurse forever. The fingerprint covers the
aliasing-prone mutable data: regions, houses, flat bookkeeping, and the four
pending-effect dicts (which ``clone`` copies by hand instead of with
``copy.deepcopy``, and which the engine writes into in place).
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from agot.agents.mcts import MCTS
from agot.engine.game import Game, GameConfig
from agot.engine.state import GameState


def _freeze(value, owner=None):
    """Hashable deep snapshot of a pending-effect value.

    Dataclass instances are expanded field by field rather than compared
    wholesale, so a leaked object surfaces as a changed fingerprint. The one
    edge that must not be followed is a ``GameState``: a parked battle's
    ``BattleContext`` back-references the state it is parked in, which would
    recurse forever and drag the whole board into the snapshot. It is recorded as
    a marker instead, and the marker says whether it points at the state this
    snapshot belongs to — that is the invariant the clone has to preserve.
    """
    if isinstance(value, dict):
        return (
            "dict",
            tuple(sorted((key, _freeze(item, owner)) for key, item in value.items())),
        )
    if isinstance(value, list):
        return ("list", tuple(_freeze(item, owner) for item in value))
    if isinstance(value, GameState):
        return "self-state" if value is owner else "foreign-state"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__name__,
            tuple(
                (f.name, _freeze(getattr(value, f.name), owner))
                for f in dataclasses.fields(value)
            ),
        )
    return value


def _fingerprint(state) -> tuple:
    regions = tuple(
        (
            rid,
            r.controller,
            r.owner,
            tuple(sorted((u.value, c) for u, c in r.units.items())),
            tuple(sorted((u.value, c) for u, c in r.routed.items())),
            r.neutral_force,
            r.garrison,
            r.mustering_points,
            r.influence_token,
            r.order,
            r.order_house,
        )
        for rid, r in sorted(state.regions.items())
    )
    houses = tuple(
        (
            hid,
            h.power,
            h.supply_level,
            tuple(h.orders_available),
            tuple(h.orders_placed),
            tuple(h.house_cards_available),
            tuple(h.house_cards_used),
            h.blade_used,
            h.passed,
            h.eliminated,
        )
        for hid, h in sorted(state.houses.items())
    )
    return (
        state.round,
        state.phase,
        state.action_step,
        state.active_house,
        state.winner,
        state.action_house_last,
        state.raven_used,
        state.return_phase,
        state.wildling_threat,
        tuple(sorted((str(k), tuple(v)) for k, v in state.tracks.items())),
        tuple(sorted(state.westeros_deck_pos.items())),
        tuple(sorted(state.westeros_drawn.items())),
        tuple(state.order_restrictions),
        tuple(state.pending_planning),
        tuple(sorted(state.pending_bids.items())),
        tuple(state.bid_tracks),
        None if state.pending_westeros is None else tuple(state.pending_westeros),
        _freeze(state.wildling_attack, state),
        _freeze(state.westeros_choice, state),
        _freeze(state.pending_battle, state),
        _freeze(state.pending_march, state),
        regions,
        houses,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clone_and_step_never_touches_the_source(seed: int) -> None:
    game = Game.new(GameConfig(seed=seed, max_rounds=10))
    rng = random.Random(seed)
    steps = 0
    while not game.is_terminal() and steps < 4000:
        actions = game.legal_actions()
        if not actions:
            break
        steps += 1
        baseline = _fingerprint(game.state)
        for action in actions:
            child = game.clone()
            child.step(action)
            assert _fingerprint(game.state) == baseline, (
                f"seed {seed}: stepping {action!r} on a clone mutated the source"
            )
        game.step(rng.choice(actions))
    assert steps > 0


def test_clone_shares_regions_until_they_are_written() -> None:
    """The copy-on-write contract: untouched regions are shared, written ones not."""
    game = Game.new(GameConfig(seed=0))
    region_id = "winterfell"

    child = game.clone()
    assert child.state.regions[region_id] is game.state.regions[region_id]

    before = game.state.regions[region_id].controller
    child.state.mut(region_id).controller = "lannister"

    assert child.state.regions[region_id] is not game.state.regions[region_id]
    assert game.state.regions[region_id].controller == before


def test_mut_is_idempotent_and_only_copies_once() -> None:
    game = Game.new(GameConfig(seed=0))
    child = game.clone()
    region_id = "winterfell"
    assert child.state.regions[region_id] is game.state.regions[region_id]
    first = child.state.mut(region_id)
    second = child.state.mut(region_id)
    assert first is second
    assert first is child.state.regions[region_id]


def test_a_region_taken_private_before_the_clone_is_still_copied_after_it() -> None:
    """Regression: ``mut`` must not trust ``_owned`` across a clone.

    The source takes a region private, *then* a clone is made and shares that
    private object. ``mut`` used to see the id still in ``_owned`` and hand the
    shared object back for the next write, so the source wrote straight into the
    clone. Ownership has to be dropped when a clone appears.
    """
    game = Game.new(GameConfig(seed=0))
    region_id = "winterfell"
    game.clone()  # a virgin state shares nothing, but it cannot *stay* virgin
    private = game.state.mut(region_id)
    assert region_id in game.state._owned

    child = game.clone()
    assert child.state.regions[region_id] is private

    game.state.mut(region_id).controller = "lannister"

    assert child.state.regions[region_id] is private
    assert child.state.regions[region_id].controller != "lannister"
    assert game.state.regions[region_id] is not private


def _parked_battle_result():
    """A position where a resolved battle is waiting on casualties or a retreat.

    Battles park their ``BattleResult`` on ``pending_battle`` and apply the
    post-combat House-card abilities a step or two later, so this is the moment
    the back-reference matters.
    """
    for seed in range(12):
        game = Game.new(GameConfig(seed=seed, max_rounds=10))
        rng = random.Random(seed)
        for _ in range(4000):
            if game.is_terminal():
                break
            actions = game.legal_actions()
            if not actions:
                break
            battle = game.state.pending_battle
            if battle is not None and battle.get("result") is not None:
                return game
            game.step(rng.choice(actions))
    raise AssertionError("no parked battle result found")  # pragma: no cover


def test_a_cloned_battle_context_points_at_the_clone() -> None:
    """A parked battle must keep its ``state`` back-reference inside the clone.

    ``BattleResult.context.state`` is not merely read: ``apply_post_battle``
    writes Power (Tywin), unit upgrades (Renly), Order removals (Cersei) and card
    returns (Roose, Patchface) through it. On the live game that reference *is*
    the game's own state, so a clone's must be the clone's own state too --
    otherwise the clone resolves those abilities against a detached ghost board
    and the search quietly plays by different rules than the game being played.
    """
    game = _parked_battle_result()
    source = game.state
    assert source.pending_battle["result"].context.state is source

    clone = game.clone()
    assert clone.state.pending_battle["result"].context.state is clone.state
    # Rebinding the clone's context must leave the source pointing at itself.
    assert source.pending_battle["result"].context.state is source


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_stepping_the_source_never_touches_an_existing_clone(seed: int) -> None:
    """The other direction of isolation: an existing clone must stay frozen.

    The engine writes a handful of regions per step. If any of those writes
    reached a region object a clone is sharing, the clone would silently change
    under whoever is holding it -- which is exactly the shape of a held
    :class:`~agot.engine.game.GameRecord` or an open search root.
    """
    game = Game.new(GameConfig(seed=seed, max_rounds=10))
    rng = random.Random(seed)
    checked = 0
    while not game.is_terminal() and checked < 1500:
        actions = game.legal_actions()
        if not actions:
            break
        child = game.clone()
        baseline = _fingerprint(child.state)
        game.step(rng.choice(actions))
        assert _fingerprint(child.state) == baseline, (
            f"seed {seed}: stepping the source mutated an existing clone"
        )
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("seed", [4])
def test_mcts_search_does_not_mutate_the_source(seed: int) -> None:
    game = Game.new(GameConfig(seed=seed, max_rounds=10))
    rng = random.Random(seed)
    checked = 0
    for _ in range(60):
        if game.is_terminal():
            break
        actions = game.legal_actions()
        if not actions:
            break
        baseline = _fingerprint(game.state)
        MCTS(simulations=12, determinizations=3, seed=seed).search(game)
        assert _fingerprint(game.state) == baseline, (
            "MCTS search mutated the game it searched"
        )
        checked += 1
        game.step(rng.choice(actions))
    assert checked > 0


def test_copy_pending_reproduces_every_nested_container() -> None:
    """``clone`` copies the pending dicts by hand, so the copy must be deep.

    The engine writes into ``pending_battle["support_choices"]`` and
    ``wildling_attack["bids"]`` *in place*. If the hand-copy shared those inner
    dicts, a Support declaration in one clone would appear in every clone.
    """
    from agot.engine.state import _copy_pending

    source = {
        "scalar": 1,
        "flag": None,
        "list": ["a", "b"],
        "dict": {"a": 1},
        "nested": {"list": [1, 2]},
    }
    copied = _copy_pending(source)

    assert copied == source
    assert copied is not source
    for key in ("list", "dict", "nested"):
        assert copied[key] is not source[key]
    assert copied["nested"]["list"] is not source["nested"]["list"]

    copied["dict"]["b"] = 2
    copied["list"].append("c")
    copied["nested"]["list"].append(3)
    assert source["dict"] == {"a": 1}
    assert source["list"] == ["a", "b"]
    assert source["nested"]["list"] == [1, 2]

    assert _copy_pending(None) is None
