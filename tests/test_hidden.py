"""Tests for root determinization (PIMC) of hidden information."""

from __future__ import annotations

import random

from agot.agents.mcts import MCTS
from agot.engine.enums import ORDERS_BY_KEY, OrderType, Phase, UnitType
from agot.engine.game import Game, GameConfig
from agot.engine.hidden import determinize

OBSERVER = "stark"
RIVAL = "baratheon"


def _game_in_phase(phase: Phase, seed: int = 0) -> Game:
    """A game parked in ``phase`` with every Order token cleared off the map."""
    game = Game.new(GameConfig(seed=seed))
    state = game.state
    state.phase = phase
    state.order_restrictions = []
    for region in state.regions.values():
        region.order = None
        region.order_house = None
    return game


def _place(game: Game, region_id: str, house: str, order: str) -> None:
    region = game.state.regions[region_id]
    region.units[UnitType.FOOTMAN] = 1
    region.order = order
    region.order_house = house
    placed = game.state.houses[house].orders_placed
    if order not in placed:
        placed.append(order)


def test_determinize_does_not_mutate_the_original() -> None:
    game = _game_in_phase(Phase.PLANNING)
    _place(game, "winterfell", OBSERVER, "defend_1a")
    _place(game, "kings_landing", RIVAL, "march")

    determinize(game, OBSERVER, random.Random(0))

    assert game.state.regions["winterfell"].order == "defend_1a"
    assert game.state.regions["kings_landing"].order == "march"


def test_determinize_keeps_the_observers_own_orders() -> None:
    game = _game_in_phase(Phase.PLANNING)
    _place(game, "winterfell", OBSERVER, "support_a")
    _place(game, "kings_landing", RIVAL, "march")

    for seed in range(10):
        world = determinize(game, OBSERVER, random.Random(seed))
        assert world.state.regions["winterfell"].order == "support_a"

    # The rival's token is genuinely resampled, not merely copied along.
    sampled = {
        determinize(game, OBSERVER, random.Random(seed))
        .state.regions["kings_landing"]
        .order
        for seed in range(50)
    }
    assert len(sampled) > 1


def test_hidden_orders_stay_within_the_valid_pool() -> None:
    game = _game_in_phase(Phase.PLANNING)
    state = game.state
    state.order_restrictions = ["march"]
    _place(game, "kings_landing", RIVAL, "march")
    _place(game, "dragonstone", RIVAL, "march_plus")
    _place(game, "storms_end", RIVAL, "march_minus")

    for seed in range(50):
        world = determinize(game, OBSERVER, random.Random(seed))
        hidden = [
            region.order
            for region in world.state.regions.values()
            if region.order_house == RIVAL
        ]
        assert len(hidden) == 3
        assert len(set(hidden)) == 3, "one house cannot place the same token twice"
        starred = 0
        for key in hidden:
            assert key in ORDERS_BY_KEY
            spec = ORDERS_BY_KEY[key]
            assert spec.order_type is not OrderType.MARCH, "restrictions apply"
            starred += spec.starred
        assert starred <= game.state.star_allowance(RIVAL)


def test_resampled_pool_bookkeeping_is_consistent() -> None:
    game = _game_in_phase(Phase.PLANNING)
    _place(game, "kings_landing", RIVAL, "defend_1a")
    _place(game, "dragonstone", RIVAL, "support_a")

    world = determinize(game, OBSERVER, random.Random(4))
    house = world.state.houses[RIVAL]
    on_board = [
        region.order
        for region in world.state.regions.values()
        if region.order_house == RIVAL
    ]
    assert sorted(house.orders_placed) == sorted(on_board)
    assert sorted(house.orders_available) == sorted(
        set(ORDERS_BY_KEY) - set(house.orders_placed)
    )


def test_orders_are_public_in_the_action_phase() -> None:
    game = _game_in_phase(Phase.ACTION)
    _place(game, "kings_landing", RIVAL, "march")

    world = determinize(game, OBSERVER, random.Random(0))
    assert world.state.regions["kings_landing"].order == "march"


def test_seeded_determinization_is_reproducible() -> None:
    game = _game_in_phase(Phase.PLANNING)
    _place(game, "kings_landing", RIVAL, "march")
    _place(game, "dragonstone", RIVAL, "defend_1a")

    first = determinize(game, OBSERVER, random.Random(7))
    second = determinize(game, OBSERVER, random.Random(7))
    assert first.state.regions["kings_landing"].order == second.state.regions[
        "kings_landing"
    ].order
    assert first.state.regions["dragonstone"].order == second.state.regions[
        "dragonstone"
    ].order


def test_bids_of_other_houses_are_resampled() -> None:
    game = _game_in_phase(Phase.BIDDING)
    state = game.state
    state.pending_bids = {OBSERVER: 3, RIVAL: 5, "lannister": 0}

    sampled = set()
    for seed in range(30):
        world = determinize(game, OBSERVER, random.Random(seed))
        bids = world.state.pending_bids
        assert bids[OBSERVER] == 3
        assert 0 <= bids[RIVAL] <= world.state.houses[RIVAL].power
        sampled.add(bids[RIVAL])
    assert len(sampled) > 1

    # The real game is untouched.
    assert state.pending_bids[RIVAL] == 5


def test_wildling_bids_are_resampled() -> None:
    game = _game_in_phase(Phase.WILDLING_ATTACK)
    state = game.state
    state.wildling_attack = {
        "bids": {OBSERVER: 2, RIVAL: 4},
        "strength": 6,
        "card": None,
    }

    sampled = {
        determinize(game, OBSERVER, random.Random(seed))
        .state.wildling_attack["bids"][RIVAL]
        for seed in range(30)
    }
    assert len(sampled) > 1
    for amount in sampled:
        assert 0 <= amount <= state.houses[RIVAL].power
    assert state.wildling_attack["bids"][RIVAL] == 4


def test_pimc_search_aggregates_over_determinizations() -> None:
    game = _game_in_phase(Phase.PLANNING, seed=3)
    _place(game, "kings_landing", RIVAL, "march")

    search = MCTS(simulations=40, determinizations=4, seed=0)
    result = search.search(game)

    assert result.actions == game.legal_actions()
    assert result.visits.sum() > 0
    assert abs(result.probabilities().sum() - 1.0) < 1e-9


def test_pimc_decide_returns_a_legal_action() -> None:
    game = _game_in_phase(Phase.PLANNING, seed=5)
    _place(game, "kings_landing", RIVAL, "march")

    search = MCTS(simulations=24, determinizations=3, seed=1)
    action, probabilities = search.decide(game)

    assert action in game.legal_actions()
    assert len(probabilities) == len(game.legal_actions())
