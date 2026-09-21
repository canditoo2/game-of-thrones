"""Tests for battle resolution and retreat."""

from __future__ import annotations

from agot.engine.combat import resolve_and_apply, resolve_battle
from agot.engine.enums import Track, UnitType
from agot.engine.game import Game, GameConfig
from agot.engine.map import load_house_cards, load_map


def _clean_game() -> Game:
    game = Game.new(GameConfig(seed=0))
    for region in game.state.regions.values():
        region.units.clear()
        region.routed.clear()
        region.controller = None
        region.order = None
        region.order_house = None
    return game


def _adjacent_land_pair() -> tuple[str, str]:
    game_map = load_map()
    for region_id in game_map.land_region_ids:
        for neighbour in game_map.land_neighbours_of(region_id):
            return region_id, neighbour
    raise AssertionError("no adjacent land regions")  # pragma: no cover


def test_strength_counts_units_and_march_order() -> None:
    game = _clean_game()
    _, target = _adjacent_land_pair()
    result = resolve_battle(
        game.state,
        target,
        "stark",
        "lannister",
        {UnitType.FOOTMAN: 3, UnitType.KNIGHT: 1},
        {UnitType.FOOTMAN: 2},
        None,
        None,
        attacker_order_strength=1,
        defender_order_strength=1,
    )
    assert result.attacker.total == 3 * 1 + 1 * 2 + 1
    assert result.defender.total == 2 + 1
    assert result.winner == "stark"


def test_merging_forces_would_be_a_bug_regression() -> None:
    """The defender's units must never be credited to the attacker."""
    game = _clean_game()
    _, target = _adjacent_land_pair()
    result = resolve_battle(
        game.state,
        target,
        "stark",
        "lannister",
        {UnitType.FOOTMAN: 1},
        {UnitType.KNIGHT: 4},
        None,
        None,
    )
    assert result.attacker.total == 1
    assert result.defender.total == 8
    assert result.winner == "lannister"


def test_ties_go_to_the_higher_fiefdoms_position() -> None:
    game = _clean_game()
    _, target = _adjacent_land_pair()
    result = resolve_battle(
        game.state,
        target,
        "stark",
        "lannister",
        {UnitType.FOOTMAN: 2},
        {UnitType.FOOTMAN: 2},
        None,
        None,
    )
    assert result.tiebreak
    stark_position = game.state.track_position("stark", Track.FIEFDOMS)
    lannister_position = game.state.track_position("lannister", Track.FIEFDOMS)
    expected = "stark" if stark_position < lannister_position else "lannister"
    assert result.winner == expected


def test_casualties_are_swords_minus_forts() -> None:
    game = _clean_game()
    _, target = _adjacent_land_pair()
    cards = load_house_cards()
    attacker_card = max(cards.deck("stark"), key=lambda card: card["swords"])
    defender_card = max(cards.deck("lannister"), key=lambda card: card["forts"])
    result = resolve_battle(
        game.state,
        target,
        "stark",
        "lannister",
        {UnitType.FOOTMAN: 5},
        {UnitType.FOOTMAN: 1},
        attacker_card["id"],
        defender_card["id"],
        attacker_order_strength=1,
    )
    assert result.winner == "stark"
    assert result.casualties == max(
        0, attacker_card["swords"] - defender_card["forts"]
    )


def test_losing_attacker_retreats_to_its_origin() -> None:
    game = _clean_game()
    origin, target = _adjacent_land_pair()
    game.state.regions[origin].units = {}  # the army has already marched out
    result = resolve_and_apply(
        game.state,
        target,
        "stark",
        "lannister",
        {UnitType.FOOTMAN: 1},
        {UnitType.KNIGHT: 3},
        None,
        None,
        attacker_origin=origin,
    )
    assert result.winner == "lannister"
    assert game.state.regions[target].controller == "lannister"
    # The surviving attacker is routed in its origin region.
    assert sum(game.state.regions[origin].routed.values()) == 1
