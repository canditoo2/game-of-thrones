"""Regression tests for player-chosen casualties and retreats.

The Combat phase pauses after the dice-free strength comparison so that the
*defeated player* destroys the units they choose (rulebook page 21) and picks a
retreat destination (rulebook page 21). Before this change the engine removed
units cheapest-first and retreated to the first legal area.
"""

from __future__ import annotations

from agot.engine.actions import Action, ActionKind
from agot.engine.enums import ORDERS_BY_KEY, Phase, UnitType
from agot.engine.game import Game, GameConfig
from agot.engine.map import load_map


def _clear_board(game: Game) -> None:
    for region in game.state.regions.values():
        region.units.clear()
        region.routed.clear()
        region.controller = None
        region.order = None
        region.order_house = None
        region.garrison = 0
        region.influence_token = None
        region.mustering_points = 0
        region.neutral_force = 0
    for house in game.state.houses.values():
        house.orders_placed = []
        house.orders_available = [spec.key for spec in ORDERS_BY_KEY.values()]


def _adjacent_land_pair() -> tuple[str, str]:
    game_map = load_map()
    for region_id in sorted(game_map.land_region_ids):
        for neighbour in sorted(game_map.land_neighbours_of(region_id)):
            return region_id, neighbour
    raise AssertionError("no adjacent land regions")  # pragma: no cover


def _setup_battle(
    attacker: str,
    defender: str,
    attacker_units: dict[UnitType, int],
    defender_units: dict[UnitType, int],
    *,
    attacker_card: str | None = None,
    defender_card: str | None = None,
) -> tuple[Game, str, str]:
    """Put two armies in an adjacent pair and let the engine settle the phase."""
    game = Game.new(GameConfig(seed=0))
    state = game.state
    _clear_board(game)
    origin, target = _adjacent_land_pair()
    state.regions[origin].controller = attacker
    state.regions[origin].units = dict(attacker_units)
    state.regions[target].controller = defender
    state.regions[target].units = dict(defender_units)
    state.phase = Phase.COMBAT
    state.active_house = None
    # Cards and the Blade are treated as already decided so the test starts at
    # the strength comparison.
    state.pending_battle = {
        "region": target,
        "attacker": attacker,
        "defender": defender,
        "origin": origin,
        "attacker_order": None,
        "attacker_card": attacker_card,
        "defender_card": defender_card,
        "attacker_card_chosen": True,
        "defender_card_chosen": True,
        "attacker_units": dict(attacker_units),
        "defender_units": dict(defender_units),
        "attacker_order_strength": 0,
        "defender_order_strength": 0,
        "blade_done": True,
        "attacker_blade": False,
        "defender_blade": False,
        "stage": "cards",
    }
    game._advance()
    return game, origin, target


def _stage(game: Game) -> str:
    battle = game.state.pending_battle
    return "" if battle is None else battle.get("stage", "")


def test_defender_is_asked_which_unit_dies() -> None:
    # Melisandre (Baratheon) has one Sword icon; the defender has a Footman and
    # a Knight, so the choice of casualty changes the surviving army.
    game, _origin, _target = _setup_battle(
        "baratheon",
        "stark",
        {UnitType.FOOTMAN: 4},
        {UnitType.FOOTMAN: 1, UnitType.KNIGHT: 1},
        attacker_card="melisandre",
    )
    assert _stage(game) == "casualties"
    assert game.state.active_house == "stark"
    legal = game.legal_actions()
    assert {action.kind for action in legal} == {ActionKind.CHOOSE_CASUALTY}
    assert {action.unit for action in legal} == {"footman", "knight"}

    # The defeated player sacrifices the Footman instead of the more valuable
    # Knight, which the old auto-resolution would never have allowed.
    game.step(Action.choose_casualty("stark", _target, "footman"))
    pool = game.state.pending_battle["result"].context.units_of("stark")
    assert pool == {UnitType.KNIGHT: 1}


def test_last_defender_destroyed_area_falls() -> None:
    # One Sword icon kills the single defending Footman; the player must still
    # choose it, after which there is nobody left and the area falls.
    game, _origin, target = _setup_battle(
        "baratheon",
        "stark",
        {UnitType.FOOTMAN: 4},
        {UnitType.FOOTMAN: 1},
        attacker_card="melisandre",
    )
    assert _stage(game) == "casualties"
    assert game.state.active_house == "stark"
    game.step(Action.choose_casualty("stark", target, "footman"))
    assert game.state.pending_battle is None
    assert game.state.regions[target].controller == "baratheon"
    assert game.state.regions[target].units.get(UnitType.FOOTMAN, 0) == 4


def test_retreat_destination_is_a_player_choice() -> None:
    # The defender keeps one Knight (no Sword icons on the attacker card), so
    # it must choose where to retreat among several empty neighbours.
    game, _origin, target = _setup_battle(
        "baratheon",
        "stark",
        {UnitType.FOOTMAN: 4},
        {UnitType.KNIGHT: 2},
        attacker_card="melisandre",
    )
    game.step(Action.choose_casualty("stark", target, "knight"))
    assert _stage(game) == "retreat"
    assert game.state.active_house == "stark"
    options = [
        action.target
        for action in game.legal_actions()
        if action.kind is ActionKind.RETREAT
    ]
    assert len(options) >= 1
    chosen = options[0]
    game.step(Action.retreat("stark", target, chosen))
    region = game.state.regions[chosen]
    # The survivor was routed into the chosen area (and may already have
    # recovered if the rest of the empty round advanced to Cleanup).
    recovered = region.units.get(UnitType.KNIGHT, 0) + region.routed.get(
        UnitType.KNIGHT, 0
    )
    assert recovered == 1
    assert game.state.regions[target].controller == "baratheon"


def test_blackfish_prevents_casualties_entirely() -> None:
    # The Blackfish takes no casualties, so the phase skips the casualty step.
    game, _origin, _target = _setup_battle(
        "baratheon",
        "stark",
        {UnitType.FOOTMAN: 4},
        {UnitType.FOOTMAN: 2},
        attacker_card="melisandre",
        defender_card="blackfish",
    )
    assert _stage(game) == "retreat"
    assert game.state.active_house == "stark"


def test_robb_stark_lets_the_winner_pick_the_retreat() -> None:
    game, _origin, target = _setup_battle(
        "stark",
        "martell",
        {UnitType.FOOTMAN: 4},
        {UnitType.FOOTMAN: 2},
        attacker_card="robb_stark",
    )
    # Clear any casualties first (Robb may carry Sword icons).
    guard = 0
    while _stage(game) == "casualties" and guard < 5:
        guard += 1
        first = game.legal_actions()[0]
        game.step(first)
    assert _stage(game) == "retreat"
    # The winner, not the loser, is the one choosing.
    assert game.state.active_house == "stark"
    assert game.state.active_house != "martell"
    options = {action.target for action in game.legal_actions()}
    assert options
    game.step(Action.retreat("stark", target, sorted(options)[0]))
    assert game.state.regions[target].controller == "stark"


def test_attacker_loss_does_not_offer_a_retreat_choice() -> None:
    # A defeated attacker always retreats to where it marched from, so the
    # engine must not ask it to pick an area.
    game, origin, target = _setup_battle(
        "stark",
        "baratheon",
        {UnitType.FOOTMAN: 1},
        {UnitType.FOOTMAN: 4},
    )
    # Skip any casualty choices for the attacker.
    guard = 0
    while _stage(game) == "casualties" and guard < 5:
        guard += 1
        game.step(game.legal_actions()[0])
    # Whatever remains, the attacker is never asked where to go.
    assert game.state.active_house != "stark" or _stage(game) != "retreat"
    if game.state.pending_battle is None:
        # Battle resolved: the attacker's survivors are back at the origin.
        assert game.state.regions[origin].controller == "stark"
        assert game.state.regions[target].controller == "baratheon"
