"""Tests for the play-layer pruning of guaranteed no-op orders."""

from __future__ import annotations

from types import SimpleNamespace

from agot.engine.actions import Action, ActionKind
from agot.tactics import is_wasted_order


def _game(port_units, sea_units, controller="baratheon"):
    port = SimpleNamespace(controller=controller, units=port_units)
    sea = SimpleNamespace(controller=None, units=sea_units)
    land = SimpleNamespace(controller="baratheon", units={})
    state = SimpleNamespace(
        regions={"dragonstone_port": port, "shipbreaker_bay": sea, "winterfell": land}
    )
    regions = {
        "dragonstone_port": SimpleNamespace(is_port=True, sea="shipbreaker_bay"),
        "shipbreaker_bay": SimpleNamespace(is_port=False, sea=None),
        "winterfell": SimpleNamespace(is_port=False, sea=None),
    }
    return SimpleNamespace(state=state, map=SimpleNamespace(regions=regions))


def _order(region: str, order: str = "defend_1b") -> Action:
    return Action(
        kind=ActionKind.PLACE_ORDER,
        house="baratheon",
        region=region,
        order=order,
    )


def test_defend_in_an_empty_port_is_wasted() -> None:
    game = _game(port_units={}, sea_units={"baratheon": 2})

    assert is_wasted_order(game, _order("dragonstone_port"))


def test_defend_in_a_shielded_port_is_wasted() -> None:
    """The owner holds the sea with ships and no enemy fleet is present."""
    game = _game(port_units={"ship": 1}, sea_units={"baratheon": 3})

    assert is_wasted_order(game, _order("dragonstone_port"))


def test_defend_in_a_threatened_port_is_kept() -> None:
    game = _game(port_units={"ship": 1}, sea_units={"baratheon": 1, "stark": 2})

    assert not is_wasted_order(game, _order("dragonstone_port"))


def test_defend_in_a_port_without_a_fleet_ashore_is_kept() -> None:
    """An empty sea can be entered by an enemy, so the order can still matter."""
    game = _game(port_units={"ship": 1}, sea_units={})

    assert not is_wasted_order(game, _order("dragonstone_port"))


def test_land_orders_and_other_order_types_are_never_pruned() -> None:
    game = _game(port_units={}, sea_units={"baratheon": 2})

    assert not is_wasted_order(game, _order("winterfell"))
    assert not is_wasted_order(game, _order("dragonstone_port", "raid_a"))
