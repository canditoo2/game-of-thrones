"""Play-layer pruning of decisions that cannot change the game.

These helpers never alter what the engine allows: they only recognise orders
that are guaranteed to be wasted, so desktop play can leave them out of the
list the human sees and the bots can refuse to pick them.
"""

from __future__ import annotations

from agot.engine.actions import Action, ActionKind


def _is_defend(action: Action) -> bool:
    return action.kind is ActionKind.PLACE_ORDER and (action.order or "").startswith(
        "defend"
    )


def is_wasted_order(game, action: Action) -> bool:
    """True for a Defend order in a port where it cannot ever matter.

    A port can only be attacked from its adjacent sea. A Defend order there is
    pointless when the port holds no unit at all, or when the port's owner
    already holds the adjacent sea with ships and no enemy ship is present: the
    attacker would have to destroy that fleet first, and the order would have
    been spent long before.
    """
    if not _is_defend(action) or not action.region:
        return False
    region = game.state.regions.get(action.region)
    if region is None:
        return False
    definition = game.map.regions.get(action.region)
    if definition is None or not definition.is_port:
        return False
    if sum(region.units.values()) == 0:
        return True
    if not definition.sea:
        return False
    sea = game.state.regions.get(definition.sea)
    if sea is None or region.controller is None:
        return False
    enemy = sum(
        count for house, count in sea.units.items() if house != region.controller
    )
    own = sea.units.get(region.controller, 0)
    return enemy == 0 and own > 0
