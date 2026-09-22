"""A hand-written greedy baseline.

The greedy agent is not meant to play *well* - it is meant to be a stable,
interpretable yardstick. If the trained network cannot beat a few hundred lines
of heuristics, something is wrong with the training pipeline, so this policy
doubles as a smoke test for the engine's strategic depth.

Scores are deliberately simple and additive. The one rule that matters is the
victory condition: castles and strongholds are what win the game, so almost
every branch of the heuristic rewards taking or holding them.
"""

from __future__ import annotations

import random

import numpy as np

from agot.agents.base import Policy, one_hot_target
from agot.engine.actions import Action, ActionKind
from agot.engine.combat import order_modifier, unit_strength
from agot.engine.enums import ORDERS_BY_KEY, OrderType, Phase, UnitType
from agot.engine.game import Game

_UNIT_VALUE = {
    UnitType.KNIGHT: 2.5,
    UnitType.FOOTMAN: 1.2,
    UnitType.SIEGE: 1.0,
    UnitType.SHIP: 1.0,
}

_TRACK_VALUE = {
    "kings_court": 2.0,  # starred orders compound every round
    "iron_throne": 1.5,  # first player picks mustering and breaks ties
    "fiefdoms": 1.4,  # the Valyrian Steel Blade
}


class GreedyPolicy(Policy):
    """One-step lookahead with a small amount of noise for tie-breaking."""

    name = "greedy"

    def __init__(self, seed: int | None = None, noise: float = 0.05) -> None:
        self.rng = random.Random(seed)
        self.noise = noise

    # -- interface -------------------------------------------------------

    def decide(self, game: Game) -> tuple[Action, np.ndarray]:
        actions = game.legal_actions()
        if not actions:  # pragma: no cover - defensive
            raise RuntimeError("no legal actions to choose from")
        scores = np.array(
            [self._score(game, action) for action in actions], dtype=np.float64
        )
        if self.noise:
            scores += np.array(
                [self.rng.uniform(-self.noise, self.noise) for _ in actions]
            )
        best = int(np.argmax(scores))
        return actions[best], one_hot_target(best, len(actions))

    # -- scoring ---------------------------------------------------------

    def _score(self, game: Game, action: Action) -> float:
        kind = action.kind
        if kind is ActionKind.PASS:
            return -1.0
        if kind is ActionKind.PLACE_ORDER:
            return self._score_place_order(game, action)
        if kind is ActionKind.MARCH:
            return self._score_march(game, action)
        if kind is ActionKind.RESOLVE_RAID:
            return self._score_raid(game, action)
        if kind is ActionKind.RESOLVE_CONSOLIDATE:
            return self._score_consolidate(game, action)
        if kind is ActionKind.MUSTER:
            return self._score_muster(game, action)
        if kind is ActionKind.UPGRADE:
            return self._score_upgrade(game, action)
        if kind is ActionKind.PLAY_HOUSE_CARD:
            return self._score_card(game, action)
        if kind is ActionKind.BID:
            return self._score_bid(game, action)
        if kind is ActionKind.WESTEROS_CHOICE:
            return self._score_westeros_choice(game, action)
        if kind is ActionKind.RETREAT:
            return self._score_retreat(game, action)
        if kind is ActionKind.DECLARE_SUPPORT:
            return self._score_support(game, action)
        if kind in (ActionKind.DISBAND, ActionKind.CHOOSE_CASUALTY):
            # Destroying a unit is a pure loss, so prefer the cheapest one.
            return -_UNIT_VALUE.get(UnitType(action.unit), 1.0)
        return 0.0

    def _score_place_order(self, game: Game, action: Action) -> float:
        state = game.state
        game_map = game.map
        region_id = action.region or ""
        region = state.regions[region_id]
        static = game_map.regions[region_id]
        spec = ORDERS_BY_KEY[action.order or ""]
        army = region.total(include_routed=False)
        score = 0.0

        if spec.order_type is OrderType.MARCH:
            score += 1.2 * army
            for neighbour in game_map.neighbours_of(region_id):
                target = game_map.regions[neighbour]
                if target.is_fortification and state.regions[neighbour].controller not in (
                    None,
                    action.house,
                ):
                    score += 2.0
        elif spec.order_type is OrderType.DEFEND:
            if static.is_fortification:
                score += 4.0
            score += 0.6 * self._threat(game, region_id, action.house)
            score += 0.4 * spec.strength
        elif spec.order_type is OrderType.SUPPORT:
            contested = 0
            for neighbour in game_map.neighbours_of(region_id):
                target = state.regions[neighbour]
                if target.controller not in (None, action.house) or target.order:
                    contested += 1
            score += 1.6 * contested + 0.5 * spec.strength
        elif spec.order_type is OrderType.RAID:
            for neighbour in game_map.neighbours_of(region_id):
                order = state.regions[neighbour].order
                if order is None:
                    continue
                order_type = ORDERS_BY_KEY[order].order_type
                if order_type in (OrderType.SUPPORT, OrderType.CONSOLIDATE_POWER):
                    score += 2.0
                elif spec.starred and order_type is OrderType.DEFEND:
                    score += 1.5
        elif spec.order_type is OrderType.CONSOLIDATE_POWER:
            score += 1.0 * static.power
            score += 1.0 if spec.starred else 0.0
        return score

    def _score_march(self, game: Game, action: Action) -> float:
        state = game.state
        game_map = game.map
        origin_id = action.region or ""
        target_id = action.target or ""
        origin = state.regions[origin_id]
        target = state.regions[target_id]
        static = game_map.regions[target_id]
        house = action.house or ""

        attack = unit_strength(state, origin_id, house) + order_modifier(
            state, origin_id, house, "attacker"
        )
        score = 0.0
        if static.is_fortification:
            score += 8.0
        if target.neutral_force > 0:
            # A Neutral Force must be destroyed outright to take the area.
            if attack >= target.neutral_force:
                score += 6.0
            else:
                score -= 4.0
        elif target.controller not in (None, house):
            defender = target.controller or ""
            defense = unit_strength(state, target_id, defender) + order_modifier(
                state, target_id, defender, "defender"
            )
            if attack > defense:
                score += 5.0 + 0.2 * (attack - defense)
            elif attack == defense:
                score += 0.5
            else:
                score -= 3.0
        else:
            score += 1.5
        # Penalise abandoning a fortification with no defenders left behind.
        if game_map.regions[origin_id].is_fortification and origin.total(
            include_routed=False
        ) <= attack:
            score -= 1.5
        return score

    def _score_raid(self, game: Game, action: Action) -> float:
        target = game.state.regions[action.target or ""]
        if target.order is None:
            return 0.0
        order_type = ORDERS_BY_KEY[target.order].order_type
        if order_type is OrderType.SUPPORT:
            return 3.0
        if order_type is OrderType.CONSOLIDATE_POWER:
            return 2.0
        if order_type is OrderType.DEFEND:
            return 1.0
        return 0.5

    def _score_consolidate(self, game: Game, action: Action) -> float:
        static = game.map.regions[action.region or ""]
        region = game.state.regions[action.region or ""]
        spec = ORDERS_BY_KEY[region.order or ""] if region.order else None
        score = 1.0 + 1.0 * static.power
        if spec is not None and spec.starred:
            # A Special Consolidate Power Order may also muster in its area.
            score += 0.5
        return score

    def _score_muster(self, game: Game, action: Action) -> float:
        game_map = game.map
        region_id = action.region or ""
        static = game_map.regions[region_id]
        unit = UnitType(action.unit) if action.unit else UnitType.FOOTMAN
        score = _UNIT_VALUE.get(unit, 1.0)
        if static.is_fortification:
            score += 1.0
        if unit is UnitType.SHIP and action.target:
            score += 0.5
        return score

    def _score_upgrade(self, game: Game, action: Action) -> float:
        unit = UnitType(action.unit) if action.unit else UnitType.KNIGHT
        return _UNIT_VALUE.get(unit, 1.0) + 0.5

    def _score_card(self, game: Game, action: Action) -> float:
        if action.card is None:
            return -2.0
        from agot.engine.map import load_house_cards

        for card in load_house_cards().deck(action.house or ""):
            if card["id"] == action.card:
                return card["strength"] + 0.8 * card["swords"] + 0.5 * card["forts"]
        return 0.0

    def _score_bid(self, game: Game, action: Action) -> float:
        state = game.state
        amount = action.amount or 0
        house = action.house or ""
        power = state.houses[house].power
        if state.phase is Phase.WILDLING_ATTACK:
            # Every bid is pooled into the Night's Watch strength, so a modest
            # contribution is usually worth making.
            target = min(max(1.0, power * 0.5), float(power))
            return 1.0 - abs(amount - target) / max(1.0, power) * 2.0
        if not state.bid_tracks:
            return -float(amount)
        base = max(_TRACK_VALUE.get(name, 1.0) for name in state.bid_tracks)
        # Aim to spend a fraction of power: more when the prize is big.
        target = max(1.0, power * (0.35 + 0.15 * base))
        target = min(target, float(power))
        return base - abs(amount - target) / max(1.0, power) * 4.0

    _RESTRICTION_BY_CARD = {
        "rains_of_autumn": "march_plus",
        "storm_of_swords": "defend",
    }

    def _score_westeros_choice(self, game: Game, action: Action) -> float:
        """Prefer the holder's most useful option; 'none' is the fallback."""
        card = action.card or "none"
        if card == "none":
            return 0.0
        state = game.state
        house = action.house or ""
        if card == "mustering":
            return 2.5
        if card == "supply":
            return 3.0 if not state.within_supply(house) else 1.0
        if card == "game_of_thrones":
            return 2.0
        if card == "clash_of_kings":
            return 1.0
        restriction = self._RESTRICTION_BY_CARD.get(card)
        if restriction is not None:
            own = enemy = 0
            for region in state.regions.values():
                order_key = region.order
                if not order_key:
                    continue
                spec = ORDERS_BY_KEY[order_key]
                blocked = (
                    order_key == restriction
                    or spec.order_type.value == restriction
                )
                if not blocked:
                    continue
                if region.controller == house:
                    own += 1
                elif region.controller is not None:
                    enemy += 1
            return 1.0 + 0.6 * (enemy - own)
        return 0.5

    def _score_retreat(self, game: Game, action: Action) -> float:
        game_map = game.map
        target_id = action.target or ""
        static = game_map.regions[target_id]
        score = 0.0
        if static.is_fortification:
            score += 2.0
        score -= self._threat(game, target_id, action.house or "")
        return score

    def _score_support(self, game: Game, action: Action) -> float:
        """Back our own side, otherwise the side that is further behind.

        The greedy agent has no diplomacy model, so refusing costs nothing; the
        underdog rule exists just to break ties and keep games from snowballing.
        """
        side = action.side or "refuse"
        if side == "refuse":
            return 0.0
        battle = game.state.pending_battle
        if battle is None:  # pragma: no cover - defensive
            return 0.0
        house = action.house or ""
        attacker = battle["attacker"]
        defender = battle.get("defender")
        ally, opponent = (
            (attacker, defender) if side == "attacker" else (defender, attacker)
        )
        if ally == house:
            return 2.0
        if opponent is None:
            # Supporting an attack on a Neutral Force: mildly worthwhile.
            return 0.5
        return 0.5 + 0.5 * (
            self._victory_points(game, opponent) - self._victory_points(game, ally or "")
        )

    # -- helpers ---------------------------------------------------------

    def _victory_points(self, game: Game, house: str) -> int:
        """Number of castles and strongholds controlled by ``house``."""
        return sum(
            1
            for region_id in game.state.controlled_regions(house)
            if game.map.regions[region_id].is_fortification
        )

    def _threat(self, game: Game, region_id: str, house: str | None) -> float:
        """Rough enemy combat strength adjacent to a region."""
        threat = 0.0
        for neighbour in game.map.neighbours_of(region_id):
            region = game.state.regions[neighbour]
            if region.controller in (None, house):
                continue
            threat += unit_strength(game.state, neighbour, region.controller)
        return threat
