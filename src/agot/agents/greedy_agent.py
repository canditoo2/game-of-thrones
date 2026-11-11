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
from agot.choice_context import (
    ChoiceContext,
    card_choice_mode,
    choice_track,
    resolve_choice_context,
)
from agot.engine.actions import Action, ActionKind
from agot.engine.enums import ORDERS_BY_KEY, OrderType, Phase, Track, UnitType
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

#: Printed unit strengths, used only by this heuristic. The heuristics never
#: count a Siege Engine (its 4 applies only when attacking a fortification).
_COMBAT_STRENGTH = {
    UnitType.FOOTMAN: 1,
    UnitType.KNIGHT: 2,
    UnitType.SIEGE: 4,
    UnitType.SHIP: 1,
}


def _unit_strength(state, region_id: str, house: str) -> int:
    """Combat strength of ``house``'s ready units in a region.

    Siege Engines are excluded: the greedy heuristic does not model whether the
    target is a fortification, and treating them as 0 is the conservative choice.
    """
    region = state.regions[region_id]
    if region.controller != house:
        return 0
    return sum(
        _COMBAT_STRENGTH[unit] * count
        for unit, count in region.units.items()
        if unit is not UnitType.SIEGE
    )


def _unit_value(unit) -> float:
    """Unit value, tolerant of a missing or already-parsed unit type."""
    if isinstance(unit, UnitType):
        return _UNIT_VALUE.get(unit, 1.0)
    try:
        return _UNIT_VALUE.get(UnitType(unit), 1.0)
    except (TypeError, ValueError):
        return 1.0


def _order_modifier(state, region_id: str, house: str, side: str) -> int:
    """March bonus for an attacker or Defend bonus for a defender."""
    region = state.regions[region_id]
    if region.order is None or region.order_house != house:
        return 0
    spec = ORDERS_BY_KEY[region.order]
    if side == "attacker" and spec.order_type is OrderType.MARCH:
        return spec.strength
    if side == "defender" and spec.order_type is OrderType.DEFEND:
        return spec.strength
    return 0


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
        context = resolve_choice_context(game, action)
        if context is not None:
            return self._score_choice(game, action, context)
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
            return -_unit_value(action.unit)
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

        attack = _unit_strength(state, origin_id, house) + _order_modifier(
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
            defense = _unit_strength(state, target_id, defender) + _order_modifier(
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
        battle = game.state.pending_battle
        if battle is not None and battle.get("stage") == "Doran":
            index = action.amount if action.amount is not None else 0
            tracks = (Track.IRON_THRONE, Track.FIEFDOMS, Track.KINGS_COURT)
            if not 0 <= index < len(tracks):
                return 0.0
            track = tracks[index]
            owner = action.house
            opponent = (
                battle.get("defender")
                if owner == battle.get("attacker")
                else battle.get("attacker")
            )
            if not opponent:
                return 0.0
            score = max(0.0, 7.0 - game.state.track_position(opponent, track)) * 0.5
            if track is Track.FIEFDOMS:
                fiefdoms = game.state.tracks.get(Track.FIEFDOMS, [])
                if fiefdoms and fiefdoms[0] == opponent:
                    score += 3.0
            if track is Track.KINGS_COURT:
                score += 0.25
            return score
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

    def _score_choice(
        self, game: Game, action: Action, context: ChoiceContext
    ) -> float:
        """Score a generic card/Wildling choice by what it actually does.

        The same action kinds are reused for very different tasks, so scoring
        them by kind (a ``PLAY_HOUSE_CARD`` is a card strength, a
        ``CHOOSE_CASUALTY`` is a lost unit) is wrong. The pending choice says
        which decision it is; these branches score the consequence instead.
        """
        purpose = context.purpose
        if purpose == "doran":
            return self._score_westeros_choice(game, action)
        if purpose == "wildling_tie":
            # The Iron Throne holder breaks the tie; every tied bidder is
            # equivalent for this heuristic, so keep the choice neutral.
            return 0.0
        if action.kind is ActionKind.PASS:
            return self._score_decline(context)
        if purpose in ("queen_order", "cersei_order"):
            return self._score_order_removal(game, action)
        if purpose == "renly_footman":
            return self._score_renly(game, action)
        if purpose == "horde_fort":
            return self._score_horde_fort(game, action, context)
        if purpose == "horde_muster":
            return self._score_horde_muster(game, action)
        if purpose in ("king_beyond_other", "king_beyond_highest"):
            return self._score_king_beyond(game, action, context)
        if purpose == "preemptive_raid":
            return self._score_preemptive(game, action)
        if purpose == "preemptive_highest":
            return self._score_preemptive_track(game, action, context)
        if purpose == "crow_replace_knight":
            return self._score_crow_knight(game, action)
        if purpose == "crow_upgrade_footman":
            return self._score_crow_upgrade(game, action)
        if purpose == "generic_destroy":
            return self._score_destroy(game, action)
        if purpose == "tyrion_use":
            option = action.amount if action.amount is not None else 0
            return 0.3 if option else 0.0
        if purpose == "tyrion_card":
            return self._score_card_action(game, action, context, "replace")
        if purpose == "aeron_card":
            return self._score_card_action(game, action, context, "aeron")
        if purpose == "robb_retreat":
            option = action.amount if action.amount is not None else 0
            return 0.3 if option else 0.0
        if purpose == "card_discard_or_retrieve":
            mode = card_choice_mode(context, action, game) or "discard"
            return self._score_card_action(game, action, context, mode)
        # A payload this version does not recognise: fall back on the kind.
        if context.kind == "card":
            mode = card_choice_mode(context, action, game) or "discard"
            return self._score_card_action(game, action, context, mode)
        if context.kind == "unit":
            return self._score_destroy(game, action)
        if context.kind == "region":
            return self._score_region_choice(game, action, context)
        if context.kind == "track":
            return self._score_king_beyond(game, action, context)
        if context.kind == "option":
            return self._score_preemptive(game, action)
        if context.kind == "muster":
            return self._score_horde_muster(game, action)
        return 0.0

    def _score_decline(self, context: ChoiceContext) -> float:
        """An optional decline is neutral; it can beat a harmful choice."""
        if context.purpose in (
            "horde_fort",
            "preemptive_raid",
            "king_beyond_other",
            "king_beyond_highest",
        ):
            return -5.0
        if context.allow_none:
            return 0.0
        if context.purpose in (
            "crow_replace_knight",
            "crow_upgrade_footman",
            "generic_destroy",
        ):
            return -2.0
        return 0.0

    def _score_order_removal(self, game: Game, action: Action) -> float:
        state = game.state
        region_id = action.region or ""
        region = state.regions.get(region_id)
        if region is None or region.order is None:
            return -0.5
        if region.order_house == action.house:
            return -2.0
        order_type = ORDERS_BY_KEY[region.order].order_type
        base = {
            OrderType.SUPPORT: 3.0,
            OrderType.CONSOLIDATE_POWER: 2.0,
            OrderType.DEFEND: 1.5,
            OrderType.MARCH: 1.0,
            OrderType.RAID: 0.5,
        }.get(order_type, 0.5)
        static = game.map.regions.get(region_id)
        if static is not None and static.is_fortification:
            base += 0.5
        if any(
            state.regions.get(neighbour) is not None
            and state.regions[neighbour].controller == action.house
            for neighbour in game.map.neighbours_of(region_id)
        ):
            base += 0.2
        return base

    def _score_renly(self, game: Game, action: Action) -> float:
        gain = _UNIT_VALUE[UnitType.KNIGHT] - _UNIT_VALUE[UnitType.FOOTMAN]
        region_id = action.region or ""
        static = game.map.regions.get(region_id)
        if static is not None and static.is_fortification:
            gain += 0.5
        gain += 0.08 * min(self._threat(game, region_id, action.house), 6.0)
        return gain

    def _score_horde_fort(
        self, game: Game, action: Action, context: ChoiceContext
    ) -> float:
        region_id = action.region or ""
        region = game.state.regions.get(region_id)
        if region is None:
            return -5.0
        count = context.count or context.remaining or 1
        values: list[float] = []
        for unit, amount in region.units.items():
            values.extend([_UNIT_VALUE.get(unit, 1.0)] * amount)
        for unit, amount in region.routed.items():
            values.extend([_UNIT_VALUE.get(unit, 1.0) * 0.7] * amount)
        values.sort()
        loss = sum(values[:count])
        if len(values) < count:
            # The card falls back to destroying anywhere; we lose control of
            # where the remaining losses land.
            loss += 0.5 * (count - len(values))
        # Take losses where the enemy is not massing on the border.
        loss += 0.06 * min(self._threat(game, region_id, action.house), 6.0)
        return -loss

    def _score_horde_muster(self, game: Game, action: Action) -> float:
        if action.kind is ActionKind.PASS:
            return 0.0
        if action.kind is ActionKind.MUSTER:
            return self._score_muster(game, action) + 1.0
        if action.kind is ActionKind.UPGRADE:
            return self._score_upgrade(game, action) + 1.0
        region_id = action.region or ""
        region = game.state.regions.get(region_id)
        static = game.map.regions.get(region_id)
        if region is None or static is None:
            return -1.0
        score = 1.5 * region.mustering_points
        if static.is_fortification:
            score += 1.0
        if region.controller == action.house:
            score += 2.0
        return score

    def _score_region_choice(
        self, game: Game, action: Action, context: ChoiceContext
    ) -> float:
        region_id = action.region or ""
        region = game.state.regions.get(region_id)
        static = game.map.regions.get(region_id)
        if region is None or static is None:
            return -1.0
        score = 0.0
        if static.is_fortification:
            score += 1.0
        if region.controller == action.house:
            score += 0.5
        if context.purpose == "horde_muster":
            score += 1.5 * region.mustering_points
        return score

    def _score_king_beyond(
        self, game: Game, action: Action, context: ChoiceContext
    ) -> float:
        track = choice_track(context, action)
        if track is None:
            return 0.0
        house = action.house or context.house or ""
        position = game.state.track_position(house, track)
        if position <= 0:
            return 0.0
        value = _TRACK_VALUE.get(track.value, 1.0)
        if context.purpose == "king_beyond_highest":
            # Moving to the top gains one position per step (position 1 is top).
            gain = 0.35 * (position - 1) * value
            if position > 1:
                gain += self._track_leader_bonus(track, 1)
            return gain
        # Moving to the bottom costs one position per step away from the top.
        loss = 0.35 * (6 - position) * value
        loss += self._track_leader_bonus(track, position) * 1.5
        return -loss

    @staticmethod
    def _track_leader_bonus(track: Track, position: int) -> float:
        """Extra value of holding the top of a track."""
        if position > 2:
            return 0.0
        if track is Track.FIEFDOMS:
            return 1.2 if position == 1 else 0.5
        if track is Track.KINGS_COURT:
            return 1.0 if position == 1 else 0.4
        return 0.8 if position == 1 else 0.3

    def _score_preemptive(self, game: Game, action: Action) -> float:
        house = action.house or ""
        option = action.amount if action.amount is not None else -1
        if option == 0:
            return -self._destroy_cost(game, house, 2)
        if option == 1:
            positions = [
                (game.state.track_position(house, track), track)
                for track in Track
            ]
            positions = [(position, track) for position, track in positions if position > 0]
            if not positions:
                return -1.0
            position, track = min(positions, key=lambda item: item[0])
            value = _TRACK_VALUE.get(track.value, 1.0)
            loss = 0.7 * value
            loss += self._track_leader_bonus(track, position)
            return -loss
        return 0.0

    def _score_preemptive_track(
        self, game: Game, action: Action, context: ChoiceContext
    ) -> float:
        track = choice_track(context, action)
        if track is None:
            return -1.0
        house = action.house or context.house or ""
        position = game.state.track_position(house, track)
        value = _TRACK_VALUE.get(track.value, 1.0)
        loss = 0.7 * value
        loss += self._track_leader_bonus(track, position)
        return -loss

    def _destroy_cost(self, game: Game, house: str, count: int) -> float:
        """Cheapest way to lose ``count`` units anywhere the house controls."""
        values: list[float] = []
        for region_id in game.state.controlled_regions(house):
            region = game.state.regions[region_id]
            for unit, amount in region.units.items():
                values.extend([_UNIT_VALUE.get(unit, 1.0)] * amount)
            for unit, amount in region.routed.items():
                values.extend([_UNIT_VALUE.get(unit, 1.0) * 0.7] * amount)
        values.sort()
        return sum(values[:count])

    def _score_crow_knight(self, game: Game, action: Action) -> float:
        loss = _UNIT_VALUE[UnitType.KNIGHT] - _UNIT_VALUE[UnitType.FOOTMAN]
        region_id = action.region or ""
        if action.amount == 1:
            # The routed Knight was already fighting at 0 strength.
            loss *= 0.7
        static = game.map.regions.get(region_id)
        if static is not None and static.is_fortification:
            loss += 0.2
        loss += 0.06 * min(self._threat(game, region_id, action.house), 6.0)
        region = game.state.regions.get(region_id)
        if region is not None:
            other_knights = region.units.get(UnitType.KNIGHT, 0)
            loss -= 0.1 * min(other_knights, 3)
        return -max(0.0, loss)

    def _score_crow_upgrade(self, game: Game, action: Action) -> float:
        gain = _UNIT_VALUE[UnitType.KNIGHT] - _UNIT_VALUE[UnitType.FOOTMAN]
        if action.amount == 1:
            gain *= 0.8  # a routed Knight starts routed
        region_id = action.region or ""
        static = game.map.regions.get(region_id)
        if static is not None and static.is_fortification:
            gain += 0.5
        gain += 0.08 * min(self._threat(game, region_id, action.house), 6.0)
        return gain

    def _score_destroy(self, game: Game, action: Action) -> float:
        value = _unit_value(action.unit)
        if action.amount == 1:
            value *= 0.7
        region_id = action.region or ""
        value += 0.05 * min(self._threat(game, region_id, action.house), 6.0)
        return -value

    def _score_card_action(
        self,
        game: Game,
        action: Action,
        context: ChoiceContext,
        mode: str,
    ) -> float:
        if action.kind is ActionKind.PASS:
            return self._score_decline(context)
        card = action.card
        if not card or card == "none":
            return -0.5
        house = self._card_owner(game, action, context)
        value = self._card_value(house, card)
        if mode == "retrieve":
            return value + 0.5
        if mode == "replace":
            return value + 0.2
        if mode == "discard":
            detail = context.detail or ""
            if "massing" in detail or context.wildling_card in (
                "massing_milkwater",
                "mammoth_riders",
            ):
                # Massing: lose the weakest card. (Mammoth never reaches discards.)
                return -value
            target = context.target
            if target and target != (action.house or context.house):
                # Patchface: discarding an opponent's strong card is a gain.
                return value + 0.15 * self._victory_points(game, target)
            # Massing: lose the weakest card.
            return -value
        if mode == "aeron":
            return value - 1.2  # 2 Power is not free
        return -value

    def _card_owner(
        self, game: Game, action: Action, context: ChoiceContext
    ) -> str:
        house = getattr(action, "card_house", None)
        if house:
            return house
        if context.target:
            return context.target
        if action.house:
            return action.house
        card = action.card
        for house_id, record in game.state.houses.items():
            if card in record.house_cards_available or card in record.house_cards_used:
                return house_id
        return ""

    def _card_value(self, house: str, card_id: str) -> float:
        from agot.engine.map import load_house_cards

        data = load_house_cards()
        decks = [data.deck(house)] if house in data.cards else list(data.cards.values())
        for deck in decks:
            for card in deck:
                if card["id"] == card_id:
                    return (
                        float(card["strength"])
                        + 0.8 * float(card["swords"])
                        + 0.5 * float(card["forts"])
                    )
        return 0.0

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
        if region_id not in game.map.neighbours:
            return 0.0
        threat = 0.0
        for neighbour in game.map.neighbours_of(region_id):
            region = game.state.regions[neighbour]
            if region.controller in (None, house):
                continue
            threat += _unit_strength(game.state, neighbour, region.controller)
        return threat
