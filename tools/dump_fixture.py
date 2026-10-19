"""Generate the differential fixture the Rust port is validated against.

Usage (from the repo root, with the project venv)::

    .venv\\Scripts\\python tools\\dump_fixture.py

Writes ``rust/crates/agot-core/tests/data/engine_fixture.json``: for each
scenario, the opening fingerprint, every decision (acting house, the legal-action
list or a hash of it, and the action chosen) and the closing fingerprint.

Why the corpus turns the Westeros decks off
-------------------------------------------
The two engines shuffle with different generators, so they cannot share a random
stream. Setup shuffles do not matter as long as nothing is *drawn*: with
``use_westeros_decks=False`` a whole game is deterministic given the action
script, which is what lets the port be compared move for move. The phases that
do draw cards (Westeros, Wildling) are covered by Rust-side unit tests instead,
plus the scenarios here that reach them through the other paths.

Why the actions are hashed
--------------------------
A ten-round game is several hundred decisions; storing every legal-action list
verbatim would make the fixture megabytes. The first `SAMPLED_DECISIONS`
decisions of each scenario keep their full list so a mismatch can be read
directly; the rest store a count plus an FNV-1a hash of the same canonical text.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from agot.engine.enums import Track, UnitType
from agot.engine.game import Game, GameConfig
from agot.engine.observation import HOUSE_ORDER

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "rust" / "crates" / "agot-core" / "tests" / "data" / "engine_fixture.json"

UNIT_ORDER = (UnitType.FOOTMAN, UnitType.KNIGHT, UnitType.SIEGE, UnitType.SHIP)
TRACK_ORDER = (Track.IRON_THRONE, Track.FIEFDOMS, Track.KINGS_COURT)
DECK_NAMES = ("deck_i", "deck_ii", "deck_iii")

#: How many decisions per scenario keep their full legal-action list.
SAMPLED_DECISIONS = 40

#: Scenarios to record. Six-house games cover the normal table; the short-handed
#: ones pin the "absent houses' home areas are still marked" quirk.
SCENARIOS = (
    {"label": "6p-seed1", "houses": None, "seed": 1, "max_rounds": 10},
    {"label": "6p-seed2", "houses": None, "seed": 2, "max_rounds": 10},
    {"label": "6p-seed3-3rounds", "houses": None, "seed": 3, "max_rounds": 3},
    {"label": "3p-seed5", "houses": ["stark", "greyjoy", "tyrell"], "seed": 5, "max_rounds": 4},
    {
        "label": "4p-seed9",
        "houses": ["baratheon", "lannister", "martell", "tyrell"],
        "seed": 9,
        "max_rounds": 4,
    },
)


def fnv1a64(text: str) -> str:
    """64-bit FNV-1a of the UTF-8 bytes, as 16 hex digits (mirrors Rust)."""
    digest = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        digest ^= byte
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"


def canonical_action(action) -> str:
    """`kind|house|region|target|order|card|unit|side|amount|units|establish`."""

    def text(value) -> str:
        return "" if value is None else str(value)

    units = "" if action.units is None else ",".join(str(count) for count in action.units)
    return "|".join(
        [
            action.kind.value,
            text(action.house),
            text(action.region),
            text(action.target),
            text(action.order),
            text(action.card),
            text(action.unit),
            text(action.side),
            text(action.amount),
            units,
            "1" if action.establish else "0",
        ]
    )


def fingerprint(game: Game) -> dict:
    """Complete, comparable description of a position (see the Rust twin)."""
    state = game.state
    game_map = game.map

    houses = {}
    for house in HOUSE_ORDER:
        house_state = state.houses.get(house)
        if house_state is None:
            continue
        houses[house] = {
            "power": house_state.power,
            "supply_level": house_state.supply_level,
            "orders_available": list(house_state.orders_available),
            "orders_placed": list(house_state.orders_placed),
            "cards_available": list(house_state.house_cards_available),
            "cards_used": list(house_state.house_cards_used),
            "blade_used": house_state.blade_used,
            "passed": house_state.passed,
            "eliminated": house_state.eliminated,
        }

    regions = {}
    for region_id in game_map.regions:
        region = state.regions[region_id]
        regions[region_id] = {
            "controller": region.controller,
            "owner": region.owner,
            "units": {unit.value: region.units.get(unit, 0) for unit in UNIT_ORDER},
            "routed": {unit.value: region.routed.get(unit, 0) for unit in UNIT_ORDER},
            "neutral_force": region.neutral_force,
            "garrison": region.garrison,
            "mustering_points": region.mustering_points,
            "influence_token": region.influence_token,
            "order": region.order,
            "order_house": region.order_house,
        }

    return {
        "round": state.round,
        "phase": state.phase.value,
        "action_step": state.action_step.value,
        "active_house": state.active_house,
        "winner": state.winner,
        "action_house_last": state.action_house_last,
        "raven_used": state.raven_used,
        "return_phase": None if state.return_phase is None else state.return_phase.value,
        "pending_planning": list(state.pending_planning),
        "order_restrictions": list(state.order_restrictions),
        "pending_bids": dict(state.pending_bids),
        "bid_tracks": list(state.bid_tracks),
        "fortifications": sorted(state.fortifications),
        "tracks": {track.value: list(state.tracks[track]) for track in TRACK_ORDER},
        "houses": houses,
        "regions": regions,
        "wildling_threat": state.wildling_threat,
        "wildling_deck": sorted(state.wildling_deck),
        "wildling_discard": sorted(state.wildling_discard),
        "westeros_decks": {
            name: {
                "cards": sorted(state.westeros_decks[name]),
                "position": state.westeros_deck_pos[name],
                "drawn": state.westeros_drawn[name],
            }
            for name in DECK_NAMES
        },
        "pending_westeros": (
            None if state.pending_westeros is None else list(state.pending_westeros)
        ),
        "westeros_choice": (
            None
            if state.westeros_choice is None
            else {
                "card": state.westeros_choice["card"],
                "holder": state.westeros_choice["holder"],
                "options": list(state.westeros_choice["options"]),
            }
        ),
        "pending_march": (
            None
            if state.pending_march is None
            else {
                "house": state.pending_march["house"],
                "origin": state.pending_march["origin"],
            }
        ),
        "pending_battle_present": state.pending_battle is not None,
        "wildling_attack_present": state.wildling_attack is not None,
    }


def build_scenario(spec: dict) -> dict:
    houses = spec["houses"]
    config = GameConfig(
        houses=tuple(houses) if houses else GameConfig().houses,
        seed=spec["seed"],
        max_rounds=spec["max_rounds"],
        # See the module docstring: no card is ever drawn, so the two engines
        # stay comparable move for move.
        use_westeros_decks=False,
    )
    game = Game.new(config)
    opening = fingerprint(game)

    chooser = random.Random(spec["seed"] * 7919 + 13)
    decisions = []
    steps = 0
    while not game.is_terminal() and steps < 5000:
        house = game.current_player()
        actions = game.legal_actions()
        if house is None or not actions:
            break
        canonical = [canonical_action(action) for action in actions]
        chosen = chooser.randrange(len(actions))
        record = {
            "house": house,
            "count": len(actions),
            "hash": fnv1a64("\n".join(canonical)),
            "chosen": chosen,
        }
        if len(decisions) < SAMPLED_DECISIONS:
            record["actions"] = canonical
        decisions.append(record)
        game.step(actions[chosen])
        steps += 1

    return {
        "label": spec["label"],
        "config": {
            "houses": list(config.houses),
            "max_rounds": config.max_rounds,
            "victory_castles": config.victory_castles,
            "seed": config.seed,
            "use_westeros_decks": config.use_westeros_decks,
        },
        "initial": opening,
        "decisions": decisions,
        "final": fingerprint(game),
    }


def main() -> int:
    scenarios = [build_scenario(spec) for spec in SCENARIOS]
    payload = {
        "generator": "tools/dump_fixture.py",
        "note": (
            "Reference fingerprints and action scripts recorded from the Python "
            "engine. The Rust port must reproduce every field exactly. Westeros "
            "decks are off so the corpus is deterministic; see the generator "
            "docstring."
        ),
        "scenarios": scenarios,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")

    print(f"wrote {FIXTURE.relative_to(REPO)} ({FIXTURE.stat().st_size / 1024:.0f} KB)")
    for scenario in scenarios:
        decisions = len(scenario["decisions"])
        final = scenario["final"]
        print(
            f"  {scenario['label']:18s} {decisions:4d} decisions, "
            f"round {final['round']}, final phase {final['phase']}, "
            f"winner {final['winner']}"
        )
    print(f"  total: {len(scenarios)} scenarios, {sum(len(s['decisions']) for s in scenarios)} decisions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
