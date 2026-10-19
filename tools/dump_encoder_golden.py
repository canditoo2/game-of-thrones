"""Generate the encoder golden data the Rust observation port is validated against.

Usage (from the repo root, with the project venv)::

    .venv\\Scripts\\python tools/dump_encoder_golden.py

Writes ``rust/crates/agot-core/tests/data/encoder_golden.json``. The Python
engine is the reference implementation, so the encoder port must reproduce it
*bit for bit*: every vector is recorded as the raw little-endian float32 bytes
of the numpy buffer, hex-encoded, never as decimal text. A single changed last
bit is a bug, and the parity test reports the first differing index.

The file has two sections:

``scenarios``
    Seeded games replayed decision by decision. Each decision carries the state
    vector and the vector of every candidate action, the canonical action
    strings, and the index of the action the script chose, so the Rust test can
    rebuild the position with ``Game::step`` and compare. The scripted path
    stays inside the phases the Rust engine can replay today (Planning and the
    Messenger Raven step) -- the same rule ``tools/dump_fixture.py`` uses.

``positions``
    Positions sampled from full games (every phase, every action kind the engine
    produces). They are recorded with a complete encoder-facing state dump
    instead of an action script, so the Rust test can check the encoder without
    depending on the rules port being finished, and so the blocks that only
    battles, bids and mustering reach are covered too.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from agot.engine import observation as obs
from agot.engine.enums import Track, UnitType
from agot.engine.game import Game, GameConfig

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "rust" / "crates" / "agot-core" / "tests" / "data" / "encoder_golden.json"

UNIT_ORDER = (UnitType.FOOTMAN, UnitType.KNIGHT, UnitType.SIEGE, UnitType.SHIP)
TRACK_ORDER = (Track.IRON_THRONE, Track.FIEFDOMS, Track.KINGS_COURT)

#: Phases whose decisions the Rust test replays with ``Game::step``.
REPLAY_PHASES = ("planning", "raven")

#: Scenarios to record. Two full tables plus the short-handed ones, which pin
#: the "absent houses' blocks stay zeroed" path.
SCENARIOS = (
    {"label": "6p-seed1", "houses": None, "seed": 1},
    {"label": "6p-seed2", "houses": None, "seed": 2},
    {"label": "3p-seed5", "houses": ["stark", "greyjoy", "tyrell"], "seed": 5},
    {"label": "4p-seed9", "houses": ["baratheon", "lannister", "martell", "tyrell"], "seed": 9},
)

#: Caps for the sampled positions: overall, and per scenario so several games
#: contribute instead of the first one exhausting the budget.
MAX_POSITIONS = 24
MAX_POSITIONS_PER_SCENARIO = 8


def make_config(spec: dict) -> GameConfig:
    if spec["houses"] is None:
        return GameConfig(seed=spec["seed"])
    return GameConfig(houses=tuple(spec["houses"]), seed=spec["seed"])


def vector_hex(vector: np.ndarray) -> str:
    """Raw little-endian float32 bytes, hex-encoded (exact, not decimal)."""
    return np.ascontiguousarray(vector, dtype=np.dtype("<f4")).tobytes().hex()


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


def encode_decision(game: Game, actions: list, actor: str | None) -> dict:
    """State vector plus every candidate action vector at one decision point."""
    cache: obs.RegionCache = {}
    vectors = [
        vector_hex(
            obs.encode_action_into(
                game, action, np.zeros(obs.ACTION_DIM, dtype=np.float32), actor, cache
            )
        )
        for action in actions
    ]
    return {
        "round": game.state.round,
        "phase": game.state.phase.value,
        "actor": actor,
        "state": vector_hex(obs.encode_state(game, actor)),
        "actions": [canonical_action(action) for action in actions],
        "vectors": vectors,
    }


def record_replay(spec: dict) -> dict:
    """Every decision along a scripted path the Rust engine can replay."""
    config = make_config(spec)
    game = Game.new(config)
    chooser = random.Random(spec["seed"] * 7919 + 13)
    decisions: list[dict] = []
    while game.state.phase.value in REPLAY_PHASES:
        actor = game.current_player()
        actions = game.legal_actions()
        if actor is None or not actions:
            break
        chosen = chooser.randrange(len(actions))
        record = encode_decision(game, actions, actor)
        record["chosen"] = chosen
        # Probe on a clone: stop *before* a step that would move the game into a
        # phase the Rust engine cannot replay yet, so every recorded position is
        # reproducible. The stopping decision itself is still recorded; the
        # Rust test simply does not step it.
        probe = game.clone()
        probe.step(actions[chosen])
        if probe.state.phase.value not in REPLAY_PHASES:
            record["chosen"] = None
            decisions.append(record)
            break
        decisions.append(record)
        game.step(actions[chosen])
    return {
        "label": spec["label"],
        "config": {
            "houses": list(config.houses),
            "max_rounds": config.max_rounds,
            "victory_castles": config.victory_castles,
            "seed": config.seed,
            "use_westeros_decks": config.use_westeros_decks,
        },
        "decisions": decisions,
    }


def state_dump(game: Game) -> dict:
    """Everything the encoder reads from a position, as plain JSON.

    Region order is the map's region order, which is also the Rust `RegionId`
    order; `fortifications` is not recorded because it is a constant of the
    board (every game starts with the map's castles and strongholds).
    """
    state = game.state
    return {
        "max_rounds": state.max_rounds,
        "victory_castles": state.victory_castles,
        "round": state.round,
        "phase": state.phase.value,
        "action_step": state.action_step.value,
        "active_house": state.active_house,
        "wildling_threat": state.wildling_threat,
        "tracks": {track.value: list(state.tracks[track]) for track in TRACK_ORDER},
        "houses": {
            house: {
                "power": house_state.power,
                "supply_level": house_state.supply_level,
                "orders_placed": list(house_state.orders_placed),
                "cards_available": list(house_state.house_cards_available),
                "eliminated": house_state.eliminated,
            }
            for house, house_state in state.houses.items()
        },
        "regions": [
            {
                "units": [state.regions[rid].units.get(unit, 0) for unit in UNIT_ORDER],
                "routed": [state.regions[rid].routed.get(unit, 0) for unit in UNIT_ORDER],
                "controller": state.regions[rid].controller,
                "owner": state.regions[rid].owner,
                "influence_token": state.regions[rid].influence_token,
                "order": state.regions[rid].order,
                "order_house": state.regions[rid].order_house,
                "garrison": state.regions[rid].garrison,
                "neutral_force": state.regions[rid].neutral_force,
                "mustering_points": state.regions[rid].mustering_points,
            }
            for rid in game.map.regions
        ],
    }


def sample_positions(scenarios: list[dict]) -> list[dict]:
    """Positions from full games, one per new action kind or new phase.

    Selection is coverage-driven: walk the scripted path of each scenario and
    record a position the first time it offers an action kind or reaches a phase
    not seen before. That keeps the file small while still touching every
    encoder branch the engine can produce. The replayed Planning and Raven
    decisions are already covered by the ``scenarios`` section, so their kinds
    start out covered.
    """
    covered_kinds: set[str] = {
        text.split("|", 1)[0]
        for scenario in scenarios
        for decision in scenario["decisions"]
        for text in decision["actions"]
    }
    covered_phases: set[str] = set(REPLAY_PHASES)
    positions: list[dict] = []
    for spec in SCENARIOS:
        game = Game.new(make_config(spec))
        chooser = random.Random(spec["seed"] * 7919 + 13)
        contributed = 0
        index = 0
        while not game.is_terminal() and len(positions) < MAX_POSITIONS:
            actions = game.legal_actions()
            if not actions:
                break
            kinds = {action.kind.value for action in actions}
            phase = game.state.phase.value
            fresh = kinds - covered_kinds
            if (fresh or phase not in covered_phases) and contributed < MAX_POSITIONS_PER_SCENARIO:
                record = encode_decision(game, actions, game.current_player())
                record["label"] = spec["label"]
                record["decision"] = index
                record["state_dump"] = state_dump(game)
                positions.append(record)
                contributed += 1
                # Only a recorded decision counts as covered: the kinds of the
                # ones skipped after the per-scenario cap must still be picked
                # up by a later scenario.
                covered_kinds |= kinds
                covered_phases.add(phase)
            game.step(actions[chooser.randrange(len(actions))])
            index += 1
    return positions


def check_card_data() -> None:
    """The Rust card lookup must agree with the Python table's key.

    Python builds its card table as ``{(house, card id): features}`` while the
    Rust test resolves a card id through the house deck (first match wins). The
    two agree as long as each deck holds `CARD_SLOTS` cards with unique ids: a
    duplicate would make Python keep the last slot and Rust the first.
    """
    from agot.engine.map import load_house_cards

    for house, deck in load_house_cards().cards.items():
        ids = [card["id"] for card in deck]
        if len(ids) != obs.CARD_SLOTS:
            raise SystemExit(f"{house} brings {len(ids)} cards, expected {obs.CARD_SLOTS}")
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{house} has duplicate card ids: {sorted(ids)}")


def main() -> int:
    check_card_data()
    scenarios = [record_replay(spec) for spec in SCENARIOS]
    positions = sample_positions(scenarios)
    payload = {
        "generator": "tools/dump_encoder_golden.py",
        "note": (
            "Reference encoder vectors recorded from the Python engine. Every "
            "vector is little-endian float32 bytes, hex-encoded; the Rust port "
            "must reproduce them exactly."
        ),
        "state_dim": obs.STATE_DIM,
        "action_dim": obs.ACTION_DIM,
        "house_order": list(obs.HOUSE_ORDER),
        "order_keys": list(obs.ORDER_KEYS),
        "unit_order": [unit.value for unit in obs.UNIT_ORDER],
        "scenarios": scenarios,
        "positions": positions,
    }
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    decisions = sum(len(scenario["decisions"]) for scenario in scenarios)
    vectors = sum(
        len(decision["vectors"])
        for scenario in scenarios
        for decision in scenario["decisions"]
    )
    sampled = sum(len(position["vectors"]) for position in positions)
    kinds = sorted({position["actions"][0].split("|")[0] for position in positions})
    print(f"wrote {GOLDEN.relative_to(REPO)} ({GOLDEN.stat().st_size / 1e6:.2f} MB)")
    for scenario in scenarios:
        print(f"  {scenario['label']:10s} {len(scenario['decisions']):3d} decisions")
    print(f"  replay: {decisions} decisions, {vectors} action vectors")
    print(f"  positions: {len(positions)} sampled, {sampled} action vectors")
    print(f"  sampled phases: {sorted({position['phase'] for position in positions})}")
    print(f"  first sampled kinds: {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
