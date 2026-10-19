"""Replay one fixture scenario in the Python engine and print a decision point.

Usage (from the repo root)::

    .venv\\Scripts\\python tools\\debug_parity.py <label> <decision> [--state-only]

Prints the canonical legal actions at that decision (and, with ``--state-only``,
the state fingerprint instead). Used to diff a failing differential test against
the reference: the Rust side prints its own list, the two are compared by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dump_fixture import FIXTURE, canonical_action, fingerprint  # noqa: E402

from agot.engine.game import Game, GameConfig  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    label = sys.argv[1]
    index = int(sys.argv[2])
    state_only = "--state-only" in sys.argv

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    scenario = next(
        (entry for entry in fixture["scenarios"] if entry["label"] == label), None
    )
    if scenario is None:
        print(f"unknown scenario {label!r}")
        return 2

    config = scenario["config"]
    game = Game.new(
        GameConfig(
            houses=tuple(config["houses"]),
            seed=config["seed"],
            max_rounds=config["max_rounds"],
            use_westeros_decks=config["use_westeros_decks"],
        )
    )
    for decision in scenario["decisions"][:index]:
        actions = game.legal_actions()
        game.step(actions[decision["chosen"]])

    payload = {
        "fingerprint": fingerprint(game),
        "actions": [canonical_action(action) for action in game.legal_actions()],
    }
    if state_only:
        payload.pop("actions")
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
