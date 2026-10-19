"""Print the actions taken around a decision index of a fixture scenario.

Usage::

    .venv\\Scripts\\python tools\\debug_step.py <label> <first> <last>

Replays the recorded script and prints, for each decision in ``[first, last]``,
the acting house, the phase, the chosen action and — when a battle is pending —
its stage and forces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dump_fixture import FIXTURE, canonical_action  # noqa: E402

from agot.engine.game import Game, GameConfig  # noqa: E402

BATTLE_KEYS = (
    "region",
    "attacker",
    "defender",
    "neutral",
    "origin",
    "stage",
    "support_queue",
    "support_index",
    "attacker_units",
    "defender_units",
)


def main() -> int:
    label, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    scenario = next(entry for entry in fixture["scenarios"] if entry["label"] == label)
    config = scenario["config"]
    game = Game.new(
        GameConfig(
            houses=tuple(config["houses"]),
            seed=config["seed"],
            max_rounds=config["max_rounds"],
            use_westeros_decks=config["use_westeros_decks"],
        )
    )
    for index, decision in enumerate(scenario["decisions"]):
        actions = game.legal_actions()
        if index >= first:
            house = game.current_player()
            print(
                f"--- decision {index}: house={house} phase={game.state.phase.value} "
                f"step={game.state.action_step.value} last={game.state.action_house_last}"
            )
            print(f"    chosen: {canonical_action(actions[decision['chosen']])}")
            battle = game.state.pending_battle
            if battle is not None:
                summary = {key: battle[key] for key in BATTLE_KEYS if key in battle}
                print(f"    battle: {summary}")
                result = battle.get("result")
                if result is not None:
                    for label, side in (("attacker", result.attacker), ("defender", result.defender)):
                        print(
                            f"    {label}: house={side.house} total={side.total} "
                            f"units={side.units_strength} order={side.order_strength} "
                            f"support={side.support_strength} garrison={side.garrison_strength} "
                            f"card={side.card_strength} swords={side.swords} forts={side.forts} "
                            f"blade={side.blade}"
                        )
                    print(
                        f"    result: winner={result.winner} loser={result.loser} "
                        f"casualties={result.casualties} tiebreak={result.tiebreak}"
                    )
        game.step(actions[decision["chosen"]])
        if index >= last:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
