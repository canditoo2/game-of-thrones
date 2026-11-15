"""Audit the bot decisions of a logged game against a deeper search.

Replays the game exactly (same engine, same RNG, recovering the dialog choices),
and at every bot decision runs MCTS with a larger budget than the one the seat
used. For each decision it reports whether the deeper search would have played
something else and by how much it prefers that move (the Q gap for the acting
house), so a "silly move" can be classified as a depth problem or a policy one.

Usage::

    python tools/audit_bot_moves.py checkpoints/play-logs/play-X.jsonl \
        --simulations 512 --top 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from agot.agents import MCTS
from agot.engine.game import Game
from agot.nn.model import NeuralEvaluator, PolicyValueNet

TOOL = Path(__file__).resolve().parent / "reanalyze_human_game.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("reanalyze_human_game", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--checkpoint", default=None, help="fallback for every seat")
    parser.add_argument(
        "--checkpoint-map",
        default=None,
        help="per-house networks, e.g. 'baratheon=run6/agot.pt,tyrell=run5/agot.pt'",
    )
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--determinizations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tool = load_tool()
    records = tool.load_records(args.log)
    start = next(record for record in records if record.get("type") == "start")
    decisions = tool.active_decisions(records)

    mapping = tool_parse_map(args.checkpoint_map)
    fallback = args.checkpoint or start.get("checkpoint")
    if fallback is None and not mapping:
        raise SystemExit("no checkpoint: pass --checkpoint or --checkpoint-map")

    models: dict[str, NeuralEvaluator] = {}

    def policy_for(house: str) -> MCTS:
        path = str(mapping.get(house) or fallback)
        evaluator = models.get(path)
        if evaluator is None:
            evaluator = NeuralEvaluator(PolicyValueNet.load(path))
            models[path] = evaluator
        return MCTS(
            evaluator,
            simulations=args.simulations,
            determinizations=args.determinizations,
            temperature=0.0,
            seed=args.seed,
        )

    solution = tool.find_solution(tool.game_config(start), decisions)
    if solution is None:
        raise SystemExit("could not reproduce the match; refusing to audit")

    game = Game.new(tool.game_config(start))
    pending = list(solution)
    stats: dict[str, list[float]] = defaultdict(list)
    disagree: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    findings: list[dict] = []

    index = 0
    while not game.is_terminal() and index < len(decisions):
        tool.settled(game)
        if game.is_terminal():
            break
        record = decisions[index]
        action = tool.action_from_record(record)
        if tool.try_step(game, action) is not None:
            if not tool._record_matches_state(record, game):
                raise SystemExit(f"log and engine disagree at step {record.get('step')}")
            if record.get("actor") == "bot":
                house = record["house"]
                totals[house] += 1
                result = policy_for(house).search(game)
                played = next(
                    (
                        position
                        for position, candidate in enumerate(result.actions)
                        if tool.canonical(candidate) == tool.canonical(action)
                    ),
                    None,
                )
                if played is not None and len(result.q_values):
                    best = int(np.argmax(result.q_values))
                    gap = float(result.q_values[best] - result.q_values[played])
                    stats[house].append(gap)
                    if gap > 0.05:
                        disagree[house] += 1
                    findings.append(
                        {
                            "step": record.get("step"),
                            "round": record.get("round"),
                            "house": house,
                            "gap": gap,
                            "played": str(action),
                            "best": str(result.actions[best]),
                            "changed": best != played,
                        }
                    )
            if not tool.apply_step(game, action):
                raise SystemExit(f"the engine refused step {record.get('step')}")
            index += 1
            continue
        if not pending:
            raise SystemExit(f"no dialog choice left at step {record.get('step')}")
        if not tool.apply_step(game, pending.pop(0)):
            raise SystemExit("a solved dialog choice no longer applies")

    print(
        f"{'casa':12s} {'decisioni':>9s} {'gap medio':>10s} {'gap>0.05':>9s} {'cambia mossa':>12s}"
    )
    for house in sorted(totals):
        gaps = stats[house]
        share = disagree[house] / max(1, totals[house])
        mean_gap = float(np.mean(gaps)) if gaps else 0.0
        changed = sum(
            1
            for finding in findings
            if finding["house"] == house and finding["changed"]
        )
        print(
            f"{house:12s} {totals[house]:9d} {mean_gap:10.3f} "
            f"{share:8.1%} {changed:12d}"
        )

    findings.sort(key=lambda finding: finding["gap"], reverse=True)
    print("\nPeggiori mosse:")
    for finding in findings[: args.top]:
        print(
            f"  round {finding['round']:2d} step {finding['step']:4d} "
            f"{finding['house']:10s} gap {finding['gap']:.3f}  "
            f"giocata: {finding['played']}"
        )
        if finding["changed"]:
            print(f"{'':34s}meglio:  {finding['best']}")
    print(json.dumps({"house_totals": dict(totals)}, indent=2))
    return 0


def tool_parse_map(text: str | None) -> dict[str, str]:
    from agot.play import parse_checkpoint_map

    return dict(parse_checkpoint_map(text) or {})


if __name__ == "__main__":
    raise SystemExit(main())
