"""Turn a logged human-vs-bot match into ReplayBuffer examples for the human seat.

The GUI's own ``.npz`` holds only the bot (MCTS) decisions on purpose, so a human
game never teaches the net anything about the house the human played. This tool
closes that gap for one house: it replays the recorded actions on a fresh engine
and, at every decision of that house, runs MCTS with a checkpoint, storing the
visit distribution as the policy target and the real final outcome as the value
target. The result is a normal ``ReplayBuffer`` archive.

Replay is exact because the engine is deterministic given the config seed: the
records carry every action, and the start record carries seed and rules. Two
decisions are not written as normal records because the GUI resolves them with
dialogs, so the replay searches them: Doran Martell's Influence track and the
Iron Throne tie-breaks. An ``undo`` record rewinds to just before the last human
decision, dropping whatever the rollback discarded.

Usage::

    python tools/reanalyze_human_game.py checkpoints/play-logs/play-X.jsonl \
        --checkpoint checkpoints/run5/agot.pt \
        --house martell --output checkpoints/play-logs/play-X.martell.npz \
        --merge checkpoints/play-logs/play-X.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from agot.agents import MCTS, outcome_vector
from agot.engine.actions import Action, ActionKind
from agot.engine.enums import Phase
from agot.engine.game import Game, GameConfig
from agot.engine.observation import HOUSE_ORDER
from agot.training.replay import Example, ReplayBuffer

#: Upper bound on dialog decisions a single replay may add (Doran, tie-breaks).
MAX_DIALOG_CHOICES = 6
#: Upper bound on full-replay attempts per log, so a bad log fails fast.
MAX_BRANCHES = 500


class ReplayMismatch(RuntimeError):
    """The recorded log cannot be reproduced on this engine build."""


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def active_decisions(records: list[dict]) -> list[dict]:
    """The decisions that survived the logged undos.

    An undo restores the snapshot taken before the last human decision, so that
    decision and everything after it never happened and must be dropped.
    """
    active: list[dict] = []
    for record in records:
        kind = record.get("type")
        if kind == "decision":
            active.append(record)
        elif kind == "undo":
            last_human = max(
                (index for index, item in enumerate(active) if item.get("actor") == "human"),
                default=-1,
            )
            if last_human >= 0:
                del active[last_human:]
    return active


def game_config(start: dict) -> GameConfig:
    """The GUI's rules, with the seed and round cap from the start record."""
    return GameConfig(
        seed=start.get("seed", 0),
        max_rounds=start.get("max_rounds", 10),
        deferred_battles=True,
        sequential_westeros=True,
        doran_choice=True,
        wildling_tie_choice=True,
        choice_choices=True,
    )


def action_from_record(record: dict) -> Action:
    payload = dict(record["action"])
    payload["kind"] = ActionKind(payload["kind"])
    if payload.get("units") is not None:
        payload["units"] = tuple(payload["units"])
    return Action(**payload)


def settled(game: Game) -> None:
    """Advance the sequential Westeros reveal the GUI advances on a timer."""
    while (
        game.state.phase is Phase.WESTEROS
        and game.state.westeros_waiting
        and not game.is_terminal()
    ):
        game.continue_westeros()


def _record_matches_state(record: dict, game: Game) -> bool:
    """The record's phase and action step must describe this decision point."""
    phase = record.get("phase")
    if phase is not None and phase != game.state.phase.value:
        return False
    action_step = record.get("action_step")
    if action_step is not None and action_step != game.state.action_step.value:
        return False
    return True


def try_step(game: Game, action: Action) -> Game | None:
    """A cloned game with ``action`` applied, or None when the engine refuses it."""
    branch = game.clone()
    try:
        branch.step(action)
    except (RuntimeError, ValueError):
        return None
    return branch


def apply_step(game: Game, action: Action) -> bool:
    """Apply ``action`` in place, keeping the engine's RNG stream intact.

    ``Game.clone`` derives the child's generator from a copy of the parent's
    stream, so a replay that cloned at every step would draw different Westeros
    cards. The main line must therefore mutate the original game; clones are
    only for exploring dialog branches.
    """
    try:
        game.step(action)
    except (RuntimeError, ValueError):
        return False
    return True


def attempt(
    config: GameConfig, decisions: list[dict], choices: list[Action]
) -> tuple[str, list[Action]]:
    """Replay in place with a fixed dialog prefix.

    Returns ``("ok", used)`` when the whole log reproduces, ``("need", legal)``
    at the first dialog the prefix does not cover, and ``("fail", [])`` when a
    record no longer applies. Replaying from scratch on every branch is what
    keeps the RNG exact: a cloned branch forks its generator and would draw
    different Westeros cards.
    """
    game = Game.new(config)
    used: list[Action] = []
    index = 0
    while True:
        settled(game)
        if game.is_terminal():
            return ("ok", used) if index == len(decisions) else ("fail", [])
        if index >= len(decisions):
            return ("fail", [])
        record = decisions[index]
        action = action_from_record(record)
        if try_step(game, action) is not None:
            current = game.current_player()
            if current is not None and current != record.get("house"):
                return ("fail", [])
            if not _record_matches_state(record, game):
                return ("fail", [])
            if not apply_step(game, action):
                return ("fail", [])
            index += 1
            continue

        # The log has no record here: the GUI asked a dialog instead.
        legal = game.legal_actions()
        if len(used) >= len(choices):
            return ("need", legal)
        if not apply_step(game, choices[len(used)]):
            return ("fail", [])
        used.append(choices[len(used)])


def search(
    config: GameConfig,
    decisions: list[dict],
    prefix: list[Action],
    limit: int,
    budget: list[int],
) -> list[Action] | None:
    budget[0] -= 1
    if budget[0] < 0:
        return None
    status, payload = attempt(config, decisions, prefix)
    if status == "ok":
        return payload
    if status == "need" and len(prefix) < limit:
        for candidate in payload:
            found = search(config, decisions, [*prefix, candidate], limit, budget)
            if found is not None:
                return found
    return None


def find_solution(config: GameConfig, decisions: list[dict]) -> list[Action] | None:
    """Iterative deepening on the number of dialog decisions.

    One budget covers every pass, so an unreproducible log fails in bounded
    time instead of exploring a fresh exponential tree per depth.
    """
    budget = [MAX_BRANCHES]
    for limit in range(0, MAX_DIALOG_CHOICES + 1):
        found = search(config, decisions, [], limit, budget)
        if found is not None:
            return found
        if budget[0] < 0:
            return None
    return None


def replay_with_choices(
    game: Game, decisions: list[dict], choices: list[Action]
) -> tuple[list[tuple[dict, Game]], Game]:
    """Walk the solved line in place, capturing each decision's pre-state.

    In-place is load-bearing: the engine's clone forks its RNG, so only the
    original stream reproduces the logged Westeros draws. Clones are taken just
    to capture the state each example is encoded from.
    """
    pending = list(choices)
    captured: list[tuple[dict, Game]] = []
    index = 0
    while not game.is_terminal() and index < len(decisions):
        settled(game)
        if game.is_terminal():
            break
        record = decisions[index]
        action = action_from_record(record)
        if try_step(game, action) is not None:
            if not _record_matches_state(record, game):
                raise ReplayMismatch(
                    f"log and engine disagree at step {record.get('step')}: "
                    f"{record.get('phase')}/{record.get('action_step')} vs "
                    f"{game.state.phase.value}/{game.state.action_step.value}"
                )
            captured.append((record, game.clone()))
            if not apply_step(game, action):
                raise ReplayMismatch(f"the engine refused step {record.get('step')}")
            index += 1
            continue
        if not pending:
            raise ReplayMismatch(f"no dialog choice left at step {record.get('step')}")
        if not apply_step(game, pending.pop(0)):
            raise ReplayMismatch(
                f"a solved dialog choice no longer applies at step {record.get('step')}"
            )
    settled(game)
    if pending:
        raise ReplayMismatch("the solved line left dialog choices unused")
    return captured, game


def canonical(action: Action) -> tuple:
    """Action identity without the engine handle, for aligning visit vectors."""
    units = getattr(action, "units", None)
    return (
        getattr(action.kind, "value", action.kind),
        action.house,
        action.region,
        action.target,
        action.order,
        action.card,
        getattr(action, "card_house", None),
        getattr(action, "choice_tag", None),
        action.unit,
        action.side,
        action.amount,
        None if units is None else tuple(units),
        action.establish,
    )


def build_examples(
    captured: list[tuple[dict, Game]],
    *,
    house: str,
    policy: MCTS,
    outcome: np.ndarray,
) -> list[Example]:
    examples: list[Example] = []
    for record, game in captured:
        if record.get("house") != house:
            continue
        observation = game.observe()
        result = policy.search(game)
        visits = np.asarray(result.visits, dtype=np.float64)
        total = float(visits.sum())
        if total <= 0 or not len(result.actions):
            continue
        visits /= total
        by_key = {
            canonical(action): float(visits[position])
            for position, action in enumerate(result.actions)
        }
        target = np.array(
            [by_key.get(canonical(action), 0.0) for action in observation.actions],
            dtype=np.float32,
        )
        if target.sum() <= 0:
            continue
        examples.append(
            Example(
                state=observation.state_features,
                actions=observation.action_features,
                policy=target,
                player=HOUSE_ORDER.index(house),
                outcome=outcome.astype(np.float32),
            )
        )
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="play-logs/*.jsonl")
    parser.add_argument("--checkpoint", required=True, help="network used for the reanalysis")
    parser.add_argument("--house", default="martell", help="house whose decisions to keep")
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--determinizations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "write the new examples this many times (oversampling, since the "
            "trainer samples uniformly); 1 keeps them as a plain record"
        ),
    )
    parser.add_argument("--merge", type=Path, default=None, help="extra NPZ to merge in")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.setrecursionlimit(20_000)

    records = load_records(args.log)
    start = next(record for record in records if record.get("type") == "start")
    end = next(record for record in records if record.get("type") == "end")
    if end.get("aborted"):
        raise SystemExit("the match was aborted; there is no clean outcome to learn from")

    decisions = active_decisions(records)
    solution = find_solution(game_config(start), decisions)
    if solution is None:
        raise SystemExit("could not reproduce the match on this engine build; no data written")

    replay = Game.new(game_config(start))
    captured, replay = replay_with_choices(replay, decisions, solution)
    settled(replay)
    if not replay.is_terminal():
        raise SystemExit("the reproduced line did not reach a terminal state")
    winner = replay.winner()
    if winner != end.get("winner"):
        raise SystemExit(
            f"winner mismatch: reproduced {winner!r}, log says {end.get('winner')!r}"
        )
    if replay.state.round != end.get("rounds"):
        raise SystemExit(
            f"round mismatch: reproduced {replay.state.round}, log says {end.get('rounds')}"
        )

    from agot.nn.model import NeuralEvaluator, PolicyValueNet

    evaluator = NeuralEvaluator(PolicyValueNet.load(args.checkpoint))
    policy = MCTS(
        evaluator,
        simulations=args.simulations,
        determinizations=args.determinizations,
        temperature=0.0,
        seed=args.seed,
    )
    outcome = outcome_vector(replay)
    examples = build_examples(captured, house=args.house, policy=policy, outcome=outcome)
    if not examples:
        raise SystemExit(f"no {args.house} decisions were found in the log")
    repeat = max(1, args.repeat)
    examples = examples * repeat

    merged = ReplayBuffer.load(str(args.merge)).all() if args.merge else []
    buffer = ReplayBuffer(capacity=max(1, len(merged) + len(examples)))
    buffer.extend(merged)
    buffer.extend(examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    buffer.save(str(args.output))

    print(
        json.dumps(
            {
                "log": str(args.log),
                "house": args.house,
                "winner": winner,
                "outcome": outcome.tolist(),
                "decisions_in_log": len(decisions),
                "dialog_choices_recovered": len(solution),
                "examples_new": len(examples),
                "examples_repeat": repeat,
                "examples_merged": len(merged),
                "examples_total": len(buffer),
                "output": str(args.output),
                "simulations": args.simulations,
                "determinizations": args.determinizations,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
