"""Turn the downloaded Swords and Ravens archive into round snapshots.

Each ``orders-revealed`` log entry carries the full board (units, orders,
garrisons) plus a game snapshot (round, Influence tracks, Wildling strength,
power, Supply, House cards, victory points). That is one complete, labelled
position per round, and the label is the eventual winner.

The output is a compact gzipped NDJSON of those positions plus a summary of the
archive. It is the dataset the human-game study and the auxiliary value target
are built on.

Usage::

    python tools/analyze_human_games.py
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = ROOT / "checkpoints" / "human-games" / "games.ndjson.gz"
DEFAULT_ROUNDS = ROOT / "checkpoints" / "human-games" / "rounds.ndjson.gz"
DEFAULT_SUMMARY = ROOT / "checkpoints" / "human-games" / "summary.json"


def read_games(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        try:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        except EOFError:
            return


def winner_of(game: dict) -> str | None:
    ended = game.get("childGameState", {}).get("childGameState") or {}
    if ended.get("type") == "game-ended":
        return ended.get("winner")
    for log in reversed(game["childGameState"]["gameLogManager"]["logs"]):
        if log["data"].get("type") == "winner-declared":
            return log["data"].get("winner")
    return None


def snapshots(game: dict) -> list[dict]:
    logs = game["childGameState"]["gameLogManager"]["logs"]
    out = []
    for log in logs:
        data = log.get("data") or {}
        if data.get("type") != "orders-revealed":
            continue
        snap = data.get("gameSnapshot") or {}
        world = data.get("worldState") or []
        regions = []
        for region in world:
            units = [[unit.get("type"), unit.get("house")] for unit in region.get("units", [])]
            order = (region.get("order") or {}).get("type")
            entry = {"id": region.get("id"), "units": units}
            if order:
                entry["order"] = order
            if region.get("garrison"):
                entry["garrison"] = region["garrison"]
            regions.append(entry)
        houses = []
        for house in snap.get("housesOnVictoryTrack", []):
            cards = house.get("houseCards") or []
            houses.append(
                {
                    "id": house.get("id"),
                    "supply": house.get("supply"),
                    "power": house.get("powerTokens"),
                    "land": house.get("landAreaCount"),
                    "vp": house.get("victoryPoints"),
                    "cards": [[card.get("id"), card.get("state")] for card in cards],
                }
            )
        # Older games only logged the board, without the round snapshot.
        ordinal = len(out) + 1
        out.append(
            {
                "round": snap.get("round") or ordinal,
                "full": bool(snap),
                "tracks": {
                    "iron_throne": snap.get("ironThroneTrack"),
                    "fiefdoms": snap.get("fiefdomsTrack"),
                    "kings_court": snap.get("kingsCourtTrack"),
                },
                "wildling": snap.get("wildlingStrength"),
                "houses": houses,
                "regions": regions,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_ROUNDS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    winners: Counter = Counter()
    rounds: Counter = Counter()
    orders_by_round: dict[int, Counter] = {}
    port_ships = 0
    games = 0
    snapshots_total = 0
    full_snapshots = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        for game in read_games(args.archive):
            games += 1
            winner = winner_of(game)
            snap_list = snapshots(game)
            if winner:
                winners[winner] += 1
            rounds[len(snap_list)] += 1
            first = snap_list[0] if snap_list else None
            if first:
                port = next((r for r in first["regions"] if r["id"] == "port-of-lannisport"), None)
                if port and any(unit[1] == "lannister" for unit in port["units"]):
                    port_ships += 1
            for snap in snap_list:
                snapshots_total += 1
                if snap["full"]:
                    full_snapshots += 1
                for region in snap["regions"]:
                    if region.get("order"):
                        orders_by_round.setdefault(snap["round"], Counter())[region["order"]] += 1
                out.write(
                    json.dumps(
                        {"game": game["id"], "winner": winner, **snap},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    summary = {
        "games": games,
        "snapshots": snapshots_total,
        "full_snapshots": full_snapshots,
        "winners": dict(winners.most_common()),
        "snapshots_per_game": dict(sorted(rounds.items())),
        "lannister_port_ship_at_round_1": port_ships,
        "orders_by_round": {
            str(round_number): dict(counter)
            for round_number, counter in sorted(orders_by_round.items())
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    size_mb = args.out.stat().st_size / 1e6
    print(
        f"games {games}, snapshots {snapshots_total} ({full_snapshots} full), "
        f"rounds file {size_mb:.1f} MB"
    )
    total = sum(winners.values())
    for house, wins in winners.most_common():
        print(f"  {house:10s} {wins:4d}  {wins / total * 100:5.1f}%")
    print(f"Lannister port ship at round 1: {port_ships}/{games}")
    print("orders round 1:", dict(orders_by_round.get(1, {})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
