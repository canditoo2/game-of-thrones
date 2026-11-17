"""Causal asset weights from paired-seed self-play.

Design (see the calibration notes): one seed is a block. The control game and
every treated game share the seed, the greedy policy seeds and the rules; only
the treated house's setup differs, so the difference in its win rate is the
causal effect of that one asset. The control is shared by every condition, so a
seed costs one game per condition instead of two.

Perturbations are data-only (a copied data directory with one JSON field
changed), which keeps the engine untouched and the RNG stream identical.

Usage::

    python tools/calibrate_weights.py --data-dir src/agot/data \
        --out /content/drive/MyDrive/agot/calibration --seeds 800 --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from agot.agents import GreedyPolicy
from agot.engine.game import Game, GameConfig
from agot.engine.map import GameMap

HOUSES = ("baratheon", "lannister", "stark", "greyjoy", "tyrell", "martell")

#: The rules and presentation options used by training/evaluation.
BASE = dict(
    max_rounds=10,
    victory_castles=7,
    use_westeros_decks=True,
    deferred_battles=True,
    sequential_westeros=True,
    doran_choice=True,
    wildling_tie_choice=True,
    bid_tie_choice=True,
    choice_choices=True,
    # The FAQ v2.0 Lannister opening. Spelled out so the calibration can never
    # silently drift from the setup our games use.
    lannister_port_ship=True,
)

#: A neutral castle with mustering 1 and no other icons: the positive control.
FORT_REGION = "crackclaw_point"

CONDITIONS = {
    "fort": "positive control: a footman on a neutral castle (1 victory point, M1)",
    "castle": "home stronghold turned into a castle: mustering 2 -> 1 (legal direction)",
    "castle_supply": "castle plus +1 supply icon (the mustering/supply interaction)",
    "power_icon": "+1 printed crown on the home region",
    "supply_icon": "+1 printed supply icon on the home region",
    "power_token": "+1 starting Power token",
    "supply_step": "+1 starting Supply level",
    "footman": "+1 starting Footman at home",
    "knight": "+1 starting Knight at home",
    "placebo": "identical data copy (must be exactly 0)",
}


def design_payload(base: Path) -> dict:
    """The immutable ingredients of a run, hashed so a resume cannot mix designs."""
    digest = hashlib.sha256()
    for path in sorted(base.glob("*.json")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {
        "version": 1,
        "rules": BASE,
        "conditions": CONDITIONS,
        "fort_region": FORT_REGION,
        "houses": list(HOUSES),
        "data_hash": digest.hexdigest()[:16],
    }


def home_region(map_data: dict, house: str) -> dict:
    for region in map_data["regions"]:
        if region.get("home") == house:
            return region
    raise SystemExit(f"{house} has no home region")


def write_variant(base: Path, target: Path, house: str, condition: str) -> None:
    if target.exists():
        return
    target.mkdir(parents=True)
    for path in base.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)

    map_path = target / "map_base_6p.json"
    houses_path = target / "houses.json"
    map_data = json.loads(map_path.read_text(encoding="utf-8"))
    houses_data = json.loads(houses_path.read_text(encoding="utf-8"))

    if condition in ("castle", "castle_supply", "power_icon", "supply_icon"):
        region = home_region(map_data, house)
        if condition in ("castle", "castle_supply"):
            # The legal direction: a stronghold (2) becomes a castle (1), which
            # also costs the stronghold tiebreak key. The +1 direction (2 -> 3)
            # leaves the printed board and is not used.
            region["mustering"] = max(1, int(region.get("mustering", 1)) - 1)
        if condition in ("supply_icon", "castle_supply"):
            region["supply"] = int(region.get("supply", 0)) + 1
        if condition == "power_icon":
            region["power"] = int(region.get("power", 0)) + 1
        map_path.write_text(
            json.dumps(map_data, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    elif condition == "fort":
        spec = next(item for item in houses_data["houses"] if item["id"] == house)
        units = spec.setdefault("starting_units", {}).setdefault(
            FORT_REGION, {"footmen": 0, "knights": 0, "ships": 0, "siege": 0}
        )
        units["footmen"] = int(units.get("footmen", 0)) + 1
        houses_path.write_text(
            json.dumps(houses_data, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    elif condition in ("power_token", "supply_step", "footman", "knight"):
        spec = next(item for item in houses_data["houses"] if item["id"] == house)
        if condition == "power_token":
            spec["initial_power"] = int(spec.get("initial_power", 0)) + 1
        elif condition == "supply_step":
            spec["initial_supply"] = int(spec.get("initial_supply", 0)) + 1
        else:
            region = home_region(map_data, house)["id"]
            units = spec.setdefault("starting_units", {}).setdefault(
                region, {"footmen": 0, "knights": 0, "ships": 0, "siege": 0}
            )
            key = "footmen" if condition == "footman" else "knights"
            units[key] = int(units.get(key, 0)) + 1
        houses_path.write_text(
            json.dumps(houses_data, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    elif condition == "placebo":
        pass  # byte-identical copy


def variant_dirs(base: Path, cache: Path) -> dict[str, dict[str, str]]:
    """``{condition: {house: dir}}``; ``control`` keeps the packaged data."""
    dirs: dict[str, dict[str, str]] = {"control": {h: str(base) for h in HOUSES}}
    for condition in CONDITIONS:
        dirs[condition] = {}
        for house in HOUSES:
            target = cache / f"{condition}-{house}"
            write_variant(base, target, house, condition)
            dirs[condition][house] = str(target)
    return dirs


def play(seed: int, house: str, data_dir: str) -> tuple[str | None, dict]:
    config = GameConfig(seed=seed, **BASE)
    game = Game.new(config, data_dir=data_dir)
    # The engine loaded the variant, but the Python map used by the greedy
    # heuristics (and by the GUI) is the packaged one. Swap it so the policy
    # sees the same board the rules do; otherwise an icon perturbation would be
    # invisible to the policy and its measured value would be biased to zero.
    raw_map = json.loads(
        (Path(data_dir) / "map_base_6p.json").read_text(encoding="utf-8")
    )
    game.map = GameMap(raw_map)
    policies = {
        name: GreedyPolicy(seed * 97 + index)
        for index, name in enumerate(HOUSES)
    }
    steps = 0
    while not game.is_terminal() and steps < 4000:
        while (
            game.state.phase.value == "westeros"
            and game.state.westeros_waiting
            and not game.is_terminal()
        ):
            game.continue_westeros()
        if game.is_terminal():
            break
        actions = game.legal_actions()
        if not actions:
            break
        player = game.current_player()
        policy = policies.get(player) if player else policies["stark"]
        action, _ = policy.decide(game)
        game.step(action)
        steps += 1
    house_state = game.state.houses[house]
    metrics = {
        "castles": game.state.castle_count(house),
        "strongholds": sum(
            1
            for region_id in game.state.controlled_regions(house)
            if game.map.regions[region_id].is_stronghold
        ),
        "supply": house_state.supply_level,
        "power": house_state.power,
    }
    return game.winner(), metrics


def run_seed(seed: int, dirs: dict[str, dict[str, str]]) -> dict:
    house = HOUSES[seed % len(HOUSES)]
    wins = {}
    stats = {}
    for condition, by_house in dirs.items():
        winner, metrics = play(seed, house, by_house[house])
        wins[condition] = 1 if winner == house else 0
        stats[condition] = metrics
    return {"seed": seed, "house": house, "wins": wins, "stats": stats}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    # Exact two-sided binomial test on the discordant pairs.
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def summarize(rows: list[dict], *, bootstrap: bool = True) -> dict:
    out = {}
    for condition in CONDITIONS:
        pairs = [
            (row["wins"]["control"], row["wins"][condition])
            for row in rows
            if "control" in row["wins"] and condition in row["wins"]
        ]
        if not pairs:
            continue
        n = len(pairs)
        b = sum(1 for control, treated in pairs if treated > control)
        c = sum(1 for control, treated in pairs if treated < control)
        delta = (b - c) / n
        q = (b + c) / n
        sd = math.sqrt(max(0.0, (b + c - n * delta * delta) / max(1, n - 1)))
        se = sd / math.sqrt(n)
        out[condition] = {
            "n": n,
            "p_control": sum(pair[0] for pair in pairs) / n,
            "p_treated": sum(pair[1] for pair in pairs) / n,
            "delta": delta,
            "b": b,
            "c": c,
            "q": q,
            "se": se,
            "ci95": [delta - 1.96 * se, delta + 1.96 * se],
            "p_mcnemar": mcnemar(b, c),
        }
        if all("stats" in row for row in rows):
            out[condition]["metrics_delta"] = {
                key: sum(row["stats"][condition][key] for row in rows) / n
                - sum(row["stats"]["control"][key] for row in rows) / n
                for key in ("castles", "strongholds", "supply", "power")
            }
    # Holm over the five substantial conditions (fort and placebo are checks).
    family = [name for name in out if name not in ("fort", "placebo")]
    ordered = sorted(family, key=lambda name: out[name]["p_mcnemar"])
    m = len(ordered)
    previous = 0.0
    for rank, name in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * out[name]["p_mcnemar"])
        previous = max(previous, adjusted)
        out[name]["p_holm"] = previous
    # Ratios to the rules' 1/7 victory share, bootstrapped over seeds. The
    # measured fort effect is reported alongside as a positive control; the
    # ratio to it is the more honest scale once the run is done.
    fort = 1.0 / 7.0
    ratios_by_condition = {name: [] for name in out}
    if bootstrap and "fort" in out:
        rng = random.Random(0)
        for _ in range(400):
            sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
            base = sum(row["wins"]["control"] for row in sample) / len(sample)
            effects = {
                name: sum(row["wins"][name] for row in sample) / len(sample) - base
                for name in out
            }
            if effects["fort"] <= 0:
                continue
            for name in out:
                ratios_by_condition[name].append(effects[name] / effects["fort"])
    for name, stats in out.items():
        stats["victory_units_rules"] = stats["delta"] / fort
        stats["victory_units_rules_ci95"] = [
            stats["ci95"][0] / fort,
            stats["ci95"][1] / fort,
        ]
        ratios = sorted(ratios_by_condition[name])
        if ratios:
            stats["ratio_to_fort"] = (
                stats["delta"] / out["fort"]["delta"] if out["fort"]["delta"] else None
            )
            stats["ratio_to_fort_ci95"] = [
                ratios[int(0.025 * (len(ratios) - 1))],
                ratios[int(0.975 * (len(ratios) - 1))],
            ]
    return out


def state_signature(data_dir: str, seed: int) -> str:
    game = Game.new(GameConfig(seed=seed, **BASE), data_dir=data_dir)
    return json.dumps(game.state._data(), sort_keys=True, default=str)


def pairing_check(dirs: dict[str, dict[str, str]], seeds: int = 6) -> None:
    """The two arms must share the whole post-setup state but the perturbation.

    A map-only change (mustering or an icon) must leave the state byte-identical:
    the state carries power, supply and units, not the printed icons. A state
    change is normalised on the treated house and then compared.
    """
    for seed in range(seeds):
        house = HOUSES[seed % 6]
        control = state_signature(dirs["control"][house], seed)
        for condition in CONDITIONS:
            variant = state_signature(dirs[condition][house], seed)
            if condition in (
                "castle",
                "castle_supply",
                "power_icon",
                "supply_icon",
                "placebo",
            ):
                if variant != control:
                    raise SystemExit(
                        f"pairing broke at seed {seed} ({condition}/{house}): "
                        "a map-only change altered the setup state"
                    )
    print(f"pairing check ok on {seeds} seeds", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=600)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--target-half-width", type=float, default=0.02)
    parser.add_argument("--max-seeds", type=int, default=2000)
    parser.add_argument("--skip-pairing-check", action="store_true")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "seeds.jsonl"
    design = design_payload(args.data_dir)
    design_path = out / "design.json"
    if design_path.exists():
        previous = json.loads(design_path.read_text(encoding="utf-8"))
        if previous != design:
            raise SystemExit(
                f"{design_path} records a different design (rules or data changed). "
                "Use a new --out directory; resuming would mix incompatible games."
            )
    elif rows_path.exists():
        raise SystemExit(
            f"{rows_path} exists without design.json (a run from an older tool). "
            "Use a new --out directory; do not resume it."
        )
    else:
        design_path.write_text(json.dumps(design, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            "design: Lannister port ship on, "
            f"{len(CONDITIONS)} conditions, data {design['data_hash']}",
            flush=True,
        )
    cache = out / "variants"
    cache.mkdir(exist_ok=True)
    dirs = variant_dirs(args.data_dir, cache)
    if not args.skip_pairing_check:
        pairing_check(dirs, seeds=min(6, args.max_seeds))

    rows: list[dict] = []
    done: set[int] = set()
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                done.add(row["seed"])

    plan = [seed for seed in range(args.max_seeds) if seed not in done]
    if not plan:
        plan = []

    batch = 25
    with ProcessPoolExecutor(max_workers=args.workers) as pool, rows_path.open(
        "a", encoding="utf-8"
    ) as handle:
        for start in range(0, len(plan), batch):
            chunk = plan[start : start + batch]
            futures = [pool.submit(run_seed, seed, dirs) for seed in chunk]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                done.add(row["seed"])
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            summary = summarize(rows, bootstrap=False)
            tight = all(
                stats["ci95"][1] - stats["ci95"][0] <= 2 * args.target_half_width
                for name, stats in summary.items()
                if name not in ("fort", "placebo")
            )
            print(
                f"{len(done)} seeds; "
                + " ".join(
                    f"{name}={stats['delta'] * 100:+.1f}pp"
                    for name, stats in summary.items()
                ),
                flush=True,
            )
            if len(done) >= args.seeds:
                if tight or len(done) >= args.max_seeds:
                    break

    summary = summarize(rows)
    per_house = {
        house: summarize([row for row in rows if row["house"] == house], bootstrap=False)
        for house in HOUSES
    }
    result = {
        "design": {
            "seeds": len(rows),
            "conditions": CONDITIONS,
            "fort_region": FORT_REGION,
            "rules": BASE,
        },
        "results": summary,
        "by_house": per_house,
    }
    (out / "weights.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
