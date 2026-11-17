"""Download finished Swords and Ravens games as compact NDJSON.

Swords and Ravens serves a replay over a WebSocket: connect to
``wss://play.swordsandravens.net``, send a zlib+base64 ``authenticate`` message
with the per-user page token, and the first reply carries the whole game as
JSON, game log included. This script drives that protocol headlessly, so the
965-game archive costs minutes instead of clicking through 965 pages.

The token is a session credential: pass it with ``--token`` or ``SAR_TOKEN``
and never commit it.

Usage::

    python tools/download_human_games.py --token <token> \
        --user-id <uuid> --request-user-id <uuid>

The output is one gzipped NDJSON line per game, each line its own gzip member,
so an interrupted run never corrupts the games already written. Re-running the
same command skips the IDs already in the file.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import json
import os
import sys
import time
import zlib
from pathlib import Path

try:
    import websockets
except ImportError:  # pragma: no cover - depends on the environment
    sys.exit("pip install websockets to run this tool")

HOST = "wss://play.swordsandravens.net"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = ROOT / "checkpoints" / "human-games" / "index.json"
DEFAULT_OUT = ROOT / "checkpoints" / "human-games" / "games.ndjson.gz"


def encode(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.b64encode(zlib.compress(raw)).decode()


def decode(raw: str) -> dict:
    return json.loads(zlib.decompress(base64.b64decode(raw)))


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        try:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        except EOFError:  # a member cut by a killed run: keep what we have
            pass
    return done


def append_game(path: Path, game: dict) -> None:
    """One gzip member per line: a killed run loses at most the current game."""
    line = (json.dumps(game, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("ab") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(line)


async def fetch(game_id: str, auth: dict, timeout: float) -> dict:
    async with websockets.connect(HOST, max_size=64 * 1024 * 1024, open_timeout=timeout) as ws:
        await ws.send(encode({"type": "authenticate", "authData": {**auth, "gameId": game_id}}))
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        message = decode(raw)
    if message.get("type") != "authenticate-response":
        raise RuntimeError(f"unexpected reply {message.get('type')!r}")
    game = message.get("game")
    if not game or game.get("id") != game_id:
        raise RuntimeError("reply carried no matching game")
    return game


async def worker(
    queue: asyncio.Queue,
    auth: dict,
    args,
    lock: asyncio.Lock,
    stats: dict,
) -> None:
    while not stats["stop"]:
        try:
            game_id = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        game = None
        for attempt in (1, 2):
            try:
                game = await fetch(game_id, auth, args.timeout)
                stats["consecutive"] = 0
                break
            except Exception as error:  # noqa: BLE001 - reported and retried
                stats["consecutive"] += 1
                if stats["consecutive"] >= args.max_failures:
                    stats["stop"] = True
                    stats["error"] = f"stopped after {stats['consecutive']} failures: {error}"
                    return
                await asyncio.sleep(args.cooldown)
                if attempt == 2:
                    stats["failed"].append({"id": game_id, "error": str(error)[:200]})
        if game is not None:
            async with lock:
                append_game(args.out, game)
                stats["done"] += 1
                if stats["done"] % 25 == 0:
                    print(
                        f"{stats['done']}/{stats['total']} games, "
                        f"{len(stats['failed'])} failed",
                        flush=True,
                    )
        if args.delay:
            await asyncio.sleep(args.delay)
        queue.task_done()


async def run(args) -> int:
    games = json.loads(args.index.read_text(encoding="utf-8"))
    done = already_done(args.out)
    todo = [game for game in games if game["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(games)} games in the index, {len(done)} already downloaded, {len(todo)} to go")

    token = args.token or os.environ.get("SAR_TOKEN")
    user = args.user_id or os.environ.get("SAR_USER_ID")
    request_user = args.request_user_id or user
    if not token or not user:
        sys.exit("--token/--user-id (or SAR_TOKEN/SAR_USER_ID) are required")
    auth = {"userId": user, "requestUserId": request_user, "authToken": token}

    queue: asyncio.Queue = asyncio.Queue()
    for game in todo:
        queue.put_nowait(game["id"])
    stats = {
        "total": len(todo),
        "done": 0,
        "failed": [],
        "consecutive": 0,
        "stop": False,
        "error": None,
    }
    lock = asyncio.Lock()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    workers = [
        asyncio.create_task(worker(queue, auth, args, lock, stats))
        for _ in range(args.workers)
    ]
    await asyncio.gather(*workers)

    failures = args.out.parent / "failures.json"
    failures.write_text(json.dumps(stats["failed"], indent=2), encoding="utf-8")
    print(f"downloaded {stats['done']}, failed {len(stats['failed'])}")
    if stats["error"]:
        print(stats["error"], file=sys.stderr)
    return 0 if not stats["failed"] and not stats["stop"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--token")
    parser.add_argument("--user-id")
    parser.add_argument("--request-user-id")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.75)
    parser.add_argument("--cooldown", type=float, default=20.0)
    parser.add_argument("--max-failures", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    started = time.monotonic()
    code = asyncio.run(run(args))
    print(f"{time.monotonic() - started:.1f}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
