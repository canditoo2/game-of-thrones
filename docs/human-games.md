# Human games (Swords and Ravens)

A second data source next to self-play: the finished six-player base games
logged on [Swords and Ravens](https://swordsandravens.net). The archive is
local; the site and the replays are not redistributed here.

## What was downloaded

The statistics filter is **2nd Edition Base Game (3-6p), 6 players, 6
human-controlled seats, no Tides of Battle, no vassals**: 965 finished games
(the site counts 710 with a known winner; the rest ended without one).

The replay protocol, discovered from the client bundle:

1. `GET /play/<game-id>` embeds a small JSON blob
   `{userId, requestUserId, gameId, authToken}`. The token is per user, not per
   game.
2. Connect to `wss://play.swordsandravens.net`, send
   `{"type":"authenticate","authData":{...}}` compressed with zlib and base64.
3. The first reply, `authenticate-response`, carries the **whole game**:
   metadata, players, settings, the full log (all events), and the final state.

The token is a session credential and is passed to the downloader at runtime
(`--token` or `SAR_TOKEN`); it is never stored in the repository.

`tools/download_human_games.py` drives the protocol. The server rate-limits
bursts (new handshakes start timing out after about a hundred quick
connections), so the defaults are deliberately slow: two workers, a delay
between games, a cooldown on failure, and a stop after a few consecutive
failures. Resuming is safe: the output is one gzip member per line, so a killed
run cannot corrupt the games already written.

## Archive format

`checkpoints/human-games/games.ndjson.gz` is gzipped NDJSON, one game per line.
Each line is the server's `game` object unchanged:

- `gameSettings`: `setupId` (`base-game`), `playerCount`, `pbem`;
- `users` and the first log entry's `user-house-assignments`;
- `childGameState.game`: the final board, Houses, power, Supply, tracks,
  Westeros decks, Wildling deck and strength;
- `childGameState.gameLogManager.logs`: the full event stream (about 950
  entries per game) with types such as `player-action`, `orders-revealed`,
  `player-mustered`, `raid-done`, `march-resolved`, `attack`,
  `combat-house-card-chosen`, `retreat-region-chosen`,
  `consolidate-power-order-resolved`, `clash-of-kings-bidding-done`,
  `westeros-card-executed`;
- `childGameState.childGameState`: the end state, `{type: "game-ended",
  winner}`.

## Round snapshots

`tools/analyze_human_games.py` writes
`checkpoints/human-games/rounds.ndjson.gz`, one line per `orders-revealed`
event: the moment after planning, when every order is face up and before the
action phase starts. Each line carries

- `game`, `winner`, `round`, `full`;
- `tracks`: Iron Throne, Fiefdoms, King's Court;
- `wildling`: the Wildling threat;
- `houses`: supply, power, land areas, victory points and House-card states;
- `regions`: units (`[type, house]`), the placed order, the garrison.

`full` marks the games that also logged `gameSnapshot` (tracks and Houses);
older games carry the board only. With the full archive that is roughly 8,700
positions, about two thirds of them complete.

The Lannister port Ship from the FAQ v2.0 errata is present in **all** games,
which is why the Python `GameConfig` now enables it.

## What the log does and does not give

- **Value labels are reliable**: every position has the eventual winner.
- **The board at each reveal is complete**: units, orders, garrisons, plus
  tracks, power, Supply and House cards when `gameSnapshot` exists.
- **Public actions are logged** with region and unit detail: mustering, raids,
  marches, attacks, casualties, retreats, consolidates, bids, House cards.
- **Order placement is not logged**: orders are secret until the reveal, so the
  archive gives the joint revealed set, not the sequential decisions that
  produced it.
- **Intermediate states are not logged**. Reconstructing the board between two
  events means replaying the events with Swords and Ravens' own rules (their
  engine is open source, GPL-3.0, in TypeScript) or writing a tracker. The
  reveals alone are already a usable value dataset.

## From a snapshot to an engine state

`Game.from_snapshot(line)` (on `agot.engine.rust.RustGame`) builds a read-only
engine position from one exported line. It fills what the state encoder reads:
regions, units, controllers, orders and garrisons, each House's power, Supply
and available cards, the three tracks, the round and the Wildling threat. It
then picks the first real Action-Phase decision (Raid, then March, then
Consolidate) so `legal_actions()` and `encode_state()` work and the value head
sees the same action context it was trained with. It is not a simulator: the
position is for evaluation, not for stepping.

The import is in `rust/crates/agot-core/src/snapshot.rs`, exposed as
`Game.from_snapshot` in the extension and wrapped by the Python facade.

## First benchmark

`tools/benchmark_human_value.py` scores a checkpoint on the full snapshots:
state encoder, real legal actions at the first decision, `value_context`
pooling exactly as in training.

On the first 119 games (640 full snapshots), run 7:

| metric | value |
|---|---|
| winner argmax | **30.3%** (chance 16.7%) |
| castle-count baseline | 28.7% |
| MSE (targets ±1) | 0.767 |
| Brier | 0.192 |
| mean value, eventual winner | −0.445 |
| mean value, eventual loser | −0.728 |

The values are absolute per-House values with five losers per winner, so a
negative mean is expected; the winner–loser gap is the signal. The ranking is
the finding: the network beats chance but barely beats counting castles, and it
is strongly House-biased (Greyjoy winners are recognised 71% of the time,
Lannister 50%, Stark/Tyrell/Martell 0%). By round the accuracy is flat at about
26% in rounds 1–4 and rises to 34–37% from round 5.

That is the case for using the human positions as an auxiliary value target:
the archive is large, the labels are clean, and the current value head does not
transfer to human play.

## Planned use

1. **Value benchmark**: encode the reveal positions and measure how well the
   current network predicts the eventual winner, by round and by House. This
   says whether the network's evaluation transfers to human play.
2. **Auxiliary value target**: add the human positions to training with a small
   weight, so the value head learns from human outcomes without touching the
   policy target.
3. **Behaviour cloning**, if it earns its place: action-phase decisions
   (mustering, raids, marches, retreats) are logged with enough detail to
   reconstruct state-action pairs; order placement is not.
