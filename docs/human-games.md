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
older games carry the board only. Across the full archive that is roughly 8,700
reveals, but only the 2023+ games are complete (see below); the first 347
downloaded games yield 640 full snapshots.

The Lannister port Ship from the FAQ v2.0 errata is present in every round-1
snapshot the archive can check, which is why the Python `GameConfig` now
enables it.

## House balance

The 740 downloaded games with a known winner are representative of the site's
own table (which uses 710 games): Greyjoy 23.6%, Stark 23.4%, Baratheon 17.0%,
Martell 13.6%, Lannister 11.6%, Tyrell 10.7%.

Lannister is the interesting one because the port Ship arrives only in the
modern rules. Before 2023 (no Ship, 575 games) Lannister won 11.1%; from 2023
(Ship, 165 games) 13.3%. That is a +2.2 pp difference with a standard error of
about 2.9 pp — not distinguishable from zero, and confounded by the era's
players and meta. The archive can describe the ship's effect; it cannot isolate
it.

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

On the 164 games with full snapshots downloaded so far (1,315 positions),
run 7:

| metric | value |
|---|---|
| winner argmax | **28.3%** (chance 16.7%) |
| castle-count baseline | 29.9% |
| MSE (targets ±1) | 0.792 |
| Brier | 0.198 |
| mean value, eventual winner | −0.478 |
| mean value, eventual loser | −0.719 |

The values are absolute per-House values with five losers per winner, so a
negative mean is expected; the winner–loser gap is the signal. The ranking is
the finding: the network beats chance but does **not** beat counting castles,
and it is strongly House-biased (Greyjoy winners are recognised 69.8% of the
time, Lannister 48.3%, Baratheon 28.8%, Stark/Tyrell/Martell 0%). By round the
accuracy is 21–27% in rounds 1–4 and rises to 29–37% from round 5.

That is the case for using the human positions as an auxiliary value target:
the archive is large, the labels are clean, and the current value head does not
transfer to human play.

## What the archive actually covers

The log format changed over the site's life, and the difference matters:

- **2020–2021**: no `orders-revealed` entries at all. The log is a complete
  event stream, but it never records a full board, so no round snapshots.
- **2022**: a few `orders-revealed` entries, board only (`gameSnapshot` absent).
- **2023 and later**: every reveal carries the board plus the round snapshot
  (tracks, power, Supply, House cards), which is the dataset used here.

So the complete-snapshot set is the 2023+ subset, not all 965 games. In the 740
games downloaded so far that is 164 games with at least one full reveal
(1,315 full snapshots, 162 of them at round 1). The rest of the archive is still
useful for outcomes, House win rates and the action-event stream, but not for
board-state training.

The Lannister port Ship is present in **162 of 162** round-1 full snapshots; in
older games the first logged reveal can already be mid-game, so the port has
sometimes changed hands and cannot answer the question.

## Training integration

`tools/export_human_value.py` writes `checkpoints/human-games/values.npz`: one
row per full snapshot with the encoded state, the concatenated legal-action
rows and the ±1 outcome. `agot train --human-value <npz>` runs one value-only
pass over it after every self-play iteration (`--human-value-lr-scale`, default
0.1). The pass trains only the value head and the shared trunk — there is no
policy target, because the archive does not record the orders during planning —
and it scales the learning rate instead of the loss weight, since AdamW
normalises away a small weight but not a small learning rate.
`--human-value-head-only` freezes the trunk and trains only the value head.

The pass is off by default. Run 8 stays a pure self-play run; the human value
target is an opt-in experiment until a game-level A/B shows it helps.

The A/B (`tools/ab_human_value.py`, 164 games split 131/33 by game) says the
pass calibrates but does not rank better. Twenty epochs lower the held-out MSE
from 0.848 to 0.700 and move winner argmax from 23.5% to 24.8% on 294 snapshots
from 33 games — inside noise. A hundred epochs push the training-game ranking
from 29.7% to 37.7% while the held-out ranking stays at 24.1%: that is
overfitting, not transfer. The head-only variant behaves the same way.

So the pass stays off for run 8. The archive's training value is limited by
what the log exposes: full board snapshots only for 2023+ games and no
order-placement decisions at all. It remains a good benchmark and a House
balance study; the next training idea would have to reconstruct action-phase
states, which means replaying Swords and Ravens' open-source engine.

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
