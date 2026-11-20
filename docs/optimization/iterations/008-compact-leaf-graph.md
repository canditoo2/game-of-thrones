# 008 — compact leaf graph: export only the trunk units that fire

- Track: SPEED
- Status: kept
- Commit: `c5b7bfa` (code + tests), docs in the following commit

## Hypothesis

Leaf inference is 72.5 % of self-play with int8 (run7's net, 16 games: 12.4 s of 17.1 s,
246 us per call, 6.36 leaves and 112.7 candidate rows per call). Most trunk units of a
trained net never fire: over run7's whole replay buffer (61 701 states, 1 004 115 candidate
rows) the max pre-activation is <= 0 for 202/256 units of `state_trunk[0]`, 173 of
`state_trunk[2]`, 173 of `action_trunk[0]` and 165 of `action_trunk[2]`
(`checkpoints/perf-2x/dead_mask_study.py`). None of those units fired on 40 179 real search
leaves (716 817 rows, `leaves.npz`). A ReLU unit that outputs zero contributes nothing
downstream, so a graph without those rows and columns is exact on every input where they
stay silent. The heads are different: replay-dead units of `policy_head`'s and
`value_head`'s hidden layers fire on real leaves (policy KL up to 2.5e-4, value change up to
1.08), so they keep full width.

## Prediction (written before the change)

- Metric: `seconds_selfplay` of one production-shaped iteration (`ab_iter.py`, run7 resumed,
  32 games, int8), and per-call leaf time in situ (`share.py 16`).
- Expected: per call, compact int8 ~1.6x faster than full int8 and compact fp32 ~2.6x faster
  than full fp32 (prototype: 453 / 232 / 170 / 142 us). End to end, with ~70 % of self-play in
  the leaf callback, self-play 1.3–1.5x faster on int8. Calibration + export adds ~2 s per
  iteration outside both timers.
- Falsified if: compact fp32 differs from full fp32 by more than round-off on real leaves, or
  self-play improves by less than 3 % / twice the round spread.

## Change

- `onnx_backend.calibrate_keep(model, states, actions, margin=0)`: trunk-only forward over the
  replay buffer's packed arrays (pre-transposed weights, 8192-row `addmm` chunks, running
  per-unit max). A unit is kept iff its max pre-activation is `> -margin`; every layer keeps at
  least its most active unit. No subsampling: the whole buffer costs 1.15–1.6 s on two threads
  at 61.7k examples (2.2 s measured inside a slow-clock A/B round), inside the ~2 s budget, and
  exactness on leaves matters more than a second.
- `PackedGraph(model, keep)`: the trunks keep only the kept rows (and the previous layer's
  kept columns); `policy_head[0]`'s state and action halves and `value_merge` lose the dropped
  columns; both heads are untouched. `keep=None` is the old forward. `GRAPH_REVISION` is now
  `packed-v2`.
- Cache: `keep` is a key field (sidecar stores the lists; `keep_widths` is provenance) and a
  compact graph lives at `<stem>.<backend>.compact.onnx`, so evaluation's full graph beside the
  same checkpoint never evicts it. A pre-compaction sidecar, another keep-set or the full graph
  are all refused (`StaleGraphError` with `rebuild=False`) or rebuilt.
- `agot train --leaf-compact {auto,off}` (default auto). `_prepare_leaf_graph` calibrates in
  the parent when the buffer holds at least `MIN_CALIBRATION_EXAMPLES` (50 000) examples, builds
  the compact graph there, and returns the keep-sets; they ride `SelfPlayPlan.leaf_keep` ->
  worker options -> `batched_self_play(leaf_keep=)` -> `try_batched_self_play` ->
  `leaf_evaluator(keep=)` -> `LeafBackend.from_model(keep=)`, so the workers get a cache hit
  on the parent's graph. The Rust driver spec is unchanged. The log records `leaf_compact`
  (kept widths, `null` = full), `seconds_leaf_calibration` and `seconds_leaf_graph`.
  `gate`, `evaluate`, `gauntlet`, `play` and the deferred `GateSpec` are untouched and stay on
  the full graph. The buffer is read through its existing arrays (`_states`, `_actions`,
  `_offsets`) by a small `cli._replay_rows` helper; no change to `replay.py`.
- `python -m agot.nn.onnx_backend <ckpt> --leaves <leaves.npz> --replay <agot.replay.npz>`:
  full vs compact, fp32 vs int8 on real captured calls (per-call time, policy KL and value
  |difference| against torch, and compact-vs-full at equal precision). The synthetic
  benchmark now reports value drift for fp32 and int8 too.

Minimum data, and why not a margin. Sample-size study on run7 (keep-sets from a subset,
drift of the masked net on `leaves.npz` against the full net):

| calibration examples | how drawn | dropped units that fire on leaves (activations) | leaf policy KL max | leaf value \|diff\| max |
|---|---|---|---|---|
| 1 000 | random | 201–621 | 2.8e-3 – 0.41 | 0.14 – 0.54 |
| 5 000 | random | 13–14 | 1.2e-5 | 0 |
| 10 000 | random | 0–4 | 0 | 0 |
| 20 000 | random | 0–1 | 0 | 0 |
| 1 000 | most recent | 21 315 | 7.0e-3 | 2.00 |
| 10 000 | most recent | 1 551 | 4.2e-4 | 0.45 |
| 20 000 | most recent | 591 | 2.4e-4 | 0.45 |
| 33 000 | most recent (~1 iteration) | 93 | 5.9e-5 | 0.027 |
| 40 000 | most recent | 1 unit missed | — | — |
| 45 000 / 50 000 / 55 000 | most recent | 0 | 0 | 0 |
| 61 701 | whole buffer | 0 | 0 | 0 |

A contiguous block (what a young buffer is: few games) needs far more examples than a random
draw; coverage of games, not row count, is what matters. 50 000 is the first round number
above the last miss (45k clean, 40k missed one unit); a production buffer (80k cap, a quarter
dropped when full) holds 60–80k from its second iteration on, so in practice only a fresh run's
first iterations and small-buffer runs stay on the full graph.

A margin does not buy that safety cheaply: the missed units peaked at -0.38 to -15 on the
small sets, while the action trunk's dead units sit just below zero, so widths explode first:

| margin | kept (s1, s2, a1, a2) on the whole buffer |
|---|---|
| 0 | 54, 83, 83, 91 |
| 0.01 | 54, 83, 93, 93 |
| 0.05 | 55, 83, 138, 116 |
| 0.1 | 60, 83, 174, 175 |
| 0.25 | 87, 83, 231, 238 |

So the margin stays an option (default 0) and training uses none.

## Gates

- `$env:PYTHONPATH="<worktree>\src"; .venv\Scripts\python -m pytest -q` -> exit 0, 252 tests
  (2 skipped); the summary line is swallowed by the shell, the dot count and exit code are the
  record. `tests/test_onnx_backend.py`: 22 (was 13).
- `.venv\Scripts\python -m ruff check src tests` -> `All checks passed!`
- No Rust change, so the cargo gates are untouched (the copied `_core.pyd` is the main
  checkout's build of `7952ae2`).

New tests: calibration finds exactly the hand-built dead units (also with a 7-row chunk); a
unit that fires on a single calibration row is kept, and the margin acts as specified; no layer
ends up empty; compact fp32 equals the torch `forward_packed` within the existing fp32
tolerances on inputs from the calibration distribution, and the int8 compact graph has real
int8 kernels; keeping every unit reproduces the full graph; no keep-sets is the full graph
(file name, key); another keep-set, a reordered/duplicated one, an out-of-range one and a
pre-compaction sidecar are handled as specified; keep-sets reach `leaf_evaluator` through the
plan and the driver but not the Rust spec; the torch engine ignores them; `_prepare_leaf_graph`
uses the full graph below the minimum and for `torch`, the compact one above it, and a worker
given the keep-sets gets a cache hit.

## Measurement

### Real leaves, one thread (`python -m agot.nn.onnx_backend checkpoints\run7\agot.pt --leaves checkpoints\perf-2x\leaves.npz --replay checkpoints\run7\agot.replay.npz`)

1 581 calls, 9 991 leaves, 176 342 rows; machine in a slow clock state (the prototype measured
full fp32 at 453 us), so read the ratios. Kept units `[54, 83, 83, 91]`.

| engine | us/call min | median | vs full int8 |
|---|---|---|---|
| full int8 (today's training engine) | 490.1 | 582.9 | 1.00x |
| full fp32 | 1018.8 | 1206.0 | 0.48x |
| **compact int8** | 310.6 | 376.3 | **1.58x** |
| **compact fp32** | 386.4 | 521.6 | **1.27x** |

Compact fp32 / full fp32: 2.64x. Compact int8 / full int8: 1.58x. Prototype (faster clock):
453 / 232 / 170 / 142 us -> the same ratios (2.66x, 1.63x).

Drift against torch `forward_packed` on the same leaves:

| engine | policy KL mean | p95 | max | value \|diff\| mean | max |
|---|---|---|---|---|---|
| full fp32 | 4.3e-13 | 1.6e-12 | 1.9e-11 | 1.7e-7 | 1.1e-5 |
| compact fp32 | 4.3e-13 | 1.6e-12 | 1.9e-11 | 2.0e-7 | 1.1e-5 |
| full int8 | 1.05e-3 | 3.99e-3 | 1.35e-2 | **5.93e-2** | **1.06** |
| compact int8 | 1.07e-3 | 4.10e-3 | 1.55e-2 | 6.07e-2 | 1.14 |

Compaction alone (compact vs full at equal precision): fp32 KL exactly 0, value |diff| mean
8.9e-8 / max 2.6e-6; int8 KL mean 1.4e-3 / max 3.1e-2, value mean 1.3e-2 / max 0.40 (int8
re-quantizes narrower matrices, so its rounding moves; that is int8 noise, not compaction
error).

### End to end (`ab_iter.py --arm CA=<baseline 7952ae2> --arm CB=<worktree> --rounds 3 --games 32`, int8)

| round | CA selfplay | CB selfplay | ratio | CA seconds | CB seconds | ratio |
|---|---|---|---|---|---|---|
| 0 | 47.25 | 24.84 | 1.903 | 95.60 | 68.19 | 1.402 |
| 1 | 37.52 | 33.60 | 1.117 | 86.51 | 149.92 | 0.577 |
| 2 | 57.40 | 30.96 | 1.854 | 124.51 | 74.09 | 1.681 |

`seconds_selfplay` median **1.854x** (min 1.117, max 1.903); CA median 47.25 s (37.52–57.40),
CB 30.95 s (24.84–33.60). Round 1 is a load spike on the CB run (its `seconds_train`, which this
change does not touch, went from 37 s to 107 s), which is also why the `seconds` ratio is
unreadable that round; `seconds_train` itself has median ratio 1.17x for the same reason
(noise, not an effect). Calibration + export in CB (round 2 log): `seconds_leaf_calibration`
2.18 s, `seconds_leaf_graph` 3.05 s (calibration + fp32 export + quantize), against ~0.9 s
for today's int8 export: **+~2.2 s per iteration**, paid inside `seconds`, against ~16 s saved
in self-play at 32 games (~2x that at 64). Only `agot.onnx-int8.compact.onnx` was written in
CB's directory: the workers hit the parent's cache. Data: CA 17 594 examples, CB 17 735 (int8
rounding differs on the narrower matrices, so the games differ; see the fp32 check below).

### In situ (`share.py 16 <backend>`, `PYTHONPATH` per arm, interleaved)

| round | engine | selfplay s | in leaf callback s | us/call | rest s |
|---|---|---|---|---|---|
| 0 | baseline int8 | 25.89 | 19.01 (73.4 %) | 376 | 6.88 |
| 0 | baseline fp32 | 45.52 | 37.62 (82.6 %) | 764 | 7.90 |
| 0 | compact int8 | 20.17 | 12.81 (63.5 %) | 253 | 7.37 |
| 0 | compact fp32 | 21.15 | 14.05 (66.4 %) | 285 | 7.10 |
| 1 | baseline int8 | 47.17 | 35.59 | 704 | 11.58 |
| 1 | baseline fp32 | 71.94 | 60.16 | 1222 | 11.78 |
| 1 | compact int8 | 15.08 | 9.43 | 186 | 5.66 |
| 1 | compact fp32 | 14.37 | 9.57 | 194 | 4.80 |
| 2 | compact fp32 | 15.24 | 10.15 (66.6 %) | 206 | 5.10 |
| 2 | compact int8 | 12.69 | 7.80 (61.5 %) | 154 | 4.89 |
| 2 | baseline fp32 | 31.72 | 26.13 (82.4 %) | 531 | 5.59 |
| 2 | baseline int8 | 20.05 | 14.60 (72.8 %) | 289 | 5.45 |

"rest" (self-play outside the leaf callback, the same work in every arm) is the clock probe:
rounds 0 and 2 are flat (6.9–7.9 s, 4.9–5.6 s), round 1 drifted ~2x between the baseline and
compact halves and is not used for ratios. Rounds 0 / 2:

- per call, full int8 / compact int8: 1.49x / 1.88x; full fp32 / compact fp32: 2.68x / 2.58x
- per call, full int8 (today) / compact fp32: 1.32x / 1.40x; compact fp32 / compact int8:
  1.13x / 1.34x slower
- self-play, baseline int8 / compact int8: 1.28x / 1.58x; baseline int8 / compact fp32:
  1.22x / 1.32x

Exactness check from the same runs: compact fp32 made **exactly** the leaf calls (49 246) and
examples (8 670) of baseline fp32, every round; compact int8 differs from baseline int8
(50 564 vs 50 569 calls, 8 162 vs 8 655 examples), as int8 noise predicts.

## Verdict

Kept. Prediction met or beaten: per call 1.5–1.9x on int8 and 2.6–2.7x on fp32, self-play
1.85x median end to end on int8 (predicted 1.3–1.5x; the prototype's leaf share understated
how much of the in-situ time the trunks were). The compaction itself is exact on everything
measured (fp32 games identical, KL 0), for ~2.2 s of calibration + export per iteration.

Engine choice (not changed here; the numbers for the decision): with compaction, fp32 costs
only 1.13–1.34x int8 per call and is already 1.3–1.4x faster than today's full int8, while
int8 moves the leaf value by 0.06 on average and up to ~1.1 (the value lives in [-1, 1]) on
real leaves, a drift the int8 budget (`test_int8_policy_kl_stays_inside_the_budget`, policy KL
on a synthetic net) never measured. Switching `auto` for training to `onnx-fp32` would give up
~15–25 % of self-play speed relative to compact int8 to remove that value noise entirely.

## Follow-ups

- Decide the training engine (above). If fp32 is chosen, the int8 quantize step (~0.5 s per
  iteration) also disappears.
- Add a value-drift bound to the int8 budget test, on a trained-like net.
- Skipped micro-levers, by arithmetic: folding `value_merge` into `value_head[0]` saves one
  256x256 matmul on ~6 leaf rows per call and the one-hot pooling is a 113x6x91 matmul, each
  ~1–2 % of the compact graph's MACs, below the 3 % keep threshold before measurement.
- Calibration uses root states; leaves are deeper states. Zero misses on 40k leaves at the
  50k minimum, but the next run should spot-check `leaf_compact` widths in the log against a
  fresh `--leaves` capture once in a while.
- `cli._replay_rows` reads `ReplayBuffer._states/_actions/_offsets`; if the buffer's layout
  changes, it falls back to the full graph (by design), so a public read-only accessor would
  be cleaner.

## Ledger rows and findings to merge

LEDGER.md:

| id | date | track | change | metric | before | after | delta | verdict | commit |
|---|---|---|---|---|---|---|---|---|---|
| 008 | 2026-11-20 | SPEED | compact leaf graph (trunk units that never fire on the replay buffer are not exported) | `seconds_selfplay`, 1 run7 iteration, 32 games, int8, 3 interleaved rounds | 47.25 s | 30.95 s | 1.85x (1.12–1.90) | kept | `c5b7bfa` |

FINDINGS.md:

- Trained nets here have ~65–80 % dead trunk units; dropping them from the leaf graph is exact
  (fp32 self-play games identical to the full graph) and makes a leaf call 2.6x cheaper in fp32
  and 1.6x in int8. Command: `python -m agot.nn.onnx_backend <ckpt> --leaves <leaves.npz>
  --replay <agot.replay.npz>`.
- Head hidden layers must not be pruned this way: replay-dead head units fire on real leaves.
- Calibration needs coverage of games, not rows: 33k contiguous examples miss units that fire
  on leaves, 45k+ do not; random 10k subsets are nearly clean. A margin is a bad substitute (it
  widens the action trunk long before it catches the misses).
- int8 is not round-off on real leaves: full int8 vs torch value |diff| mean 0.059, max 1.06;
  policy KL mean 1.05e-3, p95 4.0e-3, max 1.35e-2 (run7, 1 581 calls).
