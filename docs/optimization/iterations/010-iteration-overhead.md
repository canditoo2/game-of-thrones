# 010 — the time outside both phase timers: background snapshot, calibration over distinct rows

- Track: SPEED
- Status: kept (timers, background snapshot, distinct-row calibration); rejected (block
  `extend`, cheaper int8 quantize, unique→argsort, all-core calibration)
- Commits: `6831223` (timers), `92d9d87` (background snapshot), `2ad2317` (distinct-row
  calibration), docs in the following commit
- Branch: `optimization/train-2x`, on top of `e1c9c71` (iterations 006–008)

## Hypothesis

After 006–008, one production-shaped iteration on this machine was 25.16 s, of which
`seconds - seconds_selfplay - seconds_train` was 4.33 s (17 %), up from 2.9 s at `7952ae2`.
In-process timing of each step on run7's buffer after one extend (79 201 examples, 1 291 417
candidate rows, two torch threads, `prof_overhead.py`):

| step | seconds |
|---|---|
| `ReplayBuffer.save` (`np.savez_compressed`, single-threaded zlib, 10.9 MB) | 1.87–1.95 |
| `onnx_backend.calibrate_keep` (whole buffer) | 1.32–1.37 |
| `ensure_graph` miss (export + int8 quantize) | 0.33–0.58 |
| `buffer.extend` of 17.5k examples | 0.22 |
| `weights_digest`, `checkpoint_digest`, `graph_key`, cache hit | 0.001–0.02 |
| `model.save`, `run.save`, `trainer.state_dict`, log append, `target_entropy`, `model.to` | < 0.006 each |

With the new timers in the log, the second iteration of a two-iteration process was fully
attributed: 4.12 s = leaf graph 1.97 (calibration 1.49) + snapshot 1.93 + extend 0.21 +
checkpoint 0.01.

Two facts made the two big items reducible without changing any result:

- The snapshot writes arrays nobody reads until the next resume, and the next thing that
  changes them is the next iteration's `extend`, ~12 s later (calibration + self-play).
- The replay buffer repeats candidate rows heavily: 53 964 distinct of 1 004 115 rows (5.4 %)
  in run7's buffer, 61 498 of 1 246 239 after two more iterations (`distinct.py`). A unit's
  maximum over a repeated row is its maximum over one copy.

## Prediction (written before the change)

- Metric: `seconds - seconds_selfplay - seconds_train` of the **second** iteration of a
  two-iteration process (`ab_iters.py`, a copy of `ab_multi.py` that runs `--iterations 502`
  from run7 and reports the last, warm iteration; one iteration per process would let the
  background snapshot fall entirely outside `seconds`, joined at exit, and overstate it), plus
  the per-step timers.
- Expected: snapshot −1.9 s (fully hidden behind calibration + self-play on this 22-core box);
  calibration −0.8 to −1.0 s if distinct rows can be found and verified in ~0.3 s. Together
  4.3 s → ~1.5 s, which would take the whole branch past 2.0x on `seconds`.
- Falsified if: any of `examples_new`, `target_entropy`, `policy_loss`, `value_loss`,
  `leaf_compact` or the snapshot's arrays differ from the `before` arm; or `seconds_selfplay`
  gets worse by more than the snapshot saves.

## Change

1. **Timers** (`cli.py`). The record gains `seconds_extend`, `seconds_trainer_restore`,
   `seconds_checkpoint` (model + run state), `seconds_replay_save` (main-thread time at the
   snapshot point), and with lead 2, `seconds_replay_wait` (time the next iteration blocked on
   the background snapshot) and `seconds_replay_snapshot` (that snapshot's own duration,
   measured in its thread).
2. **Background, atomic replay snapshot** (`replay.py`, `cli.py`).
   `ReplayBuffer.save_async(path)` builds the same array dict `save` writes (one
   `_snapshot_arrays`, views of the storage plus the same `astype` copies) and compresses it
   in a non-daemon thread. Every mutation (`add`, `extend`, `clear`, `_adopt`) and every new
   snapshot first calls `wait_save()`, which joins and re-raises the thread's error once, so
   the thread always reads exactly the arrays a synchronous save would have written at that
   point, and a failure can't be lost by mutating past it. Both `save` and `save_async` now
   write to `<name>.tmp-<pid>` beside the target and `os.replace` it, so an interrupted write
   leaves the previous snapshot loadable (before this, `np.savez_compressed` wrote straight to
   the final name). The `.npz` suffix rule of `np.savez_compressed(name)` is kept. Train loop:
   `save_async` where `save` was (same `--buffer-save-every` condition); after self-play and
   before `extend`, `_finish_replay_snapshot` waits and *reports* a failure ("replay buffer not
   saved: …", as before) instead of stopping the run; the loop's `finally` does the same, so a
   run never ends, normally or on an exception, with the last snapshot half-written. zlib
   level unchanged (default 6), so the file does not grow. zlib and file I/O release the GIL;
   with `--workers 2` the parent only waits on futures during self-play.
3. **Calibration over distinct rows** (`onnx_backend.py`). `distinct_rows(rows)`:
   `_row_keys` projects every row onto two seeded random float32 columns and bit-casts the pair
   to one uint64 key; `np.unique` groups by key; every row is then compared **bit for bit**
   (as int32) with its group's first row (`torch.ne` over `index_select`, 64k-row chunks) and
   any row that differs is kept too. Every input row is therefore either returned or bitwise
   equal to a returned row: a key collision can only add rows, never drop one. Equal rows
   nearly always share a key, but not always (the matmul rounded one repeated row differently
   at another batch position in the synthetic test), which only scores a repeat twice.
   `calibrate_keep(..., dedupe=True)` (the default) runs the unchanged trunk forward over
   `distinct_rows(states)` and `distinct_rows(actions)`. `MIN_CALIBRATION_EXAMPLES`, the
   margin rule, the keep-set format and the graph are untouched.

Tests (`tests/test_replay_snapshot.py`, new, 14; `tests/test_onnx_backend.py`, +4 functions /
+5 cases):

- background snapshot arrays are byte-identical (dtype, shape, bytes) to a synchronous save,
  with and without eviction, and for an empty buffer; a resume restores identical storage;
- `extend`, `add` and `clear` started during a held background write block until it finishes,
  and the file holds the pre-mutation arrays (`add` and `extend` would also evict in place);
- a new snapshot waits for the previous one;
- a failing write surfaces once from `wait_save`, and from a mutation that did not wait
  (before anything changes); an interrupted write (partial bytes, then `OSError`), in either
  mode, leaves the previous snapshot intact and no temporary behind;
- `_finish_replay_snapshot` reports instead of raising;
- end to end: a two-iteration `agot train` leaves a complete snapshot holding the logged
  buffer size, the second record shows the wait, and `--resume` restores it;
- `distinct_rows` keeps every bit pattern (including a `-0.0` twin of a repeated row) and drops
  the repeats; with every key forced equal it still keeps every differing row;
- `calibrate_keep(dedupe=True) == calibrate_keep(dedupe=False)` on repeated rows, with real
  and with forced-colliding keys, including a row that differs from a heavily repeated one in
  one entry and is the only row that fires a unit.

## Gates

- `.venv\Scripts\python -m pytest -q` → exit 0; 292 passed, 0 skipped or failed (the summary
  line is swallowed when piped; count from the dot lines).
- `.venv\Scripts\ruff check src tests` → `All checks passed!`
- No Rust change; the cargo gates are untouched and no `maturin develop` was run.

## Measurement

Machine: this 22-core box, run7's net, production flags (`ab_multi.py` / `ab_iters.py`: 32
games, 2 workers, batch 16, 128 simulations, phase budgets, int8, 2 train threads,
`--buffer-save-every 1`), arms interleaved. "other" = `seconds - seconds_selfplay -
seconds_train`.

### The bucket, before and after (second iteration, `ab_iters.py --iters 2`, final run below)

| component | `before` (`e1c9c71`) | this branch |
|---|---|---|
| leaf graph (`seconds_leaf_graph`) | 2.42–2.48 | 1.11–1.35 |
| of which calibration (`seconds_leaf_calibration`) | 1.90–1.95 | 0.61–0.86 |
| of which export + int8 quantize | 0.49–0.55 | 0.49–0.53 |
| replay snapshot | ~2.2–2.3 (synchronous; `before` has no timer: the bucket minus the leaf graph is 2.39–2.55, which also holds `extend` and the saves) | **0.000** wait (`seconds_replay_wait`) + 0.001–0.002 to start; 2.77–3.34 s in its thread, overlapped |
| `buffer.extend` (`seconds_extend`) | ~0.21 (same code; timed in the branch only) | 0.21–0.24 |
| `model.save` + `run.save` (`seconds_checkpoint`) | ~0.01 | 0.009–0.011 |
| unattributed residual | — | 0.003–0.004 |
| **total ("other")** | **4.81–5.02, median 4.98** | **1.36–1.61, median 1.36** |

### Whole branch, steady state: `ab_iters.py --arm OZ=<baseline 7952ae2> --arm OA=<before> --arm OB=<branch> --rounds 1 --games 32 --iters 2`, run three times (one round per command), second iteration

| metric | OZ baseline median (min–max) | OA before | OB branch | OZ/OB per round | OZ/OB median | OA/OB per round | OA/OB median |
|---|---|---|---|---|---|---|---|
| `seconds` | 49.38 (48.75–52.16) | 27.22 (25.89–28.02) | 24.39 (23.60–24.48) | 1.991, 2.093, 2.138 | **2.093x** | 1.112, 1.097, 1.149 | 1.112x |
| `seconds_selfplay` | 21.64 (17.60–22.31) | 10.18 (9.89–10.66) | 9.83 (9.62–10.83) | 1.790, 2.249, 2.061 | 2.061x | 1.036, 1.028, 0.985 | 1.028x |
| `seconds_train` | 26.99 (25.03–28.06) | 12.06 (11.19–12.34) | 12.62 (11.96–13.30) | 2.111, 1.984, 2.258 | 2.111x | 0.907, 0.887, 1.032 | 0.907x |
| other | 2.85 (2.71–3.09) | 4.98 (4.81–5.02) | 1.36 (1.36–1.61) | 2.277, 1.992, 1.768 | 1.992x | 3.670, 3.542, 3.110 | **3.542x** |
| process wall (2 iterations) | 111.93 (102.47–112.56) | 63.18 (56.12–64.77) | 58.03 (55.32–58.26) | 1.759, 2.035, 1.929 | 1.929x | 1.084, 1.014, 1.116 | 1.084x |
| sum of `seconds` (2 iterations) | 107.06 | 58.50 | 51.72 | 1.895, 2.176, 2.064 | 2.064x | 1.131, 1.063, 1.165 | 1.131x |

All nine processes reproduced their arm's results exactly (OA = OB in both iterations:
17 235 / 17 223 examples, losses and keep-sets as quoted under Equivalence; OZ, the baseline
without compaction, 17 594 / 17 249). The branch's `seconds_train` came out 0.89–0.91x of
`before` in two of these rounds (and 0.83 / 0.95 / 1.03 in lead 1's A/B), although the
snapshot has always finished before training starts (`seconds_replay_wait` 0.000). Isolated
in process (`train_after_save.py`: 20k examples, an epoch right after a synchronous save vs
right after a background save + join, alternating, 4 pairs) it is 2.527 vs 2.531 s median
with identical losses, and in the one-iteration runs below the branch trained *faster*
(1.057x). So it is noise, not an effect.

### Whole branch, prescribed protocol: `ab_multi.py --arm OZ=<baseline> --arm OA=<before> --arm OB=<branch> --rounds 3 --games 32` (one iteration per process)

| metric | OZ median (min–max) | OA | OB | OZ/OB per round | OZ/OB median | OA/OB median |
|---|---|---|---|---|---|---|
| `seconds` | 51.01 (42.22–53.65) | 26.41 (25.95–28.04) | 23.61 (22.47–24.42) | 1.788, 2.197, 2.271 | **2.197x** | 1.148x (1.119, 1.148, 1.155) |
| `seconds_selfplay` | 22.83 (19.23–23.90) | 11.79 (10.14–11.95) | 11.26 (10.73–11.55) | 1.708, 2.069, 2.128 | 2.069x | — |
| `seconds_train` | 25.22 (20.28–26.63) | 11.48 (9.83–12.05) | 10.86 (10.31–11.52) | 1.867, 2.312, 2.445 | 2.312x | — |
| other | 2.96 (2.71–3.12) | 4.33 (4.23–4.61) | 1.42 (1.35–1.49) | 1.820, 2.304, 2.084 | 2.084x | 3.046x (2.835, 3.405, 3.046) |
| process wall | 55.31 (46.38–58.30) | 29.94 (29.49–31.56) | 29.59 (28.04–30.32) | 1.567, 1.923, 1.973 | 1.923x | 1.041x |

This protocol flatters lead 1: with one iteration per process the snapshot runs after the log
record with nothing to overlap and is joined at exit, so it leaves `seconds` but not the
process wall (OB wall − `seconds` ≈ 6 s vs 3.5 s for OA). The two-iteration table is the
honest one. OA round 2 hit the training non-reproducibility below (same examples, `policy_loss`
1.575503 vs 1.575518); its timings are kept.

### Per lead (`ab_iters.py --iters 2 --rounds 3 --games 32`, second iteration)

| A/B | A other (per round) | B other | A/B per round | median | spread (B, max−min) |
|---|---|---|---|---|---|
| lead 1: `before` (OA) vs background snapshot (OB) | 4.316, 4.331, 5.007 | 2.620, 2.483, 2.891 | 1.647, 1.744, 1.732 | 1.732x (−1.71 s) | 0.41 s |
| lead 2: lead 1 (OL, `92d9d87`) vs + distinct rows (OD) | 2.595, 2.564, 2.650 | 1.345, 1.355, 1.269 | 1.929, 1.892, 2.088 | 1.929x (−1.25 s) | 0.09 s |

Both are far beyond the keep rule (3 % of the bucket and twice the round spread).

### Equivalence

- Every round in which neither arm hit the training non-reproducibility below: `examples_new`,
  `target_entropy`, `policy_loss`, `value_loss` and `leaf_compact` identical between `before`
  and the branch, in both iterations of the two-iteration runs (e.g. 17 235 / 1.4203 /
  1.5755182494993534 / 0.14317705972207714 / [54, 83, 83, 91], then 17 223 / 1.4350 /
  1.5662052781641984 / 0.13844703266068417 / [54, 88, 83, 91]).
- `agot.replay.npz` in the arms' scratch directories: identical names, dtypes, shapes and
  bytes for `before` vs lead 1 (`work-OA` / `work-OB`, 76 159 examples, 1 246 239 rows) and
  lead 1 vs lead 2 (`work-OL` / `work-OD`).
- Calibration (`check_dedupe.py`, `peaks.py`): keep-sets identical with and without dedupe on
  run7's buffer as restored (61 701 examples) and after a simulated 17.5k extend (79 201), and
  on the two-iteration successor buffer and net (`work-OB`, 76 159 and 73 659 examples, widths
  [53, 89, 83, 91]). Beyond the keep-sets, the per-unit peak pre-activations are
  **bit-identical** on all eight trunk layers of both nets. Nearest unit to the threshold:
  dropped at −0.00171 (action trunk, layer 1), kept at +0.00407 (state trunk, layer 1).

Training is not bit-reproducible on this box under load, and that is pre-existing: in 3 of
30 process runs the self-play examples matched but the losses differed in the 5th–6th digit.
One was the `before` arm (`ab_multi` round 2: `policy_loss` 1.575503 vs 1.575518, same
17 235 examples), which contains none of these changes; one was the lead-1 arm's *first*
iteration (OL round 0), before any snapshot thread exists; one was lead 2's second iteration
during a load spike (OD round 1, `seconds_train` 24.9 s instead of ~10.5 s), after its
snapshot had finished. Not investigated further; a likely cause is
MKL's dynamic threading (`MKL_DYNAMIC`) picking fewer threads when the box is busy, which
changes the gemm reduction split. Rounds with such a divergence are flagged in the tables and
excluded from the equivalence claims only, not from the timings.

### Leads

| lead | measured | verdict |
|---|---|---|
| 1. background, atomic snapshot | second-iteration other 4.33 → 2.62 s (per round 1.647 / 1.744 / 1.732x vs `before`); `seconds_replay_wait` 0.000 every round; `seconds_replay_snapshot` 2.27–2.83 s in its thread (1.9 s synchronous: it shares the CPU with calibration) | **kept** |
| 2. calibration over distinct rows | in process 1.37 → 0.46 s (79k examples); inside the loop `seconds_leaf_calibration` 1.85–2.15 → 0.53–0.64 s; second-iteration other 2.60 → 1.34 s (per round 1.929 / 1.892 / 2.088x vs lead 1) | **kept** |
| 2b. live columns for the second trunk layer | not built: after dedupe the trunk forward is ~0.13 s of the 0.48 s calibration (54k action rows); live columns would save at most ~1/3 of that, < 3 % of the bucket | rejected (ceiling) |
| 2c. calibration on every core | not built: Colab, where the runs are, has 2 vCPUs (already used by the 2 train threads), and the box is shared with another agent's benchmarks; after dedupe the forward is 0.13 s anyway | rejected (does not transfer) |
| 2d. `np.unique` → `argsort` grouping | 0.10 → ~0.06 s, i.e. ~0.04 s = 3 % of the new bucket | rejected (below threshold) |
| 3a. block `extend` | 17.5k examples: `add` loop 0.193–0.201 s, block append with `add`'s per-example assignments 0.194–0.223 s, `np.concatenate` block 0.236–0.270 s; contents identical in all three. The cost is not per example: into a pre-grown buffer the same extend is 0.065 s; the rest is the one-time growth copy after a restore (0.158 s, the restored storage is exactly sized) and the eviction memmove when the buffer fills (0.175–0.183 s) | rejected (≤ 0.035 s possible) |
| 3b. int8 quantize | `quantize_dynamic` 0.24 s warm / 0.40 s first; 0.16 s of it is ONNX Runtime's own temp-file and external-data I/O; the fp32 export is 0.03 s warm | rejected (library internals) |
| 3c. hashing the checkpoint / weights, cache hit, device moves, `target_entropy`, log | 0.001–0.02 s each | nothing to take |

## Verdict

Kept. Prediction met: the second iteration's time outside both phase timers went from
4.3–5.0 s to 1.36 s (3.2–3.5x), with every result identical. The snapshot is fully hidden on
this box (never waited for); calibration costs a third of what it did. The branch now clears
the 2.0x goal on `seconds` against `7952ae2`: 2.093x median in steady state (second of two
iterations; per round 1.991, 2.093, 2.138) and 2.197x on the one-iteration protocol (which
overstates lead 1, see above).

On Colab (2 vCPUs, two self-play workers) the background snapshot is not free: zlib competes
with the workers for the two cores. The run logs put the synchronous snapshot at 7–9 s there,
during which one core sat idle, so overlapping it should still recover roughly half of that per
iteration. `seconds_replay_wait` and `seconds_replay_snapshot` in the log will show it on the
next run. The calibration change is pure work reduction and transfers as is.

## Follow-ups

- Eviction memmove (0.18 s per eviction, every iteration in production once the 80k buffer is
  full) and the post-restore growth copy (0.16 s once per session): a head-offset layout would
  remove the memmove, but `cli._replay_rows` and the tests' `_contents` read the private
  arrays assuming the data starts at 0, so it needs a public read-only accessor first.
- int8 quantize writes and reads temp files through ONNX Runtime (0.16 s of 0.24 s here,
  likely less on Linux). If fp32 becomes the training engine (008's open decision), it goes
  away.
- Training non-reproducibility under load (above): pin `MKL_DYNAMIC=FALSE` /
  `MKL_CBWR` in a measurement harness and see whether it disappears; if so, A/B harnesses
  should set it so equivalence checks never need to drop rounds.
- `ab_multi.py` measures one iteration per process, which hides any work moved after the
  record (here, the snapshot joined at exit). `checkpoints/perf-2x/ab_iters.py` (two
  iterations, reports the last) is the honest harness for overlap changes.

## Ledger rows and findings to merge

LEDGER.md:

| id | date | track | change | metric | before | after | delta | verdict | commit |
|---|---|---|---|---|---|---|---|---|---|
| 010a | 2026-11-20 | SPEED | replay snapshot written in a background thread, atomically | other (`seconds` − selfplay − train), 2nd of 2 iterations, run7, 32 games, 3 interleaved rounds | 4.33 s | 2.62 s | 1.73x (1.65–1.74) | kept | `92d9d87` |
| 010b | 2026-11-20 | SPEED | compact-graph calibration over distinct (bit-verified) rows | same | 2.60 s | 1.34 s | 1.93x (1.89–2.09) | kept | `2ad2317` |
| 010 | 2026-11-20 | SPEED | whole branch vs `7952ae2` (006–008 + 010) | `seconds`, 2nd of 2 iterations | 49.38 s | 24.39 s | 2.09x (1.99–2.14) | — | `2ad2317` |

FINDINGS.md:

- One iteration's time outside `seconds_selfplay`/`seconds_train` is now attributed in the log
  (`seconds_leaf_calibration`, `seconds_leaf_graph`, `seconds_extend`, `seconds_checkpoint`,
  `seconds_replay_save`, `seconds_replay_wait`, `seconds_replay_snapshot`,
  `seconds_trainer_restore`).
- The replay snapshot can be written in the background with no change to its arrays as long as
  every buffer mutation waits for it; it is hidden behind calibration + self-play on a
  many-core box (`seconds_replay_wait` 0.000). Command: `ab_iters.py --iters 2`.
- Replay candidate rows are ~95 % repeats (run7: 54k distinct of 1.0–1.3M); calibration over
  distinct rows (hash + bitwise verification) gives bit-identical per-unit peaks at a third of
  the cost.
- `ReplayBuffer.extend` is dominated by storage growth after a restore and by eviction
  memmoves, not by the per-example Python loop (0.065 s for 17.5k examples into a pre-grown
  buffer).
- Training is not bit-reproducible on this box under load (losses differ in the 5th–6th digit
  with identical examples, ~1 run in 8, also on unchanged code).
