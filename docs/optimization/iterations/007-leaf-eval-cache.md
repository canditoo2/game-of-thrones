# 007 — exact cross-pass leaf evaluation cache in the Rust self-play driver

- Track: SPEED
- Status: kept (cache, timers); hygiene round: one lead measured and reverted, the others
  rejected on their measured ceiling
- Commits: `c4af05f` (cache), `d455eb2` (per-section timers)

## Hypothesis

72.5% of in-process self-play time is the leaf callback (`share.py 16 onnx-int8`: 12.4 of
17.1 s), and 72.8% of the leaves one `selfplay()` call sends to the network were already scored
earlier in the same call (`dup_study.py 16`: 321 453 leaves, 87 418 distinct; an LRU of 2 000
entries gets the same hit rate). Two causes: every decision restarts its tree
(`search.rs` `open_search_worlds` clears the arena), so the next decision scores the root and
subtree again; and the 97-float state row carries no board, so many distinct positions encode to
the same network input. The in-pass `LeafDedupe` (`selfplay.rs`) keeps nothing across passes.
The weights are fixed for one call, so remembering each input's output is exact for an
evaluator that is a pure row-wise function of its input.

## Prediction (written before the change)

- Metric: `seconds_selfplay` from `ab_iter.py` (baseline `7952ae2` vs this branch, interleaved,
  32 games, production flags, int8) and the in-situ split (`share.py 16`).
- Expected: ~70% fewer leaves sent. If the callback cost scaled with leaves, int8 self-play
  would go from ~17 s to ~4.7 + 0.3 × 12.4 ≈ 8.5 s (2x). The per-call fixed cost does not shrink
  with the rows, so expect 1.5–1.9x for int8, more for fp32 (its per-row cost is higher).
- Falsified if: records differ cache on vs off under a row-wise pure evaluator, or the speed-up
  is under 1.3x.

## Change

`rust/crates/agot-core/src/leaf_cache.rs` (new, engine crate, no network dependency):

- **Key.** The exact network input of one leaf: the candidate count, the state row and the
  ordered candidate rows, as `f32` bit patterns (so `-0.0` ≠ `0.0`, as in `LeafDedupe`). The
  cache does not keep the ~10 KB per leaf; it keeps a 256-bit digest from two independent NH
  lanes (UMAC's universal hash: 64-bit words, 128-bit accumulators, keys from a fixed SplitMix64
  stream). NH is 2^-64-almost-universal per lane on equal-length messages, so a fixed pair of
  distinct inputs collides with probability ≤ 2^-128; different candidate counts have different
  lengths and the count is also stored and compared exactly. **Collision bound relied on:** a
  false hit needs some lookup to collide with some stored entry, ≤ lookups × capacity × 2^-128;
  at the default capacity (2^16) and even 2^30 lookups in one call (a production 16-game call
  makes ~3×10^5) that is below 2^-82.
- **Verification mode.** `leaf_cache_verify` (spec key, default off) keeps every entry's full
  input and compares it byte for byte on each hit, and also checks in-pass repeats; a mismatch
  raises instead of returning a value. Every Python equality test runs with it on.
- **Value.** The raw logits and the value vector exactly as the evaluator returned them; the
  driver's segment softmax, root Dirichlet noise and backup run downstream unchanged.
- **Bound.** Two generations: entries go into `current`; when it holds half the entry cap or
  half the logit cap, it becomes `previous` and the old `previous` is dropped. A hit in
  `previous` moves the entry back. Defaults: ≤ 2^16 entries and ≤ 2^22 logits per evaluator
  (16 MiB of logits + ~100 B per entry ≈ 23 MiB worst case; one oversized entry can overshoot
  the logit half-cap once). Measured peak in a production 16-game call: 65 444 entries,
  1.24 M logits (~12 MB), 2 rotations, ~32.7 k entries evicted — evictions only cost
  re-evaluations.

`rust/crates/agot-py/src/selfplay.rs`:

- One cache per evaluator per `selfplay()` call (created lazily, dropped with the call), so a
  seat model never sees another model's outputs and new weights never see old ones.
- Per pass and model group: hits are copied from the cache; misses are packed in
  first-occurrence order with in-pass repeats merged by key (so the cache subsumes
  `dedupe_leaves`), sent in **one** callback, and inserted; a pass with no miss makes **no**
  callback. The cache-off path (`evaluate_direct`) is the old code moved into a function.
- Spec keys: `leaf_cache` (default `true`), `leaf_cache_verify`, `leaf_cache_entries`,
  `leaf_cache_logits`.
- `_core.selfplay_stats()` returns the counters of the last call in the process (`leaves`,
  `slots`, `cache_hits`, `cache_hit_slots`, `pass_duplicates`, `evaluator_calls`,
  `calls_skipped`, `rows_sent`, `slots_sent`, cache bookkeeping and, since `d455eb2`,
  `seconds_*` sections). The return tuple is unchanged, so `rust_engine._unpack` and
  `evaluate._play_model_games` are untouched.

Python plumbing (unlike `dedupe_leaves`, reachable from a training run):
`SelfPlayPlan.leaf_cache` → `_options` → `batched_self_play(leaf_cache=)` →
`try_batched_self_play(leaf_cache=)` → spec; `spec_for(..., leaf_cache=)`; and
`agot train --leaf-cache on|off` (default on). While there, `rust_engine.self_play_records`
stopped playing its games twice (a merge leftover ran a discarded first `selfplay()` call and
dropped `phase_budgets` from the call that counted); it has no production caller.

`d455eb2` adds per-section wall-clock timers (a few clock reads per pass, never per leaf).

## Gates

From `rust/`:

```
cargo fmt --all -- --check                        -> clean
cargo clippy --all-targets -- -D warnings         -> clean
cargo clippy -p agot-py --all-targets -- -D warnings -> clean
cargo test --release -> 242 + 38 + 3 + 3 + 4 + 12 passed, 0 failed (6 new in leaf_cache)
```

Frozen pins (`engine_fixture.json`, `encoder_golden.json`), `search_semantics` and
`phase_budgets` untouched and green.

Python (worktree `.pyd` built with `maturin build --release`, extracted to `src/agot/`;
`PYTHONPATH` = worktree `src`, verified with `agot.__file__` / `agot._core.__file__`):

```
python -m pytest -q   -> exit 0 (249 collected, 1 skipped)
python -m ruff check src tests -> All checks passed!
```

New tests (`tests/test_leaf_cache.py`, all with `leaf_cache_verify=True` on the cached arm),
using an evaluator that mixes the float bit patterns with exact `uint64` arithmetic so its
output cannot depend on batch shape:

- training-shaped spec (temperature schedule, phase budgets that cap, sequential Westeros,
  deferred battles, Doran/choice actions, Lannister ship), det 1 and det 2: all ten returned
  arrays/lists bit-identical cache on vs off; counters consistent (hits + in-pass + sent =
  leaves, calls + skipped passes = cache-off calls);
- cache vs the oldest reference path (`leaf_cache=False, dedupe_leaves=False`);
- a 16-entry / 256-logit cache (rotations and evictions) still bit-identical;
- evaluate-style two-model spec (`seat_models`, two different fake networks): identical, so
  the per-evaluator caches are not shared;
- plumbing: `SelfPlayPlan` → `spec_for`, and `try_batched_self_play(leaf_cache=)`.

**One existing test changed its input, flagged here:** `test_selfplay_batches_the_network_calls`
asserts the *callback* shape (≥0.8 × batch leaves per call) with a uniform evaluator, whose
leaves are nearly all cache hits (average fell to 1.44 leaves per call). It now passes
`leaf_cache=False` explicitly, with its assertions unchanged, and a new sibling
`test_selfplay_batches_each_pass_with_the_leaf_cache` asserts the same batching per pass with
the cache on (≥0.8 × batch leaves per pass, one callback or none per pass).

## Measurement

Machine state: another agent's ONNX benchmarks (`work-CA/CB`, `work-share-onnx`) ran on this
box during part of these runs, which shows up in the absolute numbers (training time of the
*same* code varied 35–108 s). Only the paired ratios are readable.

### End to end — `ab_iter.py`, baseline `7952ae2` (RA) vs branch (RB), 32 games, int8

Rounds 0–2 (branch at `c4af05f`):

| round | RA selfplay | RB selfplay | ratio |
|---|---|---|---|
| 0 | 57.79 s | 21.16 s | 2.731x |
| 1 | 32.20 s | 18.66 s | 1.726x |
| 2 | 32.53 s | 26.82 s | 1.213x |

The spread exceeded 15%, so two more interleaved rounds were added later (branch at
`d455eb2`, which adds only the timers), on a quiet machine (RA within 0.2% of itself):

| round | RA selfplay | RB selfplay | ratio |
|---|---|---|---|
| 3 | 24.85 s | 15.20 s | 1.635x |
| 4 | 24.82 s | 13.74 s | 1.807x |

All five rounds: **median 1.726x**, min 1.213x, max 2.731x; the two quiet rounds 1.64–1.81x.
Whole iteration (`seconds`) in the quiet rounds: 1.22x / 1.28x (training time unchanged,
1.00x / 1.03x).

`examples_new` RA 17 594 / RB 17 050 every round (int8 trajectories differ, see below);
`target_entropy` 1.410 vs 1.423.

### In-situ split — `share.py 16 <backend>`, interleaved, PYTHONPATH per arm

| backend | round | baseline selfplay | cache selfplay | ratio | leaf callback | rest |
|---|---|---|---|---|---|---|
| int8 | 1 | 18.43 s | 11.03 s | 1.671x | 13.47 → 6.08 s | 4.95 → 4.95 s |
| int8 | 2 | 17.69 s | 10.80 s | 1.638x | 12.79 → 5.88 s | 4.90 → 4.93 s |
| int8 | 3 | 20.52 s | 10.57 s | 1.941x | 15.04 → 5.78 s | 5.47 → 4.79 s |
| fp32 | 1 | 34.34 s | 16.32 s | 2.104x | 28.19 → 10.37 s | 6.14 → 5.96 s |
| fp32 | 2 | 34.75 s | 15.92 s | 2.183x | 28.64 → 10.08 s | 6.10 → 5.84 s |
| fp32 | 3 | 31.87 s | 15.29 s | 2.084x | 26.13 → 9.82 s | 5.74 → 5.47 s |

Median: **int8 1.67x** (min 1.64, max 1.94), **fp32 2.10x** (min 2.08, max 2.18). Leaf calls
50 569 → 35 884 (int8) and 49 246 → 34 429 (fp32); the per-call cost falls too (266 → 169 us)
because the calls carry fewer rows. "Rest" (Rust + glue) is unchanged: the digest and lookups
cost no more than the `LeafDedupe` they replace.

### Hit rates (production chunk: 16 games, 128 sims, det 1, default budgets, batch 16)

| backend | leaves | cache hits | hit slots | in-pass repeats | rows sent | slots sent | passes with no callback |
|---|---|---|---|---|---|---|---|
| fp32 | 303 040 | 73.6% | 70.5% | 0.04% | 26.3% | 29.5% | 29.3% |
| int8 | 294 093 | 73.5% | 70.7% | 0.04% | 26.5% | 29.3% | 32.0% |

### Record equality on run7's net (`cache_eq_rust.py`)

- **Fake row-wise pure evaluator:** bit-identical (tests above).
- **onnx-fp32:** records **bit-identical** cache on vs off for the 16-game production chunk
  (8 058 examples). Not guaranteed, though: replaying the cache-off trajectory and comparing
  every repeat of an input with its first evaluation, 99.8% of 220 173 repeats are
  bit-identical, and the other 0.2% differ by ≤ 2.98e-7 in a value (policy KL exactly 0). ORT
  fp32 is not strictly row-invariant across batch shapes (the packed graph's one-hot pooling
  matmul is a plausible source); in this sample no visit count moved.
- **onnx-int8:** records differ (as expected: dynamic quantization takes its activation range
  from the batch). Cached vs fresh for 200 897 repeated leaves (76 423 distinct): only 3.1%
  bit-identical; policy KL(cached‖fresh) mean 1.02e-3, p50 4.2e-4, p95 4.0e-3, p99 7.0e-3,
  max 3.2e-2 nats; value |diff| mean 0.020, p95 0.099, max 1.24. This is int8's own
  batch-composition noise (fresh-vs-fresh for the same input differs just as much), not
  something the cache adds: with the cache, every repeat of an input inside a call gets the
  same score; without it, the score of a leaf depends on what else happened to be in its batch.

## Hygiene round (after the cache)

Section split from the driver's timers (`split_rust.py onnx-fp32 16`, cache on, `d455eb2`):
total 14.8 s = callback 9.6 · scripted greedy seats (Python decider) 1.7 · prepare 1.28 ·
encode 0.93 · evaluate glue (hash, lookup, copies, softmax, scatter) 0.83 · finish 0.29 ·
advance excluding scripted 0.15 · record 0.04. Rust + glue ≈ 3.5 s of 14.8 s.

| lead | verdict | evidence |
|---|---|---|
| root encoded twice (pass 0 leaf + `observe` for the record) | rejected, not implemented | the whole `record` section (aggregate, pick, `observe`, step) is 0.04 s = 0.3% |
| `legal_actions()` only for `.len()` | rejected, not implemented | the whole non-scripted `advance` section is 0.15 s = 1% |
| `leaf_game()` clones `State` per leaf | **measured, reverted** | `leaf_state()` borrow: `encode` 0.90/0.99/0.79 → 0.70/0.55/0.62 s (median −0.28 s, ~1.8% of self-play), records digest identical; below the 3% bar and inside the ±12% round spread |
| `RegionCache::new()` per pass, `clear()` per leaf; `fortifications.contains` | rejected on ceiling | lives inside `encode`, which is 0.62 s (4%) even after the clone removal; any share of it is < 3% |
| dedupe `to_vec()` copies, all-ones `mask` | moot / rejected on ceiling | the cache path no longer runs `LeafDedupe`; the mask is part of the 0.83 s evaluate glue together with hashing and softmax |
| `march_actions(..).is_empty()` twice per March step | not attempted | inside `prepare`/`finish` (1.6 s total, all engine stepping); touches `game.rs` rule code for a fraction of that |
| `with_node_game` move-out/re-box | not attempted | inside `prepare` (1.28 s); same ceiling argument |

## Verdict

Predicted 1.5–1.9x for int8; measured 1.67x in situ (median of 3) and 1.73x end to end
(median of 5 rounds; 1.64–1.81x in the two quiet rounds), 2.10x for fp32 in situ.
Bit-identical with a pure evaluator (pinned by tests, verify mode on) and, on this sample, with
onnx-fp32. Kept, default on.

## Follow-ups

- The callback is still 65% of cached self-play, and 29% of passes now make no callback at all;
  the remaining calls carry ~2.5 leaves. Larger batches (`--batch-games`) now pay more than
  before, since the cache raises the per-call fixed cost's share.
- The scripted greedy seats (`--opponent-rate 0.2`) cost 1.7 s of 14.8 s (11%), the
  largest non-network item: a Rust greedy policy would remove a Python call per scripted
  decision. Out of this driver's scope.
- int8 batch-composition noise (value |diff| up to 1.24 for the same input) is worth its own
  look: per-tensor dynamic quantization of a batch with an outlier row coarsens every row.
- The deferred gate and evaluation get the cache by default (Rust default on); their records
  are not training data, but with fp32 they stay identical on the sample measured.

## Ledger rows and findings to merge

LEDGER.md:

| id | date | track | change | metric | before | after | delta | verdict | commit |
|---|---|---|---|---|---|---|---|---|---|
| 007 | 2026-11-20 | SPEED | exact cross-pass leaf evaluation cache in the Rust driver | `share.py 16` selfplay (int8 / fp32, median of 3) | 18.43 s / 34.34 s | 10.80 s / 15.92 s | 1.67x / 2.10x | kept | `c4af05f` |
| 007 | 2026-11-20 | SPEED | (same) | `ab_iter.py` `seconds_selfplay`, int8, 32 games, median of 5 | 32.20 s | 18.66 s | 1.73x (quiet rounds 1.64–1.81x) | kept | `c4af05f` |
| 007b | 2026-11-20 | HYGIENE | per-section timers in `selfplay_stats` | overhead | — | few ms/call | — | kept | `d455eb2` |
| 007c | 2026-11-20 | HYGIENE | borrow the leaf `State` instead of cloning it | `encode` section / selfplay | 0.90 s | 0.62 s | −1.8% of selfplay | reverted (< 3%) | none |

FINDINGS.md:

- ~73% of the leaves of one production `selfplay()` call repeat an input already scored in that
  call; the repeats are recent (an LRU of 2 000 suffices). `split_rust.py` /
  `cache_eq_rust.py` print the counters (`_core.selfplay_stats()`).
- With the cache on, int8 self-play is callback 55% / Rust+glue 45% in situ; Rust+glue splits
  as prepare 1.28 · encode 0.93 · evaluate glue 0.83 · finish 0.29 s per 16 games, and the
  Python greedy seats cost 1.7 s (`split_rust.py onnx-fp32 16`).
- ORT fp32 is row-invariant for 99.8% of repeated leaves across batch shapes; the rest differ
  by ≤ 3e-7 in a value. ORT int8 is not: the same leaf in two batches differs by up to 1.24 in a
  value and 3e-2 nats in policy (`cache_eq_rust.py onnx-int8 16 drift`).
