# 009 — self-play driver, round 2: the non-network half

- Track: SPEED
- Status: kept (seven code commits + one test commit); one lead measured and reverted, the
  rest rejected on their ceiling
- Commits: `84b2865` (scripted-seat pin test), `be208ae` + `94dbe85` (snapshot),
  `ae1a9af` + `fd749d4` (scripted decider), `b13f67c` (per-game scheduling),
  `b3e3659` + `71c0b2a` (hygiene bundle); the docs in the commit after them

## Hypothesis

After 007 (leaf cache) and 008 (compact graph) the network is no longer most of self-play.
Re-measured on the merged tree (`a95445a`, compact onnx-fp32 graph of run7's net, the
production 16-game chunk, `split_compact.py onnx-fp32 16`): **8.10 s = leaf callback 3.75 ·
Python scripted greedy seats 1.50 · prepare 1.01 · encode 0.76 · evaluate glue 0.68 · finish
0.23 · advance (non-scripted) 0.12 · record 0.04**. 4.35 s (54 %) is not the network.

Two things the earlier split did not show:

1. The scripted seats' 630 us per decision is mostly not the greedy heuristic: the engine's
   `snapshot()` built a `serde_json::Value`, serialised it and had Python `json.loads` it
   (312 us), the view converted every legal action twice, and each decision's `RustGame` /
   `StateView` pair is a reference cycle, so its few thousand objects waited for Python 3.14's
   incremental cyclic GC (~1 300 collections per 8.6 k decisions, ~60 us per decision).
2. Passes are 41 % full: a pass runs one simulation for every game whose decision still has
   budget, and the batch waits for the deepest decision (phase budgets make decisions very
   unequal) before any game moves on. 45 758 passes carried 303 040 leaves (6.6 of 16).
   Per leaf call the compact fp32 graph costs ~40 us fixed + ~22 us per leaf
   (`call_curve.py`, real leaves), so half-empty passes cost ~40 us each in fixed cost.

## Prediction (written before the change)

- Metric: `seconds_selfplay` from `ab_multi.py` (integration `7a81c5e` vs this branch,
  interleaved, 3 rounds, 32 games, production flags, int8 compact) and the in-situ split.
- Expected: scripted seats −0.5 to −0.8 s, per-game scheduling −0.6 to −1.0 s (fewer, fuller
  callbacks), hygiene −0.3 to −0.5 s per 16-game chunk: 1.2–1.35x on `seconds_selfplay`.
- Falsified if: any record differs under a row-wise pure evaluator, or onnx-fp32 records
  change, or the end-to-end median is under 1.2x.

## Change

One logical change per commit; every commit was checked against the record digests below
before it was kept.

| commit | change |
|---|---|
| `84b2865` | `tests/test_scripted_driver.py`, written and green **against the unmodified code**: scripted greedy seats on the Rust driver record identical games cache on/off (det 1/2, record on/off), and every driver decision equals `GreedyPolicy.decide` on a full `RustGame` clone with a copy of the seat's RNG (same index, same RNG state afterwards, same target). |
| `be208ae` | `PyGame.snapshot()` converts the fingerprint `Value` with the existing `to_python` instead of `serde_json::to_string` + Python `json.loads`. Same dict: sorted keys, `bool`/`int` apart (`tests/test_rust_snapshot.py`). |
| `ae1a9af` | Scripted decider: legal actions converted once (the policy gets a fresh list of the same objects; the index is still the first *equal* action, so duplicate actions resolve as before); `StateView.regions` answers a current memo without `_sync`; `_units` maps names by dict lookup (same enum members). |
| `b13f67c` | **Per-game scheduling.** The batch loop no longer runs `max(per_world)` passes and then advances every game: each game keeps `(roots, per_world, done)`; after every pass a game that spent its budget aggregates, picks, records and steps, and opens its next decision in the next advance. Inside a pass, games are visited in slot order and a game's roots in root order, so each game runs exactly the same `prepare_simulation`/`finish_simulation` sequence on its own searcher as before; only which games share a leaf callback changes. Every game is still seeded, scripted and stepped only by its own state. (A game whose `open_search_worlds` returns no root is now marked over instead of being re-opened every round; it could never move again either way.) |
| `b3e3659` | Hygiene bundle, Rust: `Mcts::leaf_state` borrows the leaf's `State` for the encoder instead of `leaf_game` cloning it; one `RegionCache` per batch whose `clear()` bumps a generation stamp instead of rewriting 64 entries; `encode_state_into` tests fortifications with a 64-bit set instead of `SmallVec::contains` per controlled region. New test `a_cleared_region_cache_serves_the_new_state`. |
| `71c0b2a` | Hygiene bundle, Python: `LeafBackend.__call__` calls the CPU session's native `_sess.run` (the same C call `InferenceSession.run` makes after re-validating names, OrtValue ownership and WebGPU capture) and keeps its output-name list; any other session uses the public `run`. Test: native path is byte-identical to the public `run` for fp32 and int8. |
| `94dbe85` | `snapshot_py.rs` writes the snapshot straight into Python objects (no `serde_json::Value` tree), keys inserted in sorted order with a stable sort (a repeated key keeps the last value at the first position, as `BTreeMap` does). `Game._snapshot_reference()` keeps the fingerprint path; a test compares both, with exact key order and types, over four greedy games with battles, bids and choices. |
| `fd749d4` | The decider sets `view._state_view = None` after the policy decides, breaking the `RustGame` ↔ `StateView` cycle so the decision's objects are freed by refcount instead of the cyclic GC. |

Not changed: the leaf callback protocol (still `(state, actions, index, mask)` with an all-ones
mask), the Rust spec, the cache, `search_semantics`, `phase_budgets`, legal-action order,
canonical action text, the frozen fixtures.

## Gates

From `rust/` (stable `x86_64-pc-windows-gnu`):

```
cargo fmt --all -- --check                            -> exit 0
cargo clippy --all-targets -- -D warnings             -> exit 0
cargo clippy -p agot-py --all-targets -- -D warnings  -> exit 0
cargo test --release -> 243 + 38 + 3 + 3 + 4 + 12 passed, 0 failed (1 new: region cache)
```

`.pyd` rebuilt with `maturin build --release` and extracted into the worktree's `src/agot`;
`PYTHONPATH=<worktree>\src`, `agot.__file__` and `agot._core.__file__` both in the worktree:

```
python -m pytest -q -> exit 0 (287 collected, 1 skipped; new: test_scripted_driver.py (6),
                       test_rust_snapshot.py (5), one onnx native-run test x2)
python -m ruff check src tests -> All checks passed!
```

## Measurement

Machine: shared box, other agents' benchmarks possibly running; only interleaved ratios are
read. Integration arm = `7a81c5e`, baseline arm = `7952ae2` (each with its own `.pyd`).

### Record equality

- **Fake row-wise pure evaluator** (`eq_fake.py`: the `hashing_evaluator` of
  `test_leaf_cache.py`; 11 cases: det 1/2, cache on/off, scripted seats det 1/2 with
  `record_opponents` off/on, an 8-game 4-round case with scripted seats and three-phase
  budgets, a no-cache case): every returned array digest identical to the merged tree's
  before **every** commit (`eq_base.txt` = `eq_new` = `eq_async` = `eq_hyg` = `eq_gc` =
  `eq_fresh`), including the leaf counts.
- **Scripted-only driver** (all six seats greedy, 16 games, 8 569 decisions, records on):
  digest `165cb99630d5bf88` before and after.
- **Snapshots**: 577 snapshots of three greedy games, exact key order and leaf types,
  identical across the JSON path, the `to_python` path and the direct builder.
- **onnx-fp32, compact graph, run7's net, in situ** (`split_compact.py onnx-fp32 16`):
  records digest `797d7baab19803d0`, 8 058 examples, 303 040 leaves, identical in every arm
  and every commit (the evaluator call count changes, the leaves do not).
- **onnx-fp32 end to end** (`ab_multi.py`, 3 rounds): `examples_new` 17 582 and
  `target_entropy` 1.4334 identical in both arms; the **replay buffers written by the
  iteration are bit-identical** (all six arrays of `agot.replay.npz`, 79 283 states,
  1 293 893 candidate rows). `policy_loss` 1.573700 vs 1.573691 and `value_loss`
  0.170577 vs 0.169287 differ because the integration arm `7a81c5e` does **not** contain the
  006 trainer pass (`df10927`, `09f17a4`, `3e54423`) that this branch inherits from its base
  (`git merge-base --is-ancestor df10927 7a81c5e` -> false), which also explains its 2.1x
  `seconds_train`. Same data, different trainer arithmetic; nothing in this iteration touches
  training.
- **onnx-int8**: records differ between arms, as expected (int8 dynamic quantization depends
  on batch composition, and the scheduling change regroups the leaves).

### Section split, in situ (`split_compact.py onnx-fp32 16`, one 16-game chunk, seconds)

| section | merged tree `a95445a` | HEAD `fd749d4` (median of 3) |
|---|---|---|
| total | 8.10 | 6.25 |
| leaf callback | 3.75 | 3.11 |
| scripted (Python greedy) | 1.50 | 0.72 |
| prepare | 1.01 | 0.99 |
| encode | 0.76 | 0.41 |
| evaluate glue | 0.68 | 0.63 |
| finish | 0.23 | 0.23 |
| advance, non-scripted | 0.12 | 0.12 |
| record | 0.04 | 0.04 |
| passes / evaluator calls | 45 758 / 32 359 | 26 330 / 23 589 |
| leaves per pass / rows per call | 6.6 / 2.47 | 11.5 / 3.38 |

### Per change, in situ, interleaved (`split_compact.py onnx-fp32 16`)

| change | arm A | arm B | per-round `total` A / B | median ratio |
|---|---|---|---|---|
| scripted view + snapshot `to_python` (`be208ae`, `ae1a9af`) | integration | working tree | 7.33 / 6.91, 7.40 / 7.10 | 1.05x (1.061, 1.042; scripted 1.34 -> 1.03 s) |
| per-game scheduling (`b13f67c`) | integration | working tree | 10.13 / 7.58, 8.96 / 8.22 | (noisy machine; callbacks 32 359 -> 23 589) |
| hygiene bundle (`b3e3659` + `71c0b2a`) | `b13f67c` | working tree | 7.87 / 7.64, 7.97 / 7.30, 7.90 / 7.53 | **1.049x** (1.030, 1.092, 1.049); encode 0.77 -> 0.45 s, callback −0.1 s |
| everything up to `71c0b2a` | integration | HEAD | 7.38 / 6.34, 7.64 / 6.87, 7.50 / 6.75 | 1.11x (1.164, 1.112, 1.111) |
| everything (`fd749d4`) | integration | HEAD | 8.21 / 5.87, 7.55 / 6.47, 8.37 / 6.25 | **1.34x** (1.399, 1.167, 1.339) |
| everything, int8 | integration | HEAD | 8.38 / 7.44, 8.72 / 6.37, 9.07 / 8.22 | 1.13x raw; per example 1.19x (games differ: 7 743 vs 8 151 examples) |

Harnesses: `checkpoints/perf-2x/` (gitignored): `split_compact.py` (split with the compact
graph and keep-sets as `agot train` passes them), `eq_fake.py`, `scripted_only.py`, `gc_probe.py`,
`call_curve.py`, `pool_split.py`, `spawn_cost.py`.

Micro numbers: snapshot 281–290 us (JSON) -> 198–209 (`to_python`) -> 101–128 (direct);
a per-conversion string cache was tried on top of `to_python` and was *slower* (209–215 us,
SipHash lookups cost more than they save), not committed. Scripted-only decisions:
466 -> 402 -> 381 -> 322 us. GC collections in a scripted-only run: ~1 313 -> 34.

The in-situ ratio overstates the GC fix: the harness process holds the model, torch and the
calibration buffer, so each cyclic collection scans a larger heap than a self-play worker's.
A warm 2-worker pool (`pool_split.py`, the executor `agot train` uses, 32 games, fp32):
integration 8.23 s -> HEAD 6.89 s (**1.19x**), cold pool 9.88 -> 8.66 s (worker spawn
~1.7–2.5 s per arm, `spawn_cost.py`).

### End to end (`ab_multi.py --rounds 3 --games 32`, production flags, one run7 iteration)

`python checkpoints\perf-2x\ab_multi.py --arm "R2A=<integ>" --arm "R2B=<worktree>" --rounds 3 --games 32` (int8, the production engine):

| round | R2A selfplay | R2B selfplay | ratio |
|---|---|---|---|
| 0 | 13.365 | 11.125 | 1.201 |
| 1 | 12.544 | 11.465 | 1.094 |
| 2 | 14.245 | 11.710 | 1.216 |

`seconds_selfplay` **median 1.201x** (min 1.094, max 1.216); R2A 13.37 s (12.54–14.24),
R2B 11.46 s (11.12–11.71). `examples_new` 17 235 vs 17 192 (int8, games differ).

Same with `@--leaf-backend,onnx-fp32` on both arms (identical data, see above):

| round | R2F selfplay | R2G selfplay | ratio |
|---|---|---|---|
| 0 | 14.458 | 12.475 | 1.159 |
| 1 | 14.309 | 12.755 | 1.122 |
| 2 | 10.935 | 9.628 | 1.136 |

`seconds_selfplay` **median 1.136x** (min 1.122, max 1.159).

Baseline first (`R2Z=<baseline 7952ae2>`, int8): `seconds_selfplay` **median 2.401x**
(2.451, 2.401, 2.398); R2Z 30.08 s (28.57–30.13), R2B 12.27 s (11.90–12.56). Whole iteration
`seconds` 2.094x, but that includes the 006 trainer pass inherited from the base.

`seconds_selfplay` of a one-iteration `agot train` includes the lazy spawn of both workers
(~2.5 s, `spawn_cost.py`), result pickling (~80 MB per 16-game chunk: 0.13 s pickle +
0.07 s unpickle plus the pipe) and the slower of the two workers; none of that changed, so
the end-to-end ratio is diluted relative to the per-worker 1.19x.

## Verdict

Predicted 1.2–1.35x on `seconds_selfplay`. Measured end to end **1.20x with int8**
(production engine, median of 3, min 1.09) and **1.14x with fp32** (median of 3, min 1.12);
per warm worker 1.19x; in situ 1.34x (overstated by the GC fix, see above). The target
(≥ 1.2x) is met on the production engine only at the median and not with fp32: the code has
about 1.15–1.2x here, not more. All kept changes are exact: bit-identical records under a
pure evaluator for every commit, bit-identical replay data with onnx-fp32 end to end.

| lead | verdict | evidence |
|---|---|---|
| 1. scripted greedy seats | **kept** (`be208ae`, `ae1a9af`, `94dbe85`, `fd749d4`) | scripted section 1.50 -> 0.72 s per chunk; the cost was the JSON snapshot (312 us), the cyclic GC (~60 us/decision) and a double action conversion, not the heuristic |
| 2. evaluate overhead per call | **partly kept** (native ORT run, in the bundle); mask **rejected** | ORT wrapper ~3–8 us/call over ~24 k calls; the all-ones mask is ~1 us per call to build and part of the documented protocol, so it stays |
| 3. hygiene bundle | **kept as a bundle** (`b3e3659`, `71c0b2a`): 1.049x in situ (> 3 %, > 2x the arm's 1.3 % round spread) | leaf `State` borrow + RegionCache generation + fortification bit set: encode 0.77 -> 0.45 s |
| 3. `with_node_game` move-out/re-box (+ `legal_actions()` generated twice for a fresh leaf) | **measured, reverted** | computing a freshly stepped leaf's actions inside `materialize`: prepare 1.21/1.23/1.10 -> 1.00/0.98/1.05 s (≈ −0.17 s, ~2.4 % of self-play), records identical, total ratio inside noise (0.96–1.14); below the 3 % bar |
| 3. `legal_actions().len()` for the budget | rejected on ceiling | the whole non-scripted advance section is 0.12 s (1.9 %) |
| 3. root encoded twice | rejected on ceiling | the whole record section is 0.04 s |
| 3. `march_actions(..).is_empty()` twice | not attempted | inside engine stepping (`prepare`/`finish`), touches rule code; the previous lead's measurement puts all of `prepare`'s non-stepping overhead at ~0.17 s |
| 4. fuller calls by scheduling | **kept** (`b13f67c`) | passes 45 758 -> 26 330, callbacks 32 359 -> 23 589 per chunk; callback 3.75 -> ~3.1 s at the same rows; compact fp32 call = ~40 us fixed + ~22 us/leaf (`call_curve.py`) |
| snapshot string cache | tried, not committed | 209–215 us vs 198–209 us without |

## Follow-ups

- **Remaining split per 16-game chunk (fp32 compact, in situ): callback 3.1 · prepare 1.0 ·
  scripted 0.7 · evaluate glue 0.6 · encode 0.4 · finish 0.2.** The callback is again half of
  self-play; with ~11.5 of 16 leaves per pass the remaining empty slots are the tail of games
  that end at different times.
- `--batch-games 32` (one call per worker instead of two sequential 16-game calls at 64
  games / 2 workers) should now pay more: fuller passes and one cache across both chunks.
  It does not change records under a pure evaluator; it is the run's choice, not measured here.
- The leaf-cache key hashes every leaf's full input with two NH lanes: 1.06 us per
  production leaf, ~0.32 s per chunk (~5 %). Action rows are mostly zeros; a sparse
  (position, value) universal hash with a proper variable-length construction could cut most
  of it, but needs its own collision argument.
- Worker results cross the process boundary as ~80 MB of pickled per-example objects per
  16-game chunk; returning the driver's packed arrays would remove that, but it touches the
  record/replay types (another agent's area).
- Worker spawn (~2.5 s) is paid once per `agot train` process and lands in the first
  iteration's `seconds_selfplay`; one-iteration A/Bs are diluted by it.
- The leftover legal-action conversion (13 getters per action, ~40 us per scripted decision)
  and the `RegionView` build (~60 us) are the next scripted-seat items; a Rust greedy policy
  would remove the seat's Python entirely but must reproduce its `random.Random` draws.

## Ledger rows and findings to merge

LEDGER.md:

| id | date | track | change | metric | before | after | delta | verdict | commit |
|---|---|---|---|---|---|---|---|---|---|
| 009 | 2026-11-20 | SPEED | scripted greedy seats: direct snapshot, one action conversion, no GC cycle | `split_compact.py onnx-fp32 16` scripted section | 1.50 s | 0.72 s | −52 % | kept | `be208ae` `ae1a9af` `94dbe85` `fd749d4` |
| 009b | 2026-11-20 | SPEED | per-game decision scheduling in the batch | passes / leaf calls per 16-game chunk | 45 758 / 32 359 | 26 330 / 23 589 | −42 % / −27 % | kept | `b13f67c` |
| 009c | 2026-11-20 | HYGIENE | leaf State borrow, RegionCache generation, fortification bit set, native ORT run | `split_compact.py` total, median of 3 | 7.91 s | 7.53 s | 1.049x | kept | `b3e3659` `71c0b2a` |
| 009d | 2026-11-20 | HYGIENE | fresh leaf's legal actions computed in `materialize` | prepare section | 1.18 s | 1.01 s | ~2.4 % of self-play | reverted (< 3 %) | none |
| 009 | 2026-11-20 | SPEED | (all of the above) | `ab_multi.py` `seconds_selfplay`, 32 games, int8 / fp32, median of 3 | 13.37 s / 14.31 s | 11.46 s / 12.47 s | 1.20x / 1.14x | kept | `fd749d4` |

FINDINGS.md:

- A scripted greedy decision on the Rust driver cost ~630 us, of which ~312 us was the
  JSON round trip of `Game.snapshot()` and ~60 us Python 3.14's cyclic GC collecting the
  `RustGame`/`StateView` cycle of every decision. Command: `scripted_only.py` (all seats
  greedy) under cProfile; `gc_probe.py` prints the collection count.
- With phase budgets, a fixed per-batch pass loop leaves passes ~40 % full (6.6 of 16
  leaves): decisions in one batch differ widely in budget. Scheduling each game's decisions
  independently fills them to ~72 % and is exact.
- Compact fp32 leaf call: ~40 us fixed + ~22 us per leaf on real leaves (`call_curve.py`).
- In-process split ratios overstate GC-related wins relative to a self-play worker (bigger
  heap); confirm with a warm `ProcessPoolExecutor` (`pool_split.py`).
- The leaf-cache key costs 1.06 us per production leaf (~5 % of cached self-play).
