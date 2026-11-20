# Findings — what is now known

Durable facts only, each with the command that proves it. Measured 2026-11-11 on this box
(1.4–4 GHz, see the baseline for why only interleaved ratios are trustworthy).

## Fast

- **The fused runtime is the leaf engine.** `onnx-fp32` is 2.15–2.24x the torch path at 1–2
  leaves, 1.5x at 4–8, 1.17x at 16. `checkpoints/perf-ab/ab_leaf.py`.
- **`onnx-int8` is another 1.4x on top of fp32** (2.5–3.4x versus torch at every batch size).
  It is the default for `agot train` since iteration 005 (2.35x end-to-end self-play at batch 8,
  2.44x at batch 4), and stays opt-in elsewhere: its ~4e-4 nat policy KL changes recorded games,
  while `onnx-fp32` is a round-off away from torch and is the evaluate/gauntlet default.
  Same command, `--leaf-backend onnx-int8`.
- **Quantizing the graph costs 0.40 s per iteration** (export 0.09 s + quantize 0.40 s +
  session 0.06 s, measured). Negligible at production scale, and roughly cancelled by the
  self-play win at meso scale — which is why the meso benchmark cannot show int8's 2x.
- **A full random game runs at 722 k decisions/s** with the trace off, up from 532 k before
  the `note_args` change (`cargo bench --bench clone_step`).
- **`State::clone` is ~380 ns for a 4872-byte state** — that is ~12.8 GB/s, i.e. memcpy-bound,
  not allocation-bound. There is no cheap 2x there.
- **`torch._dynamo` costs 1.48 s** and is imported by the first `torch.optim` optimizer, not
  by `import torch` (`python -c "import torch; import torch._dynamo"`). It cannot be avoided:
  `torch/optim/optimizer.py:_use_grad` imports it unconditionally inside the first
  `optimizer.step()`, and `Optimizer.add_param_group` is wrapped by
  `torch._compile._disable_dynamo` on the way in. Pre-seeding that wrapper's
  `__dynamo_disable` cache skips the constructor import but not the step's, so the only lever
  is to overlap the import with self-play (iteration 004).

## Slow

- **Leaf inference is 91–94 % of a self-play step.** Instrumented evaluator callback, 8 games
  with 16 simulations: 29.8 s of 31.8 s at batch 2, 67.5 s of 73.8 s at batch 8.
- **The leaf forward at small batches is dispatch-bound.** A 15-candidate leaf is ~11 MFLOP ≈
  40 µs at this CPU's ~125 GFLOP/s single-thread f32; measured 150–200 µs before, 88–105 µs
  with the fp32 graph. Nothing in Python closes the rest.
- **Per-leaf cost falls 2.4x from batch 1 to batch 16** (torch: 525 → 107 µs/leaf). `batch_games`
  is the single largest knob in the driver, and bigger than any code change in this ledger.
- **The engine is not the bottleneck for training**: self-play is ~93 % network. A 1.36x
  engine win shows up on `agot demo`, the gauntlet and the rules tests, not on training.

- **Adam moments of dead ReLU units live in the float32 subnormal range forever** and made
  the optimizer step 2-6x slower. A dead unit's gradient is exactly zero, the EMA
  `m*beta` of a subnormal rounds back to itself, and run7's `agot.state.pt` was 73 %
  subnormal (761 900 of 1 046 542 entries). Activations and gradients have none (50 run7
  batches, every Linear). Fixed by FTZ during the gradient pass plus a per-epoch scrub
  (iteration 006); the scrub is the portable half, because GNU OpenMP (Linux/Colab wheels)
  does not propagate the FP flags to an existing pool, Intel OpenMP (Windows) does.
- **MKL's vector `sqrt` is slow on exact zeros, not only on subnormals, and FTZ/DAZ do not
  help it**: 131 072 floats, 1 thread: normal 20 µs, zeros 170 µs, subnormal 860 µs. The
  unfused AdamW calls it every step on second moments that are mostly zero; the fused CPU
  kernel does not (optimizer step 4x, bit-identical; iteration 006).
- **70-76 % of the action trunk's ReLU columns are zero for a whole run7 batch** (state
  trunk 75-79 %, value hidden 99 %, policy hidden 41 %). Feeding the next matmul only the
  live columns makes the training backward 1.36-1.40x faster (iteration 006).

## Wrong / corrected beliefs

- **"The planning phase is the heavy legal-action path, so that is where the engine loses
  time."** The planning line did not move in either direction across builds (1.98–2.30 M/s,
  noise). The cost that mattered was eager trace formatting on the *full game*, which the
  planning-only bench never exercised.
- **"ONNX export costs 0.6 s per iteration, too much for small runs."** The export is 0.31 s
  and session creation 0.10 s cold / 0.006 s warm; with one export per iteration it is 5 % of
  a 7 s meso and 0.5 % of a production iteration. `do_constant_folding=False` makes the export
  0.05 s but slows every inference by 7 %, so the export is not the place to save time.
- **"Rebuilding the ONNX session per batch is a per-batch cost"** (backlog item 8). It is not:
  the CLI prepares the graph once per iteration in the parent and workers get a cache hit. The
  only real bug was the *first* iteration of a fresh run, where the checkpoint does not exist
  yet and every worker exported concurrently. Fixed in 003.
- **"int8 is fine on small batches below ~1500 rows"** (prompt §8) — measured the opposite:
  int8 is 3.4x at 2 leaves and 2.5x at 16. The real cost of int8 is numeric (KL 4e-4 on a
  trained net), not a small-batch cliff.

- **"Training on run7 is reproducible bit for bit only on the same code."** It is not even
  that: changing the torch thread count (1 vs 2) moves the weights by up to 0.11 after 300
  AdamW steps and the epoch loss in the 5th digit, because Adam turns round-off in near-zero
  gradients into lr-sized steps. That is the floor any summation-order change is judged
  against (iteration 006, `bench_epoch.py` with `BENCH_THREADS`).
- **`Optimizer.load_state_dict` restores `fused`/`foreach` from the file** and keeps the
  caller's moment tensors when dtype/device already match (two optimizers loaded from one
  dict share their state). Both bit an A/B harness here; `Trainer.load_state_dict` re-applies
  the kernel choice.
- **numpy's float32 `sum` is not reproducible by `np.add.reduceat` or `cumsum`** (pairwise,
  SIMD-blocked). Bit-identical vectorised collation has to keep the per-example `.sum()`.

## Unmeasurable on this box

- **A ~10 % effect in a whole-epoch subprocess A/B**: rounds swing 0.87-1.38x from clock drift
  alone. Same-process, batch-interleaved arms (two trainer copies, A,B / B,A each batch) are
  stable to ~2 % and A-vs-A is bit-identical (iteration 006).
- Anything that needs more than ~2 minutes of sustained CPU: the clock collapses and the last
  measurement of a run is not comparable to the first. All numbers here come from interleaved
  runs or a reference-GEMM-normalised harness.
- Wall-clock comparisons against the documented §6.1 table (recorded on a different machine
  state). The numbers in `iterations/000-baseline.md` were re-measured today.

## Self-play and the iteration around it (2026-11-20, iterations 007–010)

- **~73 % of the leaves of one production `selfplay()` call repeat an input already scored in
  that call**, and the repeats are recent (an LRU of 2 000 reaches the same hit rate). Every
  decision restarts its tree, and the 97-float state carries no board, so many positions encode
  to the same network input. `checkpoints/perf-2x/dup_study.py 16`; counters in
  `_core.selfplay_stats()`.
- **Trained nets here have 65–80 % dead trunk units** (run7: 202/173/173/165 of 256 never fire
  on the whole replay buffer; run5 and run6 look the same) **and a nearly dead value head**
  (251 of 256 hidden units). Dropping the dead trunk units from the leaf graph is exact (fp32
  self-play games identical to the full graph) and makes a leaf call 2.6x cheaper in fp32 and
  1.6x in int8. `python -m agot.nn.onnx_backend <ckpt> --leaves <leaves.npz> --replay
  <agot.replay.npz>`. Whether the dead units cost playing strength has not been tested.
- **Head hidden layers must not be pruned this way**: replay-dead head units fire on real leaves.
- **Calibration needs coverage of games, not rows**: 33k contiguous examples miss units that
  fire on leaves, 45k+ do not; random 10k subsets are nearly clean. A margin is a bad substitute.
- **int8 is not round-off on real leaves.** Full int8 against torch on run7's leaves: value
  |diff| mean 0.059, max 1.06; policy KL mean 1.05e-3, p95 4.0e-3, max 1.35e-2 (1 581 calls).
  The same leaf scored in two different batches differs by up to 1.24 in a value
  (`cache_eq_rust.py onnx-int8 16 drift`). ORT fp32 is row-invariant for 99.8 % of repeated
  leaves, the rest differ by ≤ 3e-7. Iteration 005 only ever measured the policy KL on a
  synthetic net.
- **With the cache and the compact graph, fp32 leaves cost about what int8 leaves do**: whole
  iteration 2.30x against 2.33x over the old int8 baseline (`ab_iters.py`, 5 rounds).
- **A scripted greedy decision on the Rust driver cost ~630 µs**, of which ~312 µs was the JSON
  round trip of `Game.snapshot()` and ~60 µs Python 3.14's cyclic GC collecting the
  `RustGame`/`StateView` cycle. `scripted_only.py` under cProfile; `gc_probe.py`.
- **With phase budgets a fixed per-batch pass loop leaves passes ~40 % full** (6.6 of 16
  leaves). Scheduling each game's decisions independently fills them to ~72 % and is exact.
- **Replay candidate rows are ~95 % repeats** (run7: 54k distinct of 1.0–1.3M); calibration
  over distinct, bit-verified rows gives bit-identical per-unit peaks at a third of the cost.
- **The time outside both phase timers is attributed in the log** (`seconds_leaf_calibration`,
  `seconds_leaf_graph`, `seconds_extend`, `seconds_checkpoint`, `seconds_replay_save`,
  `seconds_replay_wait`, `seconds_replay_snapshot`, `seconds_trainer_restore`).
- **`ab_multi.py` (one iteration per process) hides work moved after the log record**; use
  `ab_iters.py --iters 2` and read the second iteration for anything that overlaps phases.
- **Training is not bit-reproducible on this box under load**: with identical examples the
  losses differ in the 5th–6th digit about once in eight runs, on unchanged code too.

## Not measured

- Nothing in iterations 006–010 was run on the 2-vCPU Colab runtime. The subnormal penalty, the
  thread flags and the contention between the background snapshot and two self-play workers
  all depend on that CPU; the new log timers will show them on the next run.
