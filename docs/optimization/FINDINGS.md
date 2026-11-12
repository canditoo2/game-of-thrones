# Findings — what is now known

Durable facts only, each with the command that proves it. Measured 2026-11-11 on this box
(1.4–4 GHz, see the baseline for why only interleaved ratios are trustworthy).

## Fast

- **The fused runtime is the leaf engine.** `onnx-fp32` is 2.15–2.24x the torch path at 1–2
  leaves, 1.5x at 4–8, 1.17x at 16. `checkpoints/perf-ab/ab_leaf.py`.
- **`onnx-int8` is another 1.4x on top of fp32** (2.5–3.4x versus torch at every batch size).
  Same command, `--leaf-backend onnx-int8`.
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

## Unmeasurable on this box

- Anything that needs more than ~2 minutes of sustained CPU: the clock collapses and the last
  measurement of a run is not comparable to the first. All numbers here come from interleaved
  runs or a reference-GEMM-normalised harness.
- Wall-clock comparisons against the documented §6.1 table (recorded on a different machine
  state). The numbers in `iterations/000-baseline.md` were re-measured today.
