# 005 — train on the int8 leaf engine (the one run6 already uses)

- Track: SPEED (with an explicit, measured learning-signal note)
- Status: kept
- Commit: branch head (`src/agot/training/rust_engine.py`, `src/agot/cli.py`)

## Hypothesis

Iteration 003 left `auto` = onnx-fp32 for every command. That is the right engine for
evaluation (it is a round-off away from torch) but the wrong one for training: the leaf batch
in a training run is large and long-lived, and int8 is the engine the project's own next run
selects (`AGENTS.md`, run6). Iteration 003 measured int8 at 2.44x the torch path end to end
(8 games / 16 sims / batch 4) against 1.53x for fp32, so making it the *training* default is
the difference between a 1.5x and a 2.4x self-play loop.

## Prediction (written before the change)

- Metric: meso wall clock (5 interleaved rounds against `b01c99a`) and self-play at a
  production-shaped batch (`ab_selfplay.py 8 16 8`).
- Expected: self-play 2.2–2.5x; meso wall 1.4–1.6x — the meso pays ~0.45 s/iteration more for
  `quantize_dynamic` (measured: export 0.09 s, quantize 0.40 s, session 0.06 s) than fp32, and
  saves ~0.35 s/iteration in self-play, so the two nearly cancel at that scale.
- Falsified if: meso wall regresses below the fp32 default (1.42x), or the int8 KL budget test
  stops passing.

## Change

- `resolve_leaf_backend(..., training=True)` returns `onnx-int8` for `auto`; every other
  command keeps `onnx-fp32` (evaluate, gauntlet) or inherits from the run (`gate`).
- `cmd_train` passes `training=True`.
- Tests pin the new resolution on both paths.

The graph is still built **once per iteration**, in the parent, before the games
(`_prepare_leaf_graph`), so quantization is paid once and every worker gets a cache hit. That
was already true for fp32 in 003; int8 rides the same path.

## Gates

`pytest -q` 216 passed (13 in `test_onnx_backend.py`, including
`test_int8_policy_kl_stays_inside_the_budget`: mean KL < 1e-3, p95 < 5e-3) · `ruff` clean ·
`cargo` gates untouched · a real `agot train` writes `i8def.onnx-int8.onnx` beside its
checkpoint, which is how the default was verified.

## Measurement

Meso (§5 command), 5 interleaved rounds against `b01c99a`. The box was ~3x slower during this
run than during 003's (absolute numbers moved from 10 s to 20–36 s), so only the paired ratios
are readable — which is exactly why the harness is interleaved.

| round | before | after | ratio | sp before | sp after | sp ratio |
|---|---|---|---|---|---|---|
| 0 | 19.35 s | 13.00 s | 1.49x | 10.71 s | 4.61 s | 2.32x |
| 1 | 19.75 s | 12.74 s | 1.55x | 11.31 s | 4.61 s | 2.46x |
| 2 | 27.11 s | 23.12 s | 1.17x | 17.02 s | 10.32 s | 1.65x |
| 3 | 36.62 s | 21.04 s | 1.74x | 23.68 s | 9.38 s | 2.53x |
| 4 | 25.86 s | 13.43 s | 1.93x | 15.85 s | 5.03 s | 3.15x |

Median wall **1.55x** (mean 1.58x), median `seconds_selfplay` **2.46x** (mean 2.42x).

A 4-iteration meso (same command, `--iterations 4`, 3 rounds) confirms the split: wall
28.44–35.58 s → 19.05–19.67 s (**1.53x**), `seconds_selfplay` 18.70–23.13 s → 7.11–7.58 s
(**2.69x**). The after-wall decomposes as 5.2 s of process startup (measured at that clock:
`import torch` 3.69 s, `import torch._dynamo` another 2.84 s, overlapped), ~2 s of
`quantize_dynamic` over 4 iterations, ~2 s of training, and 7.5 s of self-play. Startup and
quantization are the two terms no leaf-engine change can move, which is why the wall ratio sits
below the self-play ratio on a micro-run and converges to it on a real one.

The wall ratio is smaller than the self-play ratio for a measured reason: the meso is 2
iterations, so ~30 % of its before-wall is process startup (imports alone are 1.94 s, and
`torch._dynamo`'s 1.48 s can only be overlapped, see 004) and int8 adds ~0.55 s per iteration
of `quantize_dynamic`. Amortise those over a real iteration and the loop itself runs at the
self-play ratio: `seconds_train` is unchanged, so a steady-state iteration is ~2.3x faster.

Self-play at a production-shaped batch (8 games / 16 sims / batch 8, Rust driver):

| engine | wall | decisions/s | vs torch | same records |
|---|---|---|---|---|
| torch | 31.35 s | 147 | 1.00x | — |
| onnx-fp32 | 24.92 s | 184 | 1.26x | yes |
| onnx-int8 | 13.33 s | 342 | **2.35x** | no |

## Verdict

Kept. The training default is now the engine the project already trains with, and self-play —
91–94 % of a training iteration — runs **2.46x** faster on the meso and **2.35x** faster at a
production batch size. The end-to-end meso wall is 1.55x because it is startup-dominated; the
ledger records both numbers rather than picking the flattering one.

The learning-signal cost is stated, not hidden: int8 moves the leaf policy by ~4e-4 nats (mean)
on a trained net and can flip a close action choice, so recorded games differ from a torch run
of the same seed. `--leaf-backend torch` or `onnx-fp32` restores the old numerics exactly;
`onnx-fp32` is still 1.5–2.2x faster than torch.

## Follow-ups

The experiment card `experiments/001-fp32-leaf-at-run6-scale.md` now has a third arm to run on
Colab: it is the only way to confirm the 2.3x at 64 games × 128 sims with real phase budgets.
