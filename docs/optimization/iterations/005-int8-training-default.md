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
run than during 003's (absolute numbers moved from 10 s to 30 s), so only the paired ratios are
readable — which is exactly why the harness is interleaved.

| round | before | after | ratio | sp before | sp after | sp ratio |
|---|---|---|---|---|---|---|
| 0 | 23.75 s | 19.63 s | 1.21x | 16.74 s | 11.65 s | 1.44x |
| 1 | 29.86 s | 17.72 s | 1.69x | 22.25 s | 10.87 s | 2.05x |
| 2 | 36.13 s | 23.32 s | 1.55x | 23.30 s | 13.68 s | 1.70x |
| 3 | 34.58 s | 22.55 s | 1.53x | 22.16 s | 13.25 s | 1.67x |
| 4 | 34.00 s | 22.94 s | 1.48x | 21.70 s | 13.38 s | 1.62x |

Median wall **1.53x** (mean 1.49x), median `seconds_selfplay` **1.67x** (mean 1.70x).
The same harness with the fp32 default measured 1.42x / 1.60x, so int8 is not a regression at
meso scale and is a large win above it.

Self-play at a production-shaped batch (8 games / 16 sims / batch 8, Rust driver):

| engine | wall | decisions/s | vs torch | same records |
|---|---|---|---|---|
| torch | 31.35 s | 147 | 1.00x | — |
| onnx-fp32 | 24.92 s | 184 | 1.26x | yes |
| onnx-int8 | 13.33 s | 342 | **2.35x** | no |

## Verdict

Kept. The training default is now the engine the project already trains with, and the measured
2.35–2.44x on self-play is where the 2x actually lives: the meso benchmark cannot show it
because 2.9 s of its ~10 s is process startup and the remaining self-play window is short
enough that int8's one-time quantization offsets its win.

The learning-signal cost is stated, not hidden: int8 moves the leaf policy by ~4e-4 nats (mean)
on a trained net and can flip a close action choice, so recorded games differ from a torch run
of the same seed. `--leaf-backend torch` or `onnx-fp32` restores the old numerics exactly;
`onnx-fp32` is still 1.5–2.2x faster than torch.

## Follow-ups

The experiment card `experiments/001-fp32-leaf-at-run6-scale.md` now has a third arm to run on
Colab: it is the only way to confirm the 2.3x at 64 games × 128 sims with real phase budgets.
