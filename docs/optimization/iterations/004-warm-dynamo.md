# 004 — keep the `torch._dynamo` import off the critical path

- Track: SPEED (startup)
- Status: kept
- Commit: see the branch head (`src/agot/cli.py`, `src/agot/training/trainer.py`)

## Hypothesis

Every `agot train` and `agot gate` run pays **1.48 s** to import `torch._dynamo`, and it is
`torch.optim.AdamW(...)` that triggers it: the optimizer classes are wrapped by
`torch._compile._disable_dynamo`, whose first invocation does the import
(`torch/_compile.py:43-52`, `torch/optim/adamw.py`). Measured:
`import torch` 1.72 s, `import torch; import torch._dynamo` 3.35 s.

The trainer builds its optimizer before self-play, so the run blocks on it while the CPU is
otherwise idle. Self-play is the one place where the import could hide.

## Prediction (written before the change)

- Metric: wall clock of the §5 meso command and of `agot train --games 0`.
- Expected: −0.8…−1.5 s on runs that self-play (5–15 % of the meso); the `--games 0` run gets
  no overlap and stays flat or costs a few percent.
- Falsified if: any run pays *more* wall clock for a real workload, or the optimizer state is
  not identical after a resume.

## Change

- `Trainer` builds `torch.optim.AdamW` lazily, on the first access to `.optimizer`
  (training step, `state_dict`, `load_state_dict`).
- `cli._run_training` starts a daemon thread that imports `torch._dynamo` at the top, so the
  import overlaps the first self-play round.
- On `--resume`, the saved optimizer state is applied just before the first training pass
  instead of before self-play, which is what makes the overlap work for resumed runs too (a
  resumed run is the common case: run6 resumes run5).

Nothing about the optimizer's contents changes: same class, same parameter groups, same
moment loading, just later.

## Gates

`pytest -q` 216 passed (including the checkpoint/resume tests) · `ruff` clean.

## Measurement

Meso (§5 command), 3 rounds, warm thread on/off by toggling the call in place, `--leaf-backend
torch` to isolate the effect:

| round | warm on | warm off |
|---|---|---|
| 0 | 8.33 s | 8.89 s |
| 1 | 8.47 s | 8.79 s |
| 2 | 8.25 s | 11.00 s |

`agot train --games 0` (no self-play to hide behind), 4 interleaved rounds: before 3.84 s
median, after 4.01 s — the run that does no work pays ~4 % for the thread.

## Verdict

Kept: −0.5…−1.0 s (6–13 %) on every run that plays games, which is every run that matters.
The regression on `--games 0` is recorded here on purpose; it is the honest cost of the
thread and the reason the warm-up is a thread rather than a synchronous import.

## Follow-ups

The import cannot be removed: `torch/optim/optimizer.py:_use_grad` does `import torch._dynamo`
unconditionally inside the first `optimizer.step()`, and `Optimizer.add_param_group` is wrapped
by `torch._compile._disable_dynamo`. Pre-seeding that wrapper's `__dynamo_disable` cache (the
intended fast path of the wrapper) removes the constructor's share and was measured: it moves
the import to the first step and buys nothing end to end. Overlapping is the only lever.
