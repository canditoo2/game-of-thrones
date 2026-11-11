# Iteration 000 — baseline

- Track: baseline
- Status: n/a
- Commit: `b01c99a` (branch `optimization/speed-leaf-engine`)

## Environment

| item | value |
|---|---|
| Python | 3.14.6 (MSC v.1944, 64 bit) |
| torch | 2.14.0+cpu |
| onnxruntime | 1.31.1 |
| cargo / rustc | 1.99.1 / 1.99.1 |
| logical CPUs | 22 (Alder Lake, 1.4 GHz base, boosts under short load) |
| tree | clean except the untracked prompt document |

**Machine caveat, and it matters for every number below.** The box free-runs between
1.4 GHz and ~4 GHz. A 10 s benchmark can come back 2.5x slower two minutes later with no
change to the code. Absolute numbers are therefore only comparable inside one session, and
every A/B in this ledger is interleaved (A, B, A, B...) with a median over rounds. The leaf
benchmark additionally normalises each round against a fixed reference GEMM
(`checkpoints/perf-ab/ab_leaf.py`).

## Gates (all green, before any change)

| gate | command | result |
|---|---|---|
| Python suite | `.venv\Scripts\python -m pytest -q` | 216 passed, 1 skipped, **36.0 s** |
| lint | `.venv\Scripts\ruff check src tests` | clean |
| rust fmt | `cargo fmt --all -- --check` | clean, 0.4 s |
| rust clippy | `cargo clippy --all-targets -- -D warnings` | clean |
| rust clippy (py) | `cargo clippy -p agot-py --all-targets -- -D warnings` | clean |
| rust tests | `cargo test --release` | 292 passed (232+38+3+3+4+12), 1.3 s warm |

## Benchmarks (three repeats, median unless noted)

| metric | command | baseline |
|---|---|---|
| `State::clone` | `cargo bench --bench clone_step` | 389–414 ns |
| `Game::clone` | same | 422–445 ns |
| `legal_actions` (opening, 45) | same | 355–422 ns |
| `clone + step` | same | 509–540 ns |
| planning throughput | same | 2.41–2.47 M decisions/s |
| full game, trace off (new line) | same | 531 k–533 k decisions/s |
| full game, trace on (new line) | same | 430 k–455 k decisions/s |
| leaf forward, torch, 1/2/4/8/16 leaves | `checkpoints/perf-ab/ab_leaf.py` | 525 / 397 / 601 / 1015 / 1707 µs |
| leaf forward, ORT fp32, same | same | 243 / 177 / 391 / 691 / 1442 µs |
| leaf forward, ORT int8, same | same | 165 / 118 / 202 / 337 / 674 µs |
| self-play, 8 games / 16 sims / batch 4 | `checkpoints/perf-ab/ab_selfplay.py 8 16 4` | 9.71 s torch (1.53x fp32, 2.44x int8) |
| meso iteration | §5 command, 2 iterations | **9.78 s** wall; `seconds_selfplay` 2.67+2.46, `seconds_train` 0.22+0.32 |
| CLI startup floor | `.venv\Scripts\python -c "import agot.cli"` / `import torch` | 0.22 s / 1.72 s |
| `torch._dynamo` import | `python -c "import torch; import torch._dynamo"` | +1.48 s |
| no-work training run | `agot train --games 0` | 3.77–3.84 s |

## Where the time goes (measured, not assumed)

- A self-play step is **91–94 % leaf inference**: a callback probe around the driver's
  evaluator measured 29.8 s of 31.8 s (batch 2) and 67.5 s of 73.8 s (batch 8).
- The leaf forward at 1–2 leaves runs at **2.5–3x its arithmetic floor**. A 256x256 f32 GEMM on
  this CPU sustains ~125 GFLOP/s, so a 15-candidate leaf is ~40 µs of arithmetic against
  150–200 µs measured; the rest is per-op dispatch.
- A planning-phase step is dominated by `legal_actions`, and a *full game* additionally pays the
  eager `format!` at 73 `state.note(...)` call sites even though search runs with `log == None`.

## Ambiguities (§6.3)

1. `--determinizations 1` semantics: not resolved here; nothing in this ledger depends on it.
2. Policy-loss §2.0 contradiction: not used; this ledger reports no loss numbers.
3. run6 projection: not used; no extrapolation.
4. Stale docs: the counts in §6.1 of the prompt lag the code (292 rust tests, 216 python tests).
   This file records what was measured today, not what the docs claim.

## Backlog selected for this session

The measurements above pick the levers themselves: the eager trace formatting (every step),
the per-op dispatch in the leaf forward (90 % of every run), the leaf *engine* (a fused runtime
exists and is barely used) and the 1.5 s `torch._dynamo` import paid by every training run.
