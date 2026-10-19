# AGENTS.md

AlphaZero-style RL agent for **A Game of Thrones: The Board Game (2E)**, six players: a
numpy-only rules engine + PUCT MCTS (`agents/`), an optional PyTorch scorer network (`nn/`) and
the self-play training loop (`training/`). Everything is driven by the `agot` CLI.

**Current direction (2026-10-09):** the pipeline was rebuilt after two void runs (below); the
first credible training run (`run3`) has not happened yet. Heavy training is meant for a separate
machine — this PC is the build-and-verify machine.

## Commands (Windows/PowerShell — invoke the venv explicitly)

- Tests: `.venv\Scripts\python -m pytest -q` (172 tests). When piped in PowerShell the final
  summary line can be swallowed — trust `$LASTEXITCODE` (0 = pass).
- One file: `.venv\Scripts\python -m pytest tests/test_nn.py -q`
- Lint: `.venv\Scripts\ruff check src tests` (line-length 100, py310 target).
- CLI: `.venv\Scripts\agot <demo|legal|evaluate|gauntlet|round-robin|train|report>`. One full
  game is ~0.1 s: `agot demo --policy greedy --seed 1`. Gauntlet = single-seat win rate vs a
  field; any policy argument accepts `net:<checkpoint>` (MCTS over that saved network), and
  `round-robin` builds a league table across checkpoints.
- Most of the suite runs without torch: only `test_nn` and `test_checkpoint` skip entirely
  (module-level `importorskip`), plus a few individual cases in `test_batched_selfplay` and
  `test_report`. CI (`.github/workflows/ci.yml`) runs the numpy-only suite on Python 3.10/3.12
  plus an allowed-to-fail torch job; the local venv is Python 3.14 — don't assume CI parity.
- Rust core (engine port in progress, `rust/`): run from `rust/` with the stable
  `x86_64-pc-windows-gnu` toolchain (not on `PATH`; use `%USERPROFILE%\.cargo\bin\cargo`).
  `cargo test --release`, `cargo clippy --all-targets -- -D warnings`,
  `cargo bench --bench clone_step`. `tools\dump_fixture.py` regenerates the differential
  fixture the parity tests replay — regenerate it from Python, never hand-edit it. The
  other generators/debug aids: `tools\dump_encoder_golden.py` (bit-exact encoder vectors),
  `tools\debug_parity.py` (print the reference position at a decision),
  `tools\debug_step.py` (print the actions around a decision),
  `tools\diff_state.py` (diff a Rust dump against a Python one).
- The Python extension (`rust/crates/agot-py`, PyO3 → the `agot_core_ext` module) is **not** in
  the default workspace build, so the engine's own loop needs no PyO3 toolchain:
  `cargo build -p agot-py`, or `maturin build --release --out dist` from
  `rust/crates/agot-py`. On Windows that needs binutils (`dlltool`) on `PATH` — WinLibs'
  UCRT build — plus `PYO3_PYTHON` pointing at the venv interpreter. The wheel is abi3
  (Python ≥3.10), so one build covers the local venv, CI and Colab. `tests/test_rust_engine.py`
  skips itself when the module is missing.
- Training self-play already runs on the Rust engine when the extension is installed:
  `agot.training.rust_engine` adapts it, `selfplay.batched_self_play` takes that path whenever
  it can (falling back to the Python driver for scripted opponent seats) and records which ran
  in `selfplay.LAST_BACKEND`. The rules engine itself (`agot.engine`) is still the Python one:
  switching it over needs a view tier over the Rust state and is the remaining cutover step.
- `checkpoints/` and `Rulebook/` are gitignored (run artifacts; the rulebook scans are
  copyrighted and must not be committed).

## Architecture constraints (easy to break)

- Layering: `engine/` is numpy-only and must stay importable without torch; never import torch
  from `engine/` or `agents/`.
- The Rust port (`rust/crates/agot-core`) is the future home of the engine, search and encoder.
  `agot-core` must never depend on torch or PyO3 (the Python binding is a separate crate), and
  it must not reach for the network. While the port runs, the Python engine is the reference
  oracle: ported behaviour is pinned by the differential fixture, and a disagreement is a bug in
  the Rust side (or in the fixture's generator) — never something to fix by editing the fixture.
- Copy-on-write regions: every write to a `RegionState` must go through `GameState.mut(id)`;
  direct writes silently corrupt clones and search. `tests/test_clone_isolation.py` pins this —
  run it after any state-mutation change.
- `Game.step` applies one player decision and auto-advances every phase that needs no input;
  legal actions are non-empty only when a player must decide.
- MCTS stores a value *vector* per house (multiplayer; no sign flipping). The policy head is a
  scorer MLP over `[h, a_i]` — deliberately NOT a dot product: a product has a zero fixed point
  that killed runs 1-2, and `PolicyValueNet.load` refuses those checkpoints. Don't "simplify" it
  back.
- Training uses AdamW with decoupled decay on weight matrices only (`trainer._parameter_groups`);
  plain Adam + L2 weight decay is what ground the old policy head to zero.
- Self-play defaults are load-bearing: 128 simulations / 2 determinizations (PIMC) and a
  temperature schedule (1.0 for 3 rounds, then 0.25). The recorded policy target is built from
  the temperature, so flat 1.0 or too few sims trains on near-uniform noise.
- Hidden information: `engine/hidden.py` resamples face-down orders and secret bids (PIMC);
  observation masks hidden orders. Known remaining leak: the first chooser's House card in a
  battle (revealed simultaneously in the physical game).

## Training runs

- **Do not start training runs on this machine unless the user asks** (user directive): it is
  CPU-only (no CUDA), and heavy runs belong on a separate machine. Prepare/verify here, run there.
  For Google Colab, start from `colab/agot-colab.ipynb` (private-repo token as a Colab secret
  `GITHUB_TOKEN`; checkpoints on Drive; the launch cell auto-resumes).
- `checkpoints/run1` and `run2` are **void**: trained with the old broken policy head (policy loss
  flat at ~2.0 = uniform over candidates). Don't resume them; the loader refuses them.
- A run writes `agot.pt` (weights), `agot.state.pt` (iteration/best-eval/optimizer for
  `--resume`), `agot.log.jsonl` (one line per iteration) and `agot-best.pt` (best gauntlet eval).
  The replay buffer is deliberately not saved — a resumed run refills it and shows a transient
  loss spike.
- Smoke gate before any long run: policy loss must fall **below ~2.0** and keep falling. Flat
  ~2.0 is the old failure signature, not convergence.
- Launch/monitor/resume commands: README "Quickstart". `agot report --checkpoint <ckpt>
  --watch 15 --open` renders the log to a self-contained HTML page.

## Docs are part of the deliverable

- `docs/roadmap.md` — plan of record and honest gap list; update statuses when features land.
- `docs/methods.md` — decision record + measured baselines (greedy = 38.3% vs 5×random, 23.3% vs
  5×greedy; parity = 1/6 ≈ 16.7%).
- `PROVENANCE.md` + `verified` flags in `src/agot/data/*.json` — board-data confidence; data
  fixes need no engine code.
- `docs/rulebook-audit.md` — rule-fidelity audit trail; `tests/test_rules_fixes.py` holds one
  regression per audited rule fix (add one when fixing a rule).
