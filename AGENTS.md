# AGENTS.md

AlphaZero-style RL agent for **A Game of Thrones: The Board Game (2E)**, six players: a Rust
rules engine, search and encoder (`rust/`) behind a thin Python facade, an optional PyTorch
scorer network (`nn/`) and the self-play training loop (`training/`). Everything is driven by
the `agot` CLI.

**Current direction (2026-10-21):** `run3` is training on Colab (checkpoints under
`/content/drive/MyDrive/agot/run3`); this PC remains the build-and-verify machine and must not
start heavy runs unless asked. The pipeline was rebuilt after the two void runs (below), so
watch the policy loss and the best-checkpoint eval rather than assuming the last checkpoint is
strongest.

## Commands (Windows/PowerShell — invoke the venv explicitly)

- Tests: `.venv\Scripts\python -m pytest -q` (116 tests; the rules/combat/encoder
  regressions live in Rust — see `cargo test --release` below). When piped in PowerShell
  the final summary line can be swallowed — trust `$LASTEXITCODE` (0 = pass).
- One file: `.venv\Scripts\python -m pytest tests/test_nn.py -q`
- Lint: `.venv\Scripts\ruff check src tests` (line-length 100, py310 target).
- CLI: `.venv\Scripts\agot <demo|play|legal|evaluate|gauntlet|round-robin|train|report>`. One
  full game is ~0.2 s: `agot demo --policy greedy --seed 1`. `agot play --checkpoint <dir|pt>
  --house stark` opens the local Tkinter desktop game: the setup screen chooses the human
  house, opponent and checkpoint, then the full board and action list are shown. When
  `Rulebook/board.jpg` is present (Pillow installed), the game uses the local scan as the
  board; otherwise it falls back to the schematic region board. Desktop play enables
  `deferred_battles`, so all March destinations are chosen before the one battle starts;
  training/evaluation keep the legacy immediate-battle default for the frozen fixtures.
  AI seats default to 64 simulations over 2 determinizations (2 keeps face-down orders and bids
  hidden), and `--opponents greedy` is the fast fallback. Gauntlet = single-seat win rate vs
  a field; any policy argument accepts `net:<checkpoint>` (MCTS over that saved network), and
  `round-robin` builds a league table across checkpoints.
- Most of the suite runs without torch: only `test_nn` and `test_checkpoint` skip entirely
  (module-level `importorskip`), plus a few individual cases in `test_batched_selfplay` and
  `test_report`. CI (`.github/workflows/ci.yml`) installs a Rust toolchain and builds the
  extension, then runs the suite on Python 3.10/3.12 plus an allowed-to-fail torch job; the
  local venv is Python 3.14 — don't assume CI parity.
- Rust engine (`rust/`): run from `rust/` with the stable `x86_64-pc-windows-gnu` toolchain
  (not on `PATH`; use `%USERPROFILE%\.cargo\bin\cargo`). Gates:
  `cargo test --release`, `cargo fmt --all -- --check`,
  `cargo clippy --all-targets -- -D warnings`,
  `cargo clippy -p agot-py --all-targets -- -D warnings`, `cargo bench --bench clone_step`.
  The differential fixture (`crates/agot-core/tests/data/engine_fixture.json`) and the encoder
  golden file are **frozen regression pins** — the Python generators that produced them were
  retired with the Python engine, so never hand-edit either file.
- Packaging: `pyproject.toml` uses maturin; `pip install -e .` builds the abi3 extension into
  `agot._core`. `rust/crates/agot-py` is not a default workspace member, so the engine's own
  `cargo test` loop needs no PyO3 toolchain. On Windows a direct `maturin build` needs binutils
  (`dlltool`) on `PATH` — WinLibs' UCRT build — plus `PYO3_PYTHON` pointing at the venv
  interpreter; plain `pip install -e .` works with the toolchain on `PATH` as it is here. The
  wheel is abi3 (Python ≥3.10), so one build covers the local venv, CI and Colab.
- Self-play: `agot.training.rust_engine` drives the batch tier and
  `selfplay.batched_self_play` takes it whenever the evaluator is model-backed and the configs
  share one game configuration, recording the path in `selfplay.LAST_BACKEND`. Scripted seats
  run on the Rust driver too (one Python policy call per scripted decision, search seats stay
  native). The reference driver remains only for hand-written evaluators and mixed configs.
- `checkpoints/` and `Rulebook/` are gitignored (run artifacts; the rulebook scans are
  copyrighted and must not be committed).

## Architecture constraints (easy to break)

- Layering: the Python `engine/` facade is numpy-only and must stay importable without torch;
  never import torch from `engine/` or `agents/`.
- `rust/crates/agot-core` is the engine, search and encoder. It must never depend on torch or
  PyO3 (the Python binding is a separate crate), and it must not reach for the network. The
  differential fixture and encoder golden vectors are frozen regression pins: a disagreement is
  a bug in the Rust side, never something to fix by editing a fixture or a golden file.
- `Game.step` applies one player decision and auto-advances every phase that needs no input;
  legal actions are non-empty only when a player must decide. `RustGame` caches its snapshot
  view per step (`StateView._version`); any new mutation path must invalidate it.
- MCTS stores a value *vector* per house (multiplayer; no sign flipping). The policy head is a
  scorer MLP over `[h, a_i]` — deliberately NOT a dot product: a product has a zero fixed point
  that killed runs 1-2, and `PolicyValueNet.load` refuses those checkpoints. Don't "simplify" it
  back.
- Training uses AdamW with decoupled decay on weight matrices only (`trainer._parameter_groups`);
  plain Adam + L2 weight decay is what ground the old policy head to zero.
- Self-play defaults are load-bearing: 128 simulations / 2 determinizations (PIMC) and a
  temperature schedule (1.0 for 3 rounds, then 0.25). The recorded policy target is built from
  the temperature, so flat 1.0 or too few sims trains on near-uniform noise.
- Hidden information: `agot.engine.hidden.determinize` delegates to the Rust resampling of
  face-down orders and secret bids (PIMC); the encoder masks hidden orders. Known remaining
  leak: the first chooser's House card in a battle (revealed simultaneously in the physical
  game).
- Rulebook scans are local reference material: `Rulebook/` is gitignored and must never be
  committed or packaged. The desktop GUI reads `Rulebook/board.jpg` when it exists and
  otherwise falls back to the schematic region board; do not commit derived board images.

## Training runs

- **Do not start training runs on this machine unless the user asks** (user directive): it is
  CPU-only (no CUDA), and heavy runs belong on a separate machine. Prepare/verify here, run there.
  For Google Colab, start from `colab/agot-colab.ipynb` (private-repo token as a Colab secret
  `GITHUB_TOKEN`; checkpoints on Drive; the install cell caches the abi3 wheel on Drive per
  commit; the launch cell auto-resumes).
- `checkpoints/run1` and `run2` are **void**: trained with the old broken policy head (policy loss
  flat at ~2.0 = uniform over candidates). Don't resume them; the loader refuses them.
- A run writes `agot.pt` (weights), `agot.state.pt` (iteration/best-eval/optimizer for
  `--resume`), `agot.log.jsonl` (one line per iteration) and `agot-best.pt` (best gauntlet eval).
  The replay buffer is deliberately not saved — a resumed run refills it and shows a transient
  loss spike.
- The GUI writes human-vs-bot matches under `checkpoints/play-logs/`: a JSONL decision/result
  log and, for MCTS bot seats, a ReplayBuffer-compatible NPZ of those decisions. Undo truncates
  the NPZ but leaves an undo event in the JSONL.
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
- `docs/rulebook-audit.md` — rule-fidelity audit trail; `crates/agot-core/src/rules_fixes.rs`
  holds one regression per audited rule fix (add one when fixing a rule).
