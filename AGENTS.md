# AGENTS.md

AlphaZero-style RL agent for **A Game of Thrones: The Board Game (2E)**, six players: a Rust
rules engine, search and encoder (`rust/`) behind a thin Python facade, an optional PyTorch
scorer network (`nn/`) and the self-play training loop (`training/`). Everything is driven by
the `agot` CLI.

**Current direction (2026-10-29):** the next Colab run is **`run5`** (checkpoints under
`/content/drive/MyDrive/agot/run5`), on code commit `9443761`: first-play value from the seen
children, raw-visit policy targets, a value head that also reads the candidate actions, the
neutral-support legality fix and a 200-game gate. This PC remains the build-and-verify machine
and must not start heavy runs unless asked. `run4` is the pre-fix baseline (60 iterations, policy
loss 1.98 -> 0.62, gate never promoted; its final beat the gate best 119-81). Watch the policy
target's entropy and the gate trend, not the policy loss alone.

## Commands (Windows/PowerShell — invoke the venv explicitly)

- Tests: `.venv\Scripts\python -m pytest -q` (180 tests; the rules/combat/encoder
  regressions live in Rust — see `cargo test --release` below). When piped in PowerShell
  the final summary line can be swallowed — trust `$LASTEXITCODE` (0 = pass).
- One file: `.venv\Scripts\python -m pytest tests/test_nn.py -q`
- Lint: `.venv\Scripts\ruff check src tests` (line-length 100, py310 target).
- CLI: `.venv\Scripts\agot <demo|play|legal|evaluate|gauntlet|round-robin|train|report>`. One
  full game is ~0.2 s: `agot demo --policy greedy --seed 1`. `agot play --checkpoint <dir|pt>
  --house stark` opens the local Tkinter desktop game: the setup screen chooses the human
  house, opponent and checkpoint, then the full board and action list are shown. When
  `Rulebook/board.jpg` is present (Pillow installed), the game uses the local scan as the
  board; otherwise it falls back to the schematic region board. Desktop and console
  play, training and evaluation enable `deferred_battles` (all March destinations are
  chosen before the one battle starts) and `sequential_westeros`, which reveals each of
  the three cards, resolves its own mustering/supply/bidding/Wildling step, and only then
  draws the next card; `doran_choice` prompts for Doran Martell's Influence track. Combat card text
  resolves before the blade step. Python training/evaluation include Doran's choice as
  three legal actions; the raw Rust fixture default stays deterministic.
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
- Packaging: `pyproject.toml` uses maturin. `pip install -e .` installs the editable
  Python path (`agot.pth` -> `src`) and the console script; it does **not** refresh the
  compiled extension. After any Rust change run `python -m maturin develop --release`
  (cargo on `PATH`, `PYO3_PYTHON` pointing at the venv interpreter), which rebuilds and
  copies `src/agot/_core.pyd`; a stale `.pyd` otherwise keeps the old engine alive while
  the Rust tests still pass. `rust/crates/agot-py` is not a default workspace member, so
  the engine's own `cargo test` loop needs no PyO3 toolchain. On Windows a direct
  `maturin build` needs binutils (`dlltool`) on `PATH` - WinLibs' UCRT build. The wheel is
  abi3 (Python >=3.10), so one build covers the local venv, CI and Colab.
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
  face-down orders, secret bids, the first chooser's House card while the second chooser
  decides, and the order of the cards still face down in the Westeros and Wildling decks
  (PIMC); the encoder masks hidden orders, and the snapshot hides both battle cards until
  both players commit.
- House-card and Wildling sub-choices are real actions in Python training/evaluation and
  desktop play (`GameConfig.choice_choices=True`); the low-level Rust helpers keep their
  deterministic `resolve_and_apply` fallbacks for the frozen parity fixtures.
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
- `run3` predates the sequential Westeros resolution and the PIMC deck resampling; treat its
  numbers as a pre-change baseline and start a fresh run rather than resuming it.
- `run4` uses a relative checkpoint gate: every 10 iterations it plays 60 games of the
  candidate (3 seats) against `agot-best.pt` (3 seats), swapping seats, and only replaces the
  best when the overall win rate clears `--gate-threshold` and every house stays within
  `--gate-house-tolerance` (default 0.10, about one standard error at 30 games per house) of the
  previous best in that seat; `--gate-house-margin` is an optional *required* advantage on top of
  that. Three consecutive failures stop the run with the best preserved, and the counter is
  stored in `agot.state.pt` so patience also covers resumed sessions.
- `run4` finished 60 iterations but the gate never promoted past its iteration-10 seed: the
  overall rate against it rose to 0.70 while the strict per-house rule kept rejecting. That
  evidence is why the tolerance exists; judge a checkpoint by a direct head-to-head, not by
  the `agot-best.pt` name.
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
