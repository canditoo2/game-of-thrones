# AGENTS.md

AlphaZero-style RL agent for **A Game of Thrones: The Board Game (2E)**, six players: a Rust
rules engine, search and encoder (`rust/`) behind a thin Python facade, an optional PyTorch
scorer network (`nn/`) and the self-play training loop (`training/`). Everything is driven by
the `agot` CLI.

**Current direction (2026-11-11):** the next Colab run is **`run6`** (checkpoints under
`/content/drive/MyDrive/agot/run6`). The notebook launches `--simulations 128
--determinizations 1 --phase-budgets default --leaf-backend onnx-int8 --games 64
--batch-games 16 --buffer 80000 --gate-games 120 --gate-mode deferred`. The single PIMC
world is exact (the two determinizations encode bit-identically), the phase caps cut scored
rows by ~40%, and ONNX int8 cuts the leaf forward 2.6x. This PC remains the build-and-verify
machine and must not start heavy runs unless asked. `run5` reached 105 iterations and is the
warm-start candidate (`--resume` from its `agot.pt`); a fresh start is also valid.

## Commands (Windows/PowerShell — invoke the venv explicitly)

- Tests: `.venv\Scripts\python -m pytest -q` (180 tests; the rules/combat/encoder
  regressions live in Rust — see `cargo test --release` below). When piped in PowerShell
  the final summary line can be swallowed — trust `$LASTEXITCODE` (0 = pass).
- One file: `.venv\Scripts\python -m pytest tests/test_nn.py -q`
- Lint: `.venv\Scripts\ruff check src tests` (line-length 100, py310 target).
- CLI: `.venv\Scripts\agot <demo|play|legal|evaluate|gauntlet|round-robin|train|gate|report>`. One
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
  three legal actions; the raw Rust fixture default stays deterministic. All Python games also use the FAQ v2.0
  Lannister opening: one Ship in Lannisport Port (`GameConfig.lannister_port_ship`,
  default on; the raw Rust default stays off so the frozen parity pins keep the
  printed setup).
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
- Leaf inference: `--leaf-backend {auto,torch,onnx-int8,onnx-fp32}` (train/gate/evaluate/gauntlet,
  default `auto`) picks the engine that scores search leaves (`agot/nn/onnx_backend.py`).
  `auto` resolves to `onnx-int8` for `agot train` and to `onnx-fp32` for evaluate/gauntlet when
  `pip install -e ".[onnx]"` (onnxruntime + onnx) is present, and to `torch` otherwise, so
  `torch` stays the only engine CI needs and a pure-PyTorch install keeps working; it is
  resolved before the spec is built, so a checkpoint or a queued gate always records a concrete
  engine. Training gets int8 because that is the engine run6 already selects and the only one
  that halves self-play (2.35–2.44x end to end, mean policy KL ~4e-4 nats, inside the budget
  `tests/test_onnx_backend.py` enforces); evaluation keeps fp32, which is a round-off away from
  torch (7e-8 nats) and still 1.5–2.2x faster. The ONNX backends cache
  `<checkpoint stem>.<backend>.onnx` plus a JSON key beside the checkpoint and refuse a graph
  whose key (checkpoint hash, weight digest, model config, backend, `GRAPH_REVISION`) no
  longer matches. Measured at 24 leaves x 15 candidates and one thread: torch 1316 us,
  fp32 1099 (1.2x), int8 499 (2.6x) per leaf call, int8 policy KL p95 ~1.3e-3 nats on a
  trained-like net; on the 1-2 leaf batches that gate/evaluate/play actually use, fp32 is
  2.2x the torch path and int8 3.4x (`docs/optimization/iterations/003-auto-leaf-engine.md`).
  The choice travels in the self-play spec (`leaf_backend`) and in the
  deferred `GateSpec`, so a queued gate is judged with the engine its training run used;
  `agot gate --leaf-backend` overrides it. Repro: `python -m agot.nn.onnx_backend <ckpt>`.
- Compact leaf graph (`agot train --leaf-compact {auto,off}`, default `auto`, ONNX backends
  only): most trunk ReLU units of a trained net never fire (run7: 202/173/173/165 of 256 dead
  on the replay buffer, none of them firing on 40k real search leaves). Once per iteration,
  in the parent, `onnx_backend.calibrate_keep` measures which trunk units fire on the replay
  buffer (over its distinct rows only, bit-verified: ~0.5 s at 79k examples, two threads)
  and `_prepare_leaf_graph` exports
  `<stem>.<backend>.compact.onnx` without the others; the keep-sets ride the plan
  (`SelfPlayPlan.leaf_keep`) so the workers open that same graph. Below
  `MIN_CALIBRATION_EXAMPLES` (50k: 33k contiguous examples still missed units that fire on
  leaves) the run keeps the full graph. Compact fp32 is exact against the full graph on real
  leaves (policy KL 0, value |diff| <= 2.6e-6) and 2.6x cheaper per call; compact int8 is
  1.6x cheaper than full int8. Gate, evaluate, gauntlet and play always use the full graph.
  The log records `leaf_compact` (kept widths, `null` = full) and the calibration/export
  seconds. Measure real leaves with `python -m agot.nn.onnx_backend <ckpt> --leaves
  <leaves.npz> --replay <agot.replay.npz>` (per-call time, policy KL and value drift of full
  vs compact, fp32 vs int8, against torch). Note that int8 is not round-off: on run7's real
  leaves full int8 moves the value by 0.059 on average and up to 1.06
  (`docs/optimization/iterations/008-compact-leaf-graph.md`), and at game level that noise
  flips a near-tied move within the first few decisions of every self-play game on both the
  old and the new driver (0/32 identical games vs the fp32 twins; a same-seed rerun is
  bit-identical, `docs/optimization/FINDINGS.md`), so compare pipelines on their buffers in
  fp32, never in int8.
- Leaf evaluation cache (Rust driver, on by default; `agot train --leaf-cache {on,off}`, spec
  key `leaf_cache`): about 73% of the leaves of one `selfplay()` call repeat a network input
  already scored in that call, so the driver keeps each input's raw logits and values
  (`agot-core/src/leaf_cache.rs`, keyed by a 256-bit digest of the encoded state and candidate
  rows, one cache per evaluator, bounded at 2^16 entries) and only sends misses to Python.
  Records are bit-identical with the cache on and off for any evaluator that is a pure
  row-wise function (pinned by `tests/test_leaf_cache.py`; onnx-fp32 is one in practice). int8
  is not, with or without the cache: its dynamic quantization range depends on the batch, so
  the same leaf in two batches differs by up to 1.24 in a value. Gate and evaluation matches
  get the cache too. Counters and per-section timers: `_core.selfplay_stats()`.
- Speed record of 2026-11-20 (`docs/optimization/LEDGER.md`, iterations 006-010): one run8-shaped
  iteration is 2.33x faster than commit `7952ae2` on this PC (self-play 2.57x, training pass
  2.22x), and 2.30x with `--leaf-backend onnx-fp32`, which now costs about what int8 does and
  has none of its value drift. None of it has been measured on Colab yet; with the whole A/B
  tree pinned to two logical CPUs (the Colab shape, LEDGER 2026-11-22) the same protocol gives
  2.78x median over four paired rounds — the advantage does not shrink when cores are scarce.
- `checkpoints/` and `Rulebook/` are gitignored (run artifacts; the rulebook scans are
  copyrighted and must not be committed).

## Architecture constraints (easy to break)

- Layering: the Python `engine/` facade is numpy-only and must stay importable without torch;
  never import torch from `engine/` or `agents/`.
- `rust/crates/agot-core` is the engine, search and encoder. It must never depend on torch or
  PyO3 (the Python binding is a separate crate), and it must not reach for the network. The
  differential fixture and encoder golden vectors are frozen regression pins: a disagreement is
  a bug in the Rust side, never something to fix by editing a fixture or a golden file.
- `agot/nn/onnx_backend.py` is an optional leaf scorer. It imports torch at module level (like
  `nn/model.py`) but `onnxruntime`/`onnx` only inside functions, and nothing imports it at
  import time, so `engine/`, `agents/` and the torch-free CLI stay torch-free. Its
  `PackedGraph` is a *second* copy of `forward_packed`'s arithmetic (a one-hot matmul instead
  of `index_add_`, which does not export); `tests/test_onnx_backend.py` pins the two against
  each other. Changing `forward_packed` means changing `PackedGraph` too and bumping
  `GRAPH_REVISION`, or a stale cached graph would quietly score the old arithmetic.
  `PackedGraph(model, keep)` is the compact variant: only the trunks shrink (rows of each trunk
  layer, the matching columns of the next layer, of both halves of `policy_head[0]` and of
  `value_merge`); `policy_head` and `value_head` must stay full width, because their
  replay-dead units do fire on real leaves (value change up to 1.08). The keep-sets are part
  of the cache key and the file name, so a graph for other keep-sets, or the full graph, is
  rebuilt rather than reused; `keep=None` is exactly the full graph. Dropping a unit is exact
  only while it would have output zero, so calibrate on enough data (coverage, not a margin:
  the dead action-trunk units sit just below zero, and a margin of 0.05 already keeps 138
  instead of 83 units of `action_trunk[0]`).
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
- `--gate-mode deferred` moves the gate outside the training iteration for flaky Colab
  runtimes: after the iteration commit the loop only writes
  `<stem>.gate/candidates/iter-NNNNNN/{candidate.pt,best.pt,spec.json}` (spec last), then
  `agot gate --checkpoint <ckpt>` plays it in `--batch-games` chunks, one atomic
  `shards/chunk-XX.json` each. A verdict (and `agot-best.pt` promotion on a pass) waits until
  every chunk exists; patience is recomputed from `gate.log.jsonl`, where a candidate without
  a verdict is not a strike. `agot.state.pt` records `gate_mode`, and a resume refuses to
  switch modes.
- `run4` finished 60 iterations but the gate never promoted past its iteration-10 seed: the
  overall rate against it rose to 0.70 while the strict per-house rule kept rejecting. That
  evidence is why the tolerance exists; judge a checkpoint by a direct head-to-head, not by
  the `agot-best.pt` name.
- A run writes `agot.pt` (weights), `agot.state.pt` (iteration/best-eval/optimizer for
  `--resume`), `agot.log.jsonl` (one line per iteration), `agot-best.pt` (gate-approved best)
  and `agot.replay.npz` (compressed replay snapshot, ~15-60 MB at 100k examples, written every
  `--buffer-save-every` iterations and restored on resume, which removes the empty-buffer loss
  spike). The snapshot is compressed in a background thread (`ReplayBuffer.save_async`) that
  every buffer mutation waits for, and written to a temporary that is renamed into place, so
  an interrupted write keeps the previous one. The log carries `seconds_selfplay`, `seconds_train`, `target_entropy` and
  `buffer_restored`; with raw-visit targets the policy-loss floor is the mean target entropy
  (~2.2-2.6 nats), so the old "<2.0" smoke reading no longer applies.
- The GUI writes human-vs-bot matches under `checkpoints/play-logs/`: a JSONL decision/result
  log and, for MCTS bot seats, a ReplayBuffer-compatible NPZ of those decisions. Undo truncates
  the NPZ but leaves an undo event in the JSONL.
- Human games: the Swords and Ravens archive and its round snapshots live under
  `checkpoints/human-games/` (gitignored); `docs/human-games.md` has the protocol and the
  coverage caveats. `tools/export_human_value.py` builds `values.npz`, and
  `agot train --human-value` adds an opt-in value-only pass (off by default: the game-level
  A/B improves calibration but not winner ranking at the current sample size).
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
