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
- `checkpoints/` and `Rulebook/` are gitignored (run artifacts; the rulebook scans are
  copyrighted and must not be committed).

## Architecture constraints (easy to break)

- Layering: `engine/` is numpy-only and must stay importable without torch; never import torch
  from `engine/` or `agents/`.
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
