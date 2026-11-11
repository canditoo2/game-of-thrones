# Optimization loop — operator prompt

A copy-paste prompt that turns an agent session into a measurement-driven improvement loop over
this repository. It is written for the agent, in English, like the rest of `docs/`.

## How to launch

**Interactive, one iteration at a time (recommended).**

1. Open a session at the repo root, on a clean tree or a dedicated branch.
2. Send:

   > Read `docs/optimization-loop-prompt.md` and execute it. Start with Iteration 0 and stop after
   > reporting the baseline table. Do not change any code before that.

3. Then advance with: `Continue the optimization loop: one iteration, then stop and report.`
4. `Run until a stop condition in §9` lets the agent chain iterations by itself.

If the agent cannot read files, paste §0–§12 as the prompt instead of the one-liner.

**Chained / headless.** Re-send the resume line in §12 for every new session. The only memory the
loop needs is `docs/optimization/LEDGER.md` plus the per-iteration files.

**Prerequisites.** `.venv` working, Rust toolchain at `%USERPROFILE%\.cargo\bin`, optional
`pip install -e ".[onnx]"` for the int8 path. Do not start from a dirty tree.

---

# THE PROMPT

## 0. Role and mission

You are the optimization lead for this repository: a Rust rules engine, search and encoder behind
a thin Python facade, an optional PyTorch scorer, and a self-play training loop.

Your job is a sequence of **small, measured, reversible changes**. You are judged on evidence, not
activity: three kept changes with numbers beat twenty with opinions. Every change gets a metric, a
prediction written before the measurement, an A/B result, and a verdict.

You have subagents. Use them to parallelize *reading* and *verification*. Never parallelize writing
into the same working tree.

The optimization target is a vector, not a scalar, and it is lexicographic (§2). "Faster" is only
an improvement if correctness and the learning signal are intact. Conversely, a correctness fix
that also removes work is the best kind of iteration.

## 1. Non-negotiables

Bind this section verbatim into every subagent brief.

**Environment**

- Windows / PowerShell. Invoke the venv explicitly: `.venv\Scripts\python`.
- `cargo` is not on `PATH`: use `%USERPROFILE%\.cargo\bin\cargo`.
- After **any** Rust change run `python -m maturin develop --release` with cargo and the WinLibs
  `dlltool` on `PATH` and `PYO3_PYTHON` pointing at the venv interpreter. A stale
  `src/agot/_core.pyd` is the classic false green: Rust tests pass, Python still runs the old
  engine.
- This PC is the build-and-verify machine. **No heavy runs unless the user asks.** Heavy means a
  single job over ~15 minutes, more than ~100 games, any Colab-scale configuration, or anything
  that saturates the machine for minutes. If a change needs a heavy run to be validated, do not
  run it: write an experiment card for Colab (§11) and move on.

**Frozen regression pins**

- `rust/crates/agot-core/tests/data/engine_fixture.json` and
  `rust/crates/agot-core/tests/data/encoder_golden.json` are frozen. Never edit, never regenerate,
  never relax `assert_bits_equal`, `MIN_VECTORS` or `total_decisions`. A disagreement is a bug in
  the Rust side.
- Legal-action order and the canonical action text are wire format. Reordering, de-duplicating,
  sorting or renaming breaks the pin even when the game is identical.
- `STATE_DIM = 97`, `ACTION_DIM = 141`, the house order and the order-spec keys are checkpoint
  format. Changing them invalidates every saved network.

**Architecture**

- No torch in `engine/` or `agents/`; `agot-core` must not depend on torch, PyO3 or the network.
- `forward_packed` ↔ `PackedGraph` ↔ `GRAPH_REVISION` move together, or a cached ONNX graph
  silently scores old arithmetic.
- `Game.step` semantics, `StateView._version` invalidation, PIMC determinism, the temperature
  schedule and AdamW parameter groups are **behaviour, not optimization targets**.

**Process**

- One writer at a time. One logical change per iteration.
- Work on a branch. Commit kept changes locally; never push, never rewrite history.
- Never delete, skip or weaken a test to make a gate green. Never change an expected value to match
  new behaviour unless that behaviour change *is* the iteration, and then say so in the ledger.
- No secrets, no `Rulebook/`, no `checkpoints/` in git.

## 2. What counts as an improvement

Every iteration declares exactly one **track**:

- `SPEED` — wall clock, throughput, memory, IO.
- `CORRECT` — rule fidelity, determinism, hidden information, protocol semantics.
- `LEARN` — target quality, schedule, evaluation protocol; only when it can be validated cheaply
  and without changing the training semantics by accident.
- `HYGIENE` — dead work, duplicated computation, artifacts and atomicity.

The objective across tracks is lexicographic. A change may not trade a higher priority for a lower
one.

1. **Correctness** — pins green, determinism green, no new ambiguity about the rules or the
   protocol.
2. **Learning-signal integrity** — target distribution, visit counts, temperature schedule and
   hidden-information handling unchanged, unless the iteration is explicitly about them.
3. **Speed** — median wall clock of a named, reproducible benchmark.
4. **Memory and IO.**
5. **Maintainability** — only when it removes measured work (dead code, duplicated computation),
   never as taste.

A change is **KEPT** only if all of these hold:

- every gate in §5 is green;
- the primary metric improves beyond the noise floor (§5);
- every guard metric stays inside tolerance;
- the ledger entry is complete, with the prediction recorded *before* the measurement.

Otherwise **REVERT** and record the negative result. A negative result that stops the loop from
retrying is worth as much as a win.

Effect threshold: keep a speed change only if the median improves by **more than 3% and more than
twice the observed run-to-run spread**. Below that, declare "no measurable effect" and revert — a
1% win that costs complexity is a loss.

## 3. The loop

### Iteration 0 — baseline (mandatory, once)

1. Clean tree. Record `git rev-parse HEAD`, Python/cargo/rustc versions, torch and onnxruntime
   versions, CPU count.
2. Run every gate in §5 and record counts and wall clock.
3. Run the baseline benchmarks three times each and record median, min, max and spread.
4. Resolve or explicitly flag the ambiguities in §6.3. Do not silently pick one.
5. Fill any missing measurement. If a metric has no harness, adding one is a valid iteration.
6. Create `docs/optimization/LEDGER.md`, `docs/optimization/FINDINGS.md`,
   `docs/optimization/iterations/`.
7. Report the baseline table **before changing any code**.

### Iterations 1..N — exactly one change each

**A. Select.** From the backlog (§7) or a fresh profile. One sentence: metric, current value,
target.

**B. Evidence.** Read the code and get a profile or a counter. No hypothesis without a
`file:line` and a number.

**C. Predict — before touching code.** Write down: metric, expected delta, plausible range, and
what result would falsify the hypothesis. A prediction recorded after the measurement is invalid
and the iteration must be re-run or discarded.

**D. Implement** the minimal diff. If the change needs a second change to be measurable, split it
into two iterations.

**E. Gates** (§5). A red pin stops the iteration: revert, record, and — if the pin is telling you
something real — escalate instead of "fixing" the fixture.

**F. Measure blind** (§5): the measurer receives only the two artifacts and the command, not the
hypothesis.

**G. Decide.** Actual versus predicted. Keep only if §2 holds. Update the prediction block with
the outcome; do not rewrite the prediction.

**H. Record.** One ledger row plus one iteration file (§10). Negative results are results.

## 4. Subagents

| Role | Type | Writes? | Input | Output |
|---|---|---|---|---|
| Orchestrator | you | ledger only | everything | selection, verdicts, final report |
| Scout ×1–3 | `explore` | no | one subsystem each | `file:line` evidence, ranked levers, risk |
| Implementer | `general` | **yes, sole writer** | one lever + prediction | minimal diff, gates run, diff summary |
| Verifier | `explore`/`general` | no | diff + claim | independent reproduction, semantic drift check, adversarial cases |
| Measurer | `general` | artifacts only | two build paths + command | A/B table, blind to the hypothesis |
| Scribe | orchestrator | `docs/optimization/**` | all of the above | ledger, findings, experiment cards |

Rules:

- Scouts run in parallel; writers never do. Parallel implementers only in separate worktrees, only
  for disjoint files, and only when the user asks for parallel tracks.
- The verifier is never the implementer and never shares a session with it.
- A subagent that finds a rules bug **stops and escalates**. It does not touch a frozen pin and
  does not "make the test pass".
- Every subagent brief starts with §1 verbatim.
- Every subagent return must contain the exact commands run and their output. Prose-only claims
  are rejected.

Copy-paste briefs:

```text
SCOUT — read-only.
Repo: <root>. Read §1 of docs/optimization-loop-prompt.md first.
Subsystem: <files/modules>. Question: <one question>.
Return: (a) the call path with file:line; (b) every measured number you find with its source;
(c) 5 ranked levers with expected gain, risk, and the test that would catch a regression;
(d) anything that looks like dead work, double work or a correctness ambiguity.
Do not modify anything. Do not run anything longer than ~2 minutes.
```

```text
IMPLEMENTER — sole writer for this iteration.
Repo: <root>, branch <name>. Read §1 of docs/optimization-loop-prompt.md first.
Change: <one lever>. Metric: <name>. Prediction: <delta, range, falsifier>.
Constraints: minimal diff; no test edits; no fixture edits; no behaviour change outside the lever.
Run the gates in §5 yourself and paste the commands and results. If a gate fails and the fix is
not inside the lever, stop and report instead of widening the diff.
```

```text
VERIFIER — read-only, adversarial.
Repo: <root>, branch <name>. Read §1 of docs/optimization-loop-prompt.md first.
Claim to attack: <diff + claimed effect>.
Do: reproduce the gates from scratch; look for semantic drift (ordering, RNG draw order, f32
re-association, hidden-info leakage, changed legal actions); propose the smallest input that would
break it. Return PASS/FAIL with evidence and the exact commands. Do not edit files.
```

```text
MEASURER — blind.
You get two paths A and B and one command. Run the command three times each, interleaved, same
machine state, same seeds. Report median/min/max per path and the difference with its spread.
Do not ask what changed and do not read the branch diff.
```

## 5. Gates, benchmarks, measurement protocol

### Gates (all must be green before a keep)

```powershell
# Python
.venv\Scripts\python -m pytest -q
.venv\Scripts\ruff check src tests

# Rust
cd rust
& "$env:USERPROFILE\.cargo\bin\cargo" fmt --all -- --check
& "$env:USERPROFILE\.cargo\bin\cargo" clippy --all-targets -- -D warnings
& "$env:USERPROFILE\.cargo\bin\cargo" clippy -p agot-py --all-targets -- -D warnings
& "$env:USERPROFILE\.cargo\bin\cargo" test --release
```

`cargo bench --bench clone_step` is a measurement, not a pass/fail gate: run it in the baseline and
in the measurement step, record the numbers, and treat a large regression as a revert signal.

After a Rust change, rebuild before any Python measurement:

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
$env:PYO3_PYTHON = "$PWD\.venv\Scripts\python.exe"
# WinLibs dlltool must also be on PATH
.venv\Scripts\python -m maturin develop --release
```

Record the gate counts you observe; do not hard-code counts from this document. Docs in this repo
are known to lag the code.

### Benchmarks

- Micro: `cargo bench --bench clone_step` (clone, step, legal actions, planning throughput).
- Leaf: `.venv\Scripts\python -m agot.nn.onnx_backend <checkpoint>` (torch / fp32 / int8 per-call
  cost and the int8 policy KL).
- Meso: a tiny non-heavy iteration, second iteration is the warm one:

```powershell
.venv\Scripts\agot train --iterations 2 --games 4 --workers 1 --batch-games 2 `
  --simulations 16 --determinizations 1 --max-rounds 4 --seed 7 `
  --train-threads 2 --device cpu --buffer 4000 --buffer-save-every 0 `
  --eval-games 0 --gate-games 0 --gate-mode inline `
  --checkpoint checkpoints\perf-ab\base.pt
Get-Content checkpoints\perf-ab\base.log.jsonl
```

- Training epoch in isolation: same command with `--games 0 --resume <checkpoint-with-replay>`.
- End-to-end only when micro and meso agree.

### Protocol

- Same machine state for A and B, interleaved order, same seeds and seed ranges.
- Warm-up run discarded; three measured repetitions minimum; report median, min, max.
- If the spread exceeds ~15%, add repetitions or pick a quieter metric before drawing conclusions.
- Never compare across days without re-running the control on the same day.
- Micro → meso → end-to-end, in that order. Do not jump to end-to-end to "confirm" a micro win.
- A benchmark that changes the work being done (fewer simulations, fewer rows, smaller batch) is
  not a speedup. It is a protocol change and belongs in the learning-quality track with its own
  target-KL evidence.

## 6. Baseline and oracles

### 6.1 Machine baseline (verify these on this machine; they are documented, not guaranteed)

| Metric | Documented value | Source |
|---|---|---|
| `State::clone` | ~380 ns | `cargo bench --bench clone_step` |
| `Game::clone` | ~432 ns | same |
| `legal_actions` (opening, 45 candidates) | ~348 ns | same |
| clone + step | ~472 ns | same |
| Planning throughput | ~2.5 M decisions/s | same |
| `State` size | 4872 B | same |
| Leaf call, torch / fp32 / int8 | 1316 / 1099 / 499 µs | 24 leaves × 15 candidates, 1 thread |
| Python suite | 213 tests, ~45 s | local venv, torch present |
| Rust suite | 292 tests, ~7 s warm | `cargo test --release` |
| `agot demo --policy greedy --seed 1` | ~0.4 s | local |

An iteration that changes these numbers must re-measure them, not cite this table.

### 6.2 Strength oracles

Parity is 1/6 ≈ 16.7%. Greedy vs 5×random 38.3% (CI 27–51%), random vs 5×greedy 3.3%, greedy vs
5×greedy 23.3%. **These anchors predate sequential Westeros and PIMC deck resampling**: re-run the
gauntlet before comparing a new number to them.

A checkpoint is judged by a direct head-to-head, never by the `agot-best.pt` name. Per-house rates
matter more than the aggregate: a single-seat aggregate can hide a house regression.

### 6.3 Open questions that block a trustworthy baseline

Resolve these in Iteration 0, or flag them as unresolved and keep them out of every claim.

1. **`--determinizations 1` semantics.** The code says a single perfect-information root
   (`search.rs:442`, comment at `search.rs:472`), while the docs call the single world "exact".
   Those are different claims. If run6's baseline is treated as fair-information PIMC, verify it
   with a same-seed 128/1 vs 128/2 A/B on per-phase target KL before trusting any strength number.
2. **The policy-loss success criterion contradicts itself** in `AGENTS.md`: "below ~2.0" in one
   place, "floor is mean target entropy 2.2–2.6" in another. Use `target_entropy` and the trend,
   never the absolute `< 2.0`.
3. **The run6 projection does not reconcile** with its component gains. Measure
   `seconds_selfplay`, `seconds_train`, `examples_new`, `target_entropy`; never extrapolate.
4. **Stale docs**: test counts, `rules_fixes` count, leaf dedupe listed as "not landed" while it is
   default-on, and the methods runbook still describing run4. Fixing the doc is cheap; fixing the
   *belief* requires the measurement.

## 7. Seed backlog (verified leads, not a to-do list in order)

Profile first, then pick. Each entry: change — where — why — risk.

**Speed, low risk**

1. Guard every `state.note(format!(...))` behind `log.is_some()` — `state.rs:1000` + ~87 call
   sites — eager `String` allocation on every step even when the log is `None` in all search paths.
2. Stop generating legal actions twice per decision — `agot-py/src/selfplay.rs:389-395` — the
   count for the budget, then the list again in the search.
3. Reserve the child vector before expansion — `search.rs:978-1000` — repeated reallocation of a
   ~200 B/element `Vec<Node>`.
4. Reuse the leaf buffers and skip the dedupe copies when every row is unique —
   `selfplay.rs:479-596`, `leaf_dedupe.rs:53-164` — full copies of states and actions per pass.
5. Cache the parsed `DataSet` — `map.rs:834`, `agot-py/src/lib.rs:387` — five JSON files re-parsed
   per `Game.new`/per batch.
6. Vectorise buffer insertion and compaction — `replay.py:85-98,113-133` — per-example `add`,
   quarter-drop memcpys.
7. Atomic replay snapshot, cheaper compression, or a larger `--buffer-save-every` —
   `replay.py:163-182` — the only non-atomic write in a run, and the cost is currently
   unattributed.
8. Cache the ONNX leaf backend per process instead of rebuilding per batch —
   `rust_engine.py:424-429`, `onnx_backend.py:520-548` — per-call weight and checkpoint digests
   plus a fresh session.

**Speed, medium risk (bit-exactness or RNG order)**

9. Kill the re-boxing in `with_node_game` — `search.rs:334-361` — one ~3.2 KB allocation and
   memcpy per call, roughly twice per simulation.
10. Encoder micro-opts: hoist `fortifications.contains(&id)` out of the 58-region loop, reuse one
    `RegionCache` per pass — `observation.rs:259-281`, `selfplay.rs:512` — must stay bit-identical
    against the golden vectors.
11. Memoise the repeated board scans in mustering and march generation — `phases.rs:974-1113`,
    `action_phase.rs:100-125,282-343` — the pattern already used by `supply_profile`.
12. Single-pass `select_child` — `search.rs:868-908` — changes f64 summation order and therefore
    visit ties; only with the PUCT regression tests as a hard gate.

**Correctness / hygiene**

13. Deck recycling across ten rounds is unverified (`docs/roadmap.md`, item 11).
14. `rust_engine.self_play_records` plays every game twice and discards the first result —
    `rust_engine.py:289-302` — dead today, a 2× trap tomorrow.
15. `PendingBattle` is deep-cloned on every chooser step — `action_phase.rs:1129`.
16. Westeros card resolution clones a `serde_json::Value` per card — `phases.rs:126,226`.
17. `PyGame::snapshot` round-trips through JSON instead of using the existing `to_python` —
    `agot-py/src/lib.rs:662`.

## 8. Do not retry (measured and rejected, or actively dangerous)

- `torch.compile` at production batch sizes; bf16/fp16 on CPU; Candle/burn; C++/libtorch; a PyO3
  trainer; tied trunks.
- Rewriting the non-NN path for speed: the Amdahl ceiling is ~1.11–1.14×.
- Top-K root pruning as currently conceived; post-hoc structured pruning.
- int8 on small batches (below ~1500 rows); ORT defaults (one thread per core, spinning on).
- More than 16 `--batch-games`; more workers than vCPUs.
- A narrower hidden width as a warm start for a 256-wide checkpoint.
- Resuming run1/run2 (void head) or run3 (pre-change baseline); switching `--gate-mode` on resume.
- Adding `index_add_` back to the exported graph; `torch.export` for the packed forward.
- Treating a lower policy loss as strength without a head-to-head.

## 9. Stop conditions

Stop and write the final report when any of these happens:

- 12 iterations completed, or 6 hours of local compute, whichever comes first (configurable).
- 3 consecutive iterations with no measurable effect or a revert.
- A frozen pin fails and the cause is not inside the current lever.
- The only way to validate a change is a heavy run.
- A baseline ambiguity in §6.3 turns out to invalidate the measured baseline.

## 10. Ledger schema

`docs/optimization/LEDGER.md` — one row per iteration, append-only:

| id | date | track | change | metric | before | after | delta | verdict | commit |
|---|---|---|---|---|---|---|---|---|---|
| 000 | … | baseline | — | suite/s | … | … | — | — | `HEAD` |

`docs/optimization/iterations/NNN-slug.md`:

```markdown
# NNN — <one line>

- Track: SPEED | CORRECT | LEARN | HYGIENE
- Status: kept | reverted | no-effect | blocked
- Commit: <sha or none>

## Hypothesis
<one paragraph, file:line evidence, the counter or profile that motivates it>

## Prediction (written before the change)
- Metric: <name and command>
- Expected: <delta and range>
- Falsified if: <condition>

## Change
<files, diff summary, why this is the minimal version>

## Gates
<command → result, including counts>

## Measurement
<A/B table: 3 reps, median/min/max, spread, machine state>

## Verdict
<actual vs predicted; kept/reverted and why>

## Follow-ups
<what this teaches the next iteration; links to experiment cards>
```

`docs/optimization/FINDINGS.md` — durable knowledge only: what is now known to be fast, slow,
wrong, or unmeasurable, with the command that proves it. No narrative.

## 11. Experiment cards (for anything heavy)

When a change needs Colab-scale validation, do not run it. Write
`docs/optimization/experiments/NNN-slug.md` with: question, config diff versus run6, seeds, games,
expected effect and the decision rule, cost estimate, and the exact launch cell. The loop's job is
to make the card launchable, not to launch it.

## 12. Final report

1. Table of iterations: kept / reverted / no-effect, with the metric delta.
2. Net effect on the primary metrics and on the guard metrics.
3. What is now known that was not known before (including negative results).
4. Experiment cards produced, ranked by expected value per hour of compute.
5. The next three moves, each with its expected gain and its risk.

Resume line for a new session:

> Continue the optimization loop. Read `docs/optimization/LEDGER.md`, `FINDINGS.md` and the last
> three iteration files, then run the next iteration per `docs/optimization-loop-prompt.md`.
