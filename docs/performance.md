# Performance budget and tuning decisions (November 2026)

Measured on run5 (64 games x 256 simulations x 2 determinizations, 2-vCPU Colab
runtime, ~1370 s per iteration) and on the 22-thread build machine. Six audits
plus four follow-ups produced the numbers below; this file is the record of what
was measured, what was changed, and what was rejected.

## Where an iteration goes

| Slice | Share | Evidence |
|---|---|---|
| Neural forward on search leaves | 88-92% of self-play | null-evaluator A/B: 2.35 s vs 45-49 s per 12 games |
| Rust engine + search + encoder | 3-5% | 4.71 us/simulation, 161k simulations/game |
| Python-Rust boundary | 0.2-1% | 6.4 us per crossing, 418k crossings per iteration |
| Training epoch (100k buffer) | 3-5% | 1563 steps at ~65 ms |
| Gate + eval, amortised | 17-25% | 200 games in one process, ~1.7-2 h per gate |
| Checkpoint + Drive IO | <1% | 6.3 MB, 7 file operations |

Amdahl ceiling for rewriting every non-NN line in another language: **1.11x**.
The engine is not the bottleneck and a port is explicitly rejected.

## What was changed for run6

- `--simulations 128` (from 256): the target changes by less than the binomial
  noise in the visit counts (KL 0.06 -> 0.08 nats at planning) and doubles the
  games per hour.
- `--iterations 300`, `--games 64`, `--buffer 80000`: 19200 games overall, same
  data-per-update ratio as run4, ~14 min per iteration.
- `--batch-games 16` (32 games per worker = two exact batches).
- Gate: 120 games, every 25 iterations, threshold 0.55, per-house tolerance
  0.25, patience 4. Same decision power as 200 games at 60% of the cost.
- `--eval-games 40 --eval-every 25`: the anchor is informational; the gate is
  the selector.
- `--buffer-save-every 1`: `agot.replay.npz` (compressed, ~15-60 MB at 100k
  examples) is written atomically next to the checkpoint and restored on
  resume, removing the empty-buffer loss spike.
- Phase timers (`seconds_selfplay`, `seconds_train`), `target_entropy`,
  `examples_new` and `buffer_restored` in `agot.log.jsonl`.
- Policy head split: the state half of the first head layer is computed once per
  leaf instead of once per candidate. Algebraically identical (per-leaf KL
  < 1e-6), 1.28-1.31x on the head, no checkpoint change.

## Measured-but-not-landed (run6 candidates)

- **Exact leaf dedupe in the Rust driver**: 22% duplicate leaves / 32% duplicate
  rows over a full game at `determinizations=2`; bit-identical scatter, ~1.47x
  on inference.
- **Phase-scaled simulation budget** (planning 96, rest 192): 0.45x cost, 2.2x
  games, planning target unchanged within noise.
- **Greedy planning teacher**: the planning prior is uniform (H/log A = 1.00)
  and the root Q spread across order choices is ~0.001, so search cannot
  improve planning under any fixed budget. Recording the greedy seats'
  planning decisions as extra targets is the only lever that raises the ceiling.
- **Narrower net** (hidden 128): 3.38x inference, and held-out cross-entropy
  shows no capacity loss (128 -> 1.328 vs 256 -> 1.322), but it breaks
  checkpoint lineage and belongs to a fresh run.

## Rejected

- Porting to C++/libtorch (<=1.06x for 10-16 person-weeks, high risk to the
  frozen parity pins), dropping torch for candle/burn (likely slower).
- `torch.compile` (0.28-0.96x at production batch sizes), bf16/fp16 on CPU
  (0.03x, no AMX), ONNX/TorchScript (the boundary is 0.8% of a call).
- int8 quantization (helps only large batches and perturbs the policy that
  generates the targets), candidate pruning (2% duplicate rows, no dominance),
  batch-games above 16 (fixed cost is already ~5%).
- A 4th/8th worker on 2 vCPU: two single-threaded workers beat one two-threaded
  worker by ~20%; more processes have nowhere to run.

## Reading the log now

With raw-visit targets the policy loss floor is the mean target entropy
(~2.2-2.6 nats), not the old ~2.0. Watch `target_entropy` (should stay ~1.0-1.2
for sharpened data, ~2.5 with raw visits), `seconds_selfplay` per game, and the
gate trend. A policy loss below ~2.0 with raw-visit targets means the search got
sharper or the net is fitting visit noise, not automatic progress.

## Decisions from the second audit round (November 2026)

Twelve further audits measured the same workload from new angles. The engine is
not the lever and no rewrite is: torch already runs at ~80% of the machine's
single-thread fp32 kernel ceiling (89 of 112 GFLOPS on a 2048^3 GEMM), candle
measures **0.41x** of torch, C++/libtorch is bounded by 1.05x, and the Amdahl
ceiling for rewriting every non-NN line is 1.14x.

Accepted, in order of measured gain per week of work:

1. **Leaves on ONNX Runtime with dynamic int8** - 2.11x end-to-end measured
   (45.6 s -> 21.5 s on 8 games), policy KL drift 4.9e-4 nats, ~100x inside the
   PIMC noise floor already accepted. No checkpoint, fixture or encoder change.
   Spike rules: >=1.8x on Colab, per-leaf KL p95 < 5e-3, 60-game int8-vs-fp32
   within one SE, no golden/fixture touched; fall back to ORT fp32 (1.18x) if the
   Colab vCPU lacks VNNI. Two known traps: `index_add_` does not export to ONNX
   (the value-context pooling must become a one-hot matmul) and the ONNX file is
   derived state, so it must be cached next to the checkpoint and keyed by its
   hash.
2. **`--determinizations 1`** - the two PIMC worlds are observationally
   identical: `encode_state`/`encode_actions` come out bit-identical across
   determinizations, and KL(128/2||64/1) = 0.00000 on ten positions in every
   phase. The second world costs +33% rows for zero signal.
3. **Per-phase simulation budgets** *(landed: `--phase-budgets default`)* -
   planning 32, wildling 16, mustering / bidding / action 32, raven / combat /
   westeros 64: -44% rows on top of the above, with a pooled target change of
   0.075 nats, 8-20x below the label noise it replaces. Re-measured on two real
   games at 64 simulations / 2 determinizations: 75 942 -> 45 333 scored rows
   (-40.3%), 63.7 -> 36.5 rows per decision, Planning 64.0 -> 37.7 and Wildling
   attacks 64.0 -> 16.0. The cap is per decision and is divided by the
   determinizations exactly like `simulations`; it never goes below one
   simulation per legal action, which is why a 45-candidate Planning decision
   lands on 46 rather than 32. At 16 simulations the preset is a no-op (every
   cap is at or above the ceiling), so it only pays off at the training budget
   of 128 or more. `off` (the default) is the fixed budget, bit for bit.
4. **Sharpened target plus a planning teacher, and a per-candidate Q head** - the
   root cause of weak play is that the 97-dim state has no board: the children of
   a planning node encode to bit-identical vectors (verified on 40/40 positions),
   so the value cannot rank orders, the visits are uniform and the target is
   uniform by construction. A fresh net trained on 4k greedy planning decisions
   reaches 40% held-out top-1 where the self-play policy reaches 10%, so the
   representation can learn and the search target cannot teach. Raw-visit targets
   (commit 9443761) put the policy cross-entropy on the uniform floor; they must
   be sharpened again *together with* a non-uniform prior, never sharpened alone.
5. **`--hidden 160`** for a fresh run - 1.85x measured end-to-end at +0.007 nats
   of held-out cross-entropy (128 reaches 2.38x at +0.012). Not compatible with
   warm-starting a 256-wide checkpoint.
6. Free items already applied: `--batch-games 16`, `OMP_NUM_THREADS=1` per
   worker, `--buffer-save-every 1`, deferred gate.

Rejected with measurements: candle/burn/tch rewrite (0.41x), C++/libtorch
(<=1.05x), a PyO3 trainer (<=0.8%), top-K root pruning (loses the baseline top-1
in 12% of cases), post-hoc structured pruning (+0.34 nats at 50%), tied trunks
(no MAC change, measured slower), fp16/bf16 on CPU (17x slower), and int8 on
batches smaller than ~1500 rows.
