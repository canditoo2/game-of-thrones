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
