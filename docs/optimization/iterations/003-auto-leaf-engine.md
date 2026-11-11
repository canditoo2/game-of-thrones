# 003 — make the fused leaf engine the default (`--leaf-backend auto`)

- Track: SPEED
- Status: kept
- Commit: see the branch head (`src/agot/cli.py`, `src/agot/training/rust_engine.py`)

## Hypothesis

The repository ships three leaf engines and defaults to the slowest one. The ONNX Runtime
float32 graph is the *same arithmetic* (the export traces `PackedGraph`, which
`tests/test_onnx_backend.py` pins against `forward_packed`), but it executes the whole forward
in one C++ pass instead of 20 Python-dispatched ops. On a 1–2 leaf batch that dispatch is
2.5–3x the arithmetic (iteration 000), and those are exactly the batches that `agot gate`,
`agot evaluate`, `agot play` and the tiny training runs use.

## Prediction (written before the change)

- Metric: leaf forward (`ab_leaf.py`) and end-to-end self-play (`ab_selfplay.py`).
- Expected: 1.5–2.2x on the leaf forward at 1–4 leaves, ~1.2x at 16; 1.4–1.7x end-to-end
  self-play at batch 3–4; meso wall clock +1.3x.
- Falsified if: the games' records change (same seeds, same visits), or the fp32 KL against
  torch exceeds 1e-6 nats.

## Change

- `--leaf-backend auto` (new default for train/gate/evaluate/gauntlet) resolves to
  `onnx-fp32` when both `onnxruntime` and `onnx` import, and to `torch` otherwise. `auto` is
  a CLI-only alias: it is resolved before a spec is built, so a checkpoint or a queued gate
  always records a concrete engine.
- `_prepare_leaf_graph` now runs before the first self-play round unconditionally. The graph
  key covers the weights, not just the checkpoint file, so this is also correct on the first
  iteration of a fresh run — previously every worker in the pool exported the same graph
  concurrently because the checkpoint did not exist yet.
- The duplicate export after `model.save(...)` is gone: the next iteration exports once, from
  exactly those weights, right before the games.

## Gates

`pytest -q` 216 passed · `ruff` clean · `cargo` gates unchanged (no Rust change in this
iteration) · `agot demo --policy greedy --seed 1` 0.284 s.

## Measurement

Leaf forward, normalised A/B (`leaves`, `candidates`):

| leaves | torch (after 002) | onnx-fp32 | speedup |
|---|---|---|---|
| 1 | 475.1 µs | 243.1 µs | **2.15x** |
| 2 | 302.6 µs | 177.2 µs | **2.24x** |
| 4 | 525.1 µs | 391.3 µs | 1.53x |
| 8 | 865.5 µs | 691.3 µs | 1.51x |
| 16 | 1591.0 µs | 1442.3 µs | 1.17x |

End-to-end self-play, Rust driver, 8 games / 16 simulations / batch 4, two rounds:

| engine | wall | decisions/s | vs torch | same records |
|---|---|---|---|---|
| torch | 9.71 s | 530 | 1.00x | — |
| onnx-fp32 | 6.35 s | 811 | **1.53x** | yes |
| onnx-int8 | 3.98 s | 1272 | 2.44x | no |

(6 games / batch 3 confirms it: 7.94 s torch, 5.11 s fp32 (1.55x), 4.03 s int8 (1.97x).)

Numerics, 40 random leaf batches, logits softmaxed per leaf:

| model | fp32 KL mean / p95 | int8 KL mean / p95 | fp32 max abs value drift |
|---|---|---|---|
| fresh net | 3.4e-09 / 2.2e-08 | 3.0e-08 / 1.7e-07 | 3.8e-08 |
| run4-best | 6.9e-09 / 7.2e-08 | 4.1e-04 / 8.4e-04 | 8.3e-06 |

## Verdict

Kept. The leaf forward doubles at the batch sizes used by the non-training commands, the
self-play records are byte-identical on the sampled games, and the fp32 graph is numerically
a round-off away from torch. int8 stays opt-in: it is 1.6x faster again but it moves the
policy by ~4e-4 nats and changes recorded games, which is a learning-signal decision, not a
speed one (run6 already selects it explicitly).
