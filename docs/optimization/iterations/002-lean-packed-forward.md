# 002 — a lean inference path for the packed forward

- Track: SPEED
- Status: kept
- Commit: see the branch head (`src/agot/nn/model.py`)

## Hypothesis

`PolicyValueNet.forward_packed` walks 14 `nn.Linear` modules per call. A leaf batch is 1–16
rows, so the arithmetic is tiny (a 15-candidate leaf is ~11 MFLOP) and the per-op dispatch
dominates. Measured on this box: `nn.Linear` on a 2x97 input costs 10.5 µs while the matmul
alone is 6.5 µs, and `torch.mm` on 30x256x256 is 29.6 µs — i.e. the module wrapper is
~35–40 % of a small layer call, and there are 14 of them plus the intermediate tensor
allocations.

## Prediction (written before the change)

- Metric: `checkpoints/perf-ab/ab_leaf.py`, `before` (frozen copy of the old body) vs `after`.
- Expected: 1.25–1.6x at 1–2 leaves, 1.05–1.15x at 16 (there the GEMMs dominate).
- Falsified if: logits/values drift by more than 1e-5, or a test that pins `forward` vs
  `forward_packed` fails.

## Change

`forward_packed` keeps its module path when grad is enabled (training needs the autograd
graph) and dispatches to `_forward_packed_inference` under `torch.inference_mode()`, which:

- caches the transposed weight matrices (`_packed_weights`), keyed on each parameter's
  `data_ptr()` and `_version`, so an optimizer step or `load_state_dict` invalidates them
  without the trainer cooperating; `_apply` drops the cache on any device/dtype move;
- runs one `addmm`/`mm` per layer with `relu_()` in place;
- keeps the *same* arithmetic in the same order, so `PackedGraph` (the ONNX export) and
  `GRAPH_REVISION` do not move.

## Gates

`pytest -q` 216 passed · `ruff` clean · `tests/test_nn.py` and `tests/test_onnx_backend.py`
green, including the test that pins `forward_packed` against the ONNX graph.

## Measurement

Normalised interleaved A/B (each round measures a fixed reference GEMM first; the reported
ratio is the median of paired per-round ratios).

| leaves | before | after | speedup | max abs drift |
|---|---|---|---|---|
| 1 | 525.0 µs | 475.1 µs | 1.20x | 1.5e-08 |
| 2 | 397.2 µs | 302.6 µs | 1.17x | 0 |
| 4 | 600.5 µs | 525.1 µs | 1.08x | 0 |
| 8 | 1014.7 µs | 865.5 µs | 1.12x | 0 |
| 16 | 1707.1 µs | 1591.0 µs | 1.07x | 0 |

## Verdict

Kept. Below the predicted low end at 1–2 leaves (1.20x vs 1.25x) but the effect is
consistent across five batch sizes and the drift is round-off. It is the only one of the two
leaf engines with no optional dependency, so it is what a torch-only install gets.

## Follow-ups

The remaining gap to the arithmetic floor at 1–2 leaves is per-op dispatch that Python cannot
remove; a fused runtime closes it (iteration 003). `int8` closes it further at a measurable
policy cost (see FINDINGS).
