# 006 — the optimizer pass of `agot train`, 2x without changing what is learned

- Track: SPEED
- Status: kept (levers 1, 2, 3, 4, 5-restore); rejected (foreach AdamW, a lower snapshot
  zlib level, live columns on the 64-row state trunk)
- Commits: `b522018` (subnormals), `ba7c23e` (batch gather), `df10927` (fused AdamW),
  `09f17a4` (live columns), `3e54423` (replay restore)
- Branch: `optimization/train-2x`, based on `7952ae2`

## Hypothesis

cProfile of one resumed run7 iteration with `--games 0` (`checkpoints/perf-2x/prof_iter.py 0
tottime`, 61 701 examples, 965 steps of 64, 2 torch threads) put `seconds_train` at 31.7 s:

- `torch/optim/adam.py:_single_tensor_adam` 13.2 s cumulative (`lerp_` 3.5, `sqrt` 2.6,
  `addcdiv_` 2.5, `mul_` 2.5, `addcmul_` 1.2; 17 370 calls each = 18 tensors x 965 steps).
  The saved state `checkpoints/run7/agot.state.pt` is 73 % subnormal (761 900 of 1 046 542
  moment entries): dead ReLU units get exactly-zero gradients, the EMA of zeros never reaches
  zero in float32, and every elementwise op on a subnormal takes the microcode slow path.
- `trainer.py:150 collate_packed` 2.5 s own / 3.3 s cumulative, 308 505 `torch.from_numpy`.
- `run_backward` 8.3 s, `linear` 3.8 s, with 70-76 % of the action trunk's ReLU columns zero for
  a whole batch (`dead` columns per batch, measured on 100 run7 batches: state L1 79 %, state
  L2 75 %, action L1 71 %, action L2 76 %, policy hidden 41 %, value hidden 99 %).
- Replay restore: `replay.py:113 add` x 61 701; snapshot `zlib.compress` 1.9 s.

## Prediction (written before the change)

- Metric: `seconds_train` of one resumed run7 iteration (`ab_iter.py --games 0`), interleaved
  against `7952ae2`; component times from a same-process, batch-interleaved harness.
- Expected: subnormals 1.3x, batch assembly another 1.1-1.2x, fused optimizer 1.1x, live
  columns 1.1-1.2x; together >= 2x.
- Falsified if: `policy_loss`/`value_loss` of the `--games 0` iteration differ from the
  baseline beyond round-off for levers 1-3 (they must be bit-identical, the arithmetic is the
  same), or the final weights of a full epoch differ.

## Change

1. **Subnormals** (`trainer.py`: `flush_denormals`, `scrub_subnormals`). FTZ/DAZ is held only
   for the gradient pass (`train_epoch`, `train_value_only`) and switched back off, so
   inference and self-play in the same process keep IEEE arithmetic. At the start of every
   epoch the optimizer's moments are scrubbed of subnormals. The scrub is the portable half:
   Intel OpenMP (Windows wheels) propagates the FP flags to its pool threads (verified: 0
   subnormals left in either half of a 1 M-element `mul_` with 2 threads), GNU OpenMP (Linux /
   Colab wheels) does not propagate them to an already-running pool, and a dead unit's moment
   stays exactly zero once zeroed (`0 * beta + 0`).
   Why `sqrt` did not improve under FTZ: the optimizer's `sqrt` goes through MKL's vector
   sqrt, which is slow on subnormal **and on exact-zero** input regardless of the flags
   (131 072 floats, 1 thread: normal 20 µs, zeros 170 µs, subnormal 860 µs, flush on or
   off). Lever 3 removes it.
2. **Batch assembly** (`ReplayBuffer.packed_batch`, `PackedBatch.from_arrays`). Fancy
   indexing on the buffer's concatenated storage. The policy normalisation keeps the
   per-example `ndarray.sum` over the same storage slice: `np.add.reduceat` was tried first
   and is *not* bit-identical (numpy's float32 sum is pairwise/SIMD-blocked); the pin test
   caught it on 3 of 4 cases. `collate_packed` stays for plain lists of examples.
3. **Fused AdamW** (`trainer._adamw`). `fused=True` is probed at construction (older torch
   raises on CPU, falls back to the default). `Optimizer.load_state_dict` overwrites `fused`
   with the saved group value, so `Trainer.load_state_dict` re-applies the kernel choice, and
   `Trainer.state_dict` writes `fused=None` so an older checkout resumes the file unchanged.
   Parameter groups (decoupled decay on matrices only) are untouched.
4. **Live columns** (`model._live_linear`, `_run_live`, `PolicyValueNet.live_columns`). In the
   grad-enabled `forward_packed` only: the action trunk's second layer and the action half of
   the scorer's first layer read the columns of their ReLU input that are non-zero somewhere
   in the batch (`index_select` on activation and weight; autograd scatters back). Exact up
   to summation order because the input is a ReLU output. Inference
   (`_forward_packed_inference`) and the ONNX `PackedGraph` are not touched, so
   `GRAPH_REVISION` does not move.
5. **Replay restore** (`ReplayBuffer._adopt`). A snapshot that fits the buffer adopts the saved
   arrays in one copy each; larger, legacy (object policies) or non-contiguous files keep the
   per-example path, whose quarter-at-a-time eviction defines what survives.

## Gates

```
.venv\Scripts\python -m pytest -q        -> exit 0 (255 collected, 254 passed, 1 skipped)
.venv\Scripts\ruff check src tests       -> All checks passed!
```

No Rust change, no maturin rebuild, no fixture touched. New tests:
`tests/test_trainer_pass.py` (scrub, flush scoping, subnormal-state epoch vs IEEE epoch, gather
vs `collate_packed` byte equality on random batches up to 300 candidates, buffer epoch vs list
epoch bit equality, old<->new optimizer-state resume, live vs dense gradients, vectorised vs
sequential restore with eviction).

## Measurement

### Same-process, batch-interleaved harness (component attribution)

`scratchpad/bench_il.py`: two trainer copies on the same run7 weights/state/replay, same
batches, steps alternating A,B / B,A every batch so clock drift hits both arms equally. A
harness bug found on the way: `Optimizer.load_state_dict` keeps the caller's tensors when
dtype/device match, so two optimizers loaded from one dict share their moments; the harness
deep-copies per arm (A-vs-A is then bit-identical).

| comparison | steps | A total | B total | A/B | per component | final weights |
|---|---|---|---|---|---|---|
| baseline recipe vs levers 1+2 | 965 | 83.87 s | 38.54 s | **2.18x** | collate 9.42x, opt 5.99x, fwd 1.05x, bwd 1.01x | bit-identical |
| 1+2 vs 1+2+fused | 300 | 4.18 s | 3.57 s | 1.17x | opt 4.00x | bit-identical |
| 1+2 vs 1+2+foreach | 300 | 4.97 s | 5.08 s | 0.98x | opt 0.98x | bit-identical — **rejected** |
| 1+2+3 vs 1+2+3+live | 300 | 4.47 s | 3.80 s | 1.18x | bwd 1.31x | round-off drift (below) |
| same | 965 | 13.20 s | 10.53 s | 1.25x | bwd 1.40x, fwd 1.15x | same |
| same | 965 | 11.23 s | 9.28 s | 1.21x | bwd 1.36x, fwd 1.09x | same |
| live also on the state trunk | 400 | 5.31 s | 4.20 s | 1.26x | — | no gain over 1.21-1.25x — **rejected** |

Absolute times differ by 6x between rows: the clock moved between runs; only the ratio inside
one row is meaningful.

### Production CLI A/B (`ab_iter.py --games 0`, run8 notebook flags, 4 interleaved rounds)

A = untouched `7952ae2` checkout with its own extension, B = branch head (`3e54423`).
`.venv\Scripts\python checkpoints\perf-2xb_iter.py --arm "A=<baseline>" --arm "B=<repo>"
--rounds 4 --games 0`.

| round | A `seconds_train` | B `seconds_train` | A/B | A `seconds` | B `seconds` | A/B |
|---|---|---|---|---|---|---|
| 0 | 23.08 | 12.11 | 1.907 | 25.40 | 15.71 | 1.617 |
| 1 | 19.70 | 10.69 | 1.843 | 21.91 | 13.48 | 1.626 |
| 2 | 24.37 | 11.43 | 2.133 | 26.80 | 14.20 | 1.888 |
| 3 | 25.78 | 10.55 | 2.444 | 28.14 | 13.34 | 2.110 |
| **median** | 23.72 (min 19.70, max 25.78) | 11.06 (min 10.54, max 12.11) | **2.020x** | 26.10 | 13.84 | **1.757x** |

Wall clock of the whole command (startup included): median 1.684x (1.388-1.942).
Losses: A `1.576067 / 0.121023` every round; B `1.576113 / 0.120446` every round (the live
column drift explained below; deterministic run to run).

Reverse resume: the `7952ae2` checkout resumed B's `agot.state.pt` (saved groups carry
`fused: None`) and trained one more iteration normally (`policy_loss 1.5752`).

### Equivalence

- Levers 1-3 and 5: final weights of a full 965-step run7 epoch through the real `Trainer`
  are **bit-identical** to the `7952ae2` baseline (0 of 523 271 entries differ;
  `policy_loss 1.5760665541`, `value_loss 0.1210232100` in both). Optimizer state after the
  epoch: 0 subnormals (baseline: 761 900). Batches: all 965 run7 batches byte-identical.
  Replay restore: byte-identical buffer contents on run7.
- Lever 4: per-step gradients vs the dense path on 50 run7 batches: max abs difference
  1.4e-9, at most 3.5e-7 of each tensor's max. The dense path's own difference between 1 and
  2 torch threads on the same batches is 6.7e-6 abs (3.0e-5 relative), ~5000x larger. Over an
  epoch AdamW amplifies round-off in near-zero gradients into lr-sized steps, so trajectories
  separate: after 300 steps live vs dense max weight difference 0.116, dense 1 thread vs
  dense 2 threads 0.110 (`policy_loss` shifts in the 5th digit in both cases). The epoch's
  losses therefore no longer print the baseline's digits: `policy_loss 1.57611275`,
  `value_loss 0.12044641` against `1.57606655` / `0.12102321`. That is the run-to-run
  difference any thread-count or BLAS change already produces (Colab vs this box), not a
  change of the objective; `PolicyValueNet.live_columns = False` restores the bit-exact
  dense path.
- Activations, activation gradients and parameter gradients carry **no** subnormals (50 run7
  batches, every Linear), so FTZ matters only for the optimizer state.

## Verdict

Predicted >= 2x; measured 2.02x median on `seconds_train` (1.84-2.44x per round) in the
production CLI, and 2.18x x 1.17x x 1.21-1.25x in the attribution harness. The subnormal
lever paid as predicted in the optimizer (6x there with the gather), fused AdamW paid more
than predicted (4x on the step, because it also sidesteps MKL's slow `sqrt` on zeros), live
columns paid at the top of the predicted range. Levers 1, 2, 3, 5 are bit-identical; lever 4
is round-off per step and at the thread-count floor per epoch, and is the only reason the
`--games 0` losses no longer print the baseline's digits. Kept all five; rejected foreach
AdamW (0.98x), live columns on the state trunk (no measurable gain) and a lower snapshot zlib
level (size cost).

## Follow-ups

- Snapshot write: zlib level 1 writes 2.9x faster (0.56 s vs 1.61 s) but 3.3x bigger
  (28.3 MB vs 8.5 MB); on a Drive mount the upload likely eats it. Not changed. The lossless
  lever with no size cost is to compress in a background thread overlapping the next
  iteration's self-play — a `cli.py` change, outside this iteration's scope. `save` is also not
  atomic (direct write to the final path); a tmp + `os.replace` would match `model.save`.
- Live columns could drop the dead *output* units too, but only with a calibration of which
  units are dead, which changes behaviour when a unit revives. Not done.
- What is left of the pass (levers 1-4 in, per 965 steps): backward ~5 s, forward ~3.3 s,
  clip ~0.6 s, optimizer ~0.7 s, batch ~0.4 s at the fast clock. The backward of the
  T-row matmuls is the remaining mass.
- The 2.6 GHz-vs-4 GHz drift on this box swings a whole-epoch subprocess A/B by 0.9-1.4x per
  round; same-process batch interleaving is the only harness here that resolves a 10 % effect.
