# 001 — validate the fp32 leaf engine at run6 scale

Heavy by this machine's rules (a run6 iteration is tens of minutes), so it does not run here.
The loop's job is to make the card launchable.

## Question

Does `--leaf-backend auto` (onnx-fp32) hold its 1.5x self-play win at run6's batch size and
simulation count, and does the int8 engine still hold 2.4x there? Everything measured on this
PC used 1–8 leaf batches; run6 uses 16.

## Config diff versus run6

run6 launches (see `colab/agot-colab.ipynb`):

```
--simulations 128 --determinizations 1 --phase-budgets default --leaf-backend onnx-int8
--games 64 --batch-games 16 --buffer 80000 --gate-games 120 --gate-mode deferred
```

Card: run the same command three times, changing only `--leaf-backend`:

| arm | flag |
|---|---|
| A (baseline) | `--leaf-backend torch` |
| B | `--leaf-backend onnx-fp32` |
| C (what run6 already does) | `--leaf-backend onnx-int8` |

Same seeds, same `--checkpoint` stem per arm, 3 iterations each (the first is the warm one:
graph export + ORT session), 48–64 games.

## Expected effect (from the local measurements)

| arm | expected `seconds_selfplay` | expected `seconds_train` | expected KL vs A |
|---|---|---|---|
| A | 1.00x | 1.00x | — |
| B | 1.15–1.55x faster | unchanged | ≤ 1e-6 nats |
| C | 2.0–2.4x faster | unchanged | ~4e-4 nats (mean) |

At batch 16 the local leaf table says fp32 is only 1.17x over torch, so B may land near 1.2x.
That is the number this card exists to settle.

## Decision rule

- Keep `auto` = fp32 if B ≥ 1.15x on `seconds_selfplay` and the gate verdicts on the same
  seeds do not change sign.
- Promote int8 to the default only if C ≥ 2.0x **and** the KL stays under the 1e-3/5e-3 budget
  on a trained net (it does locally: mean 4.1e-4, p95 8.4e-4) **and** the owner accepts the
  numeric change — it is a learning-signal decision, not a speed one.

## Cost

3 arms × 3 iterations × 64 games ≈ 3 × 3 × (10–20 min) on a Colab CPU runtime: budget ~2–3 h.

## Launch cell

```python
for arm, backend in (("a", "torch"), ("b", "onnx-fp32"), ("c", "onnx-int8")):
    !cd /content/agot && python -m agot.cli train \
        --simulations 128 --determinizations 1 --phase-budgets default \
        --leaf-backend {backend} --games 64 --batch-games 16 --iterations 3 \
        --seed 11 --device cpu --buffer 80000 --buffer-save-every 0 \
        --eval-games 0 --gate-games 0 --gate-mode inline \
        --checkpoint /content/drive/MyDrive/agot/engine-ab/{arm}/agot.pt
```

Then read `seconds_selfplay` from each `agot.log.jsonl` (skip iteration 1) and compare medians.
