# Optimization ledger

Append-only. One row per iteration; the iteration file has the full story.
Baseline machine state and the measurement protocol are in `iterations/000-baseline.md`.
Branch: `optimization/speed-leaf-engine`, based on `b01c99a`.

| id | date | track | change | metric | before | after | delta | verdict | commit |
|---|---|---|---|---|---|---|---|---|---|
| 000 | 2026-11-11 | baseline | — | see `iterations/000-baseline.md` | — | — | — | — | `b01c99a` |
| 001 | 2026-11-11 | HYGIENE | lazy trace formatting (`note_args(format_args!)` at 73 sites) | full game, `cargo bench --bench clone_step` | 532 k decisions/s | 722 k | **1.36x** | kept | `a7a07c3` |
| 001 | 2026-11-11 | HYGIENE | same | clone + step | 639 ns | 520 ns | **1.22x** | kept | same |
| 002 | 2026-11-11 | SPEED | lean `forward_packed` under inference mode | leaf forward, 2 leaves | 397 µs | 303 µs | 1.17x | kept | `d804caf` |
| 002 | 2026-11-11 | SPEED | same | leaf forward, 1/4/8/16 leaves | 525/601/1015/1707 µs | 475/525/866/1591 | 1.20x/1.08x/1.12x/1.07x | kept | same |
| 003 | 2026-11-11 | SPEED | `--leaf-backend auto` = onnx-fp32 when the extra is installed; one graph export per iteration | leaf forward, 1–2 leaves | 475/303 µs | 243/177 µs | **2.15x / 2.24x** | kept | `d804caf` |
| 003 | 2026-11-11 | SPEED | same | self-play, 8 games/16 sims/batch 4 | 9.71 s torch | 6.35 s | **1.53x** | kept | same |
| 003 | 2026-11-11 | SPEED | same (opt-in int8, measured for the record) | self-play, same run | 9.71 s torch | 3.98 s | 2.44x | option, not default | same |
| 004 | 2026-11-11 | SPEED | lazy optimizer + background `torch._dynamo` import | meso wall clock (§5 command) | 8.89–11.00 s | 8.25–8.47 s | 1.06–1.30x | kept | same |
| — | 2026-11-11 | combined | 001+002+003+004 | meso wall clock, 5 rounds against `b01c99a` (`git checkout b01c99a -- src/agot`) | 10.09–11.18 s | 6.95–8.13 s | **1.36–1.51x (median 1.42x)** | — | `a9203cc` |
| — | 2026-11-11 | combined | 001+002+003+004 | `seconds_selfplay`, same rounds | 5.64–6.48 s | 3.25–4.39 s | **1.48–1.84x (median 1.60x)** | — | `a9203cc` |
| — | 2026-11-11 | combined | 001+002+003+004 | leaf forward, 1–2 leaves (the batch sizes gate/evaluate/play use) | 525/397 µs | 243/177 µs | **2.15x / 2.24x** | — | `a9203cc` |
| — | 2026-11-11 | combined | 001+002+003+004 | `agot evaluate`, 2 games / 8 sims (startup-dominated) | 3.87 s | 3.59 s | 1.08x | — | `a9203cc` |
| 005 | 2026-11-12 | SPEED | `auto` = **onnx-int8** for `agot train` (fp32 stays for evaluate/gauntlet, gate inherits) | meso wall clock, 5 interleaved rounds against `b01c99a` | 19.35–36.62 s | 12.74–23.12 s | **1.55x (median; mean 1.58x)** | kept | `b419cd7` |
| 005 | 2026-11-12 | SPEED | same | `seconds_selfplay`, same rounds | 10.71–23.68 s | 4.61–10.32 s | **2.46x (median; mean 2.42x)** | kept | `b419cd7` |
| 005 | 2026-11-12 | SPEED | same | self-play, 8 games/16 sims/**batch 8** | 31.35 s torch | 13.33 s | **2.35x** | kept | `b419cd7` |
| 005 | 2026-11-12 | SPEED | same | steady-state iteration (`sp` + `train`, startup amortised) | — | — | **~2.3x** | — | `b419cd7` |
| 005 | 2026-11-12 | SPEED | same | meso with 4 iterations (startup amortised 2x better), 3 rounds | 28.44–35.58 s | 19.05–19.67 s | 1.53x | — | `8eb7292` |
| 005 | 2026-11-12 | SPEED | same | `seconds_selfplay`, 4-iteration meso | 18.70–23.13 s | 7.11–7.58 s | **2.69x** | — | `8eb7292` |
| 006 | 2026-11-20 | SPEED | trainer pass: FTZ + subnormal scrub of Adam moments | optimizer step, run7 epoch, same-process interleaved (with 2) | 45.90 s | 7.66 s | 6.0x (bit-identical weights) | kept | `b522018` |
| 006 | 2026-11-20 | SPEED | same: batch gather from the buffer's packed arrays | batch assembly, run7 epoch | 7.05 s | 0.75 s | 9.4x (byte-identical batches) | kept | `ba7c23e` |
| 006 | 2026-11-20 | SPEED | levers 1+2 together | training pass, 965 steps, interleaved | 83.87 s | 38.54 s | **2.18x** | kept | `ba7c23e` |
| 006 | 2026-11-20 | SPEED | fused CPU AdamW (probed, kernel kept across load, saved as `fused=None`) | pass, 300 steps, on top of 1+2 | 4.18 s | 3.57 s | 1.17x (opt 4.0x, bit-identical) | kept | `df10927` |
| 006 | 2026-11-20 | SPEED | live-column training forward (action trunk L2, scorer action half) | pass, 965 steps, on top of 1-3 | 13.20 / 11.23 s | 10.53 / 9.28 s | 1.25x / 1.21x (round-off) | kept | `09f17a4` |
| 006 | 2026-11-20 | SPEED | replay restore adopts the saved arrays | run7 restore | 0.99-1.54 s | 0.69-0.84 s | 1.43-1.83x (byte-identical) | kept | `3e54423` |
| 006 | 2026-11-20 | SPEED | levers 1-5, production CLI (`ab_iter.py --games 0`, 4 rounds vs `7952ae2`) | `seconds_train` | 23.72 s (19.70-25.78) | 11.06 s (10.54-12.11) | **2.02x median** (1.84-2.44) | kept | `3e54423` |
| 006 | 2026-11-20 | SPEED | same | `seconds` (iteration) | 26.10 s | 13.84 s | 1.76x median | — | `3e54423` |

The combined rows were measured against `b01c99a` itself (`git checkout b01c99a -- src/agot`, run,
`git checkout HEAD -- src/agot`, run, five rounds) because this box's clock drifts by more than the
effect size between sessions. Read them as the authoritative before/after; the single-arm numbers
in `iterations/000-baseline.md` were taken in an earlier session and are only comparable inside it.

## Rejected / no effect (worth remembering)

| date | track | idea | measurement | verdict |
|---|---|---|---|---|
| 2026-11-11 | SPEED | int8 as the default leaf engine | 2.5–3.4x on leaves, 2.4x self-play, but KL mean 4.1e-4 / p95 8.4e-4 on a trained net and different game records | rejected as a *universal* default (evaluate/gauntlet stay on fp32 for comparability); adopted as the **training** default in 005, which is what run6 already selects |
| 2026-11-11 | SPEED | `do_constant_folding=False` on the ONNX export | export 0.31 s → 0.05 s, but inference 179 µs → 192 µs at 2 leaves | rejected |
| 2026-11-11 | SPEED | per-thread tuning of the in-process leaf forward | 1 vs 2 vs 4 torch threads swings ±15 % and changes sign with the CPU clock | rejected (not reproducible) |
| 2026-11-11 | SPEED | planning-phase micro-optimisations | planning line did not move with 001 (1.98–2.30 M/s across both builds) | nothing to win there |
| 2026-11-20 | SPEED | `foreach=True` AdamW on CPU (torch 2.14) | pass, 300 steps, interleaved: 0.98x, opt 0.98x | rejected |
| 2026-11-20 | SPEED | live columns on the 64-row state trunk as well | 1.26x vs 1.21-1.25x without it: inside the noise | rejected |
| 2026-11-20 | SPEED | replay snapshot at zlib level 1 | write 1.61 s -> 0.56 s, file 8.5 MB -> 28.3 MB (Drive upload cost) | rejected |
| 2026-11-20 | SPEED | `np.add.reduceat` for the per-example policy sums | not bit-identical to `ndarray.sum` (pairwise/SIMD) | rejected |
