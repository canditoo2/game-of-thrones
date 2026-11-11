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

The combined rows were measured against `b01c99a` itself (`git checkout b01c99a -- src/agot`, run,
`git checkout HEAD -- src/agot`, run, five rounds) because this box's clock drifts by more than the
effect size between sessions. Read them as the authoritative before/after; the single-arm numbers
in `iterations/000-baseline.md` were taken in an earlier session and are only comparable inside it.

## Rejected / no effect (worth remembering)

| date | track | idea | measurement | verdict |
|---|---|---|---|---|
| 2026-11-11 | SPEED | int8 as the default leaf engine | 2.5–3.4x on leaves, 2.4x self-play, but KL mean 4.1e-4 / p95 8.4e-4 on a trained net and different game records | rejected as a default; kept as `--leaf-backend onnx-int8`, which run6 already uses |
| 2026-11-11 | SPEED | `do_constant_folding=False` on the ONNX export | export 0.31 s → 0.05 s, but inference 179 µs → 192 µs at 2 leaves | rejected |
| 2026-11-11 | SPEED | per-thread tuning of the in-process leaf forward | 1 vs 2 vs 4 torch threads swings ±15 % and changes sign with the CPU clock | rejected (not reproducible) |
| 2026-11-11 | SPEED | planning-phase micro-optimisations | planning line did not move with 001 (1.98–2.30 M/s across both builds) | nothing to win there |

