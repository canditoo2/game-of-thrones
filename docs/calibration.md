# Causal asset weights

The reward shaping needs the marginal value of each board asset. The
correlational fit over self-play states (run6 buffer, 76,926 examples) gives the
orders of magnitude, not causal effects: forts correlate with winning because
winners end with forts, not only because forts cause wins. This calibration
measures the effects directly.

## Design

One seed is a block. The same game is played twice, once on the packaged data
and once on a copy whose only change is one asset for one house. Both arms use
the same seed, the same greedy policy seeds and the same rules, so the setup
draws are shared and the difference in the treated house's win rate is the
effect of that one asset.

- The **control is common**: one control game per seed serves every condition.
  With eight conditions a seed costs nine games, not sixteen.
- The **treated house rotates** with the seed (`seed % 6`), so the pooled
  estimate is not one house's geography. `weights.json` also reports each
  house separately.
- The perturbations are **data-only**: the data directory is copied and one
  JSON field is changed (`map_base_6p.json` for mustering/icons, `houses.json`
  for power, supply and units). The engine is untouched and the RNG stream is
  identical; a pairing check verifies that the post-setup state is byte-equal
  for map-only changes.

| condition | change | what it measures |
|---|---|---|
| `fort` | a footman on a neutral castle (crackclaw_point) | positive control, ~1/7 |
| `castle` | home stronghold (mustering 2) → castle (1) | the legal castle ↔ stronghold step |
| `castle_supply` | `castle` + one supply icon | the mustering × supply interaction |
| `power_icon` | +1 printed crown on the home region | one Consolidate/Game of Thrones icon |
| `supply_icon` | +1 printed supply icon on the home region | the Supply-track icon |
| `power_token` | +1 starting Power | one Power token |
| `supply_step` | +1 starting Supply level | one Supply step |
| `footman` | +1 starting Footman at home | one unit (1 muster point) |
| `knight` | +1 starting Knight at home | one unit (2 muster points) |
| `placebo` | identical copy | must be exactly 0 |

The home is a stronghold for every house, so `+1 mustering` (2 → 3) is not the
castle→stronghold step: it leaves the printed board, breaks the stronghold
tiebreak key and pushes the encoder's `mustering/2` feature to 1.5, outside its
training range. The calibration therefore moves the home *down* to a castle,
which is the direction the real board uses, and crosses it with a supply icon.

## Estimator

For seed `i`: `Y_i(0), Y_i(1) ∈ {0,1}`, `D_i = Y_i(1) − Y_i(0) ∈ {−1,0,+1}`,
`Δ = (b − c)/N` with `b`/`c` the discordant pairs, `q = (b + c)/N`,
`SE = sqrt(q − Δ²)/sqrt(N)`, Wilson-style 95% interval, exact McNemar p. Holm
over the five substantial conditions; `fort` and `placebo` are validation.

`q` is the central diagnostic: if the perturbation makes the games diverge
immediately, `q` approaches `2p(1−p)` and the paired design buys nothing. The
script prints it for every condition.

## Running

The Colab notebook is `colab/agot-calibra.ipynb`; the runner is
`tools/calibrate_weights.py`:

```bash
python tools/calibrate_weights.py --data-dir src/agot/data \
    --out /content/drive/MyDrive/agot/calibration \
    --seeds 800 --max-seeds 2000 --workers 2 --target-half-width 0.02
```

It resumes from `seeds.jsonl` after a disconnect and stops extending once every
substantial condition has a 95% half-width below the target (or at the cap).
On two Colab vCPUs the measured cost is about 0.25 s per game effective, so
800 seeds (nine games each) take roughly 30 minutes.

## Output

`weights.json` has, per condition: `n`, `p_control`, `p_treated`, `delta`,
`b`, `c`, `q`, `se`, `ci95`, `p_mcnemar`, `p_holm`,
`victory_units_rules` (delta ÷ 1/7) and `ratio_to_fort` (delta ÷ measured fort
effect, bootstrapped over seeds), plus a per-house breakdown.

## Validation

- **Pairing check** (runs automatically): a map-only perturbation must leave the
  post-setup state byte-identical.
- **Placebo**: an identical data copy must give exactly 0, otherwise the engine
  or the loop is not deterministic.
- **Positive control**: the neutral castle should land near +14.3 pp. If it is
  near 0 the perturbation is not reaching the rules; if it is far above, the
  extra footman and mustering are doing the work and the number is not a pure
  fortification value.

## Known limits

- The policy is the greedy heuristic, not the networked agent. It is fast and
  fixed, which is what the design needs, but an asset a smart policy would
  convert better (a printed crown) is undervalued. Re-run the key conditions
  with the net on a subsample before treating the icon weights as final.
- In the greedy probe every game ended at the round-10 tiebreak and no house
  reached seven fortifications. The win rate is then partly a tiebreak weight
  (castles first, then strongholds, supply, power), which is exactly why the
  output also reports the final castles/strongholds/supply/power deltas: read
  them together with the win delta.
- The estimate is a local derivative at the setup position, not a global value:
  +1 supply with a saturated army is not the same as +1 early.
- The castle is always `crackclaw_point`; its position is a fixed ingredient of
  the positive control and not of the other weights.
