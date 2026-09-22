# Rulebook fidelity audit

This document records **what was checked, what was changed, and what is still
uncertain** in the engine's rules and data. It is the narrative companion to
[`../PROVENANCE.md`](../PROVENANCE.md): PROVENANCE is the quick confidence
table, this file is the audit trail.

The project policy is unchanged: *implement something honest and flag what is
uncertain, rather than invent a plausible-sounding rule.*

## Sources used

* **Official 2E rulebook** — combat, casualties, tie-breaks, supply, round
  structure, setup, Wildling attacks (pp. 22-23), Neutral Forces (p. 26).
* **Bundled scans** in `Rulebook/`:
  * `board.jpg` — per-region power / supply / mustering icons and adjacency.
  * `Westeros_card.txt` — the 15 Westeros cards, their per-deck deal and the
    three holders'-choice effects.
  * `151060549-House-Cards-Reference-Sheet.txt` — all 42 House card names,
    printed stats and ability text.
  * `wildling-cards1..3.webp` — the nine Wildling cards.
* **Community implementations** as cross-references:
  `chibimagic/got-board-game` (`app/models/`), `Robin59/Got-BoardGame`,
  `buhanec/pygot`.

Local copies of the copyrighted scans are kept out of version control; only the
facts extracted from them are stored.

## Corrections made

| # | What | Before | After | Source |
| - | ---- | ------ | ----- | ------ |
| 1 | The Mountains of the Moon icon | Power 1, Supply 0 | Power 0, Supply 1 | `board.jpg`; matches `chibimagic` `area.rb` |
| 2 | House Stark starting Supply | 2 | **1** (every other house stays 2) | Physical player screen, confirmed by the user |
| 3 | House Martell House cards | Missing Darkstar; Red Viper 0, Areo Hotah 2, Obara 2, Nymeria 1, Arianne 4, Doran 3; included "Ellaria Sand" | Red Viper 4 (2 sword + 1 fort), Areo Hotah 3 (fort), Obara Sand 2 (sword), Darkstar 2 (sword), Nymeria Sand 1, Arianne Martell 1, Doran Martell 0 | `151060549-House-Cards-Reference-Sheet.txt` |
| 4 | Lannister setup (reverted) | 1 Ship in The Golden Sound only | Briefly added a port Ship from the FAQ v2.0 errata; removed again at the user's request, so the base setup stands | Physical setup, per the user |
| 5 | Holders'-choice Westeros cards | Resolved as "no effect" | Real decisions for the named Influence token's holder | `Westeros_card.txt` |
| 6 | House-card special abilities | Ignored (printed stats only) | All 42 implemented in `engine/house_cards.py` | `151060549-House-Cards-Reference-Sheet.txt` |
| 7 | Neutral Forces | Not modelled | King's Landing (5) and The Eyrie (6) placed at 6-player setup | Rulebook p. 4 and p. 26 |
| 8 | House Martell's starting units | 1 Footman + 1 Knight in Sunspear, 1 Footman in Salt Shore, 1 Ship in East Summer Sea (inferred) | Sunspear 1 Footman + 1 Knight, Salt Shore 1 Footman, **1 Ship in the Sea of Dorne** | User, against the physical board |
| 9 | Six-player Influence tracks | Martell last on all three tracks (inferred) | Iron Throne: Baratheon, Lannister, Stark, Martell, Greyjoy, Tyrell; Fiefdoms: Greyjoy, Tyrell, Martell, Stark, Baratheon, Lannister; King's Court: Lannister, Stark, Martell, Baratheon, Tyrell, Greyjoy | User, against the physical board |
| 10 | Starred-order allowance | Flagged unverified | Confirmed `(3, 3, 2, 1, 0, 0)` | `chibimagic` `KingsCourtTrack::FIVE_SIX_PLAYER_SPECIAL_ORDERS`; printed on the board |

Correction 2 also carries a (non-mechanical) fact: House Stark is the only house
to open with **both** a Stronghold (Winterfell) and a Castle (White Harbor).

## Verified correct — do not "fix" these

These were checked against the rulebook and/or two independent references and
are right:

* Unit combat strengths: Footman 1, Knight 2, Siege Engine 4 (contextual), Ship 1.
* 15 Order tokens per house and the supply / army limits table.
* Mustering costs (Footman 1, Knight 2, Ship 1, Siege Engine 2).
* Garrison strength 2; a port holds at most 3 Ships.
* Casualties = **swords on the winner's card minus forts on the loser's card**;
  the Valyrian Steel Blade adds 1 strength and is spent once per battle.
* Strength ties are broken by **Fiefdoms** track position (defender wins an
  exact tie on strength *and* track).
* Victory at **7** castles/strongholds.
* Round 1 has no Westeros phase; rounds 2-10 draw one card from each deck.
* Home-area shields and printed ownership.
* Siege Engines contribute 4 only when attacking (or supporting an attack
  against) an area containing a Castle or Stronghold.
* The 15 Westeros card names and the per-deck deal (each deck = 10 cards); the
  cards carrying a Wildling icon; *Winter is Coming* reshuffles and redraws.
* The nine Wildling card names and their effects.
* Base-game setup: 5 Power tokens per house.
* A defeated defender's surviving units **retreat**; a routed unit forced to
  retreat again is destroyed; the attacker's and defender's unit pools stay
  separate until resolution.

An earlier plan to model a "house elimination" rule was **dropped**: the base 2E
rulebook has no elimination rule, so the `eliminated` field stays unused.

## Remaining uncertainty

No data is currently flagged as unverified. The starting positions, the
six-player Influence tracks, the King's Court starred-order allowance, House
Stark's starting Supply, the House card data and the Westeros deck deal have all
been confirmed.

Everything in the table above is verified at **High** confidence.

## Fidelity of the multi-choice effects

The physical game asks a player to choose inside several effects (which unit
dies, which area is retreated to, which card is played). The engine keeps the
game *decidable* by auto-resolving those sub-choices with deterministic
defaults. The chosen defaults are:

### House cards

| Card | Sub-choice and default |
| ---- | ---------------------- |
| Renly Baratheon | Upgrade a participating Footman if present, otherwise a supporting Baratheon Footman in a neighbouring Support area. |
| Tyrion Lannister | The opponent plays their strongest remaining card; the ability is used only when that card's value is lower than the one being replaced. With no other card, the opponent fights with none. |
| Cersei Lannister | Removes the first opponent Order found scanning regions in id order. |
| Robb Stark | Picks the retreat destination that keeps the loser within Supply, otherwise the first legal area. |
| Aeron Damphair | Pays 2 Power and swaps in the strongest card when it is worth more than Aeron (0). |
| Mace Tyrell | Destroys one opposing participating Footman. |
| Queen of Thorns | Removes the first opposing Order in a neighbouring area (id order), never the attacker's origin March. |
| Doran Martell | Moves the opponent down the track on which they rank highest. |
| Patchface | Discards the opponent's strongest card in hand. |

All other card abilities are fully determined (no sub-choice).

### Wildling cards

The unit(s) destroyed, the Influence track moved to and the House card
discarded are chosen by a deterministic default; a tie for highest or lowest
bidder is broken by **Iron Throne position** rather than asking the holder.

## How to re-verify

The three flagged values above are the only things that need the physical
components. Everything else can be re-checked by running `pytest -q` (the map
validator and the `test_rules_fixes.py` regression suite) and by comparing
`src/agot/data/*.json` against the sources listed at the top of this file.
