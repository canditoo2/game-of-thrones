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
| 6 | House-card special abilities | Ignored (printed stats only) | All 42 implemented in `rust/crates/agot-core/src/house_cards.rs` | `151060549-House-Cards-Reference-Sheet.txt` |
| 7 | Neutral Forces | Not modelled | King's Landing (5) and The Eyrie (6) placed at 6-player setup | Rulebook p. 4 and p. 26 |
| 8 | House Martell's starting units | 1 Footman + 1 Knight in Sunspear, 1 Footman in Salt Shore, 1 Ship in East Summer Sea (inferred) | Sunspear 1 Footman + 1 Knight, Salt Shore 1 Footman, **1 Ship in the Sea of Dorne** | User, against the physical board |
| 9 | Six-player Influence tracks | Martell last on all three tracks (inferred) | Iron Throne: Baratheon, Lannister, Stark, Martell, Greyjoy, Tyrell; Fiefdoms: Greyjoy, Tyrell, Martell, Stark, Baratheon, Lannister; King's Court: Lannister, Stark, Martell, Baratheon, Tyrell, Greyjoy | User, against the physical board |
| 10 | Starred-order allowance | Flagged unverified | Confirmed `(3, 3, 2, 1, 0, 0)` | `chibimagic` `KingsCourtTrack::FIVE_SIX_PLAYER_SPECIAL_ORDERS`; printed on the board |
| 11 | Cross-house support | Support was automatic and only ever helped its owner | Combat opens with a Support step: each adjacent Support order's owner, in Iron Throne order, declares `attacker`, `defender` or `refuse`. A player may never support an opponent against their own units; Neutral Forces can never be supported | Rulebook pp. 17-18 and p. 26 |
| 12 | Siege Engine support | Counted only when the *supporting* area was a Castle/Stronghold | Counts only when the *embattled* area is a Castle/Stronghold, which is what the rulebook means | Rulebook p. 18 |
| 13 | Combat card-text / blade timing | Immediate House-card effects were applied with the final comparison, after the blade step | Card text and immediate effects resolve first; Doran can move the Fiefdoms leader before the blade decision. A new blade holder may use an unused blade; a blade used earlier remains used when it changes hands. Existing starred orders remain on the board when King's Court changes. | Rulebook combat steps; FFG forum archive topic 291514 |
| 14 | Siege Engines as casualties | A Siege Engine could be picked as a casualty, saving a Footman | Siege Engines cannot be chosen as casualties; the loser removes every participating Siege Engine automatically before picking the ordinary casualties | Official FFG responses, BGG thread 752416 |
| 15 | Battles against routed units | A province containing only routed units was not a battle target | The province still fights; routed units contribute 0 Combat Strength, and a routed unit can support at 0 plus the Support order's bonus. Routed units cannot be casualties and are destroyed automatically after a loss. Kevan/Victarion do not upgrade their 0 strength because their text says "instead of +1". | Official FFG responses, BGG thread 752416 |
| 16 | Wildling bid ties | Tied highest/lowest bidders were broken by Iron Throne position | After the card is revealed, the Iron Throne holder chooses among the tied bidders before rewards or penalties resolve. The choice is a real action in training and desktop play. | Official FFG responses, BGG thread 752416 |
| 17 | Salladhor Saan vs Victarion | Salladhor removed only the printed Ship strength, leaving Victarion's +1-per-Ship bonus | Salladhor's "reduce to 0" is an override: it removes both the printed Ship strength and Victarion's extra bonus; Order and House-card modifiers are untouched | Official FFG responses, BGG thread 752416 |
| 18 | Arianne Martell's token and Mace Tyrell's Order | Arianne could leave the defender's Power token in the area; an emptied area always released its Order | Arianne prevents occupation but the winner still removes the Order/Power token and resolves other winner effects. If Mace removes the last unit but the defender still wins, the defender's March/Consolidate Order remains on the empty area | Official FFG responses, BGG thread 752416 |
| 19 | House-card secrecy | The second chooser's PIMC search could see the first chooser's card | The second chooser's determinized worlds resample the first card (the physical reveal is simultaneous); the GUI/snapshot hides both cards until both players commit | Official FFG responses, BGG thread 752416 |
| 20 | Supply disbanding | Excess units were removed automatically (largest army, cheapest first) | The affected owner chooses one unit at a time from any controlled area, including routed units; the same pause handles Supply cards, Wildling Supply loss and other Supply enforcement paths | Rulebook p. 24 (Supply) |
| 21 | Wildling card reveal timing | The top Wildling card was drawn and exposed before the Power bids | It stays face down during the bids, is revealed before the Iron Throne tie-break, and resolves only then; PIMC draws its own card from the resampled deck | Rulebook Wildling attack steps; Messenger Raven: "Look in the Wildling Deck" |

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
  the Valyrian Steel Blade adds 1 strength and is spent once per round (turn).
* Strength ties are broken by **Fiefdoms** track position (defender wins an
  exact tie on strength *and* track).
* Victory at **7** castles/strongholds.
* Round 1 has no Westeros phase; rounds 2-10 draw one card from each deck.
* Home-area shields and printed ownership.
* Siege Engines contribute 4 only when attacking (or supporting an attack
  against) an area containing a Castle or Stronghold.
* Support is declared in Iron Throne order; a player may never support an
  opponent against their own units; a Support order is **not** consumed by
  declaring support and may help several battles in the same round. A Neutral
  Force can never receive support, but the attacker against one can.
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
dies, which area is retreated to, which card is played). Interactive play now
pauses for all of these choices through a generic choice queue. The extension's
`Game.new` leaves `choice_choices` off by default for the frozen fixtures;
Python training/evaluation and desktop play construct every game with
`choice_choices=True`, and the same actions are exposed to native self-play and
the neural policy. The low-level `combat.resolve_and_apply` test convenience
path (used only by the frozen regression corpus) retains deterministic
fallbacks; it is not the path used by the game or training.

Supply disbanding is offered one unit at a time, including separate ready and
routed copies of the same unit type, and is used after a Supply card, a Wildling
Supply loss and an interactive post-retreat check.

One printed "may" is deliberately not a decision: capturing a land area replaces
the enemy Port Ships up to the Port's capacity and the captor's Supply. Taking
fewer Ships has no upside (Ships add combat strength and never a penalty), so
the engine always takes the maximum rather than spending a policy action on a
dominated option.

### House cards

| Card | Real choice |
| ---- | ---------- |
| Renly Baratheon | If the winner may upgrade, choose a participating Footman or a supporting Baratheon Footman. |
| Tyrion Lannister | The opponent chooses a different card from their hand; with no other card they fight with none. |
| Cersei Lannister | The winner chooses which opponent Order to remove. |
| Robb Stark | The winner chooses the defender's retreat among the legal fewest-loss areas. |
| Aeron Damphair | The owner chooses whether to pay 2 Power and which replacement card to play. |
| Mace Tyrell | Destroys one opposing participating Footman; all participating Footmen are equivalent, so no sub-choice is needed. |
| Queen of Thorns | The owner chooses which adjacent opposing Order to remove, never the March origin. |
| Doran Martell | Three legal track choices. Desktop play prompts the card's owner; Python training/evaluation expose the same actions. Doran resolves before the Valyrian Steel Blade step, so a new holder may use an unused blade. |
| Patchface | The owner chooses which opponent card to discard, after the opponent has drawn back a recycled six-card hand. |

### Wildling cards

All player-facing sub-choices are real decisions in interactive play and
training:

* **Crow Killers:** which Knights to replace and which Footmen to upgrade.
* **The Horde Descends:** which Castle/Stronghold to use, which units are
  destroyed, and how the highest bidder musters.
* **King Beyond the Wall:** each other house chooses Fiefdoms or King's Court;
  the highest bidder chooses any Influence track.
* **Mammoth Riders:** which units are destroyed and which discarded House card
  is retrieved.
* **Massing at Milkwater:** each other house chooses which card to discard.
* **Preemptive Raid:** the lowest bidder chooses between destroying two units
  and dropping two positions on their highest Influence track.
* **Rattleshirt's Raiders:** Supply losses use the normal real Supply-disband
  queue.

A tied highest/lowest bidder is likewise a real decision for the Iron Throne
holder, taken after the card is revealed and before any reward or penalty
resolves.

## How to re-verify

The data is fully verified against the physical components. Everything can be
re-checked by running `pytest -q` (the map validator and the rule-fix
regressions) and by comparing `src/agot/data/*.json` against the sources listed
at the top of this file.

The rule-fix regression suite lives in Rust: every test in the retired
`tests/test_rules_fixes.py` has a counterpart in
`crates/agot-core/src/rules_fixes.rs`, named after its Python original. Running
`cargo test --release` from `rust/` exercises the whole audit trail, and it is
the gate that runs in CI's Rust job. The Python engine is frozen pending its
deletion, so its copies of these regressions were retired rather than kept in
parallel.
