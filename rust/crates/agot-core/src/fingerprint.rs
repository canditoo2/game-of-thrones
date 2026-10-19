//! Canonical fingerprints for differential testing against the Python engine.
//!
//! The port is validated by replaying an action script recorded from the Python
//! engine and comparing a full state fingerprint after every step. The
//! fingerprint must therefore be *structurally identical* on both sides; the
//! Python counterpart lives in `tools/dump_fixture.py` and the two are pinned
//! against each other by `tests/planning_parity.rs`.
//!
//! Decks are compared as sorted multisets: the two engines shuffle with
//! different generators, so draw *order* cannot match, while contents and the
//! draw cursor can and must.

use serde_json::{json, Map, Value};

use crate::action::Action;
use crate::enums::{HouseId, Track, UnitType};
use crate::map::DataSet;
use crate::state::State;

/// `kind|house|region|target|order|card|unit|side|amount|units|establish`
///
/// Empty fields for absent payload, units as comma-joined counts, `1`/`0` for
/// the establish flag. Mirrors `canonical_action` in `tools/dump_fixture.py`.
pub fn canonical_action(action: &Action, data: &DataSet) -> String {
    let house = action
        .house
        .map(|house| house.name().to_string())
        .unwrap_or_default();
    let region = action
        .region
        .map(|region| data.map.id_of(region).to_string())
        .unwrap_or_default();
    let target = action
        .target
        .map(|target| data.map.id_of(target).to_string())
        .unwrap_or_default();
    let order = action
        .order
        .map(|order| order.key().to_string())
        .unwrap_or_default();
    let card = match (action.house, action.card) {
        (Some(house), Some(card)) => data.house_cards.card(house, card).id.clone(),
        _ => String::new(),
    };
    let unit = action
        .unit
        .map(|unit| unit.as_str().to_string())
        .unwrap_or_default();
    let side = action
        .side
        .map(|side| side.as_str().to_string())
        .unwrap_or_default();
    let amount = action
        .amount
        .map(|amount| amount.to_string())
        .unwrap_or_default();
    let units = action
        .units
        .map(|counts| {
            counts
                .iter()
                .map(|count| count.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();

    [
        action.kind.as_str().to_string(),
        house,
        region,
        target,
        order,
        card,
        unit,
        side,
        amount,
        units,
        if action.establish { "1" } else { "0" }.to_string(),
    ]
    .join("|")
}

fn house_name(house: Option<HouseId>) -> Value {
    match house {
        Some(house) => Value::String(house.name().to_string()),
        None => Value::Null,
    }
}

/// 64-bit FNV-1a of a string's UTF-8 bytes, as 16 hex digits.
///
/// The differential fixture stores this hash of the canonical legal-action list
/// for decisions whose full list would bloat the file. It mirrors `fnv1a64` in
/// `tools/dump_fixture.py`; the two must agree bit for bit.
pub fn fnv1a64(text: &str) -> String {
    let mut digest: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in text.as_bytes() {
        digest ^= *byte as u64;
        digest = digest.wrapping_mul(0x0000_0100_0000_01B3);
    }
    format!("{digest:016x}")
}

/// A complete, comparable description of a position.
pub fn state_fingerprint(state: &State, data: &DataSet) -> Value {
    let mut root = Map::new();

    root.insert("round".into(), json!(state.round));
    root.insert("phase".into(), json!(state.phase.as_str()));
    root.insert("action_step".into(), json!(state.action_step.as_str()));
    root.insert("active_house".into(), house_name(state.active_house));
    root.insert("winner".into(), house_name(state.winner));
    root.insert(
        "action_house_last".into(),
        house_name(state.action_house_last),
    );
    root.insert("raven_used".into(), json!(state.raven_used));
    root.insert(
        "return_phase".into(),
        json!(state.return_phase.map(|phase| phase.as_str())),
    );

    root.insert(
        "pending_planning".into(),
        json!(state
            .pending_planning
            .iter()
            .map(|house| house.name())
            .collect::<Vec<_>>()),
    );
    root.insert(
        "order_restrictions".into(),
        json!(state
            .order_restrictions
            .iter()
            .map(|restriction| match restriction {
                crate::state::Restriction::Key(key) => key.key().to_string(),
                crate::state::Restriction::Type(order_type) => order_type.as_str().to_string(),
            })
            .collect::<Vec<_>>()),
    );
    let mut pending_bids = Map::new();
    for (house, amount) in state.pending_bids.iter() {
        pending_bids.insert(house.name().to_string(), json!(amount));
    }
    root.insert("pending_bids".into(), Value::Object(pending_bids));
    root.insert(
        "bid_tracks".into(),
        json!(state
            .bid_tracks
            .iter()
            .map(|track| track.as_str())
            .collect::<Vec<_>>()),
    );
    // Sorted: the Python reference keeps fortifications in a frozenset, so it
    // has no order to preserve and sorts them for comparison.
    let mut fortifications: Vec<&str> = state
        .fortifications
        .iter()
        .map(|region| data.map.id_of(*region))
        .collect();
    fortifications.sort_unstable();
    root.insert("fortifications".into(), json!(fortifications));

    // Tracks, in resolution order.
    let mut tracks = Map::new();
    for track in Track::ALL {
        tracks.insert(
            track.as_str().to_string(),
            json!(state.tracks[track.index()]
                .iter()
                .map(|house| house.name())
                .collect::<Vec<_>>()),
        );
    }
    root.insert("tracks".into(), Value::Object(tracks));

    // Houses, in canonical order.
    let mut houses = Map::new();
    for house in HouseId::ALL {
        let Some(house_state) = state.house(house) else {
            continue;
        };
        houses.insert(
            house.name().to_string(),
            json!({
                "power": house_state.power,
                "supply_level": house_state.supply_level,
                "orders_available": house_state.orders_available.iter().map(|order| order.key()).collect::<Vec<_>>(),
                "orders_placed": house_state.orders_placed.iter().map(|order| order.key()).collect::<Vec<_>>(),
                "cards_available": house_state.house_cards_available.iter().map(|card| data.house_cards.card(house, *card).id.as_str()).collect::<Vec<_>>(),
                "cards_used": house_state.house_cards_used.iter().map(|card| data.house_cards.card(house, *card).id.as_str()).collect::<Vec<_>>(),
                "blade_used": house_state.blade_used,
                "passed": house_state.passed,
                "eliminated": house_state.eliminated,
            }),
        );
    }
    root.insert("houses".into(), Value::Object(houses));

    // Board, in map order.
    let mut regions = Map::new();
    for region_id in state.region_ids() {
        let region = state.region(region_id);
        let mut units = Map::new();
        let mut routed = Map::new();
        for unit in UnitType::ALL {
            units.insert(unit.as_str().to_string(), json!(region.unit_count(unit)));
            routed.insert(unit.as_str().to_string(), json!(region.routed_count(unit)));
        }
        regions.insert(
            data.map.id_of(region_id).to_string(),
            json!({
                "controller": house_name(region.controller),
                "owner": house_name(region.owner),
                "units": Value::Object(units),
                "routed": Value::Object(routed),
                "neutral_force": region.neutral_force,
                "garrison": region.garrison,
                "mustering_points": region.mustering_points,
                "influence_token": house_name(region.influence_token),
                "order": region.order.map(|order| order.key()),
                "order_house": house_name(region.order_house),
            }),
        );
    }
    root.insert("regions".into(), Value::Object(regions));

    root.insert("wildling_threat".into(), json!(state.wildling_threat));
    let mut wildling_deck: Vec<&str> = state
        .wildling_deck
        .cards
        .iter()
        .map(|card| data.wildlings.card_id(*card))
        .collect();
    wildling_deck.sort_unstable();
    root.insert("wildling_deck".into(), json!(wildling_deck));
    let mut wildling_discard: Vec<&str> = state
        .wildling_discard
        .iter()
        .map(|card| data.wildlings.card_id(*card))
        .collect();
    wildling_discard.sort_unstable();
    root.insert("wildling_discard".into(), json!(wildling_discard));

    let mut decks = Map::new();
    for (position, name) in crate::map::WesterosData::DECK_NAMES.iter().enumerate() {
        let deck = &state.westeros_decks[position];
        let mut cards: Vec<&str> = deck
            .cards
            .iter()
            .map(|card| data.westeros.card_id(*card))
            .collect();
        cards.sort_unstable();
        decks.insert(
            (*name).to_string(),
            json!({
                "cards": cards,
                "position": deck.position,
                "drawn": deck.drawn.map(|card| data.westeros.card_id(card)),
            }),
        );
    }
    root.insert("westeros_decks".into(), Value::Object(decks));

    root.insert(
        "pending_westeros".into(),
        match state.pending_westeros.as_ref() {
            Some(queue) => json!(queue
                .iter()
                .map(|card| data.westeros.card_id(*card))
                .collect::<Vec<_>>()),
            None => Value::Null,
        },
    );
    root.insert(
        "westeros_choice".into(),
        match state.westeros_choice.as_ref() {
            Some(choice) => json!({
                "card": data.westeros.card_id(choice.card),
                "holder": choice.holder.name(),
                "options": choice
                    .options
                    .iter()
                    .map(|card| {
                        // The "no effect" option has no card row; Python keeps
                        // the literal string, so the fingerprint must too.
                        if *card == crate::state::NONE_OPTION {
                            "none".to_string()
                        } else {
                            data.westeros.card_id(*card).to_string()
                        }
                    })
                    .collect::<Vec<_>>(),
            }),
            None => Value::Null,
        },
    );
    root.insert(
        "pending_march".into(),
        match state.pending_march.as_ref() {
            Some(march) => json!({
                "house": march.house.name(),
                "origin": data.map.id_of(march.origin),
            }),
            None => Value::Null,
        },
    );
    root.insert(
        "pending_battle_present".into(),
        json!(state.pending_battle.is_some()),
    );
    root.insert(
        "wildling_attack_present".into(),
        json!(state.wildling_attack.is_some()),
    );

    Value::Object(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{Action, ActionKind};
    use crate::enums::{OrderKey, RegionId};

    #[test]
    fn canonical_actions_are_stable_strings() {
        let data = DataSet::load_default().unwrap();
        let region = data.map.id_of(RegionId(0));

        let action = Action::place_order(HouseId(2), RegionId(0), OrderKey(2));
        let expected = [
            "place_order",
            "stark",
            region,
            "",
            "march_plus",
            "",
            "",
            "",
            "",
            "",
            "0",
        ]
        .join("|");
        assert_eq!(canonical_action(&action, &data), expected);

        let pass = Action::pass(HouseId(1));
        let expected = ["pass", "lannister", "", "", "", "", "", "", "", "", "0"].join("|");
        assert_eq!(canonical_action(&pass, &data), expected);

        let bid = Action::bid(HouseId(0), 4);
        let expected = ["bid", "baratheon", "", "", "", "", "", "", "4", "", "0"].join("|");
        assert_eq!(canonical_action(&bid, &data), expected);
    }

    #[test]
    fn fingerprint_covers_the_board() {
        let data = DataSet::load_default().unwrap();
        let state = crate::state::State::blank();
        let value = state_fingerprint(&state, &data);
        assert_eq!(value["phase"], "westeros");
        assert_eq!(value["regions"].as_object().unwrap().len(), 0);
        assert_eq!(value["houses"].as_object().unwrap().len(), 0);
        assert!(value["tracks"]["iron_throne"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn canonical_action_kind_names_are_python_names() {
        assert_eq!(ActionKind::PlaceOrder.as_str(), "place_order");
        assert_eq!(ActionKind::ChooseCasualty.as_str(), "choose_casualty");
    }
}
