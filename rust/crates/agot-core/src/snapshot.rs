//! Build an engine state from a logged game snapshot.
//!
//! The human-game archive records one complete position per round: units,
//! orders, garrisons, and — for the newer logs — the Influence tracks, power,
//! Supply and House-card states. That is enough for the state encoder, so the
//! network can be evaluated (and trained) on human positions without replaying
//! the whole game.
//!
//! This is deliberately *not* a second rules engine. It fills the fields the
//! encoder reads and leaves the rest at their blank defaults.

use serde::Deserialize;
use smallvec::SmallVec;

use crate::enums::{ActionStep, CardId, HouseId, OrderKey, Phase, Track, UnitType, N_HOUSES};
use crate::map::DataSet;
use crate::state::{House, State};

/// The compact snapshot written by `tools/analyze_human_games.py`.
#[derive(Clone, Debug, Deserialize)]
pub struct Snapshot {
    #[serde(default = "default_round")]
    pub round: u8,
    #[serde(default)]
    pub wildling: u8,
    #[serde(default)]
    pub tracks: Tracks,
    #[serde(default)]
    pub houses: Vec<SnapshotHouse>,
    #[serde(default)]
    pub regions: Vec<SnapshotRegion>,
}

fn default_round() -> u8 {
    1
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Tracks {
    #[serde(default)]
    pub iron_throne: Vec<String>,
    #[serde(default)]
    pub fiefdoms: Vec<String>,
    #[serde(default)]
    pub kings_court: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SnapshotHouse {
    pub id: String,
    #[serde(default)]
    pub supply: u8,
    #[serde(default)]
    pub power: u8,
    #[serde(default)]
    pub land: u8,
    #[serde(default)]
    pub vp: u8,
    #[serde(default)]
    pub cards: Vec<(String, u8)>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SnapshotRegion {
    pub id: String,
    #[serde(default)]
    pub units: Vec<(String, String)>,
    #[serde(default)]
    pub order: Option<String>,
    #[serde(default)]
    pub garrison: Option<u8>,
}

fn unit_type(name: &str) -> Option<UnitType> {
    match name {
        "footman" => Some(UnitType::Footman),
        "knight" => Some(UnitType::Knight),
        "siege-engine" | "siege" => Some(UnitType::Siege),
        "ship" => Some(UnitType::Ship),
        _ => None,
    }
}

/// Map a Swords and Ravens order id onto the engine's order token.
///
/// The encoder only reads the number of placed orders, but keeping the real
/// token makes the imported state useful for anything stronger later.
fn order_key(name: &str) -> Option<OrderKey> {
    let key = match name {
        "march" => "march",
        "march-minus-one" => "march_minus",
        "march-plus-one" => "march_plus",
        "defense-plus-one" => "defend_1a",
        "defense-plus-two" => "defend_2",
        "support" => "support_a",
        "support-plus-one" => "support_plus",
        "raid" => "raid_a",
        "consolidate-power" => "consolidate_a",
        "special-consolidate-power" => "consolidate_star",
        _ => return None,
    };
    OrderKey::from_name(key)
}

/// Build the state an encoder reads from one round snapshot.
pub fn state_from_snapshot(data: &DataSet, snapshot: &Snapshot) -> Result<State, String> {
    let map = &data.map;
    let mut state = State::blank();
    state.max_rounds = 10;
    state.victory_castles = 7;
    state.region_count = map.region_count() as u8;
    state.fortifications = map.fortifications.clone();
    state.round = snapshot.round.max(1);
    state.wildling_threat = snapshot.wildling;
    // A reveal is the start of the Action Phase, before the first Raid order.
    state.phase = Phase::Action;
    state.action_step = ActionStep::Raid;

    for house in &snapshot.houses {
        let id =
            HouseId::from_name(&house.id).ok_or_else(|| format!("unknown house {:?}", house.id))?;
        let available = house.cards.iter().filter(|(_, state)| *state == 0).count();
        let mut entry = House::new(available.min(7), house.power, house.supply);
        entry.house_cards_available.clear();
        for index in 0..available.min(7) {
            entry.house_cards_available.push(CardId(index as u8));
        }
        entry.eliminated = house.land == 0;
        state.houses[id.index()] = Some(entry);
    }

    for region in &snapshot.regions {
        let region_id = map
            .lookup(&region.id)
            .ok_or_else(|| format!("unknown region {:?}", region.id))?;
        let mut controller = None;
        for (unit, house) in &region.units {
            let unit = unit_type(unit).ok_or_else(|| format!("unknown unit {:?}", unit))?;
            let house =
                HouseId::from_name(house).ok_or_else(|| format!("unknown house {:?}", house))?;
            let slot = &mut state.regions[region_id.index()];
            slot.units[unit.index()] = slot.units[unit.index()].saturating_add(1);
            controller = Some(house);
        }
        if controller.is_none() {
            if let Some(home) = map.region(region_id).home {
                if state.house(home).is_some() {
                    controller = Some(home);
                }
            }
        }
        let key = region.order.as_deref().and_then(order_key);
        {
            let slot = &mut state.regions[region_id.index()];
            slot.controller = controller;
            if let Some(garrison) = region.garrison {
                slot.garrison = garrison;
            }
            slot.order = key;
            slot.order_house = controller;
        }
        if let (Some(key), Some(house)) = (key, controller) {
            if let Some(entry) = state.house_mut(house) {
                entry.orders_placed.push(key);
            }
        }
    }

    let tracks = [
        &snapshot.tracks.iron_throne,
        &snapshot.tracks.fiefdoms,
        &snapshot.tracks.kings_court,
    ];
    for (track, names) in Track::ALL.into_iter().zip(tracks) {
        let mut order: SmallVec<[HouseId; N_HOUSES]> = SmallVec::new();
        for name in names {
            let Some(house) = HouseId::from_name(name) else {
                continue;
            };
            if state.house(house).is_some() && !order.contains(&house) {
                order.push(house);
            }
        }
        for house in HouseId::ALL {
            if state.house(house).is_some() && !order.contains(&house) {
                order.push(house);
            }
        }
        state.tracks[track.index()] = order;
    }
    // The first decision after a reveal belongs to the first player in Iron
    // Throne order who has an order in the current step. Walk the steps
    // (Raid, March, Consolidate) so the imported position has real legal
    // actions for the encoder's value context.
    state.active_house = state.tracks[Track::IronThrone.index()].first().copied();
    'steps: for step in ActionStep::ALL {
        for house in state.tracks[Track::IronThrone.index()].iter().copied() {
            if !state.orders_of_type(house, step.order_type()).is_empty() {
                state.action_step = step;
                state.active_house = Some(house);
                break 'steps;
            }
        }
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::{encode_state, STATE_DIM};

    fn snapshot() -> Snapshot {
        serde_json::from_str(
            r#"{
              "round": 3,
              "wildling": 4,
              "tracks": {
                "iron_throne": ["stark", "lannister"],
                "fiefdoms": ["lannister", "stark"],
                "kings_court": ["stark", "lannister"]
              },
              "houses": [
                {"id": "stark", "supply": 1, "power": 6, "land": 3, "vp": 2,
                 "cards": [["eddard-stark", 0], ["robb-stark", 0], ["catelyn-stark", 1]]},
                {"id": "lannister", "supply": 2, "power": 4, "land": 2, "vp": 1, "cards": []}
              ],
              "regions": [
                {"id": "winterfell", "units": [["footman", "stark"], ["knight", "stark"]],
                 "order": "march", "garrison": 2},
                {"id": "lannisport", "units": [["footman", "lannister"]],
                 "order": "special-consolidate-power"},
                {"id": "the_golden_sound", "units": [["ship", "lannister"]]}
              ]
            }"#,
        )
        .expect("snapshot parses")
    }

    #[test]
    fn a_snapshot_fills_the_encoder_state() {
        let data = DataSet::load_default().expect("bundled data");
        let state = state_from_snapshot(&data, &snapshot()).expect("snapshot imports");

        assert_eq!(state.round, 3);
        assert_eq!(state.wildling_threat, 4);
        assert_eq!(state.phase, Phase::Action);
        // No Raid order is on the board, so the first real decision is the
        // March step, still led by the Iron Throne holder.
        assert_eq!(state.action_step, ActionStep::March);
        assert_eq!(state.active_house, HouseId::from_name("stark"));

        let stark = HouseId::from_name("stark").unwrap();
        let house = state.house(stark).unwrap();
        assert_eq!(house.power, 6);
        assert_eq!(house.supply_level, 1);
        assert_eq!(house.house_cards_available.len(), 2);
        assert_eq!(house.orders_placed.len(), 1);

        let winterfell = data.map.lookup("winterfell").unwrap();
        let region = state.region(winterfell);
        assert_eq!(region.controller, Some(stark));
        assert_eq!(region.unit_count(UnitType::Footman), 1);
        assert_eq!(region.unit_count(UnitType::Knight), 1);
        assert_eq!(region.garrison, 2);
        assert_eq!(region.order, OrderKey::from_name("march"));
        assert_eq!(region.order_house, Some(stark));

        let sound = data.map.lookup("the_golden_sound").unwrap();
        assert_eq!(
            state.region(sound).controller,
            HouseId::from_name("lannister")
        );

        assert_eq!(encode_state(&state, &data, None).len(), STATE_DIM);
    }

    #[test]
    fn unknown_regions_are_rejected() {
        let data = DataSet::load_default().expect("bundled data");
        let mut snapshot = snapshot();
        snapshot.regions[0].id = "not-a-region".to_string();
        assert!(state_from_snapshot(&data, &snapshot).is_err());
    }
}
