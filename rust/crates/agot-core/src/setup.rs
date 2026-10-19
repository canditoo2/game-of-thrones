//! Opening position: `GameConfig` plus the setup `Game._initial_state` builds.
//!
//! Port of `GameConfig`, `Game._initial_state` and `state.new_state`.

use smallvec::SmallVec;

use crate::enums::{HouseId, RegionId, Track, GARRISON_STRENGTH, N_HOUSES};
use crate::map::{DataSet, WesterosData};
use crate::rng::Rng;
use crate::state::{DeckState, House, State};

/// Options that control a single game.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameConfig {
    pub houses: SmallVec<[HouseId; N_HOUSES]>,
    pub max_rounds: u8,
    pub victory_castles: u8,
    pub seed: u64,
    /// Draw and apply the three Westeros cards each round from round two.
    pub use_westeros_decks: bool,
}

impl Default for GameConfig {
    fn default() -> Self {
        GameConfig {
            houses: HouseId::ALL.into_iter().collect(),
            max_rounds: 10,
            victory_castles: 7,
            seed: 0,
            use_westeros_decks: true,
        }
    }
}

impl GameConfig {
    pub fn with_seed(seed: u64) -> GameConfig {
        GameConfig {
            seed,
            ..GameConfig::default()
        }
    }

    /// The same validation Python's `GameConfig.__post_init__` performs.
    pub fn validate(&self, data: &DataSet) -> Result<(), String> {
        if self.houses.is_empty() {
            return Err("at least one house is required".to_string());
        }
        for house in &self.houses {
            if data.houses.house(*house).is_none() {
                return Err(format!("unknown house: {house}"));
            }
        }
        Ok(())
    }
}

/// Builds the opening position, shuffled decks included.
///
/// Faithful to the Python original, including one quirk worth naming: the
/// home-area pass runs over *every* region that prints a home, not only the
/// houses taking part, so a short-handed game still marks the absent houses'
/// home areas as owned and garrisoned.
pub fn initial_state(data: &DataSet, config: &GameConfig, rng: &mut Rng) -> State {
    let map = &data.map;
    let mut state = State::blank();
    state.max_rounds = config.max_rounds;
    state.victory_castles = config.victory_castles;
    state.region_count = map.region_count() as u8;
    state.fortifications = map.fortifications.clone();

    // Houses: full order pool, full house deck, printed starting power/supply.
    for house in &config.houses {
        let spec = data
            .houses
            .house(*house)
            .expect("config validated before setup");
        state.houses[house.index()] = Some(House::new(
            data.house_cards.deck_size(*house),
            spec.initial_power,
            spec.initial_supply,
        ));
    }

    // Tracks: the reference ordering filtered to the houses in play, with any
    // house missing from the reference ordering going last.
    for track in Track::ALL {
        let mut order: SmallVec<[HouseId; N_HOUSES]> = data.houses.initial_tracks[track.index()]
            .iter()
            .copied()
            .filter(|house| config.houses.contains(house))
            .collect();
        for house in &config.houses {
            if !order.contains(house) {
                order.push(*house);
            }
        }
        state.tracks[track.index()] = order;
    }

    // Every region prints an owner (its home house); control comes next.
    let region_count = state.region_count as usize;
    for index in 0..region_count {
        let id = RegionId(index as u16);
        state.regions[index].owner = map.region(id).home;
    }

    // Starting units also establish control of the areas they occupy.
    for house in &config.houses {
        let spec = data
            .houses
            .house(*house)
            .expect("config validated before setup");
        for (region_id, counts) in &spec.starting_units {
            let region = &mut state.regions[region_id.index()];
            region.controller = Some(*house);
            for (index, count) in counts.iter().enumerate() {
                if *count > 0 {
                    region.units[index] += *count;
                }
            }
        }
    }

    // Home areas start under their owner's control and hold a Garrison.
    for index in 0..region_count {
        let id = RegionId(index as u16);
        if let Some(home) = map.region(id).home {
            let region = &mut state.regions[index];
            region.owner = Some(home);
            region.controller = Some(home);
            region.garrison = GARRISON_STRENGTH;
        }
    }

    // Neutral Force tokens defend the areas printed on them.
    for (region_id, strength) in map.neutral_forces.iter().copied() {
        state.regions[region_id.index()].neutral_force = strength;
        state.note(format!(
            "neutral force {strength} placed in {}",
            map.id_of(region_id)
        ));
    }

    // Shuffle the Westeros decks and the Wildling deck.
    for (position, _name) in WesterosData::DECK_NAMES.iter().enumerate() {
        let mut cards: smallvec::SmallVec<[u16; 12]> =
            data.westeros.decks[position].iter().copied().collect();
        rng.shuffle(&mut cards);
        state.westeros_decks[position] = DeckState::new(cards);
    }
    let mut wildling_deck: smallvec::SmallVec<[u16; 12]> =
        data.wildlings.deck.iter().copied().collect();
    rng.shuffle(&mut wildling_deck);
    state.wildling_deck = DeckState::new(wildling_deck);
    state.wildling_threat = data.wildlings.start_threat;

    state.pending_planning = state.tracks[Track::IronThrone.index()].clone();
    state.note(format!(
        "setup complete with {} houses",
        config.houses.len()
    ));
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::{OrderKey, UnitType};
    use crate::map::DataSet;

    fn data() -> DataSet {
        DataSet::load_default().expect("bundled data must load")
    }

    #[test]
    fn setup_fills_the_board() {
        let data = data();
        let config = GameConfig::with_seed(1);
        let mut rng = Rng::from_seed(config.seed);
        let state = initial_state(&data, &config, &mut rng);

        assert_eq!(state.region_count as usize, data.map.region_count());
        assert_eq!(state.present_houses().len(), 6);
        assert_eq!(state.pending_planning.len(), 6);
        assert_eq!(state.wildling_threat, data.wildlings.start_threat);

        // Every house has its full pool and deck.
        for house in HouseId::ALL {
            let house_state = state.house(house).unwrap();
            assert_eq!(house_state.orders_available.len(), 15);
            assert_eq!(house_state.house_cards_available.len(), 7);
            assert_eq!(house_state.power, 5);
        }

        // Home areas: owned, controlled and garrisoned.
        let winterfell = data.map.lookup("winterfell").unwrap();
        let region = state.region(winterfell);
        assert_eq!(region.owner, Some(HouseId(2)));
        assert_eq!(region.controller, Some(HouseId(2)));
        assert_eq!(region.garrison, GARRISON_STRENGTH);

        // Neutral forces sit where the map prints them.
        let kings_landing = data.map.lookup("kings_landing").unwrap();
        assert_eq!(state.region(kings_landing).neutral_force, 5);

        // Starting units landed on the board.
        let dragonstone = data.map.lookup("dragonstone").unwrap();
        assert_eq!(state.region(dragonstone).unit_count(UnitType::Footman), 1);
        assert_eq!(state.region(dragonstone).unit_count(UnitType::Knight), 1);
        let shipbreaker = data.map.lookup("shipbreaker_bay").unwrap();
        assert_eq!(state.region(shipbreaker).unit_count(UnitType::Ship), 2);

        // Decks are complete permutations of the reference composition.
        let mut deck_i = state.westeros_decks[0].cards.clone();
        deck_i.sort_unstable();
        let mut reference = data.westeros.decks[0].clone();
        reference.sort_unstable();
        assert_eq!(deck_i.as_slice(), reference.as_slice());
        assert_eq!(state.wildling_deck.cards.len(), data.wildlings.deck.len());
    }

    #[test]
    fn short_handed_games_keep_absent_homes_owned() {
        // The Python engine marks every printed home, not only the houses in
        // play; the port keeps that behaviour.
        let data = data();
        let config = GameConfig {
            houses: smallvec::smallvec![HouseId(2), HouseId(3)],
            seed: 4,
            ..GameConfig::default()
        };
        let mut rng = Rng::from_seed(config.seed);
        let state = initial_state(&data, &config, &mut rng);
        assert_eq!(state.present_houses().len(), 2);
        let dragonstone = data.map.lookup("dragonstone").unwrap();
        assert_eq!(state.region(dragonstone).controller, Some(HouseId(0)));
        assert_eq!(state.pending_planning.len(), 2);
    }

    #[test]
    fn shuffled_decks_differ_between_seeds() {
        let data = data();
        let mut first = Rng::from_seed(1);
        let mut second = Rng::from_seed(2);
        let config = GameConfig::with_seed(1);
        let a = initial_state(&data, &config, &mut first);
        let b = initial_state(&data, &config, &mut second);
        assert_ne!(a.westeros_decks[0].cards, b.westeros_decks[0].cards);
        assert_eq!(
            a.westeros_decks[0].cards.len(),
            b.westeros_decks[0].cards.len()
        );
    }

    #[test]
    fn order_pool_is_the_python_order() {
        let data = data();
        let config = GameConfig::with_seed(0);
        let mut rng = Rng::from_seed(0);
        let state = initial_state(&data, &config, &mut rng);
        let pool = &state.house(HouseId(0)).unwrap().orders_available;
        assert_eq!(pool[0], OrderKey(0));
        assert_eq!(pool[14], OrderKey(14));
    }
}
