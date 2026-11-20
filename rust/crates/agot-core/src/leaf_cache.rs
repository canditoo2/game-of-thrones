//! Exact evaluation cache for batched self-play leaves.
//!
//! Every decision restarts its search tree, so the next decision re-scores the
//! root and the subtree the previous one already scored, and the 97-float state
//! row carries no board, so many distinct positions encode to the same network
//! input. Measured on a trained net, ~73% of the leaves one `selfplay()` call
//! sends to the evaluator were already scored earlier in the same call. The
//! network weights are fixed for that call, so remembering each input's output
//! is exact for an evaluator that is a pure row-wise function of its input.
//!
//! # Key
//!
//! A leaf is its network input: the state row, the candidate count and the
//! ordered candidate rows, compared as `f32` bit patterns. The cache does not
//! keep those ~10 KB per leaf; it keeps a 256-bit digest built from two
//! independent NH lanes (the universal hash of UMAC, Black et al. 1999) over
//! the bit patterns, with 64-bit words and 128-bit accumulators:
//!
//! `NH_K(M) = sum_i (m[2i] + k[2i] mod 2^64) * (m[2i+1] + k[2i+1] mod 2^64) mod 2^128`
//!
//! NH with `w`-bit words is `2^-w`-almost-universal on equal-length messages,
//! so one lane collides on a fixed pair of distinct equal-length inputs with
//! probability at most `2^-64`, and two lanes with independent keys at most
//! `2^-128`. Inputs of different candidate counts have different lengths; the
//! count is the first message word *and* is stored in the entry and compared
//! exactly, so they can never be confused. The keys are drawn once from a
//! fixed SplitMix64 stream that no input can depend on.
//!
//! A false hit needs some lookup to collide with some distinct stored entry.
//! By the union bound that is at most `lookups x capacity x 2^-128`: with the
//! default capacity (2^16 entries) and even 2^30 lookups in one call (about
//! three thousand times a production 16-game call), below `2^-82`.
//!
//! [`LeafCache::new`] with `verify = true` also keeps every entry's full input
//! and compares it on each hit, returning [`CacheMismatch`] instead of a value
//! when the bytes differ. Tests run with it on; it doubles as an executable
//! statement of the property above.
//!
//! # Lifetime and memory
//!
//! The driver builds one cache per evaluator per `selfplay()` call, so a
//! network never sees another network's outputs and new weights never see old
//! ones. Memory is bounded by two generations: entries go into `current`; when
//! it holds half the entry cap or half the logit cap, `current` becomes
//! `previous` and the old `previous` is dropped. A hit in `previous` moves the
//! entry back into `current`, so recently used leaves survive a rotation. With
//! the defaults the cache never holds more than 2^16 entries or 2^22 stored
//! logits (16 MiB of logits, plus ~80 bytes per entry for the key, the value
//! vector and the map slot: ~22 MiB in all, without `verify`).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::enums::N_HOUSES;

/// Default cap on cached leaves (both generations together).
pub const DEFAULT_MAX_ENTRIES: usize = 1 << 16;
/// Default cap on cached candidate logits (both generations together).
pub const DEFAULT_MAX_LOGITS: usize = 1 << 22;

/// A 256-bit digest of one leaf's network input (two NH lanes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeafKey {
    lanes: [u128; 2],
}

impl std::hash::Hash for LeafKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // The low bits of an NH sum are poorly mixed (the lowest bit of a
        // product is a parity), so fold the high halves in for bucketing.
        let [first, second] = self.lanes;
        state.write_u64((first >> 64) as u64 ^ (second >> 64) as u64 ^ first as u64);
    }
}

/// A pass-through hasher for [`LeafKey`]: the key is already a digest.
#[derive(Default)]
pub struct DigestHasher(u64);

impl Hasher for DigestHasher {
    fn finish(&self) -> u64 {
        // One multiply spreads the folded digest over the bucket bits.
        self.0.wrapping_mul(0x9e37_79b9_7f4a_7c15)
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0 << 8) ^ u64::from(*byte);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 ^= value;
    }
}

/// A map keyed by [`LeafKey`] that skips re-hashing the digest.
pub type LeafKeyMap<V> = HashMap<LeafKey, V, BuildHasherDefault<DigestHasher>>;
type DigestMap<V> = LeafKeyMap<V>;

/// One cached evaluation: the raw logits and the value vector exactly as the
/// evaluator returned them.
struct Entry {
    count: u32,
    logits: Box<[f32]>,
    value: [f32; N_HOUSES],
    /// The full input (state row then candidate rows), only with `verify`.
    input: Option<Box<[f32]>>,
}

/// A verified hit whose stored input differs from the looked-up one.
///
/// Under the bound in the module docs this never happens; if it does, the
/// digest is broken and the driver must stop rather than use a wrong value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheMismatch;

impl std::fmt::Display for CacheMismatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("leaf cache digest collision: a hit had a different network input")
    }
}

impl std::error::Error for CacheMismatch {}

/// A cache hit: the stored raw logits and value vector.
pub type CachedEvaluation<'a> = (&'a [f32], &'a [f32; N_HOUSES]);

/// Counters for one cache's lifetime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups answered from the cache.
    pub hits: u64,
    /// Lookups the cache could not answer.
    pub misses: u64,
    /// Entries stored.
    pub inserts: u64,
    /// Entries dropped by a generation rotation.
    pub evicted: u64,
    /// Generation rotations.
    pub rotations: u64,
    /// Hits whose full input was compared byte for byte (`verify` only).
    pub verified: u64,
    /// Most entries held at once.
    pub peak_entries: u64,
    /// Most logits held at once.
    pub peak_logits: u64,
}

/// SplitMix64, the key stream for the NH lanes.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Fixed seed of the NH key stream. Any value works; it only has to be chosen
/// independently of the inputs, which a constant is.
const KEY_SEED: u64 = 0x6167_6f74_2d6c_6561; // "agot-lea"

/// Exact, bounded evaluation cache for one evaluator.
pub struct LeafCache {
    /// Interleaved NH keys: `keys[2 * word + lane]`.
    keys: Vec<u64>,
    key_state: u64,
    current: DigestMap<Entry>,
    previous: DigestMap<Entry>,
    current_logits: usize,
    previous_logits: usize,
    generation_entries: usize,
    generation_logits: usize,
    verify: bool,
    stats: CacheStats,
}

impl LeafCache {
    /// An empty cache holding at most `max_entries` leaves and `max_logits`
    /// candidate logits across its two generations.
    pub fn new(max_entries: usize, max_logits: usize, verify: bool) -> LeafCache {
        LeafCache {
            keys: Vec::new(),
            key_state: KEY_SEED,
            current: DigestMap::default(),
            previous: DigestMap::default(),
            current_logits: 0,
            previous_logits: 0,
            generation_entries: (max_entries / 2).max(1),
            generation_logits: (max_logits / 2).max(1),
            verify,
            stats: CacheStats::default(),
        }
    }

    /// A cache with the default bounds.
    pub fn with_defaults(verify: bool) -> LeafCache {
        LeafCache::new(DEFAULT_MAX_ENTRIES, DEFAULT_MAX_LOGITS, verify)
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Entries currently held (both generations).
    pub fn len(&self) -> usize {
        self.current.len() + self.previous.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Make sure the key stream covers `words` message words.
    fn reserve_keys(&mut self, words: usize) {
        while self.keys.len() < 2 * words {
            let value = splitmix64(&mut self.key_state);
            self.keys.push(value);
        }
    }

    /// The digest of one leaf: `state` (one row), and `actions` holding
    /// `count` candidate rows.
    ///
    /// The message is `[count, 0]`, then the state's bit patterns and then the
    /// actions', each segment zero-padded to a whole NH pair (four floats).
    /// The state length is fixed and the action length is `count` rows, so the
    /// padded message determines the input.
    pub fn key(&mut self, state: &[f32], actions: &[f32], count: usize) -> LeafKey {
        let state_words = state.len().div_ceil(4) * 2;
        let action_words = actions.len().div_ceil(4) * 2;
        self.reserve_keys(2 + state_words + action_words);
        let keys = &self.keys;
        let mut lanes = [0u128; 2];
        nh_pair(&mut lanes, keys, 0, count as u64, 0);
        let mut word = 2;
        word = nh_segment(&mut lanes, keys, word, state);
        nh_segment(&mut lanes, keys, word, actions);
        LeafKey { lanes }
    }

    /// The cached logits and value for `key`, if any.
    ///
    /// `state` and `actions` are only read with `verify`, to compare the full
    /// input; a mismatch is an error, never a value.
    pub fn lookup(
        &mut self,
        key: &LeafKey,
        count: usize,
        state: &[f32],
        actions: &[f32],
    ) -> Result<Option<CachedEvaluation<'_>>, CacheMismatch> {
        if !self.current.contains_key(key) {
            match self.previous.remove(key) {
                Some(entry) => {
                    self.previous_logits -= entry.logits.len();
                    self.store(*key, entry);
                }
                None => {
                    self.stats.misses += 1;
                    return Ok(None);
                }
            }
        }
        let verify = self.verify;
        let entry = self
            .current
            .get(key)
            .expect("the entry was just found or promoted");
        if entry.count as usize != count || entry.logits.len() != count {
            return Err(CacheMismatch);
        }
        if verify {
            let input = entry.input.as_deref().ok_or(CacheMismatch)?;
            let (stored_state, stored_actions) = input.split_at(state.len().min(input.len()));
            if !bits_equal(stored_state, state) || !bits_equal(stored_actions, actions) {
                return Err(CacheMismatch);
            }
            self.stats.verified += 1;
        }
        self.stats.hits += 1;
        Ok(Some((&entry.logits, &entry.value)))
    }

    /// Store one evaluation. `logits` has one entry per candidate and `value`
    /// one per house.
    pub fn insert(
        &mut self,
        key: LeafKey,
        logits: &[f32],
        value: &[f32],
        state: &[f32],
        actions: &[f32],
    ) {
        let mut stored = [0.0f32; N_HOUSES];
        stored.copy_from_slice(value);
        let input = if self.verify {
            let mut input = Vec::with_capacity(state.len() + actions.len());
            input.extend_from_slice(state);
            input.extend_from_slice(actions);
            Some(input.into_boxed_slice())
        } else {
            None
        };
        let entry = Entry {
            count: logits.len() as u32,
            logits: logits.into(),
            value: stored,
            input,
        };
        self.stats.inserts += 1;
        self.store(key, entry);
    }

    /// Put an entry into the current generation, rotating first when full.
    fn store(&mut self, key: LeafKey, entry: Entry) {
        if self.current.len() >= self.generation_entries
            || self.current_logits + entry.logits.len() > self.generation_logits
        {
            self.stats.rotations += 1;
            self.stats.evicted += self.previous.len() as u64;
            self.previous = std::mem::take(&mut self.current);
            self.previous_logits = self.current_logits;
            self.current_logits = 0;
        }
        self.current_logits += entry.logits.len();
        if let Some(old) = self.current.insert(key, entry) {
            self.current_logits -= old.logits.len();
        }
        let entries = self.len() as u64;
        let logits = (self.current_logits + self.previous_logits) as u64;
        self.stats.peak_entries = self.stats.peak_entries.max(entries);
        self.stats.peak_logits = self.stats.peak_logits.max(logits);
    }
}

/// Add one NH pair to both lanes.
#[inline(always)]
fn nh_pair(lanes: &mut [u128; 2], keys: &[u64], word: usize, first: u64, second: u64) {
    for (lane, sum) in lanes.iter_mut().enumerate() {
        let left = first.wrapping_add(keys[2 * word + lane]);
        let right = second.wrapping_add(keys[2 * (word + 1) + lane]);
        *sum = sum.wrapping_add(u128::from(left) * u128::from(right));
    }
}

/// Hash a float segment starting at message word `word`; returns the next
/// free word. Two floats make one 64-bit word, four make one NH pair.
#[inline]
fn nh_segment(lanes: &mut [u128; 2], keys: &[u64], mut word: usize, values: &[f32]) -> usize {
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        let first = u64::from(chunk[0].to_bits()) | (u64::from(chunk[1].to_bits()) << 32);
        let second = u64::from(chunk[2].to_bits()) | (u64::from(chunk[3].to_bits()) << 32);
        nh_pair(lanes, keys, word, first, second);
        word += 2;
    }
    let rest = chunks.remainder();
    if !rest.is_empty() {
        let mut padded = [0u32; 4];
        for (slot, value) in rest.iter().enumerate() {
            padded[slot] = value.to_bits();
        }
        let first = u64::from(padded[0]) | (u64::from(padded[1]) << 32);
        let second = u64::from(padded[2]) | (u64::from(padded[3]) << 32);
        nh_pair(lanes, keys, word, first, second);
        word += 2;
    }
    word
}

fn bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::{ACTION_DIM, STATE_DIM};

    fn leaf(tag: usize, count: usize) -> (Vec<f32>, Vec<f32>) {
        let value = |offset: usize| ((tag * 37 + offset * 11) % 101) as f32 / 128.0 - 0.25;
        let state = (0..STATE_DIM).map(value).collect();
        let actions = (0..count * ACTION_DIM)
            .map(|offset| value(offset + 7919))
            .collect();
        (state, actions)
    }

    fn outputs(tag: usize, count: usize) -> (Vec<f32>, Vec<f32>) {
        let logits = (0..count).map(|slot| (tag + slot) as f32 * 0.5).collect();
        let values = (0..N_HOUSES).map(|house| (tag * house) as f32).collect();
        (logits, values)
    }

    #[test]
    fn a_stored_leaf_is_returned_bit_for_bit() {
        let mut cache = LeafCache::with_defaults(true);
        let (state, actions) = leaf(3, 5);
        let key = cache.key(&state, &actions, 5);
        assert!(cache.lookup(&key, 5, &state, &actions).unwrap().is_none());
        let (logits, values) = outputs(3, 5);
        cache.insert(key, &logits, &values, &state, &actions);

        let again = cache.key(&state, &actions, 5);
        assert_eq!(again, key, "the digest is a pure function of the input");
        let (hit_logits, hit_value) = cache.lookup(&again, 5, &state, &actions).unwrap().unwrap();
        assert_eq!(
            hit_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(hit_value.to_vec(), values);
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.verified), (1, 1, 1));
    }

    #[test]
    fn any_bit_change_changes_the_key() {
        let mut cache = LeafCache::with_defaults(false);
        let (state, actions) = leaf(9, 3);
        let base = cache.key(&state, &actions, 3);
        for position in [0, 1, 50, STATE_DIM - 1] {
            let mut changed = state.clone();
            changed[position] = f32::from_bits(changed[position].to_bits() ^ 1);
            assert_ne!(
                cache.key(&changed, &actions, 3),
                base,
                "state float {position}"
            );
        }
        for position in [0, ACTION_DIM, 3 * ACTION_DIM - 1] {
            let mut changed = actions.clone();
            changed[position] = f32::from_bits(changed[position].to_bits() ^ 1);
            assert_ne!(
                cache.key(&state, &changed, 3),
                base,
                "action float {position}"
            );
        }
        // -0.0 and 0.0 are different inputs, exactly as the in-pass dedupe sees them.
        let mut zero = state.clone();
        zero[0] = 0.0;
        let mut negative = state.clone();
        negative[0] = -0.0;
        assert_ne!(
            cache.key(&zero, &actions, 3),
            cache.key(&negative, &actions, 3)
        );
        // Candidate order matters.
        let mut swapped = actions.clone();
        swapped[..ACTION_DIM].copy_from_slice(&actions[ACTION_DIM..2 * ACTION_DIM]);
        swapped[ACTION_DIM..2 * ACTION_DIM].copy_from_slice(&actions[..ACTION_DIM]);
        assert_ne!(cache.key(&state, &swapped, 3), base);
        // A prefix of the candidates is a different leaf.
        assert_ne!(cache.key(&state, &actions[..2 * ACTION_DIM], 2), base);
    }

    #[test]
    fn a_forged_collision_is_caught_by_verify() {
        let mut cache = LeafCache::with_defaults(true);
        let (state, actions) = leaf(1, 2);
        let (other_state, _) = leaf(2, 2);
        let key = cache.key(&state, &actions, 2);
        let (logits, values) = outputs(1, 2);
        cache.insert(key, &logits, &values, &state, &actions);
        // Look the stored digest up with a different input, as a collision would.
        assert_eq!(
            cache.lookup(&key, 2, &other_state, &actions).err(),
            Some(CacheMismatch)
        );
    }

    #[test]
    fn the_bounds_hold_and_recent_entries_survive_rotation() {
        // Four entries in all: two per generation.
        let mut cache = LeafCache::new(4, 1 << 20, true);
        let mut keys = Vec::new();
        for tag in 0..3 {
            let (state, actions) = leaf(tag, 2);
            let key = cache.key(&state, &actions, 2);
            let (logits, values) = outputs(tag, 2);
            cache.insert(key, &logits, &values, &state, &actions);
            keys.push((key, state, actions));
        }
        // Inserting the third entry rotated {0, 1} into `previous`.
        assert_eq!(cache.stats().rotations, 1);
        assert_eq!(cache.len(), 3);
        // A hit on 0 promotes it into the current generation.
        let (key, state, actions) = &keys[0];
        assert!(cache.lookup(key, 2, state, actions).unwrap().is_some());
        // Inserting 3 rotates again: {2, 0} become previous, 1 is dropped.
        let (state, actions) = leaf(3, 2);
        let key = cache.key(&state, &actions, 2);
        let (logits, values) = outputs(3, 2);
        cache.insert(key, &logits, &values, &state, &actions);
        assert!(cache.len() <= 4);
        let (key, state, actions) = &keys[1];
        assert!(cache.lookup(key, 2, state, actions).unwrap().is_none());
        let (key, state, actions) = &keys[0];
        assert!(cache.lookup(key, 2, state, actions).unwrap().is_some());
        let stats = cache.stats();
        assert!(stats.peak_entries <= 4);
        assert_eq!(stats.evicted, 1);
    }

    #[test]
    fn the_logit_bound_rotates_too() {
        let mut cache = LeafCache::new(1 << 10, 20, false);
        for tag in 0..6 {
            let (state, actions) = leaf(tag, 4);
            let key = cache.key(&state, &actions, 4);
            let (logits, values) = outputs(tag, 4);
            cache.insert(key, &logits, &values, &state, &actions);
            assert!(cache.stats().peak_logits <= 20);
        }
        assert!(cache.stats().rotations >= 2);
    }

    #[test]
    fn many_distinct_leaves_get_distinct_keys() {
        let mut cache = LeafCache::with_defaults(false);
        let mut seen = std::collections::HashSet::new();
        for tag in 0..5_000 {
            let (state, actions) = leaf(tag, 1 + tag % 7);
            let key = cache.key(&state, &actions, 1 + tag % 7);
            seen.insert((key.lanes[0], key.lanes[1]));
        }
        // `leaf` itself repeats with period 101 in `tag` for a fixed count,
        // so count the distinct inputs rather than assuming 5 000.
        let mut inputs = std::collections::HashSet::new();
        for tag in 0..5_000usize {
            let (state, actions) = leaf(tag, 1 + tag % 7);
            let bits: Vec<u32> = state.iter().chain(&actions).map(|v| v.to_bits()).collect();
            inputs.insert(bits);
        }
        assert_eq!(seen.len(), inputs.len());
    }
}
