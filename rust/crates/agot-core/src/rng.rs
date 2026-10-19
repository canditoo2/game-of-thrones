//! Deterministic, clonable PRNG for the engine.
//!
//! The Python engine uses `random.Random`, whose Mersenne Twister stream is not
//! reproducible from Rust. That is deliberate and fine: the differential
//! harness against the Python reference is *action-scripted* (both engines
//! replay the same action sequence), so only within-Rust determinism matters.
//!
//! `xoshiro256**` is small, fast, and cheap to clone — which the search needs,
//! because a cloned game must carry an independent stream.

/// A seeded `xoshiro256**` generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    state: [u64; 4],
}

#[inline]
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// A generator seeded from one integer.
    pub fn from_seed(seed: u64) -> Rng {
        let mut sm = seed;
        let mut state = [0u64; 4];
        for slot in state.iter_mut() {
            *slot = splitmix64(&mut sm);
        }
        // xoshiro must never be all-zero.
        if state == [0, 0, 0, 0] {
            state[0] = 0x9E37_79B9_7F4A_7C15;
        }
        Rng { state }
    }

    /// A child generator, the analogue of Python's
    /// `random.Random(self.rng.random())` on `Game.clone`.
    pub fn fork(&mut self) -> Rng {
        Rng::from_seed(self.next_u64())
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform in `[0, bound)` (Lemire's method, rejection-free modulo bias).
    pub fn next_bounded(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "bound must be positive");
        let mut x = self.next_u64();
        let mut m = (x as u128) * (bound as u128);
        let mut low = m as u64;
        if low < bound {
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                x = self.next_u64();
                m = (x as u128) * (bound as u128);
                low = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[low, high]`, inclusive (Python's `randint`).
    pub fn range_inclusive(&mut self, low: i64, high: i64) -> i64 {
        assert!(low <= high, "empty range");
        let span = (high - low) as u64 + 1;
        low + self.next_bounded(span) as i64
    }

    /// In-place Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        if len < 2 {
            return;
        }
        for i in (1..len).rev() {
            let j = self.next_bounded((i + 1) as u64) as usize;
            slice.swap(i, j);
        }
    }

    /// Picks one element uniformly, returning a copy.
    pub fn choose<T: Copy>(&mut self, slice: &[T]) -> Option<T> {
        if slice.is_empty() {
            return None;
        }
        let index = self.next_bounded(slice.len() as u64) as usize;
        Some(slice[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::from_seed(42);
        let mut b = Rng::from_seed(42);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Rng::from_seed(1);
        let mut b = Rng::from_seed(2);
        let differ = (0..8).any(|_| a.next_u64() != b.next_u64());
        assert!(differ);
    }

    #[test]
    fn fork_is_independent_and_deterministic() {
        let mut parent_a = Rng::from_seed(7);
        let mut parent_b = Rng::from_seed(7);
        let mut child_a = parent_a.fork();
        let mut child_b = parent_b.fork();
        assert_eq!(child_a.next_u64(), child_b.next_u64());
        assert_ne!(child_a.next_u64(), parent_a.next_u64());
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Rng::from_seed(3);
        let mut values: Vec<u8> = (0..32).collect();
        rng.shuffle(&mut values);
        let mut sorted = values.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32).collect::<Vec<u8>>());
        assert_ne!(values, sorted, "a 32-element shuffle should reorder");
    }

    #[test]
    fn bounded_stays_in_range() {
        let mut rng = Rng::from_seed(11);
        for bound in [1u64, 2, 3, 6, 15, 100] {
            for _ in 0..200 {
                assert!(rng.next_bounded(bound) < bound);
            }
        }
        for _ in 0..200 {
            let value = rng.range_inclusive(-3, 4);
            assert!((-3..=4).contains(&value));
        }
    }
}
