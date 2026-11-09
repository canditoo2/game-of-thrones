//! Exact leaf deduplication for batched evaluation.
//!
//! The self-play driver packs every search leaf as one state row plus the
//! ordered action rows it is scored against. With PIMC (`determinizations > 1`)
//! the same position is searched in several resampled worlds, and the state
//! vector has no region one-hots, so leaves from different worlds and even
//! different games can be bit-identical. The evaluator is a pure function of
//! the packed rows, so scoring each unique leaf once and copying its outputs to
//! every duplicate is exact.
//!
//! [`LeafDedupe`] indexes a packed batch by exact contents, exposes the unique
//! rows to the evaluator, and scatters the returned logits and value rows back
//! to the original row and candidate-slot order. "Exact" means bit patterns:
//! `-0.0` and `0.0` (or two different NaNs) are not merged, so a dedupe-on run
//! and a dedupe-off run see the same values.
//!
//! The scatter itself is bit-exact. An A/B run against the dedupe-off path can
//! still differ in the last bits of the network outputs because a smaller batch
//! makes torch pick a different GEMM reduction order; that is why the driver
//! exposes the `dedupe_leaves` switch.

use std::collections::HashMap;

/// A packed leaf batch with exact duplicate leaves removed.
///
/// The original layout is one state row per leaf, `states` as `rows *
/// state_dim` floats, and `actions` as `index.len() * action_dim` floats where
/// `index[slot]` names the leaf that owns candidate slot `slot`. Candidate
/// slots are grouped by leaf in ascending leaf order, which is how the
/// self-play driver builds them.
pub struct LeafDedupe {
    rows: usize,
    unique_rows: usize,
    total_slots: usize,
    unique_slots: usize,
    /// `rows + 1` candidate-slot starts in the original layout.
    row_slots: Vec<usize>,
    /// `unique_rows + 1` candidate-slot starts in the unique layout.
    unique_row_slots: Vec<usize>,
    /// For each original row, the unique row carrying its exact contents.
    unique_of_row: Vec<usize>,
    states: Vec<f32>,
    actions: Vec<f32>,
    index: Vec<i64>,
}

impl LeafDedupe {
    /// Index `rows` packed leaves, keeping the first occurrence of each exact
    /// duplicate in encounter order.
    ///
    /// Panics when the slices do not describe a well-formed packed batch: the
    /// driver builds them itself, so a malformed batch is a bug, not input.
    pub fn from_packed(
        states: &[f32],
        state_dim: usize,
        actions: &[f32],
        action_dim: usize,
        index: &[i64],
        rows: usize,
    ) -> LeafDedupe {
        assert_eq!(
            states.len(),
            rows * state_dim,
            "packed states must hold one row per leaf"
        );
        assert!(
            action_dim > 0 && actions.len() % action_dim == 0,
            "packed actions must be whole rows"
        );
        let total_slots = actions.len() / action_dim;
        assert_eq!(
            index.len(),
            total_slots,
            "packed index must name every candidate slot"
        );

        // Candidate slots are contiguous and grouped by leaf, so one scan
        // recovers each row's half-open range and rejects any other layout.
        let mut row_slots = Vec::with_capacity(rows + 1);
        row_slots.push(0);
        let mut cursor = 0usize;
        for row in 0..rows {
            while cursor < total_slots && index[cursor] == row as i64 {
                cursor += 1;
            }
            row_slots.push(cursor);
        }
        assert_eq!(
            cursor, total_slots,
            "candidate slots must be grouped by leaf in ascending order"
        );

        let mut unique_of_row = vec![0usize; rows];
        let mut row_of_unique: Vec<usize> = Vec::new();
        let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut unique_row_slots = Vec::with_capacity(rows + 1);
        unique_row_slots.push(0);
        let mut unique_states = Vec::new();
        let mut unique_actions = Vec::new();
        let mut unique_index = Vec::new();

        for row in 0..rows {
            let state = &states[row * state_dim..(row + 1) * state_dim];
            let action_start = row_slots[row] * action_dim;
            let action_end = row_slots[row + 1] * action_dim;
            let action_rows = &actions[action_start..action_end];

            let hash = state_hash(state);

            let mut duplicate = None;
            if let Some(candidates) = buckets.get(&hash) {
                for &candidate in candidates {
                    if rows_equal(
                        states,
                        state_dim,
                        actions,
                        action_dim,
                        &row_slots,
                        row,
                        row_of_unique[candidate],
                    ) {
                        duplicate = Some(candidate);
                        break;
                    }
                }
            }

            match duplicate {
                Some(candidate) => unique_of_row[row] = candidate,
                None => {
                    let candidate = row_of_unique.len();
                    let count = row_slots[row + 1] - row_slots[row];
                    row_of_unique.push(row);
                    unique_of_row[row] = candidate;
                    unique_states.extend_from_slice(state);
                    unique_actions.extend_from_slice(action_rows);
                    for _ in 0..count {
                        unique_index.push(candidate as i64);
                    }
                    unique_row_slots.push(unique_row_slots[candidate] + count);
                    buckets.entry(hash).or_default().push(candidate);
                }
            }
        }

        let unique_rows = row_of_unique.len();
        debug_assert_eq!(unique_row_slots.len(), unique_rows + 1);
        let unique_slots = unique_row_slots[unique_rows];
        debug_assert_eq!(unique_states.len(), unique_rows * state_dim);
        debug_assert_eq!(unique_index.len(), unique_slots);

        LeafDedupe {
            rows,
            unique_rows,
            total_slots,
            unique_slots,
            row_slots,
            unique_row_slots,
            unique_of_row,
            states: unique_states,
            actions: unique_actions,
            index: unique_index,
        }
    }

    /// Number of leaves in the original batch.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of leaves the evaluator has to score.
    pub fn unique_rows(&self) -> usize {
        self.unique_rows
    }

    /// Number of candidate slots in the original batch.
    pub fn total_slots(&self) -> usize {
        self.total_slots
    }

    /// Number of candidate slots the evaluator has to score.
    pub fn unique_slots(&self) -> usize {
        self.unique_slots
    }

    /// Packed state rows for the unique leaves, in first-occurrence order.
    pub fn states(&self) -> &[f32] {
        &self.states
    }

    /// Packed action rows for the unique leaves, in first-occurrence order.
    pub fn actions(&self) -> &[f32] {
        &self.actions
    }

    /// Segment index for the unique candidate slots.
    pub fn index(&self) -> &[i64] {
        &self.index
    }

    /// The unique leaf that carries original row `row`'s contents.
    pub fn unique_of_row(&self, row: usize) -> usize {
        self.unique_of_row[row]
    }

    /// Copy each unique leaf's logits to every original candidate slot.
    pub fn scatter_logits(&self, unique: &[f32]) -> Vec<f32> {
        assert_eq!(
            unique.len(),
            self.unique_slots,
            "scatter_logits needs one logit per unique candidate"
        );
        let mut out = vec![0.0f32; self.total_slots];
        for row in 0..self.rows {
            let candidate = self.unique_of_row[row];
            let destination = self.row_slots[row];
            let source = self.unique_row_slots[candidate];
            let count = self.row_slots[row + 1] - destination;
            debug_assert_eq!(
                count,
                self.unique_row_slots[candidate + 1] - self.unique_row_slots[candidate]
            );
            out[destination..destination + count].copy_from_slice(&unique[source..source + count]);
        }
        out
    }

    /// Copy each unique leaf's value row to every original row.
    pub fn scatter_values(&self, unique: &[f32], houses: usize) -> Vec<f32> {
        assert_eq!(
            unique.len(),
            self.unique_rows * houses,
            "scatter_values needs one value vector per unique leaf"
        );
        let mut out = vec![0.0f32; self.rows * houses];
        for row in 0..self.rows {
            let candidate = self.unique_of_row[row];
            out[row * houses..(row + 1) * houses]
                .copy_from_slice(&unique[candidate * houses..(candidate + 1) * houses]);
        }
        out
    }
}

/// FNV-1a over the state row's bit patterns.
///
/// Only the 97-float state is hashed. The action rows are the expensive part of
/// a leaf and the aggregate state already collides across worlds, so hashing
/// the state alone finds every duplicate; exact comparison of the action rows
/// then rejects the leaves that merely share a board. Hashing the action rows
/// up front was several times slower on a production-shaped batch.
fn state_hash(state: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in state {
        hash ^= u64::from(value.to_bits());
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Whether two packed leaves have bit-identical state and action rows.
fn rows_equal(
    states: &[f32],
    state_dim: usize,
    actions: &[f32],
    action_dim: usize,
    row_slots: &[usize],
    left: usize,
    right: usize,
) -> bool {
    let state_left = &states[left * state_dim..(left + 1) * state_dim];
    let state_right = &states[right * state_dim..(right + 1) * state_dim];
    if !bits_equal(state_left, state_right) {
        return false;
    }
    let action_left = &actions[row_slots[left] * action_dim..row_slots[left + 1] * action_dim];
    let action_right = &actions[row_slots[right] * action_dim..row_slots[right + 1] * action_dim];
    bits_equal(action_left, action_right)
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
    use crate::enums::N_HOUSES;
    use crate::observation::{ACTION_DIM, STATE_DIM};

    /// A deterministic stand-in for the network.
    ///
    /// Both outputs are pure functions of the packed leaf rows (the value head
    /// in this project reads the candidate actions too), never of a row's
    /// position in the batch. That is exactly the property that makes dedupe
    /// exact.
    struct FakeEvaluator {
        calls: usize,
        rows: usize,
        slots: usize,
    }

    impl FakeEvaluator {
        fn new() -> FakeEvaluator {
            FakeEvaluator {
                calls: 0,
                rows: 0,
                slots: 0,
            }
        }

        fn evaluate(
            &mut self,
            states: &[f32],
            actions: &[f32],
            index: &[i64],
            rows: usize,
        ) -> (Vec<f32>, Vec<f32>) {
            self.calls += 1;
            self.rows += rows;
            self.slots += index.len();

            let mut raw = vec![0.0f32; index.len()];
            for (slot, row) in index.iter().enumerate() {
                let row = *row as usize;
                let mut sum = 0.0f32;
                for (offset, value) in states[row * STATE_DIM..(row + 1) * STATE_DIM]
                    .iter()
                    .enumerate()
                {
                    sum += value * ((offset % 7) as f32 + 1.0);
                }
                for (offset, value) in actions[slot * ACTION_DIM..(slot + 1) * ACTION_DIM]
                    .iter()
                    .enumerate()
                {
                    sum += value * ((offset % 5) as f32 + 1.0);
                }
                raw[slot] = sum;
            }

            let mut values = vec![0.0f32; rows * N_HOUSES];
            for row in 0..rows {
                let mut sum = 0.0f32;
                for value in &states[row * STATE_DIM..(row + 1) * STATE_DIM] {
                    sum += value;
                }
                for (slot, index_row) in index.iter().enumerate() {
                    if *index_row == row as i64 {
                        for value in &actions[slot * ACTION_DIM..(slot + 1) * ACTION_DIM] {
                            sum += value * 0.25;
                        }
                    }
                }
                for house in 0..N_HOUSES {
                    values[row * N_HOUSES + house] = sum * (house as f32 + 1.0);
                }
            }
            (raw, values)
        }
    }

    /// The driver's per-leaf softmax, copied so the test can compare priors too.
    fn segment_softmax(logits: &[f32], index: &[i64], rows: usize) -> Vec<f32> {
        let mut maxima = vec![f32::NEG_INFINITY; rows];
        for (slot, row) in index.iter().enumerate() {
            let row = *row as usize;
            if logits[slot] > maxima[row] {
                maxima[row] = logits[slot];
            }
        }
        let mut totals = vec![0.0f32; rows];
        let mut out = vec![0.0f32; logits.len()];
        for (slot, row) in index.iter().enumerate() {
            let row = *row as usize;
            let value = (logits[slot] - maxima[row]).exp();
            out[slot] = value;
            totals[row] += value;
        }
        for (slot, row) in index.iter().enumerate() {
            let total = totals[*row as usize];
            out[slot] = if total > 0.0 { out[slot] / total } else { 0.0 };
        }
        out
    }

    fn assert_bits_equal(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (position, (a, b)) in left.iter().zip(right).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "slot {position}: {a:?} != {b:?} (bitwise)"
            );
        }
    }

    /// `candidates` values that depend only on `(leaf, depth)`.
    fn leaf_value(leaf: usize, offset: usize) -> f32 {
        ((leaf * 37 + offset * 11) % 101) as f32 / 128.0 - 0.25
    }

    /// Build a leaf from a tag; equal tags produce bit-identical rows.
    fn push_leaf(
        states: &mut Vec<f32>,
        actions: &mut Vec<f32>,
        index: &mut Vec<i64>,
        row: usize,
        tag: usize,
        candidates: usize,
    ) {
        for offset in 0..STATE_DIM {
            states.push(leaf_value(tag, offset));
        }
        for candidate in 0..candidates {
            for offset in 0..ACTION_DIM {
                actions.push(leaf_value(tag + 1000 * candidate, offset));
            }
            index.push(row as i64);
        }
    }

    #[test]
    fn known_duplicates_score_once_and_scatter_exactly() {
        // Rows 1 and 4 repeat row 0; row 3 repeats row 2. Row 0 and row 2 keep
        // distinct action counts to exercise unequal segments.
        let tags = [7usize, 7, 19, 19, 7];
        let counts = [3usize, 3, 5, 5, 3];
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut index = Vec::new();
        for (row, (&tag, &count)) in tags.iter().zip(&counts).enumerate() {
            push_leaf(&mut states, &mut actions, &mut index, row, tag, count);
        }
        let rows = tags.len();
        let total = index.len();

        let mut full_evaluator = FakeEvaluator::new();
        let (full_logits, full_values) = full_evaluator.evaluate(&states, &actions, &index, rows);

        let dedupe =
            LeafDedupe::from_packed(&states, STATE_DIM, &actions, ACTION_DIM, &index, rows);
        assert_eq!(dedupe.unique_rows(), 2);
        assert_eq!(dedupe.unique_slots(), counts[0] + counts[2]);
        assert_eq!(dedupe.unique_of_row(0), 0);
        assert_eq!(dedupe.unique_of_row(1), 0);
        assert_eq!(dedupe.unique_of_row(2), 1);
        assert_eq!(dedupe.unique_of_row(3), 1);
        assert_eq!(dedupe.unique_of_row(4), 0);

        let mut unique_evaluator = FakeEvaluator::new();
        let (unique_logits, unique_values) = unique_evaluator.evaluate(
            dedupe.states(),
            dedupe.actions(),
            dedupe.index(),
            dedupe.unique_rows(),
        );
        assert_eq!(unique_evaluator.calls, 1);
        assert_eq!(unique_evaluator.rows, 2, "only two leaves need scoring");
        assert_eq!(
            unique_evaluator.slots,
            counts[0] + counts[2],
            "only the unique leaves' candidates need scoring"
        );
        // One batch callback either way: the saving is in the rows and
        // candidate slots the evaluator has to score.
        assert_eq!(full_evaluator.rows, 5);
        assert_eq!(full_evaluator.slots, total);
        assert_eq!(unique_evaluator.rows, full_evaluator.rows - 3);
        assert_eq!(unique_evaluator.slots, full_evaluator.slots - 11);
        assert_eq!(total, 3 + 3 + 5 + 5 + 3);

        let scattered_logits = dedupe.scatter_logits(&unique_logits);
        let scattered_values = dedupe.scatter_values(&unique_values, N_HOUSES);
        assert_bits_equal(&scattered_logits, &full_logits);
        assert_bits_equal(&scattered_values, &full_values);

        // The driver's softmax runs on the scattered layout, so the final
        // priors must be bit-identical too.
        let full_priors = segment_softmax(&full_logits, &index, rows);
        let scattered_priors = segment_softmax(&scattered_logits, &index, rows);
        assert_bits_equal(&scattered_priors, &full_priors);
    }

    #[test]
    fn no_duplicates_is_a_passthrough() {
        let tags = [3usize, 5, 8, 13];
        let counts = [2usize, 4, 1, 6];
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut index = Vec::new();
        for (row, (&tag, &count)) in tags.iter().zip(&counts).enumerate() {
            push_leaf(&mut states, &mut actions, &mut index, row, tag, count);
        }

        let dedupe =
            LeafDedupe::from_packed(&states, STATE_DIM, &actions, ACTION_DIM, &index, tags.len());
        assert_eq!(dedupe.unique_rows(), tags.len());
        assert_eq!(dedupe.unique_slots(), index.len());
        assert_bits_equal(dedupe.states(), &states);
        assert_bits_equal(dedupe.actions(), &actions);
        assert_eq!(dedupe.index(), index.as_slice());
        for row in 0..tags.len() {
            assert_eq!(dedupe.unique_of_row(row), row);
        }

        let mut full = FakeEvaluator::new();
        let (full_logits, full_values) = full.evaluate(&states, &actions, &index, tags.len());
        let mut unique = FakeEvaluator::new();
        let (unique_logits, unique_values) = unique.evaluate(
            dedupe.states(),
            dedupe.actions(),
            dedupe.index(),
            dedupe.unique_rows(),
        );

        assert_eq!(unique.calls, full.calls);
        assert_eq!(unique.rows, full.rows);
        assert_eq!(unique.slots, full.slots);
        assert_bits_equal(&dedupe.scatter_logits(&unique_logits), &full_logits);
        assert_bits_equal(
            &dedupe.scatter_values(&unique_values, N_HOUSES),
            &full_values,
        );
    }

    #[test]
    fn same_state_with_different_actions_is_not_merged() {
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut index = Vec::new();
        push_leaf(&mut states, &mut actions, &mut index, 0, 42, 3);
        // The same state row, but three different candidate rows.
        let first_state = states[..STATE_DIM].to_vec();
        states.extend_from_slice(&first_state);
        // A different candidate count as well as different rows: a leaf is not
        // a prefix match.
        for candidate in 0..2 {
            for offset in 0..ACTION_DIM {
                actions.push(leaf_value(9000 + candidate, offset));
            }
            index.push(1);
        }

        let dedupe = LeafDedupe::from_packed(&states, STATE_DIM, &actions, ACTION_DIM, &index, 2);
        assert_eq!(dedupe.unique_rows(), 2);
        assert_eq!(dedupe.unique_slots(), index.len());
    }

    #[test]
    fn same_actions_with_different_state_is_not_merged() {
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut index = Vec::new();
        push_leaf(&mut states, &mut actions, &mut index, 0, 11, 4);
        // A different state row, but the exact same candidate rows.
        for offset in 0..STATE_DIM {
            states.push(leaf_value(12, offset));
        }
        let first_leaf_actions = actions.clone();
        for slot in 0..4 {
            actions
                .extend_from_slice(&first_leaf_actions[slot * ACTION_DIM..(slot + 1) * ACTION_DIM]);
        }
        index.extend([1i64; 4]);

        let dedupe = LeafDedupe::from_packed(&states, STATE_DIM, &actions, ACTION_DIM, &index, 2);
        assert_eq!(dedupe.unique_rows(), 2);
        assert_eq!(dedupe.unique_slots(), index.len());
    }

    #[test]
    fn production_shaped_batch_reduces_scored_rows() {
        // 24 leaves, ~15 candidates each, ~30% of the leaves duplicates of an
        // earlier leaf: the shape measured on a real self-play batch.
        const LEAVES: usize = 24;
        const CANDIDATES: usize = 15;
        let tags = [
            1usize, 2, 3, 1, 5, 6, 7, 2, 9, 10, 11, 12, 3, 14, 15, 5, 17, 18, 19, 7, 6, 22, 23, 10,
        ];
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut index = Vec::new();
        for (row, &tag) in tags.iter().enumerate() {
            push_leaf(&mut states, &mut actions, &mut index, row, tag, CANDIDATES);
        }

        let dedupe =
            LeafDedupe::from_packed(&states, STATE_DIM, &actions, ACTION_DIM, &index, LEAVES);
        let unique_rows = tags
            .iter()
            .enumerate()
            .filter(|(row, tag)| !tags[..*row].contains(tag))
            .count();
        assert_eq!(dedupe.unique_rows(), unique_rows);
        assert_eq!(dedupe.unique_slots(), unique_rows * CANDIDATES);
        assert_eq!(dedupe.total_slots(), LEAVES * CANDIDATES);

        let row_reduction = 100.0 * (1.0 - dedupe.unique_rows() as f64 / LEAVES as f64);
        let slot_reduction =
            100.0 * (1.0 - dedupe.unique_slots() as f64 / dedupe.total_slots() as f64);
        eprintln!(
            "production-shaped batch: leaves {} -> {} ({row_reduction:.1}%), \
             candidate rows {} -> {} ({slot_reduction:.1}%)",
            LEAVES,
            dedupe.unique_rows(),
            dedupe.total_slots(),
            dedupe.unique_slots(),
        );
        assert_eq!(dedupe.unique_rows(), 17);
        assert_eq!(dedupe.unique_slots(), 255);
    }
}
