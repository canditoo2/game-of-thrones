//! Per-phase simulation budgets for the stepwise self-play driver.
//!
//! A fixed simulation budget spreads effort evenly over a game that is not
//! evenly interesting. Measured on the batched driver, 61% of the scored leaves
//! came from Planning decisions and 25% from Wildling attacks, while a Planning
//! decision's candidate Q values spread by ~0.001 and its target sat 0.03 nats
//! from the prior: the budget there buys nothing the network can learn from.
//!
//! [`PhaseBudgets`] caps the simulations a *decision* may spend, by phase. The
//! cap is a ceiling per decision, not per PIMC world: a driver still divides it
//! by its determinizations exactly as it divides the fixed budget, so an
//! existing run keeps its per-world split. A decision with more candidates than
//! its cap still gets one simulation per candidate, because a target built from
//! a partially screened action list is worse than one built from a cheap full
//! screen.
//!
//! The cap is applied per decision, once, before any world is searched. PIMC
//! determinization never changes the acting house's own legal actions
//! ([`crate::hidden::determinize`]), so every world of one decision is scored
//! under the same budget and the aggregated visits stay balanced.

use crate::enums::{Phase, N_PHASES};

/// Per-phase simulation caps, one entry per [`Phase`].
///
/// [`PhaseBudgets::DEFAULT`] is the measured preset. Pass `None` to
/// [`decision_budget`] for the previous fixed-budget behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseBudgets {
    caps: [u32; N_PHASES],
}

impl PhaseBudgets {
    /// The measured preset: `Planning 32`, `WildlingAttack 16`, `Mustering 32`,
    /// `Bidding 32`, `Action 32`, floored at one simulation per candidate.
    ///
    /// Raven, Combat, Westeros and Cleanup keep 64: they are rare, they change
    /// the board, and their targets are far from the prior. `GameOver` never
    /// opens a root; it is listed only so the table is total. Python's
    /// `rust_engine.PHASE_BUDGETS["default"]` mirrors these numbers and is sent
    /// as an override of this preset.
    pub const DEFAULT: PhaseBudgets = PhaseBudgets {
        caps: [
            64, // Westeros
            64, // WesterosChoice
            32, // Bidding
            16, // WildlingAttack
            32, // Mustering
            32, // Planning
            64, // Raven
            32, // Action
            64, // Combat
            64, // Cleanup
            64, // GameOver
        ],
    };

    /// The cap for one phase, before the candidate-count floor.
    #[inline]
    pub fn cap(&self, phase: Phase) -> u32 {
        self.caps[phase.index()]
    }

    /// Replaces one phase's cap, for a `phase_budgets` override table.
    pub fn set_cap(&mut self, phase: Phase, cap: u32) {
        self.caps[phase.index()] = cap;
    }

    /// The same table with one phase's cap replaced (builder form).
    pub fn with_cap(mut self, phase: Phase, cap: u32) -> PhaseBudgets {
        self.set_cap(phase, cap);
        self
    }

    /// Simulations to run for one decision with `actions` candidates.
    ///
    /// The result is the phase's cap, raised to one simulation per candidate
    /// (plus the root's own expansion) and clamped to `ceiling`, which is the
    /// caller's `simulations` setting. A ceiling of zero behaves like one,
    /// matching the driver's own `max(1)`.
    pub fn for_decision(&self, phase: Phase, actions: usize, ceiling: u32) -> u32 {
        let per_candidate = u32::try_from(actions).unwrap_or(u32::MAX).saturating_add(1);
        self.cap(phase).max(per_candidate).min(ceiling.max(1))
    }
}

/// Simulations to run for one decision, `ceiling` when no budgets are set.
///
/// This is the single entrance the driver uses, so `None` (the spec's `null`)
/// is bit-identical to the fixed budget it replaced.
#[inline]
pub fn decision_budget(
    budgets: Option<&PhaseBudgets>,
    phase: Phase,
    actions: usize,
    ceiling: u32,
) -> u32 {
    match budgets {
        Some(budgets) => budgets.for_decision(phase, actions, ceiling),
        None => ceiling.max(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every phase at every interesting candidate count. A wrong cap, an
    /// off-by-one in the floor or a missing clamp by the ceiling shows up here.
    #[test]
    fn the_budget_table_matches_the_measured_preset() {
        use Phase::*;
        // (phase, cap, budgets for 1, 5, 32, 45, 200 candidates at ceiling 128)
        let table: [(Phase, u32, [u32; 5]); 11] = [
            (Westeros, 64, [64, 64, 64, 64, 128]),
            (WesterosChoice, 64, [64, 64, 64, 64, 128]),
            (Bidding, 32, [32, 32, 33, 46, 128]),
            (WildlingAttack, 16, [16, 16, 33, 46, 128]),
            (Mustering, 32, [32, 32, 33, 46, 128]),
            (Planning, 32, [32, 32, 33, 46, 128]),
            (Raven, 64, [64, 64, 64, 64, 128]),
            (Action, 32, [32, 32, 33, 46, 128]),
            (Combat, 64, [64, 64, 64, 64, 128]),
            (Cleanup, 64, [64, 64, 64, 64, 128]),
            (GameOver, 64, [64, 64, 64, 64, 128]),
        ];
        let counts = [1usize, 5, 32, 45, 200];
        for (phase, cap, expected) in table {
            assert_eq!(PhaseBudgets::DEFAULT.cap(phase), cap, "{phase}");
            for (actions, expected) in counts.iter().zip(expected) {
                assert_eq!(
                    PhaseBudgets::DEFAULT.for_decision(phase, *actions, 128),
                    expected,
                    "{phase} with {actions} candidates"
                );
            }
        }
        // The table is total, so a new phase cannot silently inherit a cap.
        for phase in Phase::ALL {
            assert_eq!(
                PhaseBudgets::DEFAULT.cap(phase),
                table[phase.index()].1,
                "{phase} is missing from the table"
            );
        }
    }

    #[test]
    fn the_ceiling_and_a_zero_simulations_setting_are_respected() {
        // `simulations` is a ceiling even when a phase cap asks for more.
        assert_eq!(PhaseBudgets::DEFAULT.for_decision(Phase::Combat, 5, 16), 16);
        // A zero budget behaves like the driver's own `max(1)`.
        assert_eq!(PhaseBudgets::DEFAULT.for_decision(Phase::Planning, 5, 0), 1);
        assert_eq!(decision_budget(None, Phase::Planning, 45, 0), 1);
        // `None` is the old fixed budget, whatever the phase or candidates are.
        for phase in Phase::ALL {
            for actions in [0usize, 1, 5, 45, 200] {
                assert_eq!(decision_budget(None, phase, actions, 128), 128);
                assert_eq!(decision_budget(None, phase, actions, 7), 7);
            }
        }
    }

    #[test]
    fn an_override_table_only_changes_the_named_phase() {
        let budgets = PhaseBudgets::DEFAULT.with_cap(Phase::Planning, 8);
        assert_eq!(budgets.cap(Phase::Planning), 8);
        assert_eq!(budgets.cap(Phase::Action), 32);
        assert_eq!(budgets.cap(Phase::Combat), 64);
        // The candidate floor still wins over a tiny override.
        assert_eq!(budgets.for_decision(Phase::Planning, 12, 128), 13);
        assert_eq!(budgets.for_decision(Phase::Planning, 3, 128), 8);
    }
}
