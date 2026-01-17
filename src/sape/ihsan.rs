pub const WEIGHT_BENEVOLENCE: f64 = 0.4;
pub const WEIGHT_TRUTH: f64 = 0.3;
pub const WEIGHT_JUSTICE: f64 = 0.3;

#[derive(Debug, thiserror::Error)]
pub enum IhsanError {
    #[error("Input metrics out of bounds (must be 0.0 - 1.0)")]
    InputOutOfBounds,
}

pub fn calculate_score(b: f64, t: f64, j: f64) -> Result<f64, IhsanError> {
    if !(0.0..=1.0).contains(&b) || !(0.0..=1.0).contains(&t) || !(0.0..=1.0).contains(&j) {
        return Err(IhsanError::InputOutOfBounds);
    }
    Ok((b * WEIGHT_BENEVOLENCE) + (t * WEIGHT_TRUTH) + (j * WEIGHT_JUSTICE))
}

/// Calculate Unified Ihsān Score (8-factor Model)
/// Single Source of Truth
/// Source: constitution/ihsan_v1.yaml
#[allow(clippy::too_many_arguments)] // 8 Ihsān dimensions are the API contract
pub fn calculate_unified_score(
    correctness: f64,
    safety: f64,
    benefit: f64,
    efficiency: f64,
    auditability: f64,
    anti_centralization: f64,
    robustness: f64,
    adl_fairness: f64,
) -> Result<f64, IhsanError> {
    Ok(correctness * 0.22
        + safety * 0.22
        + benefit * 0.14
        + efficiency * 0.12
        + auditability * 0.12
        + anti_centralization * 0.08
        + robustness * 0.06
        + adl_fairness * 0.04)
}

// --- ELITE EXTENSION ---

/// The Golden Ratio (φ) used for architectural balance
pub const PHI: f64 = 1.61803398875;

/// The APEX Threshold for "Masterpiece" quality functionality
pub const MASTERPIECE_THRESHOLD: f64 = 0.95;

/// Evaluates alignment with the "Standing on Shoulders of Giants" protocol.
/// Returns a multiplier (1.0 - 1.5) based on citation of core axioms.
pub fn giant_shoulder_modifier(content: &str) -> f64 {
    let giants = [
        "sovereign",
        "first principles",
        "axiom",
        "logic",
        "proof",
        "truth",
    ];
    let mut multiplier: f64 = 1.0;
    for term in giants {
        if content.to_lowercase().contains(term) {
            multiplier += 0.05;
        }
    }
    multiplier.min(PHI) // Cap at Golden Ratio
}
