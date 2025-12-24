use crate::models::{EvidenceBundle, IhsanScore};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ScoringError {
    #[error("Dimension score out of range [0,1]")]
    InvalidRange,
    #[error("Ihsan score below required threshold")]
    IhsanViolation,
}

// Weights from SOT Section 4
const W_QUALITY: f64 = 0.30;
const W_UTILITY: f64 = 0.30;
const W_TRUST: f64 = 0.20;
const W_FAIRNESS: f64 = 0.10;
const W_DIVERSITY: f64 = 0.10;

const IHSAN_THRESHOLD: f64 = 0.95;
const MAX_PENALTY: f64 = 0.15;
const PENALTY_KEYS: [&str; 2] = ["penalty", "negative_effects"];

pub fn calculate_poi(evidence: &EvidenceBundle) -> Result<f64, ScoringError> {
    let d = &evidence.dimensions;
    validate_range(d.quality)?;
    validate_range(d.utility)?;
    validate_range(d.trust)?;
    validate_range(d.fairness)?;
    validate_range(d.diversity)?;

    let raw_poi = d.quality * W_QUALITY
        + d.utility * W_UTILITY
        + d.trust * W_TRUST
        + d.fairness * W_FAIRNESS
        + d.diversity * W_DIVERSITY;

    let penalty = extract_penalty(&evidence.metadata);
    let final_poi = raw_poi * (1.0 - penalty.min(MAX_PENALTY));

    Ok(final_poi.max(0.0))
}

pub fn verify_ihsan(score: &IhsanScore) -> Result<bool, ScoringError> {
    validate_range(score.truthfulness)?;
    validate_range(score.dignity)?;
    validate_range(score.fairness)?;
    validate_range(score.excellence)?;
    validate_range(score.sustainability)?;

    let total = score.total();
    if total < IHSAN_THRESHOLD {
        return Err(ScoringError::IhsanViolation);
    }
    Ok(true)
}

fn validate_range(val: f64) -> Result<(), ScoringError> {
    if !val.is_finite() || val < 0.0 || val > 1.0 {
        Err(ScoringError::InvalidRange)
    } else {
        Ok(())
    }
}

fn extract_penalty(metadata: &std::collections::HashMap<String, String>) -> f64 {
    for key in PENALTY_KEYS.iter() {
        if let Some(value) = metadata.get(*key) {
            if let Ok(parsed) = value.parse::<f64>() {
                if parsed.is_finite() {
                    return parsed.max(0.0);
                }
            }
        }
    }
    0.0
}
