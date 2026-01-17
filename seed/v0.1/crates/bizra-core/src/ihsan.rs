use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct IhsanScore {
    pub truthfulness: f32, // 0..1
    pub benefit: f32,      // 0..1
    pub alignment: f32,    // 0..1
    pub composite: f32,    // 0..1
}

impl IhsanScore {
    pub fn new(truthfulness: f32, benefit: f32, alignment: f32) -> Self {
        let composite = (0.45 * truthfulness) + (0.35 * benefit) + (0.20 * alignment);
        Self { truthfulness, benefit, alignment, composite }
    }
}

/// Seed engine: deterministic, explainable, testable.
/// Later, this can be upgraded to richer evaluators.
#[derive(Clone, Debug)]
pub struct IhsanEngine {
    pub threshold: f32,
}

impl IhsanEngine {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    pub fn score_action(&self, predicted_ok: bool, observed_ok: bool, policy_ok: bool) -> IhsanScore {
        let truthfulness = if predicted_ok == observed_ok { 1.0 } else { 0.0 };
        let benefit = if observed_ok { 1.0 } else { 0.0 };
        let alignment = if policy_ok { 1.0 } else { 0.0 };
        IhsanScore::new(truthfulness, benefit, alignment)
    }

    pub fn should_commit(&self, score: IhsanScore) -> bool {
        score.composite >= self.threshold
    }
}
