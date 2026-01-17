use crate::fixed::Fixed64;
use serde::{Deserialize, Serialize};

/// Proof-of-Impact (PoI) Engine
/// Implements Layer 5 (Economic) of the APEX Architecture.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PoIEngine {
    pub weights: ImpactWeights,
    pub bzc_mint_rate: Fixed64,
    pub harberger_tax_rate: Fixed64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImpactWeights {
    pub code_quality: Fixed64,
    pub documentation: Fixed64,
    pub signal_to_noise: Fixed64,
    pub ethical_adherence: Fixed64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImpactScore {
    pub total: Fixed64,
    pub bzc_minted: Fixed64,
    pub bzt_voting_power: Fixed64,
}

impl Default for PoIEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PoIEngine {
    pub fn new() -> Self {
        Self {
            weights: ImpactWeights {
                code_quality: Fixed64::from_f64(0.4),
                documentation: Fixed64::from_f64(0.2),
                signal_to_noise: Fixed64::from_f64(0.2),
                ethical_adherence: Fixed64::from_f64(0.2),
            },
            bzc_mint_rate: Fixed64::from_f64(10.0), // 10 BZC per unit of impact
            harberger_tax_rate: Fixed64::from_f64(0.05), // 5% annual Harberger tax
        }
    }

    /// Calculate Impact Score based on agent output and system metrics.
    pub fn calculate_impact(
        &self,
        artifact_quality: Fixed64,
        ihsan_score: Fixed64,
        snr_ratio: Fixed64,
    ) -> ImpactScore {
        // total = (artifact_quality * self.weights.code_quality)
        //     + (ihsan_score * self.weights.ethical_adherence)
        //     + (snr_ratio.min(2.0) / 2.0 * self.weights.signal_to_noise);

        let snr_contribution = snr_ratio
            .min(Fixed64::from_f64(2.0))
            .saturating_div(Fixed64::from_f64(2.0))
            .saturating_mul(self.weights.signal_to_noise);

        let total = artifact_quality
            .saturating_mul(self.weights.code_quality)
            .saturating_add(ihsan_score.saturating_mul(self.weights.ethical_adherence))
            .saturating_add(snr_contribution);

        ImpactScore {
            total,
            bzc_minted: total.saturating_mul(self.bzc_mint_rate),
            bzt_voting_power: total, // 1:1 impact to voting power
        }
    }

    /// Calculate Harberger Tax for a digital asset.
    /// tax = value * rate * (time_held / year)
    pub fn calculate_harberger_tax(&self, asset_value: Fixed64, years: Fixed64) -> Fixed64 {
        asset_value
            .saturating_mul(self.harberger_tax_rate)
            .saturating_mul(years)
    }
}

/// Integration with the Bridge
pub fn record_impact(engine: &PoIEngine, quality: f64, ihsan: f64, snr: f64) -> ImpactScore {
    let score = engine.calculate_impact(
        Fixed64::from_f64(quality),
        Fixed64::from_f64(ihsan),
        Fixed64::from_f64(snr),
    );
    // In a real system, this would write to a signed ledger or blockchain.
    println!(
        "[PoI] Impact calculated: {}. Minting {} BZC.",
        score.total.to_f64(),
        score.bzc_minted.to_f64()
    );
    score
}
