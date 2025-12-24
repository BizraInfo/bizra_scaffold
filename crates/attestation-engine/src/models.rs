use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attestation {
    pub attestation_id: String,
    pub contributor: String,
    pub validator: String,
    pub epoch: u64,
    pub evidence_root: String,
    pub poi_score: f64,
    pub validation_score: f64,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationPayload {
    pub attestation_id: String,
    pub contributor: String,
    pub validator: String,
    pub epoch: u64,
    pub evidence_root: String,
    pub poi_score: f64,
    pub validation_score: f64,
}

impl Attestation {
    pub fn payload(&self) -> AttestationPayload {
        AttestationPayload {
            attestation_id: self.attestation_id.clone(),
            contributor: self.contributor.clone(),
            validator: self.validator.clone(),
            epoch: self.epoch,
            evidence_root: self.evidence_root.clone(),
            poi_score: self.poi_score,
            validation_score: self.validation_score,
        }
    }
}

impl AttestationPayload {
    pub fn into_attestation(self, signature: String) -> Attestation {
        Attestation {
            attestation_id: self.attestation_id,
            contributor: self.contributor,
            validator: self.validator,
            epoch: self.epoch,
            evidence_root: self.evidence_root,
            poi_score: self.poi_score,
            validation_score: self.validation_score,
            signature,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceBundle {
    pub content_hash: String,
    pub metadata: HashMap<String, String>,
    pub dimensions: DimensionScores,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScores {
    pub quality: f64,
    pub utility: f64,
    pub trust: f64,
    pub fairness: f64,
    pub diversity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IhsanScore {
    pub truthfulness: f64,
    pub dignity: f64,
    pub fairness: f64,
    pub excellence: f64,
    pub sustainability: f64,
}

impl IhsanScore {
    pub fn total(&self) -> f64 {
        // Weights from SOT 3.1
        self.truthfulness * 0.30
            + self.dignity * 0.20
            + self.fairness * 0.20
            + self.excellence * 0.20
            + self.sustainability * 0.10
    }
}
