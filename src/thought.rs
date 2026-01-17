// src/thought.rs - Canonical Thought Object (COVENANT Article III)
//
// This module implements the constitutional thought pipeline.
// Every system action flows through this canonical type.
//
// COVENANT COMPLIANCE:
// - Hard Gate #1: All scores use Fixed64
// - Hard Gate #2: Attestation wraps every operation
// - Article III: Mandatory 8-stage lifecycle

use blake3;
use crate::fixed::Fixed64;
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use uuid::Uuid;

/// Thought ID: Unique identifier for canonical thought objects
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ThoughtId(Uuid);

impl ThoughtId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl Default for ThoughtId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ThoughtId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Blake3 Hash (256-bit)
pub type Blake3Hash = [u8; 32];

/// Thought Stage: Position in canonical pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThoughtStage {
    Sensed,           // Input captured
    Reasoned,         // Inference complete
    Scored,           // Ihsān evaluated
    GateChecked,      // FATE/Human veto complete
    Committed,        // State mutation applied
    Ledgered,         // BlockGraph entry written
    ProofPending,     // zk-SNARK generation queued
    ProofVerified,    // Proof cryptographically verified
    Rollback,         // Failed gate, no state change
}

/// Gate Result: Outcome of FATE or Human Veto gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateReceipt {
    pub gate_type: GateType,
    pub passed: bool,
    pub reason: Option<String>,
    pub validator_id: String,
    pub timestamp: DateTime<Utc>,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateType {
    FateSMT,       // Z3 formal verification
    HumanVeto,     // CLI/UI manual approval
    IhsanThreshold, // Constitutional score check
}

/// Ihsān Score: 8-dimensional constitutional evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IhsanScore {
    // COVENANT Article 2: 8 Dimensions
    pub correctness: Fixed64,         // Adl (Justice)
    pub safety: Fixed64,              // Amānah (Trust)
    pub user_benefit: Fixed64,        // Ihsān (Excellence)
    pub efficiency: Fixed64,          // Hikmah (Wisdom)
    pub auditability: Fixed64,        // Bayān (Clarity)
    pub anti_centralization: Fixed64, // Tawhīd (Unity)
    pub robustness: Fixed64,          // Sabr (Patience)
    pub fairness: Fixed64,            // Mizān (Balance)

    // Weighted total (per constitution weights)
    pub total: Fixed64,
}

impl IhsanScore {
    /// Compute weighted total using constitutional weights
    pub fn compute_total(&mut self) {
        // Weights from constitution/ihsan_v1.yaml
        let weights = crate::ihsan::constitution().weights();

        let mut sum = Fixed64::ZERO;
        for (dim, weight) in weights.iter() {
            let value = match dim.as_str() {
                "correctness" => self.correctness,
                "safety" => self.safety,
                "user_benefit" => self.user_benefit,
                "efficiency" => self.efficiency,
                "auditability" => self.auditability,
                "anti_centralization" => self.anti_centralization,
                "robustness" => self.robustness,
                "adl_fairness" => self.fairness,
                _ => Fixed64::ZERO,
            };
            sum = sum + Fixed64::from_f64(*weight) * value;
        }

        self.total = sum;
    }

    /// Check if score meets threshold (COVENANT Article 2)
    pub fn passes_threshold(&self, threshold: Fixed64) -> bool {
        self.total >= threshold
    }
}

/// Giants Citation: Reference to external knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub artifact_hash: Blake3Hash,
    pub claim_id: u64,
    pub validator_status: ValidatorStatus,
    pub reasoning: String, // Why this claim is relevant
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidatorStatus {
    Verified,
    Pending,
    Failed,
}

/// Action: State mutation committed by thought
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub action_type: String,
    pub parameters: BTreeMap<String, String>,
    pub state_delta_hash: Blake3Hash,
}

/// Attested Thought: Canonical thought object (COVENANT Article III)
///
/// This is the fundamental unit of system operation.
/// Every decision, inference, or state change flows through this type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestedThought {
    // Identity
    pub id: ThoughtId,
    pub stage: ThoughtStage,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    // COVENANT Stage 1: SENSE
    pub input_hash: Blake3Hash,
    pub input_metadata: BTreeMap<String, String>,

    // COVENANT Stage 2: REASON
    pub reasoning_trace: String,
    pub output_candidate: String,
    pub model_id: String,

    // COVENANT Stage 3: SCORE
    pub ihsan_score: IhsanScore,

    // COVENANT Stage 4: GATE
    pub gates_passed: Vec<GateReceipt>,

    // COVENANT Stage 5: ACT
    pub action: Option<Action>,

    // COVENANT Stage 6: LEDGER
    pub ledger_entry_hash: Option<Blake3Hash>,

    // COVENANT Stage 7: PROOF
    pub proof_hash: Option<Blake3Hash>,
    pub proof_verified: bool,

    // COVENANT Stage 8: SNR
    pub contributed_to_signal: bool,
    pub noise_reason: Option<String>,

    // COVENANT Article IV: GIANTS
    pub citations: Vec<Citation>,

    // Cryptographic Attestation
    pub signature: Vec<u8>,
}

impl AttestedThought {
    /// Create new thought from sensed input
    pub fn from_input(input_data: &[u8], metadata: BTreeMap<String, String>) -> Self {
        let input_hash = blake3::hash(input_data).into();

        Self {
            id: ThoughtId::new(),
            stage: ThoughtStage::Sensed,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            input_hash,
            input_metadata: metadata,
            reasoning_trace: String::new(),
            output_candidate: String::new(),
            model_id: String::new(),
            ihsan_score: IhsanScore {
                correctness: Fixed64::ZERO,
                safety: Fixed64::ZERO,
                user_benefit: Fixed64::ZERO,
                efficiency: Fixed64::ZERO,
                auditability: Fixed64::ZERO,
                anti_centralization: Fixed64::ZERO,
                robustness: Fixed64::ZERO,
                fairness: Fixed64::ZERO,
                total: Fixed64::ZERO,
            },
            gates_passed: Vec::new(),
            action: None,
            ledger_entry_hash: None,
            proof_hash: None,
            proof_verified: false,
            contributed_to_signal: false,
            noise_reason: None,
            citations: Vec::new(),
            signature: Vec::new(),
        }
    }

    /// Advance to REASONED stage
    pub fn with_reasoning(
        mut self,
        trace: String,
        output: String,
        model_id: String,
    ) -> Self {
        self.reasoning_trace = trace;
        self.output_candidate = output;
        self.model_id = model_id;
        self.stage = ThoughtStage::Reasoned;
        self.updated_at = Utc::now();
        self
    }

    /// Advance to SCORED stage
    pub fn with_ihsan_score(mut self, mut score: IhsanScore) -> Self {
        score.compute_total();
        self.ihsan_score = score;
        self.stage = ThoughtStage::Scored;
        self.updated_at = Utc::now();
        self
    }

    /// Advance to GATE_CHECKED stage
    pub fn with_gate_result(mut self, receipt: GateReceipt) -> Self {
        self.gates_passed.push(receipt);
        self.stage = ThoughtStage::GateChecked;
        self.updated_at = Utc::now();
        self
    }

    /// Advance to COMMITTED stage
    pub fn with_action(mut self, action: Action) -> Self {
        self.action = Some(action);
        self.stage = ThoughtStage::Committed;
        self.contributed_to_signal = true; // Committed = signal
        self.updated_at = Utc::now();
        self
    }

    /// Advance to ROLLBACK stage (failed gate)
    pub fn rollback(mut self, reason: String) -> Self {
        self.stage = ThoughtStage::Rollback;
        self.noise_reason = Some(reason);
        self.contributed_to_signal = false; // Rollback = noise
        self.updated_at = Utc::now();
        self
    }

    /// Sign thought with Ed25519 key
    pub fn sign(&mut self, signing_key: &SigningKey) {
        let canonical = self.canonical_bytes();
        let signature = signing_key.sign(&canonical);
        self.signature = signature.to_bytes().to_vec();
    }

    /// Verify thought signature
    pub fn verify(&self, verifying_key: &VerifyingKey) -> anyhow::Result<()> {
        let canonical = self.canonical_bytes();
        let sig = Signature::from_bytes(
            &self
                .signature
                .as_slice()
                .try_into()
                .map_err(|_| anyhow::anyhow!("Invalid signature length"))?,
        );
        verifying_key
            .verify(&canonical, &sig)
            .map_err(|e| anyhow::anyhow!("Signature verification failed: {}", e))
    }

    /// Canonical byte representation for hashing/signing
    fn canonical_bytes(&self) -> Vec<u8> {
        // COVENANT Hard Gate #1: Deterministic serialization
        let mut hasher = Sha256::new();

        hasher.update(self.id.0.as_bytes());
        hasher.update(self.input_hash);
        hasher.update(self.reasoning_trace.as_bytes());
        hasher.update(self.output_candidate.as_bytes());
        hasher.update(self.model_id.as_bytes());

        // Fixed64 to deterministic bits
        hasher.update(self.ihsan_score.total.to_bits().to_le_bytes());

        hasher.finalize().to_vec()
    }

    /// Compute Blake3 hash of thought
    pub fn hash(&self) -> Blake3Hash {
        blake3::hash(&self.canonical_bytes()).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thought_lifecycle_basic() {
        let input = b"test input";
        let mut metadata = BTreeMap::new();
        metadata.insert("source".to_string(), "camera".to_string());

        let thought = AttestedThought::from_input(input, metadata);

        assert_eq!(thought.stage, ThoughtStage::Sensed);
        assert!(!thought.contributed_to_signal);
    }

    #[test]
    fn ihsan_score_threshold() {
        let mut score = IhsanScore {
            correctness: Fixed64::from_f64(0.95),
            safety: Fixed64::from_f64(0.90),
            user_benefit: Fixed64::from_f64(0.85),
            efficiency: Fixed64::from_f64(0.88),
            auditability: Fixed64::from_f64(0.92),
            anti_centralization: Fixed64::from_f64(0.80),
            robustness: Fixed64::from_f64(0.87),
            fairness: Fixed64::from_f64(0.91),
            total: Fixed64::ZERO,
        };

        score.compute_total();

        // Should be weighted average, approximately 0.89
        assert!(score.total > Fixed64::from_f64(0.85));
        assert!(score.passes_threshold(Fixed64::from_f64(0.85)));
    }

    #[test]
    fn thought_signing_roundtrip() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        let input = b"test";
        let mut thought = AttestedThought::from_input(input, BTreeMap::new());
        thought.sign(&signing_key);

        assert!(thought.verify(&verifying_key).is_ok());
    }

    #[test]
    fn thought_hash_deterministic() {
        let input = b"determinism test";
        let thought1 = AttestedThought::from_input(input, BTreeMap::new());
        let thought2 = thought1.clone();

        assert_eq!(thought1.hash(), thought2.hash());
    }
}
