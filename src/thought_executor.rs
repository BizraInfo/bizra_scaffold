// src/thought_executor.rs - Minimal Thought Executor (COVENANT Article III)
//
// This is the "smallest loop that forces truth to pay rent":
// A working end-to-end pipeline that demonstrates SNR measurement in practice.
//
// COVENANT COMPLIANCE:
// - Hard Gate #1: All metrics use Fixed64
// - Article III: 8-stage thought lifecycle enforced
// - Article V: SNR metrics tracked for every thought

use blake3;
use crate::fixed::Fixed64;
use crate::ihsan::{IhsanConstitution, IhsanDimensions, compute_ihsan_score};
use crate::snr_monitor::{global_monitor, ThoughtEvent};
use crate::thought::{
    Action, AttestedThought, GateReceipt, GateType, IhsanScore,
    ThoughtId, ThoughtStage,
};
use anyhow::Result;
use chrono::Utc;

/// Stub Reasoner: Minimal inference engine for testing
/// In production, this would call Ollama/OpenAI/Gemini
pub struct StubReasoner {
    model_name: String,
}

impl StubReasoner {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
        }
    }

    /// Generate reasoning trace (stub implementation)
    pub fn reason(&self, input: &str) -> String {
        format!(
            "[StubReasoner:{}] Input: '{}'\nReasoning: Applying constitutional principles...\nOutput: Action proposed",
            self.model_name, input
        )
    }

    /// Evaluate reasoning quality (stub - returns high scores for testing)
    pub fn evaluate(&self, _trace: &str) -> IhsanDimensions {
        IhsanDimensions {
            adl: Fixed64::from_f64(0.95),      // Correctness
            amanah: Fixed64::from_f64(0.92),   // Safety
            ihsan: Fixed64::from_f64(0.97),    // User benefit
            hikmah: Fixed64::from_f64(0.90),   // Efficiency
            bayan: Fixed64::from_f64(0.93),    // Auditability
            tawhid: Fixed64::from_f64(0.88),   // Anti-centralization
            sabr: Fixed64::from_f64(0.91),     // Robustness
            mizan: Fixed64::from_f64(0.94),    // Fairness
        }
    }
}

/// FATE Gate (stub): In production, calls Z3 SMT solver
pub struct StubFateGate;

impl StubFateGate {
    pub fn verify(&self, thought: &AttestedThought) -> Result<GateReceipt> {
        // Stub: Always passes unless action type contains "UNSAFE"
        let passed = !thought
            .action
            .as_ref()
            .map(|a| a.action_type.contains("UNSAFE"))
            .unwrap_or(false);

        Ok(GateReceipt {
            gate_type: GateType::FateSMT,
            passed,
            reason: Some(if passed {
                "Constraints satisfied (stub)".to_string()
            } else {
                "UNSAFE action detected".to_string()
            }),
            validator_id: "FATE_STUB".to_string(),
            timestamp: Utc::now(),
            signature: vec![], // Placeholder
        })
    }
}

/// Thought Executor: Orchestrates the 8-stage COVENANT pipeline
#[allow(dead_code)] // Reserved fields for future expansion
pub struct ThoughtExecutor {
    reasoner: StubReasoner,
    fate_gate: StubFateGate,
    constitution: &'static IhsanConstitution,
}

impl ThoughtExecutor {
    /// Create new executor with stub components
    pub fn new_stub() -> Self {
        Self {
            reasoner: StubReasoner::new("stub-v1"),
            fate_gate: StubFateGate,
            constitution: crate::ihsan::constitution(),
        }
    }

    /// Execute a thought through the full COVENANT pipeline
    ///
    /// Returns (thought, receipt) on success, or rollback reason on failure
    pub fn execute(&self, input: &str) -> Result<(AttestedThought, String)> {
        let thought_id = ThoughtId::new();
        let monitor = global_monitor();

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // STAGE 1: SENSE - Capture input + generate hash
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        monitor.record_event(ThoughtEvent::Attempted(thought_id));

        let input_hash = *blake3::hash(input.as_bytes()).as_bytes();

        tracing::info!("🔵 STAGE 1 (SENSE): thought_id={}, input_hash={}",
            thought_id.to_string(), hex::encode(input_hash));

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // STAGE 2: REASON - Inference + trace generation
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let reasoning_trace = self.reasoner.reason(input);

        tracing::info!("🔵 STAGE 2 (REASON): Generated reasoning trace ({} chars)",
            reasoning_trace.len());

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // STAGE 3: SCORE - Ihsān 8-dimensional evaluation
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let dimensions = self.reasoner.evaluate(&reasoning_trace);
        let total = compute_ihsan_score(
            dimensions.adl.to_f64(),
            dimensions.amanah.to_f64(),
            dimensions.ihsan.to_f64(),
            dimensions.hikmah.to_f64(),
            dimensions.bayan.to_f64(),
            dimensions.tawhid.to_f64(),
            dimensions.sabr.to_f64(),
            dimensions.mizan.to_f64(),
        )?;

        let ihsan_score = IhsanScore {
            correctness: dimensions.adl,
            safety: dimensions.amanah,
            user_benefit: dimensions.ihsan,
            efficiency: dimensions.hikmah,
            auditability: dimensions.bayan,
            anti_centralization: dimensions.tawhid,
            robustness: dimensions.sabr,
            fairness: dimensions.mizan,
            total: Fixed64::from_f64(total),
        };

        let threshold = Fixed64::from_f64(0.85); // COVENANT default
        tracing::info!("🔵 STAGE 3 (SCORE): Ihsān total={:.4}", ihsan_score.total.to_f64());

        // Check Ihsān threshold
        if !ihsan_score.passes_threshold(threshold) {
            let reason = format!(
                "Ihsān score {:.4} below threshold {:.4}",
                ihsan_score.total.to_f64(),
                threshold.to_f64()
            );
            monitor.record_event(ThoughtEvent::IhsanRejection(thought_id, ihsan_score.total));
            return Err(anyhow::anyhow!("ROLLBACK: {}", reason));
        }

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // STAGE 4: GATE - FATE verification (Z3 SMT in production)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let mut thought = AttestedThought {
            id: thought_id,
            stage: ThoughtStage::Sensed,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            input_hash,
            input_metadata: std::collections::BTreeMap::new(),
            reasoning_trace: reasoning_trace.clone(),
            output_candidate: reasoning_trace.clone(),
            model_id: "covenant_executor".to_string(),
            ihsan_score,
            gates_passed: vec![],
            action: Some(Action {
                action_type: "test_action".to_string(),
                parameters: std::collections::BTreeMap::new(),
                state_delta_hash: *blake3::hash(reasoning_trace.as_bytes()).as_bytes(),
            }),
            ledger_entry_hash: None,
            proof_hash: None,
            proof_verified: false,
            contributed_to_signal: false,
            noise_reason: None,
            citations: vec![],
            signature: vec![],
        };

        let fate_receipt = self.fate_gate.verify(&thought)?;

        tracing::info!("🔵 STAGE 4 (GATE): FATE gate {}",
            if fate_receipt.passed { "PASSED" } else { "FAILED" });

        if !fate_receipt.passed {
            let reason = fate_receipt.reason.clone().unwrap_or_else(|| "Unknown FATE failure".to_string());
            monitor.record_event(ThoughtEvent::FateViolation(
                thought_id,
                reason.clone(),
            ));
            return Err(anyhow::anyhow!("ROLLBACK: FATE gate failed - {}", reason));
        }

        thought.gates_passed.push(fate_receipt);

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // STAGE 5: ACT - Commit to state
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        thought.stage = ThoughtStage::Committed;
        monitor.record_event(ThoughtEvent::Committed(thought_id));

        tracing::info!("🟢 STAGE 5 (ACT): Thought committed");

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // STAGE 6: LEDGER - BlockGraph append (stub: just hash)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let ledger_entry = format!(
            "{{\"thought_id\":\"{}\",\"stage\":\"Committed\",\"timestamp\":\"{}\"}}",
            thought_id,
            Utc::now().to_rfc3339()
        );
        thought.ledger_entry_hash = Some(*blake3::hash(ledger_entry.as_bytes()).as_bytes());

        tracing::info!("🟢 STAGE 6 (LEDGER): Entry hash={}",
            hex::encode(thought.ledger_entry_hash.as_ref().unwrap()));

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // STAGE 7: PROOF - zk-SNARK generation (async in production)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        monitor.record_event(ThoughtEvent::ProofGenerated(thought_id));
        thought.stage = ThoughtStage::ProofPending;

        // Stub: Immediately mark as verified (in production, this is async)
        let proof_data = format!("stub_proof_{}", thought_id);
        thought.proof_hash = Some(*blake3::hash(proof_data.as_bytes()).as_bytes());
        thought.stage = ThoughtStage::ProofVerified;

        monitor.record_event(ThoughtEvent::ProofVerified(thought_id, true));

        tracing::info!("🟢 STAGE 7 (PROOF): Proof hash={}",
            hex::encode(thought.proof_hash.as_ref().unwrap()));

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // STAGE 8: SNR UPDATE - Metrics increment
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        thought.contributed_to_signal = true;

        let final_snr = monitor.current_snr();
        tracing::info!("🟢 STAGE 8 (SNR UPDATE): Current SNR={:.4}", final_snr.to_f64());

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // Generate receipt
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let receipt = serde_json::json!({
            "thought_id": thought_id.to_string(),
            "input_hash": hex::encode(thought.input_hash),
            "ihsan_score": thought.ihsan_score.total.to_f64(),
            "gates_passed": thought.gates_passed.len(),
            "ledger_hash": thought.ledger_entry_hash.as_ref().map(hex::encode),
            "proof_hash": thought.proof_hash.as_ref().map(hex::encode),
            "contributed_to_signal": thought.contributed_to_signal,
            "current_snr": final_snr.to_f64(),
            "timestamp": Utc::now().to_rfc3339(),
        });

        Ok((thought, serde_json::to_string_pretty(&receipt)?))
    }
}

impl Default for ThoughtExecutor {
    fn default() -> Self {
        Self::new_stub()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_pipeline_success() {
        let executor = ThoughtExecutor::new_stub();
        let result = executor.execute("Test input: safe operation");

        assert!(result.is_ok());
        let (thought, receipt) = result.unwrap();

        assert_eq!(thought.stage, ThoughtStage::ProofVerified);
        assert!(thought.contributed_to_signal);
        assert!(thought.ledger_entry_hash.is_some());
        assert!(thought.proof_hash.is_some());
        assert!(!receipt.is_empty());
    }

    #[test]
    fn test_pipeline_fate_rejection() {
        let executor = ThoughtExecutor::new_stub();
        let result = executor.execute("UNSAFE operation that should fail");

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("FATE gate failed"));
    }

    #[test]
    fn test_snr_increments() {
        let monitor = global_monitor();
        let initial_snr = monitor.current_snr();

        let executor = ThoughtExecutor::new_stub();
        let _ = executor.execute("First thought");
        let _ = executor.execute("Second thought");

        let final_snr = monitor.current_snr();
        // SNR should change after processing thoughts
        assert_ne!(initial_snr.to_bits(), final_snr.to_bits());
    }

    #[test]
    fn test_receipt_generation() {
        let executor = ThoughtExecutor::new_stub();
        let result = executor.execute("Receipt test");

        assert!(result.is_ok());
        let (_thought, receipt) = result.unwrap();

        // Verify receipt contains all required fields
        let parsed: serde_json::Value = serde_json::from_str(&receipt).unwrap();
        assert!(parsed["thought_id"].is_string());
        assert!(parsed["ihsan_score"].is_f64());
        assert!(parsed["current_snr"].is_f64());
        assert!(parsed["contributed_to_signal"].is_boolean());
    }
}
