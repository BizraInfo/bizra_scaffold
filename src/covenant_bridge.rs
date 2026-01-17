// src/covenant_bridge.rs - COVENANT Integration Layer
//
// This module bridges the existing BridgeCoordinator architecture with the
// new COVENANT Article III pipeline, maintaining backward compatibility while
// enabling full SNR measurement and constitutional compliance.
//
// ARCHITECTURE DECISION:
// Rather than replacing the existing PAT/SAT flow (which works well), we:
// 1. Wire up the global SNR monitor to track all operations
// 2. Emit AttestedThought receipts alongside existing receipts
// 3. Provide an optional "covenant mode" for full 8-stage pipeline
// 4. Map existing components to COVENANT stages
//
// WEEK 2 INTEGRATION STRATEGY:
// Phase 1: Add SNR tracking to existing BridgeCoordinator (non-breaking)
// Phase 2: Add AttestedThought receipt generation (parallel to existing)
// Phase 3: Enable covenant mode as opt-in feature flag

use blake3;
use crate::{
    fixed::Fixed64,
    ihsan::IhsanDimensions,
    snr_monitor::{global_monitor, ThoughtEvent},
    thought::{
        Action, AttestedThought, Citation, GateReceipt, GateType, IhsanScore,
        ThoughtId, ThoughtStage,
    },
    types::{AgentResult, DualAgenticRequest},
};
use chrono::Utc;
use std::collections::BTreeMap;
use tracing::info;

/// COVENANT integration layer for existing BridgeCoordinator
///
/// This struct acts as a translation layer between:
/// - Existing: DualAgenticRequest → PAT/SAT → AgentResult → Receipt
/// - COVENANT: Input → 8-stage pipeline → AttestedThought → SNR metrics
pub struct CovenantBridge {
    enable_covenant_mode: bool,
}

impl CovenantBridge {
    /// Create new COVENANT bridge
    pub fn new(enable_covenant_mode: bool) -> Self {
        Self {
            enable_covenant_mode,
        }
    }

    /// Record request start in SNR monitor (COVENANT Stage 1: SENSE)
    ///
    /// Maps existing request to thought lifecycle tracking
    pub fn record_request_start(&self, request: &DualAgenticRequest) -> ThoughtId {
        let thought_id = ThoughtId::new();
        let monitor = global_monitor();

        monitor.record_event(ThoughtEvent::Attempted(thought_id));

        info!(
            thought_id = %thought_id.to_string(),
            task = %request.task,
            "🔵 COVENANT Stage 1 (SENSE): Request captured"
        );

        thought_id
    }

    /// Record PAT execution completion (COVENANT Stage 2: REASON)
    ///
    /// The existing PAT orchestrator acts as the reasoning engine
    pub fn record_reasoning_complete(
        &self,
        thought_id: ThoughtId,
        results_count: usize,
    ) {
        info!(
            thought_id = %thought_id.to_string(),
            results_count,
            "🔵 COVENANT Stage 2 (REASON): PAT execution completed"
        );
    }

    /// Record Ihsān scoring (COVENANT Stage 3: SCORE)
    ///
    /// Maps existing Ihsān calculation to COVENANT scoring stage
    pub fn record_ihsan_scoring(
        &self,
        thought_id: ThoughtId,
        ihsan_score: Fixed64,
        passes_threshold: bool,
    ) {
        let monitor = global_monitor();

        if !passes_threshold {
            monitor.record_event(ThoughtEvent::IhsanRejection(thought_id, ihsan_score));
        }

        info!(
            thought_id = %thought_id.to_string(),
            ihsan_score = %ihsan_score.to_f64(),
            passes = passes_threshold,
            "🔵 COVENANT Stage 3 (SCORE): Ihsān evaluation complete"
        );
    }

    /// Record SAT validation (COVENANT Stage 4: GATE)
    ///
    /// Maps existing SAT consensus to COVENANT gate stage
    pub fn record_sat_validation(
        &self,
        thought_id: ThoughtId,
        consensus_reached: bool,
        rejection_codes: &[String],
    ) {
        let monitor = global_monitor();

        if !consensus_reached {
            let reason = rejection_codes.join("; ");
            monitor.record_event(ThoughtEvent::FateViolation(thought_id, reason));
        }

        info!(
            thought_id = %thought_id.to_string(),
            consensus = consensus_reached,
            "🔵 COVENANT Stage 4 (GATE): SAT validation complete"
        );
    }

    /// Record action commitment (COVENANT Stage 5: ACT)
    ///
    /// Maps successful execution to COVENANT commit stage
    pub fn record_action_committed(&self, thought_id: ThoughtId) {
        let monitor = global_monitor();
        monitor.record_event(ThoughtEvent::Committed(thought_id));

        info!(
            thought_id = %thought_id.to_string(),
            "🟢 COVENANT Stage 5 (ACT): Action committed"
        );
    }

    /// Record ledger append (COVENANT Stage 6: LEDGER)
    ///
    /// The existing receipt system already handles this
    pub fn record_ledger_append(&self, thought_id: ThoughtId, receipt_id: &str) {
        info!(
            thought_id = %thought_id.to_string(),
            receipt_id,
            "🟢 COVENANT Stage 6 (LEDGER): Receipt emitted"
        );
    }

    /// Record proof generation (COVENANT Stage 7: PROOF)
    ///
    /// Maps to zk-SNARK verification in existing system
    pub fn record_proof_generated(&self, thought_id: ThoughtId, proof_verified: bool) {
        let monitor = global_monitor();
        monitor.record_event(ThoughtEvent::ProofGenerated(thought_id));

        if proof_verified {
            monitor.record_event(ThoughtEvent::ProofVerified(thought_id, true));
        }

        info!(
            thought_id = %thought_id.to_string(),
            verified = proof_verified,
            "🟢 COVENANT Stage 7 (PROOF): Proof processing complete"
        );
    }

    /// Get current SNR metrics (COVENANT Stage 8: SNR UPDATE)
    ///
    /// Returns real-time signal-to-noise ratio
    pub fn get_current_snr(&self) -> Fixed64 {
        let monitor = global_monitor();
        let snr = monitor.current_snr();

        info!(
            snr = %snr.to_f64(),
            "🟢 COVENANT Stage 8 (SNR UPDATE): Metrics updated"
        );

        snr
    }

    /// Generate COVENANT-compliant AttestedThought from existing execution
    ///
    /// This creates a canonical thought object from the existing dual-agentic flow,
    /// enabling full COVENANT compliance without breaking existing receipts.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_attested_thought(
        &self,
        thought_id: ThoughtId,
        request: &DualAgenticRequest,
        results: &[AgentResult],
        ihsan_score: Fixed64,
        ihsan_dimensions: &IhsanDimensions,
        sat_consensus: bool,
        receipt_id: Option<String>,
    ) -> AttestedThought {
        // COVENANT Stage 1: Input hash (Blake3)
        let input_hash = *blake3::hash(request.task.as_bytes()).as_bytes();

        // COVENANT Stage 2: Reasoning trace (from PAT agents)
        let reasoning_trace = results
            .iter()
            .map(|r| format!("[{}]: {}", r.agent_name, r.contribution))
            .collect::<Vec<_>>()
            .join("\n\n");

        // COVENANT Stage 3: Ihsān score
        let ihsan_score_obj = IhsanScore {
            correctness: ihsan_dimensions.adl,
            safety: ihsan_dimensions.amanah,
            user_benefit: ihsan_dimensions.ihsan,
            efficiency: ihsan_dimensions.hikmah,
            auditability: ihsan_dimensions.bayan,
            anti_centralization: ihsan_dimensions.tawhid,
            robustness: ihsan_dimensions.sabr,
            fairness: ihsan_dimensions.mizan,
            total: ihsan_score,
        };

        // COVENANT Stage 4: Gate receipts
        let mut gates_passed = vec![];
        if sat_consensus {
            gates_passed.push(GateReceipt {
                gate_type: GateType::IhsanThreshold,
                passed: true,
                reason: Some("SAT consensus reached".to_string()),
                validator_id: "SAT".to_string(),
                timestamp: Utc::now(),
                signature: vec![], // Placeholder - should sign in production
            });
        }

        // COVENANT Stage 5: Action (synthesis of agent results)
        let mut action_params = BTreeMap::new();
        action_params.insert("num_agents".to_string(), results.len().to_string());
        action_params.insert("task".to_string(), request.task.clone());

        let action = Some(Action {
            action_type: "dual_agentic_synthesis".to_string(),
            parameters: action_params,
            state_delta_hash: *blake3::hash(reasoning_trace.as_bytes()).as_bytes(),
        });

        // COVENANT Stage 6: Ledger entry (existing receipt system)
        let ledger_entry_hash = receipt_id.map(|id| *blake3::hash(id.as_bytes()).as_bytes());

        // COVENANT Stage 7: Proof (placeholder for zk-SNARK integration)
        let proof_hash = None; // Will be populated when zk integration is complete

        // COVENANT Stage 8: SNR contribution
        let contributed_to_signal = sat_consensus && ihsan_score >= Fixed64::from_f64(0.85);

        // Giants Protocol: Citations from agent results (Week 2 Phase 2)
        let citations: Vec<Citation> = vec![]; // TODO: Extract from reasoning trace

        // Ed25519 signature (placeholder for Week 2 Phase 3)
        let signature = vec![]; // TODO: Sign with sovereign key

        AttestedThought {
            id: thought_id,
            stage: if contributed_to_signal {
                ThoughtStage::ProofVerified
            } else {
                ThoughtStage::Rollback
            },
            created_at: Utc::now(),
            updated_at: Utc::now(),
            input_hash,
            input_metadata: BTreeMap::new(),
            reasoning_trace,
            output_candidate: results.iter().map(|r| r.contribution.clone()).collect::<Vec<_>>().join("\n"),
            model_id: "dual_agentic".to_string(),
            ihsan_score: ihsan_score_obj,
            gates_passed,
            action,
            ledger_entry_hash,
            proof_hash,
            proof_verified: false,
            contributed_to_signal,
            noise_reason: if !contributed_to_signal { Some("SAT consensus failed or Ihsan below threshold".to_string()) } else { None },
            citations,
            signature,
        }
    }

    /// Check if COVENANT compliance is enabled
    pub fn is_covenant_mode(&self) -> bool {
        self.enable_covenant_mode
    }

    /// Get SNR metrics report
    pub fn get_snr_report(&self) -> String {
        let monitor = global_monitor();
        monitor.report()
    }

    /// Check if system meets COVENANT threshold (SNR ≥ 0.95)
    pub fn meets_covenant_threshold(&self) -> bool {
        let monitor = global_monitor();
        monitor.meets_covenant()
    }
}

impl Default for CovenantBridge {
    fn default() -> Self {
        // Enable covenant mode by default in development
        // In production, this should be controlled by environment variable
        let enable = std::env::var("BIZRA_COVENANT_MODE")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(true); // Default enabled

        Self::new(enable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covenant_bridge_lifecycle() {
        let bridge = CovenantBridge::new(true);

        // Create mock request
        let request = DualAgenticRequest {
            task: "Test task".to_string(),
            context: HashMap::new(),
            mode: crate::types::AdapterModes::Auto,
        };

        // Stage 1: SENSE
        let thought_id = bridge.record_request_start(&request);
        assert_ne!(thought_id.to_string(), "");

        // Stage 2: REASON
        bridge.record_reasoning_complete(thought_id, 7);

        // Stage 3: SCORE
        let ihsan_score = Fixed64::from_f64(0.92);
        bridge.record_ihsan_scoring(thought_id, ihsan_score, true);

        // Stage 4: GATE
        bridge.record_sat_validation(thought_id, true, &[]);

        // Stage 5: ACT
        bridge.record_action_committed(thought_id);

        // Stage 6: LEDGER
        bridge.record_ledger_append(thought_id, "TEST-12345");

        // Stage 7: PROOF
        bridge.record_proof_generated(thought_id, true);

        // Stage 8: SNR UPDATE
        let snr = bridge.get_current_snr();
        assert!(snr >= Fixed64::ZERO);
    }

    #[test]
    fn test_attested_thought_generation() {
        let bridge = CovenantBridge::new(true);

        let request = DualAgenticRequest {
            task: "Generate summary".to_string(),
            context: HashMap::new(),
            mode: crate::types::AdapterModes::Auto,
        };

        let thought_id = ThoughtId::new();
        let results = vec![AgentResult {
            agent_name: "TestAgent".to_string(),
            output: "Test output".to_string(),
            confidence: 0.95,
            execution_time: std::time::Duration::from_millis(100),
        }];

        let ihsan_dimensions = IhsanDimensions {
            adl: Fixed64::from_f64(0.95),
            amanah: Fixed64::from_f64(0.92),
            ihsan: Fixed64::from_f64(0.93),
            hikmah: Fixed64::from_f64(0.90),
            bayan: Fixed64::from_f64(0.91),
            tawhid: Fixed64::from_f64(0.88),
            sabr: Fixed64::from_f64(0.89),
            mizan: Fixed64::from_f64(0.94),
        };

        let thought = bridge.generate_attested_thought(
            thought_id,
            &request,
            &results,
            Fixed64::from_f64(0.92),
            &ihsan_dimensions,
            true,
            Some("EXEC-12345".to_string()),
        );

        assert_eq!(thought.id, thought_id);
        assert!(thought.contributed_to_signal);
        assert_eq!(thought.gates_passed.len(), 1);
        assert!(thought.ledger_entry_hash.is_some());
    }

    #[test]
    fn test_covenant_mode_env_var() {
        // Test default behavior
        std::env::remove_var("BIZRA_COVENANT_MODE");
        let bridge1 = CovenantBridge::default();
        assert!(bridge1.is_covenant_mode()); // Default enabled

        // Test explicit enable
        std::env::set_var("BIZRA_COVENANT_MODE", "true");
        let bridge2 = CovenantBridge::default();
        assert!(bridge2.is_covenant_mode());

        // Test explicit disable
        std::env::set_var("BIZRA_COVENANT_MODE", "false");
        let bridge3 = CovenantBridge::default();
        assert!(!bridge3.is_covenant_mode());

        // Cleanup
        std::env::remove_var("BIZRA_COVENANT_MODE");
    }
}
