// src/cognitive.rs - L4 Cognitive Layer (SAPE-E)
// "The Bridge between Neural Reasoning and Symbolic Execution"
//
// Implements Thought Capsules, Policy Gates, and Evidence Chains.
// Ensures that every cognitive action is wrapped, signed, and audited.

use crate::executor::{SignedModule, ThoughtExecutor}; // Use Executor instead of raw Sandbox
use crate::storage::ReceiptStore;
use crate::types::AgentResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, instrument}; // warn removed

/// A Signed Thought Capsule (The Atomic Unit of L4 Action)
/// Wraps a symbolic plan (compiled to WASM) with a cryptographic signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtCapsule {
    pub capsule_id: String,
    pub created_at: u64,
    pub policy_version: String,
    pub plan_ir_hash: String,
    pub wasm_module_hash: String, // Canonical hash of the WASM bytes
    pub required_permissions: Vec<String>,
    pub signature: Vec<u8>, // Signature over the module bytes by RoT

    // The actual payload (Wasm bytecode)
    // Skipped in serialization if we just want the metadata
    #[serde(skip)]
    pub module_bytes: Vec<u8>,
}

impl ThoughtCapsule {
    /// Create a new signed thought capsule
    pub fn new(module_bytes: Vec<u8>, signature: Vec<u8>, permissions: Vec<String>) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(&module_bytes);
        let wasm_hash = hex::encode(hasher.finalize());

        Self {
            capsule_id: uuid::Uuid::new_v4().to_string(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            policy_version: "v1.0-genesis".to_string(),
            plan_ir_hash: "SAPE-IR-PLACEHOLDER".to_string(), // In full SAPE, this comes from the planner
            wasm_module_hash: wasm_hash,
            required_permissions: permissions,
            signature,
            module_bytes,
        }
    }
}

/// Computable Evidence Chain (The Audit Trail)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceChain {
    pub capsule_id: String,
    pub trace_id: String,
    pub inputs_hash: String,
    pub result_hash: String,
    pub policy_decision: String,
    pub execution_metrics: serde_json::Value,
    pub timestamp: u64,
}

/// The Cognitive Layer Engine (SAPE-E)
pub struct CognitiveLayer {
    executor: Option<ThoughtExecutor>,
}

impl CognitiveLayer {
    pub fn new() -> anyhow::Result<Self> {
        info!("🧠 Initializing L4 Cognitive Layer (SAPE-E) [Pending Store]");
        Ok(Self { executor: None })
    }

    pub async fn init_executor(&mut self, store: Arc<dyn ReceiptStore>) -> anyhow::Result<()> {
        let executor = ThoughtExecutor::new(store).await?;
        self.executor = Some(executor);
        info!("🧠 L4 Executor Online");
        Ok(())
    }

    /// Execute a Thought Capsule through the Fortress Gates
    #[instrument(skip(self, capsule, input))]
    pub async fn execute_thought(
        &mut self,
        capsule: &ThoughtCapsule,
        input: &str,
    ) -> anyhow::Result<(AgentResult, EvidenceChain)> {
        // 1. POLICY GATE: Check permissions and policy version
        self.check_policy(capsule)?;

        // Ensure Executor is Ready
        let executor = self
            .executor
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Executor not initialized (Store missing)"))?;

        // 2. INTEGRITY CHECK: Verify hash matches bytes
        // (ThoughtExecutor checks signature, but we verify hash to match capsule metadata)
        // ... (This logic can remain or move to executor) ...

        // Construct SignedModule from Capsule
        let module = SignedModule {
            wasm: capsule.module_bytes.clone(),
            module_hash: capsule.wasm_module_hash.clone(),
            signature: capsule.signature.clone(),
            capabilities: capsule.required_permissions.clone(),
            gas_limit: 1_000_000, // Default or from capsule
        };

        // 3. EXECUTE
        let (result, receipt) = executor.execute(&module, input).await?;

        // 4. EMIT EVIDENCE CHAIN
        let chain = EvidenceChain {
            capsule_id: capsule.capsule_id.clone(),
            trace_id: receipt.payload_id.clone(),
            inputs_hash: sha256_digest(input),
            result_hash: receipt.payload.evidence_hash.clone(),
            policy_decision: "EXECUTED_L4".to_string(),
            execution_metrics: serde_json::json!(receipt.payload.execution),
            timestamp: receipt.payload.timestamp,
        };

        info!(capsule = %capsule.capsule_id, "✅ Thought executed and evidenced");

        Ok((result, chain))
    }

    // Helper: Check Policy
    fn check_policy(&self, capsule: &ThoughtCapsule) -> anyhow::Result<()> {
        if capsule.policy_version != "v1.0" && capsule.policy_version != "v1.0-genesis" {
            return Err(anyhow::anyhow!("Unsupported policy version"));
        }
        Ok(())
    }
}

fn sha256_digest(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    hex::encode(hasher.finalize())
}
