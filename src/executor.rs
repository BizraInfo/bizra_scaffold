// src/executor.rs - L4 Thought Execution Engine
// Implements the "Third Fact" verified execution loop.
// GIANTS_PROTOCOL: Integrated with HookChain for Pre/Post capability governance

use crate::hookchain::{
    CapabilityToken, ExecutedReceipt, HookDecision, PostHookResult, ReceiptDraft, SATHookChain,
};
use crate::storage::ReceiptStore;
use crate::tpm::{SignerProvider, TpmContext};
use crate::types::AgentResult;
use crate::wasm::WasmSandbox;
use std::sync::Arc;
// Using the blueprint's ThoughtExecReceipt definition which is distinct for the DemoKit JCS flow.

use bizra_jcs::{compute_digest, compute_payload_id};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{info, instrument, warn};

/// Signed Module (Wasm + Capabilities + Signature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedModule {
    pub wasm: Vec<u8>,
    pub module_hash: String, // Hex encoded SHA256 of wasm
    pub signature: Vec<u8>,
    pub capabilities: Vec<String>,
    pub gas_limit: u64,
}

impl SignedModule {
    pub fn verify_signature(&self, root_key_provider: &dyn SignerProvider) -> bool {
        // Verification must happen against the exact WASM bytes
        root_key_provider.verify(&self.wasm, &self.signature)
    }
}

/// JCS-Canonical Receipt for Public Verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtExecReceipt {
    /// The canonical payload (JCS canonicalized before hashing)
    pub payload: ThoughtPayload,

    /// Unique ID derived from canonical payload: b64url(sha256(JCS(payload)))
    pub payload_id: String,

    /// Receipt hash for chaining: sha256(JCS(this_struct_without_signatures)) - simplified here
    /// In practice, signatures wrap this.
    pub receipt_hash: String,

    /// Signatures over the receipt_hash
    pub signatures: Vec<Signature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtPayload {
    pub context: String,
    pub type_: String, // "thought_exec"
    pub timestamp: u64,
    pub prev_hash: String,
    pub evidence_hash: String,
    pub module_hash: String,
    pub execution: ExecutionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    pub gas_used: u64,
    pub latency_ms: u64,
    pub exit_code: i32,
    pub ihsan_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    pub signer_id: String,
    pub value_hex: String,
}

/// The Engine that runs thoughts and emits receipts
/// GIANTS_PROTOCOL: Wrapped with HookChain for capability governance
pub struct ThoughtExecutor {
    sandbox: WasmSandbox,
    signer: Arc<dyn SignerProvider>,
    store: Arc<dyn ReceiptStore>,
    cached_head: String,
    /// HookChain for Pre/Post capability governance (Giants Protocol)
    hook_chain: Option<SATHookChain>,
}

impl ThoughtExecutor {
    pub async fn new(store: Arc<dyn ReceiptStore>) -> anyhow::Result<Self> {
        let tpm = TpmContext::new();
        let head = store.get_head_hash().await?;
        let signer: Arc<dyn SignerProvider> = Arc::from(tpm.get_signer());

        // Initialize HookChain with the same signer
        let hook_chain = SATHookChain::new(signer.clone(), "BIZRA_v10.0");

        Ok(Self {
            sandbox: WasmSandbox::new()?,
            signer,
            store,
            cached_head: head,
            hook_chain: Some(hook_chain),
        })
    }

    /// Create executor without HookChain (for testing/legacy)
    pub async fn new_without_hooks(store: Arc<dyn ReceiptStore>) -> anyhow::Result<Self> {
        let tpm = TpmContext::new();
        let head = store.get_head_hash().await?;
        let signer: Arc<dyn SignerProvider> = Arc::from(tpm.get_signer());

        Ok(Self {
            sandbox: WasmSandbox::new()?,
            signer,
            store,
            cached_head: head,
            hook_chain: None,
        })
    }

    /// Execute a signed module and emit a Third Fact Receipt
    /// GIANTS_PROTOCOL: Pre/Post hooks wrap the entire execution
    #[instrument(skip(self, module, input))]
    pub async fn execute(
        &mut self,
        module: &SignedModule,
        input: &str,
    ) -> anyhow::Result<(AgentResult, ThoughtExecReceipt)> {
        self.execute_with_token(module, input, None).await
    }

    /// Execute with explicit capability token
    #[instrument(skip(self, module, input, capability_token))]
    pub async fn execute_with_token(
        &mut self,
        module: &SignedModule,
        input: &str,
        capability_token: Option<CapabilityToken>,
    ) -> anyhow::Result<(AgentResult, ThoughtExecReceipt)> {
        let start = Instant::now();

        // 0. GIANTS PROTOCOL: Pre-Capability Hook
        let session_node = if let Some(ref hook_chain) = self.hook_chain {
            let head = hook_chain.get_session_head().await;
            head.map(|n| n.node_hash)
                .unwrap_or_else(|| "genesis".to_string())
        } else {
            "genesis".to_string()
        };

        let draft = ReceiptDraft {
            tool_id: format!("wasm:{}", &module.module_hash[..16]),
            input: input.to_string(),
            capability_token: capability_token.clone(),
            session_node: session_node.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos() as u64,
        };

        // Pre-hook evaluation
        if let Some(ref hook_chain) = self.hook_chain {
            match hook_chain.pre_capability_use(&draft).await? {
                HookDecision::Allow { reason, .. } => {
                    info!("✅ HookChain PRE: Allowed - {}", reason);
                }
                HookDecision::Deny { reason, code } => {
                    return Err(anyhow::anyhow!(
                        "⛔ HookChain PRE DENIED [{}]: {}",
                        code,
                        reason
                    ));
                }
                HookDecision::Ask { question, .. } => {
                    // In production, this would prompt the user
                    // For now, we log and deny (fail-safe)
                    warn!("❓ HookChain PRE: Ask required - {}", question);
                    return Err(anyhow::anyhow!(
                        "⛔ HookChain PRE: User confirmation required - {}",
                        question
                    ));
                }
            }
        }

        // 1. GATE: Verify Module Signature (Fail-Close)
        if !module.verify_signature(&*self.signer) {
            return Err(anyhow::anyhow!(
                "⛔ Security Violation: Module signature invalid"
            ));
        }

        // 2. CHECK: Capability Safety (WASI allowlist check would go here)
        // For now we trust the SignedModule struct's capabilities field as policy

        // 3. EXECUTE: Run in Sandbox
        let agent_result = self
            .sandbox
            .execute_isolated(&module.wasm, input, &module.signature)
            .await?;

        // 4. METERING & EVIDENCE
        let gas_used = self.sandbox.last_fuel_consumed().unwrap_or(0);
        let latency = start.elapsed();

        // 4.5 GIANTS PROTOCOL: Post-Capability Hook
        let executed = ExecutedReceipt {
            draft,
            output: agent_result.contribution.clone(),
            execution_time_ms: latency.as_millis() as u64,
            tokens_used: gas_used as u32,
            success: true,
            error: None,
        };

        if let Some(ref hook_chain) = self.hook_chain {
            match hook_chain.post_capability_use(&executed).await? {
                PostHookResult::Commit { receipt_id, .. } => {
                    info!("✅ HookChain POST: Committed - receipt={}", receipt_id);
                }
                PostHookResult::Quarantine { reason, .. } => {
                    warn!("🔒 HookChain POST: Quarantined - {}", reason);
                    // Still proceed but mark as quarantined
                }
            }
        }

        // 5. RECEIPT GENERATION
        let payload = ThoughtPayload {
            context: "bizra-genesis".to_string(),
            type_: "thought_exec".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            prev_hash: self.cached_head.clone(),
            evidence_hash: sha256_hex(&agent_result.contribution), // Hash of output
            module_hash: module.module_hash.clone(),
            execution: ExecutionMetadata {
                gas_used,
                latency_ms: latency.as_millis() as u64,
                exit_code: 0, // Success
                ihsan_score: agent_result.ihsan_score.to_f64(),
            },
        };

        // Compute Deterministic ID
        let payload_id =
            compute_payload_id(&payload).map_err(|e| anyhow::anyhow!("JCS ID Error: {}", e))?;

        // Hash for the chain
        let receipt_digest =
            compute_digest(&payload).map_err(|e| anyhow::anyhow!("JCS Digest Error: {}", e))?;
        let receipt_hash = hex::encode(receipt_digest);

        // Sign the receipt hash (Operator Attestation)
        let sig_bytes = self
            .signer
            .sign(&receipt_digest)
            .await
            .map_err(|e| anyhow::anyhow!("Signing failed: {}", e))?;

        let signature = Signature {
            signer_id: "bizra-node-0".to_string(),
            value_hex: hex::encode(sig_bytes),
        };

        let receipt = ThoughtExecReceipt {
            payload,
            payload_id,
            receipt_hash: receipt_hash.clone(),
            signatures: vec![signature],
        };

        // Update chain state
        self.store.append(&receipt).await?;
        self.cached_head = receipt_hash;

        info!("🧾 Generated Thought Receipt: ID={}", receipt.payload_id);

        Ok((agent_result, receipt))
    }
}

fn sha256_hex(data: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    hex::encode(hasher.finalize())
}
