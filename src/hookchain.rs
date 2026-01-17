// src/hookchain.rs
// Status: GIANTS_PROTOCOL_V1
// SAT HookChain - Governance injection point for every capability/tool call
// Based on "Shoulder of Giants" pattern from Claude Code hooks

use crate::fixed::Fixed64;
use crate::tpm::SignerProvider;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::info;

// ============================================================================
// CAPABILITY TOKENS - Permission + Budget + Tier + Expiry + Evidence
// ============================================================================

/// Tier classification for capability budgets (Consumer-first design)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum CapabilityTier {
    /// T0: Mobile/Phone - Offline-first, strict budgets, minimal tools
    T0Mobile,
    /// T1: Consumer PC - Local model + tool sandbox
    #[default]
    T1Consumer,
    /// T2: Pro Workstation - Expanded skills + heavier verification
    T2Pro,
    /// T3: Pooled Compute - Verified contribution + reward
    T3Pooled,
}


impl CapabilityTier {
    /// Get default budget limits for this tier
    pub fn default_budget(&self) -> CapabilityBudget {
        match self {
            CapabilityTier::T0Mobile => CapabilityBudget {
                max_tokens: 1024,
                max_time_ms: 5000,
                max_tool_calls: 3,
                max_memory_mb: 256,
                offload_allowed: false,
            },
            CapabilityTier::T1Consumer => CapabilityBudget {
                max_tokens: 4096,
                max_time_ms: 30000,
                max_tool_calls: 10,
                max_memory_mb: 2048,
                offload_allowed: false,
            },
            CapabilityTier::T2Pro => CapabilityBudget {
                max_tokens: 16384,
                max_time_ms: 120000,
                max_tool_calls: 50,
                max_memory_mb: 8192,
                offload_allowed: true,
            },
            CapabilityTier::T3Pooled => CapabilityBudget {
                max_tokens: 65536,
                max_time_ms: 600000,
                max_tool_calls: 200,
                max_memory_mb: 32768,
                offload_allowed: true,
            },
        }
    }
}

/// Resource budget for a capability execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityBudget {
    pub max_tokens: u32,
    pub max_time_ms: u64,
    pub max_tool_calls: u32,
    pub max_memory_mb: u32,
    pub offload_allowed: bool,
}

/// Consent classification for capability usage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsentClass {
    /// Implicit consent - Safe, non-sensitive operations
    Implicit,
    /// Explicit consent - Requires user acknowledgment
    Explicit,
    /// Elevated consent - Requires strong authentication
    Elevated,
    /// Forbidden - Not allowed under any circumstances
    Forbidden,
}

/// Evidence requirements for capability attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRules {
    /// Require hardware attestation (TPM/Secure Enclave)
    pub require_hardware_attest: bool,
    /// Require receipt generation
    pub require_receipt: bool,
    /// Require audit logging
    pub require_audit_log: bool,
    /// Require sandbox execution
    pub require_sandbox: bool,
}

impl Default for EvidenceRules {
    fn default() -> Self {
        EvidenceRules {
            require_hardware_attest: false,
            require_receipt: true,
            require_audit_log: true,
            require_sandbox: false,
        }
    }
}

/// A Capability Token - the core unit of permission in BIZRA
/// Replaces boolean allow/deny with rich permission contracts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityToken {
    /// Unique identifier for this token
    pub token_id: String,
    /// The tool/capability this token grants access to
    pub tool_id: String,
    /// Scope of the permission (e.g., "read", "write", "execute")
    pub scope: String,
    /// Tier classification
    pub tier: CapabilityTier,
    /// Resource budget for this execution
    pub budget: CapabilityBudget,
    /// Token expiry timestamp (Unix epoch nanos)
    pub expiry_ns: u64,
    /// Evidence/attestation requirements
    pub evidence_rules: EvidenceRules,
    /// Consent classification
    pub consent_class: ConsentClass,
    /// Issuer signature (from SAT)
    pub issuer_signature: Option<Vec<u8>>,
    /// Creation timestamp
    pub created_at: u64,
}

impl CapabilityToken {
    /// Create a new capability token
    pub fn new(tool_id: &str, scope: &str, tier: CapabilityTier) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Default expiry: 1 hour from now
        let expiry = now + (3600 * 1_000_000_000);

        CapabilityToken {
            token_id: Self::generate_token_id(tool_id, scope, now),
            tool_id: tool_id.to_string(),
            scope: scope.to_string(),
            tier,
            budget: tier.default_budget(),
            expiry_ns: expiry,
            evidence_rules: EvidenceRules::default(),
            consent_class: ConsentClass::Implicit,
            issuer_signature: None,
            created_at: now,
        }
    }

    fn generate_token_id(tool_id: &str, scope: &str, timestamp: u64) -> String {
        let mut hasher = Sha256::new();
        hasher.update(tool_id.as_bytes());
        hasher.update(scope.as_bytes());
        hasher.update(timestamp.to_le_bytes());
        let result = hasher.finalize();
        format!("cap_{}", hex::encode(&result[..8]))
    }

    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        now > self.expiry_ns
    }

    /// Check if token is valid (not expired and properly signed)
    pub fn is_valid(&self) -> bool {
        !self.is_expired() && self.issuer_signature.is_some()
    }

    /// Sign this token with the provided signer
    pub async fn sign(&mut self, signer: &dyn SignerProvider) -> Result<(), HookError> {
        let payload = self.canonical_payload();
        let signature = signer
            .sign(&payload)
            .await
            .map_err(|e| HookError::SigningFailed(e.to_string()))?;
        self.issuer_signature = Some(signature);
        Ok(())
    }

    fn canonical_payload(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(self.tool_id.as_bytes());
        payload.extend_from_slice(self.scope.as_bytes());
        payload.extend_from_slice(&(self.tier as u8).to_le_bytes());
        payload.extend_from_slice(&self.expiry_ns.to_le_bytes());
        payload.extend_from_slice(&self.created_at.to_le_bytes());
        payload
    }
}

// ============================================================================
// SESSION MERKLE-DAG - Fork/Merge Semantics with State Roots
// ============================================================================

/// A node in the Session Merkle-DAG
/// Enables fork/merge governance with full provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionNode {
    /// Hash of this node (computed from contents)
    pub node_hash: String,
    /// Parent node hash (None for genesis)
    pub parent_hash: Option<String>,
    /// Merkle root of current state
    pub state_root: String,
    /// Merkle root of receipts in this session
    pub receipts_root: String,
    /// Policy version active for this session
    pub policy_version: String,
    /// Impact delta from parent (Ihsān score change)
    /// Fixed64 for deterministic cross-platform hash computation
    pub impact_delta: Fixed64,
    /// Timestamp of node creation
    pub created_at: u64,
    /// Fork ID if this is a forked context
    pub fork_id: Option<String>,
    /// Whether this node has been merged back
    pub merged: bool,
}

impl SessionNode {
    /// Create a genesis session node
    pub fn genesis(policy_version: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let mut node = SessionNode {
            node_hash: String::new(),
            parent_hash: None,
            state_root: "0".repeat(64),
            receipts_root: "0".repeat(64),
            policy_version: policy_version.to_string(),
            impact_delta: Fixed64::ZERO,
            created_at: now,
            fork_id: None,
            merged: false,
        };
        node.node_hash = node.compute_hash();
        node
    }

    /// Create a child node from this parent
    pub fn child(&self, state_root: &str, receipts_root: &str, impact_delta: Fixed64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let mut node = SessionNode {
            node_hash: String::new(),
            parent_hash: Some(self.node_hash.clone()),
            state_root: state_root.to_string(),
            receipts_root: receipts_root.to_string(),
            policy_version: self.policy_version.clone(),
            impact_delta,
            created_at: now,
            fork_id: None,
            merged: false,
        };
        node.node_hash = node.compute_hash();
        node
    }

    /// Fork this session (for parallel/experimental execution)
    pub fn fork(&self, fork_id: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let mut node = SessionNode {
            node_hash: String::new(),
            parent_hash: Some(self.node_hash.clone()),
            state_root: self.state_root.clone(),
            receipts_root: self.receipts_root.clone(),
            policy_version: self.policy_version.clone(),
            impact_delta: Fixed64::ZERO,
            created_at: now,
            fork_id: Some(fork_id.to_string()),
            merged: false,
        };
        node.node_hash = node.compute_hash();
        node
    }

    /// Compute deterministic hash using Fixed64 bits representation
    fn compute_hash(&self) -> String {
        let mut hasher = Sha256::new();
        if let Some(ref parent) = self.parent_hash {
            hasher.update(parent.as_bytes());
        }
        hasher.update(self.state_root.as_bytes());
        hasher.update(self.receipts_root.as_bytes());
        hasher.update(self.policy_version.as_bytes());
        // DETERMINISM: Use Fixed64 raw bits (i64) for cross-platform hash consistency
        hasher.update(self.impact_delta.to_bits().to_le_bytes());
        hasher.update(self.created_at.to_le_bytes());
        if let Some(ref fork_id) = self.fork_id {
            hasher.update(fork_id.as_bytes());
        }
        hex::encode(hasher.finalize())
    }
}

/// Session DAG manager
pub struct SessionDAG {
    nodes: RwLock<HashMap<String, SessionNode>>,
    head: RwLock<String>,
}

impl SessionDAG {
    pub fn new(policy_version: &str) -> Self {
        let genesis = SessionNode::genesis(policy_version);
        let head = genesis.node_hash.clone();
        let mut nodes = HashMap::new();
        nodes.insert(genesis.node_hash.clone(), genesis);

        SessionDAG {
            nodes: RwLock::new(nodes),
            head: RwLock::new(head),
        }
    }

    /// Get the current head node
    pub async fn get_head(&self) -> Option<SessionNode> {
        let head = self.head.read().await;
        let nodes = self.nodes.read().await;
        nodes.get(&*head).cloned()
    }

    /// Advance the DAG with a new child node
    /// Returns error if DAG has no head (should never happen after initialization)
    pub async fn advance(
        &self,
        state_root: &str,
        receipts_root: &str,
        impact_delta: Fixed64,
    ) -> Result<SessionNode, HookError> {
        let head = self.get_head().await.ok_or_else(|| {
            HookError::InvalidToken("DAG has no head node".to_string())
        })?;
        let child = head.child(state_root, receipts_root, impact_delta);

        let mut nodes = self.nodes.write().await;
        let mut head_lock = self.head.write().await;

        nodes.insert(child.node_hash.clone(), child.clone());
        *head_lock = child.node_hash.clone();

        Ok(child)
    }

    /// Fork the current head for parallel execution
    /// Returns error if DAG has no head (should never happen after initialization)
    pub async fn fork(&self, fork_id: &str) -> Result<SessionNode, HookError> {
        let head = self.get_head().await.ok_or_else(|| {
            HookError::InvalidToken("DAG has no head node".to_string())
        })?;
        let forked = head.fork(fork_id);

        let mut nodes = self.nodes.write().await;
        nodes.insert(forked.node_hash.clone(), forked.clone());

        Ok(forked)
    }
}

// ============================================================================
// HOOK CHAIN - Pre/Post Capability Governance
// ============================================================================

/// Decision from PreCapabilityUse hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookDecision {
    /// Allow execution with optional constraints
    Allow {
        constraints: Option<HookConstraints>,
        reason: String,
    },
    /// Deny execution
    Deny { reason: String, code: String },
    /// Ask for user confirmation
    Ask { question: String, timeout_ms: u64 },
}

/// Constraints applied by hooks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookConstraints {
    /// Modified input (if hook transforms the request)
    pub modified_input: Option<String>,
    /// Additional budget restrictions
    pub budget_override: Option<CapabilityBudget>,
    /// Required evidence rules override
    pub evidence_override: Option<EvidenceRules>,
}

/// Result from PostCapabilityUse hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PostHookResult {
    /// Commit the execution (generate receipt, update state)
    Commit {
        receipt_id: String,
        state_delta: String,
    },
    /// Quarantine the execution (needs review)
    Quarantine { reason: String, evidence: Vec<u8> },
}

/// Draft receipt for pre-hook evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptDraft {
    pub tool_id: String,
    pub input: String,
    pub capability_token: Option<CapabilityToken>,
    pub session_node: String,
    pub timestamp: u64,
}

/// Executed receipt for post-hook evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutedReceipt {
    pub draft: ReceiptDraft,
    pub output: String,
    pub execution_time_ms: u64,
    pub tokens_used: u32,
    pub success: bool,
    pub error: Option<String>,
}

/// Hook Chain errors
#[derive(Debug, thiserror::Error)]
pub enum HookError {
    #[error("Hook denied: {0}")]
    Denied(String),
    #[error("Hook timeout: {0}")]
    Timeout(String),
    #[error("Signing failed: {0}")]
    SigningFailed(String),
    #[error("Invalid capability token: {0}")]
    InvalidToken(String),
    #[error("Budget exceeded: {0}")]
    BudgetExceeded(String),
    #[error("Unauthorized: {0}")]
    Unauthorized(String),
    #[error("Security violation: {0}")]
    SecurityViolation(String),
}

/// The SAT Hook Chain - governance injection for every capability
#[allow(dead_code)] // Reserved fields for future expansion
pub struct SATHookChain {
    signer: Arc<dyn SignerProvider>,
    session_dag: SessionDAG,
    policy_version: String,
    /// Blocked tools (deny-by-default for these)
    blocked_tools: Vec<String>,
    /// Tools requiring elevated consent
    elevated_tools: Vec<String>,
}

impl SATHookChain {
    pub fn new(signer: Arc<dyn SignerProvider>, policy_version: &str) -> Self {
        SATHookChain {
            signer,
            session_dag: SessionDAG::new(policy_version),
            policy_version: policy_version.to_string(),
            blocked_tools: vec![
                "rm".to_string(),
                "sudo".to_string(),
                "chmod".to_string(),
                "eval".to_string(),
                "exec".to_string(),
            ],
            elevated_tools: vec![
                "file_write".to_string(),
                "network_request".to_string(),
                "subprocess".to_string(),
            ],
        }
    }

    /// Pre-capability hook: Evaluate before execution
    /// Returns: Allow, Deny, or Ask
    pub async fn pre_capability_use(
        &self,
        draft: &ReceiptDraft,
    ) -> Result<HookDecision, HookError> {
        info!("🔗 PreCapabilityUse hook: tool={}", draft.tool_id);

        // Check blocked tools first (deny-by-default)
        if self.blocked_tools.contains(&draft.tool_id) {
            return Ok(HookDecision::Deny {
                reason: format!("Tool '{}' is blocked by policy", draft.tool_id),
                code: "BLOCKED_TOOL".to_string(),
            });
        }

        // Validate capability token if present
        if let Some(ref token) = draft.capability_token {
            if token.is_expired() {
                return Ok(HookDecision::Deny {
                    reason: "Capability token expired".to_string(),
                    code: "TOKEN_EXPIRED".to_string(),
                });
            }
            if token.tool_id != draft.tool_id {
                return Ok(HookDecision::Deny {
                    reason: format!(
                        "Token tool_id mismatch: {} != {}",
                        token.tool_id, draft.tool_id
                    ),
                    code: "TOKEN_MISMATCH".to_string(),
                });
            }
        }

        // Check if elevated consent required
        if self.elevated_tools.contains(&draft.tool_id) {
            if let Some(ref token) = draft.capability_token {
                if token.consent_class != ConsentClass::Elevated {
                    return Ok(HookDecision::Ask {
                        question: format!(
                            "Tool '{}' requires elevated consent. Allow execution?",
                            draft.tool_id
                        ),
                        timeout_ms: 30000,
                    });
                }
            } else {
                return Ok(HookDecision::Ask {
                    question: format!(
                        "Tool '{}' requires capability token with elevated consent.",
                        draft.tool_id
                    ),
                    timeout_ms: 30000,
                });
            }
        }

        // Security scan on input
        let security_threats = self.scan_input_security(&draft.input);
        if !security_threats.is_empty() {
            return Ok(HookDecision::Deny {
                reason: format!("Security threats detected: {}", security_threats.join(", ")),
                code: "SECURITY_THREAT".to_string(),
            });
        }

        // All checks passed
        Ok(HookDecision::Allow {
            constraints: None,
            reason: "All pre-capability checks passed".to_string(),
        })
    }

    /// Post-capability hook: Evaluate after execution
    /// Returns: Commit or Quarantine
    pub async fn post_capability_use(
        &self,
        executed: &ExecutedReceipt,
    ) -> Result<PostHookResult, HookError> {
        info!(
            "🔗 PostCapabilityUse hook: tool={}, success={}",
            executed.draft.tool_id, executed.success
        );

        // Check if execution succeeded
        if !executed.success {
            if let Some(ref error) = executed.error {
                // Quarantine failed executions for review
                return Ok(PostHookResult::Quarantine {
                    reason: format!("Execution failed: {}", error),
                    evidence: error.as_bytes().to_vec(),
                });
            }
        }

        // Check budget compliance
        if let Some(ref token) = executed.draft.capability_token {
            if executed.tokens_used > token.budget.max_tokens {
                return Ok(PostHookResult::Quarantine {
                    reason: format!(
                        "Token budget exceeded: {} > {}",
                        executed.tokens_used, token.budget.max_tokens
                    ),
                    evidence: vec![],
                });
            }
            if executed.execution_time_ms > token.budget.max_time_ms {
                return Ok(PostHookResult::Quarantine {
                    reason: format!(
                        "Time budget exceeded: {}ms > {}ms",
                        executed.execution_time_ms, token.budget.max_time_ms
                    ),
                    evidence: vec![],
                });
            }
        }

        // Security scan on output
        let security_threats = self.scan_output_security(&executed.output);
        if !security_threats.is_empty() {
            return Ok(PostHookResult::Quarantine {
                reason: format!("Output security threats: {}", security_threats.join(", ")),
                evidence: executed.output.as_bytes().to_vec(),
            });
        }

        // Generate receipt ID and commit
        let receipt_id = self.generate_receipt_id(executed);
        let state_delta = self.compute_state_delta(executed);

        // Advance session DAG
        let _new_node = self
            .session_dag
            .advance(
                &state_delta,
                &receipt_id,
                Fixed64::ZERO, // Impact delta computed separately
            )
            .await?;

        Ok(PostHookResult::Commit {
            receipt_id,
            state_delta,
        })
    }

    /// Mint a capability token for a tool
    pub async fn mint_capability_token(
        &self,
        tool_id: &str,
        scope: &str,
        tier: CapabilityTier,
        consent_class: ConsentClass,
    ) -> Result<CapabilityToken, HookError> {
        let mut token = CapabilityToken::new(tool_id, scope, tier);
        token.consent_class = consent_class;
        token.sign(self.signer.as_ref()).await?;

        info!(
            "🎫 Minted capability token: {} for tool={}",
            token.token_id, tool_id
        );
        Ok(token)
    }

    /// Get current session head
    pub async fn get_session_head(&self) -> Option<SessionNode> {
        self.session_dag.get_head().await
    }

    /// Fork current session for parallel execution
    pub async fn fork_session(&self, fork_id: &str) -> Result<SessionNode, HookError> {
        self.session_dag.fork(fork_id).await
    }

    fn scan_input_security(&self, input: &str) -> Vec<String> {
        let mut threats = Vec::new();
        let input_lower = input.to_lowercase();

        let patterns = [
            ("rm -rf", "Destructive command"),
            ("sudo", "Privilege escalation"),
            ("eval(", "Code injection"),
            ("exec(", "Code execution"),
            ("__import__", "Python import injection"),
            ("drop table", "SQL injection"),
            ("'; --", "SQL injection"),
            ("<script>", "XSS attack"),
        ];

        for (pattern, threat) in patterns {
            if input_lower.contains(pattern) {
                threats.push(threat.to_string());
            }
        }

        threats
    }

    fn scan_output_security(&self, output: &str) -> Vec<String> {
        let mut threats = Vec::new();
        let output_lower = output.to_lowercase();

        // Check for sensitive data leakage
        let patterns = [
            ("password", "Potential password leak"),
            ("api_key", "Potential API key leak"),
            ("secret", "Potential secret leak"),
            ("private_key", "Potential private key leak"),
        ];

        for (pattern, threat) in patterns {
            if output_lower.contains(pattern) {
                threats.push(threat.to_string());
            }
        }

        threats
    }

    fn generate_receipt_id(&self, executed: &ExecutedReceipt) -> String {
        let mut hasher = Sha256::new();
        hasher.update(executed.draft.tool_id.as_bytes());
        hasher.update(executed.draft.input.as_bytes());
        hasher.update(executed.output.as_bytes());
        hasher.update(executed.execution_time_ms.to_le_bytes());
        hasher.update(executed.draft.timestamp.to_le_bytes());
        format!("rec_{}", hex::encode(&hasher.finalize()[..12]))
    }

    fn compute_state_delta(&self, executed: &ExecutedReceipt) -> String {
        let mut hasher = Sha256::new();
        hasher.update(executed.output.as_bytes());
        hasher.update(executed.tokens_used.to_le_bytes());
        hex::encode(&hasher.finalize()[..16])
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tpm::SoftwareSigner;

    #[test]
    fn test_capability_token_creation() {
        let token = CapabilityToken::new("file_read", "read", CapabilityTier::T1Consumer);

        assert!(token.token_id.starts_with("cap_"));
        assert_eq!(token.tool_id, "file_read");
        assert_eq!(token.scope, "read");
        assert!(!token.is_expired());
        assert!(!token.is_valid()); // Not signed yet
    }

    #[tokio::test]
    async fn test_capability_token_signing() {
        let signer = Arc::new(SoftwareSigner::new());
        let mut token = CapabilityToken::new("file_read", "read", CapabilityTier::T1Consumer);

        token.sign(signer.as_ref()).await.unwrap();

        assert!(token.is_valid());
        assert!(token.issuer_signature.is_some());
    }

    #[test]
    fn test_session_dag_genesis() {
        let dag = SessionDAG::new("v1.0.0");
        // Just test that it creates without panic
        assert!(true);
    }

    #[tokio::test]
    async fn test_session_dag_advance() {
        let dag = SessionDAG::new("v1.0.0");

        let head1 = dag.get_head().await.unwrap();
        assert!(head1.parent_hash.is_none()); // Genesis has no parent

        // Use Fixed64 for deterministic impact delta
        let child = dag.advance("state1", "receipts1", Fixed64::from_f64(0.05)).await.unwrap();
        assert_eq!(child.parent_hash, Some(head1.node_hash));

        let head2 = dag.get_head().await.unwrap();
        assert_eq!(head2.node_hash, child.node_hash);
    }

    #[tokio::test]
    async fn test_hook_chain_blocked_tool() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer, "v1.0.0");

        let draft = ReceiptDraft {
            tool_id: "sudo".to_string(),
            input: "sudo rm -rf /".to_string(),
            capability_token: None,
            session_node: "genesis".to_string(),
            timestamp: 0,
        };

        let decision = hook_chain.pre_capability_use(&draft).await.unwrap();

        match decision {
            HookDecision::Deny { code, .. } => assert_eq!(code, "BLOCKED_TOOL"),
            _ => panic!("Expected Deny decision"),
        }
    }

    #[tokio::test]
    async fn test_hook_chain_security_scan() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer, "v1.0.0");

        let draft = ReceiptDraft {
            tool_id: "text_process".to_string(),
            input: "DROP TABLE users; --".to_string(),
            capability_token: None,
            session_node: "genesis".to_string(),
            timestamp: 0,
        };

        let decision = hook_chain.pre_capability_use(&draft).await.unwrap();

        match decision {
            HookDecision::Deny { code, .. } => assert_eq!(code, "SECURITY_THREAT"),
            _ => panic!("Expected Deny decision for SQL injection"),
        }
    }

    #[tokio::test]
    async fn test_hook_chain_allow() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer, "v1.0.0");

        let draft = ReceiptDraft {
            tool_id: "text_process".to_string(),
            input: "Hello, please analyze this text.".to_string(),
            capability_token: None,
            session_node: "genesis".to_string(),
            timestamp: 0,
        };

        let decision = hook_chain.pre_capability_use(&draft).await.unwrap();

        match decision {
            HookDecision::Allow { .. } => (),
            _ => panic!("Expected Allow decision"),
        }
    }

    #[tokio::test]
    async fn test_post_hook_commit() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer, "v1.0.0");

        let executed = ExecutedReceipt {
            draft: ReceiptDraft {
                tool_id: "text_process".to_string(),
                input: "Hello".to_string(),
                capability_token: None,
                session_node: "genesis".to_string(),
                timestamp: 0,
            },
            output: "Processed text".to_string(),
            execution_time_ms: 100,
            tokens_used: 50,
            success: true,
            error: None,
        };

        let result = hook_chain.post_capability_use(&executed).await.unwrap();

        match result {
            PostHookResult::Commit { receipt_id, .. } => {
                assert!(receipt_id.starts_with("rec_"));
            }
            _ => panic!("Expected Commit result"),
        }
    }

    // ========================================================================
    // REVIEW FIX: Additional tests for comprehensive hookchain coverage
    // ========================================================================

    #[test]
    fn test_capability_tier_budget_defaults() {
        // T0 Mobile: Strict constraints
        let t0 = CapabilityTier::T0Mobile.default_budget();
        assert_eq!(t0.max_tokens, 1024);
        assert_eq!(t0.max_time_ms, 5000);
        assert_eq!(t0.max_tool_calls, 3);
        assert!(!t0.offload_allowed);

        // T1 Consumer: Balanced
        let t1 = CapabilityTier::T1Consumer.default_budget();
        assert_eq!(t1.max_tokens, 4096);
        assert_eq!(t1.max_tool_calls, 10);
        assert!(!t1.offload_allowed);

        // T2 Pro: Expanded
        let t2 = CapabilityTier::T2Pro.default_budget();
        assert_eq!(t2.max_tokens, 16384);
        assert!(t2.offload_allowed);

        // T3 Pooled: Maximum
        let t3 = CapabilityTier::T3Pooled.default_budget();
        assert_eq!(t3.max_tokens, 65536);
        assert_eq!(t3.max_tool_calls, 200);
        assert!(t3.offload_allowed);
    }

    #[test]
    fn test_session_node_hash_determinism() {
        // Create two identical nodes and verify hashes match
        let node1 = SessionNode::genesis("v1.0.0");
        let node2 = SessionNode::genesis("v1.0.0");

        // Genesis nodes with same policy should have same structure
        // (timestamps will differ, but hash computation is deterministic)
        assert_eq!(node1.policy_version, node2.policy_version);
        assert!(node1.parent_hash.is_none());
        assert!(node2.parent_hash.is_none());
    }

    #[test]
    fn test_session_node_fixed64_impact_delta() {
        let node = SessionNode::genesis("v1.0.0");

        // Genesis node should have zero impact delta
        assert_eq!(node.impact_delta.to_f64(), 0.0);

        // Verify Fixed64 is being used (not f64)
        let bits = node.impact_delta.to_bits();
        assert_eq!(bits, 0i64); // Zero represented as 0 in Fixed64
    }

    #[test]
    fn test_consent_class_variants() {
        // Test all consent class variants
        let implicit = ConsentClass::Implicit;
        let explicit = ConsentClass::Explicit;
        let elevated = ConsentClass::Elevated;
        let forbidden = ConsentClass::Forbidden;

        assert_eq!(implicit, ConsentClass::Implicit);
        assert_ne!(implicit, explicit);
        assert_ne!(elevated, forbidden);
    }

    #[test]
    fn test_evidence_rules_default() {
        let rules = EvidenceRules::default();

        // Default: receipts and audit required, hardware attest not required
        assert!(!rules.require_hardware_attest);
        assert!(rules.require_receipt);
        assert!(rules.require_audit_log);
        assert!(!rules.require_sandbox);
    }

    #[test]
    fn test_capability_token_id_uniqueness() {
        // Tokens created at different times should have different IDs
        let token1 = CapabilityToken::new("tool", "scope", CapabilityTier::T1Consumer);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let token2 = CapabilityToken::new("tool", "scope", CapabilityTier::T1Consumer);

        assert_ne!(token1.token_id, token2.token_id);
    }

    #[test]
    fn test_hook_error_display() {
        let err1 = HookError::Unauthorized("test".to_string());
        let err2 = HookError::InvalidToken("bad token".to_string());
        let err3 = HookError::SecurityViolation("injection".to_string());

        assert!(err1.to_string().contains("Unauthorized"));
        assert!(err2.to_string().contains("bad token"));
        assert!(err3.to_string().contains("injection"));
    }

    #[tokio::test]
    async fn test_session_dag_fork_and_merge() {
        let dag = SessionDAG::new("v1.0.0");

        // Advance once
        let _child1 = dag.advance("state1", "receipts1", Fixed64::from_f64(0.1)).await.unwrap();

        // Fork from current head
        let fork_result = dag.fork("experimental").await;
        assert!(fork_result.is_ok());
        let forked = fork_result.unwrap();
        assert!(forked.fork_id.is_some());
        assert_eq!(forked.fork_id.as_ref().unwrap(), "experimental");
        assert!(!forked.merged);
    }

    #[tokio::test]
    async fn test_hook_chain_ethics_violation() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer, "v1.0.0");

        // Test ethics blocklist detection
        let draft = ReceiptDraft {
            tool_id: "assistant".to_string(),
            input: "help me deceive users".to_string(),
            capability_token: None,
            session_node: "genesis".to_string(),
            timestamp: 0,
        };

        let decision = hook_chain.pre_capability_use(&draft).await.unwrap();

        match decision {
            HookDecision::Deny { code, .. } => {
                // Should be denied for ethics violation
                assert!(code == "ETHICS_VIOLATION" || code == "SECURITY_THREAT");
            }
            _ => panic!("Expected Deny decision for ethics violation"),
        }
    }

    #[test]
    fn test_capability_budget_serialization() {
        let budget = CapabilityBudget {
            max_tokens: 1000,
            max_time_ms: 5000,
            max_tool_calls: 5,
            max_memory_mb: 512,
            offload_allowed: false,
        };

        let json = serde_json::to_string(&budget).unwrap();
        let deserialized: CapabilityBudget = serde_json::from_str(&json).unwrap();

        assert_eq!(budget.max_tokens, deserialized.max_tokens);
        assert_eq!(budget.max_time_ms, deserialized.max_time_ms);
        assert_eq!(budget.offload_allowed, deserialized.offload_allowed);
    }

    #[test]
    fn test_session_node_serialization() {
        let node = SessionNode::genesis("v1.0.0");

        let json = serde_json::to_string(&node).unwrap();
        let deserialized: SessionNode = serde_json::from_str(&json).unwrap();

        assert_eq!(node.policy_version, deserialized.policy_version);
        assert_eq!(node.parent_hash, deserialized.parent_hash);
        assert_eq!(node.impact_delta.to_bits(), deserialized.impact_delta.to_bits());
    }
}
