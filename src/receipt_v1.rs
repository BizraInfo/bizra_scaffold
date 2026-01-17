// ═══════════════════════════════════════════════════════════════════════════════
// BIZRA RECEIPT SCHEMA v1 - Verifiable Kernel
// ═══════════════════════════════════════════════════════════════════════════════
//
// This module defines the FROZEN Receipt Schema v1 for the Verifiable Kernel.
//
// Design principles:
// 1. NO FLOATS - All numeric values use Fixed64 (Q32.32) or integers
// 2. JCS CANONICAL - All serialization uses RFC 8785 JSON Canonicalization
// 3. DETERMINISTIC - Same input → same bytes → same hash across all platforms
// 4. OFFLINE VERIFIABLE - Any third party can verify without network access
//
// Standing on giants:
// - RFC 8785 (JCS): https://www.rfc-editor.org/rfc/rfc8785
// - Ed25519: https://ed25519.cr.yp.to/
// - SHA-256 for hashing (FIPS 180-4)

use crate::fixed::Fixed64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ═══════════════════════════════════════════════════════════════════════════════
// FIXED-POINT IHSAN VECTOR (8 Dimensions)
// ═══════════════════════════════════════════════════════════════════════════════

/// 8-dimensional Ihsān score vector using Fixed64 (Q32.32) for determinism.
/// No floats. Ever.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct IhsanVector {
    /// 1. Correctness (ʿAdl) - Technical accuracy
    pub correctness: i64,
    /// 2. Safety (Amānah) - Risk mitigation
    pub safety: i64,
    /// 3. Benefit (Ihsān) - Positive user impact
    pub benefit: i64,
    /// 4. Efficiency (Hikmah) - Resource optimization
    pub efficiency: i64,
    /// 5. Auditability (Bayān) - Transparency
    pub auditability: i64,
    /// 6. Anti-centralization (Tawḥīd) - Distributed authority
    pub anti_centralization: i64,
    /// 7. Robustness (Ṣabr) - Fault tolerance
    pub robustness: i64,
    /// 8. Fairness (Mīzān) - Equitable treatment
    pub fairness: i64,
    /// Composite score (weighted average as Fixed64 bits)
    pub composite: i64,
}

impl IhsanVector {
    /// Scale factor for Fixed64: 2^32 = 4294967296
    pub const SCALE: i64 = 0x1_0000_0000;

    /// Convert a 0.0-1.0 f64 to Fixed64 bits (for initialization only)
    /// IMPORTANT: This is only for setup/config. Runtime uses integer math.
    pub fn from_f64(value: f64) -> i64 {
        (value * Self::SCALE as f64) as i64
    }

    /// Convert Fixed64 bits back to f64 (for display only)
    pub fn to_f64(bits: i64) -> f64 {
        bits as f64 / Self::SCALE as f64
    }

    /// Check if composite meets threshold (both as Fixed64 bits)
    pub fn meets_threshold(&self, threshold_bits: i64) -> bool {
        self.composite >= threshold_bits
    }

    /// Compute weighted composite from dimensions
    /// Weights (as Fixed64 bits):
    /// - correctness: 0.15 = 644245094
    /// - safety: 0.20 = 858993459
    /// - benefit: 0.15 = 644245094
    /// - efficiency: 0.10 = 429496730
    /// - auditability: 0.10 = 429496730
    /// - anti_centralization: 0.10 = 429496730
    /// - robustness: 0.10 = 429496730
    /// - fairness: 0.10 = 429496730
    /// Total: 1.0
    pub fn compute_composite(&mut self) {
        // Weights as Fixed64 bits (sum = SCALE)
        const W_CORRECTNESS: i64 = 644245094;      // 0.15
        const W_SAFETY: i64 = 858993459;           // 0.20
        const W_BENEFIT: i64 = 644245094;          // 0.15
        const W_EFFICIENCY: i64 = 429496730;       // 0.10
        const W_AUDITABILITY: i64 = 429496730;     // 0.10
        const W_ANTI_CENTRAL: i64 = 429496730;     // 0.10
        const W_ROBUSTNESS: i64 = 429496730;       // 0.10
        const W_FAIRNESS: i64 = 429496729;         // 0.10 (adjusted for sum=SCALE)

        // Fixed-point multiplication: (a * b) >> 32
        let sum =
            ((self.correctness as i128 * W_CORRECTNESS as i128) >> 32) +
            ((self.safety as i128 * W_SAFETY as i128) >> 32) +
            ((self.benefit as i128 * W_BENEFIT as i128) >> 32) +
            ((self.efficiency as i128 * W_EFFICIENCY as i128) >> 32) +
            ((self.auditability as i128 * W_AUDITABILITY as i128) >> 32) +
            ((self.anti_centralization as i128 * W_ANTI_CENTRAL as i128) >> 32) +
            ((self.robustness as i128 * W_ROBUSTNESS as i128) >> 32) +
            ((self.fairness as i128 * W_FAIRNESS as i128) >> 32);

        self.composite = sum as i64;
    }
}

impl Default for IhsanVector {
    fn default() -> Self {
        Self {
            correctness: 0,
            safety: 0,
            benefit: 0,
            efficiency: 0,
            auditability: 0,
            anti_centralization: 0,
            robustness: 0,
            fairness: 0,
            composite: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EVIDENCE ENVELOPE (Anti-Replay)
// ═══════════════════════════════════════════════════════════════════════════════

/// Evidence envelope that wraps every action with anti-replay protection.
/// All fields are integers or strings - no floats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Envelope {
    /// Policy hash binding (H(JCS(constitution + covenant + weights)))
    pub policy_hash: String,
    /// Session identifier
    pub session_id: String,
    /// Agent identifier (who executed)
    pub agent_id: String,
    /// Unique nonce (UUID or random hex)
    pub nonce: String,
    /// Monotonic counter for this session
    pub counter: u64,
    /// Timestamp in nanoseconds since UNIX epoch (deterministic)
    pub timestamp_ns: u64,
    /// Hash of the action payload
    pub payload_hash: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// HASH CHAIN
// ═══════════════════════════════════════════════════════════════════════════════

/// Hash chain linking receipts together
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HashChain {
    /// Previous receipt hash (hex)
    pub prev: String,
    /// Current receipt hash (hex) - computed after all other fields
    pub current: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// DECISION
// ═══════════════════════════════════════════════════════════════════════════════

/// Decision outcome
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum Decision {
    /// Action committed successfully
    Commit { action_hash: String },
    /// Action rejected
    Reject { reason: String, gate: String },
}

// ═══════════════════════════════════════════════════════════════════════════════
// RECEIPT v1 (FROZEN SCHEMA)
// ═══════════════════════════════════════════════════════════════════════════════

/// Receipt Schema v1 - FROZEN
///
/// This schema is designed to be:
/// 1. JCS-canonicalizable (RFC 8785)
/// 2. Deterministic (no floats, stable field order via BTreeMap in serde)
/// 3. Offline-verifiable (all data needed for verification is included)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptV1 {
    /// Schema version (always "1" for this schema)
    pub schema_version: String,

    /// Unique receipt identifier (e.g., "EXEC-20260117-000001" or "REJ-...")
    pub receipt_id: String,

    /// Receipt type
    pub receipt_type: String, // "execution" or "rejection"

    /// Thought identifier (unique per thought lifecycle)
    pub thought_id: String,

    /// Evidence envelope (anti-replay, policy binding)
    pub envelope: Envelope,

    /// 8-dimensional Ihsān scores (Fixed64 bits)
    pub ihsan: IhsanVector,

    /// Decision (commit or reject with reason)
    pub decision: Decision,

    /// Hash chain (prev + current)
    pub hash_chain: HashChain,

    /// Ed25519 signature over JCS-canonicalized receipt (minus signature field)
    pub signature: String,

    /// Optional: FATE/Z3 proof artifact ID (for v0.3+)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fate_proof_id: Option<String>,

    /// Optional: Shūrā Council (formerly SAT) consensus metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shura_consensus: Option<ShuraConsensus>,
}

/// Shūrā Council consensus metadata
/// (Renamed from SAT to avoid collision with "satisfiable")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShuraConsensus {
    /// Did consensus pass?
    pub passed: bool,
    /// Total weight achieved
    pub weight_achieved: i64, // Fixed64 bits
    /// Weight required (70% threshold)
    pub weight_required: i64, // Fixed64 bits
    /// Which validators approved
    pub approvals: Vec<String>,
    /// Which validators vetoed (Security, Formal, Ethics have VETO power)
    pub vetoes: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// JCS CANONICALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

/// JCS (RFC 8785) canonicalization for deterministic hashing.
///
/// Rules:
/// 1. Object keys sorted lexicographically (Unicode code points)
/// 2. No whitespace
/// 3. Numbers: no leading zeros, no trailing zeros after decimal
/// 4. Strings: minimal escaping
///
/// We use serde_json with a custom serializer config.
pub fn jcs_canonicalize<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    // serde_json already sorts keys when using BTreeMap
    // For JCS compliance, we serialize then re-parse to ensure canonical form
    let json = serde_json::to_string(value)?;

    // Parse and re-serialize to ensure canonical ordering
    let parsed: serde_json::Value = serde_json::from_str(&json)?;
    canonicalize_value(&parsed)
}

fn canonicalize_value(value: &serde_json::Value) -> Result<String, serde_json::Error> {
    match value {
        serde_json::Value::Null => Ok("null".to_string()),
        serde_json::Value::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),
        serde_json::Value::Number(n) => {
            // JCS: integers without decimal, floats with minimal representation
            if let Some(i) = n.as_i64() {
                Ok(i.to_string())
            } else if let Some(u) = n.as_u64() {
                Ok(u.to_string())
            } else if let Some(f) = n.as_f64() {
                // NOTE: We should NOT have floats in receipts!
                // This is a safety fallback only.
                Ok(format!("{}", f))
            } else {
                Ok(n.to_string())
            }
        }
        serde_json::Value::String(s) => {
            // JCS: minimal escaping
            Ok(serde_json::to_string(s)?)
        }
        serde_json::Value::Array(arr) => {
            let items: Result<Vec<String>, _> = arr.iter().map(canonicalize_value).collect();
            Ok(format!("[{}]", items?.join(",")))
        }
        serde_json::Value::Object(obj) => {
            // JCS: keys sorted by Unicode code points
            let mut keys: Vec<&String> = obj.keys().collect();
            keys.sort();

            let pairs: Result<Vec<String>, _> = keys
                .iter()
                .map(|k| {
                    let v = canonicalize_value(&obj[*k])?;
                    Ok(format!("{}:{}", serde_json::to_string(k)?, v))
                })
                .collect();

            Ok(format!("{{{}}}", pairs?.join(",")))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HASHING
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute SHA-256 hash of JCS-canonicalized data
pub fn hash_jcs<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let canonical = jcs_canonicalize(value)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute SHA-256 hash of raw bytes
pub fn hash_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

// ═══════════════════════════════════════════════════════════════════════════════
// POLICY HASH
// ═══════════════════════════════════════════════════════════════════════════════

/// Policy components for hashing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyComponents {
    /// Constitution hash
    pub constitution_hash: String,
    /// Covenant hash
    pub covenant_hash: String,
    /// Scoring weights (as Fixed64 bits)
    pub weights: IhsanWeights,
    /// Gate thresholds (as Fixed64 bits)
    pub thresholds: GateThresholds,
    /// Policy version
    pub version: String,
}

/// Ihsān scoring weights (Fixed64 bits)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IhsanWeights {
    pub correctness: i64,
    pub safety: i64,
    pub benefit: i64,
    pub efficiency: i64,
    pub auditability: i64,
    pub anti_centralization: i64,
    pub robustness: i64,
    pub fairness: i64,
}

/// Gate thresholds (Fixed64 bits)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateThresholds {
    /// Development threshold (0.80 as Fixed64)
    pub development: i64,
    /// CI threshold (0.90 as Fixed64)
    pub ci: i64,
    /// Production threshold (0.95 as Fixed64)
    pub production: i64,
}

impl PolicyComponents {
    /// Compute policy hash: H(JCS(self))
    pub fn compute_hash(&self) -> Result<String, serde_json::Error> {
        hash_jcs(self)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RECEIPT BUILDER
// ═══════════════════════════════════════════════════════════════════════════════

impl ReceiptV1 {
    /// Create a new receipt (without signature and hash_chain.current)
    pub fn new(
        receipt_id: String,
        receipt_type: String,
        thought_id: String,
        envelope: Envelope,
        ihsan: IhsanVector,
        decision: Decision,
        prev_hash: String,
    ) -> Self {
        Self {
            schema_version: "1".to_string(),
            receipt_id,
            receipt_type,
            thought_id,
            envelope,
            ihsan,
            decision,
            hash_chain: HashChain {
                prev: prev_hash,
                current: String::new(), // Computed later
            },
            signature: String::new(), // Computed later
            fate_proof_id: None,
            shura_consensus: None,
        }
    }

    /// Compute the receipt hash (for hash_chain.current)
    /// This hashes all fields except signature and hash_chain.current
    pub fn compute_hash(&self) -> Result<String, serde_json::Error> {
        // Create a copy without the fields we're computing
        let hashable = serde_json::json!({
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "receipt_type": self.receipt_type,
            "thought_id": self.thought_id,
            "envelope": self.envelope,
            "ihsan": self.ihsan,
            "decision": self.decision,
            "hash_chain": {
                "prev": self.hash_chain.prev
            },
            "fate_proof_id": self.fate_proof_id,
            "shura_consensus": self.shura_consensus,
        });

        hash_jcs(&hashable)
    }

    /// Finalize the receipt by computing hash and adding signature
    pub fn finalize(&mut self, sign_fn: impl Fn(&[u8]) -> String) -> Result<(), serde_json::Error> {
        // Compute hash
        self.hash_chain.current = self.compute_hash()?;

        // Sign the canonical form
        let canonical = jcs_canonicalize(self)?;
        self.signature = sign_fn(canonical.as_bytes());

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jcs_canonicalization_deterministic() {
        let envelope = Envelope {
            policy_hash: "abc123".to_string(),
            session_id: "sess-001".to_string(),
            agent_id: "agent-scribe".to_string(),
            nonce: "nonce-xyz".to_string(),
            counter: 42,
            timestamp_ns: 1737100800000000000,
            payload_hash: "payload123".to_string(),
        };

        let canonical1 = jcs_canonicalize(&envelope).unwrap();
        let canonical2 = jcs_canonicalize(&envelope).unwrap();

        assert_eq!(canonical1, canonical2, "JCS must be deterministic");
    }

    #[test]
    fn test_ihsan_fixed_point_no_floats() {
        let mut ihsan = IhsanVector {
            correctness: IhsanVector::from_f64(0.95),
            safety: IhsanVector::from_f64(0.98),
            benefit: IhsanVector::from_f64(0.92),
            efficiency: IhsanVector::from_f64(0.90),
            auditability: IhsanVector::from_f64(0.96),
            anti_centralization: IhsanVector::from_f64(0.88),
            robustness: IhsanVector::from_f64(0.89),
            fairness: IhsanVector::from_f64(0.93),
            composite: 0,
        };

        ihsan.compute_composite();

        // Composite should be around 0.93 (weighted average)
        let composite_f64 = IhsanVector::to_f64(ihsan.composite);
        assert!(composite_f64 > 0.90 && composite_f64 < 0.96,
            "Composite {} should be ~0.93", composite_f64);
    }

    #[test]
    fn test_hash_determinism() {
        let envelope = Envelope {
            policy_hash: "test".to_string(),
            session_id: "s".to_string(),
            agent_id: "a".to_string(),
            nonce: "n".to_string(),
            counter: 1,
            timestamp_ns: 1000,
            payload_hash: "p".to_string(),
        };

        let hash1 = hash_jcs(&envelope).unwrap();
        let hash2 = hash_jcs(&envelope).unwrap();

        assert_eq!(hash1, hash2, "Hash must be deterministic");
    }

    #[test]
    fn test_threshold_check() {
        let threshold_95 = IhsanVector::from_f64(0.95);

        let mut passing = IhsanVector::default();
        passing.composite = IhsanVector::from_f64(0.96);
        assert!(passing.meets_threshold(threshold_95));

        let mut failing = IhsanVector::default();
        failing.composite = IhsanVector::from_f64(0.94);
        assert!(!failing.meets_threshold(threshold_95));
    }
}
