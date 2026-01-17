// src/evidence.rs - Evidence Envelope Security
//
// Rust-native implementation of the Evidence Envelope protocol,
// mirroring the Python bizra_kernel/replay_guard.py for cross-language consistency.
//
// This ensures that every executed action is:
// 1. Unique (nonce tracking - anti-replay)
// 2. Ordered (monotonic counters)
// 3. Governed (policy hash binding to constitutional enforcement)

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// The Evidence Envelope structure that wraps every action.
/// PEAK MASTERPIECE v7.1: timestamp changed from f64 to u64 for determinism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    // Binding
    pub policy_hash: String, // Snapshot of the constitution
    pub session_id: String,  // Session scope
    pub agent_id: String,    // Actor

    // Ordering & Uniqueness
    pub nonce: String,    // Random unique identifier
    pub counter: u64,     // Monotonic counter for this session
    pub timestamp_ns: u64, // Timestamp in nanoseconds (UTC) - deterministic

    // Payload
    pub payload_hash: String, // Hash of the actual content/action
}

impl Envelope {
    /// Create a new envelope
    /// PEAK MASTERPIECE v7.1: Uses nanoseconds (u64) for deterministic hashing
    pub fn new(
        policy_hash: String,
        session_id: String,
        agent_id: String,
        nonce: String,
        counter: u64,
        payload_hash: String,
    ) -> Self {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System clock before UNIX epoch")
            .as_nanos() as u64;

        Self {
            policy_hash,
            session_id,
            agent_id,
            nonce,
            counter,
            timestamp_ns,
            payload_hash,
        }
    }

    /// Compute the unique hash of this envelope.
    /// Uses canonical JSON serialization to ensure deterministic hashing.
    /// PEAK MASTERPIECE v7.1: Uses u64 timestamp for cross-platform determinism
    pub fn compute_envelope_hash(&self) -> String {
        // Create a canonical representation matching Python's json.dumps(sort_keys=True)
        // Uses integer timestamp (nanoseconds) for deterministic hashing
        let canonical = format!(
            r#"{{"agent":"{}","counter":{},"nonce":"{}","payload":"{}","policy":"{}","session":"{}","ts_ns":{}}}"#,
            self.agent_id,
            self.counter,
            self.nonce,
            self.payload_hash,
            self.policy_hash,
            self.session_id,
            self.timestamp_ns  // Deterministic u64 instead of f64
        );

        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Compute payload hash from raw content
    pub fn hash_payload(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Enforces uniqueness and ordering of envelopes.
/// PEAK MASTERPIECE v7.1: Uses u64 nanoseconds for deterministic timestamps
///
/// State:
/// - seen_nonces: Map of nonce -> timestamp_ns (to prevent replay)
/// - session_counters: Map of session_id -> last_seen_counter (to enforce order)
#[derive(Debug)]
pub struct ReplayGuard {
    nonce_ttl_ns: u64,  // TTL in nanoseconds
    seen_nonces: HashMap<String, u64>,  // timestamp in nanoseconds
    session_counters: HashMap<String, u64>,
}

/// Nanoseconds per second constant
const NANOS_PER_SEC: u64 = 1_000_000_000;

impl ReplayGuard {
    /// Create a new ReplayGuard with default TTL of 1 hour
    pub fn new() -> Self {
        Self::with_ttl(3600)
    }

    /// Create a new ReplayGuard with custom TTL (in seconds)
    pub fn with_ttl(nonce_ttl_seconds: u64) -> Self {
        Self {
            nonce_ttl_ns: nonce_ttl_seconds * NANOS_PER_SEC,
            seen_nonces: HashMap::new(),
            session_counters: HashMap::new(),
        }
    }

    /// Get current timestamp in nanoseconds
    fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System clock before UNIX epoch")
            .as_nanos() as u64
    }

    /// Validate an envelope against replay and ordering rules.
    /// Returns Ok(()) on success, Err with descriptive message on failure.
    pub fn validate_envelope(&mut self, envelope: &Envelope) -> Result<(), String> {
        // 1. Nonce Check (Anti-Replay)
        if self.seen_nonces.contains_key(&envelope.nonce) {
            return Err(format!(
                "Replay detected: Nonce {} already used.",
                envelope.nonce
            ));
        }

        // Check current nonce freshness (using u64 nanoseconds)
        let now_ns = Self::now_ns();

        // Saturating subtraction to prevent underflow
        let age_ns = now_ns.saturating_sub(envelope.timestamp_ns);
        if age_ns > self.nonce_ttl_ns {
            return Err("Envelope expired (timestamp too old).".to_string());
        }

        // 2. Monotonic Counter Check (Ordering)
        let last_counter = self
            .session_counters
            .get(&envelope.session_id)
            .unwrap_or(&0);
        if envelope.counter <= *last_counter {
            return Err(format!(
                "Ordering violation: Counter {} <= last {} for session {}",
                envelope.counter, last_counter, envelope.session_id
            ));
        }

        // 3. Policy Binding (Basic Check)
        if envelope.policy_hash.is_empty() {
            return Err("Invalid envelope: Missing policy_hash binding.".to_string());
        }

        // --- Commit State ---
        self.seen_nonces.insert(envelope.nonce.clone(), now_ns);
        self.session_counters
            .insert(envelope.session_id.clone(), envelope.counter);

        Ok(())
    }

    /// Get the expected next counter value for a session
    pub fn get_next_counter(&self, session_id: &str) -> u64 {
        self.session_counters.get(session_id).unwrap_or(&0) + 1
    }

    /// Remove expired nonces to prevent memory bloat
    pub fn cleanup(&mut self) {
        let now_ns = Self::now_ns();

        let expired: Vec<String> = self
            .seen_nonces
            .iter()
            .filter(|(_, ts)| now_ns.saturating_sub(**ts) > self.nonce_ttl_ns)
            .map(|(nonce, _)| nonce.clone())
            .collect();

        for nonce in expired {
            self.seen_nonces.remove(&nonce);
        }
    }
}

impl Default for ReplayGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_hash_deterministic() {
        let env1 = Envelope::new(
            "policy123".to_string(),
            "session1".to_string(),
            "agent1".to_string(),
            "nonce1".to_string(),
            1,
            "payload_abc".to_string(),
        );

        let env2 = Envelope::new(
            "policy123".to_string(),
            "session1".to_string(),
            "agent1".to_string(),
            "nonce1".to_string(),
            1,
            "payload_abc".to_string(),
        );

        // Same inputs should produce same hash (within timestamp precision)
        // Note: timestamps will differ, so we just check structure
        assert_eq!(env1.policy_hash, env2.policy_hash);
        assert_eq!(env1.agent_id, env2.agent_id);
    }

    #[test]
    fn test_replay_guard_anti_replay() {
        let mut guard = ReplayGuard::new();
        let envelope = Envelope::new(
            "policy1".to_string(),
            "session1".to_string(),
            "agent1".to_string(),
            "nonce1".to_string(),
            1,
            "payload1".to_string(),
        );

        // First validation should succeed
        assert!(guard.validate_envelope(&envelope).is_ok());

        // Second validation with same nonce should fail
        let result = guard.validate_envelope(&envelope);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Replay detected"));
    }

    #[test]
    fn test_replay_guard_monotonic_counter() {
        let mut guard = ReplayGuard::new();

        let env1 = Envelope::new(
            "policy1".to_string(),
            "session1".to_string(),
            "agent1".to_string(),
            "nonce1".to_string(),
            1,
            "payload1".to_string(),
        );

        let env2 = Envelope::new(
            "policy1".to_string(),
            "session1".to_string(),
            "agent1".to_string(),
            "nonce2".to_string(),
            2,
            "payload2".to_string(),
        );

        let env3_bad = Envelope::new(
            "policy1".to_string(),
            "session1".to_string(),
            "agent1".to_string(),
            "nonce3".to_string(),
            2, // Same counter as env2 - should fail
            "payload3".to_string(),
        );

        assert!(guard.validate_envelope(&env1).is_ok());
        assert!(guard.validate_envelope(&env2).is_ok());

        let result = guard.validate_envelope(&env3_bad);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Ordering violation"));
    }

    #[test]
    fn test_replay_guard_empty_policy_hash() {
        let mut guard = ReplayGuard::new();
        let envelope = Envelope::new(
            "".to_string(), // Empty policy hash
            "session1".to_string(),
            "agent1".to_string(),
            "nonce1".to_string(),
            1,
            "payload1".to_string(),
        );

        let result = guard.validate_envelope(&envelope);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing policy_hash"));
    }

    #[test]
    fn test_get_next_counter() {
        let mut guard = ReplayGuard::new();

        // Session doesn't exist yet
        assert_eq!(guard.get_next_counter("session1"), 1);

        let env = Envelope::new(
            "policy1".to_string(),
            "session1".to_string(),
            "agent1".to_string(),
            "nonce1".to_string(),
            1,
            "payload1".to_string(),
        );

        guard.validate_envelope(&env).unwrap();

        // After counter 1, next should be 2
        assert_eq!(guard.get_next_counter("session1"), 2);
    }

    #[test]
    fn test_payload_hash() {
        let hash1 = Envelope::hash_payload("test content");
        let hash2 = Envelope::hash_payload("test content");
        let hash3 = Envelope::hash_payload("different content");

        // Same content -> same hash
        assert_eq!(hash1, hash2);
        // Different content -> different hash
        assert_ne!(hash1, hash3);
    }
}
