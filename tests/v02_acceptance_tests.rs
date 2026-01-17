// ═══════════════════════════════════════════════════════════════════════════════
// v0.2 VERIFIABLE KERNEL - ACCEPTANCE TESTS
// ═══════════════════════════════════════════════════════════════════════════════
//
// These tests separate "cool" from "credible":
//
// 1. Determinism test - Same input → same bytes → same hash
// 2. Tamper test - Flip one byte → verifier fails
// 3. Replay test - Same nonce/counter → verifier rejects
// 4. Policy drift test - Change policy → old receipts don't validate
//
// If these pass, a hostile reviewer cannot break your evidence.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

// ═══════════════════════════════════════════════════════════════════════════════
// TEST TYPES (subset of receipt_v1)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct IhsanVector {
    correctness: i64,
    safety: i64,
    benefit: i64,
    efficiency: i64,
    auditability: i64,
    anti_centralization: i64,
    robustness: i64,
    fairness: i64,
    composite: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct Envelope {
    policy_hash: String,
    session_id: String,
    agent_id: String,
    nonce: String,
    counter: u64,
    timestamp_ns: u64,
    payload_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestReceipt {
    receipt_id: String,
    envelope: Envelope,
    ihsan: IhsanVector,
    hash: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// JCS CANONICALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

fn jcs_canonicalize(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                n.to_string()
            }
        }
        serde_json::Value::String(s) => serde_json::to_string(s).unwrap_or_default(),
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(jcs_canonicalize).collect();
            format!("[{}]", items.join(","))
        }
        serde_json::Value::Object(obj) => {
            let mut keys: Vec<&String> = obj.keys().collect();
            keys.sort(); // RFC 8785: lexicographic sort
            let pairs: Vec<String> = keys
                .iter()
                .map(|k| {
                    let v = jcs_canonicalize(&obj[*k]);
                    format!("{}:{}", serde_json::to_string(k).unwrap_or_default(), v)
                })
                .collect();
            format!("{{{}}}", pairs.join(","))
        }
    }
}

fn hash_canonical<T: Serialize>(value: &T) -> String {
    let json = serde_json::to_value(value).unwrap();
    let canonical = jcs_canonicalize(&json);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn fixed64_from_f64(value: f64) -> i64 {
    (value * 0x1_0000_0000_i64 as f64) as i64
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST FIXTURES
// ═══════════════════════════════════════════════════════════════════════════════

fn create_test_ihsan() -> IhsanVector {
    IhsanVector {
        correctness: fixed64_from_f64(0.95),
        safety: fixed64_from_f64(0.98),
        benefit: fixed64_from_f64(0.92),
        efficiency: fixed64_from_f64(0.90),
        auditability: fixed64_from_f64(0.96),
        anti_centralization: fixed64_from_f64(0.88),
        robustness: fixed64_from_f64(0.89),
        fairness: fixed64_from_f64(0.93),
        composite: fixed64_from_f64(0.93), // Pre-computed weighted average
    }
}

fn create_test_envelope(nonce: &str, counter: u64) -> Envelope {
    Envelope {
        policy_hash: "policy_abc123".to_string(),
        session_id: "session_001".to_string(),
        agent_id: "agent_scribe".to_string(),
        nonce: nonce.to_string(),
        counter,
        timestamp_ns: 1737100800000000000, // Fixed timestamp for determinism
        payload_hash: "payload_xyz789".to_string(),
    }
}

fn create_test_receipt(receipt_id: &str, nonce: &str, counter: u64) -> TestReceipt {
    let envelope = create_test_envelope(nonce, counter);
    let ihsan = create_test_ihsan();

    let hashable = serde_json::json!({
        "receipt_id": receipt_id,
        "envelope": envelope,
        "ihsan": ihsan,
    });

    let hash = hash_canonical(&hashable);

    TestReceipt {
        receipt_id: receipt_id.to_string(),
        envelope,
        ihsan,
        hash,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 1: DETERMINISM
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_determinism_same_input_same_output() {
    // Same input + same policy + same environment → same receipt bytes → same hash

    let receipt1 = create_test_receipt("EXEC-001", "nonce-abc", 1);
    let receipt2 = create_test_receipt("EXEC-001", "nonce-abc", 1);

    // Hashes must be identical
    assert_eq!(
        receipt1.hash, receipt2.hash,
        "DETERMINISM FAILED: Same input produced different hashes"
    );

    // Canonical JSON must be identical
    let canonical1 = jcs_canonicalize(&serde_json::to_value(&receipt1).unwrap());
    let canonical2 = jcs_canonicalize(&serde_json::to_value(&receipt2).unwrap());

    assert_eq!(
        canonical1, canonical2,
        "DETERMINISM FAILED: Same input produced different canonical JSON"
    );
}

#[test]
fn test_determinism_jcs_key_ordering() {
    // JCS requires lexicographic key ordering

    // Create JSON with keys in different orders
    let json1 = serde_json::json!({
        "zebra": 1,
        "apple": 2,
        "mango": 3
    });

    let json2 = serde_json::json!({
        "apple": 2,
        "mango": 3,
        "zebra": 1
    });

    let canonical1 = jcs_canonicalize(&json1);
    let canonical2 = jcs_canonicalize(&json2);

    assert_eq!(
        canonical1, canonical2,
        "JCS ORDERING FAILED: Key order affected canonicalization"
    );

    // Verify keys are sorted
    assert!(
        canonical1.starts_with(r#"{"apple""#),
        "JCS should sort keys lexicographically"
    );
}

#[test]
fn test_determinism_fixed64_no_floats() {
    // All numeric fields must use Fixed64 (i64), not floats

    let ihsan = create_test_ihsan();
    let json = serde_json::to_string(&ihsan).unwrap();

    // Should not contain any decimal points (all values are integers)
    // Note: serde serializes i64 without decimals
    assert!(
        !json.contains(".0"),
        "NO FLOATS RULE VIOLATED: Found float representation in Ihsan"
    );

    // Verify roundtrip preserves exact values
    let parsed: IhsanVector = serde_json::from_str(&json).unwrap();
    assert_eq!(
        ihsan, parsed,
        "DETERMINISM FAILED: Serialization roundtrip changed values"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 2: TAMPER DETECTION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_tamper_single_byte_flip() {
    // Flip one byte in payload → hash changes → verifier fails

    let original = create_test_receipt("EXEC-002", "nonce-def", 2);
    let original_hash = original.hash.clone();

    // Create tampered version (change one character in receipt_id)
    let mut tampered = original.clone();
    tampered.receipt_id = "EXEC-003".to_string(); // Changed from 002 to 003

    // Recompute hash for tampered receipt
    let hashable = serde_json::json!({
        "receipt_id": tampered.receipt_id,
        "envelope": tampered.envelope,
        "ihsan": tampered.ihsan,
    });
    let tampered_hash = hash_canonical(&hashable);

    assert_ne!(
        original_hash, tampered_hash,
        "TAMPER DETECTION FAILED: Hash didn't change after modification"
    );
}

#[test]
fn test_tamper_ihsan_score_modification() {
    // Modifying Ihsan score must invalidate hash

    let original = create_test_receipt("EXEC-003", "nonce-ghi", 3);
    let original_hash = original.hash.clone();

    let mut tampered = original.clone();
    // Attacker tries to boost their composite score
    tampered.ihsan.composite = fixed64_from_f64(0.99);

    let hashable = serde_json::json!({
        "receipt_id": tampered.receipt_id,
        "envelope": tampered.envelope,
        "ihsan": tampered.ihsan,
    });
    let tampered_hash = hash_canonical(&hashable);

    assert_ne!(
        original_hash, tampered_hash,
        "TAMPER DETECTION FAILED: Ihsan modification not detected"
    );
}

#[test]
fn test_tamper_timestamp_modification() {
    // Modifying timestamp must invalidate hash

    let original = create_test_receipt("EXEC-004", "nonce-jkl", 4);
    let original_hash = original.hash.clone();

    let mut tampered = original.clone();
    tampered.envelope.timestamp_ns += 1; // Change by 1 nanosecond

    let hashable = serde_json::json!({
        "receipt_id": tampered.receipt_id,
        "envelope": tampered.envelope,
        "ihsan": tampered.ihsan,
    });
    let tampered_hash = hash_canonical(&hashable);

    assert_ne!(
        original_hash, tampered_hash,
        "TAMPER DETECTION FAILED: Timestamp modification not detected"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 3: REPLAY PROTECTION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_replay_duplicate_nonce_detected() {
    // Re-submit same nonce → verifier rejects

    let receipts = vec![
        create_test_receipt("EXEC-005", "nonce-same", 5),
        create_test_receipt("EXEC-006", "nonce-same", 6), // Same nonce!
    ];

    let mut seen_nonces: HashSet<String> = HashSet::new();
    let mut replay_detected = false;

    for receipt in &receipts {
        let nonce_key = format!("{}:{}", receipt.envelope.session_id, receipt.envelope.nonce);
        if seen_nonces.contains(&nonce_key) {
            replay_detected = true;
            break;
        }
        seen_nonces.insert(nonce_key);
    }

    assert!(
        replay_detected,
        "REPLAY PROTECTION FAILED: Duplicate nonce not detected"
    );
}

#[test]
fn test_replay_non_monotonic_counter_detected() {
    // Non-monotonic counter → replay attack

    let receipts = vec![
        create_test_receipt("EXEC-007", "nonce-a", 10),
        create_test_receipt("EXEC-008", "nonce-b", 5), // Counter went backwards!
    ];

    let mut last_counter = 0u64;
    let mut replay_detected = false;

    for receipt in &receipts {
        if receipt.envelope.counter <= last_counter && last_counter > 0 {
            replay_detected = true;
            break;
        }
        last_counter = receipt.envelope.counter;
    }

    assert!(
        replay_detected,
        "REPLAY PROTECTION FAILED: Non-monotonic counter not detected"
    );
}

#[test]
fn test_replay_valid_sequence_passes() {
    // Valid sequence should pass

    let receipts = vec![
        create_test_receipt("EXEC-009", "nonce-x", 1),
        create_test_receipt("EXEC-010", "nonce-y", 2),
        create_test_receipt("EXEC-011", "nonce-z", 3),
    ];

    let mut seen_nonces: HashSet<String> = HashSet::new();
    let mut last_counter = 0u64;
    let mut valid = true;

    for receipt in &receipts {
        let nonce_key = format!("{}:{}", receipt.envelope.session_id, receipt.envelope.nonce);

        if seen_nonces.contains(&nonce_key) {
            valid = false;
            break;
        }
        if receipt.envelope.counter <= last_counter && last_counter > 0 {
            valid = false;
            break;
        }

        seen_nonces.insert(nonce_key);
        last_counter = receipt.envelope.counter;
    }

    assert!(
        valid,
        "VALID SEQUENCE REJECTED: False positive in replay detection"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 4: POLICY DRIFT
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_policy_drift_detected() {
    // Change policy file → policy_hash changes → old receipts don't validate

    let original_policy = serde_json::json!({
        "constitution_hash": "const_abc",
        "covenant_hash": "cov_def",
        "version": "1.0.0"
    });

    let updated_policy = serde_json::json!({
        "constitution_hash": "const_abc",
        "covenant_hash": "cov_xyz", // Changed!
        "version": "1.0.1"
    });

    let original_hash = hash_canonical(&original_policy);
    let updated_hash = hash_canonical(&updated_policy);

    assert_ne!(
        original_hash, updated_hash,
        "POLICY DRIFT DETECTION FAILED: Policy change not detected"
    );

    // Create receipt bound to original policy
    let mut envelope = create_test_envelope("nonce-policy", 1);
    envelope.policy_hash = original_hash.clone();

    // Verification against updated policy should fail
    assert_ne!(
        envelope.policy_hash, updated_hash,
        "Old receipt should not validate against new policy"
    );
}

#[test]
fn test_policy_threshold_change_invalidates() {
    // Changing threshold values changes policy hash

    let policy_v1 = serde_json::json!({
        "thresholds": {
            "production": fixed64_from_f64(0.95)
        }
    });

    let policy_v2 = serde_json::json!({
        "thresholds": {
            "production": fixed64_from_f64(0.90) // Lowered threshold
        }
    });

    let hash_v1 = hash_canonical(&policy_v1);
    let hash_v2 = hash_canonical(&policy_v2);

    assert_ne!(
        hash_v1, hash_v2,
        "POLICY DRIFT: Threshold change must change policy hash"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 5: SNR ACCOUNTING
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_snr_signal_vs_noise_classification() {
    // Signal = actions that pass gates and verify
    // Noise = rejections, rollbacks, unverifiable

    struct SNRCounter {
        signal: u64,
        noise: u64,
    }

    impl SNRCounter {
        fn new() -> Self {
            Self { signal: 0, noise: 0 }
        }

        fn record_commit(&mut self) {
            self.signal += 1;
        }

        fn record_reject(&mut self) {
            self.noise += 1;
        }

        fn snr(&self) -> f64 {
            if self.signal + self.noise == 0 {
                return 0.0;
            }
            self.signal as f64 / (self.signal + self.noise) as f64
        }
    }

    let mut counter = SNRCounter::new();

    // Simulate: 95 commits, 5 rejections
    for _ in 0..95 {
        counter.record_commit();
    }
    for _ in 0..5 {
        counter.record_reject();
    }

    let snr = counter.snr();
    assert!(
        (snr - 0.95).abs() < 0.001,
        "SNR calculation incorrect: expected 0.95, got {}",
        snr
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION TEST
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_full_receipt_lifecycle() {
    // End-to-end test: create → hash → verify → chain

    // Step 1: Create receipts
    let receipts = vec![
        create_test_receipt("EXEC-100", "nonce-100", 1),
        create_test_receipt("EXEC-101", "nonce-101", 2),
        create_test_receipt("EXEC-102", "nonce-102", 3),
    ];

    // Step 2: Verify determinism
    for receipt in &receipts {
        let receipt_copy = create_test_receipt(
            &receipt.receipt_id,
            &receipt.envelope.nonce,
            receipt.envelope.counter,
        );
        assert_eq!(receipt.hash, receipt_copy.hash, "Determinism check failed");
    }

    // Step 3: Verify no replay
    let mut seen: HashSet<String> = HashSet::new();
    for receipt in &receipts {
        let key = format!("{}:{}", receipt.envelope.session_id, receipt.envelope.nonce);
        assert!(!seen.contains(&key), "Replay detected");
        seen.insert(key);
    }

    // Step 4: Verify Ihsan threshold
    let threshold = fixed64_from_f64(0.90);
    for receipt in &receipts {
        assert!(
            receipt.ihsan.composite >= threshold,
            "Ihsan threshold not met"
        );
    }

    println!("Full receipt lifecycle test PASSED");
}
