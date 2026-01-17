// ═══════════════════════════════════════════════════════════════════════════════
// BIZRA VERIFIER CLI - Offline Receipt Validation
// ═══════════════════════════════════════════════════════════════════════════════
//
// A hostile reviewer can run this and say:
// "I don't trust you, but I can't break your evidence."
//
// Verification checks:
// 1. JCS canonicalization + hash chain integrity
// 2. Ed25519 signature verification
// 3. Policy hash matches local policy files
// 4. Replay constraints (nonce uniqueness, counter monotonicity)
// 5. Ihsān threshold compliance
//
// Usage:
//   bizra_verify receipt.json --policy policy.json
//   bizra_verify evidence/receipts/*.json --policy policy.json --chain
//   bizra_verify --help

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use clap::Parser;
use ed25519_dalek::{Signature, VerifyingKey, Verifier};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ═══════════════════════════════════════════════════════════════════════════════
// CLI ARGUMENTS
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Parser, Debug)]
#[command(name = "bizra_verify")]
#[command(about = "BIZRA Receipt Verifier - Offline validation of Third Fact receipts")]
#[command(version = "0.2.0")]
struct Args {
    /// Receipt file(s) to verify (JSON)
    #[arg(required = true)]
    receipts: Vec<PathBuf>,

    /// Policy file for policy_hash verification
    #[arg(short, long)]
    policy: Option<PathBuf>,

    /// Public key file (hex-encoded Ed25519 public key)
    #[arg(short = 'k', long)]
    pubkey: Option<PathBuf>,

    /// Verify hash chain continuity across multiple receipts
    #[arg(short, long)]
    chain: bool,

    /// Check replay constraints (nonce uniqueness)
    #[arg(short, long)]
    replay: bool,

    /// Persistent nonce journal file for stateful replay detection
    /// Survives process restarts - use for continuous monitoring
    #[arg(long)]
    nonce_journal: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    output: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// RECEIPT TYPES (matching receipt_v1.rs)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReceiptV1 {
    schema_version: String,
    receipt_id: String,
    receipt_type: String,
    thought_id: String,
    envelope: Envelope,
    ihsan: IhsanVector,
    decision: Decision,
    hash_chain: HashChain,
    signature: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    fate_proof_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shura_consensus: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HashChain {
    prev: String,
    current: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum Decision {
    Commit { action_hash: String },
    Reject { reason: String, gate: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PolicyComponents {
    constitution_hash: String,
    covenant_hash: String,
    weights: serde_json::Value,
    thresholds: serde_json::Value,
    version: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// VERIFICATION RESULT
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
struct VerificationResult {
    receipt_id: String,
    passed: bool,
    checks: Vec<CheckResult>,
}

#[derive(Debug, Serialize)]
struct CheckResult {
    check: String,
    passed: bool,
    message: String,
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
            keys.sort();
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

fn hash_jcs(value: &serde_json::Value) -> String {
    let canonical = jcs_canonicalize(value);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    format!("{:x}", hasher.finalize())
}

// ═══════════════════════════════════════════════════════════════════════════════
// VERIFICATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

fn verify_hash_chain(receipt: &ReceiptV1) -> CheckResult {
    // Reconstruct hashable content (without signature and current hash)
    let hashable = serde_json::json!({
        "schema_version": receipt.schema_version,
        "receipt_id": receipt.receipt_id,
        "receipt_type": receipt.receipt_type,
        "thought_id": receipt.thought_id,
        "envelope": receipt.envelope,
        "ihsan": receipt.ihsan,
        "decision": receipt.decision,
        "hash_chain": {
            "prev": receipt.hash_chain.prev
        },
        "fate_proof_id": receipt.fate_proof_id,
        "shura_consensus": receipt.shura_consensus,
    });

    let computed_hash = hash_jcs(&hashable);
    let matches = computed_hash == receipt.hash_chain.current;

    CheckResult {
        check: "hash_chain".to_string(),
        passed: matches,
        message: if matches {
            format!("Hash chain valid: {}", &receipt.hash_chain.current[..16])
        } else {
            format!(
                "Hash mismatch! Expected: {}, Got: {}",
                &receipt.hash_chain.current[..16],
                &computed_hash[..16]
            )
        },
    }
}

fn verify_signature(receipt: &ReceiptV1, pubkey_hex: Option<&str>) -> CheckResult {
    let Some(pubkey_hex) = pubkey_hex else {
        return CheckResult {
            check: "signature".to_string(),
            passed: true, // Skip if no pubkey provided
            message: "Signature check skipped (no public key provided)".to_string(),
        };
    };

    // Parse public key
    let pubkey_bytes = match hex::decode(pubkey_hex) {
        Ok(bytes) => bytes,
        Err(e) => {
            return CheckResult {
                check: "signature".to_string(),
                passed: false,
                message: format!("Invalid public key hex: {}", e),
            };
        }
    };

    let pubkey: [u8; 32] = match pubkey_bytes.try_into() {
        Ok(arr) => arr,
        Err(_) => {
            return CheckResult {
                check: "signature".to_string(),
                passed: false,
                message: "Public key must be 32 bytes".to_string(),
            };
        }
    };

    let verifying_key = match VerifyingKey::from_bytes(&pubkey) {
        Ok(vk) => vk,
        Err(e) => {
            return CheckResult {
                check: "signature".to_string(),
                passed: false,
                message: format!("Invalid Ed25519 public key: {}", e),
            };
        }
    };

    // Parse signature
    let sig_bytes = match hex::decode(&receipt.signature) {
        Ok(bytes) => bytes,
        Err(e) => {
            return CheckResult {
                check: "signature".to_string(),
                passed: false,
                message: format!("Invalid signature hex: {}", e),
            };
        }
    };

    let sig_array: [u8; 64] = match sig_bytes.try_into() {
        Ok(arr) => arr,
        Err(_) => {
            return CheckResult {
                check: "signature".to_string(),
                passed: false,
                message: "Signature must be 64 bytes".to_string(),
            };
        }
    };

    let signature = Signature::from_bytes(&sig_array);

    // Canonicalize receipt for signature verification
    let receipt_value = serde_json::to_value(receipt).unwrap();
    let canonical = jcs_canonicalize(&receipt_value);

    match verifying_key.verify(canonical.as_bytes(), &signature) {
        Ok(_) => CheckResult {
            check: "signature".to_string(),
            passed: true,
            message: "Ed25519 signature valid".to_string(),
        },
        Err(e) => CheckResult {
            check: "signature".to_string(),
            passed: false,
            message: format!("Signature verification failed: {}", e),
        },
    }
}

fn verify_policy_hash(receipt: &ReceiptV1, policy: Option<&PolicyComponents>) -> CheckResult {
    let Some(policy) = policy else {
        return CheckResult {
            check: "policy_hash".to_string(),
            passed: true,
            message: "Policy hash check skipped (no policy file provided)".to_string(),
        };
    };

    let policy_value = serde_json::to_value(policy).unwrap();
    let computed_policy_hash = hash_jcs(&policy_value);
    let matches = computed_policy_hash == receipt.envelope.policy_hash;

    CheckResult {
        check: "policy_hash".to_string(),
        passed: matches,
        message: if matches {
            format!("Policy hash matches: {}", &receipt.envelope.policy_hash[..16])
        } else {
            format!(
                "Policy hash mismatch! Receipt: {}, Computed: {}",
                &receipt.envelope.policy_hash[..16],
                &computed_policy_hash[..16]
            )
        },
    }
}

fn verify_ihsan_threshold(receipt: &ReceiptV1, threshold_bits: i64) -> CheckResult {
    let passes = receipt.ihsan.composite >= threshold_bits;
    let composite_f64 = receipt.ihsan.composite as f64 / 0x1_0000_0000_i64 as f64;
    let threshold_f64 = threshold_bits as f64 / 0x1_0000_0000_i64 as f64;

    CheckResult {
        check: "ihsan_threshold".to_string(),
        passed: passes,
        message: format!(
            "Ihsān composite: {:.4} (threshold: {:.4}) - {}",
            composite_f64,
            threshold_f64,
            if passes { "PASS" } else { "FAIL" }
        ),
    }
}

fn verify_no_floats(receipt: &ReceiptV1) -> CheckResult {
    // Check that ihsan vector uses only integers
    let receipt_json = serde_json::to_string(receipt).unwrap();

    // Simple heuristic: look for decimal points followed by digits
    // This isn't perfect but catches obvious float usage
    let has_floats = receipt_json.contains(".0,") ||
        receipt_json.contains(".0}") ||
        receipt_json.contains(".0]");

    CheckResult {
        check: "no_floats".to_string(),
        passed: !has_floats,
        message: if has_floats {
            "WARNING: Possible float values detected in receipt".to_string()
        } else {
            "No float values detected (determinism OK)".to_string()
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAIN VERIFICATION
// ═══════════════════════════════════════════════════════════════════════════════

fn verify_chain_continuity(receipts: &[ReceiptV1]) -> Vec<CheckResult> {
    let mut results = Vec::new();

    if receipts.len() < 2 {
        results.push(CheckResult {
            check: "chain_continuity".to_string(),
            passed: true,
            message: "Single receipt - no chain to verify".to_string(),
        });
        return results;
    }

    // Sort by timestamp
    let mut sorted: Vec<&ReceiptV1> = receipts.iter().collect();
    sorted.sort_by_key(|r| r.envelope.timestamp_ns);

    let mut prev_hash = &sorted[0].hash_chain.current;
    let mut chain_valid = true;

    for i in 1..sorted.len() {
        let receipt = sorted[i];
        if receipt.hash_chain.prev != *prev_hash {
            chain_valid = false;
            results.push(CheckResult {
                check: "chain_continuity".to_string(),
                passed: false,
                message: format!(
                    "Chain break at {}: expected prev={}, got prev={}",
                    receipt.receipt_id,
                    &prev_hash[..16],
                    &receipt.hash_chain.prev[..16]
                ),
            });
        }
        prev_hash = &receipt.hash_chain.current;
    }

    if chain_valid {
        results.push(CheckResult {
            check: "chain_continuity".to_string(),
            passed: true,
            message: format!("Hash chain valid across {} receipts", receipts.len()),
        });
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════════
// REPLAY VERIFICATION
// ═══════════════════════════════════════════════════════════════════════════════

fn verify_no_replay(receipts: &[ReceiptV1]) -> Vec<CheckResult> {
    let mut results = Vec::new();
    let mut seen_nonces: HashSet<String> = HashSet::new();
    let mut session_counters: HashMap<String, u64> = HashMap::new();

    for receipt in receipts {
        let nonce_key = format!("{}:{}", receipt.envelope.session_id, receipt.envelope.nonce);

        // Check nonce uniqueness
        if seen_nonces.contains(&nonce_key) {
            results.push(CheckResult {
                check: "replay_nonce".to_string(),
                passed: false,
                message: format!("REPLAY DETECTED: Duplicate nonce {} in {}", receipt.envelope.nonce, receipt.receipt_id),
            });
        } else {
            seen_nonces.insert(nonce_key);
        }

        // Check counter monotonicity
        let session = &receipt.envelope.session_id;
        if let Some(&last_counter) = session_counters.get(session) {
            if receipt.envelope.counter <= last_counter {
                results.push(CheckResult {
                    check: "replay_counter".to_string(),
                    passed: false,
                    message: format!(
                        "REPLAY DETECTED: Non-monotonic counter in session {}: {} <= {}",
                        session, receipt.envelope.counter, last_counter
                    ),
                });
            }
        }
        session_counters.insert(session.clone(), receipt.envelope.counter);
    }

    if results.is_empty() {
        results.push(CheckResult {
            check: "replay".to_string(),
            passed: true,
            message: format!("No replay detected across {} receipts", receipts.len()),
        });
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERSISTENT NONCE JOURNAL (v0.2)
// ═══════════════════════════════════════════════════════════════════════════════

/// Entry in the persistent nonce journal
#[derive(Debug, Serialize, Deserialize)]
struct NonceJournalEntry {
    nonce: String,
    session_id: String,
    counter: u64,
    receipt_id: String,
    verified_at_ns: u64,
}

/// Load persistent nonce journal from disk
fn load_nonce_journal(path: &PathBuf) -> std::io::Result<(HashSet<String>, HashMap<String, u64>)> {
    let mut nonces = HashSet::new();
    let mut counters = HashMap::new();

    if !path.exists() {
        return Ok((nonces, counters));
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        if let Ok(entry) = serde_json::from_str::<NonceJournalEntry>(&line) {
            let nonce_key = format!("{}:{}", entry.session_id, entry.nonce);
            nonces.insert(nonce_key);

            // Track max counter per session
            let current = counters.entry(entry.session_id.clone()).or_insert(0);
            *current = (*current).max(entry.counter);
        }
    }

    Ok((nonces, counters))
}

/// Append entries to persistent nonce journal
fn append_to_nonce_journal(
    path: &PathBuf,
    receipts: &[ReceiptV1],
) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    for receipt in receipts {
        let entry = NonceJournalEntry {
            nonce: receipt.envelope.nonce.clone(),
            session_id: receipt.envelope.session_id.clone(),
            counter: receipt.envelope.counter,
            receipt_id: receipt.receipt_id.clone(),
            verified_at_ns: now_ns,
        };

        let line = serde_json::to_string(&entry)?;
        writeln!(file, "{}", line)?;
    }

    file.sync_all()?;
    Ok(())
}

/// Verify replay using persistent nonce journal
fn verify_no_replay_persistent(
    receipts: &[ReceiptV1],
    journal_path: &PathBuf,
) -> Vec<CheckResult> {
    let mut results = Vec::new();

    // Load existing journal
    let (mut seen_nonces, mut session_counters) = match load_nonce_journal(journal_path) {
        Ok((n, c)) => (n, c),
        Err(e) => {
            results.push(CheckResult {
                check: "nonce_journal_load".to_string(),
                passed: false,
                message: format!("Failed to load nonce journal: {}", e),
            });
            return results;
        }
    };

    let journal_entries_before = seen_nonces.len();
    let mut new_receipts_for_journal = Vec::new();

    for receipt in receipts {
        let nonce_key = format!("{}:{}", receipt.envelope.session_id, receipt.envelope.nonce);

        // Check nonce uniqueness (including historical)
        if seen_nonces.contains(&nonce_key) {
            results.push(CheckResult {
                check: "replay_nonce_persistent".to_string(),
                passed: false,
                message: format!(
                    "REPLAY DETECTED (historical): Duplicate nonce {} in {}",
                    receipt.envelope.nonce, receipt.receipt_id
                ),
            });
        } else {
            seen_nonces.insert(nonce_key);
            new_receipts_for_journal.push(receipt.clone());
        }

        // Check counter monotonicity (including historical)
        let session = &receipt.envelope.session_id;
        if let Some(&last_counter) = session_counters.get(session) {
            if receipt.envelope.counter <= last_counter {
                results.push(CheckResult {
                    check: "replay_counter_persistent".to_string(),
                    passed: false,
                    message: format!(
                        "REPLAY DETECTED (historical): Non-monotonic counter in session {}: {} <= {}",
                        session, receipt.envelope.counter, last_counter
                    ),
                });
            }
        }
        let current = session_counters.entry(session.clone()).or_insert(0);
        *current = (*current).max(receipt.envelope.counter);
    }

    // Append new valid entries to journal
    if !new_receipts_for_journal.is_empty() {
        if let Err(e) = append_to_nonce_journal(journal_path, &new_receipts_for_journal) {
            results.push(CheckResult {
                check: "nonce_journal_write".to_string(),
                passed: false,
                message: format!("Failed to update nonce journal: {}", e),
            });
        }
    }

    if results.is_empty() {
        results.push(CheckResult {
            check: "replay_persistent".to_string(),
            passed: true,
            message: format!(
                "No replay detected ({} historical entries, {} new entries)",
                journal_entries_before,
                new_receipts_for_journal.len()
            ),
        });
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Load policy if provided
    let policy: Option<PolicyComponents> = if let Some(ref path) = args.policy {
        let content = fs::read_to_string(path)?;
        Some(serde_json::from_str(&content)?)
    } else {
        None
    };

    // Load public key if provided
    let pubkey: Option<String> = if let Some(ref path) = args.pubkey {
        Some(fs::read_to_string(path)?.trim().to_string())
    } else {
        None
    };

    // Load all receipts
    let mut receipts: Vec<ReceiptV1> = Vec::new();
    for path in &args.receipts {
        let content = fs::read_to_string(path)?;
        let receipt: ReceiptV1 = serde_json::from_str(&content)?;
        receipts.push(receipt);
    }

    // Production threshold (0.95 as Fixed64)
    let threshold_bits = (0.95 * 0x1_0000_0000_i64 as f64) as i64;

    // Verify each receipt
    let mut all_results: Vec<VerificationResult> = Vec::new();
    let mut all_passed = true;

    for receipt in &receipts {
        let mut checks = Vec::new();

        checks.push(verify_hash_chain(receipt));
        checks.push(verify_signature(receipt, pubkey.as_deref()));
        checks.push(verify_policy_hash(receipt, policy.as_ref()));
        checks.push(verify_ihsan_threshold(receipt, threshold_bits));
        checks.push(verify_no_floats(receipt));

        let passed = checks.iter().all(|c| c.passed);
        if !passed {
            all_passed = false;
        }

        all_results.push(VerificationResult {
            receipt_id: receipt.receipt_id.clone(),
            passed,
            checks,
        });
    }

    // Chain verification
    if args.chain && receipts.len() > 1 {
        let chain_checks = verify_chain_continuity(&receipts);
        for check in &chain_checks {
            if !check.passed {
                all_passed = false;
            }
        }
        if args.verbose || !all_passed {
            for check in chain_checks {
                println!("[CHAIN] {} - {}", if check.passed { "PASS" } else { "FAIL" }, check.message);
            }
        }
    }

    // Replay verification (choose persistent or in-memory)
    if args.replay || args.nonce_journal.is_some() {
        let replay_checks = if let Some(ref journal_path) = args.nonce_journal {
            // Use persistent nonce journal for stateful verification
            verify_no_replay_persistent(&receipts, journal_path)
        } else if receipts.len() > 1 {
            // Use in-memory verification for this batch only
            verify_no_replay(&receipts)
        } else {
            vec![CheckResult {
                check: "replay".to_string(),
                passed: true,
                message: "Single receipt - replay check skipped".to_string(),
            }]
        };

        for check in &replay_checks {
            if !check.passed {
                all_passed = false;
            }
        }
        if args.verbose || !all_passed {
            for check in replay_checks {
                println!("[REPLAY] {} - {}", if check.passed { "PASS" } else { "FAIL" }, check.message);
            }
        }
    }

    // Output results
    match args.output.as_str() {
        "json" => {
            let output = serde_json::json!({
                "passed": all_passed,
                "receipts_verified": receipts.len(),
                "results": all_results,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        _ => {
            println!("\n═══════════════════════════════════════════════════════════════════════");
            println!("  BIZRA RECEIPT VERIFIER - Results");
            println!("═══════════════════════════════════════════════════════════════════════\n");

            for result in &all_results {
                let status = if result.passed { "PASS" } else { "FAIL" };
                println!("[{}] {}", status, result.receipt_id);

                if args.verbose || !result.passed {
                    for check in &result.checks {
                        let mark = if check.passed { "+" } else { "!" };
                        println!("  [{}] {}: {}", mark, check.check, check.message);
                    }
                }
            }

            println!("\n───────────────────────────────────────────────────────────────────────");
            println!(
                "  Summary: {} receipts verified, {} passed",
                receipts.len(),
                all_results.iter().filter(|r| r.passed).count()
            );
            println!(
                "  Overall: {}",
                if all_passed { "ALL CHECKS PASSED" } else { "VERIFICATION FAILED" }
            );
            println!("───────────────────────────────────────────────────────────────────────\n");
        }
    }

    std::process::exit(if all_passed { 0 } else { 1 });
}
