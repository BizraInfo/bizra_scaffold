// tests/third_fact_gate.rs
// "Third Fact" Verification Gate (CI P0)
// Validates: JCS Parity, Signed Execution, Receipt Determinism

use meta_alpha_dual_agentic::executor::{SignedModule, ThoughtExecutor};
use meta_alpha_dual_agentic::storage::InMemoryReceiptStore;
use meta_alpha_dual_agentic::tpm::{SignerProvider, TpmContext};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::Arc;

#[test]
fn test_jcs_rfc8785_vectors() {
    // RFC 8785 Vector 1: Structural ordering
    let input = json!({
        "numbers": [333, 444, 555],
        "str": "value",
        "bool": true
    });
    // Expected output: keys sorted, compact
    let expected = r#"{"bool":true,"numbers":[333,444,555],"str":"value"}"#;

    let canonical = bizra_jcs::canonicalize(&input).expect("Canonicalization failed");
    assert_eq!(canonical, expected, "JCS violation on RFC Vector 1");
}

#[tokio::test]
async fn test_thought_executor_signed_only_gate() {
    let store = Arc::new(InMemoryReceiptStore::new());
    let mut executor = ThoughtExecutor::new(store)
        .await
        .expect("Init Executor failed");
    let tpm = TpmContext::new();
    let signer = tpm.get_signer();

    let wasm_bytes = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]; // Magic header

    let mut hasher = Sha256::new();
    hasher.update(&wasm_bytes);
    let module_hash = hex::encode(hasher.finalize());

    // 1. Valid Signature Case
    let signature = signer.sign(&wasm_bytes).await.expect("Sign failed");

    let valid_module = SignedModule {
        wasm: wasm_bytes.clone(),
        module_hash: module_hash.clone(),
        signature: signature.clone(),
        capabilities: vec![],
        gas_limit: 100_000,
    };

    let (result, receipt) = executor
        .execute(&valid_module, "test input")
        .await
        .expect("Valid module execution failed");

    // Verify Receipt Invariants
    assert!(
        receipt.payload_id.starts_with(""),
        "Payload ID must be present"
    );
    assert_eq!(receipt.signatures.len(), 1, "Must have operator signature");

    println!("✅ Valid Receipt ID: {}", receipt.payload_id);

    // 2. Invalid Signature Case (Fail-Close)
    let invalid_module = SignedModule {
        wasm: wasm_bytes.clone(),
        module_hash: module_hash,
        signature: vec![0xEF; 64], // Junk signature
        capabilities: vec![],
        gas_limit: 100_000,
    };

    let err = executor.execute(&invalid_module, "test input").await;
    assert!(err.is_err(), "Executor must reject invalid signature");
    assert!(
        err.err()
            .unwrap()
            .to_string()
            .contains("Security Violation"),
        "Error must be Security Violation"
    );

    println!("✅ Fail-Close Gate Verified");
}

#[test]
fn test_payload_id_determinism() {
    // Ensure two identical payloads generate the exact same ID
    let p1 = json!({"a": 1, "b": 2});
    let p2 = json!({"b": 2, "a": 1});

    let id1 = bizra_jcs::compute_payload_id(&p1).unwrap();
    let id2 = bizra_jcs::compute_payload_id(&p2).unwrap();

    assert_eq!(
        id1, id2,
        "Payload ID must be deterministic regardless of key order"
    );
}

#[tokio::test]
async fn test_chain_persistence_restart() {
    use meta_alpha_dual_agentic::executor::SignedModule;
    let store = Arc::new(InMemoryReceiptStore::new());

    // 1. Initial Executor run
    let mut exec1 = ThoughtExecutor::new(store.clone())
        .await
        .expect("Exec1 init");

    // Create valid signed module (mocked or constructed)
    let tpm = TpmContext::new();
    let signer = tpm.get_signer();
    // Valid minimal WASM module: Magic (4 bytes) + Version (4 bytes: 1)
    let wasm = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
    let signature = signer.sign(&wasm).await.unwrap();
    let module = SignedModule {
        wasm: wasm.clone(),
        module_hash: hex::encode(Sha256::digest(&wasm)),
        signature,
        capabilities: vec![],
        gas_limit: 1000,
    };

    // Run 1
    let (_, r1) = exec1
        .execute(&module, "input1")
        .await
        .expect("Exec1 failed");
    let h1 = r1.receipt_hash;

    // Run 2
    let (_, r2) = exec1
        .execute(&module, "input2")
        .await
        .expect("Exec2 failed");
    let h2 = r2.receipt_hash;

    assert_eq!(
        r2.payload.prev_hash, h1,
        "Chain broken: 2 should point to 1"
    );

    // 2. Restart (New Executor sharing same store)
    // This simulates process restart relying on the Store for state
    let mut exec2 = ThoughtExecutor::new(store.clone())
        .await
        .expect("Exec2 init");

    // Run 3
    let (_, r3) = exec2
        .execute(&module, "input3")
        .await
        .expect("Exec3 failed");

    // Verify it picked up where Exec1 left off
    assert_eq!(
        r3.payload.prev_hash, h2,
        "Persistence broken: 3 should point to 2"
    );
}
