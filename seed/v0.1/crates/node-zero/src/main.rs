use bizra_core::{Constitution, IhsanEngine, FileLedger, Ledger, LedgerEvent};
use bizra_core::evidence::HashChain;
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let constitution = Constitution::seed();
    let ihsan = IhsanEngine::new(constitution.ihsan_threshold);

    // Seed ledger
    let mut ledger = FileLedger::new("./genesis_ledger.jsonl");
    let mut chain = HashChain::new();

    // --- Sense (stub) ---
    let sensed = 42u8; // placeholder for sensor reading

    // --- Reason (stub) ---
    let predicted_ok = sensed >= 40;

    // --- Verify (policy) ---
    let policy_ok = true; // replace with FATE gate

    // --- Observe (stub reality check) ---
    let observed_ok = true;

    let score = ihsan.score_action(predicted_ok, observed_ok, policy_ok);
    let commit = ihsan.should_commit(score);

    let payload = json!({
        "sensed": sensed,
        "predicted_ok": predicted_ok,
        "observed_ok": observed_ok,
        "policy_ok": policy_ok,
        "commit": commit,
    });

    let bytes = serde_json::to_vec(&payload)?;
    let h = chain.step(&bytes);

    let event = LedgerEvent {
        ts_unix_ms: FileLedger::now_ms(),
        node_id: "node-zero".to_string(),
        kind: "first_verified_thought".to_string(),
        payload_json: payload,
        ihsan: score,
        hash: hex::encode(h),
    };

    ledger.append(event)?;

    println!("Node-0 executed. Ihsan={:.3} commit={}", score.composite, commit);
    println!("Ledger appended: genesis_ledger.jsonl");
    Ok(())
}
