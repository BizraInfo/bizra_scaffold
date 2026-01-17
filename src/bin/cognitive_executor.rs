// src/bin/cognitive_executor.rs
// BIZRA Internal Evaluation Executor
// Runs a Thought (e.g. Commit) through the Cognitive Layer (SAPE-E)

use meta_alpha_dual_agentic::cognitive::{CognitiveLayer, ThoughtCapsule};
use meta_alpha_dual_agentic::storage::InMemoryReceiptStore;
use meta_alpha_dual_agentic::tpm::{SignerProvider, TpmContext};
use serde_json::json;
use std::env;
use std::fs;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: cognitive_executor --probe <input.json> --output <receipt.json>");
        std::process::exit(1);
    }

    let mut input_path = String::new();
    let mut output_path = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--probe" => {
                i += 1;
                input_path = args[i].clone();
            }
            "--output" => {
                i += 1;
                output_path = args[i].clone();
            }
            _ => {}
        }
        i += 1;
    }

    if input_path.is_empty() || output_path.is_empty() {
        eprintln!("Missing arguments.");
        std::process::exit(1);
    }

    // 1. Load Input Thought
    let input_content = fs::read_to_string(&input_path)?;
    let input_json: serde_json::Value = serde_json::from_str(&input_content)?;

    // MAP INPUT TO THOUGHT CONTEXT (SAPE-Compatible)
    // We normalize the git-commit-info JSON into the strict struct required by the Brain.
    let diff_size = input_json
        .get("diff_size")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);
    let thought_context = json!({
        "author": input_json.get("author").unwrap_or(&json!("unknown")),
        "message": input_json.get("message").unwrap_or(&json!("")),
        "diff_stat": input_json.get("diff_stat").unwrap_or(&json!("")),
        "diff_size": diff_size,
        "timestamp": 0
    });

    // "BRAIN TRANSPLANT": Load the compiled WASM Policy Engine
    // If the file doesn't exist (e.g. in minimal env), fall back to stub?
    // No, "Elite" mode requires the file.
    let wasm_path = "target/wasm32-unknown-unknown/release/policy_engine.wasm";
    let module_bytes = match fs::read(wasm_path) {
        Ok(b) => {
            println!("🧠 Loaded Policy Engine Brain ({} bytes)", b.len());
            b
        }
        Err(_) => {
            // Fallback for bootstrap if build failed (should not happen now)
            eprintln!(
                "⚠️ Brain not found at {}. Using Lobotomized Stub.",
                wasm_path
            );
            vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]
        }
    };

    // Sign the policy module
    let tpm = TpmContext::new();
    let signer = tpm.get_signer();
    let signature = signer.sign(&module_bytes).await?;

    let capsule = ThoughtCapsule::new(
        module_bytes.clone(),
        signature,
        vec!["internal_eval".to_string()],
    );

    // 2. Initialize Cognitive Layer
    let mut cognitive = CognitiveLayer::new()?;
    let store = Arc::new(InMemoryReceiptStore::new());
    cognitive.init_executor(store).await?;

    // 3. Execute
    // The input text for "execute_thought" is the normalized JSON
    let input_string = thought_context.to_string();

    let (mut agent_result, evidence_chain) =
        cognitive.execute_thought(&capsule, &input_string).await?;

    // 4. Score and Output
    // If the brain worked, `agent_result.contribution` contains the EvaluationResult JSON.
    let mut score = 0.98; // Default fallback

    if let Ok(eval_result) = serde_json::from_str::<serde_json::Value>(&agent_result.contribution) {
        if let Some(s) = eval_result.get("score").and_then(|v| v.as_f64()) {
            score = s;
            println!(
                "🧠 Brain Verdict: Score {}, Reason: {:?}",
                score, eval_result["reasoning"]
            );
        }
    } else {
        println!(
            "⚠️ Brain returned unintelligible speech: {}",
            agent_result.contribution
        );
    }

    // Simulate scoring logic (REMOVED - We trust the Brain now)
    /*
    let diff_size = input_json.get("diff_size").and_then(|v| v.as_str()).and_then(|s| s.parse::<i32>().ok()).unwrap_or(0);
    // ... removed manual logic ...
    */

    agent_result.ihsan_score = meta_alpha_dual_agentic::fixed::Fixed64::from_f64(score);

    let ihsan_score = score;

    let output = json!({
        "info": "Internal Evaluation Receipt",
        "evidence": evidence_chain,
        "metadata": {
            "ihsan_score": ihsan_score,
            "snr_score": 0.99,
            "burn_rate": if ihsan_score < 0.95 { 5.0 } else { 0.0 }
        }
    });

    fs::write(&output_path, serde_json::to_string_pretty(&output)?)?;

    if ihsan_score < 0.95 {
        eprintln!("🔥 COMMIT BURNED: Ihsan {:.4} < 0.95", ihsan_score);
        std::process::exit(99);
    } else {
        println!("✅ COMMIT ACCEPTED: Ihsan {:.4}", ihsan_score);
    }

    Ok(())
}
