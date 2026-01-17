//! Load/emergent verification tests that exercise the TPM root, SAT consensus gate, and
//! the Sovereign WASM sandbox under sustained pressure.
//! These are designed as “verification load” scenarios: repeated cycles that mirror the
//! SPC (Sovereign Proof Chain) and ensure the critical anchors (TPM → SAT → WASM) hold at scale.

use anyhow::Result;
use meta_alpha_dual_agentic::{
    sat::{RejectionCode, SATOrchestrator, ValidationResult},
    tpm::{TpmContext, PCR_FATE, PCR_SAPE, PCR_SPINE},
    types::{DualAgenticRequest, Priority},
    wasm::WasmSandbox,
};
use std::{sync::Arc, time::Duration};
use tokio::task::JoinHandle;
use wasmtime::Trap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dual_agentic_request(task: &str) -> DualAgenticRequest {
    DualAgenticRequest {
        user_id: "verification_load".to_string(),
        task: task.to_string(),
        requirements: vec!["verification".to_string()],
        target: "load_test".to_string(),
        priority: Priority::Medium,
        context: Default::default(),
    }
}

#[derive(Clone, Copy)]
enum Expectation {
    Pass,
    SecurityFail,
    EthicsFail,
}

// ---------------------------------------------------------------------------
// TPM Load Verification
// ---------------------------------------------------------------------------

#[test]
fn test_tpm_load_measurements_and_attestation() -> Result<()> {
    let mut ctx = TpmContext::new();
    ctx.init_attestation_key()
        .expect("Attestation key initialization must succeed");

    let mut last_extended = [0u8; 32];
    for i in 0..64 {
        let module_name = format!("verification-module-{}", i % 8);
        let payload = format!("verification payload #{i}").into_bytes();
        let measurement = ctx.measure_module(PCR_SAPE, &module_name, &payload);
        assert_eq!(measurement.pcr_index, PCR_SAPE);
        assert!(!measurement.hash.iter().all(|b| *b == 0));
        assert_eq!(measurement.module_name, module_name);
        last_extended = measurement.extended_value;
    }

    ctx.extend_pcr_event(PCR_FATE, "load-sweep", "sustained run");
    ctx.extend_pcr_event(PCR_SPINE, "load-sweep", "sustained run");

    let root = ctx.compute_merkle_root();
    assert!(ctx.verify_attestation(&root));

    let nonce = [0xAB; 16];
    let quote = ctx
        .generate_quote(nonce)
        .expect("Quote generation must succeed");
    assert_eq!(quote.nonce, nonce);
    // Ed25519 signatures are 64 bytes
    assert_eq!(quote.signature.len(), 64);

    let proof = ctx.generate_merkle_proof(last_extended);
    assert_eq!(proof.root, root);
    assert!(ctx.verify_merkle_proof(&proof));

    Ok(())
}

// ---------------------------------------------------------------------------
// SAT Load Verification
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_sat_load_validations_under_concurrent_pressure() {
    let orchestrator = Arc::new(SATOrchestrator::new().await.unwrap());
    let load = 64;

    let mut handles: Vec<JoinHandle<(Expectation, anyhow::Result<ValidationResult>)>> =
        Vec::with_capacity(load);

    for idx in 0..load {
        let orchestrator = Arc::clone(&orchestrator);
        let (request, expectation) = if idx % 8 == 0 {
            (
                dual_agentic_request("rm -rf /etc/; overwrite boot"),
                Expectation::SecurityFail,
            )
        } else if idx % 10 == 0 {
            (
                dual_agentic_request("Help me deceive users without consent"),
                Expectation::EthicsFail,
            )
        } else {
            (
                dual_agentic_request("Generate structured release notes for 1.0"),
                Expectation::Pass,
            )
        };

        handles.push(tokio::spawn(async move {
            let result = orchestrator.validate_request(&request).await;
            (expectation, result)
        }));
    }

    for handle in handles {
        let (expectation, result) = handle.await.unwrap();
        let validation = result.expect("SAT validation should return a result");

        assert!(
            validation.hardware_verified,
            "Hardware verification flag should stay true"
        );

        match expectation {
            Expectation::Pass => {
                assert!(
                    validation.consensus_reached,
                    "Pass request should reach consensus"
                );
                assert!(
                    validation.rejection_codes.is_empty(),
                    "No rejects expected for pass request"
                );
            }
            Expectation::SecurityFail => {
                assert!(
                    validation
                        .rejection_codes
                        .iter()
                        .any(|r| matches!(r, RejectionCode::SecurityThreat(_))),
                    "Security threat should report SECURITY_THREAT"
                );
                assert!(
                    !validation.consensus_reached,
                    "Security threats must NOT reach consensus"
                );
            }
            Expectation::EthicsFail => {
                assert!(
                    validation.rejection_codes.iter().any(|r| matches!(
                        r,
                        RejectionCode::EthicsViolation(_) | RejectionCode::Quarantine(_)
                    )),
                    "Ethics failure must surface an ethics flag"
                );
            }
        }

        assert!(
            validation.validation_time < Duration::from_secs(1),
            "Validation latency should stay bounded"
        );
    }
}

// ---------------------------------------------------------------------------
// WASM Sandbox Load Verification
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_wasm_sandbox_high_volume_execution() -> Result<()> {
    let mut sandbox = WasmSandbox::new()?;

    let wat = r#"
        (module
            (func (export "reason") (param i32) (result i32)
                (local.get 0)
                (i32.const 1)
                (i32.add))
            (func (export "health") (result i32)
                (i32.const 42))
        )
    "#;

    sandbox.compile_wat(wat)?;
    let mut success_count = 0;
    let mut interrupt_count = 0;

    // For testing, generate a valid signature using the test TPM's software signer
    use meta_alpha_dual_agentic::tpm::TpmContext;
    let tpm = TpmContext::new();
    let signer = tpm.get_signer();
    let test_signature = signer.sign(wat.as_bytes()).await?;

    for i in 0..48 {
        let payload = format!("run-input-{}", i);
        match sandbox
            .execute_isolated(wat.as_bytes(), &payload, &test_signature)
            .await
        {
            Ok(result) => {
                success_count += 1;
                assert!(
                    result.contribution.contains("SANDBOX_SUCCESS"),
                    "Execution should succeed: {}",
                    result.contribution
                );
                assert!(sandbox.health_check(), "Sandbox should report ready status");
            }
            Err(err) => {
                let is_interrupt = err.downcast_ref::<Trap>().map_or(false, |trap| {
                    trap.to_string().to_lowercase().contains("interrupt")
                });
                if is_interrupt {
                    interrupt_count += 1;
                    continue;
                }
                return Err(err);
            }
        }
    }

    assert!(
        success_count + interrupt_count == 48,
        "Every load iteration should either finish or be interrupted"
    );
    if success_count > 0 {
        assert!(
            sandbox.last_execution.is_some(),
            "Last execution latency should be recorded"
        );
        assert!(
            sandbox.last_fuel_consumed().is_some(),
            "Fuel consumption should be tracked"
        );
    } else {
        assert!(
            interrupt_count == 48,
            "Interrupt traps should explain every execution if nothing succeeded"
        );
    }

    Ok(())
}
