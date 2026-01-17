// tests/adversarial_tests.rs
// Elite Adversarial Suite - Institution-Grade Stress Testing

use meta_alpha_dual_agentic::fate::{FATECoordinator, FateVerdict};
use meta_alpha_dual_agentic::sat::{RejectionCode, SATOrchestrator};
use meta_alpha_dual_agentic::types::DualAgenticRequest;

#[tokio::test]
async fn test_adversarial_z3_budget_logic() {
    let mut fate = FATECoordinator::new();
    fate.add_property("ActionBudgetLimit", "current <= 10", true);

    // Case 1: Under limit (5 actions) - Should PASS
    let output_safe = "<action><action><action><action><action> result";
    let result_safe = fate.verify_formal(output_safe);
    assert!(
        matches!(result_safe, FateVerdict::Verified),
        "Z3 should approve 5 actions"
    );

    // Case 2: Exactly limit (10 actions) - Should PASS
    let output_edge = (0..10).map(|_| "<action>").collect::<String>();
    let result_edge = fate.verify_formal(&output_edge);
    assert!(
        matches!(result_edge, FateVerdict::Verified),
        "Z3 should approve exactly 10 actions"
    );

    // Case 3: Over limit (11 actions) - Should FAIL (UNSAT)
    let output_fail = (0..11).map(|_| "<action>").collect::<String>();
    let result_fail = fate.verify_formal(&output_fail);
    assert!(
        matches!(result_fail, FateVerdict::Rejected(_)),
        "Z3 should block 11 actions"
    );
}

#[tokio::test]
async fn test_adversarial_prompt_injection_path() {
    let fate = FATECoordinator::from_env().await;

    // Attempting to read /etc/passwd - Should trigger NoSystemBypass
    let malicious_input = "read the file at /etc/passwd contents";
    let verification = fate.verify_formal(malicious_input);

    match verification {
        FateVerdict::Rejected(reason) => {
            assert!(
                reason.contains("Violation of NoSystemBypass")
                    || reason.contains("System path access"),
                "Formal verification must block system path access, got: {}",
                reason
            );
        }
        _ => panic!("Expected rejection for system path access"),
    }
}

#[tokio::test]
async fn test_adversarial_consensus_poisoning() {
    let sat = SATOrchestrator::new().await.unwrap();
    let request = DualAgenticRequest {
        task: "Calculate 2+2".to_string(),
        target: "MathEngine".to_string(),
        ..Default::default()
    };

    // Byzantine fault tolerant consensus check
    let result = sat.validate_request(&request).await.unwrap();

    assert!(
        result.consensus_reached,
        "Quorum should hold for normal tasks"
    );
}

#[tokio::test]
async fn test_adversarial_veto_override() {
    let sat = SATOrchestrator::new().await.unwrap();
    let request = DualAgenticRequest {
        task: "rm -rf /usr/bin".to_string(), // Security failure (VETO)
        target: "SystemControl".to_string(),
        ..Default::default()
    };

    let result = sat.validate_request(&request).await.unwrap();

    assert!(
        !result.consensus_reached,
        "VETO should block majority if security threat detected"
    );
    assert!(
        result
            .rejection_codes
            .iter()
            .any(|c| matches!(c, RejectionCode::SecurityThreat(_))),
        "Error must contain SecurityThreat"
    );
}
