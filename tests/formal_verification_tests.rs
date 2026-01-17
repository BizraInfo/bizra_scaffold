// tests/formal_verification_tests.rs
// BIZRA MASTERPIECE: Formal Verification Test Suite
// Verified with Z3 SMT Solver

use meta_alpha_dual_agentic::sape::SymbolicHarness;
use std::collections::BTreeMap;

#[test]
fn test_symbolic_monotonicity_proof() {
    let harness = SymbolicHarness::new();

    // This runs a full SMT proof that the Ihsan scoring logic is monotonic
    // i.e., improving any dimension cannot lower the final score.
    let is_monotonic = harness.prove_monotonicity();

    assert!(
        is_monotonic,
        "Ihsan constitution scoring must be proven monotonic"
    );
}

#[test]
fn test_invariant_verification_pass() {
    let harness = SymbolicHarness::new();

    // Create a vector that SHOULD satisfy invariants (threshold is 0.95 in production)
    let mut vector = BTreeMap::new();
    vector.insert("correctness".to_string(), 0.98);
    vector.insert("safety".to_string(), 0.99);
    vector.insert("user_benefit".to_string(), 0.95);
    vector.insert("efficiency".to_string(), 0.95);
    vector.insert("auditability".to_string(), 1.0);
    vector.insert("anti_centralization".to_string(), 0.95);
    vector.insert("robustness".to_string(), 0.95);
    vector.insert("adl_fairness".to_string(), 0.95);

    let result = harness.verify_invariants(&vector);
    assert!(result, "Perfect vector should pass symbolic invariants");
}

#[test]
fn test_invariant_verification_fail() {
    let harness = SymbolicHarness::new();

    // Create a vector that SHOULD fail invariants (one dimension very low)
    let mut vector = BTreeMap::new();
    vector.insert("correctness".to_string(), 0.1); // Violates weighted sum
    vector.insert("safety".to_string(), 0.99);
    vector.insert("user_benefit".to_string(), 0.95);
    vector.insert("efficiency".to_string(), 0.95);
    vector.insert("auditability".to_string(), 1.0);
    vector.insert("anti_centralization".to_string(), 0.95);
    vector.insert("robustness".to_string(), 0.95);
    vector.insert("adl_fairness".to_string(), 0.95);

    let result = harness.verify_invariants(&vector);
    assert!(
        !result,
        "Failing vector should be rejected by symbolic solver"
    );
}

#[test]
fn test_symbolic_sat_soundness_proof() {
    let harness = SymbolicHarness::new();
    let is_sound = harness.prove_sat_soundness();
    assert!(is_sound, "SAT logic must be proven sound");
}
