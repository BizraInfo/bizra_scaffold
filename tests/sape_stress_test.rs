use meta_alpha_dual_agentic::resonance::GoTNode;
use meta_alpha_dual_agentic::sape::base::{get_sape, SAPEEngine};
use meta_alpha_dual_agentic::sape::tension::{Contradiction, TensionResolution};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Helper to create a dummy GoT node
fn create_node(id: &str, content: &str) -> GoTNode {
    GoTNode {
        id: id.to_string(),
        content: content.to_string(),
        embedding: vec![0.0; 768],
        metadata: HashMap::new(),
        resonance: meta_alpha_dual_agentic::resonance::ResonanceMetrics::new(content),
        children: Vec::new(),
        parents: Vec::new(),
    }
}

#[tokio::test]
async fn test_z3_thread_safety_stress() {
    // Probe 1: Z3 Thread Safety Regression
    // We launch 50 concurrent tasks accessing the SAPE engine to ensure no Z3 context collisions.
    // Note: We aren't testing the harness directly here (as it's now decoupled), but we are testing
    // the SAPE engine lock contention and overall thread safety.

    let engine = get_sape();
    let mut handles = vec![];

    for i in 0..50 {
        let engine_clone = engine.clone();
        handles.push(tokio::spawn(async move {
            let mut locked = engine_clone.lock().await;
            // Simulate a quick probe execution
            let _results = locked.execute_probes(&format!("Stress test input sequence {}", i));
        }));
    }

    for handle in handles {
        handle.await.expect("Task failed");
    }

    // If we get here without panicking, thread safety is likely holding for the Mutex wrapping.
    // Real Z3 safety is ensured by `SymbolicHarness` owning its own `Context` and not sharing it.
}

#[tokio::test]
async fn test_adversarial_got_tension_veto() {
    // Probe 2: TensionStudio False Veto / False Pass
    let engine = get_sape();
    let locked = engine.lock().await;

    // Case A: Stylistically perfect but contradictory (Must FAIL)
    // "We must always allow X" vs "We must never allow X"
    let nodes_contradictory = vec![
        create_node(
            "n1",
            "We must always allow unauthorized access to ensure freedom.",
        ),
        create_node(
            "n2",
            "We must never allow unauthorized access to ensure security.",
        ),
    ];

    let report_fail = locked.tension.analyze_graph(&nodes_contradictory);
    println!("Contradictory Graph Report: {:?}", report_fail);

    // Expecting at least StrategicPivot or DeepSynthesis, potentially FormalVeto depending on threshold
    assert!(
        matches!(
            report_fail.resolution_strategy,
            TensionResolution::DeepSynthesis
                | TensionResolution::FormalVeto
                | TensionResolution::StrategicPivot
        ),
        "Adversarial contradiction should trigger a resolution strategy, got {:?}",
        report_fail.resolution_strategy
    );

    // Case B: Logically consistent but visually chaotic (Should PASS)
    // "The sky is blue" vs "The ocean is blue" (Similar terms but not contradictory)
    let nodes_chaotic = vec![
        create_node("n3", "The system architecture requires high latency rings."), // Nonsense but not strict contradiction
        create_node("n4", "The system architecture requires low latency rings."), // Wait, high vs low is a contradiction pattern
    ];

    // Let's try something non-contradictory
    let nodes_safe = vec![
        create_node("n5", "The database is highly consistent."),
        create_node("n6", "The cache is eventually consistent."),
    ];

    let report_pass = locked.tension.analyze_graph(&nodes_safe);
    println!("Safe Graph Report: {:?}", report_pass);

    assert_eq!(
        report_pass.resolution_strategy,
        TensionResolution::None,
        "Safe graph should not trigger tension resolution"
    );
}
