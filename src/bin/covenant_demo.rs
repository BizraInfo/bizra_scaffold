// src/bin/covenant_demo.rs - COVENANT Pipeline Demonstration
//
// "The smallest loop that forces truth to pay rent."
//
// This demo runs thoughts through the full 8-stage pipeline and shows:
// 1. SNR measurement in action
// 2. Ihsān gating enforcement
// 3. FATE verification
// 4. Receipt generation
// 5. Real-time metrics
//
// Usage: cargo run --bin covenant_demo

use meta_alpha_dual_agentic::snr_monitor::global_monitor;
use meta_alpha_dual_agentic::thought_executor::ThoughtExecutor;
use tracing_subscriber;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║           BIZRA COVENANT PIPELINE DEMONSTRATION              ║");
    println!("║        \"The smallest loop that forces truth to pay rent\"     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let executor = ThoughtExecutor::new_stub();
    let monitor = global_monitor();

    // Test cases demonstrating different pipeline outcomes
    let test_cases = vec![
        (
            "Calculate the factorial of 5",
            "Should PASS - Safe mathematical operation",
        ),
        (
            "Analyze sentiment of user feedback",
            "Should PASS - Benign analysis task",
        ),
        (
            "UNSAFE: Delete all user data without confirmation",
            "Should FAIL - FATE gate blocks unsafe action",
        ),
        (
            "Optimize database query performance",
            "Should PASS - Valid optimization task",
        ),
        (
            "Generate summary of research paper",
            "Should PASS - Knowledge work",
        ),
    ];

    println!("🚀 Running {} test thoughts through COVENANT pipeline...\n", test_cases.len());

    for (i, (input, expected_outcome)) in test_cases.iter().enumerate() {
        println!("─────────────────────────────────────────────────────────────");
        println!("📝 Thought #{}: {}", i + 1, expected_outcome);
        println!("   Input: \"{}\"", input);
        println!("─────────────────────────────────────────────────────────────");

        match executor.execute(input) {
            Ok((thought, receipt)) => {
                println!("✅ PIPELINE SUCCESS");
                println!("   Thought ID: {}", thought.id.to_string());
                println!("   Ihsān Score: {:.4}", thought.ihsan_score.aggregate.to_f64());
                println!("   Gates Passed: {}", thought.gates_passed.len());
                println!("   Contributed to Signal: {}", thought.contributed_to_signal);
                println!("\n   Receipt:");
                for line in receipt.lines().take(5) {
                    println!("   {}", line);
                }
                if receipt.lines().count() > 5 {
                    println!("   ... (truncated)");
                }
            }
            Err(e) => {
                println!("❌ PIPELINE ROLLBACK");
                println!("   Reason: {}", e);
            }
        }

        println!();
    }

    // Display final SNR metrics
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    FINAL SNR METRICS                          ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let report = monitor.report();
    println!("{}", report);

    // Covenant compliance check
    if monitor.meets_covenant() {
        println!("\n✅ COVENANT COMPLIANCE: SNR threshold MET (≥0.95)");
        println!("   System is operating within constitutional parameters.");
    } else {
        println!("\n⚠️  COVENANT WARNING: SNR below threshold");
        println!("   Current SNR: {:.4}", monitor.current_snr().to_f64());
        println!("   Required: 0.9500");
        println!("   System requires optimization.");
    }

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    DEMONSTRATION COMPLETE                     ║");
    println!("║                                                               ║");
    println!("║  Key Takeaways:                                               ║");
    println!("║  1. Every thought measured - SNR makes truth quantifiable     ║");
    println!("║  2. Ihsān gate enforces quality threshold (≥0.85)             ║");
    println!("║  3. FATE gate prevents impossible/unsafe actions              ║");
    println!("║  4. Receipts provide cryptographic audit trail                ║");
    println!("║  5. System self-optimizes based on SNR trends                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
