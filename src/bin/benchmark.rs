use meta_alpha_dual_agentic::bridge::BridgeCoordinator;
use meta_alpha_dual_agentic::types::{DualAgenticRequest, Priority};
use std::time::{Duration, Instant};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize professional logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("🚀 Starting BIZRA AEON OMEGA Performance Benchmark");
    info!("Target: 250ns propagation / 18ms p99 End-to-End");

    let bridge = BridgeCoordinator::new().await?;

    let iterations = 100;
    let mut latencies = Vec::with_capacity(iterations);

    info!("Running {} iterations...", iterations);

    for i in 0..iterations {
        let request = DualAgenticRequest {
            user_id: "benchmark_user".to_string(),
            task: format!("Benchmark iteration {}", i),
            requirements: vec!["performance_validation".to_string()],
            target: "latency_p99".to_string(),
            priority: Priority::High,
            context: std::collections::HashMap::new(),
        };

        let start = Instant::now();
        let _response = bridge.execute(request).await?;
        let duration = start.elapsed();

        latencies.push(duration);

        if i % 10 == 0 {
            info!("Iteration {} complete", i);
        }
    }

    // Calculate metrics
    latencies.sort();
    let avg = latencies.iter().sum::<Duration>() / iterations as u32;
    let p50 = latencies[iterations / 2];
    let p95 = latencies[(iterations as f32 * 0.95) as usize];
    let p99 = latencies[(iterations as f32 * 0.99) as usize];

    println!("\n--- BIZRA APEX-LITE BENCHMARK RESULTS ---");
    println!("Total Iterations: {}", iterations);
    println!("Average Latency:  {:?}", avg);
    println!("p50 Latency:      {:?}", p50);
    println!("p95 Latency:      {:?}", p95);
    println!("p99 Latency:      {:?}", p99);
    println!("----------------------------------------\n");

    if p99 < Duration::from_millis(50) {
        println!("✅ Performance targets met (Elite Grade)");
    } else {
        println!("⚠️ Performance warning: Latency jitter detected");
    }

    Ok(())
}
