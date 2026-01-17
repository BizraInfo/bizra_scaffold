// benches/sovereign_benchmark.rs
//
// BIZRA SOVEREIGN BENCHMARK HARNESS v9.0
// =======================================
// Elite Performance Verification aligned with SRE Standards
//
// SLOs (Service Level Objectives):
// - Reasoning Latency: <1ms (P99)
// - GoT Construction: <500µs
// - SNR Calculation: <100µs
// - Ihsān Verification: <50µs
//
// Run with: cargo bench --bench sovereign_benchmark

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

// Import BIZRA types
use meta_alpha_dual_agentic::{
    fixed::Fixed64,
    snr::SNREngine,
    types::{AgentResult, DualAgenticRequest, Priority},
};

/// Benchmark: SNR Calculation Speed
/// Target: <100µs per agent result
fn bench_snr_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("SNR Engine");
    group.measurement_time(Duration::from_secs(10));

    // Create test agent results with varying complexity
    let results = vec![
        create_agent_result("strategic_visionary", 100),
        create_agent_result("analytical_optimizer", 500),
        create_agent_result("quality_guardian", 1000),
        create_agent_result("integration_coordinator", 2000),
    ];

    for result in results {
        group.bench_with_input(
            BenchmarkId::new("score", result.contribution.len()),
            &result,
            |b, r| {
                b.iter(|| {
                    let score = SNREngine::score(black_box(r));
                    black_box(score)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Fixed64 Arithmetic (Deterministic Math)
/// Target: <10µs for complex operations
fn bench_fixed64_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fixed64 Determinism");

    let a = Fixed64::from_f64(0.85);
    let b = Fixed64::from_f64(0.95);
    let c_val = Fixed64::from_f64(0.90);

    group.bench_function("addition", |bench| {
        bench.iter(|| black_box(a + b + c_val));
    });

    group.bench_function("multiplication", |bench| {
        bench.iter(|| black_box(a * b * c_val));
    });

    group.bench_function("division_safe", |bench| {
        bench.iter(|| black_box(a.saturating_div(b)));
    });

    group.bench_function("ihsan_formula", |bench| {
        // Ihsān = 0.4*Excellence + 0.3*Benevolence + 0.3*Justice
        let w1 = Fixed64::from_f64(0.4);
        let w2 = Fixed64::from_f64(0.3);
        let w3 = Fixed64::from_f64(0.3);
        bench.iter(|| black_box(w1 * a + w2 * b + w3 * c_val));
    });

    group.finish();
}

/// Benchmark: Request Construction
/// Target: <50µs for full request initialization
fn bench_request_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Request Construction");

    group.bench_function("default_request", |b| {
        b.iter(|| {
            let req = DualAgenticRequest::default();
            black_box(req)
        });
    });

    group.bench_function("full_request", |b| {
        b.iter(|| {
            let req = DualAgenticRequest {
                user_id: "benchmark_user".to_string(),
                task: "Generate comprehensive performance analysis for the system".to_string(),
                requirements: vec![
                    "latency".to_string(),
                    "throughput".to_string(),
                    "reliability".to_string(),
                ],
                target: "benchmark".to_string(),
                priority: Priority::High,
                context: [
                    ("session_id".to_string(), "bench-001".to_string()),
                    ("environment".to_string(), "production".to_string()),
                ]
                .into_iter()
                .collect(),
            };
            black_box(req)
        });
    });

    group.finish();
}

/// Benchmark: Shannon Entropy Calculation (SNR Component)
/// Target: <200µs for 10KB text
fn bench_entropy_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Entropy Calculation");
    group.measurement_time(Duration::from_secs(10));

    // Generate text of varying sizes
    let sizes = [100, 500, 1000, 5000, 10000];

    for size in sizes {
        let text = generate_benchmark_text(size);
        group.bench_with_input(BenchmarkId::new("shannon_entropy", size), &text, |b, t| {
            let agent_result = AgentResult {
                agent_name: "benchmark_agent".to_string(),
                contribution: t.clone(),
                confidence: Fixed64::from_f64(0.90),
                ihsan_score: Fixed64::from_f64(0.85),
                execution_time: Duration::from_millis(10),
                metadata: std::collections::HashMap::new(),
            };
            b.iter(|| {
                let score = SNREngine::score(black_box(&agent_result));
                black_box(score.information_density)
            });
        });
    }

    group.finish();
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn create_agent_result(name: &str, text_length: usize) -> AgentResult {
    let contribution = generate_benchmark_text(text_length);
    AgentResult {
        agent_name: name.to_string(),
        contribution,
        confidence: Fixed64::from_f64(0.90),
        ihsan_score: Fixed64::from_f64(0.85),
        execution_time: Duration::from_millis(10),
        metadata: std::collections::HashMap::new(),
    }
}

fn generate_benchmark_text(length: usize) -> String {
    // Generate text with technical markers to simulate real agent output
    let base =
        "Formal verification protocol optimization ihsan sovereign genesis latency benchmark ";
    let mut result = String::with_capacity(length);
    while result.len() < length {
        result.push_str(base);
    }
    result.truncate(length);
    result
}

// ============================================================================
// CRITERION GROUPS
// ============================================================================

criterion_group!(
    name = sovereign_benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3));
    targets =
        bench_snr_calculation,
        bench_fixed64_operations,
        bench_request_construction,
        bench_entropy_calculation
);

criterion_main!(sovereign_benches);
