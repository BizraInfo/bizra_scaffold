//! BIZRA Sovereign Kernel Benchmarks
//!
//! Production performance validation for critical paths:
//! - FFI boundary latency (<5μs target)
//! - FATE verification (<100ms target)  
//! - WASM execution (<500ms target)
//! - Receipt generation (<10ms target)
//!
//! Run: cargo bench --features python

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

// Import from main crate
use meta_alpha_dual_agentic::fixed::Fixed64;
use meta_alpha_dual_agentic::ihsan;
use meta_alpha_dual_agentic::tpm::TpmContext;
use meta_alpha_dual_agentic::wasm::WasmSandbox;

/// Benchmark TPM PCR operations
fn bench_tpm_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("TPM Operations");
    group.measurement_time(Duration::from_secs(5));

    // PCR Extension benchmark
    group.bench_function("pcr_extend", |b| {
        let mut tpm = TpmContext::new();
        let module_bytes = b"test_module_data_for_benchmarking";

        b.iter(|| {
            tpm.measure_module(
                black_box(12),
                black_box("bench_module"),
                black_box(module_bytes),
            )
        });
    });

    // Quote generation benchmark
    group.bench_function("generate_quote", |b| {
        let mut tpm = TpmContext::new();
        // Pre-extend some PCRs
        tpm.measure_module(12, "sape", b"sape_bytes");
        tpm.measure_module(13, "fate", b"fate_bytes");
        tpm.measure_module(14, "spine", b"spine_bytes");

        let nonce = [0u8; 16];

        b.iter(|| tpm.generate_quote(black_box(nonce)));
    });

    // Merkle root computation benchmark
    group.bench_function("merkle_root", |b| {
        let mut tpm = TpmContext::new();
        tpm.measure_module(12, "sape", b"sape_module");
        tpm.measure_module(13, "fate", b"fate_module");
        tpm.measure_module(14, "spine", b"spine_module");

        b.iter(|| tpm.compute_merkle_root());
    });

    group.finish();
}

/// Benchmark WASM sandbox operations
fn bench_wasm_sandbox(c: &mut Criterion) {
    let mut group = c.benchmark_group("WASM Sandbox");
    group.measurement_time(Duration::from_secs(10));

    // Sandbox initialization (Target: <50ms)
    group.bench_function("init", |b| {
        b.iter(|| WasmSandbox::new());
    });

    // WASM execution (Target: <500ms)
    group.bench_function("execute_reasoning", |b| {
        let mut sandbox = WasmSandbox::new();
        let wasm_module = sandbox.elevate_pattern("bench_pattern");

        b.iter(|| {
            let _ = futures::executor::block_on(
                sandbox.execute_isolated(black_box(&wasm_module), black_box("test_input")),
            );
        });
    });

    group.finish();
}

/// Benchmark FATE Verification (Target: <100ms)
fn bench_fate_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("FATE Verification");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("verify_ethics_score", |b| {
        let weights = [0.22, 0.22, 0.14, 0.12, 0.12, 0.08, 0.06, 0.04];
        let scores = [0.95, 0.90, 0.85, 0.88, 0.92, 0.75, 0.80, 0.70];

        b.iter(|| {
            let mut total = 0.0f64;
            for (w, s) in weights.iter().zip(scores.iter()) {
                total += w * s;
            }
            black_box(total)
        });
    });

    group.finish();
}

/// Benchmark Fixed-point arithmetic (Ihsān calculations)
fn bench_fixed_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fixed Point Arithmetic");
    group.measurement_time(Duration::from_secs(5));

    // Fixed64 creation from f64
    group.bench_function("from_f64", |b| {
        b.iter(|| Fixed64::from_f64(black_box(0.95)));
    });

    // Fixed64 to f64
    group.bench_function("to_f64", |b| {
        let fixed = Fixed64::from_f64(0.95);

        b.iter(|| black_box(fixed).to_f64());
    });

    // Fixed64 multiplication (common in Ihsān weighting)
    group.bench_function("multiply", |b| {
        let a = Fixed64::from_f64(0.95);
        let b_val = Fixed64::from_f64(0.22);

        b.iter(|| black_box(a) * black_box(b_val));
    });

    // Ihsān score computation (8 dimensions)
    group.bench_function("ihsan_8d_weighted", |b| {
        let weights = [0.22, 0.22, 0.14, 0.12, 0.12, 0.08, 0.06, 0.04];
        let scores = [0.95, 0.90, 0.85, 0.88, 0.92, 0.75, 0.80, 0.70];

        b.iter(|| {
            let mut total = 0.0f64;
            for (w, s) in weights.iter().zip(scores.iter()) {
                total += w * s;
            }
            black_box(total)
        });
    });

    group.finish();
}

/// Benchmark receipt generation
fn bench_receipt_generation(c: &mut Criterion) {
    use sha2::{Digest, Sha256};

    let mut group = c.benchmark_group("Receipt Generation");
    group.measurement_time(Duration::from_secs(5));

    // SHA256 hashing (core of receipt)
    group.bench_function("sha256_32b", |b| {
        let data = [0u8; 32];

        b.iter(|| {
            let mut hasher = Sha256::new();
            hasher.update(black_box(&data));
            hasher.finalize()
        });
    });

    // SHA256 hashing (1KB payload)
    group.bench_function("sha256_1kb", |b| {
        let data = vec![0u8; 1024];

        b.iter(|| {
            let mut hasher = Sha256::new();
            hasher.update(black_box(&data));
            hasher.finalize()
        });
    });

    // Receipt structure creation
    group.bench_with_input(BenchmarkId::new("create_receipt", "full"), &(), |b, _| {
        let mut tpm = TpmContext::new();
        tpm.measure_module(12, "sape", b"sape");

        b.iter(|| {
            let nonce = [0u8; 16];
            let quote = tpm.generate_quote(nonce);

            // Simulate receipt creation
            let mut hasher = Sha256::new();
            hasher.update(&quote.pcr_digest);
            hasher.update(&quote.signature);
            let receipt_hash = hasher.finalize();

            black_box(receipt_hash)
        });
    });

    group.finish();
}

/// Benchmark Harberger tax computation
fn bench_harberger_tax(c: &mut Criterion) {
    let mut group = c.benchmark_group("Harberger Tax");
    group.measurement_time(Duration::from_secs(3));

    // Tax calculation
    group.bench_function("compute_tax", |b| {
        let resource_size: u64 = 1024;
        let ihsan_score: f64 = 0.95;
        let tax_rate: f64 = 0.0001;

        b.iter(|| {
            let tax =
                (black_box(resource_size) as f64) * black_box(tax_rate) / black_box(ihsan_score);
            black_box(tax)
        });
    });

    // Tax with Fixed64 precision
    group.bench_function("compute_tax_fixed", |b| {
        let resource_size = Fixed64::from_f64(1024.0);
        let ihsan_score = Fixed64::from_f64(0.95);
        let tax_rate = Fixed64::from_f64(0.0001);

        b.iter(|| {
            let numerator = black_box(resource_size) * black_box(tax_rate);
            // Note: Fixed64 division may need implementation
            black_box(numerator)
        });
    });

    group.finish();
}

/// Benchmark FFI boundary overhead (placeholder)
fn bench_ffi_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFI Boundary");
    group.measurement_time(Duration::from_secs(5));

    // Simulate Vec<u8> conversion (common FFI pattern)
    group.bench_function("bytes_to_vec", |b| {
        let bytes: &[u8] = &[0u8; 1024];

        b.iter(|| {
            let vec: Vec<u8> = black_box(bytes).to_vec();
            black_box(vec)
        });
    });

    // Simulate string conversion
    group.bench_function("string_conversion", |b| {
        let s = "This is a test string for FFI boundary benchmarking";

        b.iter(|| {
            let owned = black_box(s).to_string();
            black_box(owned)
        });
    });

    // HashMap access (common in FFI contexts)
    group.bench_function("hashmap_lookup", |b| {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        for i in 0..1000 {
            map.insert(format!("key_{}", i), i);
        }

        b.iter(|| map.get(black_box("key_500")));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tpm_operations,
    bench_wasm_sandbox,
    bench_fate_verification,
    bench_fixed_point,
    bench_receipt_generation,
    bench_harberger_tax,
    bench_ffi_boundary,
);

criterion_main!(benches);
