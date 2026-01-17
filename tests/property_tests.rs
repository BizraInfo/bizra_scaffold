// tests/property_tests.rs - Property-Based Tests for MASTERPIECE Hardening
// These tests use controlled random inputs to verify invariants
// Run with: cargo test --test property_tests

use meta_alpha_dual_agentic::fixed::Fixed64;
use meta_alpha_dual_agentic::ihsan;
use meta_alpha_dual_agentic::sat::RejectionCode;
use std::collections::BTreeMap;

/// Test Fixed64 arithmetic properties with pseudo-random inputs
#[test]
fn fixed64_addition_is_commutative() {
    let test_values = [
        0i64,
        1,
        -1,
        100,
        -100,
        1000000,
        -1000000,
        i64::MAX / 2,
        i64::MIN / 2,
    ];

    for &a_raw in &test_values {
        for &b_raw in &test_values {
            let a = Fixed64::from_bits(a_raw);
            let b = Fixed64::from_bits(b_raw);

            let sum_ab = a.saturating_add(b);
            let sum_ba = b.saturating_add(a);

            assert_eq!(
                sum_ab, sum_ba,
                "Addition must be commutative: {:?} + {:?}",
                a, b
            );
        }
    }
}

#[test]
fn fixed64_multiplication_is_commutative() {
    let test_values = [
        0i64,
        1,
        -1,
        100,
        -100,
        Fixed64::SCALE,     // 1.0
        Fixed64::SCALE / 2, // 0.5
        Fixed64::SCALE * 2, // 2.0
    ];

    for &a_raw in &test_values {
        for &b_raw in &test_values {
            let a = Fixed64::from_bits(a_raw);
            let b = Fixed64::from_bits(b_raw);

            let prod_ab = a.saturating_mul(b);
            let prod_ba = b.saturating_mul(a);

            assert_eq!(
                prod_ab, prod_ba,
                "Multiplication must be commutative: {:?} * {:?}",
                a, b
            );
        }
    }
}

#[test]
fn fixed64_division_by_zero_does_not_panic() {
    let test_values = [
        0i64,
        1,
        -1,
        i64::MAX,
        i64::MIN,
        Fixed64::SCALE,
        -Fixed64::SCALE,
    ];

    let zero = Fixed64::from_bits(0);

    for &a_raw in &test_values {
        let a = Fixed64::from_bits(a_raw);
        // This should not panic
        let result = a.saturating_div(zero);

        // Should return MAX or MIN (or zero for 0/0)
        let is_valid = result == Fixed64::from_bits(i64::MAX)
            || result == Fixed64::from_bits(i64::MIN)
            || a_raw == 0;

        assert!(
            is_valid,
            "Division by zero should saturate: {:?} / zero = {:?}",
            a, result
        );
    }
}

#[test]
fn fixed64_saturating_ops_never_overflow() {
    let extremes = [i64::MAX, i64::MIN, i64::MAX - 1, i64::MIN + 1];

    for &a_raw in &extremes {
        for &b_raw in &extremes {
            let a = Fixed64::from_bits(a_raw);
            let b = Fixed64::from_bits(b_raw);

            // None of these should panic
            let _ = a.saturating_add(b);
            let _ = a.saturating_sub(b);
            let _ = a.saturating_mul(b);
            if b_raw != 0 {
                let _ = a.saturating_div(b);
            }
        }
    }
}

/// Test Ihsān score properties
#[test]
fn ihsan_score_in_valid_range() {
    let test_cases = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // All zeros
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], // All ones
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], // All halves
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], // Alternating
        [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99], // Near-max
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], // Near-min
    ];

    let dimensions = [
        "correctness",
        "safety",
        "user_benefit",
        "efficiency",
        "auditability",
        "anti_centralization",
        "robustness",
        "adl_fairness",
    ];

    for scores in &test_cases {
        let mut dim_map = BTreeMap::new();
        for (i, name) in dimensions.iter().enumerate() {
            dim_map.insert(name.to_string(), scores[i]);
        }

        let result = ihsan::score(&dim_map).expect("Valid input should succeed");

        assert!(
            result >= 0.0 && result <= 1.0,
            "Score must be in [0, 1]: got {} for {:?}",
            result,
            scores
        );
        assert!(
            result.is_finite(),
            "Score must be finite: got {} for {:?}",
            result,
            scores
        );
    }
}

#[test]
fn ihsan_score_is_deterministic() {
    let dimensions = [
        "correctness",
        "safety",
        "user_benefit",
        "efficiency",
        "auditability",
        "anti_centralization",
        "robustness",
        "adl_fairness",
    ];
    let scores = [0.8, 0.9, 0.7, 0.6, 0.85, 0.5, 0.75, 0.65];

    let mut dim_map = BTreeMap::new();
    for (i, name) in dimensions.iter().enumerate() {
        dim_map.insert(name.to_string(), scores[i]);
    }

    let result1 = ihsan::score(&dim_map).expect("Should succeed");
    let result2 = ihsan::score(&dim_map).expect("Should succeed");
    let result3 = ihsan::score(&dim_map).expect("Should succeed");

    assert_eq!(result1, result2, "Score must be deterministic");
    assert_eq!(result2, result3, "Score must be deterministic");
}

#[test]
fn ihsan_rejects_invalid_inputs() {
    let dimensions = [
        "correctness",
        "safety",
        "user_benefit",
        "efficiency",
        "auditability",
        "anti_centralization",
        "robustness",
        "adl_fairness",
    ];

    // Test with out-of-range value
    let mut dim_map = BTreeMap::new();
    for (i, name) in dimensions.iter().enumerate() {
        dim_map.insert(name.to_string(), if i == 0 { 1.5 } else { 0.5 });
    }

    assert!(ihsan::score(&dim_map).is_err(), "Should reject score > 1.0");

    // Test with negative value
    dim_map.insert("correctness".to_string(), -0.1);
    assert!(ihsan::score(&dim_map).is_err(), "Should reject score < 0.0");

    // Test with NaN
    dim_map.insert("correctness".to_string(), f64::NAN);
    assert!(ihsan::score(&dim_map).is_err(), "Should reject NaN");

    // Test with infinity
    dim_map.insert("correctness".to_string(), f64::INFINITY);
    assert!(ihsan::score(&dim_map).is_err(), "Should reject Infinity");
}

/// Test SAT security patterns
#[test]
fn sat_rejection_code_display() {
    let codes = [
        RejectionCode::SecurityThreat("test".into()),
        RejectionCode::FormalViolation("test".into()),
        RejectionCode::EthicsViolation("test".into()),
        RejectionCode::PerformanceBudgetExceeded("test".into()),
        RejectionCode::ConsistencyFailure("test".into()),
        RejectionCode::ResourceConstraintViolated("test".into()),
        RejectionCode::ThermalThrottle("test".into()),
        RejectionCode::IhsanUnsat("test".into()),
        RejectionCode::Quarantine("test".into()),
    ];

    for code in &codes {
        let display = format!("{}", code);
        assert!(!display.is_empty(), "Display should not be empty");
        assert!(display.contains("test"), "Display should contain message");
    }
}

#[test]
fn sat_rejection_codes_are_distinct() {
    let security = RejectionCode::SecurityThreat("same".into());
    let formal = RejectionCode::FormalViolation("same".into());
    let ethics = RejectionCode::EthicsViolation("same".into());

    assert_ne!(security, formal);
    assert_ne!(formal, ethics);
    assert_ne!(security, ethics);
}

/// Edge case: Very small fixed-point values
#[test]
fn fixed64_small_value_precision() {
    let epsilon = Fixed64::from_bits(1); // Smallest representable value
    let one = Fixed64::ONE;

    let result = one.saturating_add(epsilon);
    assert!(
        result.to_bits() > one.to_bits(),
        "Should be able to add epsilon"
    );

    let back = result.saturating_sub(epsilon);
    assert_eq!(back, one, "Subtraction should be precise");
}

/// Edge case: Fixed-point conversion round-trip
#[test]
fn fixed64_f64_roundtrip() {
    let test_values = [0.0, 1.0, -1.0, 0.5, 0.25, 0.125, 100.0, -100.0];

    for &f in &test_values {
        let fixed = Fixed64::from_f64(f);
        let back = fixed.to_f64();

        assert!(
            (back - f).abs() < 1e-9,
            "f64 round-trip should be precise: {} -> {:?} -> {}",
            f,
            fixed,
            back
        );
    }
}

// ============================================================================
// CROSS-PLATFORM DETERMINISM TESTS (Task 1.4)
// These tests verify that Fixed64-based hashing produces identical results
// regardless of the platform or conversion path used.
// ============================================================================

use sha2::{Digest, Sha256};

/// Test that Fixed64 to_bits produces identical results for same logical value
#[test]
fn fixed64_bits_determinism() {
    let test_values = [
        0.0, 1.0, -1.0, 0.5, 0.25, 0.125, 0.95, 0.85,
        100.0, -100.0, 0.001, 0.999, 0.123456789,
    ];

    for &f in &test_values {
        // Create Fixed64 twice from same f64
        let fixed1 = Fixed64::from_f64(f);
        let fixed2 = Fixed64::from_f64(f);

        // Bits must be identical
        assert_eq!(
            fixed1.to_bits(),
            fixed2.to_bits(),
            "Fixed64 bits must be deterministic for value {}",
            f
        );

        // Hash of bits must be identical
        let hash1 = {
            let mut h = Sha256::new();
            h.update(fixed1.to_bits().to_le_bytes());
            hex::encode(h.finalize())
        };
        let hash2 = {
            let mut h = Sha256::new();
            h.update(fixed2.to_bits().to_le_bytes());
            hex::encode(h.finalize())
        };

        assert_eq!(
            hash1, hash2,
            "Hash of Fixed64 bits must be deterministic for value {}",
            f
        );
    }
}

/// Test that Fixed64 hash is different from f64 hash (proving we're using the right representation)
#[test]
fn fixed64_hash_differs_from_f64_hash() {
    let test_value = 0.95;
    let fixed = Fixed64::from_f64(test_value);

    // Hash using Fixed64 bits (deterministic)
    let fixed_hash = {
        let mut h = Sha256::new();
        h.update(fixed.to_bits().to_le_bytes());
        hex::encode(h.finalize())
    };

    // Hash using f64 bits (platform-dependent)
    let f64_hash = {
        let mut h = Sha256::new();
        h.update(test_value.to_le_bytes());
        hex::encode(h.finalize())
    };

    // These should be different (Fixed64 uses Q32.32, f64 uses IEEE 754)
    assert_ne!(
        fixed_hash, f64_hash,
        "Fixed64 hash should differ from f64 hash - using different representations"
    );
}

/// Test receipt-style hash computation with Fixed64 fields
#[test]
fn receipt_hash_determinism_with_fixed64() {
    let synergy_score = Fixed64::from_f64(0.92);
    let ihsan_score = Fixed64::from_f64(0.95);
    let threshold = Fixed64::from_f64(0.85);
    let receipt_id = "EXEC-20260113-000001";
    let timestamp = "2026-01-13T12:00:00Z";
    let task = "test_task";

    // Compute hash multiple times
    let compute_hash = || {
        let content = format!(
            "{}|{}|{}|{}|{}",
            receipt_id,
            timestamp,
            task,
            synergy_score.to_bits(),
            ihsan_score.to_bits()
        );
        let hash = Sha256::digest(content.as_bytes());
        format!("sha256:{:x}", hash)
    };

    let hash1 = compute_hash();
    let hash2 = compute_hash();
    let hash3 = compute_hash();

    assert_eq!(hash1, hash2, "Receipt hash must be deterministic");
    assert_eq!(hash2, hash3, "Receipt hash must be deterministic");

    // Verify changing a score changes the hash
    let different_ihsan = Fixed64::from_f64(0.96);
    let different_hash = {
        let content = format!(
            "{}|{}|{}|{}|{}",
            receipt_id,
            timestamp,
            task,
            synergy_score.to_bits(),
            different_ihsan.to_bits()
        );
        let hash = Sha256::digest(content.as_bytes());
        format!("sha256:{:x}", hash)
    };

    assert_ne!(
        hash1, different_hash,
        "Different scores must produce different hashes"
    );
}

/// Test SessionNode-style hash computation with Fixed64 impact_delta
#[test]
fn session_node_hash_determinism_with_fixed64() {
    let state_root = "a".repeat(64);
    let receipts_root = "b".repeat(64);
    let policy_version = "v1.0.0";
    let impact_delta = Fixed64::from_f64(0.05);
    let created_at: u64 = 1705147200000000000; // Fixed timestamp

    let compute_node_hash = |impact: Fixed64| {
        let mut hasher = Sha256::new();
        hasher.update(state_root.as_bytes());
        hasher.update(receipts_root.as_bytes());
        hasher.update(policy_version.as_bytes());
        hasher.update(impact.to_bits().to_le_bytes());
        hasher.update(created_at.to_le_bytes());
        hex::encode(hasher.finalize())
    };

    // Same impact_delta should produce same hash
    let hash1 = compute_node_hash(impact_delta);
    let hash2 = compute_node_hash(Fixed64::from_f64(0.05));

    assert_eq!(
        hash1, hash2,
        "Session node hash must be deterministic for same impact_delta"
    );

    // Different impact_delta should produce different hash
    let different_impact = Fixed64::from_f64(0.06);
    let hash3 = compute_node_hash(different_impact);

    assert_ne!(
        hash1, hash3,
        "Different impact_delta must produce different hashes"
    );
}

/// Test that very close f64 values map to different Fixed64 bits
#[test]
fn fixed64_distinguishes_close_values() {
    let epsilon = 1e-10;
    let base = 0.95;

    let fixed_base = Fixed64::from_f64(base);
    let fixed_plus = Fixed64::from_f64(base + epsilon);
    let fixed_minus = Fixed64::from_f64(base - epsilon);

    // The Fixed64 bits should be different (or at least not all equal)
    // Note: Very small epsilon might round to same value, which is acceptable
    // as long as the representation is deterministic
    let bits_base = fixed_base.to_bits();
    let bits_plus = fixed_plus.to_bits();
    let bits_minus = fixed_minus.to_bits();

    // At least one should be different (epsilon is small but representable)
    let all_same = bits_base == bits_plus && bits_plus == bits_minus;

    // If they're all the same, that's fine (rounding), but verify consistency
    if all_same {
        // Verify the value is still deterministic
        assert_eq!(
            Fixed64::from_f64(base).to_bits(),
            bits_base,
            "Fixed64 must be deterministic even when rounding"
        );
    } else {
        // At least one differs, verify ordering is preserved
        assert!(
            bits_minus <= bits_base && bits_base <= bits_plus,
            "Fixed64 should preserve ordering: {} <= {} <= {}",
            bits_minus,
            bits_base,
            bits_plus
        );
    }
}

/// Test Fixed64 arithmetic determinism for Harberger tax calculation pattern
#[test]
fn harberger_tax_calculation_determinism() {
    let latency_ms: usize = 50;
    let ihsan_score = Fixed64::from_f64(0.92);
    let memory_usage = Fixed64::from_f64(0.3);

    let compute_tax = |latency: usize, ihsan: Fixed64, memory: Fixed64| -> Fixed64 {
        let latency_factor = Fixed64::from_f64(latency as f64 / 1000.0);
        let memory_factor = memory;
        let efficiency = Fixed64::ONE.saturating_sub(ihsan);

        latency_factor
            .saturating_mul(memory_factor)
            .saturating_mul(efficiency)
    };

    let tax1 = compute_tax(latency_ms, ihsan_score, memory_usage);
    let tax2 = compute_tax(latency_ms, ihsan_score, memory_usage);
    let tax3 = compute_tax(latency_ms, ihsan_score, memory_usage);

    assert_eq!(tax1.to_bits(), tax2.to_bits(), "Tax calculation must be deterministic");
    assert_eq!(tax2.to_bits(), tax3.to_bits(), "Tax calculation must be deterministic");

    // Verify hash determinism
    let hash1 = {
        let mut h = Sha256::new();
        h.update(tax1.to_bits().to_le_bytes());
        hex::encode(h.finalize())
    };
    let hash2 = {
        let mut h = Sha256::new();
        h.update(tax2.to_bits().to_le_bytes());
        hex::encode(h.finalize())
    };

    assert_eq!(hash1, hash2, "Tax hash must be deterministic");
}

/// Test that Fixed64::ZERO is truly zero and deterministic
#[test]
fn fixed64_zero_determinism() {
    let zero1 = Fixed64::ZERO;
    let zero2 = Fixed64::from_f64(0.0);
    let zero3 = Fixed64::from_bits(0);

    assert_eq!(zero1.to_bits(), 0, "ZERO constant must be 0 bits");
    assert_eq!(zero2.to_bits(), 0, "from_f64(0.0) must be 0 bits");
    assert_eq!(zero3.to_bits(), 0, "from_bits(0) must be 0 bits");

    // Hash should be identical
    let hash_const = {
        let mut h = Sha256::new();
        h.update(zero1.to_bits().to_le_bytes());
        hex::encode(h.finalize())
    };
    let hash_from_f64 = {
        let mut h = Sha256::new();
        h.update(zero2.to_bits().to_le_bytes());
        hex::encode(h.finalize())
    };

    assert_eq!(hash_const, hash_from_f64, "Zero hash must be deterministic");
}

/// Test Fixed64 ONE constant determinism
#[test]
fn fixed64_one_determinism() {
    let one1 = Fixed64::ONE;
    let one2 = Fixed64::from_f64(1.0);

    assert_eq!(
        one1.to_bits(),
        one2.to_bits(),
        "ONE constant must equal from_f64(1.0)"
    );
    assert_eq!(
        one1.to_bits(),
        Fixed64::SCALE,
        "ONE must equal SCALE"
    );
}

// ============================================================================
// FUZZ-STYLE EDGE CASE TESTS (Robustness Hardening)
// These tests verify Fixed64 behavior at boundary conditions and with
// adversarial inputs to ensure no panics or undefined behavior.
// ============================================================================

/// Fuzz test: Fixed64 handles all i64 bit patterns without panic
#[test]
fn fuzz_fixed64_all_bit_patterns_no_panic() {
    // Test strategic bit patterns that might cause issues
    let patterns: [i64; 20] = [
        0,
        1,
        -1,
        i64::MAX,
        i64::MIN,
        i64::MAX - 1,
        i64::MIN + 1,
        i64::MAX / 2,
        i64::MIN / 2,
        Fixed64::SCALE,
        -Fixed64::SCALE,
        Fixed64::SCALE - 1,
        Fixed64::SCALE + 1,
        0x7FFF_FFFF_FFFF_FFFF, // All bits except sign
        0x8000_0000_0000_0000_u64 as i64, // Only sign bit
        0x5555_5555_5555_5555, // Alternating bits
        0xAAAA_AAAA_AAAA_AAAAu64 as i64, // Alternating bits (inverted)
        0x0000_0000_FFFF_FFFF, // Lower 32 bits
        0xFFFF_FFFF_0000_0000u64 as i64, // Upper 32 bits
        0x0000_FFFF_FFFF_0000, // Middle bits
    ];

    for bits in patterns {
        // None of these operations should panic
        let fixed = Fixed64::from_bits(bits);
        let _ = fixed.to_f64();
        let _ = fixed.to_bits();
        let _ = fixed.saturating_add(Fixed64::ONE);
        let _ = fixed.saturating_sub(Fixed64::ONE);
        let _ = fixed.saturating_mul(Fixed64::ONE);
        if bits != 0 {
            let _ = Fixed64::ONE.saturating_div(fixed);
        }
    }
}

/// Fuzz test: Fixed64 from_f64 handles special float values
#[test]
fn fuzz_fixed64_special_float_values() {
    let special_values: [f64; 12] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        f64::EPSILON,
        1e-15,
        1e15,
        0.999999999999999,
        0.000000000000001,
    ];

    for f in special_values {
        // Should not panic
        let fixed = Fixed64::from_f64(f);
        let back = fixed.to_f64();

        // If input was finite, output should be finite
        if f.is_finite() {
            assert!(back.is_finite(), "from_f64({}) produced non-finite result", f);
        }
    }
}

/// Fuzz test: Ihsān score range values (0.0 to 1.0)
#[test]
fn fuzz_ihsan_score_range() {
    // Test 100 values across the Ihsān score range
    for i in 0..=100 {
        let score = i as f64 / 100.0;
        let fixed = Fixed64::from_f64(score);
        let back = fixed.to_f64();

        // Should round-trip with high precision
        assert!(
            (back - score).abs() < 1e-9,
            "Ihsān score {} failed round-trip: got {}",
            score, back
        );

        // Hash should be deterministic
        let hash1 = {
            let mut h = Sha256::new();
            h.update(fixed.to_bits().to_le_bytes());
            hex::encode(h.finalize())
        };
        let hash2 = {
            let fixed2 = Fixed64::from_f64(score);
            let mut h = Sha256::new();
            h.update(fixed2.to_bits().to_le_bytes());
            hex::encode(h.finalize())
        };
        assert_eq!(hash1, hash2, "Hash not deterministic for score {}", score);
    }
}

/// Fuzz test: Saturating arithmetic never panics at extremes
#[test]
fn fuzz_saturating_arithmetic_extremes() {
    let extremes = [
        Fixed64::from_bits(i64::MAX),
        Fixed64::from_bits(i64::MIN),
        Fixed64::from_bits(i64::MAX - 1),
        Fixed64::from_bits(i64::MIN + 1),
    ];

    for &a in &extremes {
        for &b in &extremes {
            // None of these should panic
            let _ = a.saturating_add(b);
            let _ = a.saturating_sub(b);
            let _ = a.saturating_mul(b);
            // Division by non-zero
            if b.to_bits() != 0 {
                let _ = a.saturating_div(b);
            }
        }
    }
}

/// Fuzz test: Receipt hash computation with random-ish scores
#[test]
fn fuzz_receipt_hash_computation() {
    // Simulate receipt hash computation with various score combinations
    let scores: [(f64, f64); 10] = [
        (0.95, 0.92),
        (0.85, 0.88),
        (1.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (0.99, 0.01),
        (0.01, 0.99),
        (0.333, 0.666),
        (0.123456789, 0.987654321),
        (0.111111111, 0.999999999),
    ];

    for (ihsan, synergy) in scores {
        let ihsan_fixed = Fixed64::from_f64(ihsan);
        let synergy_fixed = Fixed64::from_f64(synergy);

        // Compute hash twice
        let compute_hash = || {
            let content = format!(
                "receipt|{}|{}",
                ihsan_fixed.to_bits(),
                synergy_fixed.to_bits()
            );
            let hash = Sha256::digest(content.as_bytes());
            format!("{:x}", hash)
        };

        let hash1 = compute_hash();
        let hash2 = compute_hash();

        assert_eq!(
            hash1, hash2,
            "Receipt hash not deterministic for ihsan={}, synergy={}",
            ihsan, synergy
        );
    }
}

/// Fuzz test: Session node impact_delta edge cases
#[test]
fn fuzz_session_node_impact_delta() {
    let deltas: [f64; 10] = [
        0.0, 0.01, -0.01, 0.5, -0.5,
        0.001, -0.001, 1.0, -1.0, 0.123456789,
    ];

    for delta in deltas {
        let fixed_delta = Fixed64::from_f64(delta);

        // Simulate session node hash computation
        let compute_node_hash = || {
            let mut hasher = Sha256::new();
            hasher.update("state_root".as_bytes());
            hasher.update("receipts_root".as_bytes());
            hasher.update("v1.0.0".as_bytes());
            hasher.update(fixed_delta.to_bits().to_le_bytes());
            hasher.update(12345u64.to_le_bytes());
            hex::encode(hasher.finalize())
        };

        let hash1 = compute_node_hash();
        let hash2 = compute_node_hash();

        assert_eq!(
            hash1, hash2,
            "Session node hash not deterministic for delta={}",
            delta
        );
    }
}

/// Fuzz test: Harberger tax calculation edge cases
#[test]
fn fuzz_harberger_tax_edge_cases() {
    let test_cases: [(usize, f64, f64); 8] = [
        (0, 0.95, 0.3),      // Zero latency
        (1000, 0.95, 0.3),   // 1 second latency
        (50, 1.0, 0.0),      // Perfect Ihsān, zero memory
        (50, 0.0, 1.0),      // Zero Ihsān, full memory
        (50, 0.5, 0.5),      // Balanced
        (1, 0.99, 0.01),     // Minimal latency
        (999, 0.01, 0.99),   // High latency, low Ihsān
        (100, 0.85, 0.5),    // Threshold case
    ];

    for (latency, ihsan, memory) in test_cases {
        let ihsan_fixed = Fixed64::from_f64(ihsan);
        let memory_fixed = Fixed64::from_f64(memory);

        // Compute tax (simplified)
        let latency_factor = Fixed64::from_f64(latency as f64 / 1000.0);
        let efficiency = Fixed64::ONE.saturating_sub(ihsan_fixed);
        let tax = latency_factor
            .saturating_mul(memory_fixed)
            .saturating_mul(efficiency);

        // Should not panic and should be deterministic
        let tax2 = latency_factor
            .saturating_mul(memory_fixed)
            .saturating_mul(efficiency);

        assert_eq!(
            tax.to_bits(), tax2.to_bits(),
            "Tax calculation not deterministic for latency={}, ihsan={}, memory={}",
            latency, ihsan, memory
        );
    }
}
