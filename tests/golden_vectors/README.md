# BIZRA AEON OMEGA - Golden Vectors for Determinism Testing
# ═══════════════════════════════════════════════════════════════════════════════
#
# This directory contains golden vectors (expected input/output pairs) for
# verifying deterministic behavior of the BIZRA cognitive system.
#
# Reproducibility is a core Ihsān principle (أمانة - Trustworthiness)
#
# Structure:
#   golden_vectors/
#   ├── README.md                    # This file
#   ├── manifest.yaml                # Index of all vectors with metadata
#   ├── snr_scoring/                 # SNR computation vectors
#   ├── got_reasoning/               # Graph-of-Thoughts vectors
#   ├── ethical_constraints/         # Ihsān metric vectors
#   ├── thermodynamic/               # Energy computation vectors
#   ├── value_oracle/                # Oracle convergence vectors
#   └── integration/                 # End-to-end pipeline vectors
#
# Vector Format:
#   Each vector is a JSON file with:
#   - input: The input to the function/system
#   - expected_output: The exact expected output
#   - tolerance: Optional numeric tolerance for floating-point comparisons
#   - hash: SHA-256 of the expected output for integrity
#   - description: Human-readable description
#   - created_at: ISO timestamp
#   - version: Vector format version
#
# Usage:
#   python -m pytest tests/golden_vectors/ -v
#   bizra-core verify --mode determinism
#
# Adding New Vectors:
#   1. Run the function with known-good code
#   2. Capture the output
#   3. Create a JSON file with input/expected_output
#   4. Compute the hash: sha256(json.dumps(expected_output, sort_keys=True))
#   5. Add to manifest.yaml
#
# ═══════════════════════════════════════════════════════════════════════════════

## Vector Categories

### 1. SNR Scoring Vectors (snr_scoring/)

Test deterministic Signal-to-Noise Ratio computation:

- `snr_001_basic_scoring.json` - Basic SNR computation with simple inputs
- `snr_002_high_signal.json` - High signal, low noise case
- `snr_003_edge_case_zero_noise.json` - Edge case with zero noise

### 2. Graph-of-Thoughts Vectors (got_reasoning/)

Test deterministic thought graph construction and evaluation:

- `got_001_linear_chain.json` - Simple linear thought chain
- `got_002_branching.json` - Branching thought exploration
- `got_003_convergence.json` - Multiple paths converging

### 3. Ethical Constraint Vectors (ethical_constraints/)

Test Ihsān metric enforcement:

- `eth_001_high_ihsan.json` - High ethical score acceptance
- `eth_002_low_ihsan_rejection.json` - Low ethical score rejection
- `eth_003_boundary_case.json` - Edge case at threshold

### 4. Thermodynamic Vectors (thermodynamic/)

Test energy computation and entropy:

- `thermo_001_carnot_efficiency.json` - Carnot cycle efficiency
- `thermo_002_entropy_change.json` - Entropy computation
- `thermo_003_energy_conservation.json` - Energy balance

### 5. Value Oracle Vectors (value_oracle/)

Test pluralistic oracle convergence:

- `oracle_001_unanimous.json` - All oracles agree
- `oracle_002_disagreement.json` - Oracles with disagreement
- `oracle_003_weights.json` - Weighted oracle combination

### 6. Integration Vectors (integration/)

End-to-end pipeline tests:

- `int_001_full_pipeline.json` - Complete observation processing
- `int_002_with_caching.json` - Cached intermediate results
- `int_003_error_recovery.json` - Error handling paths

## Verification Process

```bash
# Run all golden vector tests
python -m pytest tests/test_golden_vectors.py -v

# Run specific category
python -m pytest tests/test_golden_vectors.py -k "snr" -v

# Generate new vectors (development mode)
python scripts/generate_golden_vectors.py --category snr --count 5

# Verify vector integrity
python scripts/verify_vector_integrity.py
```

## Reproducibility Guarantees

All golden vectors are tested with:

1. **Fixed random seeds** - `random.seed(42)`, `numpy.random.seed(42)`
2. **Sorted dictionary keys** - Consistent JSON serialization
3. **Floating-point tolerance** - Configurable epsilon for comparisons
4. **Hash verification** - SHA-256 integrity check on expected outputs
5. **Environment recording** - Python version, package versions captured

## Failure Handling

If a golden vector test fails:

1. **Check for intentional changes** - Was the algorithm improved?
2. **Verify environment** - Same Python version, dependencies?
3. **Check floating-point precision** - Adjust tolerance if needed
4. **Regenerate if valid** - Update vector if change is correct

## Versioning

Vector format version: 1.0.0

Changes to vector format require:
- Version bump
- Migration script
- Documentation update
