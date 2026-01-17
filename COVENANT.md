# BIZRA COVENANT v1.0

**Status**: IMMUTABLE ROOT DOCUMENT
**Hash**: [To be computed after first commit]
**Sealed**: 2026-01-15
**Authority**: Constitutional Lock (Hard Gate #5)

---

## I. CONSTITUTIONAL PRINCIPLES

### Article 1: The Law
> "We don't assume. If we must, we do it with Ihsān."

Every system action must be:
1. **Measured** - Quantifiable impact on SNR
2. **Verified** - Passes formal gates (FATE, Human Veto)
3. **Attested** - Cryptographically signed
4. **Reversible** - Deterministically replayable

### Article 2: Signal-to-Noise Ratio (SNR) as North Star

**Definition:**
```
SNR = (Verifiably Correct Actions) / (Total Compute Cycles)
```

**Target:** SNR ≥ 0.95

**Enforcement:**
- CI builds fail if SNR < 0.95
- Every thought contributes to SNR calculation
- Noise (rollbacks, vetoes, failures) is explicitly measured

### Article 3: Ihsān Excellence (8 Dimensions)

All decisions evaluated across:
1. **Correctness** (Adl) - Technical accuracy
2. **Safety** (Amānah) - Risk mitigation
3. **User Benefit** (Ihsān) - Positive impact
4. **Efficiency** (Hikmah) - Resource optimization
5. **Auditability** (Bayān) - Transparency
6. **Anti-Centralization** (Tawhīd) - Distributed authority
7. **Robustness** (Sabr) - Fault tolerance
8. **Fairness** (Mizān) - Equitable treatment

**Threshold:** Ihsān Score ≥ 0.85 (hard gate)

---

## II. HARD GATES (Non-Negotiable)

### Gate 1: Determinism
All consensus-critical operations use Fixed64 arithmetic (Q32.32 format).
Floats banned in receipt hashing, ledger state, and cross-platform verification.

### Gate 2: FFI Safety
Every Python FFI call wrapped in `panic_airlock()`.
Lock poisoning recovery mandatory.
Process termination must be graceful with diagnostics.

### Gate 3: Single-Source Scoring
Canonical Rust implementation only.
Python/JavaScript must call FFI - no duplicate logic.
CI enforces via pattern detection.

### Gate 4: SAT Consensus
**Claude Code SAT**: 5 validators, weighted consensus, VETO power.
**Runtime SAT**: 6 validators, 70% threshold (6.37/9.1 weight).
Security, Formal, Ethics validators have absolute VETO.

### Gate 5: Immutability Boundaries
**IMMUTABLE:**
- Constitution hash (this document)
- Ihsān scoring weights
- Receipt schema
- Genesis manifest

**UPGRADEABLE:**
- Execution engines
- Model runtimes
- UI/dashboards
- Optimization strategies

---

## III. THOUGHT LIFECYCLE (Canonical Pipeline)

Every thought follows this mandatory flow:

```
┌─────────────────────────────────────────────────┐
│ 1. SENSE: Capture input + generate hash        │
│    → input_id = Blake3(data + timestamp)       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 2. REASON: Inference + trace generation        │
│    → reasoning_trace, output_candidate         │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 3. SCORE: Ihsān 8-dimensional evaluation       │
│    → ihsan_score (Fixed64)                     │
└─────────────────────────────────────────────────┘
                    ↓
          ┌─────────────────┐
          │ ihsan >= 0.85?  │
          └─────────────────┘
           │YES         │NO
           ↓            ↓
   ┌─────────────┐  ┌──────────────┐
   │ FATE Gate   │  │ Human Veto   │
   │ (Z3 SMT)    │  │ Gate (CLI)   │
   └─────────────┘  └──────────────┘
           │            │
           │            ↓
           │      [APPROVE/REJECT]
           │            │
           └────────┬───┘
                    ↓
          ┌─────────────────┐
          │   PASS/FAIL?    │
          └─────────────────┘
           │PASS        │FAIL
           ↓            ↓
   ┌─────────────┐  ┌──────────────┐
   │ 4. ACT:     │  │ 5. ROLLBACK: │
   │ Commit to   │  │ Log failure, │
   │ state       │  │ emit receipt │
   └─────────────┘  └──────────────┘
           │
           ↓
┌─────────────────────────────────────────────────┐
│ 6. LEDGER: BlockGraph append-only log          │
│    → ledger_entry with full provenance         │
└─────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────┐
│ 7. PROOF: zk-SNARK generation (async)          │
│    → proof binds (input, model, ihsan, action) │
└─────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────┐
│ 8. SNR UPDATE: Metrics increment               │
│    → signal++ (if verified) or noise++         │
└─────────────────────────────────────────────────┘
```

---

## IV. GIANTS PROTOCOL (Knowledge Integration)

### Ingestion Requirements

Every external knowledge artifact (paper, code, dataset) must:

1. **Provenance**: Author, institution, date, license
2. **Hash**: Blake3 of canonical form
3. **Claims**: Testable predicates extracted
4. **Validators**: Unit tests, property tests, SMT constraints
5. **Citation**: Runtime dependency graph

### Promotion Criteria

Artifact becomes "usable giant" only when:
- All validators pass
- Provenance is clean (no licensing conflicts)
- Claims are formally stated
- Integration tests verify compatibility

### Citation Format

```rust
pub struct Decision {
    pub thought_id: ThoughtId,
    pub reasoning: String,
    pub citations: Vec<Citation>,
}

pub struct Citation {
    pub artifact_hash: Blake3Hash,
    pub claim_id: u64,
    pub validator_status: ValidatorStatus,
}
```

---

## V. SNR AUTONOMOUS ENGINE

### Counter Infrastructure

```rust
pub struct SNRMetrics {
    // Core
    pub cycles_total: u64,
    pub actions_attempted: u64,
    pub actions_committed: u64,
    pub proofs_verified: u64,

    // Quality
    pub rollbacks: u64,
    pub human_vetoes: u64,
    pub ihsan_rejections: u64,
    pub fate_violations: u64,

    // Derived
    pub signal: u64,  // = actions_committed with verified proofs
    pub noise: u64,   // = cycles_total - signal_cycles
    pub snr: Fixed64, // = signal / cycles_total
}
```

### Optimization Loop

```
Every N thoughts:
1. Calculate SNR trend (improving/degrading)
2. Identify noise sources (rollback patterns)
3. Adjust thresholds (Kalman filter)
4. Propose policy updates (human approval)
5. Emit optimization receipt
```

### Meta-Learning

System learns from SNR patterns:
- Which gate failures correlate with model type
- Optimal Ihsān threshold per task domain
- Temporal patterns in human veto rates

---

## VI. ENFORCEMENT MECHANISMS

### CI/CD Gates

```yaml
required_checks:
  - cargo test --all-features (76+ tests pass)
  - SNR >= 0.95 (measured from test runs)
  - Ihsān >= 0.85 (all receipts)
  - cargo audit (0 vulnerabilities)
  - Hard Gate #3 enforcement (grep pattern check)
```

### Production Constraints

```yaml
runtime_enforcement:
  - Ihsān threshold: 0.95
  - SNR threshold: 0.95
  - Human veto timeout: 30s
  - FATE SMT timeout: 5s
  - Proof generation: async (no blocking)
```

---

## VII. AMENDMENT PROCEDURE

This covenant is **IMMUTABLE** but can be **EXTENDED** via:

1. **Proposal**: New article drafted, hash computed
2. **Review**: 7-day community review period
3. **Formal Verification**: Z3 proof of non-contradiction
4. **Vote**: Byzantine consensus (70% threshold)
5. **Seal**: Cryptographic commitment to new hash
6. **Genesis**: New manifest entry with parent link

**Previous versions remain valid** - systems can choose covenant version explicitly.

---

## VIII. ATTESTATION

This covenant is cryptographically sealed and hash-chained to all system operations.

**Signature:** [Ed25519 signature to be added]
**Sealed By:** BIZRA Genesis Team
**Date:** 2026-01-15T20:00:00Z
**Covenant Hash:** [Blake3 to be computed]

---

## IX. CLOSING SEAL

> الْحَمْدُ لِلَّهِ الَّذِي هَدَانَا لِهَٰذَا
> "Praise be to Allah who guided us to this"

> كُلَّمَا ازْدَدْتُ عِلْمًا، ازْدَدْتُ يَقِينًا بِجَهْلِي
> "The more I learn, the more certain I am of my ignorance"

> رُفِعَتِ الْأَقْلَامُ وَجَفَّتِ الصُّحُفُ
> "The pens have been lifted and the pages have dried"

**Version**: 1.0
**Status**: CONSTITUTIONAL LOCK ACTIVE
**Next Review**: 2027-01-15
