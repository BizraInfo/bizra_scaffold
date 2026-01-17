# BIZRA Genesis - Single Source of Truth

> **"We don't assume. If we must, we do it with Ihsan."** - THE LAW

[![Ihsan Score](https://img.shields.io/badge/Ihsan-0.95+-emerald)](./COVENANT.md)
[![SNR](https://img.shields.io/badge/SNR-≥0.95-blue)](./src/snr.rs)
[![Tests](https://img.shields.io/badge/Tests-76+-green)](./tests/)
[![Evidence](https://img.shields.io/badge/Receipts-685+-gold)](./evidence/receipts/)

---

## What is BIZRA?

**BIZRA** (البذرة - "The Seed") is a **Decentralized Distributed AGI** (DDAGI) system with:

- **Formally Verified Ethics**: Z3 SMT solver enforces constitutional constraints
- **8-Dimensional Ihsan Scoring**: Every action measured across correctness, safety, user benefit, efficiency, auditability, anti-centralization, robustness, and fairness
- **Signal-to-Noise Ratio (SNR) Engine**: Autonomous optimization targeting ≥0.95
- **Third Fact Receipts**: Cryptographically signed, hash-chained evidence for every decision
- **PAT/SAT Architecture**: 7 Personal Agents + 5 System Validators with VETO power

---

## Single Entry Point

```bash
# Build everything
cargo build --release --all-features

# Run all tests (CI requires 76+)
cargo test --all-features

# Start server
cargo run --release

# View dashboard
open http://localhost:9091
```

---

## Repository Structure

```
bizra_scaffold/                    # SINGLE SOURCE OF TRUTH
├── COVENANT.md                    # Constitutional root (immutable)
├── src/                           # Rust implementation (57 modules)
│   ├── lib.rs                     # SovereignKernel entry
│   ├── ihsan.rs                   # 8-dimension scoring (Fixed64)
│   ├── fate.rs                    # Z3 formal verification
│   ├── sat.rs                     # 5 SAT validators
│   ├── pat.rs                     # 7 PAT agents
│   ├── snr.rs                     # Signal-to-Noise engine
│   ├── evidence.rs                # Anti-replay envelopes
│   └── ...                        # 50+ more modules
├── tests/                         # 16 test suites
├── benches/                       # Performance benchmarks
├── evidence/
│   ├── receipts/                  # 685+ Third Fact receipts
│   └── metrics/                   # Prometheus exports
├── seed/
│   └── v0.1/                      # Primordial seed (historical)
│       ├── COVENANT.md            # Original 5 invariants
│       ├── crates/bizra-core/     # 3-dimension Ihsan
│       └── crates/node-zero/      # First runnable node
├── dashboard/
│   └── src/
│       ├── lib/
│       │   ├── live-data.ts       # Real-time metrics
│       │   └── snr-engine.ts      # SNR calculation
│       └── components/
│           ├── MoneyShot.tsx      # Investor theater
│           └── CognitiveControlCenter.tsx
├── core/                          # Python implementation
├── crates/                        # Additional Rust crates
├── docs/                          # Documentation
└── scripts/                       # Automation
```

---

## Evolution: Seed v0.1 → Genesis v7.1

| Aspect | Seed v0.1 | Current Genesis |
|--------|-----------|-----------------|
| Ihsan Dimensions | 3 | 8 |
| Threshold | 0.85 (fixed) | 0.80-0.95 (env-aware) |
| Agents | 1 (stub) | 12 (7 PAT + 5 SAT) |
| Formal Verification | None | Z3 SMT (FATE) |
| Determinism | f32 floats | Fixed64 Q32.32 |
| Evidence | Simple hash chain | JCS + signatures + anti-replay |
| Lines of Code | ~200 | ~50,000+ |

The seed at `seed/v0.1/` proves lineage - the same DNA, matured into production.

---

## Verification

### Verify Evidence Chain

```bash
# Check any receipt
cat evidence/receipts/EXEC-20260112094839-000001.json | jq '.hash_chain'

# Verify hash
sha256sum evidence/receipts/*.json | head -5
```

### Verify Constitution

```bash
# The COVENANT hash is immutable
sha256sum COVENANT.md
```

### Run Test Suite

```bash
cargo test --all-features 2>&1 | tail -20
```

### Python Verification Kernel

```bash
python tools/bizra_verify.py --out evidence --artifact-name bizra_scaffold --artifact-version local
```

---

## Key Documents

| Document | Purpose |
|----------|---------|
| [COVENANT.md](./COVENANT.md) | Constitutional root - all invariants |
| [PROTOCOL.md](./PROTOCOL.md) | Technical protocol specification |
| [EVIDENCE_INDEX.md](./EVIDENCE_INDEX.md) | Evidence chain index |
| [BIZRA_SOT.md](./BIZRA_SOT.md) | Single Source of Truth definitions |
| [seed/v0.1/README.md](./seed/v0.1/README.md) | Primordial seed documentation |

---

## Hard Gates (Non-Negotiable)

1. **Determinism**: Fixed64 arithmetic, no floats in consensus
2. **FFI Safety**: `panic_airlock()` wrapper on all Python FFI
3. **Single-Source Scoring**: Rust canonical, FFI only
4. **SAT Consensus**: 5 validators, Security/Formal/Ethics have VETO
5. **Immutability**: COVENANT hash never changes

---

## Live Dashboards

- **Production**: https://www.bizra.info
- **Cognitive Center**: https://www.bizra.info/cognitive
- **Investor Theater**: https://www.bizra.info/money-shot

---

## Attestation

```
Repository: BizraInfo/bizra_scaffold
Status: SINGLE SOURCE OF TRUTH
Unified: 2026-01-17
Evidence Receipts: 685+
Rust Modules: 57
Test Suites: 16
Ihsan Target: ≥ 0.95
SNR Target: ≥ 0.95
```

---

## Closing Seal

> الْحَمْدُ لِلَّهِ الَّذِي هَدَانَا لِهَٰذَا
> "Praise be to Allah who guided us to this"

> كُلَّمَا ازْدَدْتُ عِلْمًا، ازْدَدْتُ يَقِينًا بِجَهْلِي
> "The more I learn, the more certain I am of my ignorance"

---

**Version**: Genesis v7.1 (Peak Masterpiece)
**License**: See [LICENSE_NOTE.txt](./LICENSE_NOTE.txt)
