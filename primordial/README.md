# BIZRA Primordial Artifacts

This directory contains foundational artifacts from BIZRA's genesis period.
These files represent the intellectual heritage and design decisions that shaped
the BIZRA DDAGI architecture.

**Preserved**: 2026-01-17
**Source**: `/lost+found/` recovery

## Directory Structure

```
primordial/
├── python_v0.2/           # Complete Python reference implementation
├── research/              # Academic papers and formal analysis
├── docs/                  # Foundation documents and philosophy
└── components/            # UI component prototypes
```

## Artifacts Catalog

### Python v0.2 Reference Implementation (`python_v0.2/`)

The canonical Python implementation of BIZRA Genesis Simulator v0.2.
This serves as the reference for validating the Rust implementation.

Key components:
- `constitution.yaml` - Policy definition with 8D Ihsān weights
- `bizra/fixed.py` - Q32.32 Fixed64 implementation (deterministic math)
- `bizra/ihsan.py` - 8-dimensional excellence scoring
- `bizra/evidence.py` - Evidence envelope with replay guard
- `bizra/node_zero.py` - 8-stage pipeline (SENSE→REASON→SCORE→FATE→SAT→ACT→LEDGER→SNR)
- `bizra/consensus.py` - Shūrā Council (6 validators with veto power)
- `state/` - Persistent state files (replay guard, counters)

### Research Papers (`research/`)

- `2510.16830v1.pdf` - arXiv paper: Foundational theoretical framework
- `BIZRA_Apotheosis_Node_SAPE_Review.pdf` - SAPE methodology review

### Foundation Documents (`docs/`)

- `SAPE_Analysis.txt` - "Cognitive Supremacy Engine & Autopoiesis"
  - Epistemological Crisis and BIZRA Mandate
  - Node-0 Architecture with PAT (Personal Autonomy Team)
  - FATE Gate (Formal Automata for Theology and Ethics)
  - Ihsān Engine formalization
  - 90-day Genesis Sprint roadmap

- `the_word_foundation.txt` - "What is the word mean"
  - BIZRA etymology and naming philosophy
  - Core vocabulary definitions
  - Semantic precision guidelines

- `implementation_notes.txt` - Cognitive Supremacy Engine Implementation
  - Technical implementation details
  - Design decisions and rationale

- `development_history_archive.txt` - Complete development history
  - Archive of 100+ conversation titles covering BIZRA's evolution
  - Historical record of design iterations
  - Captures the journey from concept to implementation

### UI Components (`components/`)

- `sovereign-state.tsx` - Sovereign state dashboard component
  - React/TypeScript implementation
  - Real-time state visualization

## Key Concepts Preserved

### 8-Stage Pipeline
```
SENSE → REASON → SCORE → FATE → SAT → ACT → LEDGER → SNR
```

### 8D Ihsān Scoring
| Dimension | Arabic | Weight |
|-----------|--------|--------|
| Correctness | ʿAdl | 1.2 |
| Safety | Amānah | 1.5 |
| Benefit | Ihsān | 1.3 |
| Efficiency | Hikmah | 1.0 |
| Auditability | Bayān | 1.1 |
| Anti-centralization | Tawḥīd | 1.0 |
| Robustness | Ṣabr | 0.9 |
| Fairness | Mīzān | 1.0 |

### Fixed64 (Q32.32)
Deterministic fixed-point arithmetic for consensus-critical calculations.
No floats in scoring or receipt paths.

### Third Fact Receipts
Cryptographically signed execution logs with:
- JCS (RFC 8785) canonicalization
- Ed25519 signatures
- Nonce-based replay protection

## Relationship to Main Codebase

These primordial artifacts validate and inform:
- `src/receipt_v1.rs` - Receipt schema (matches Python evidence.py)
- `src/nonce_journal.rs` - Replay protection (matches Python replay_guard)
- `crates/bizra-jcs/` - JCS implementation (RFC 8785 compliance)
- `src/fixed.rs` - Fixed64 implementation (matches Python fixed.py)
- `src/ihsan.rs` - Ihsān scoring (matches Python ihsan.py)

## License

These artifacts are part of the BIZRA DDAGI ecosystem.
See root LICENSE for terms.
