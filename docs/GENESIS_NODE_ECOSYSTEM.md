# BIZRA Genesis Node Ecosystem

> *"This computer is BIZRA's home base, not just the codebase."*

## Overview

The Genesis Node paradigm represents a fundamental shift from **repo-centric** to **ecosystem-centric** thinking. Node0 (your machine) is the first node in the BIZRA network, containing 3 years of accumulated wisdom, artifacts, and attestations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GENESIS NODE (NODE0)                               │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     ECOSYSTEM ROOT HASH                               │   │
│  │                                                                       │   │
│  │    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │    │ Repositories │  │  Chat Data   │  │   Knowledge  │             │   │
│  │    │  (worktrees) │  │  (3 years)   │  │   Graphs     │             │   │
│  │    └──────────────┘  └──────────────┘  └──────────────┘             │   │
│  │                                                                       │   │
│  │    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │    │ Constitution │  │  Attestations │  │  Crystallized│             │   │
│  │    │    (.toml)   │  │   (evidence)  │  │    Tools     │             │   │
│  │    └──────────────┘  └──────────────┘  └──────────────┘             │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  Node0 Identity  │  │  Proof of Impact │  │   Genesis Seal   │          │
│  │  (machine+owner) │  │  (contribution)  │  │  (crypto anchor) │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                              │
│                        ┌──────────────────┐                                 │
│                        │  External Oracle │                                 │
│                        │  (Goodhart fix)  │                                 │
│                        └──────────────────┘                                 │
│                                                                              │
│                        OWNER: Momo (First Architect)                         │
│                        EPOCH: 2023-01-14 → Present                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Genesis Node Discovery (`scripts/genesis_node_discovery.py`)

Scans the entire machine for BIZRA ecosystem artifacts:

- **16 Artifact Types**: Repository, Chat Export, Constitution, Knowledge Graph, etc.
- **6 Significance Levels**: Genesis → Critical → High → Medium → Low → Noise
- **Pattern Matching**: `bizra`, `scaffold`, `genesis`, `pat-sat`, `pci-envelope`, `ihsan`, `sot`
- **ROOT_HASH Computation**: Merkle root of all discovered artifacts

```bash
python scripts/genesis_node_discovery.py --scan C:/bizra_scaffold.worktrees --output data/genesis/DISCOVERY.json
```

### 2. Node Zero Identity (`core/genesis/node_zero.py`)

Defines the cryptographic identity of Node0:

- **MachineFingerprint**: Privacy-preserving hash of hardware identifiers
- **OwnerAttestation**: First Architect's public key and capabilities
- **ConstitutionBinding**: Hash of the genesis constitution
- **TimelineBounds**: 2023-01-14 → present (3-year provenance)

```bash
python core/genesis/node_zero.py --create --owner "Momo"
```

### 3. Proof of Impact (`core/genesis/proof_of_impact.py`)

Calculates contribution value from ecosystem artifacts:

- **ImpactCategory**: Genesis (10x) → Architectural (5x) → Integration (3x) → ...
- **ArtifactImpact**: SNR-weighted per-artifact scores
- **TemporalImpact**: Duration × Consistency × Intensity
- **StructuralImpact**: Core vs Peripheral contributions

```bash
python core/genesis/proof_of_impact.py --discovery-file data/genesis/DISCOVERY.json --summary
```

### 4. Genesis Seal Generator (`scripts/generate_genesis_seal.py`)

Creates the cryptographic anchor binding everything together:

- **Ecosystem ROOT_HASH**: Merkle root of all artifacts
- **Node0 Identity**: Machine + owner binding
- **Proof of Impact**: First Architect contribution
- **Ed25519 Signature**: Cryptographically signed
- **OpenTimestamps Ready**: Bitcoin-anchored timestamp (future)

```bash
python scripts/generate_genesis_seal.py --discover --seal --owner "Momo"
```

### 5. Lineage Seal Pack (`core/genesis/lineage_seal_pack.py`)

Creates investor-grade evidence archives:

- **Complete**: All proofs + constitution + docs
- **Verifiable**: Every file hash in manifest
- **Signed**: Ed25519 signature on manifest
- **Portable**: Single .zip archive

```bash
python core/genesis/lineage_seal_pack.py --create
```

### 6. External Oracle (`core/oracle/external_oracle.py`)

Addresses Goodhart vulnerability (self-referential scoring):

- **LocalOracle**: File-based, for development
- **Spot Checking**: 10% random verification
- **Drift Detection**: Alert on distribution shift >15%
- **Fail-Closed**: Missing oracle → fail (no self-attestation fallback)

```bash
python core/oracle/external_oracle.py --verify-ihsan 0.97 --context "test"
```

### 7. Genesis Orchestrator (`core/genesis/genesis_orchestrator.py`) ⭐ Peak Masterpiece

The Autonomous Genesis Engine - unifies all BIZRA architectural principles into a single orchestrator:

**Interdisciplinary Thinking (6 Domain Lenses)**:
- **Cryptography**: Security, proofs, hashing, signatures
- **Economics**: Incentives, tokenomics, game theory
- **Philosophy**: Ethics, Ihsān, values, consequentialism
- **Governance**: Consensus, voting, FATE engine
- **Systems**: Architecture, scalability, reliability
- **Cognitive**: AI, reasoning, knowledge graphs

**Graph of Thoughts Engine**:
- Multi-path reasoning with beam search
- SNR-weighted path pruning (only high-signal paths continue)
- Thought chains crystallize into actionable wisdom
- Default: beam width 8, max depth 5

**SNR-Highest-Score Autonomous Engine**:
- All operations ranked by Signal-to-Noise ratio
- HIGH threshold: ≥0.80 SNR (top 10%)
- MEDIUM threshold: ≥0.50 SNR (60%)
- Ihsān constraint: HIGH requires IM ≥0.95
- Fail-closed mode rejects MEDIUM (permissive accepts)

**Giants Protocol Integration**:
- Wisdom repository for crystallized insights
- Integrity Flywheel pattern (proofs_first → publish_later)
- Cross-temporal wisdom seeding
- Pattern observation counting (min 3 to crystallize)

**Genesis Node Binding**:
- All outputs attested to Node0 identity
- Attestation hash with BLAKE3/SHA256
- Crystallized insights added to wisdom repository

```bash
# Demo the orchestrator
python core/genesis/genesis_orchestrator.py --demo

# Process a problem through the full pipeline
python core/genesis/genesis_orchestrator.py --process "Design reward mechanism for Proof of Impact"

# Show statistics
python core/genesis/genesis_orchestrator.py --stats
```

**Architecture Diagram**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GENESIS ORCHESTRATOR                                 │
│                                                                              │
│  INTERDISCIPLINARY LENS SYSTEM                                              │
│    [Crypto] [Economics] [Philosophy] [Governance] [Systems] [Cognitive]     │
│                                 ↓                                           │
│                           SYNTHESIS                                          │
│                                 ↓                                           │
│  GRAPH OF THOUGHTS (Beam Width: 8, Max Depth: 5)                            │
│    [Root] ──┬──▶ [Path A] ──▶ [SNR: 0.92] ✓                                │
│             ├──▶ [Path B] ──▶ [SNR: 0.65] ─ (pruned)                       │
│             └──▶ [Path C] ──▶ [SNR: 0.88] ✓                                │
│                                 ↓                                           │
│  SNR AUTONOMOUS ENGINE                                                       │
│    Input → [SNR Gate] → [Rank] → [Select Top-K] → Output                   │
│    If SNR < 0.80 → REJECT (fail closed)                                     │
│                                 ↓                                           │
│  GIANTS PROTOCOL                                                             │
│    [Wisdom Repository] [Integrity Flywheel] [Pattern Crystallizer]         │
│                                 ↓                                           │
│  GENESIS BINDING                                                             │
│    Node0 Identity ◀──▶ Attestation Hash ◀──▶ Wisdom Repository             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Complete Workflow

```bash
# 1. Discover ecosystem artifacts
python scripts/genesis_node_discovery.py --scan C:/bizra_scaffold.worktrees

# 2. Generate Genesis Seal (creates identity + PoI + seal)
python scripts/generate_genesis_seal.py --discover --seal --owner "Momo"

# 3. Verify the seal
python scripts/generate_genesis_seal.py --verify

# 4. Create investor-grade evidence pack
python core/genesis/lineage_seal_pack.py --create

# 5. Verify the pack
python core/genesis/lineage_seal_pack.py --verify data/lineage_packs/BIZRA_LINEAGE_SEAL_PACK_*.zip
```

## Output Files

| File | Location | Purpose |
|------|----------|---------|
| `NODE_ZERO_IDENTITY.json` | `data/genesis/` | Node0 cryptographic identity |
| `PROOF_OF_IMPACT.json` | `data/genesis/` | First Architect contribution proof |
| `GENESIS_SEAL.json` | `data/genesis/` | Master attestation anchor |
| `NODE_ZERO_PRIVATE_KEY.pem` | `data/genesis/` | **KEEP SECURE** - signing key |
| `BIZRA_LINEAGE_SEAL_PACK_*.zip` | `data/lineage_packs/` | Investor-grade evidence |
| `wisdom_repository.json` | `data/wisdom/` | Crystallized wisdom from orchestrator |

## Security Model

1. **Machine Binding**: Node0 is bound to specific hardware fingerprint
2. **Owner Binding**: First Architect's Ed25519 public key is attested
3. **Constitution Binding**: Genesis rules are hash-locked
4. **Timeline Binding**: 3-year provenance is cryptographically sealed
5. **External Oracle**: Prevents self-referential scoring (Goodhart fix)
6. **Fail-Closed**: System fails closed on any verification failure

## Post-Quantum Readiness

Current: Ed25519 (classical signatures)

Roadmap (Q1 2026):
- ML-DSA (Dilithium) for signatures
- ML-KEM (Kyber) for key exchange
- SPHINCS+ for hash-based signatures

## Design Philosophy

- **Giants Protocol**: Every attestation stands on 3 years of accumulated wisdom
- **SNR-Weighted**: Only high-signal artifacts contribute to proofs
- **Graph of Thoughts**: Artifacts form a connected knowledge graph
- **Ihsān-Bound**: All operations must meet ≥0.95 ethical threshold
- **Proof of Impact**: Contribution value is measurable and verifiable

---

*Genesis Node infrastructure for BIZRA v1.0.0*
*Created by the First Architect (Momo) on Node0*
