# Protocol Comparison: PCI vs ZK-Ihsān Layer

**Date**: 2025-12-26  
**Purpose**: Compare implemented FATE-PCI Gate with proposed ZK-Ihsān architecture

---

## Executive Summary

| Aspect | PCI Protocol (Implemented) | ZK-Ihsān Layer (Proposed) |
|--------|---------------------------|---------------------------|
| **Trust Model** | Cryptographic Signatures (Ed25519) | Zero-Knowledge Proofs (STARKs) |
| **Verification** | Real-time gate chain (~10-150ms) | Proof verification (~300ms on-chain) |
| **Constitution** | `BIZRA_SOT.md` + `policy_hash` binding | `constitution.toml` (TOML format) |
| **Enforcement** | PAT/SAT dual-agent at runtime | Arithmetic circuit + Solidity verifier |
| **Scalability** | O(n) - linear with actions | O(1) - recursive proof composition |
| **Maturity** | ✅ Implemented (35 tests passing) | 📋 Specification only |

---

## 1. Architectural Comparison

### 1.1 Trust Model

**PCI Protocol (Current)**:
```
[PAT Agent] → [Signed Envelope] → [SAT Agent] → [Gate Chain] → [Receipt]
     │                                  │
     └── Ed25519 Signature ─────────────┴── Verification at Runtime
```
- Trust: Verifier (SAT) re-executes gate checks
- Proof: Digital signature binds identity to action
- Limitation: Verifier must be trusted and online

**ZK-Ihsān Layer (Proposed)**:
```
[Agent] → [ZK-VM] → [STARK Proof] → [Verifier.sol] → [On-Chain Receipt]
    │         │
    │         └── Arithmetic Circuit (guest/src/lib.rs)
    └── Input: IhsanReceipt struct
```
- Trust: Mathematical proof (cannot be forged)
- Proof: STARK/SNARK verifiable by anyone
- Advantage: Verification without re-execution

### 1.2 Constitutional Enforcement

**PCI Protocol** - Runtime Policy Binding:
```python
# core/pci/envelope.py
payload = Payload(
    action="state.mutate",
    data={...},
    policy_hash="<blake3-of-constitution>",  # ← Binding
)

# SAT agent verifies:
if envelope.payload.policy_hash != current_policy_hash:
    return reject(REJECT_POLICY_MISMATCH)
```

**ZK-Ihsān** - Compile-Time Invariants:
```rust
// zkiphsan/guest/src/lib.rs
const IHSAN_THRESHOLD_FIXED: u64 = 950; // Hardcoded in circuit

assert!(
    input.ihsan_score >= IHSAN_THRESHOLD_FIXED,
    "CONSTITUTIONAL VIOLATION"
);
```

**Analysis**:
| Feature | PCI | ZK-Ihsān |
|---------|-----|----------|
| Threshold Update | Change config, restart | Recompile circuit, redeploy verifier |
| Flexibility | High | Low (by design) |
| Tamper Resistance | Medium (config can be modified) | High (circuit is immutable) |
| Audit Trail | SAT receipts | On-chain events |

---

## 2. Component Mapping

### 2.1 Constitution / Policy

| PCI Protocol | ZK-Ihsān Layer | Notes |
|--------------|----------------|-------|
| `BIZRA_SOT.md` | `constitution.toml` | Human-readable source of truth |
| `policy_hash` in envelope | Circuit constants | Machine-enforced binding |
| `IhsanThresholdEnforcer` | `IHSAN_THRESHOLD_FIXED = 950` | 0.95 threshold |

### 2.2 Agents / Actors

| PCI Protocol | ZK-Ihsān Layer | Notes |
|--------------|----------------|-------|
| `PATAgent` (Prover/Builder) | Agent generating `IhsanReceipt` | Proposal creation |
| `SATAgent` (Verifier/Governor) | RiscZero `guest` + `Verifier.sol` | Verification logic |
| `PCIEnvelope` | `IhsanReceipt` struct | Wire format |
| `CommitReceipt` | On-chain event `TransactionValidated` | Proof of execution |

### 2.3 Rejection / Failure Modes

**PCI RejectCodes** (16 stable IDs):
```python
class RejectCode(IntEnum):
    SUCCESS = 0
    REJECT_SCHEMA = 1
    REJECT_SIGNATURE = 2
    REJECT_NONCE_REPLAY = 3
    REJECT_TIMESTAMP_STALE = 4
    REJECT_IHSAN_BELOW_MIN = 6
    # ...
```

**ZK-Ihsān Failures**:
```rust
// Circuit fails to generate proof:
assert!(input.ihsan_score >= 950, "CONSTITUTIONAL VIOLATION");

// Or Solidity revert:
require(ihsanScore >= 950, "Constitutional Breach: Ihsan < 0.95");
```

**Key Difference**: PCI provides granular rejection codes for debugging; ZK-Ihsān fails silently (proof doesn't generate) or reverts with generic message.

---

## 3. Performance Characteristics

### 3.1 Latency Comparison

| Operation | PCI Protocol | ZK-Ihsān |
|-----------|--------------|----------|
| Single verification | ~8ms (CHEAP tier) | ~300ms (proof verify) |
| With Ihsān check | ~15ms (MEDIUM tier) | ~300ms |
| With formal verification | ~2000ms (EXPENSIVE) | ~300ms (proof verify only) |
| 1000 actions | ~15,000ms | ~350ms (recursive proof) |

### 3.2 Scalability

**PCI Protocol**: Linear O(n)
```
Time = n × (gate_chain_latency)
1000 actions = 1000 × 15ms = 15,000ms
```

**ZK-Ihsān**: Constant O(1) with recursion
```
Time = proof_generation + recursive_aggregation + verification
1000 actions = 500ms + 100ms + 300ms = 900ms (amortized)
```

**Winner**: ZK-Ihsān for high-throughput scenarios

---

## 4. Security Properties

### 4.1 Cryptographic Guarantees

| Property | PCI Protocol | ZK-Ihsān |
|----------|--------------|----------|
| **Authenticity** | Ed25519 signatures | STARK/Groth16 proofs |
| **Integrity** | BLAKE3 domain-separated digest | SHA256 commitment in circuit |
| **Non-repudiation** | Signed envelope + receipt | On-chain event + proof |
| **Replay Resistance** | Nonce + timestamp TTL cache | Transaction hash in circuit |
| **Confidentiality** | Not addressed | AES-GCM encrypted inputs |
| **Post-Quantum** | Dilithium-5 migration path | STARKs are PQ-resistant |

### 4.2 Attack Vectors

| Attack | PCI Mitigation | ZK-Ihsān Mitigation |
|--------|----------------|---------------------|
| **Signature Forgery** | Ed25519 hardness | Proof forgery requires breaking STARK |
| **Replay Attack** | Nonce + timestamp validation | Transaction hash commitment |
| **Policy Bypass** | SAT checks `policy_hash` | Circuit hardcodes threshold |
| **Verifier Collusion** | Single SAT trust assumption | On-chain verifier is trustless |
| **Time Manipulation** | ±120s window, fail-closed | Block timestamp (Ethereum consensus) |

---

## 5. Integration Path

### 5.1 Current State (PCI Implemented)

```
bizra_scaffold/
├── core/
│   ├── pci/
│   │   ├── __init__.py          ✅
│   │   ├── envelope.py          ✅ (PCIEnvelope, canonical_json)
│   │   ├── reject_codes.py      ✅ (RejectCode enum)
│   │   ├── replay_guard.py      ✅ (TTL cache)
│   │   └── gate.py              ✅ (PCIGate boundary)
│   └── agents/
│       ├── pat.py               ✅ (PATAgent)
│       └── sat.py               ✅ (SATAgent)
├── schemas/
│   ├── pci_envelope.schema.json ✅
│   └── commit_receipt.schema.json ✅
├── tests/
│   ├── vectors/
│   │   └── pci_envelope_v1.json ✅
│   └── test_pci_protocol.py     ✅ (35 tests passing)
└── PROTOCOL.md                  ✅
```

### 5.2 Missing for ZK-Ihsān

```
bizra_scaffold/
├── constitution.toml            ❌ (Need to create)
├── .github/workflows/
│   └── evidence-spine.yml       ❌ (CI/CD gates)
├── zkiphsan/
│   └── guest/src/lib.rs         ❌ (Rust circuit)
├── contracts/
│   └── Verifier.sol             ❌ (Solidity verifier)
└── scripts/
    └── rotate_keys.py           ❌ (Key rotation)
```

### 5.3 Recommended Hybrid Architecture

The two approaches are **complementary**, not mutually exclusive:

```
┌─────────────────────────────────────────────────────────────────┐
│                    BIZRA Unified Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 7 (Philosophy)    constitution.toml                      │
│         │                      │                                │
│         ▼                      ▼                                │
│  ┌─────────────┐    ┌──────────────────┐                       │
│  │ PCI Protocol│    │  ZK-Ihsān Layer  │                       │
│  │ (Real-time) │    │ (Settlement)     │                       │
│  └──────┬──────┘    └────────┬─────────┘                       │
│         │                    │                                  │
│         ▼                    ▼                                  │
│  ┌─────────────┐    ┌──────────────────┐                       │
│  │ PAT → SAT   │───▶│ Proof Generation │                       │
│  │ Gate Chain  │    │ (Batch)          │                       │
│  └─────────────┘    └────────┬─────────┘                       │
│                              │                                  │
│                              ▼                                  │
│                     ┌──────────────────┐                       │
│                     │  Verifier.sol    │                       │
│                     │  (L1 Settlement) │                       │
│                     └──────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Workflow**:
1. **Real-time**: PCI gate chain (PAT → SAT) validates actions in ~15ms
2. **Batching**: Every N receipts or T seconds, aggregate into ZK proof
3. **Settlement**: Submit proof to Verifier.sol for final "Third Fact"

---

## 6. Recommendation: Evolutionary Path

### Phase 1: Current (COMPLETE)
- ✅ PCI Protocol with PAT/SAT agents
- ✅ Ed25519 signatures
- ✅ Gate chain (35 tests passing)

### Phase 2: Constitutional Activation (NEXT)
- ❌ Create `constitution.toml`
- ❌ Implement CI/CD evidence spine
- ❌ Key rotation automation

### Phase 3: ZK Settlement Layer (FUTURE)
- ❌ RiscZero circuit implementation
- ❌ Solidity verifier deployment
- ❌ Recursive proof batching

### Phase 4: Full Integration
- PCI for real-time enforcement
- ZK proofs for trustless settlement
- On-chain "Third Fact" for disputes

---

## 7. Code Alignment Requirements

To bridge PCI → ZK-Ihsān, the following mappings are needed:

### 7.1 IhsanReceipt ↔ PCIEnvelope

```rust
// ZK-Ihsān (to be implemented)
pub struct IhsanReceipt {
    pub agent_id: u64,
    pub transaction_hash: [u32; 8],
    pub snr_score: u64,
    pub ihsan_score: u64,
    pub impact_score: u64,
}
```

```python
# PCI Protocol (already implemented)
@dataclass
class PCIEnvelope:
    envelope_id: str           # → transaction_hash
    sender: Sender             # → agent_id
    metadata: Metadata         # → ihsan_score, snr_score
    # ...
```

### 7.2 Conversion Function

```python
def envelope_to_ihsan_receipt(envelope: PCIEnvelope) -> dict:
    """Convert PCIEnvelope to IhsanReceipt format for ZK proof."""
    return {
        "agent_id": hash(envelope.sender.agent_id) % (2**64),
        "transaction_hash": blake3(envelope.digest().encode()).digest()[:32],
        "snr_score": int(envelope.metadata.snr_score * 1000) if envelope.metadata.snr_score else 0,
        "ihsan_score": int(envelope.metadata.ihsan_score * 1000),
        "impact_score": 0,  # TBD
    }
```

---

## 8. Conclusion

| Criterion | Winner | Rationale |
|-----------|--------|-----------|
| **Immediate Use** | PCI ✅ | Already implemented, 35 tests |
| **Trustlessness** | ZK-Ihsān | Mathematical proof > signature |
| **Scalability** | ZK-Ihsān | O(1) recursive proofs |
| **Flexibility** | PCI | Runtime config vs compile-time |
| **Complexity** | PCI | No ZK toolchain required |
| **Audit Trail** | Tie | Both produce receipts |

**Final Recommendation**: 
1. **Keep PCI** as the real-time enforcement layer
2. **Add constitution.toml** for machine-readable policy
3. **Plan ZK-Ihsān** as the settlement layer for "Third Fact" finality

The PCI protocol and ZK-Ihsān layer are **complementary layers** in the BIZRA sovereignty stack.
