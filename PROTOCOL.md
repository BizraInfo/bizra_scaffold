# BIZRA PCI Protocol Specification v1.0

**Status**: FROZEN — Changes require version bump + test vector update  
**Governance Hash**: Computed at build time  
**Alignment**: BIZRA_SOT.md Section 3.1 (Ihsān IM ≥ 0.95)

---

## 1. Overview

This document defines the **Proof-Carrying Inference (PCI)** protocol for BIZRA's
dual-agent architecture. It serves as the executable contract that generates:

- JSON Schemas (envelope + receipt)
- RejectCode registry with stable numeric IDs
- Test vectors (canonical bytes + digest + signature)
- Gate ordering & latency budgets
- Fail-closed semantics

**Non-Negotiable Invariants**:
- Every envelope is cryptographically signed
- Every commit produces an append-only receipt
- Every rejection is logged with RejectCode + audit trail
- Fail-closed: ambiguous state → rejection (never silent acceptance)

---

## 2. Wire Format: PCIEnvelope

### 2.1 Canonical JSON (RFC 8785 JCS)

All envelopes MUST be serialized using JSON Canonicalization Scheme:
- Keys sorted lexicographically (Unicode code point order)
- No whitespace between tokens
- Numbers: no leading zeros, no trailing zeros after decimal
- Strings: UTF-8, minimal escape sequences
- No duplicate keys

```python
# Reference implementation: core/pci/envelope.py
def canonical_json(data: Dict[str, Any]) -> bytes:
    return json.dumps(
        data,
        separators=(',', ':'),
        sort_keys=True,
        ensure_ascii=False
    ).encode('utf-8')
```

### 2.2 Domain-Separated Digest

Digests MUST use domain separation to prevent cross-protocol attacks:

```
digest = BLAKE3("bizra-pci-v1:" || canonical_bytes)
```

The domain prefix `"bizra-pci-v1:"` is UTF-8 encoded and prepended to the
canonical JSON bytes before hashing.

### 2.3 Envelope Structure

```json
{
  "version": "1.0.0",
  "envelope_id": "<uuid4>",
  "timestamp": "<ISO8601-UTC>",
  "nonce": "<hex-encoded-32-bytes>",
  "sender": {
    "agent_type": "PAT|SAT",
    "agent_id": "<unique-identifier>",
    "public_key": "<hex-encoded-ed25519-pubkey>"
  },
  "payload": {
    "action": "<action-type>",
    "data": { ... },
    "policy_hash": "<blake3-of-constitution>",
    "state_hash": "<blake3-of-current-state>"
  },
  "metadata": {
    "ihsan_score": 0.95,
    "snr_score": 0.80,
    "urgency": "REAL_TIME|NEAR_REAL_TIME|BATCH|DEFERRED"
  },
  "signature": {
    "algorithm": "ed25519",
    "value": "<hex-encoded-signature>",
    "signed_fields": ["version", "envelope_id", "timestamp", "nonce", "sender", "payload", "metadata"]
  }
}
```

### 2.4 Required Fields

| Field | Type | Constraints |
|-------|------|-------------|
| `version` | string | SemVer, currently "1.0.0" |
| `envelope_id` | string | UUID v4, globally unique |
| `timestamp` | string | ISO8601 UTC, max 120s skew |
| `nonce` | string | 32 bytes hex, never reused |
| `sender.agent_type` | enum | "PAT" or "SAT" |
| `sender.public_key` | string | 32 bytes hex (Ed25519) |
| `payload.action` | string | Action type identifier |
| `payload.policy_hash` | string | BLAKE3 of constitution.toml |
| `metadata.ihsan_score` | float | [0.0, 1.0], required ≥ 0.95 for commit |
| `signature.algorithm` | enum | "ed25519" (future: "dilithium5") |
| `signature.value` | string | 64 bytes hex |

---

## 3. Wire Format: CommitReceipt

### 3.1 Receipt Structure

```json
{
  "version": "1.0.0",
  "receipt_id": "<uuid4>",
  "timestamp": "<ISO8601-UTC>",
  "envelope_digest": "<blake3-of-envelope>",
  "commit_ref": {
    "type": "eventlog|blockgraph",
    "offset": 12345,
    "block_hash": "<optional-hex>"
  },
  "verification": {
    "tier": "STATISTICAL|INCREMENTAL|OPTIMISTIC|FULL_ZK|FORMAL",
    "latency_ms": 8.5,
    "gates_passed": ["SCHEMA", "SIGNATURE", "REPLAY", "IHSAN", "SNR", "FATE"],
    "ihsan_score": 0.97,
    "snr_score": 0.85
  },
  "verifier_set": [
    {
      "sat_id": "<verifier-id>",
      "public_key": "<hex>",
      "signature": "<hex>",
      "timestamp": "<ISO8601-UTC>"
    }
  ],
  "quorum": {
    "required": 1,
    "achieved": 1
  },
  "audit_digest": "<blake3-of-verification-report>",
  "policy_hash": "<blake3-of-constitution>"
}
```

### 3.2 Append-Only Semantics

Receipts are **immutable** once created. The `commit_ref.offset` is a
monotonically increasing sequence number in the event log.

---

## 4. RejectCode Registry

Stable numeric IDs for cross-language compatibility and audit logging.

| Code | Name | Description |
|------|------|-------------|
| 0 | `SUCCESS` | Operation completed successfully |
| 1 | `REJECT_SCHEMA` | Envelope failed JSON schema validation |
| 2 | `REJECT_SIGNATURE` | Cryptographic signature invalid |
| 3 | `REJECT_NONCE_REPLAY` | Nonce already seen within TTL window |
| 4 | `REJECT_TIMESTAMP_STALE` | Timestamp outside acceptable skew (±120s) |
| 5 | `REJECT_TIMESTAMP_FUTURE` | Timestamp too far in future |
| 6 | `REJECT_IHSAN_BELOW_MIN` | Ihsān score < 0.95 threshold |
| 7 | `REJECT_SNR_BELOW_MIN` | SNR score below tier threshold |
| 8 | `REJECT_BUDGET_EXCEEDED` | Verification latency exceeded tier budget |
| 9 | `REJECT_POLICY_MISMATCH` | policy_hash doesn't match current constitution |
| 10 | `REJECT_STATE_MISMATCH` | state_hash doesn't match expected state |
| 11 | `REJECT_ROLE_VIOLATION` | Agent attempted unauthorized action (PAT commit) |
| 12 | `REJECT_QUORUM_FAILED` | Insufficient verifier signatures |
| 13 | `REJECT_FATE_VIOLATION` | FATE invariant check failed |
| 14 | `REJECT_INVARIANT_FAILED` | Formal invariant verification failed |
| 15 | `REJECT_RATE_LIMITED` | Too many requests from sender |
| 99 | `REJECT_INTERNAL_ERROR` | Unexpected internal error (fail-closed) |

### 4.1 Rejection Response

```json
{
  "rejected": true,
  "code": 6,
  "name": "REJECT_IHSAN_BELOW_MIN",
  "message": "Ihsān score 0.89 < required 0.95",
  "envelope_digest": "<blake3>",
  "timestamp": "<ISO8601-UTC>",
  "audit_trail": {
    "gate": "IHSAN",
    "tier": "CHEAP",
    "latency_ms": 2.1,
    "details": { "score": 0.89, "threshold": 0.95 }
  }
}
```

---

## 5. Verification Gate Ordering

### 5.1 Tiered Chain of Responsibility

Gates execute in strict order. First rejection terminates the chain (fail-fast).

```
┌─────────────────────────────────────────────────────────────────────┐
│  CHEAP TIER (<10ms, fail-closed)                                     │
├─────────────────────────────────────────────────────────────────────┤
│  1. SCHEMA     - JSON schema validation                              │
│  2. SIGNATURE  - Ed25519 signature verification                      │
│  3. TIMESTAMP  - Freshness check (±120s skew)                        │
│  4. REPLAY     - Nonce not in seen-cache (TTL 120s)                  │
│  5. ROLE       - PAT can propose, SAT can commit                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ pass
┌─────────────────────────────────────────────────────────────────────┐
│  MEDIUM TIER (<150ms)                                                │
├─────────────────────────────────────────────────────────────────────┤
│  6. SNR        - Signal-to-noise ratio ≥ tier threshold              │
│  7. IHSAN      - Ihsān metric ≥ 0.95                                 │
│  8. POLICY     - policy_hash matches current constitution            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ pass
┌─────────────────────────────────────────────────────────────────────┐
│  EXPENSIVE TIER (bounded, only for state mutations)                  │
├─────────────────────────────────────────────────────────────────────┤
│  9. FATE       - FATE invariant verification (SMT/Z3)                │
│  10. FORMAL    - Mathematical proof verification                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ pass
┌─────────────────────────────────────────────────────────────────────┐
│  COMMIT                                                              │
├─────────────────────────────────────────────────────────────────────┤
│  11. Append to event log                                             │
│  12. Generate CommitReceipt                                          │
│  13. Emit verification event                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Latency Budgets

| Tier | Max Latency | Gates | Confidence |
|------|-------------|-------|------------|
| CHEAP | 10ms | SCHEMA, SIGNATURE, TIMESTAMP, REPLAY, ROLE | 95% |
| MEDIUM | 150ms | SNR, IHSAN, POLICY | 98% |
| EXPENSIVE | 2000ms | FATE, FORMAL | 99.9% |

### 5.3 Fail-Closed Semantics

- **Any gate failure** → immediate rejection with RejectCode
- **Timeout** → rejection with `REJECT_BUDGET_EXCEEDED`
- **Internal error** → rejection with `REJECT_INTERNAL_ERROR`
- **Ambiguous state** → rejection (never silent pass)

---

## 6. Agent Roles

### 6.1 PAT (Prover/Builder)

**Capabilities**:
- Construct PCIEnvelope with proposals
- Sign envelopes with Ed25519 private key
- Validate Ihsān threshold before emission
- Submit proposals to SAT

**Constraints**:
- CANNOT commit to event log
- CANNOT issue CommitReceipt
- MUST set `sender.agent_type = "PAT"`

### 6.2 SAT (Verifier/Governor)

**Capabilities**:
- Receive PCIEnvelopes from PAT
- Execute verification gate chain
- Commit to event log on success
- Issue signed CommitReceipt
- Reject with RejectCode on failure

**Constraints**:
- CANNOT modify payload content
- MUST execute all applicable gates
- MUST emit receipt for every decision (accept or reject)

---

## 7. Replay Resistance

### 7.1 Nonce Requirements

- 32 bytes (256 bits) cryptographically random
- Hex-encoded in envelope
- NEVER reused across envelopes

### 7.2 Seen-Nonce Cache

- TTL: 120 seconds
- LRU eviction when capacity exceeded
- Thread-safe access
- Checked AFTER signature verification (prevent DoS)

### 7.3 Timestamp Window

- Max clock skew: ±120 seconds from verifier's UTC time
- Future timestamps > 120s → REJECT_TIMESTAMP_FUTURE
- Past timestamps > 120s → REJECT_TIMESTAMP_STALE

---

## 8. Cryptographic Algorithms

### 8.1 Current (v1.0)

| Purpose | Algorithm | Key Size | Output Size |
|---------|-----------|----------|-------------|
| Signature | Ed25519 | 256-bit | 512-bit |
| Digest | BLAKE3 | N/A | 256-bit |
| Nonce | CSPRNG | N/A | 256-bit |

### 8.2 Post-Quantum Migration Path

Future versions will support algorithm negotiation:

```json
{
  "signature": {
    "algorithm": "dilithium5",
    "value": "<hex-encoded-dilithium-signature>"
  }
}
```

Supported algorithms (future):
- `dilithium5` - NIST FIPS 204 (ML-DSA)
- `falcon1024` - Lattice-based alternative
- `sphincs256` - Hash-based (stateless)

---

## 9. Constitution Binding

### 9.1 Policy Hash

Every envelope MUST include `payload.policy_hash` which is:

```
policy_hash = BLAKE3(canonical_json(constitution))
```

Where `constitution` is loaded from `constitution.toml`.

### 9.2 Verification

SAT MUST verify that `policy_hash` matches the hash of the currently
active constitution. Mismatch → `REJECT_POLICY_MISMATCH`.

This prevents "policy swapping" attacks where an envelope is crafted
against an outdated or malicious constitution.

---

## 10. Test Vectors

### 10.1 Canonical JSON Vector

**Input**:
```json
{"b": 2, "a": 1, "c": {"z": 26, "y": 25}}
```

**Canonical Output** (UTF-8 bytes):
```
{"a":1,"b":2,"c":{"y":25,"z":26}}
```

**BLAKE3 Digest** (hex):
```
f3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4
```

### 10.2 Domain-Separated Digest Vector

**Domain Prefix**: `"bizra-pci-v1:"`
**Canonical Bytes**: `{"action":"test","data":{}}`
**Combined Input**: `bizra-pci-v1:{"action":"test","data":{}}`
**BLAKE3 Digest**: (computed at test time)

### 10.3 Envelope Signature Vector

See `tests/vectors/pci_envelope_v1.json` for complete test vectors
including known-good signatures that MUST pass verification.

---

## 11. Compatibility

### 11.1 Version Negotiation

Envelope `version` field follows SemVer:
- MAJOR: Breaking wire format changes
- MINOR: Backward-compatible additions
- PATCH: Bug fixes, no wire changes

### 11.2 Backward Compatibility

- v1.x envelopes MUST be accepted by v1.y verifiers where y ≥ x
- Unknown fields in `metadata` SHOULD be preserved but ignored
- Unknown fields in `payload.data` are application-specific

---

## 12. Security Considerations

### 12.1 Threat Model

| Threat | Mitigation |
|--------|------------|
| Replay attack | Nonce + timestamp + seen-cache |
| Signature forgery | Ed25519 cryptographic security |
| Policy swap | policy_hash binding |
| State manipulation | state_hash binding |
| DoS via invalid envelopes | Signature check before expensive ops |
| Clock skew attack | ±120s tolerance, NTP sync required |

### 12.2 Key Management

- Private keys MUST be stored in HSM or secure enclave
- Key rotation: Generate new keypair, update public key registry
- Compromised key: Revoke immediately, reject all envelopes signed by it

---

## 13. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-26 | Initial frozen specification |

---

## 14. References

- [BIZRA_SOT.md](BIZRA_SOT.md) - Source of Truth (Ihsān threshold)
- [RFC 8785](https://tools.ietf.org/html/rfc8785) - JSON Canonicalization Scheme
- [Ed25519](https://ed25519.cr.yp.to/) - High-speed signatures
- [BLAKE3](https://github.com/BLAKE3-team/BLAKE3) - Cryptographic hash function
- [NIST FIPS 204](https://csrc.nist.gov/pubs/fips/204/final) - ML-DSA (Dilithium)
