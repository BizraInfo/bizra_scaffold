# Post-Quantum Cryptography Migration Roadmap

**Status**: PLANNING | **Priority**: HIGH | **Target**: Q2 2026

This document outlines the migration path from classical cryptographic primitives to post-quantum (PQ) secure alternatives within the BIZRA attestation and cognitive security systems.

## 1. Current Cryptographic Stack

| Component | Algorithm | Quantum Vulnerability | Location |
|-----------|-----------|----------------------|----------|
| Signatures | Ed25519 | ❌ HIGH (Shor's algorithm) | `crypto.rs`, `cognitive_sovereign.py` |
| Content Hashing | Blake3 | ⚠️ MEDIUM (Grover's algorithm) | `crypto.rs` |
| Temporal Chains | SHA3-512 | ✅ LOW (512-bit → 256-bit effective) | `cognitive_sovereign.py` |
| Key Exchange | N/A | N/A | Not currently used |

## 2. Threat Model

### 2.1 "Harvest Now, Decrypt Later" (HNDL)

Adversaries may collect signed attestations today and break them when quantum computers become available. This is the **primary threat** for BIZRA attestations, which are designed to be immutable and long-lived.

### 2.2 Timeline Estimates

| Milestone | Conservative | Aggressive |
|-----------|--------------|------------|
| Cryptographically relevant QC | 2035+ | 2030 |
| Early warning signals | 2028 | 2026 |
| NIST PQ standards finalization | ✅ 2024 (ML-DSA, ML-KEM) | ✅ 2024 |

## 3. Target Post-Quantum Algorithms

Based on NIST FIPS 203/204/205 (August 2024):

| Use Case | Algorithm | Standard | Library |
|----------|-----------|----------|---------|
| Digital Signatures | ML-DSA (Dilithium) | FIPS 204 | `pqcrypto`, `liboqs` |
| Key Encapsulation | ML-KEM (Kyber) | FIPS 203 | `pqcrypto`, `liboqs` |
| Stateless Signatures | SLH-DSA (SPHINCS+) | FIPS 205 | `pqcrypto` |
| Hybrid Approach | Ed25519 + ML-DSA | Custom | Dual-signature |

## 4. Migration Strategy

### Phase 1: Hybrid Signatures (Q1 2026)

Implement dual-signature scheme: attestations signed with **both** Ed25519 and ML-DSA.

```rust
// crypto.rs - Proposed hybrid signature structure
pub struct HybridSignature {
    pub classical: Ed25519Signature,  // Current compatibility
    pub quantum_safe: MlDsaSignature, // Future-proof
    pub algorithm_id: u8,             // Version marker
}

impl HybridSignature {
    pub fn verify_either(&self, message: &[u8], 
                         ed_key: &VerifyingKey,
                         ml_key: &MlDsaPublicKey) -> bool {
        // Accept if EITHER signature is valid (graceful migration)
        self.verify_classical(message, ed_key) || 
        self.verify_quantum_safe(message, ml_key)
    }
    
    pub fn verify_both(&self, message: &[u8],
                       ed_key: &VerifyingKey,
                       ml_key: &MlDsaPublicKey) -> bool {
        // Require BOTH valid (maximum security)
        self.verify_classical(message, ed_key) && 
        self.verify_quantum_safe(message, ml_key)
    }
}
```

### Phase 2: Key Infrastructure (Q2 2026)

1. **Generate ML-DSA keypairs** for all validators
2. **Store PQ public keys** in attestation metadata
3. **Update key derivation** to use hybrid entropy sources

### Phase 3: Protocol Transition (Q3 2026)

1. **Soft fork**: Accept hybrid signatures, prefer PQ verification
2. **Hard fork deadline**: Reject Ed25519-only attestations after epoch X
3. **Legacy attestation archival**: Re-sign historical attestations with PQ keys

### Phase 4: Classical Deprecation (Q4 2026)

1. **Remove Ed25519** from verification path
2. **Audit and attest** migration completeness
3. **Update SOT** with new cryptographic invariants

## 5. Python Implementation Notes

For `cognitive_sovereign.py` and `ihsan_bridge.py`:

```python
# Proposed hybrid security class
from typing import Tuple
import hashlib

class QuantumResistantSecurity:
    """
    Post-quantum secure temporal sequencing.
    Phase 1: Hybrid Ed25519 + ML-DSA signatures.
    """
    
    PQ_ALGORITHM_ID: int = 0x02  # Version marker
    
    def __init__(self, use_hybrid: bool = True):
        self.use_hybrid = use_hybrid
        
        # Classical keys (current)
        self._ed25519_signing_key = ed25519.Ed25519PrivateKey.generate()
        
        # Post-quantum keys (Phase 1+)
        if use_hybrid:
            try:
                from pqcrypto.sign import dilithium3
                self._ml_dsa_keypair = dilithium3.generate_keypair()
            except ImportError:
                logger.warning("PQ library not available, using classical-only")
                self.use_hybrid = False
    
    def sign_hybrid(self, message: bytes) -> Tuple[bytes, bytes, int]:
        """
        Produce hybrid signature: (ed25519_sig, ml_dsa_sig, algorithm_id)
        """
        ed_sig = self._ed25519_signing_key.sign(message)
        
        if self.use_hybrid:
            ml_sig = dilithium3.sign(self._ml_dsa_keypair[1], message)
            return (ed_sig, ml_sig, self.PQ_ALGORITHM_ID)
        else:
            return (ed_sig, b'', 0x01)  # Classical-only marker
```

## 6. Hash Function Considerations

| Current | Post-Quantum | Action |
|---------|--------------|--------|
| Blake3 (256-bit) | Blake3 (512-bit output) | Increase output size |
| SHA3-512 | SHA3-512 | ✅ Already PQ-resistant |

Grover's algorithm halves effective security. A 256-bit hash provides 128-bit PQ security, which is considered adequate for most applications. However, for long-term attestation integrity, we recommend:

1. **Temporal chains**: Continue using SHA3-512 (256-bit PQ security)
2. **Content hashing**: Upgrade Blake3 to 512-bit output mode
3. **Attestation IDs**: Derive from SHA3-512 instead of Blake3

## 7. Dependency Updates

### Rust (`Cargo.toml`)

```toml
[dependencies]
# Current
ed25519-dalek = "2.1"
blake3 = "1.5"

# Post-quantum (Phase 1+)
pqcrypto = "0.17"
pqcrypto-dilithium = "0.5"
pqcrypto-kyber = "0.8"

# Hybrid wrapper
hybrid-pq-signatures = "0.1"  # Internal crate
```

### Python (`requirements.txt`)

```
# Current
cryptography==45.0.7

# Post-quantum (Phase 1+)
pqcrypto>=0.1.0
liboqs-python>=0.10.0
```

## 8. Testing and Validation

### 8.1 Interoperability Tests

- [ ] Rust hybrid signatures verifiable by Python
- [ ] Python hybrid signatures verifiable by Rust
- [ ] Legacy Ed25519-only attestations remain valid
- [ ] New PQ-only attestations rejected by old validators (expected)

### 8.2 Performance Benchmarks

| Algorithm | Sign (μs) | Verify (μs) | Signature Size |
|-----------|-----------|-------------|----------------|
| Ed25519 | 50 | 120 | 64 bytes |
| ML-DSA-65 (Dilithium3) | 200 | 180 | 3,309 bytes |
| Hybrid | 250 | 300 | 3,373 bytes |

**Note**: ML-DSA signatures are ~50x larger than Ed25519. This impacts:
- Attestation storage
- Network bandwidth
- Merkle proof sizes

### 8.3 Migration Simulation

1. Generate 10,000 hybrid attestations
2. Verify with classical-only verifier (should pass)
3. Verify with PQ-only verifier (should pass)
4. Simulate key compromise: invalidate Ed25519, verify PQ still works

## 9. Evidence Requirements

For SOT compliance, the migration must produce:

| Artifact | Description | Status |
|----------|-------------|--------|
| EVID-PQ-001 | Hybrid signature implementation (Rust) | PENDING |
| EVID-PQ-002 | Hybrid signature implementation (Python) | PENDING |
| EVID-PQ-003 | Interoperability test results | PENDING |
| EVID-PQ-004 | Performance benchmark logs | PENDING |
| EVID-PQ-005 | Migration simulation report | PENDING |
| EVID-PQ-006 | Updated SOT with PQ invariants | PENDING |

## 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PQ algorithm breaks | LOW | CRITICAL | Multi-algorithm fallback (SLH-DSA backup) |
| Performance regression | MEDIUM | MEDIUM | Signature caching, batch verification |
| Ecosystem fragmentation | MEDIUM | HIGH | Clear migration timeline, deprecation notices |
| Key management complexity | HIGH | MEDIUM | Automated key rotation, HSM integration |

## 11. Timeline Summary

```
2025 Q4: Research and planning (CURRENT)
2026 Q1: Hybrid signature implementation
2026 Q2: Key infrastructure deployment
2026 Q3: Soft fork activation
2026 Q4: Classical deprecation
2027 Q1: Full PQ-only mode
```

## 12. References

- [NIST FIPS 203 - ML-KEM](https://csrc.nist.gov/pubs/fips/203/final)
- [NIST FIPS 204 - ML-DSA](https://csrc.nist.gov/pubs/fips/204/final)
- [NIST FIPS 205 - SLH-DSA](https://csrc.nist.gov/pubs/fips/205/final)
- [liboqs - Open Quantum Safe](https://openquantumsafe.org/)
- [pqcrypto Rust crate](https://crates.io/crates/pqcrypto)

---

*Document Version: 1.0.0 | Last Updated: 2025-12-23 | Owner: BIZRA Security Council*
