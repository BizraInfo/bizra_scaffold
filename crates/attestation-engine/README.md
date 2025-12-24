# Sovereign Attestation Engine (Crate)

**Status:** `v0.2.0` (Hardened Attestation Core)
**Integrity:** Run `cargo test` to verify.

This crate implements the core consensus logic for BIZRA NodeZero, as defined in `Genesis_NodeZero_Attestation_Spec_v1.0.md` and `BIZRA_SOT.md`.

## Modules

- **`models`**: Canonical structs (`Attestation`, `AttestationPayload`, `EvidenceBundle`, `IhsanScore`) with stable serialization guarantees.
- **`scoring`**: Deterministic PoI calculation engine. Implements the `Fail-Closed` invariant for Ihsan verification (`score >= 0.95`).
- **`crypto`**: Canonical hashing (JCS), attestation ID derivation, and Ed25519 signing/verification.

## Usage

```rust
use attestation_engine::{issue_attestation, verify_attestation};

// Issue and verify an attestation
let attestation = issue_attestation("contributor", 1, &evidence_bundle, &signing_key, 0.99)?;
verify_attestation(&attestation, &evidence_bundle)?;
```

## Verification

Run tests to verify logic correctness:

```bash
cargo test
```
