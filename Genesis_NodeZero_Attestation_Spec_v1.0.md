# Genesis Node Zero Attestation Specification v1.0

This document defines the canonical **Genesis Node Zero** attestation format and verification rules for the BIZRA system. It serves as the reference for developers implementing Node0 consensus and for auditors verifying attestations on-chain.

## 1. Purpose and Scope

The purpose of this specification is to describe how contributions (tasks, computations, research, etc.) are attested in the genesis environment of BIZRA. These attestations form the foundation of the Proof-of-Impact (PoI) consensus and must be unambiguous, reproducible, and cryptographically verifiable. This spec covers:

1. Structure of attestation data
2. Scoring rules and PoI calculation
3. Validation workflow and quorum requirements
4. Finality and weight assignment
5. Evidence bundling and cryptographic commitments

## 2. Actors and Components

- **Contributor:** Generates a contribution (code, data, compute cycles) and submits evidence.
- **Validator:** A Node0 entity that verifies contributions, computes PoI scores, and participates in consensus.
- **Attestation Engine:** A Rust/NAPI module that canonicalizes inputs, computes PoI scores, and emits attestation records.
- **BlockGraph Consensus Layer:** The DAG-based ledger that stores attestations and derives finality via weighted quality (WQ) metrics.

## 3. Attestation Structure

Each attestation is a JSON object (or CBOR/MsgPack equivalent) with the following canonical fields:

| Field | Type | Description |
|------|------|-------------|
| `attestation_id` | `string` | Deterministic ID derived from the contributor key, epoch, and evidence root (e.g. Blake3 hash). |
| `contributor` | `string` | Public key (Ed25519) of the contributor. |
| `validator` | `string` | Public key of the validator producing the attestation. |
| `epoch` | `integer` | Epoch at which the contribution is evaluated. |
| `evidence_root` | `string` | Blake3 hash of the canonicalized evidence bundle. |
| `poi_score` | `float` | Deterministic impact score computed from evidence (see Section 4). |
| `validation_score` | `float` | Composite confidence score from peer, AI, beneficiary, and other validation layers. |
| `signature` | `string` | Ed25519 signature over the canonical serialization of the attestation (excluding this field). |

### Canonicalization

All inputs (contribution metadata and evidence) must be canonicalized using a stable serialization (e.g. sorted JSON keys, fixed number formatting) before hashing or signing. Floating point values should be encoded in fixed-point or string form to ensure determinism.

## 4. Scoring and PoI Calculation

PoI scoring follows the parameters defined in `BIZRA_SOT.md`. Given an evidence bundle `E`, the Attestation Engine computes:

1. **Dimension scores** - `D_quality(E)`, `D_utility(E)`, `D_trust(E)`, `D_fairness(E)`, `D_diversity(E)`, each returning a value in `[0,1]`.
2. **Raw PoI** - `raw_poi = sum(w_d * D_d(E))` using weights from the SOT.
3. **Penalty** - `penalty = min(max_penalty, f_negative(E))`, where `f_negative` captures negative side effects.
4. **Final PoI** - `poi_score = max(0, raw_poi * (1 - penalty))`.

The resulting `poi_score` must be reproducible by all honest validators. Any discrepancy triggers rejection.

## 5. Validation and Quorum

Validation occurs in two phases:

1. **Local Checks** - The validator verifies the signature, evidence hash, canonical serialization, and recomputes `poi_score`. It discards the attestation if:
   - The evidence bundle is malformed or fails to hash.
   - The signature is invalid.
   - The computed PoI differs from the attestation's `poi_score`.

2. **Consensus Voting** - If local checks pass, validators broadcast their acceptance. An attestation is considered **finalized** when:
   - At least `2f+1` validators (out of `n >= 3f+1`) vote to accept.
   - The aggregate validation score across votes exceeds `ihsan_threshold`.

Rejected attestations must not be propagated further. Validators who equivocate (e.g. double sign conflicting scores) are subject to slashing.

## 6. Weight and Finality in BlockGraph

Once an attestation is finalized, its `poi_score` contributes to the validator's cumulative impact (`I_v(t)`) and thus to the **effective weight** `W_v(t)` used in BlockGraph consensus. The weight computation follows the model in the SOT:

```
I_v(t) = lambda * I_v(t-1) + sum(poi_score_i for epoch t)
W_v(t) = cap(base + a * I_v(t) + b * reputation_v + c * sqrt(stake_v))
```

Blocks reference finalized attestations via **Weighted Quality (WQ)** edges. The BlockGraph fork choice rule selects the path maximizing cumulative WQ. A block is considered final when the cumulative WQ of attestations beyond it exceeds a defined threshold for `k` epochs.

## 7. Evidence Bundling and Proofs

Evidence must be bundled in a reproducible archive format (e.g. TAR + gzip) with deterministic ordering. The evidence bundle should include:

1. Raw data files or metrics produced by the contribution (code diff, inference outputs, dataset rows, etc.).
2. Metadata describing the environment (hardware, software versions) and execution context.
3. Verification artifacts (unit test results, coverage reports, benchmark outputs) supporting the dimension scores.

The bundle is hashed (Blake3) and referenced by `evidence_root`. Optionally, a Merkle tree may be constructed for efficient sub-proofs.

## 8. Implementation Notes

- Use `Blake3` for hashing and `Ed25519` for signatures; these are widely supported and performant, but not post-quantum. If post-quantum security is required, add a PQ signature scheme and document a migration path.
- Validators should maintain a local cache of recent evidence hashes and PoI scores to prevent replay attacks.
- The Attestation Engine should be modular, with separate functions for serialization, hashing, scoring, validation, and signature generation.
- Logs of attestation processing should be persisted for audit; include timestamps, validator IDs, and reasons for rejection.

## 9. Future Extensions

This specification covers the minimum viable attestation model for Genesis Node Zero. Future versions may introduce:

1. **Zero-Knowledge PoI** - allow validators to verify scores without learning sensitive data.
2. **Staged attestation types** - separate formats for compute contributions, data contributions, governance actions, etc.
3. **Advanced reputation systems** - incorporate long-term behavior and community feedback into `reputation_v`.

Any extension must remain backward-compatible or include migration procedures.

---

*End of specification.*
