# BIZRA Single Source of Truth (SOT)

Status: DRAFT - pending evidence and governance approval.

This file is the canonical reference for the unified BIZRA project. All other documentation, code, and tests must align with the definitions and rules set here.

## 1. Identity and Version

- **`sot_id`**: `bizra-sot`
- **`version`**: `0.1.0-draft`
- **`last_updated`**: `2025-12-22`
- **`status`**: `DRAFT`
- **`owners`**: `SAT Council (TBD)`
- **`compatibility`**: defines the minimum versions of other components (e.g. Node0, protocol) that are compatible with this SOT.

## 2. Canonical Names and Roles

### 2.1 Tokens

- **Governance Token**: `BZT` - used for governance voting and weight in consensus.
- **Utility Token**: `BZC` - used as a stable medium of exchange within BIZRA (payment for compute, storage, data).

### 2.2 Layer Names

| Layer | Name | Description |
|---|---|---|
| **L1** | Knowledge Foundation | Immutable ledger and knowledge graph storage |
| **L2** | Compute Infrastructure | Unified Resource Pool (URP) providing CPU/GPU/RAM |
| **L3** | Consensus and Networking | BlockGraph DAG, finality gadget (HotStuff-class) |
| **L4** | Security and Resilience | Zero-trust substrate, cryptographic enforcement |
| **L5** | Agentic OS | Dual-Agentic system, bicameral reasoning, SAPE |
| **L6** | Economy and PoI | Proof-of-Impact scoring, dual tokenomics |
| **L7** | Governance | Decentralized decision-making (SAT Council) |

## 3. Invariants and Hard Rules

- **Evidence Integrity**: Every claim labeled as `FACT` must link to reproducible evidence (test logs, benchmarks, signed attestation). Claims without evidence must be tagged `TARGET` (goal) or `HYPOTHESIS` (research direction).
- **Fail-Closed**: Any unverifiable action or ambiguous state transition must result in rejection rather than acceptance. Unverified plans are not executed.
- **Privacy**: No personally identifiable information (PII) may be stored on-chain; only hashes or aggregated statistics are recorded. PII must remain local and encrypted.
- **Ethics Enforcement**: All actions must satisfy the **Ihsan Metric** (IM >= 0.95). Plans that produce IM below this threshold must be blocked.
- **Token Cap**: Influence from stake (`BZC`/`BZT`) must be capped (e.g. `max_weight = 10.0`) to prevent plutocratic takeover.

### 3.1 Ihsan Metric Definition (Required)

Define the Ihsan Metric in a reproducible way so independent validators can compute the same result. At minimum specify:
- Inputs and measurement sources.
- Normalization and weighting.
- Range and interpretation (e.g. 0.0 to 1.0).
- Evidence artifacts required for audit.

Status: **TBD** (must be defined before production use).

## 4. PoI Parameters (Initial Values)

- **`ihsan_threshold`**: `0.85` (minimum validation score for attestation acceptance)
- **Dimension Weights**:
  - Quality: `0.30`
  - Utility: `0.30`
  - Trust: `0.20`
  - Fairness: `0.10`
  - Diversity: `0.10`
- **Penalty Factor**: `max_penalty = 0.15` (reduces score if negative side effects)
- **Carry Decay**: `lambda = 0.90` (decays PoI influence each epoch)
- **Weight Coefficients**:
  - Base Floor: `0.10`
  - PoI Coeff (`a`): `1.00`
  - Reputation Coeff (`b`): `0.50`
  - Stake Coeff (`c`): `0.05`
  - Cap Max: `10.0`

## 5. Change Control

- **Change Process**: Any update to this SOT file must be accompanied by:
  1. Evidence justifying the change (benchmark logs, new governance decision, spec update).
  2. A version bump (`version` field).
  3. Updates to affected code modules and tests.
  4. A new or updated entry in `EVIDENCE_INDEX.md`.
- **CI Enforcement**: The CI pipeline must validate that no code change violates the invariants or deviates from the canonical definitions herein.

## 6. Evidence Policy

List the acceptable forms of evidence and how they must be referenced:

- **Benchmarks**: Raw output from benchmark tools (k6, wrk, etc.), signed with commit hash and environment details.
- **Test Reports**: Coverage reports, mutation testing reports, formal verification outputs.
- **Attestation Proofs**: Signed Merkle proofs and ZK proofs produced by the PoI engine.
- **CI Logs**: Verified logs from CI runs showing successful passes.

Use relative paths or URLs to link evidence artifacts in other repositories or storage buckets. Track evidence links in `EVIDENCE_INDEX.md`.

## 7. Evidence Citation Format

When referencing evidence, use a consistent, stable format so that reviewers and future developers can easily verify claims:

- **Local Repository Evidence:** For code or data stored within this monorepo, cite the path and line numbers, e.g. `./src/utils/poi.ts#L10-L20`.
- **External Evidence:** For artifacts in other repositories, include the repository name, commit SHA, and file path (e.g. `bizra-genesis-node@69a2d0e/src/consensus.rs#L42-L60`). Avoid referencing unpinned branches.
- **CI Logs and Metrics:** Provide the commit SHA and CI run identifier or job URL so that the logs can be retrieved and re-evaluated.

All citations in documentation should be stable across time; if evidence is updated or invalidated, the associated claim must be re-evaluated, its status updated (FACT/TARGET/HYPOTHESIS), and the link adjusted accordingly.

## Conclusion

This SOT file is the master reference for the BIZRA project. It prevents contradictions, enforces consistent naming, and anchors technical decisions to evidence and ethics. Future changes should be rare and made only with due diligence and consensus among the SAT Council and core developers. All updates must preserve the project's ethical foundations, security posture, and technical excellence.
