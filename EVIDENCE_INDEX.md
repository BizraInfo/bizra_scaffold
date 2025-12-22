# Evidence Index

This index tracks evidence artifacts referenced across the scaffold. Replace placeholders with concrete artifacts and update status as evidence is verified.

Status values: `PENDING`, `VERIFIED`, `INVALIDATED`.

| ID | Claim summary | Source doc | Evidence artifact | Status | Notes |
|---|---|---|---|---|---|
| EVID-001 | Node0 architecture diagrams and system documentation | `report.md` | `bizra-node0-genesis@<commit>/docs/...` | PENDING | Add exact doc paths and commit SHA. |
| EVID-002 | Node0 performance metrics (throughput, latency) | `report.md` | `bizra-node0-genesis@<commit>/benchmarks/...` | PENDING | Include benchmark tool output and environment details. |
| EVID-003 | Node0 security posture (zero-trust, SBOM, DR) | `report.md` | `bizra-node0-genesis@<commit>/docs/security/...` | PENDING | Add SBOM artifact and security scan logs. |
| EVID-004 | Test coverage and formal verification coverage | `report.md` | `CI run <id> / coverage reports` | PENDING | Link CI job URL and commit SHA. |
| EVID-005 | bizra-core Week 1 plan and blueprint framing | `report.md` | `bizra-core@<commit>/README.md` | PENDING | Confirm repository and commit. |
| EVID-006 | PoI pack gaps and TypeScript build failure | `report.md` | `bizra-poi-v0.1@<commit>/...` | PENDING | Point to failing build logs and missing typings. |

Update this file whenever claims are added, removed, or reclassified.
