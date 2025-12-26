# BIZRA Verification Standard (BUILD-VERIFY-METRICS)

This folder is the authoritative proof boundary for BIZRA builds.

## Guarantees
- Every CI run produces an Evidence Receipt and Metrics.
- No claims are true unless backed by an Evidence Receipt.

## Outputs
- `evidence/receipts/<receipt_id>.json`
- `evidence/metrics/latest.json`
- `evidence/checks/*.json` (tool outputs, if available)

## Gates (recommended)
1. Correctness: tests pass.
2. Security: dependency audits pass (or explicit waiver).
3. Policy: OPA/Rego decision = allow.
4. Ihsan: vector score >= threshold (start 0.85; target 0.95).
5. Performance: latency regression budget.
6. Evidence: receipt completeness = 1.0.

## Next wiring steps
- Replace placeholder policy engine with OPA/Rego.
- Replace heuristic Ihsan vector with real checks (correctness, safety, efficiency, auditability).
