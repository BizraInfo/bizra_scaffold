# BridgeCoordinator ↔ COVENANT Integration Plan

**Status**: Week 2 Phase 1 - Integration Design Complete
**Date**: 2026-01-15

---

## Architecture Overview

### Current Flow (Existing)
```
DualAgenticRequest
    ↓
BridgeCoordinator::execute()
    ├─ Step 1: SAT validates request
    ├─ Step 2: PAT executes (parallel agents)
    ├─ Step 2.5: SNR filtering (>1.5 threshold)
    ├─ Step 3: SAT evaluates PAT results
    ├─ Step 4: Calculate synergy + Ihsān scores
    ├─ Step 5: FATE escalation (if needed)
    ├─ Step 6: Emit execution receipt
    └─ Step 7: Return DualAgenticResponse
```

### COVENANT Mapping
```
COVENANT Stage          →  Existing Component
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1: SENSE          →  Request capture + Blake3 hash
Stage 2: REASON         →  PAT parallel execution
Stage 3: SCORE          →  Ihsān calculation (Step 4)
Stage 4: GATE           →  SAT validation (Step 1 + 3)
Stage 5: ACT            →  Synthesis + commit
Stage 6: LEDGER         →  Receipt emission (Step 6)
Stage 7: PROOF          →  ZK verifier (existing)
Stage 8: SNR UPDATE     →  **NEW**: global_monitor() tracking
```

---

## Integration Code (Non-Breaking Changes)

### 1. Add CovenantBridge to BridgeCoordinator Struct

```rust
// src/bridge.rs
use crate::covenant_bridge::CovenantBridge;

pub struct BridgeCoordinator {
    // ... existing fields ...

    // COVENANT integration (Week 2 Phase 1)
    covenant: CovenantBridge,
}

impl BridgeCoordinator {
    pub async fn new() -> anyhow::Result<Self> {
        // ... existing initialization ...

        let covenant = CovenantBridge::default(); // Respects BIZRA_COVENANT_MODE env var

        Ok(Self {
            // ... existing fields ...
            covenant,
        })
    }
}
```

### 2. Add SNR Tracking to execute() Method

```rust
// At the start of execute()
pub async fn execute(
    &self,
    request: DualAgenticRequest,
) -> anyhow::Result<DualAgenticResponse> {
    let start = Instant::now();

    // COVENANT Stage 1: SENSE
    let thought_id = self.covenant.record_request_start(&request);

    // ... existing code continues ...
```

### 3. Track PAT Execution (COVENANT Stage 2: REASON)

```rust
// After PAT execution completes
let pat_results = pat_results_future.await?;

// COVENANT Stage 2: REASON
self.covenant.record_reasoning_complete(thought_id, pat_results.len());
```

### 4. Track Ihsān Scoring (COVENANT Stage 3: SCORE)

```rust
// After Ihsān calculation (existing Step 4)
let ihsan_passes_threshold = ihsan_score >= ihsan_threshold_fixed;

// COVENANT Stage 3: SCORE
self.covenant.record_ihsan_scoring(
    thought_id,
    ihsan_score,
    ihsan_passes_threshold,
);
```

### 5. Track SAT Validation (COVENANT Stage 4: GATE)

```rust
// After SAT validation (existing Step 1)
let validation = self.sat.validate_request(&request).await?;

// COVENANT Stage 4: GATE (SAT)
self.covenant.record_sat_validation(
    thought_id,
    validation.consensus_reached,
    &validation.rejection_codes.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
);
```

### 6. Track Action Commitment (COVENANT Stage 5: ACT)

```rust
// After successful synthesis (before receipt emission)
// COVENANT Stage 5: ACT
self.covenant.record_action_committed(thought_id);
```

### 7. Track Ledger Append (COVENANT Stage 6: LEDGER)

```rust
// After receipt emission
let receipt = self.receipts.emit_execution(/* ... */);

// COVENANT Stage 6: LEDGER
self.covenant.record_ledger_append(thought_id, &receipt.receipt_id);
```

### 8. Track Proof Generation (COVENANT Stage 7: PROOF)

```rust
// After ZK verification (if enabled)
let proof_verified = /* zk verification result */;

// COVENANT Stage 7: PROOF
self.covenant.record_proof_generated(thought_id, proof_verified);
```

### 9. Get SNR Metrics (COVENANT Stage 8: SNR UPDATE)

```rust
// At the end of execute(), before returning response
let current_snr = self.covenant.get_current_snr();

info!(
    thought_id = %thought_id.to_string(),
    current_snr = %current_snr.to_f64(),
    "COVENANT pipeline complete"
);
```

### 10. Optional: Generate AttestedThought Receipt

```rust
// If covenant mode is enabled, generate canonical thought object
if self.covenant.is_covenant_mode() {
    let attested_thought = self.covenant.generate_attested_thought(
        thought_id,
        &request,
        &snr_filtered_results,
        ihsan_score,
        &ihsan_dimensions,
        validation.consensus_reached,
        Some(receipt.receipt_id.clone()),
    );

    // Emit parallel COVENANT receipt
    let covenant_receipt = serde_json::to_string_pretty(&attested_thought)?;
    let covenant_path = format!("docs/evidence/receipts/THOUGHT-{}.json", thought_id.to_string());
    tokio::fs::write(&covenant_path, covenant_receipt).await?;

    info!(
        path = %covenant_path,
        "✅ COVENANT AttestedThought receipt emitted"
    );
}
```

---

## SNR Metrics Endpoint (HTTP API)

### Add to HTTP Router

```rust
// src/http.rs (or appropriate HTTP module)
use crate::covenant_bridge::CovenantBridge;

// Add new endpoint
async fn covenant_metrics() -> impl axum::response::IntoResponse {
    let covenant = CovenantBridge::default();
    let report = covenant.get_snr_report();

    (
        axum::http::StatusCode::OK,
        [("Content-Type", "text/plain")],
        report,
    )
}

// Register route
app.route("/api/v1/covenant/metrics", axum::routing::get(covenant_metrics))
```

### Example Response

```
GET /api/v1/covenant/metrics

╔══════════════════════════════════════════════════════════════╗
║                  SNR METRICS REPORT                          ║
╠══════════════════════════════════════════════════════════════╣
║ Signal (Verified Actions):        142                       ║
║ Noise (Wasted Cycles):          8500                       ║
║ Total Cycles:                  150000                       ║
║                                                              ║
║ SNR Ratio:                      0.0009                       ║
║ Threshold:                      0.9500                       ║
║ Status:                          ❌ FAIL                     ║
║                                                              ║
║ FAILURE MODES:                                               ║
║   Rollbacks:                        3                       ║
║   Human Vetoes:                     0                       ║
║   Ihsān Rejections:                 8                       ║
║   FATE Violations:                  5                       ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Environment Variables

### Enable/Disable COVENANT Mode

```bash
# Enable covenant mode (default)
export BIZRA_COVENANT_MODE=true

# Disable covenant mode (legacy behavior only)
export BIZRA_COVENANT_MODE=false
```

### Production Configuration

```bash
# .env.production
BIZRA_COVENANT_MODE=true          # Enable SNR tracking
BIZRA_IHSAN_ENFORCE=true          # Enforce Ihsān threshold
RUST_LOG=info,meta_alpha_dual_agentic::covenant_bridge=debug
```

---

## Testing Strategy

### Unit Tests (Already Included)

```bash
# Test covenant bridge in isolation
cargo test covenant_bridge -- --nocapture

# Test specific integration scenarios
cargo test test_covenant_bridge_lifecycle
cargo test test_attested_thought_generation
```

### Integration Tests

```bash
# Create new integration test
cargo test --test covenant_integration -- --nocapture
```

Example test file:

```rust
// tests/covenant_integration.rs
use meta_alpha_dual_agentic::{
    bridge::BridgeCoordinator,
    covenant_bridge::CovenantBridge,
    snr_monitor::global_monitor,
    types::DualAgenticRequest,
};

#[tokio::test]
async fn test_full_covenant_pipeline() {
    // Enable covenant mode
    std::env::set_var("BIZRA_COVENANT_MODE", "true");

    let bridge = BridgeCoordinator::new().await.unwrap();

    let request = DualAgenticRequest {
        task: "Test COVENANT integration".to_string(),
        context: std::collections::HashMap::new(),
        mode: crate::types::AdapterModes::Auto,
    };

    let result = bridge.execute(request).await;
    assert!(result.is_ok());

    // Verify SNR was tracked
    let monitor = global_monitor();
    let snr = monitor.current_snr();
    assert!(snr >= Fixed64::ZERO);

    std::env::remove_var("BIZRA_COVENANT_MODE");
}
```

---

## Rollout Plan (Week 2)

### Phase 1: Non-Breaking Integration (Days 1-2)
- [x] Create CovenantBridge module
- [ ] Add to BridgeCoordinator struct
- [ ] Wire up 8 tracking calls in execute()
- [ ] Run existing test suite (ensure no regressions)
- [ ] Deploy with BIZRA_COVENANT_MODE=false (tracking only, no enforcement)

### Phase 2: Parallel Receipts (Days 3-4)
- [ ] Enable AttestedThought receipt generation
- [ ] Add HTTP endpoint for SNR metrics
- [ ] Dashboard integration (WebSocket for live metrics)
- [ ] Deploy with BIZRA_COVENANT_MODE=true (parallel receipts)

### Phase 3: CI Enforcement (Days 5-7)
- [ ] Add GitHub Actions workflow step
- [ ] Fail builds if SNR < 0.95 on test runs
- [ ] Update documentation
- [ ] Full production deployment

---

## Success Metrics

### Week 2 Phase 1 Complete When:
1. ✅ CovenantBridge module created (DONE)
2. ✅ Integration plan documented (DONE)
3. ⏳ BridgeCoordinator wired up (IN PROGRESS)
4. ⏳ All existing tests pass
5. ⏳ SNR metrics visible in logs
6. ⏳ HTTP endpoint returns real metrics
7. ⏳ No performance degradation (< 5ms overhead)

### SNR Improvement Target:
- **Baseline**: Measure current SNR with tracking enabled
- **Week 2 Goal**: SNR ≥ 0.50 (50% signal ratio)
- **Week 4 Goal**: SNR ≥ 0.75 (75% signal ratio)
- **Production Goal**: SNR ≥ 0.95 (COVENANT threshold)

---

## Notes for Implementation

### Minimal Overhead Design
- All tracking calls are non-blocking
- SNR monitor uses lock-free counters where possible
- Only compute expensive metrics on-demand (report generation)

### Backward Compatibility
- Existing receipts unchanged
- COVENANT receipts are parallel/additional
- Can disable with environment variable
- No breaking API changes

### Future Enhancements (Week 3+)
- Real-time SNR streaming via WebSocket
- Per-agent SNR breakdown
- Temporal SNR trend graphs
- Automated threshold adjustment (Kalman optimization)
- Integration with Giants Protocol (citation tracking)

---

**Status**: Ready for implementation in BridgeCoordinator
**Estimated Effort**: 4-6 hours for full integration
**Risk Level**: LOW (non-breaking, can be disabled)
