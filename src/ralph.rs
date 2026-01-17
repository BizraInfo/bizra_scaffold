// src/ralph.rs - Sovereign Ralph Autonomous Orchestrator
//
// Implements the Ralph autonomous loop technique from:
// - ralph-claude-code: Dual-exit gates, rate limiting
// - ralph-orchestrator: Multi-agent orchestration, git checkpointing
//
// Giants Protocol Synthesis:
// - Al-Ghazali: Knowledge-Action Unity - Dual-exit gate = knowledge (completion) + action (EXIT_SIGNAL)
// - Ibn Khaldun: Asabiyyah Lifecycle - Iteration decay: enthusiasm → stagnation → renewal
// - Ibn Rushd: Multi-Path Truth - Multiple exit conditions
// - Nash: Game Theory - Equilibrium detection
// - Kalman: State Estimation - Convergence tracking

use crate::fixed::Fixed64;
use crate::identity::{CovenantCheckResult, KalmanState};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Ralph configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RalphConfig {
    /// Maximum iterations before forced exit
    pub max_iterations: usize,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
    /// Convergence threshold for Kalman filter
    pub convergence_threshold: f64,
    /// Whether to require explicit exit signal
    pub require_exit_signal: bool,
    /// The exit signal marker string
    pub exit_signal: String,
    /// Circuit breaker failure trip count
    pub failure_trip: usize,
    /// Circuit breaker recovery time in milliseconds
    pub recovery_ms: u64,
    /// Rate limit requests per minute
    pub rate_limit_per_minute: u32,
    /// Token bucket capacity
    pub token_bucket_capacity: u32,
    /// Ihsan threshold for completion
    pub ihsan_threshold: f64,
    /// SNR threshold for memory writes
    pub snr_threshold: f64,
}

impl Default for RalphConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            timeout_ms: 5 * 60 * 1000, // 5 minutes
            convergence_threshold: 0.01,
            require_exit_signal: true,
            exit_signal: "EXIT_SIGNAL".to_string(),
            failure_trip: 3,
            recovery_ms: 30_000,
            rate_limit_per_minute: 60,
            token_bucket_capacity: 30,
            ihsan_threshold: 0.95,
            snr_threshold: 0.70,
        }
    }
}

// ============================================================================
// DUAL EXIT GATE
// ============================================================================

/// Dual-Exit Gate - Requires BOTH conditions (ralph-claude-code pattern)
///
/// # Giants Protocol
/// - Al-Ghazali: Knowledge (completion) + Action (explicit signal) unity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DualExitGate {
    /// Task appears complete (quality thresholds met)
    pub completion_indicator: bool,
    /// Explicit EXIT_SIGNAL received
    pub explicit_exit_signal: bool,
    /// Timestamp when gate was last evaluated (Unix ms)
    pub gate_timestamp_ms: Option<u64>,
}

impl DualExitGate {
    /// Create a new dual exit gate
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if BOTH exit conditions are satisfied
    pub fn is_satisfied(&self) -> bool {
        self.completion_indicator && self.explicit_exit_signal
    }

    /// Update gate with new evaluation
    pub fn update(&mut self, completion: bool, exit_signal: bool) {
        self.completion_indicator = completion;
        self.explicit_exit_signal = exit_signal;
        self.gate_timestamp_ms = Some(unix_utc_ms());
    }

    /// Check completion only (ignoring exit signal requirement)
    pub fn is_complete(&self) -> bool {
        self.completion_indicator
    }
}

// ============================================================================
// EXIT CONDITIONS
// ============================================================================

/// Exit conditions for autonomous loops
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExitCondition {
    /// Successful completion with marker
    Success { marker: String, iteration: usize },
    /// SAT validator VETO
    SATVeto { validator: String, reason: String },
    /// Maximum iterations reached
    MaxIterations { limit: usize },
    /// Timeout exceeded
    Timeout { elapsed_ms: u64, limit_ms: u64 },
    /// Resource exhausted (memory, API, rate limit)
    ResourceExhausted { resource: String, usage: f64 },
    /// Convergence detected (iterations yielding diminishing returns)
    Convergence { delta: f64, threshold: f64 },
    /// Circuit breaker opened due to failures
    CircuitOpen { failures: usize },
    /// User interrupt requested
    UserInterrupt,
    /// Covenant rule violation
    CovenantViolation { rule_id: String, reason: String },
}

impl ExitCondition {
    /// Check if this is a successful exit
    pub fn is_success(&self) -> bool {
        matches!(self, ExitCondition::Success { .. })
    }

    /// Check if this is a failure exit
    pub fn is_failure(&self) -> bool {
        matches!(
            self,
            ExitCondition::SATVeto { .. }
                | ExitCondition::CircuitOpen { .. }
                | ExitCondition::CovenantViolation { .. }
        )
    }

    /// Get human-readable description
    pub fn description(&self) -> String {
        match self {
            ExitCondition::Success { marker, iteration } => {
                format!("Success at iteration {} with marker: {}", iteration, marker)
            }
            ExitCondition::SATVeto { validator, reason } => {
                format!("SAT VETO by {}: {}", validator, reason)
            }
            ExitCondition::MaxIterations { limit } => {
                format!("Maximum iterations reached: {}", limit)
            }
            ExitCondition::Timeout { elapsed_ms, limit_ms } => {
                format!("Timeout: {}ms / {}ms", elapsed_ms, limit_ms)
            }
            ExitCondition::ResourceExhausted { resource, usage } => {
                format!("Resource exhausted: {} ({}%)", resource, usage * 100.0)
            }
            ExitCondition::Convergence { delta, threshold } => {
                format!("Convergence: delta {} < threshold {}", delta, threshold)
            }
            ExitCondition::CircuitOpen { failures } => {
                format!("Circuit breaker open after {} failures", failures)
            }
            ExitCondition::UserInterrupt => "User interrupt".to_string(),
            ExitCondition::CovenantViolation { rule_id, reason } => {
                format!("Covenant violation ({}): {}", rule_id, reason)
            }
        }
    }
}

// ============================================================================
// CIRCUIT BREAKER
// ============================================================================

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum CircuitState {
    /// Normal operation - requests allowed
    #[default]
    Closed,
    /// Failing - requests rejected
    Open,
    /// Testing recovery - limited requests
    HalfOpen,
}


/// Circuit breaker for fault tolerance
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Current state
    pub state: CircuitState,
    /// Consecutive failure count
    failures: usize,
    /// When the circuit opened
    opened_at: Option<Instant>,
    /// Failures needed to trip
    trip_threshold: usize,
    /// Recovery timeout
    recovery_timeout: Duration,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(trip_threshold: usize, recovery_ms: u64) -> Self {
        Self {
            state: CircuitState::Closed,
            failures: 0,
            opened_at: None,
            trip_threshold: trip_threshold.max(1),
            recovery_timeout: Duration::from_millis(recovery_ms.max(1)),
        }
    }

    /// Record a successful operation
    pub fn on_success(&mut self) {
        self.failures = 0;
        self.state = CircuitState::Closed;
        self.opened_at = None;
    }

    /// Record a failed operation
    pub fn on_failure(&mut self) {
        self.failures += 1;
        if self.failures >= self.trip_threshold {
            self.state = CircuitState::Open;
            self.opened_at = Some(Instant::now());
        }
    }

    /// Check if a request is allowed
    pub fn allow_request(&mut self) -> Result<(), ExitCondition> {
        match self.state {
            CircuitState::Closed => Ok(()),
            CircuitState::Open => {
                // Check if recovery timeout has elapsed
                if let Some(opened) = self.opened_at {
                    if opened.elapsed() >= self.recovery_timeout {
                        self.state = CircuitState::HalfOpen;
                        return Ok(());
                    }
                }
                Err(ExitCondition::CircuitOpen {
                    failures: self.failures,
                })
            }
            CircuitState::HalfOpen => Ok(()),
        }
    }

    /// Get current failure count
    pub fn failures(&self) -> usize {
        self.failures
    }

    /// Get current state
    pub fn state(&self) -> CircuitState {
        self.state
    }
}

// ============================================================================
// TOKEN BUCKET RATE LIMITER
// ============================================================================

/// Token bucket rate limiter
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Bucket capacity
    capacity: u32,
    /// Current token count
    tokens: f64,
    /// Refill rate (tokens per millisecond)
    refill_per_ms: f64,
    /// Last refill timestamp
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a new token bucket
    pub fn new(capacity: u32, per_minute: u32) -> Self {
        let cap = capacity.max(1);
        let rpm = per_minute.max(1) as f64;
        let refill_per_ms = rpm / 60_000.0;

        Self {
            capacity: cap,
            tokens: cap as f64,
            refill_per_ms,
            last_refill: Instant::now(),
        }
    }

    /// Try to acquire tokens
    pub fn try_acquire(&mut self, n: u32) -> bool {
        self.refill();
        let need = n as f64;
        if self.tokens >= need {
            self.tokens -= need;
            true
        } else {
            false
        }
    }

    /// Get current token count
    pub fn available_tokens(&self) -> u32 {
        self.tokens as u32
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let elapsed_ms = self.last_refill.elapsed().as_millis() as f64;
        if elapsed_ms > 0.0 {
            self.tokens = (self.tokens + elapsed_ms * self.refill_per_ms).min(self.capacity as f64);
            self.last_refill = Instant::now();
        }
    }
}

// ============================================================================
// QUALITY METRICS
// ============================================================================

/// Quality metrics for iteration evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Signal-to-noise ratio (0.0 - 1.0)
    pub snr: f64,
    /// Ihsan score (0.0 - 1.0)
    pub ihsan: f64,
    /// SAT consensus (0.0 - 1.0)
    pub sat_consensus: f64,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            snr: 0.0,
            ihsan: 0.0,
            sat_consensus: 0.0,
            confidence: 0.0,
        }
    }
}

impl QualityMetrics {
    /// Compute weighted composite score
    pub fn composite(&self) -> f64 {
        // Weights tuned for BIZRA quality gates
        (0.30 * self.snr) + (0.35 * self.ihsan) + (0.25 * self.sat_consensus) + (0.10 * self.confidence)
    }

    /// Convert SNR to Fixed64 for deterministic operations
    pub fn snr_fixed(&self) -> Fixed64 {
        Fixed64::from_f64(self.snr.clamp(0.0, 1.0))
    }

    /// Convert Ihsan to Fixed64 for deterministic operations
    pub fn ihsan_fixed(&self) -> Fixed64 {
        Fixed64::from_f64(self.ihsan.clamp(0.0, 1.0))
    }

    /// Check if metrics pass quality gates
    pub fn passes_gates(&self, ihsan_threshold: f64, snr_threshold: f64) -> bool {
        self.ihsan >= ihsan_threshold && self.snr >= snr_threshold
    }
}

// ============================================================================
// ITERATION OUTCOME
// ============================================================================

/// Outcome of a single iteration
#[derive(Debug, Clone)]
pub struct IterationOutcome<R> {
    /// The response from execution
    pub response: R,
    /// Dual exit gate state
    pub gate: DualExitGate,
    /// Quality metrics
    pub metrics: QualityMetrics,
    /// SAT VETO information (if any)
    pub veto: Option<(String, String)>, // (validator, reason)
    /// Exit signal marker found
    pub marker: String,
    /// Additional notes
    pub notes: String,
    /// Covenant check result
    pub covenant_result: Option<CovenantCheckResult>,
}

// ============================================================================
// ITERATION RECEIPT
// ============================================================================

/// Receipt for a single iteration (deterministic, auditable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationReceipt {
    /// Iteration number
    pub iteration: usize,
    /// Creation timestamp (Unix UTC milliseconds)
    pub created_utc_ms: u64,
    /// SHA-256 hash of request
    pub request_hash: String,
    /// SHA-256 hash of response
    pub response_hash: String,
    /// Quality metrics
    pub metrics: QualityMetrics,
    /// Gate state
    pub gate: GateReceipt,
    /// Exit condition (if terminal)
    pub exit_condition: Option<ExitCondition>,
}

/// Gate state for receipt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateReceipt {
    /// Completion indicator
    pub completion_indicator: bool,
    /// Explicit exit signal
    pub explicit_exit_signal: bool,
    /// Marker found
    pub marker: String,
}

// ============================================================================
// EXECUTOR TRAIT
// ============================================================================

/// Trait for execution backends (BridgeCoordinator, etc.)
#[allow(clippy::type_complexity)] // Complex return type needed for async trait
pub trait RalphExecutor: Send + Sync {
    /// Request type
    type Request: Send + Sync + Clone + Serialize + 'static;
    /// Response type
    type Response: Send + Sync + Clone + Serialize + 'static;

    /// Execute a single iteration
    fn execute_once(
        &self,
        req: Self::Request,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<IterationOutcome<Self::Response>>> + Send>>;
}

// ============================================================================
// SOVEREIGN RALPH ORCHESTRATOR
// ============================================================================

/// Sovereign Ralph - Autonomous execution orchestrator
///
/// Ralph wraps the execution pipeline with:
/// - Dual-exit gates (completion + explicit signal)
/// - Circuit breaker for fault tolerance
/// - Rate limiting for resource protection
/// - Kalman-filtered convergence detection
/// - Deterministic receipt generation
pub struct SovereignRalph<E: RalphExecutor> {
    /// The underlying executor
    executor: E,
    /// Configuration
    config: RalphConfig,
    /// Circuit breaker
    circuit: CircuitBreaker,
    /// Rate limiter
    limiter: TokenBucket,
    /// Kalman state for convergence
    kalman: KalmanState,
    /// Iteration receipts
    receipts: VecDeque<IterationReceipt>,
    /// Last successful response
    last_response: Option<E::Response>,
}

impl<E: RalphExecutor> SovereignRalph<E> {
    /// Create a new Sovereign Ralph orchestrator
    pub fn new(executor: E, config: RalphConfig) -> Self {
        Self {
            circuit: CircuitBreaker::new(config.failure_trip, config.recovery_ms),
            limiter: TokenBucket::new(config.token_bucket_capacity, config.rate_limit_per_minute),
            executor,
            config,
            kalman: KalmanState::default(),
            receipts: VecDeque::with_capacity(256),
            last_response: None,
        }
    }

    /// Get iteration receipts
    pub fn receipts(&self) -> &VecDeque<IterationReceipt> {
        &self.receipts
    }

    /// Get circuit breaker state
    pub fn circuit_state(&self) -> CircuitState {
        self.circuit.state()
    }

    /// Execute until exit condition is met
    pub async fn execute_until_exit(
        &mut self,
        req: E::Request,
    ) -> anyhow::Result<(E::Response, ExitCondition)> {
        let start = Instant::now();
        let timeout = Duration::from_millis(self.config.timeout_ms);
        let mut last_estimate = 0.0_f64;

        info!("🔄 SovereignRalph: Starting autonomous execution");

        for iteration in 1..=self.config.max_iterations {
            // === Exit Condition Evaluation Order (deterministic) ===

            // 1. Timeout check
            let elapsed_ms = start.elapsed().as_millis() as u64;
            if start.elapsed() >= timeout {
                warn!("Ralph: Timeout after {}ms", elapsed_ms);
                return self.finalize_exit(ExitCondition::Timeout {
                    elapsed_ms,
                    limit_ms: self.config.timeout_ms,
                });
            }

            // 2. Rate limit check
            if !self.limiter.try_acquire(1) {
                warn!("Ralph: Rate limit exceeded");
                return self.finalize_exit(ExitCondition::ResourceExhausted {
                    resource: "rate_limit".to_string(),
                    usage: 1.0,
                });
            }

            // 3. Circuit breaker check
            if let Err(exit) = self.circuit.allow_request() {
                warn!("Ralph: Circuit breaker open");
                return self.finalize_exit(exit);
            }

            // 4. Execute iteration
            let outcome = self.executor.execute_once(req.clone()).await;

            match outcome {
                Ok(out) => {
                    self.circuit.on_success();
                    self.last_response = Some(out.response.clone());

                    // Generate receipt
                    let receipt = self.create_receipt(iteration, &req, &out);
                    self.receipts.push_back(receipt);

                    // 5. Check SAT VETO (highest priority after infrastructure gates)
                    if let Some((validator, reason)) = &out.veto {
                        warn!("Ralph: SAT VETO by {}", validator);
                        return Ok((
                            out.response,
                            ExitCondition::SATVeto {
                                validator: validator.clone(),
                                reason: reason.clone(),
                            },
                        ));
                    }

                    // 6. Check Covenant violation
                    if let Some(covenant_result) = &out.covenant_result {
                        if !covenant_result.allowed {
                            if let Some(violation) = covenant_result.violations.first() {
                                warn!("Ralph: Covenant violation: {}", violation.message);
                                return Ok((
                                    out.response,
                                    ExitCondition::CovenantViolation {
                                        rule_id: violation.rule_id.clone(),
                                        reason: violation.message.clone(),
                                    },
                                ));
                            }
                        }
                    }

                    // 7. Convergence tracking
                    let composite = out.metrics.composite();
                    let estimate = self.kalman.update(composite);
                    let delta = (estimate - last_estimate).abs();
                    last_estimate = estimate;

                    if delta < self.config.convergence_threshold
                        && self.kalman.has_converged(self.config.convergence_threshold)
                    {
                        info!("Ralph: Convergence detected at iteration {}", iteration);
                        return Ok((
                            out.response,
                            ExitCondition::Convergence {
                                delta,
                                threshold: self.config.convergence_threshold,
                            },
                        ));
                    }

                    // 8. Check dual exit gate
                    let dual_exit_ok = out.gate.completion_indicator
                        && (!self.config.require_exit_signal || out.gate.explicit_exit_signal);

                    if dual_exit_ok {
                        info!("Ralph: Dual exit gate satisfied at iteration {}", iteration);
                        return Ok((
                            out.response,
                            ExitCondition::Success {
                                marker: out.marker,
                                iteration,
                            },
                        ));
                    }
                }
                Err(e) => {
                    self.circuit.on_failure();
                    warn!("Ralph: Iteration {} failed: {}", iteration, e);
                    // Continue to next iteration unless circuit trips
                    continue;
                }
            }
        }

        // 9. Max iterations reached
        warn!("Ralph: Max iterations reached");
        self.finalize_exit(ExitCondition::MaxIterations {
            limit: self.config.max_iterations,
        })
    }

    /// Create a receipt for an iteration
    fn create_receipt(
        &self,
        iteration: usize,
        req: &E::Request,
        out: &IterationOutcome<E::Response>,
    ) -> IterationReceipt {
        let req_hash = deterministic_hash(&canonical_json_bytes(req));
        let resp_hash = deterministic_hash(&canonical_json_bytes(&out.response));

        IterationReceipt {
            iteration,
            created_utc_ms: unix_utc_ms(),
            request_hash: req_hash,
            response_hash: resp_hash,
            metrics: out.metrics.clone(),
            gate: GateReceipt {
                completion_indicator: out.gate.completion_indicator,
                explicit_exit_signal: out.gate.explicit_exit_signal,
                marker: out.marker.clone(),
            },
            exit_condition: None,
        }
    }

    /// Finalize exit with last response
    fn finalize_exit(
        &self,
        condition: ExitCondition,
    ) -> anyhow::Result<(E::Response, ExitCondition)> {
        if let Some(response) = &self.last_response {
            Ok((response.clone(), condition))
        } else {
            anyhow::bail!(
                "Ralph: No successful response captured. Exit condition: {}",
                condition.description()
            )
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Get current Unix UTC timestamp in milliseconds
fn unix_utc_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Canonicalize JSON value for deterministic hashing
fn canonicalize_json_value(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(map) => {
            let mut btm = BTreeMap::new();
            for (k, vv) in map {
                btm.insert(k, canonicalize_json_value(vv));
            }
            let mut out = serde_json::Map::new();
            for (k, vv) in btm {
                out.insert(k, vv);
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(canonicalize_json_value).collect())
        }
        other => other,
    }
}

/// Convert to canonical JSON bytes
fn canonical_json_bytes<T: Serialize>(t: &T) -> Vec<u8> {
    let v = serde_json::to_value(t).unwrap_or(serde_json::Value::Null);
    let canon = canonicalize_json_value(v);
    serde_json::to_vec(&canon).unwrap_or_default()
}

/// Compute deterministic SHA-256 hash
fn deterministic_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_exit_gate() {
        let mut gate = DualExitGate::new();
        assert!(!gate.is_satisfied());

        gate.update(true, false);
        assert!(!gate.is_satisfied());
        assert!(gate.is_complete());

        gate.update(true, true);
        assert!(gate.is_satisfied());
    }

    #[test]
    fn test_circuit_breaker_states() {
        let mut cb = CircuitBreaker::new(3, 1000);
        assert_eq!(cb.state(), CircuitState::Closed);

        // Fail twice - still closed
        cb.on_failure();
        cb.on_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        // Third failure trips the breaker
        cb.on_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Should reject requests when open
        assert!(cb.allow_request().is_err());

        // Success resets
        cb.on_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow_request().is_ok());
    }

    #[test]
    fn test_token_bucket() {
        let mut bucket = TokenBucket::new(10, 60);

        // Should have tokens initially
        assert!(bucket.try_acquire(5));
        assert!(bucket.try_acquire(5));

        // Should be empty now
        assert!(!bucket.try_acquire(1));
    }

    #[test]
    fn test_quality_metrics_composite() {
        let metrics = QualityMetrics {
            snr: 0.9,
            ihsan: 0.95,
            sat_consensus: 0.8,
            confidence: 0.85,
        };

        let composite = metrics.composite();
        assert!(composite > 0.8);
        assert!(composite < 1.0);
    }

    #[test]
    fn test_quality_metrics_gates() {
        let metrics = QualityMetrics {
            snr: 0.75,
            ihsan: 0.96,
            sat_consensus: 0.8,
            confidence: 0.85,
        };

        assert!(metrics.passes_gates(0.95, 0.70));
        assert!(!metrics.passes_gates(0.95, 0.80)); // SNR too low
        assert!(!metrics.passes_gates(0.98, 0.70)); // Ihsan too low
    }

    #[test]
    fn test_exit_condition_classification() {
        let success = ExitCondition::Success {
            marker: "done".to_string(),
            iteration: 5,
        };
        assert!(success.is_success());
        assert!(!success.is_failure());

        let veto = ExitCondition::SATVeto {
            validator: "security".to_string(),
            reason: "threat detected".to_string(),
        };
        assert!(!veto.is_success());
        assert!(veto.is_failure());

        let timeout = ExitCondition::Timeout {
            elapsed_ms: 5000,
            limit_ms: 3000,
        };
        assert!(!timeout.is_success());
        assert!(!timeout.is_failure()); // Timeout is not a "failure" per se
    }

    #[test]
    fn test_canonical_json() {
        use serde_json::json;

        let v1 = json!({"b": 2, "a": 1});
        let v2 = json!({"a": 1, "b": 2});

        let c1 = canonical_json_bytes(&v1);
        let c2 = canonical_json_bytes(&v2);

        assert_eq!(c1, c2); // Order should not matter after canonicalization
    }

    #[test]
    fn test_deterministic_hash() {
        let data = b"test data for hashing";
        let hash1 = deterministic_hash(data);
        let hash2 = deterministic_hash(data);

        assert_eq!(hash1, hash2);
        assert!(!hash1.is_empty());
    }

    #[test]
    fn test_ralph_config_defaults() {
        let config = RalphConfig::default();

        assert_eq!(config.max_iterations, 50);
        assert_eq!(config.ihsan_threshold, 0.95);
        assert_eq!(config.snr_threshold, 0.70);
        assert!(config.require_exit_signal);
    }
}
