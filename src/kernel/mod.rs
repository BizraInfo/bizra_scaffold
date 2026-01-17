// src/kernel/mod.rs - Apotheosis Node Kernel
//
// The Kernel is the root runtime that owns the event bus, hook chain,
// policy enforcement, and resource governance. All operations flow through
// the kernel via deterministic event dispatch.
//
// Giants Protocol Synthesis:
// - Al-Ghazali: Knowledge-Action Unity - Events bind knowledge to execution
// - Dijkstra: Formal Methods - Deterministic ordering prevents race conditions
// - Ibn Rushd: Multi-Path Truth - Multiple listeners can observe events

use crate::identity::{Covenant, CovenantCheckResult, GoalStack, IdentityAnchor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, RwLock};
use tracing::info;

// ============================================================================
// KERNEL EVENTS
// ============================================================================

/// All events that flow through the kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelEvent {
    /// Request intake - entry point
    Intake {
        request_id: String,
        task: String,
        priority: u8,
    },
    /// Identity loaded for session
    IdentityLoaded {
        user_id: String,
        display_name: String,
    },
    /// Pre-plan hook
    PlanPre {
        plan_id: String,
        description: String,
    },
    /// Post-plan hook
    PlanPost {
        plan_id: String,
        success: bool,
    },
    /// Pre-action hook (before tool/capability use)
    ActionPre {
        action_id: String,
        action_type: String,
        agent_id: String,
    },
    /// Post-action hook
    ActionPost {
        action_id: String,
        success: bool,
        duration_ms: u64,
    },
    /// Pre-verify hook (before SAT validation)
    VerifyPre {
        validation_id: String,
    },
    /// Post-verify hook
    VerifyPost {
        validation_id: String,
        consensus: f64,
        vetoed: bool,
    },
    /// Pre-memory-commit hook
    MemoryCommitPre {
        memory_type: String,
        content_hash: String,
    },
    /// Post-memory-commit hook
    MemoryCommitPost {
        receipt_id: String,
    },
    /// Receipt finalized
    ReceiptFinalized {
        receipt_id: String,
        receipt_hash: String,
    },
    /// Exit event (session termination)
    Exit {
        reason: String,
        success: bool,
    },
    /// Custom event (for extensions)
    Custom {
        event_type: String,
        payload: serde_json::Value,
    },
}

impl KernelEvent {
    /// Get a short type name for this event
    pub fn event_type(&self) -> &'static str {
        match self {
            KernelEvent::Intake { .. } => "intake",
            KernelEvent::IdentityLoaded { .. } => "identity_loaded",
            KernelEvent::PlanPre { .. } => "plan_pre",
            KernelEvent::PlanPost { .. } => "plan_post",
            KernelEvent::ActionPre { .. } => "action_pre",
            KernelEvent::ActionPost { .. } => "action_post",
            KernelEvent::VerifyPre { .. } => "verify_pre",
            KernelEvent::VerifyPost { .. } => "verify_post",
            KernelEvent::MemoryCommitPre { .. } => "memory_commit_pre",
            KernelEvent::MemoryCommitPost { .. } => "memory_commit_post",
            KernelEvent::ReceiptFinalized { .. } => "receipt_finalized",
            KernelEvent::Exit { .. } => "exit",
            KernelEvent::Custom { .. } => "custom",
        }
    }
}

// ============================================================================
// EVENT BUS
// ============================================================================

/// Event bus for kernel event distribution
pub struct EventBus {
    /// Broadcast sender for events
    sender: broadcast::Sender<KernelEvent>,
    /// Event history for audit
    history: Arc<RwLock<VecDeque<EventRecord>>>,
    /// Maximum history size
    max_history: usize,
}

/// A recorded event with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRecord {
    /// Event sequence number
    pub sequence: u64,
    /// Timestamp (Unix ms)
    pub timestamp_ms: u64,
    /// The event
    pub event: KernelEvent,
    /// SHA-256 hash of the event
    pub event_hash: String,
}

impl EventBus {
    /// Create a new event bus
    pub fn new(max_history: usize) -> Self {
        let (sender, _) = broadcast::channel(256);
        Self {
            sender,
            history: Arc::new(RwLock::new(VecDeque::with_capacity(max_history))),
            max_history,
        }
    }

    /// Subscribe to events
    pub fn subscribe(&self) -> broadcast::Receiver<KernelEvent> {
        self.sender.subscribe()
    }

    /// Publish an event
    pub async fn publish(&self, event: KernelEvent) -> anyhow::Result<u64> {
        let mut history = self.history.write().await;

        // Generate sequence number
        let sequence = history.back().map(|r| r.sequence + 1).unwrap_or(1);

        // Create record
        let timestamp_ms = unix_utc_ms();
        let event_hash = hash_event(&event, sequence, timestamp_ms);

        let record = EventRecord {
            sequence,
            timestamp_ms,
            event: event.clone(),
            event_hash,
        };

        // Add to history
        if history.len() >= self.max_history {
            history.pop_front();
        }
        history.push_back(record);

        // Broadcast (ignore if no subscribers)
        let _ = self.sender.send(event);

        Ok(sequence)
    }

    /// Get event history
    pub async fn history(&self) -> Vec<EventRecord> {
        let history = self.history.read().await;
        history.iter().cloned().collect()
    }

    /// Get history count
    pub async fn history_count(&self) -> usize {
        let history = self.history.read().await;
        history.len()
    }

    /// Clear history
    pub async fn clear_history(&self) {
        let mut history = self.history.write().await;
        history.clear();
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(1000)
    }
}

// ============================================================================
// RESOURCE GOVERNOR
// ============================================================================

/// Resource limits for the kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage (MB)
    pub max_memory_mb: u64,
    /// Maximum CPU time (ms per request)
    pub max_cpu_time_ms: u64,
    /// Maximum concurrent operations
    pub max_concurrent: u32,
    /// Maximum events per second
    pub max_events_per_second: u32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 4096,
            max_cpu_time_ms: 60_000,
            max_concurrent: 16,
            max_events_per_second: 100,
        }
    }
}

/// Resource governor for budget enforcement
pub struct ResourceGovernor {
    /// Resource limits
    limits: ResourceLimits,
    /// Current memory usage estimate (MB)
    current_memory_mb: Arc<RwLock<u64>>,
    /// Active operation count
    active_operations: Arc<RwLock<u32>>,
    /// Events in current second
    events_this_second: Arc<RwLock<u32>>,
    /// Last reset timestamp
    last_reset_ms: Arc<RwLock<u64>>,
}

impl ResourceGovernor {
    /// Create a new resource governor
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            limits,
            current_memory_mb: Arc::new(RwLock::new(0)),
            active_operations: Arc::new(RwLock::new(0)),
            events_this_second: Arc::new(RwLock::new(0)),
            last_reset_ms: Arc::new(RwLock::new(unix_utc_ms())),
        }
    }

    /// Check if resources are available
    pub async fn check_resources(&self) -> ResourceCheckResult {
        let memory = *self.current_memory_mb.read().await;
        let operations = *self.active_operations.read().await;
        let events = self.get_events_this_second().await;

        let mut issues = Vec::new();

        if memory > self.limits.max_memory_mb {
            issues.push(format!(
                "Memory exceeded: {} > {} MB",
                memory, self.limits.max_memory_mb
            ));
        }

        if operations >= self.limits.max_concurrent {
            issues.push(format!(
                "Concurrent limit reached: {} >= {}",
                operations, self.limits.max_concurrent
            ));
        }

        if events >= self.limits.max_events_per_second {
            issues.push(format!(
                "Event rate exceeded: {} >= {}/s",
                events, self.limits.max_events_per_second
            ));
        }

        ResourceCheckResult {
            allowed: issues.is_empty(),
            issues,
            memory_usage_percent: memory as f64 / self.limits.max_memory_mb as f64,
            operation_usage_percent: operations as f64 / self.limits.max_concurrent as f64,
        }
    }

    /// Reserve resources for an operation
    pub async fn reserve(&self) -> anyhow::Result<ResourceReservation> {
        let check = self.check_resources().await;
        if !check.allowed {
            anyhow::bail!("Resource check failed: {:?}", check.issues);
        }

        let mut ops = self.active_operations.write().await;
        *ops += 1;

        Ok(ResourceReservation {
            active_operations: self.active_operations.clone(),
        })
    }

    /// Increment event counter
    pub async fn record_event(&self) {
        let now = unix_utc_ms();
        let mut last_reset = self.last_reset_ms.write().await;
        let mut events = self.events_this_second.write().await;

        // Reset if new second
        if now - *last_reset >= 1000 {
            *events = 0;
            *last_reset = now;
        }

        *events += 1;
    }

    /// Get events this second
    async fn get_events_this_second(&self) -> u32 {
        let now = unix_utc_ms();
        let last_reset = *self.last_reset_ms.read().await;
        let events = *self.events_this_second.read().await;

        if now - last_reset >= 1000 {
            0
        } else {
            events
        }
    }

    /// Update memory estimate
    pub async fn update_memory(&self, memory_mb: u64) {
        let mut current = self.current_memory_mb.write().await;
        *current = memory_mb;
    }
}

impl Default for ResourceGovernor {
    fn default() -> Self {
        Self::new(ResourceLimits::default())
    }
}

/// Result of a resource check
#[derive(Debug, Clone)]
pub struct ResourceCheckResult {
    /// Whether resources are available
    pub allowed: bool,
    /// Issues preventing access
    pub issues: Vec<String>,
    /// Memory usage percentage
    pub memory_usage_percent: f64,
    /// Operation usage percentage
    pub operation_usage_percent: f64,
}

/// Reservation that releases resources on drop
pub struct ResourceReservation {
    active_operations: Arc<RwLock<u32>>,
}

impl Drop for ResourceReservation {
    fn drop(&mut self) {
        // Use block_on to release in sync context
        // In production, this should be done via an async task
        let ops = self.active_operations.clone();
        tokio::spawn(async move {
            let mut ops = ops.write().await;
            *ops = ops.saturating_sub(1);
        });
    }
}

// ============================================================================
// KERNEL
// ============================================================================

/// The Apotheosis Kernel - Root runtime coordinator
///
/// The Kernel owns:
/// - Event bus for deterministic event dispatch
/// - Hook chain integration for capability governance
/// - Resource governor for budget enforcement
/// - Identity and covenant state
pub struct Kernel {
    /// Event bus for kernel events
    pub event_bus: EventBus,
    /// Resource governor
    pub resource_governor: ResourceGovernor,
    /// Current identity (if loaded)
    pub identity: Arc<RwLock<Option<IdentityAnchor>>>,
    /// Active goals
    pub goals: Arc<RwLock<GoalStack>>,
    /// Covenant rules
    pub covenant: Arc<RwLock<Covenant>>,
    /// Kernel state
    pub state: Arc<RwLock<KernelState>>,
}

/// Kernel lifecycle state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum KernelState {
    /// Kernel is initializing
    #[default]
    Initializing,
    /// Kernel is ready to process events
    Ready,
    /// Kernel is processing a request
    Processing,
    /// Kernel is shutting down
    ShuttingDown,
    /// Kernel has terminated
    Terminated,
}


impl Kernel {
    /// Create a new kernel
    pub fn new() -> Self {
        Self {
            event_bus: EventBus::new(1000),
            resource_governor: ResourceGovernor::default(),
            identity: Arc::new(RwLock::new(None)),
            goals: Arc::new(RwLock::new(GoalStack::new("anonymous"))),
            covenant: Arc::new(RwLock::new(Covenant::new("anonymous"))),
            state: Arc::new(RwLock::new(KernelState::Initializing)),
        }
    }

    /// Initialize the kernel
    pub async fn initialize(&self) -> anyhow::Result<()> {
        info!("🔧 Kernel: Initializing...");

        // Dispatch initialization event
        self.event_bus
            .publish(KernelEvent::Custom {
                event_type: "kernel_init".to_string(),
                payload: serde_json::json!({"version": "1.0.0"}),
            })
            .await?;

        // Set state to ready
        let mut state = self.state.write().await;
        *state = KernelState::Ready;

        info!("✅ Kernel: Ready");
        Ok(())
    }

    /// Dispatch an event through the kernel
    pub async fn dispatch(&self, event: KernelEvent) -> anyhow::Result<u64> {
        // Check resources
        let check = self.resource_governor.check_resources().await;
        if !check.allowed {
            anyhow::bail!("Resource limit exceeded: {:?}", check.issues);
        }

        // Record event
        self.resource_governor.record_event().await;

        // Publish event
        let sequence = self.event_bus.publish(event).await?;

        Ok(sequence)
    }

    /// Load user identity
    pub async fn load_identity(&self, anchor: IdentityAnchor) -> anyhow::Result<()> {
        info!("👤 Kernel: Loading identity for {}", anchor.user_id);

        // Verify integrity
        if !anchor.verify_integrity() {
            anyhow::bail!("Identity integrity check failed");
        }

        // Store identity
        let mut identity = self.identity.write().await;
        *identity = Some(anchor.clone());

        // Update goals and covenant owner
        let mut goals = self.goals.write().await;
        *goals = GoalStack::new(&anchor.user_id);

        let mut covenant = self.covenant.write().await;
        *covenant = Covenant::new(&anchor.user_id);

        // Dispatch event
        self.dispatch(KernelEvent::IdentityLoaded {
            user_id: anchor.user_id,
            display_name: anchor.display_name,
        })
        .await?;

        Ok(())
    }

    /// Check covenant before action
    pub async fn check_covenant(&self, action_type: &str, context: &str) -> CovenantCheckResult {
        let covenant = self.covenant.read().await;

        // Check all active VETO rules
        let veto_rules = covenant.veto_rules();

        for rule in veto_rules {
            match &rule.constraint {
                crate::identity::CovenantConstraint::ForbiddenAction {
                    action_type: forbidden_type,
                    context: forbidden_context,
                } => {
                    if action_type == forbidden_type {
                        if let Some(ctx) = forbidden_context {
                            if context.contains(ctx) {
                                return CovenantCheckResult::veto(vec![
                                    crate::identity::CovenantViolation {
                                        rule_id: rule.rule_id.clone(),
                                        enforcement: rule.enforcement,
                                        message: format!(
                                            "Forbidden action: {} with context {}",
                                            action_type, ctx
                                        ),
                                    },
                                ]);
                            }
                        } else {
                            return CovenantCheckResult::veto(vec![
                                crate::identity::CovenantViolation {
                                    rule_id: rule.rule_id.clone(),
                                    enforcement: rule.enforcement,
                                    message: format!("Forbidden action: {}", action_type),
                                },
                            ]);
                        }
                    }
                }
                crate::identity::CovenantConstraint::Pattern { pattern, target } => {
                    if target == "content" && regex_matches(context, pattern) {
                        return CovenantCheckResult::veto(vec![
                            crate::identity::CovenantViolation {
                                rule_id: rule.rule_id.clone(),
                                enforcement: rule.enforcement,
                                message: format!("Pattern violation: {}", pattern),
                            },
                        ]);
                    }
                }
                _ => {}
            }
        }

        CovenantCheckResult::allow()
    }

    /// Add a covenant rule
    pub async fn add_covenant_rule(&self, rule: crate::identity::CovenantRule) {
        let mut covenant = self.covenant.write().await;
        covenant.add_rule(rule);
    }

    /// Get current identity
    pub async fn get_identity(&self) -> Option<IdentityAnchor> {
        let identity = self.identity.read().await;
        identity.clone()
    }

    /// Get current state
    pub async fn get_state(&self) -> KernelState {
        *self.state.read().await
    }

    /// Shutdown the kernel
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        info!("🔌 Kernel: Shutting down...");

        // Set state
        let mut state = self.state.write().await;
        *state = KernelState::ShuttingDown;
        drop(state);

        // Dispatch exit event
        self.dispatch(KernelEvent::Exit {
            reason: "shutdown".to_string(),
            success: true,
        })
        .await?;

        // Set terminated state
        let mut state = self.state.write().await;
        *state = KernelState::Terminated;

        info!("✅ Kernel: Terminated");
        Ok(())
    }
}

impl Default for Kernel {
    fn default() -> Self {
        Self::new()
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

/// Hash an event for audit
fn hash_event(event: &KernelEvent, sequence: u64, timestamp_ms: u64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(sequence.to_le_bytes());
    hasher.update(timestamp_ms.to_le_bytes());
    hasher.update(event.event_type().as_bytes());

    // Include event-specific data
    if let Ok(json) = serde_json::to_string(event) {
        hasher.update(json.as_bytes());
    }

    format!("{:x}", hasher.finalize())
}

/// Simple regex matching (without full regex crate dependency)
fn regex_matches(text: &str, pattern: &str) -> bool {
    // Simple wildcard matching for common patterns
    if pattern == ".*" {
        return true;
    }
    if pattern.starts_with(".*") && pattern.ends_with(".*") {
        let inner = &pattern[2..pattern.len() - 2];
        return text.contains(inner);
    }
    text.contains(pattern)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_event_types() {
        let event = KernelEvent::Intake {
            request_id: "req1".to_string(),
            task: "test task".to_string(),
            priority: 5,
        };
        assert_eq!(event.event_type(), "intake");

        let event = KernelEvent::Exit {
            reason: "done".to_string(),
            success: true,
        };
        assert_eq!(event.event_type(), "exit");
    }

    #[tokio::test]
    async fn test_event_bus_publish() {
        let bus = EventBus::new(100);

        let seq = bus
            .publish(KernelEvent::Intake {
                request_id: "req1".to_string(),
                task: "test".to_string(),
                priority: 1,
            })
            .await
            .unwrap();

        assert_eq!(seq, 1);
        assert_eq!(bus.history_count().await, 1);

        let seq2 = bus
            .publish(KernelEvent::Exit {
                reason: "done".to_string(),
                success: true,
            })
            .await
            .unwrap();

        assert_eq!(seq2, 2);
        assert_eq!(bus.history_count().await, 2);
    }

    #[tokio::test]
    async fn test_event_bus_history_limit() {
        let bus = EventBus::new(3);

        for i in 0..5 {
            bus.publish(KernelEvent::Custom {
                event_type: format!("event_{}", i),
                payload: serde_json::Value::Null,
            })
            .await
            .unwrap();
        }

        // Should only have last 3 events
        assert_eq!(bus.history_count().await, 3);

        let history = bus.history().await;
        assert_eq!(history[0].sequence, 3);
        assert_eq!(history[2].sequence, 5);
    }

    #[tokio::test]
    async fn test_resource_governor() {
        let limits = ResourceLimits {
            max_concurrent: 2,
            ..Default::default()
        };
        let governor = ResourceGovernor::new(limits);

        // Should allow first two operations
        let _res1 = governor.reserve().await.unwrap();
        let _res2 = governor.reserve().await.unwrap();

        // Third should fail
        let result = governor.reserve().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_kernel_initialization() {
        let kernel = Kernel::new();

        assert_eq!(kernel.get_state().await, KernelState::Initializing);

        kernel.initialize().await.unwrap();

        assert_eq!(kernel.get_state().await, KernelState::Ready);
    }

    #[tokio::test]
    async fn test_kernel_identity_loading() {
        let kernel = Kernel::new();
        kernel.initialize().await.unwrap();

        let anchor = IdentityAnchor::new("user123", "Test User", crate::identity::IdentityOrigin::Human);
        kernel.load_identity(anchor).await.unwrap();

        let identity = kernel.get_identity().await;
        assert!(identity.is_some());
        assert_eq!(identity.unwrap().user_id, "user123");
    }

    #[tokio::test]
    async fn test_kernel_covenant_check() {
        let kernel = Kernel::new();
        kernel.initialize().await.unwrap();

        // Add a VETO rule
        let rule = crate::identity::CovenantRule::new(
            "no_rm",
            "No rm commands",
            crate::identity::CovenantConstraint::ForbiddenAction {
                action_type: "shell".to_string(),
                context: Some("rm -rf".to_string()),
            },
        )
        .with_enforcement(crate::identity::EnforcementLevel::Veto);

        kernel.add_covenant_rule(rule).await;

        // Check allowed action
        let result = kernel.check_covenant("shell", "ls -la").await;
        assert!(result.allowed);

        // Check forbidden action
        let result = kernel.check_covenant("shell", "rm -rf /").await;
        assert!(!result.allowed);
        assert_eq!(result.decision, CovenantDecision::Veto);
    }

    #[tokio::test]
    async fn test_kernel_shutdown() {
        let kernel = Kernel::new();
        kernel.initialize().await.unwrap();

        assert_eq!(kernel.get_state().await, KernelState::Ready);

        kernel.shutdown().await.unwrap();

        assert_eq!(kernel.get_state().await, KernelState::Terminated);
    }
}
