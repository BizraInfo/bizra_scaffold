// src/identity.rs - Identity Memory System for Apotheosis Node
//
// Giants Protocol Synthesis:
// - Al-Ghazali: Knowledge-Action Unity - IdentityAnchor binds who you are to what you do
// - Ibn Khaldun: Asabiyyah Lifecycle - Memory decay curves personalized per user domain
// - Tulving: Encoding Specificity - Identity memory retrieval context-bound
// - Dijkstra: Formal Methods - Covenant rules formally verified before execution
// - Kalman: State Estimation - Goal progress tracking via filtered measurements

use crate::fixed::Fixed64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// IDENTITY ANCHOR - The Soul of the Node
// ============================================================================

/// Origin classification for identity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum IdentityOrigin {
    /// Human user identity
    #[default]
    Human,
    /// Agent identity (PAT/SAT member)
    Agent,
    /// System identity (kernel, service)
    System,
    /// Federation node identity
    FederationNode,
}


/// IdentityAnchor - Stable user identity binding
///
/// The IdentityAnchor is the "soul" of the Apotheosis Node. It binds the user's
/// identity to all experiences, memories, and actions. This creates continuity
/// across sessions - the agent doesn't start fresh each time.
///
/// # Giants Protocol
/// - Al-Ghazali: Binds knowledge (who you are) to action (what you do)
/// - Tulving: Context-bound retrieval ensures encoding specificity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityAnchor {
    /// Unique identifier for this identity
    pub user_id: String,
    /// Human-readable display name
    pub display_name: String,
    /// Origin classification (Human, Agent, System)
    pub origin: IdentityOrigin,
    /// Creation timestamp (Unix epoch nanoseconds)
    pub created_at: u64,
    /// Core values and principles that guide behavior
    pub principles: Vec<String>,
    /// Link to Resonance Mesh wisdom root (if any)
    pub wisdom_root_id: Option<String>,
    /// User-specific metadata (skills, preferences, etc.)
    pub metadata: HashMap<String, serde_json::Value>,
    /// SHA-256 hash of the identity for integrity verification
    pub identity_hash: String,
}

impl Default for IdentityAnchor {
    fn default() -> Self {
        Self::new("anonymous", "Anonymous User", IdentityOrigin::Human)
    }
}

impl IdentityAnchor {
    /// Create a new IdentityAnchor
    pub fn new(user_id: &str, display_name: &str, origin: IdentityOrigin) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let mut anchor = Self {
            user_id: user_id.to_string(),
            display_name: display_name.to_string(),
            origin,
            created_at: now,
            principles: Vec::new(),
            wisdom_root_id: None,
            metadata: HashMap::new(),
            identity_hash: String::new(),
        };

        anchor.identity_hash = anchor.compute_hash();
        anchor
    }

    /// Add a principle to the identity
    pub fn with_principle(mut self, principle: &str) -> Self {
        self.principles.push(principle.to_string());
        self.identity_hash = self.compute_hash();
        self
    }

    /// Link to a Resonance Mesh wisdom root
    pub fn with_wisdom_root(mut self, root_id: &str) -> Self {
        self.wisdom_root_id = Some(root_id.to_string());
        self.identity_hash = self.compute_hash();
        self
    }

    /// Compute deterministic SHA-256 hash of the identity
    fn compute_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.user_id.as_bytes());
        hasher.update(self.display_name.as_bytes());
        hasher.update([self.origin as u8]);
        hasher.update(self.created_at.to_le_bytes());
        for p in &self.principles {
            hasher.update(p.as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// Verify identity integrity
    pub fn verify_integrity(&self) -> bool {
        self.compute_hash() == self.identity_hash
    }
}

// ============================================================================
// GOAL STACK - Active Goals with Priority and Progress
// ============================================================================

/// Goal priority classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
#[derive(Default)]
pub enum GoalPriority {
    /// Low priority - nice to have
    Low = 0,
    /// Medium priority - should do
    #[default]
    Medium = 1,
    /// High priority - must do soon
    High = 2,
    /// Critical priority - do immediately
    Critical = 3,
}


/// Goal time horizon
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum GoalHorizon {
    /// Immediate - within this session
    Immediate,
    /// Short-term - within a day
    #[default]
    Short,
    /// Medium-term - within a week
    Medium,
    /// Long-term - weeks to months
    Long,
}


/// Goal status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum GoalStatus {
    /// Not yet started
    #[default]
    Pending,
    /// Currently being worked on
    Active,
    /// Temporarily paused
    Paused,
    /// Successfully completed
    Completed,
    /// Abandoned or cancelled
    Abandoned,
    /// Blocked by dependency or constraint
    Blocked,
}


/// Kalman state for convergence detection and progress tracking
///
/// # Giants Protocol
/// - Kalman: State estimation via filtered measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanState {
    /// Current estimated progress (0.0 - 1.0)
    pub estimate: f64,
    /// Error covariance (uncertainty)
    pub error_covariance: f64,
    /// Process noise (expected variance in progress)
    pub process_noise: f64,
    /// Measurement noise (expected variance in measurements)
    pub measurement_noise: f64,
}

impl Default for KalmanState {
    fn default() -> Self {
        Self {
            estimate: 0.0,
            error_covariance: 1.0,
            process_noise: 0.001,
            measurement_noise: 0.01,
        }
    }
}

impl KalmanState {
    /// Update the Kalman filter with a new measurement
    ///
    /// Returns the Kalman gain as a convergence indicator
    pub fn update(&mut self, measurement: f64) -> f64 {
        // Prediction step
        let predicted_error = self.error_covariance + self.process_noise;

        // Update step
        let kalman_gain = predicted_error / (predicted_error + self.measurement_noise);
        self.estimate += kalman_gain * (measurement - self.estimate);
        self.error_covariance = (1.0 - kalman_gain) * predicted_error;

        kalman_gain
    }

    /// Check if the filter has converged below threshold
    pub fn has_converged(&self, threshold: f64) -> bool {
        self.error_covariance < threshold
    }

    /// Convert to Fixed64 for deterministic operations
    pub fn estimate_fixed(&self) -> Fixed64 {
        Fixed64::from_f64(self.estimate.clamp(0.0, 1.0))
    }
}

/// Goal - A tracked objective with progress and priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Unique identifier for this goal
    pub goal_id: String,
    /// Human-readable description
    pub description: String,
    /// Priority classification
    pub priority: GoalPriority,
    /// Time horizon
    pub horizon: GoalHorizon,
    /// Current status
    pub status: GoalStatus,
    /// Kalman-filtered progress tracking
    pub progress: KalmanState,
    /// Parent goal (for hierarchical goals)
    pub parent_goal: Option<String>,
    /// Child goals (sub-objectives)
    pub child_goals: Vec<String>,
    /// Creation timestamp (Unix epoch nanoseconds)
    pub created_at: u64,
    /// Optional deadline (Unix epoch nanoseconds)
    pub deadline: Option<u64>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Goal-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Goal {
    /// Create a new goal
    pub fn new(goal_id: &str, description: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        Self {
            goal_id: goal_id.to_string(),
            description: description.to_string(),
            priority: GoalPriority::default(),
            horizon: GoalHorizon::default(),
            status: GoalStatus::default(),
            progress: KalmanState::default(),
            parent_goal: None,
            child_goals: Vec::new(),
            created_at: now,
            deadline: None,
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: GoalPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set horizon
    pub fn with_horizon(mut self, horizon: GoalHorizon) -> Self {
        self.horizon = horizon;
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline_ns: u64) -> Self {
        self.deadline = Some(deadline_ns);
        self
    }

    /// Update progress with a new measurement
    pub fn update_progress(&mut self, measurement: f64) -> f64 {
        self.progress.update(measurement)
    }

    /// Check if goal is complete (progress estimate >= 0.95)
    pub fn is_complete(&self) -> bool {
        self.status == GoalStatus::Completed || self.progress.estimate >= 0.95
    }

    /// Check if goal is overdue
    pub fn is_overdue(&self) -> bool {
        if let Some(deadline) = self.deadline {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            now > deadline
        } else {
            false
        }
    }
}

/// GoalStack - Active goals with priority and progress
///
/// The GoalStack tracks the user's objectives with hierarchical organization
/// and Kalman-filtered progress tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalStack {
    /// Identity anchor owning these goals
    pub owner_id: String,
    /// All goals indexed by ID
    pub goals: HashMap<String, Goal>,
    /// Root goals (no parent)
    pub root_goals: Vec<String>,
}

impl GoalStack {
    /// Create a new empty GoalStack for an identity
    pub fn new(owner_id: &str) -> Self {
        Self {
            owner_id: owner_id.to_string(),
            goals: HashMap::new(),
            root_goals: Vec::new(),
        }
    }

    /// Add a goal to the stack
    pub fn add_goal(&mut self, goal: Goal) {
        let goal_id = goal.goal_id.clone();
        let parent = goal.parent_goal.clone();

        self.goals.insert(goal_id.clone(), goal);

        if let Some(parent_id) = parent {
            // Add as child of parent
            if let Some(parent_goal) = self.goals.get_mut(&parent_id) {
                parent_goal.child_goals.push(goal_id);
            }
        } else {
            // Root goal
            self.root_goals.push(goal_id);
        }
    }

    /// Get a goal by ID
    pub fn get_goal(&self, goal_id: &str) -> Option<&Goal> {
        self.goals.get(goal_id)
    }

    /// Get mutable goal by ID
    pub fn get_goal_mut(&mut self, goal_id: &str) -> Option<&mut Goal> {
        self.goals.get_mut(goal_id)
    }

    /// Get active goals sorted by priority
    pub fn active_goals(&self) -> Vec<&Goal> {
        let mut active: Vec<_> = self
            .goals
            .values()
            .filter(|g| g.status == GoalStatus::Active)
            .collect();
        active.sort_by(|a, b| b.priority.cmp(&a.priority));
        active
    }

    /// Get critical goals
    pub fn critical_goals(&self) -> Vec<&Goal> {
        self.goals
            .values()
            .filter(|g| g.priority == GoalPriority::Critical && g.status == GoalStatus::Active)
            .collect()
    }

    /// Get overdue goals
    pub fn overdue_goals(&self) -> Vec<&Goal> {
        self.goals.values().filter(|g| g.is_overdue()).collect()
    }

    /// Calculate overall progress (weighted by priority)
    pub fn overall_progress(&self) -> f64 {
        let active_goals = self.active_goals();
        if active_goals.is_empty() {
            return 0.0;
        }

        let total_weight: f64 = active_goals
            .iter()
            .map(|g| (g.priority as u8 + 1) as f64)
            .sum();
        let weighted_sum: f64 = active_goals
            .iter()
            .map(|g| g.progress.estimate * (g.priority as u8 + 1) as f64)
            .sum();

        weighted_sum / total_weight
    }
}

// ============================================================================
// COVENANT - Immutable "Cannot Do" Rules
// ============================================================================

/// Enforcement level for covenant rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum EnforcementLevel {
    /// Log violation but allow action
    Log,
    /// Warn user but allow action
    Warn,
    /// Block action (soft VETO)
    #[default]
    Block,
    /// Hard VETO - absolutely forbidden
    Veto,
}


/// Constraint type for covenant rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CovenantConstraint {
    /// Pattern-based constraint (regex or glob)
    Pattern { pattern: String, target: String },
    /// Action-based constraint
    ForbiddenAction { action_type: String, context: Option<String> },
    /// Resource constraint
    ResourceLimit { resource: String, max_value: f64 },
    /// Time-based constraint
    TimeWindow { allowed_hours: Vec<u8>, timezone: String },
    /// Custom Z3 formula for formal verification
    Formal { z3_formula: String },
}

/// CovenantRule - A single immutable constraint
///
/// # Giants Protocol
/// - Dijkstra: Formal verification via Z3 formulas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovenantRule {
    /// Unique identifier for this rule
    pub rule_id: String,
    /// Human-readable description
    pub description: String,
    /// The constraint definition
    pub constraint: CovenantConstraint,
    /// Enforcement level
    pub enforcement: EnforcementLevel,
    /// Optional Z3 formula for formal verification
    pub z3_formula: Option<String>,
    /// Rule creation timestamp
    pub created_at: u64,
    /// Whether this rule is active
    pub active: bool,
    /// Rule metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl CovenantRule {
    /// Create a new covenant rule
    pub fn new(rule_id: &str, description: &str, constraint: CovenantConstraint) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        Self {
            rule_id: rule_id.to_string(),
            description: description.to_string(),
            constraint,
            enforcement: EnforcementLevel::default(),
            z3_formula: None,
            created_at: now,
            active: true,
            metadata: HashMap::new(),
        }
    }

    /// Set enforcement level
    pub fn with_enforcement(mut self, level: EnforcementLevel) -> Self {
        self.enforcement = level;
        self
    }

    /// Set Z3 formula for formal verification
    pub fn with_z3_formula(mut self, formula: &str) -> Self {
        self.z3_formula = Some(formula.to_string());
        self
    }
}

/// Covenant - The immutable ethics and constraints container
///
/// The Covenant defines what the system can NEVER do. These rules are
/// checked before every action and violations trigger immediate VETOs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Covenant {
    /// Identity anchor owning this covenant
    pub owner_id: String,
    /// All rules indexed by ID
    pub rules: HashMap<String, CovenantRule>,
    /// SHA-256 hash of the covenant for integrity
    pub covenant_hash: String,
    /// Version number (incremented on rule changes)
    pub version: u64,
}

impl Covenant {
    /// Create a new empty Covenant
    pub fn new(owner_id: &str) -> Self {
        let mut covenant = Self {
            owner_id: owner_id.to_string(),
            rules: HashMap::new(),
            covenant_hash: String::new(),
            version: 1,
        };
        covenant.covenant_hash = covenant.compute_hash();
        covenant
    }

    /// Add a rule to the covenant
    pub fn add_rule(&mut self, rule: CovenantRule) {
        self.rules.insert(rule.rule_id.clone(), rule);
        self.version += 1;
        self.covenant_hash = self.compute_hash();
    }

    /// Get a rule by ID
    pub fn get_rule(&self, rule_id: &str) -> Option<&CovenantRule> {
        self.rules.get(rule_id)
    }

    /// Get all active rules
    pub fn active_rules(&self) -> Vec<&CovenantRule> {
        self.rules.values().filter(|r| r.active).collect()
    }

    /// Get all VETO rules (highest enforcement)
    pub fn veto_rules(&self) -> Vec<&CovenantRule> {
        self.rules
            .values()
            .filter(|r| r.active && r.enforcement == EnforcementLevel::Veto)
            .collect()
    }

    /// Compute deterministic SHA-256 hash
    fn compute_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.owner_id.as_bytes());
        hasher.update(self.version.to_le_bytes());

        // Sort rule IDs for determinism
        let mut rule_ids: Vec<_> = self.rules.keys().collect();
        rule_ids.sort();

        for rule_id in rule_ids {
            if let Some(rule) = self.rules.get(rule_id) {
                hasher.update(rule.rule_id.as_bytes());
                hasher.update(rule.description.as_bytes());
                hasher.update([rule.enforcement as u8]);
            }
        }

        format!("{:x}", hasher.finalize())
    }

    /// Verify covenant integrity
    pub fn verify_integrity(&self) -> bool {
        self.compute_hash() == self.covenant_hash
    }
}

/// Result of a covenant check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovenantCheckResult {
    /// Whether the action is allowed
    pub allowed: bool,
    /// Violated rules (if any)
    pub violations: Vec<CovenantViolation>,
    /// Overall enforcement decision
    pub decision: CovenantDecision,
}

/// A single covenant violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovenantViolation {
    /// Rule that was violated
    pub rule_id: String,
    /// Enforcement level of the rule
    pub enforcement: EnforcementLevel,
    /// Human-readable violation message
    pub message: String,
}

/// Covenant decision outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CovenantDecision {
    /// Action is allowed
    Allow,
    /// Action logged but allowed
    AllowWithLog,
    /// Action warned but allowed
    AllowWithWarn,
    /// Action blocked (soft VETO)
    Block,
    /// Action absolutely forbidden (hard VETO)
    Veto,
}

impl CovenantCheckResult {
    /// Create an "allow" result
    pub fn allow() -> Self {
        Self {
            allowed: true,
            violations: Vec::new(),
            decision: CovenantDecision::Allow,
        }
    }

    /// Create a "veto" result
    pub fn veto(violations: Vec<CovenantViolation>) -> Self {
        Self {
            allowed: false,
            violations,
            decision: CovenantDecision::Veto,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_anchor_creation() {
        let anchor = IdentityAnchor::new("user123", "Test User", IdentityOrigin::Human);
        assert_eq!(anchor.user_id, "user123");
        assert_eq!(anchor.display_name, "Test User");
        assert_eq!(anchor.origin, IdentityOrigin::Human);
        assert!(!anchor.identity_hash.is_empty());
        assert!(anchor.verify_integrity());
    }

    #[test]
    fn test_identity_anchor_with_principles() {
        let anchor = IdentityAnchor::new("user123", "Test User", IdentityOrigin::Human)
            .with_principle("Always verify before trusting")
            .with_principle("Prioritize user safety");

        assert_eq!(anchor.principles.len(), 2);
        assert!(anchor.verify_integrity());
    }

    #[test]
    fn test_kalman_state_update() {
        let mut kalman = KalmanState::default();
        assert_eq!(kalman.estimate, 0.0);

        // Update with measurements
        kalman.update(0.5);
        assert!(kalman.estimate > 0.0);
        assert!(kalman.estimate < 0.5);

        kalman.update(0.5);
        kalman.update(0.5);
        kalman.update(0.5);

        // Should converge towards 0.5
        assert!(kalman.estimate > 0.3);
        assert!(kalman.estimate < 0.6);
    }

    #[test]
    fn test_goal_creation_and_progress() {
        let mut goal = Goal::new("goal1", "Complete the project")
            .with_priority(GoalPriority::High)
            .with_horizon(GoalHorizon::Medium);

        assert_eq!(goal.priority, GoalPriority::High);
        assert_eq!(goal.horizon, GoalHorizon::Medium);
        assert!(!goal.is_complete());

        // Update progress
        for _ in 0..10 {
            goal.update_progress(1.0);
        }

        assert!(goal.is_complete());
    }

    #[test]
    fn test_goal_stack_operations() {
        let mut stack = GoalStack::new("user123");

        let goal1 = Goal::new("goal1", "First goal").with_priority(GoalPriority::High);
        let mut goal2 = Goal::new("goal2", "Second goal").with_priority(GoalPriority::Critical);
        goal2.status = GoalStatus::Active;

        stack.add_goal(goal1);
        stack.add_goal(goal2);

        assert_eq!(stack.goals.len(), 2);
        assert_eq!(stack.root_goals.len(), 2);

        let active = stack.active_goals();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].priority, GoalPriority::Critical);
    }

    #[test]
    fn test_covenant_rule_creation() {
        let rule = CovenantRule::new(
            "no_rm",
            "Never execute rm -rf commands",
            CovenantConstraint::ForbiddenAction {
                action_type: "shell_command".to_string(),
                context: Some("rm -rf".to_string()),
            },
        )
        .with_enforcement(EnforcementLevel::Veto);

        assert_eq!(rule.enforcement, EnforcementLevel::Veto);
        assert!(rule.active);
    }

    #[test]
    fn test_covenant_integrity() {
        let mut covenant = Covenant::new("user123");

        let rule = CovenantRule::new(
            "rule1",
            "Test rule",
            CovenantConstraint::Pattern {
                pattern: ".*dangerous.*".to_string(),
                target: "content".to_string(),
            },
        );

        covenant.add_rule(rule);

        assert!(covenant.verify_integrity());
        assert_eq!(covenant.version, 2); // Started at 1, incremented on add
    }

    #[test]
    fn test_covenant_veto_rules() {
        let mut covenant = Covenant::new("user123");

        let veto_rule = CovenantRule::new(
            "veto1",
            "Hard VETO rule",
            CovenantConstraint::ForbiddenAction {
                action_type: "delete".to_string(),
                context: None,
            },
        )
        .with_enforcement(EnforcementLevel::Veto);

        let warn_rule = CovenantRule::new(
            "warn1",
            "Warning rule",
            CovenantConstraint::Pattern {
                pattern: ".*".to_string(),
                target: "test".to_string(),
            },
        )
        .with_enforcement(EnforcementLevel::Warn);

        covenant.add_rule(veto_rule);
        covenant.add_rule(warn_rule);

        let veto_rules = covenant.veto_rules();
        assert_eq!(veto_rules.len(), 1);
        assert_eq!(veto_rules[0].rule_id, "veto1");
    }
}
