// src/types.rs - Core types and data structures

use crate::fixed::Fixed64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Enhanced agent with full arsenal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedAgentCapabilities {
    /// MCP: Tool access
    pub mcp_tools: Vec<String>,

    /// A2A: Agent communication
    pub a2a_capabilities: Vec<String>,

    /// Reasoning methods
    pub reasoning_methods: Vec<ReasoningMethod>,

    /// Sub-agent generation
    pub can_spawn_sub_agents: bool,
    pub max_sub_agents: usize,

    /// Swarm capabilities
    pub swarm_modes: Vec<SwarmMode>,

    /// Memory access
    pub memory_tiers: Vec<MemoryTier>,

    /// Hook support
    pub hooks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReasoningMethod {
    ChainOfThought,
    TreeOfThought,
    GraphOfThought,
    ReAct,
    Reflexion,
    SovereignApotheosis,
    /// RLM (Recursive Language Models) for infinite context handling
    /// Decomposes complex problems into subproblems recursively
    RecursiveLanguage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmMode {
    Independent,
    Collaborative,
    HiveMind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryTier {
    Working,
    Episodic,
    Semantic,
    Procedural,
}

/// Base Dual Agentic Request
///
/// This is the primary input type for the BIZRA system. It contains
/// the task description, priority, and context for SAT validation
/// and PAT execution.
///
/// # Fields
///
/// - `task`: The primary task description (required)
/// - `priority`: Execution priority (Low/Medium/High/Critical)
/// - `context`: Key-value metadata for agent context
///
/// # Example
///
/// ```rust
/// use meta_alpha_dual_agentic::types::{DualAgenticRequest, Priority};
///
/// let request = DualAgenticRequest {
///     task: "Generate code review summary".to_string(),
///     priority: Priority::High,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DualAgenticRequest {
    /// User identifier for audit trails
    pub user_id: String,
    /// Primary task description (required)
    pub task: String,
    /// Optional requirements/constraints
    pub requirements: Vec<String>,
    /// Target artifact or deliverable
    pub target: String,
    /// Execution priority
    #[serde(default)]
    pub priority: Priority,
    /// Key-value context for agents
    #[serde(default)]
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum Priority {
    Low,
    #[default]
    Medium,
    High,
    Critical,
}

/// Enhanced request with full control
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnhancedDualAgenticRequest {
    /// Base request
    pub base: DualAgenticRequest,

    /// Advanced controls
    pub reasoning_preference: Option<ReasoningMethod>,
    #[serde(default)]
    pub enable_sub_agents: bool,
    pub enable_swarm: Option<SwarmMode>,
    pub mcp_tools_whitelist: Option<Vec<String>>,
    pub memory_context: Option<serde_json::Value>,
    #[serde(default)]
    pub hooks_config: HashMap<String, serde_json::Value>,

    /// Slash command support
    pub slash_command: Option<SlashCommand>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SlashCommand {
    Reason { method: ReasoningMethod },
    Spawn { role: String, task: String },
    Swarm { count: usize, mode: SwarmMode },
    Memory { tier: MemoryTier, query: String },
    Hook { name: String, action: HookAction },
    Tools { filter: String },
    Delegate { agent: String, task: String },
    Synthesize,
    Reflect { depth: usize },
    Export { format: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookAction {
    Enable,
    Disable,
    Configure(serde_json::Value),
}

/// Dual Agentic Response - Output from BIZRA System
///
/// Contains the aggregated results from PAT execution,
/// SAT validation contributions, and quality metrics.
///
/// # Key Metrics
///
/// - `synergy_score`: Measure of agent collaboration quality (0.0-1.0)
/// - `ihsan_score`: Ethical excellence score verified by FATE (0.0-1.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualAgenticResponse {
    /// Contributions from PAT (7 agents)
    pub pat_contributions: Vec<String>,
    /// Validation feedback from SAT (5 agents)
    pub sat_contributions: Vec<String>,
    /// Agent collaboration quality (Fixed64 for determinism)
    pub synergy_score: Fixed64,
    /// Ihsān ethical score (0.0-1.0, must meet threshold)
    pub ihsan_score: Fixed64,
    /// Total request latency
    #[serde(with = "duration_serde")]
    pub latency: Duration,
    /// Additional metadata (JSON)
    pub meta: serde_json::Value,
}

/// Agent Execution Result
///
/// Individual contribution from a single PAT or SAT agent.
/// Uses `Fixed64` for deterministic confidence scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Agent identifier (e.g., "strategic_visionary")
    pub agent_name: String,
    /// Agent's contribution text
    pub contribution: String,
    /// Confidence in contribution (Fixed64, 0.0-1.0)
    pub confidence: Fixed64,
    /// Ihsān score (Quality index)
    pub ihsan_score: Fixed64,
    /// Time taken for execution
    #[serde(with = "duration_serde")]
    pub execution_time: Duration,
    /// Rich metadata for tracing and visualization
    pub metadata: HashMap<String, String>,
}

/// Primordial Hardware Metrics (Bare Metal Telemetry)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareState {
    pub cpu_cores: usize,
    pub memory_total_mb: u64,
    pub tpm_active: bool,
    pub secure_boot: bool,
    pub instruction_set: String,
    pub entropy_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterMode {
    Real,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterModes {
    pub pat: AdapterMode,
    pub sat: AdapterMode,
    pub mcp: AdapterMode,
    pub a2a: AdapterMode,
    pub reasoning: AdapterMode,
}

impl AdapterModes {
    /// Get current adapter modes from environment.
    ///
    /// Simulation mode has been removed. Always returns Real.
    pub fn current() -> Self {
        let mode = AdapterMode::Real;
        Self {
            pat: mode.clone(),
            sat: mode.clone(),
            mcp: mode.clone(),
            a2a: mode.clone(),
            reasoning: mode,
        }
    }
}

// Custom Duration serialization
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}
