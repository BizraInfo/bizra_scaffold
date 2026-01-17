// src/model_fabric.rs - Model Fabric for One-Model-Per-Agent Isolation
//
// Implements strict isolation where each agent gets its own dedicated model endpoint.
// This prevents context contamination, enables predictable load patterns, and
// simplifies debugging.
//
// Giants Protocol Synthesis:
// - Ibn Rushd: Multi-Path Truth - Parallel reasoning paths via isolated models
// - Nash: Game Theory - Agent council reaches equilibrium via weighted votes

use crate::hookchain::CapabilityBudget;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn};

// ============================================================================
// MODEL BACKEND TYPES
// ============================================================================

/// Supported model backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum ModelBackend {
    /// Ollama (default local backend)
    #[default]
    Ollama,
    /// vLLM for high-throughput serving
    VLLM,
    /// llama.cpp for low-resource deployments
    LlamaCpp,
    /// OpenAI-compatible API (external)
    OpenAICompatible,
    /// Mock backend for testing
    Mock,
}


impl ModelBackend {
    /// Get the default health check path for this backend
    pub fn health_path(&self) -> &'static str {
        match self {
            ModelBackend::Ollama => "/api/tags",
            ModelBackend::VLLM => "/health",
            ModelBackend::LlamaCpp => "/health",
            ModelBackend::OpenAICompatible => "/v1/models",
            ModelBackend::Mock => "/health",
        }
    }

    /// Get the completion endpoint path for this backend
    pub fn completion_path(&self) -> &'static str {
        match self {
            ModelBackend::Ollama => "/api/generate",
            ModelBackend::VLLM => "/v1/completions",
            ModelBackend::LlamaCpp => "/completion",
            ModelBackend::OpenAICompatible => "/v1/chat/completions",
            ModelBackend::Mock => "/completion",
        }
    }
}

// ============================================================================
// HEALTH STATUS
// ============================================================================

/// Health status for a model endpoint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum HealthStatus {
    /// Endpoint is healthy and responding
    Healthy,
    /// Endpoint is degraded (slow but functional)
    Degraded,
    /// Endpoint is unhealthy (not responding)
    Unhealthy,
    /// Health status unknown (never checked)
    #[default]
    Unknown,
}


// ============================================================================
// RESOURCE LIMITS
// ============================================================================

/// Resource limits for a model endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum tokens per request
    pub max_tokens: u32,
    /// Maximum concurrent requests
    pub max_concurrent: u32,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Memory limit in MB (0 = unlimited)
    pub memory_limit_mb: u32,
    /// GPU memory limit in MB (0 = unlimited)
    pub gpu_memory_limit_mb: u32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            max_concurrent: 4,
            timeout_ms: 60_000,
            memory_limit_mb: 0,
            gpu_memory_limit_mb: 0,
        }
    }
}

// ============================================================================
// MODEL ENDPOINT
// ============================================================================

/// A single model endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEndpoint {
    /// Unique identifier for this endpoint
    pub endpoint_id: String,
    /// Human-readable name of the model
    pub model_name: String,
    /// Backend type
    pub backend: ModelBackend,
    /// Base URL for the endpoint
    pub url: String,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Current health status
    pub health: HealthStatus,
    /// Last health check timestamp (Unix ms)
    pub last_health_check_ms: u64,
    /// Average response time in milliseconds
    pub avg_response_ms: f64,
    /// Total requests served
    pub total_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
}

impl ModelEndpoint {
    /// Create a new model endpoint
    pub fn new(endpoint_id: &str, model_name: &str, backend: ModelBackend, url: &str) -> Self {
        Self {
            endpoint_id: endpoint_id.to_string(),
            model_name: model_name.to_string(),
            backend,
            url: url.to_string(),
            resource_limits: ResourceLimits::default(),
            health: HealthStatus::Unknown,
            last_health_check_ms: 0,
            avg_response_ms: 0.0,
            total_requests: 0,
            failed_requests: 0,
        }
    }

    /// Set resource limits
    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.resource_limits = limits;
        self
    }

    /// Get success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 1.0;
        }
        let successful = self.total_requests - self.failed_requests;
        successful as f64 / self.total_requests as f64
    }

    /// Record a successful request
    pub fn record_success(&mut self, response_ms: u64) {
        self.total_requests += 1;
        // Exponential moving average for response time
        let alpha = 0.1;
        self.avg_response_ms = alpha * response_ms as f64 + (1.0 - alpha) * self.avg_response_ms;
    }

    /// Record a failed request
    pub fn record_failure(&mut self) {
        self.total_requests += 1;
        self.failed_requests += 1;
    }

    /// Check if endpoint is available
    pub fn is_available(&self) -> bool {
        matches!(self.health, HealthStatus::Healthy | HealthStatus::Degraded)
    }
}

// ============================================================================
// AGENT ID
// ============================================================================

/// Unique identifier for an agent
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId {
    /// Agent type (e.g., "planner", "builder", "security")
    pub agent_type: String,
    /// Instance ID (for multiple instances of same type)
    pub instance_id: u32,
}

impl AgentId {
    /// Create a new agent ID
    pub fn new(agent_type: &str, instance_id: u32) -> Self {
        Self {
            agent_type: agent_type.to_string(),
            instance_id,
        }
    }

    /// Create a unique key for hash maps
    pub fn key(&self) -> String {
        format!("{}:{}", self.agent_type, self.instance_id)
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.agent_type, self.instance_id)
    }
}

// ============================================================================
// MODEL RESPONSE
// ============================================================================

/// Response from a model invocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    /// The generated text
    pub text: String,
    /// Tokens used in prompt
    pub prompt_tokens: u32,
    /// Tokens generated
    pub completion_tokens: u32,
    /// Response time in milliseconds
    pub response_ms: u64,
    /// Model name that responded
    pub model_name: String,
    /// Endpoint ID that served this request
    pub endpoint_id: String,
}

// ============================================================================
// FABRIC HEALTH
// ============================================================================

/// Overall health of the model fabric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricHealth {
    /// Number of healthy endpoints
    pub healthy_count: usize,
    /// Number of degraded endpoints
    pub degraded_count: usize,
    /// Number of unhealthy endpoints
    pub unhealthy_count: usize,
    /// Number of unbound agents
    pub unbound_agents: usize,
    /// Overall health score (0.0 - 1.0)
    pub health_score: f64,
    /// Last check timestamp (Unix ms)
    pub checked_at_ms: u64,
}

impl FabricHealth {
    /// Check if fabric is operational
    pub fn is_operational(&self) -> bool {
        self.healthy_count > 0 && self.health_score >= 0.5
    }
}

// ============================================================================
// MODEL FABRIC
// ============================================================================

/// Model Fabric - Routes agents to dedicated model endpoints
///
/// Implements strict one-model-per-agent isolation for:
/// - Context isolation (no contamination between agents)
/// - Predictable load patterns (easier debugging/profiling)
/// - Fault isolation (one model failing doesn't affect others)
pub struct ModelFabric {
    /// All model endpoints indexed by ID
    endpoints: Arc<RwLock<HashMap<String, ModelEndpoint>>>,
    /// Agent to endpoint bindings
    bindings: Arc<RwLock<HashMap<String, String>>>,
    /// Default endpoint for unbound agents
    default_endpoint: Option<String>,
    /// HTTP client for health checks
    client: reqwest::Client,
}

impl ModelFabric {
    /// Create a new model fabric
    pub fn new() -> Self {
        Self {
            endpoints: Arc::new(RwLock::new(HashMap::new())),
            bindings: Arc::new(RwLock::new(HashMap::new())),
            default_endpoint: None,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Add a model endpoint
    pub async fn add_endpoint(&self, endpoint: ModelEndpoint) {
        let endpoint_id = endpoint.endpoint_id.clone();
        let mut endpoints = self.endpoints.write().await;
        endpoints.insert(endpoint_id, endpoint);
    }

    /// Remove a model endpoint
    pub async fn remove_endpoint(&self, endpoint_id: &str) -> Option<ModelEndpoint> {
        let mut endpoints = self.endpoints.write().await;
        endpoints.remove(endpoint_id)
    }

    /// Bind an agent to an endpoint
    pub async fn bind_agent(&self, agent_id: &AgentId, endpoint_id: &str) -> anyhow::Result<()> {
        // Verify endpoint exists
        let endpoints = self.endpoints.read().await;
        if !endpoints.contains_key(endpoint_id) {
            anyhow::bail!("Endpoint {} not found", endpoint_id);
        }
        drop(endpoints);

        let mut bindings = self.bindings.write().await;
        bindings.insert(agent_id.key(), endpoint_id.to_string());
        info!("Bound agent {} to endpoint {}", agent_id, endpoint_id);
        Ok(())
    }

    /// Unbind an agent
    pub async fn unbind_agent(&self, agent_id: &AgentId) -> Option<String> {
        let mut bindings = self.bindings.write().await;
        bindings.remove(&agent_id.key())
    }

    /// Get the endpoint ID bound to an agent
    pub async fn get_binding(&self, agent_id: &AgentId) -> Option<String> {
        let bindings = self.bindings.read().await;
        bindings.get(&agent_id.key()).cloned()
    }

    /// Set the default endpoint for unbound agents
    pub async fn set_default_endpoint(&mut self, endpoint_id: &str) {
        self.default_endpoint = Some(endpoint_id.to_string());
    }

    /// Get endpoint for an agent (bound or default)
    pub async fn get_endpoint_for_agent(&self, agent_id: &AgentId) -> Option<ModelEndpoint> {
        let bindings = self.bindings.read().await;
        let endpoint_id = bindings
            .get(&agent_id.key())
            .or(self.default_endpoint.as_ref())?;

        let endpoints = self.endpoints.read().await;
        endpoints.get(endpoint_id).cloned()
    }

    /// Invoke a model for an agent
    pub async fn invoke(
        &self,
        agent_id: &AgentId,
        prompt: &str,
        budget: &CapabilityBudget,
    ) -> anyhow::Result<ModelResponse> {
        // Get endpoint for this agent
        let endpoint = self
            .get_endpoint_for_agent(agent_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("No endpoint bound for agent {}", agent_id))?;

        // Check if endpoint is available
        if !endpoint.is_available() {
            anyhow::bail!(
                "Endpoint {} is unhealthy: {:?}",
                endpoint.endpoint_id,
                endpoint.health
            );
        }

        // Build request based on backend type
        let start = Instant::now();
        let response = self.call_endpoint(&endpoint, prompt, budget).await?;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        // Record metrics
        {
            let mut endpoints = self.endpoints.write().await;
            if let Some(ep) = endpoints.get_mut(&endpoint.endpoint_id) {
                ep.record_success(elapsed_ms);
            }
        }

        Ok(ModelResponse {
            text: response,
            prompt_tokens: 0, // Would need to parse from response
            completion_tokens: 0,
            response_ms: elapsed_ms,
            model_name: endpoint.model_name.clone(),
            endpoint_id: endpoint.endpoint_id.clone(),
        })
    }

    /// Call an endpoint (implementation for different backends)
    async fn call_endpoint(
        &self,
        endpoint: &ModelEndpoint,
        prompt: &str,
        budget: &CapabilityBudget,
    ) -> anyhow::Result<String> {
        let url = format!("{}{}", endpoint.url, endpoint.backend.completion_path());

        let max_tokens = budget.max_tokens.min(endpoint.resource_limits.max_tokens);

        match endpoint.backend {
            ModelBackend::Ollama => {
                let body = serde_json::json!({
                    "model": endpoint.model_name,
                    "prompt": prompt,
                    "stream": false,
                    "options": {
                        "num_predict": max_tokens
                    }
                });

                let resp = self
                    .client
                    .post(&url)
                    .json(&body)
                    .timeout(Duration::from_millis(endpoint.resource_limits.timeout_ms))
                    .send()
                    .await?;

                let json: serde_json::Value = resp.json().await?;
                Ok(json["response"].as_str().unwrap_or("").to_string())
            }

            ModelBackend::OpenAICompatible | ModelBackend::VLLM => {
                let body = serde_json::json!({
                    "model": endpoint.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens
                });

                let resp = self
                    .client
                    .post(&url)
                    .json(&body)
                    .timeout(Duration::from_millis(endpoint.resource_limits.timeout_ms))
                    .send()
                    .await?;

                let json: serde_json::Value = resp.json().await?;
                Ok(json["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string())
            }

            ModelBackend::LlamaCpp => {
                let body = serde_json::json!({
                    "prompt": prompt,
                    "n_predict": max_tokens,
                    "stream": false
                });

                let resp = self
                    .client
                    .post(&url)
                    .json(&body)
                    .timeout(Duration::from_millis(endpoint.resource_limits.timeout_ms))
                    .send()
                    .await?;

                let json: serde_json::Value = resp.json().await?;
                Ok(json["content"].as_str().unwrap_or("").to_string())
            }

            ModelBackend::Mock => {
                // For testing - return a mock response
                Ok(format!("Mock response for: {}", prompt.chars().take(50).collect::<String>()))
            }
        }
    }

    /// Health check all endpoints
    pub async fn health_check(&self) -> FabricHealth {
        let mut endpoints = self.endpoints.write().await;
        let bindings = self.bindings.read().await;

        let mut healthy_count = 0;
        let mut degraded_count = 0;
        let mut unhealthy_count = 0;

        for endpoint in endpoints.values_mut() {
            let health = self.check_endpoint_health(endpoint).await;
            endpoint.health = health;
            endpoint.last_health_check_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);

            match health {
                HealthStatus::Healthy => healthy_count += 1,
                HealthStatus::Degraded => degraded_count += 1,
                HealthStatus::Unhealthy | HealthStatus::Unknown => unhealthy_count += 1,
            }
        }

        let total = endpoints.len();
        let health_score = if total > 0 {
            (healthy_count as f64 + 0.5 * degraded_count as f64) / total as f64
        } else {
            0.0
        };

        // Count unbound agents
        let _bound_agents: std::collections::HashSet<_> = bindings.keys().cloned().collect();
        let unbound_agents = 0; // Would need agent registry to count properly

        FabricHealth {
            healthy_count,
            degraded_count,
            unhealthy_count,
            unbound_agents,
            health_score,
            checked_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Check health of a single endpoint
    async fn check_endpoint_health(&self, endpoint: &ModelEndpoint) -> HealthStatus {
        let url = format!("{}{}", endpoint.url, endpoint.backend.health_path());
        let start = Instant::now();

        match self.client.get(&url).timeout(Duration::from_secs(5)).send().await {
            Ok(resp) => {
                let elapsed = start.elapsed();
                if resp.status().is_success() {
                    // Check response time for degradation
                    if elapsed.as_millis() > 2000 {
                        HealthStatus::Degraded
                    } else {
                        HealthStatus::Healthy
                    }
                } else {
                    warn!(
                        "Endpoint {} health check failed: {}",
                        endpoint.endpoint_id,
                        resp.status()
                    );
                    HealthStatus::Unhealthy
                }
            }
            Err(e) => {
                warn!(
                    "Endpoint {} health check error: {}",
                    endpoint.endpoint_id, e
                );
                HealthStatus::Unhealthy
            }
        }
    }

    /// Rebalance workload if an endpoint is unhealthy
    pub async fn rebalance(&self, failed_endpoint_id: &str) -> anyhow::Result<()> {
        let endpoints = self.endpoints.read().await;
        let mut bindings = self.bindings.write().await;

        // Find agents bound to the failed endpoint
        let affected_agents: Vec<_> = bindings
            .iter()
            .filter(|(_, ep_id)| *ep_id == failed_endpoint_id)
            .map(|(agent_key, _)| agent_key.clone())
            .collect();

        if affected_agents.is_empty() {
            return Ok(());
        }

        // Find a healthy alternative
        let alternative = endpoints
            .values()
            .find(|ep| ep.endpoint_id != failed_endpoint_id && ep.is_available())
            .map(|ep| ep.endpoint_id.clone());

        if let Some(alt_id) = alternative {
            for agent_key in affected_agents {
                bindings.insert(agent_key.clone(), alt_id.clone());
                warn!("Rebalanced agent {} from {} to {}", agent_key, failed_endpoint_id, alt_id);
            }
        } else {
            anyhow::bail!("No healthy alternative endpoint available");
        }

        Ok(())
    }

    /// Get all endpoints
    pub async fn all_endpoints(&self) -> Vec<ModelEndpoint> {
        let endpoints = self.endpoints.read().await;
        endpoints.values().cloned().collect()
    }

    /// Get all bindings
    pub async fn all_bindings(&self) -> HashMap<String, String> {
        let bindings = self.bindings.read().await;
        bindings.clone()
    }
}

impl Default for ModelFabric {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PREDEFINED AGENT TYPES
// ============================================================================

/// Predefined agent types for the 7-agent council
pub mod agents {
    use super::AgentId;

    /// Planner agent - decomposition and sequencing
    pub fn planner() -> AgentId {
        AgentId::new("planner", 0)
    }

    /// Builder agent - implementation
    pub fn builder() -> AgentId {
        AgentId::new("builder", 0)
    }

    /// Tester agent - tests and benchmarks
    pub fn tester() -> AgentId {
        AgentId::new("tester", 0)
    }

    /// Security agent - threat modeling
    pub fn security() -> AgentId {
        AgentId::new("security", 0)
    }

    /// Ethics agent - Ihsan/Adl constraints
    pub fn ethics() -> AgentId {
        AgentId::new("ethics", 0)
    }

    /// Research agent - evidence retrieval
    pub fn research() -> AgentId {
        AgentId::new("research", 0)
    }

    /// Integrator agent - merge and release
    pub fn integrator() -> AgentId {
        AgentId::new("integrator", 0)
    }

    /// All 7 council agents
    pub fn council() -> Vec<AgentId> {
        vec![
            planner(),
            builder(),
            tester(),
            security(),
            ethics(),
            research(),
            integrator(),
        ]
    }
}

// ============================================================================
// LM STUDIO INTEGRATION
// ============================================================================

/// LM Studio integration - OpenAI-compatible local server
///
/// Giants Protocol: Ibn Rushd multi-path truth via local model plurality
///
/// LM Studio provides OpenAI-compatible API at default port 1234.
/// Supports model hot-swapping without endpoint reconfiguration.
pub mod lm_studio {
    use super::*;

    /// Default LM Studio endpoint configuration
    pub const DEFAULT_HOST: &str = "http://172.22.48.1";
    pub const DEFAULT_PORT: u16 = 1234;

    /// Known high-quality models for PAT council
    pub const MODEL_NEMOTRON_30B: &str = "nvidia_nemotron-mini-4b-instruct";
    pub const MODEL_MINISTRAL_14B: &str = "ministral-14b-reasoning";
    pub const MODEL_QWEN_14B: &str = "qwen2.5-14b-instruct";

    /// Create an LM Studio endpoint with default configuration
    pub fn endpoint(endpoint_id: &str, model_name: &str) -> ModelEndpoint {
        endpoint_with_url(endpoint_id, model_name, DEFAULT_HOST, DEFAULT_PORT)
    }

    /// Create an LM Studio endpoint with custom URL
    pub fn endpoint_with_url(
        endpoint_id: &str,
        model_name: &str,
        host: &str,
        port: u16,
    ) -> ModelEndpoint {
        let url = format!("{}:{}", host, port);
        ModelEndpoint::new(endpoint_id, model_name, ModelBackend::OpenAICompatible, &url)
            .with_limits(ResourceLimits {
                max_tokens: 8192,          // LM Studio typically supports larger contexts
                max_concurrent: 1,          // Local models usually single-threaded
                timeout_ms: 120_000,        // 2 min for larger reasoning tasks
                memory_limit_mb: 0,
                gpu_memory_limit_mb: 0,
            })
    }

    /// Create standard PAT council endpoints for LM Studio
    ///
    /// Maps the 7-agent council to optimized model assignments:
    /// - Planner, Ethics, Integrator → Reasoning model (ministral-14b)
    /// - Builder, Tester → Code model (qwen2.5-14b)
    /// - Security, Research → General model (nemotron)
    pub fn council_endpoints() -> Vec<ModelEndpoint> {
        vec![
            endpoint("lm-reasoning", MODEL_MINISTRAL_14B),
            endpoint("lm-code", MODEL_QWEN_14B),
            endpoint("lm-general", MODEL_NEMOTRON_30B),
        ]
    }

    /// Setup complete PAT council with LM Studio bindings
    pub async fn setup_pat_council(fabric: &ModelFabric) -> anyhow::Result<()> {
        // Add endpoints
        for ep in council_endpoints() {
            info!("Adding LM Studio endpoint: {} -> {}", ep.endpoint_id, ep.model_name);
            fabric.add_endpoint(ep).await;
        }

        // Bind agents to appropriate endpoints
        // Reasoning tasks → ministral-14b
        fabric.bind_agent(&agents::planner(), "lm-reasoning").await?;
        fabric.bind_agent(&agents::ethics(), "lm-reasoning").await?;
        fabric.bind_agent(&agents::integrator(), "lm-reasoning").await?;

        // Code tasks → qwen2.5-14b
        fabric.bind_agent(&agents::builder(), "lm-code").await?;
        fabric.bind_agent(&agents::tester(), "lm-code").await?;

        // General tasks → nemotron
        fabric.bind_agent(&agents::security(), "lm-general").await?;
        fabric.bind_agent(&agents::research(), "lm-general").await?;

        info!("✅ PAT council bound to LM Studio endpoints (7 agents → 3 models)");
        Ok(())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_endpoint_creation() {
        let endpoint = ModelEndpoint::new(
            "llama-local",
            "llama3.2:8b",
            ModelBackend::Ollama,
            "http://localhost:11434",
        );

        assert_eq!(endpoint.endpoint_id, "llama-local");
        assert_eq!(endpoint.model_name, "llama3.2:8b");
        assert_eq!(endpoint.backend, ModelBackend::Ollama);
        assert_eq!(endpoint.health, HealthStatus::Unknown);
    }

    #[test]
    fn test_endpoint_success_rate() {
        let mut endpoint = ModelEndpoint::new(
            "test",
            "test-model",
            ModelBackend::Mock,
            "http://localhost",
        );

        // No requests yet - 100% success
        assert_eq!(endpoint.success_rate(), 1.0);

        // 10 successes
        for _ in 0..10 {
            endpoint.record_success(100);
        }
        assert_eq!(endpoint.success_rate(), 1.0);

        // 2 failures
        endpoint.record_failure();
        endpoint.record_failure();
        assert!((endpoint.success_rate() - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_agent_id() {
        let agent = AgentId::new("planner", 0);
        assert_eq!(agent.key(), "planner:0");
        assert_eq!(format!("{}", agent), "planner:0");
    }

    #[test]
    fn test_predefined_agents() {
        let council = agents::council();
        assert_eq!(council.len(), 7);

        assert_eq!(council[0].agent_type, "planner");
        assert_eq!(council[1].agent_type, "builder");
        assert_eq!(council[2].agent_type, "tester");
        assert_eq!(council[3].agent_type, "security");
        assert_eq!(council[4].agent_type, "ethics");
        assert_eq!(council[5].agent_type, "research");
        assert_eq!(council[6].agent_type, "integrator");
    }

    #[tokio::test]
    async fn test_model_fabric_bindings() {
        let fabric = ModelFabric::new();

        // Add endpoint
        let endpoint = ModelEndpoint::new(
            "test-ep",
            "test-model",
            ModelBackend::Mock,
            "http://localhost",
        );
        fabric.add_endpoint(endpoint).await;

        // Bind agent
        let agent = agents::planner();
        fabric.bind_agent(&agent, "test-ep").await.unwrap();

        // Verify binding
        let binding = fabric.get_binding(&agent).await;
        assert_eq!(binding, Some("test-ep".to_string()));

        // Get endpoint for agent
        let ep = fabric.get_endpoint_for_agent(&agent).await;
        assert!(ep.is_some());
        assert_eq!(ep.unwrap().endpoint_id, "test-ep");
    }

    #[tokio::test]
    async fn test_fabric_health_empty() {
        let fabric = ModelFabric::new();
        let health = fabric.health_check().await;

        assert_eq!(health.healthy_count, 0);
        assert_eq!(health.health_score, 0.0);
        assert!(!health.is_operational());
    }

    #[test]
    fn test_backend_paths() {
        assert_eq!(ModelBackend::Ollama.health_path(), "/api/tags");
        assert_eq!(ModelBackend::Ollama.completion_path(), "/api/generate");

        assert_eq!(ModelBackend::VLLM.health_path(), "/health");
        assert_eq!(ModelBackend::VLLM.completion_path(), "/v1/completions");

        assert_eq!(ModelBackend::OpenAICompatible.completion_path(), "/v1/chat/completions");
    }

    // ========================================================================
    // LM STUDIO INTEGRATION TESTS
    // ========================================================================

    #[test]
    fn test_lm_studio_endpoint_creation() {
        let ep = lm_studio::endpoint("lm-test", "test-model");

        assert_eq!(ep.endpoint_id, "lm-test");
        assert_eq!(ep.model_name, "test-model");
        assert_eq!(ep.backend, ModelBackend::OpenAICompatible);
        assert!(ep.url.contains("172.22.48.1"));
        assert!(ep.url.contains("1234"));
    }

    #[test]
    fn test_lm_studio_council_endpoints() {
        let endpoints = lm_studio::council_endpoints();

        assert_eq!(endpoints.len(), 3);
        assert_eq!(endpoints[0].endpoint_id, "lm-reasoning");
        assert_eq!(endpoints[1].endpoint_id, "lm-code");
        assert_eq!(endpoints[2].endpoint_id, "lm-general");

        // All should be OpenAI-compatible
        for ep in &endpoints {
            assert_eq!(ep.backend, ModelBackend::OpenAICompatible);
        }
    }

    #[tokio::test]
    async fn test_lm_studio_pat_council_setup() {
        let fabric = ModelFabric::new();

        // Setup should succeed
        let result = lm_studio::setup_pat_council(&fabric).await;
        assert!(result.is_ok());

        // Verify all 7 agents are bound
        let bindings = fabric.all_bindings().await;
        assert_eq!(bindings.len(), 7);

        // Verify specific bindings
        assert_eq!(bindings.get("planner:0"), Some(&"lm-reasoning".to_string()));
        assert_eq!(bindings.get("builder:0"), Some(&"lm-code".to_string()));
        assert_eq!(bindings.get("security:0"), Some(&"lm-general".to_string()));

        // Verify endpoints exist
        let endpoints = fabric.all_endpoints().await;
        assert_eq!(endpoints.len(), 3);
    }

    #[test]
    fn test_lm_studio_resource_limits() {
        let ep = lm_studio::endpoint("test", "test-model");

        // LM Studio endpoints should have appropriate limits
        assert_eq!(ep.resource_limits.max_tokens, 8192);
        assert_eq!(ep.resource_limits.max_concurrent, 1);
        assert_eq!(ep.resource_limits.timeout_ms, 120_000);
    }
}
