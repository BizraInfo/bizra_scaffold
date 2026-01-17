// src/a2a_external.rs - External AI Integration Layer
//
// Extends the A2A (Agent-to-Agent) protocol to support external AI providers:
// - OpenAI Codex for code generation
// - Google Gemini for data mining and synthesis
//
// Philosophy: "We don't assume. If we must, we do it with Ihsān."
// All external AI calls are validated, metered, and auditable.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, info, warn};

use crate::a2a::{Agent, AgentCard, AgentResponse, Capability, Task};
use crate::metrics;

/// Maximum timeout for external AI API calls (30 seconds default)
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Maximum retries for failed API calls
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Exponential backoff base delay (milliseconds)
const BACKOFF_BASE_MS: u64 = 1000;

// ═══════════════════════════════════════════════════════════════════════════
// AI PROVIDER ENUM
// ═══════════════════════════════════════════════════════════════════════════

/// External AI provider configuration
#[derive(Debug, Clone)]
pub enum AIProvider {
    /// OpenAI API (GPT-4, Codex, etc.)
    OpenAI {
        api_key: String,
        model: String,
        organization: Option<String>,
    },
    /// Google Gemini API
    GoogleGemini { api_key: String, model: String },
}

impl AIProvider {
    /// Get provider name for metrics/logging
    pub fn name(&self) -> &'static str {
        match self {
            AIProvider::OpenAI { .. } => "openai",
            AIProvider::GoogleGemini { .. } => "gemini",
        }
    }

    /// Get model name
    pub fn model(&self) -> &str {
        match self {
            AIProvider::OpenAI { model, .. } => model,
            AIProvider::GoogleGemini { model, .. } => model,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EXTERNAL AI ADAPTER
// ═══════════════════════════════════════════════════════════════════════════

/// Adapter for external AI providers that implements the Agent trait
pub struct ExternalAIAdapter {
    agent_id: String,
    provider: AIProvider,
    client: HttpClient,
    capabilities: Vec<Capability>,
    timeout: Duration,
    max_retries: u32,
}

impl ExternalAIAdapter {
    /// Create a new External AI adapter
    pub fn new(
        agent_id: String,
        provider: AIProvider,
        capabilities: Vec<Capability>,
    ) -> Result<Self> {
        let timeout_secs = std::env::var("EXTERNAL_AI_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        let max_retries = std::env::var("EXTERNAL_AI_MAX_RETRIES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_RETRIES);

        let client = HttpClient::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            agent_id,
            provider,
            client,
            capabilities,
            timeout: Duration::from_secs(timeout_secs),
            max_retries,
        })
    }

    /// Create OpenAI adapter from environment variables
    pub fn from_env_openai() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY environment variable not set")?;
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4".to_string());
        let organization = std::env::var("OPENAI_ORG").ok();

        let provider = AIProvider::OpenAI {
            api_key,
            model,
            organization,
        };

        let capabilities = vec![
            Capability::CodeGeneration,
            Capability::DataPipeline,
            Capability::Analysis,
            Capability::Synthesis,
        ];

        Self::new("codex".to_string(), provider, capabilities)
    }

    /// Create Google Gemini adapter from environment variables
    pub fn from_env_gemini() -> Result<Self> {
        let api_key = std::env::var("GOOGLE_API_KEY")
            .context("GOOGLE_API_KEY environment variable not set")?;
        let model = std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-1.5-pro".to_string());

        let provider = AIProvider::GoogleGemini { api_key, model };

        let capabilities = vec![
            Capability::DataMining,
            Capability::Search,
            Capability::Synthesis,
            Capability::Analysis,
        ];

        Self::new("gemini".to_string(), provider, capabilities)
    }

    /// Execute API call with retry logic
    async fn execute_with_retry(&self, task: &Task) -> Result<String> {
        let mut attempt = 0;
        let mut last_error = None;

        while attempt < self.max_retries {
            match self.execute_api_call(task).await {
                Ok(response) => {
                    metrics::EXTERNAL_AI_CALLS
                        .with_label_values(&[
                            self.provider.name(),
                            self.provider.model(),
                            "success",
                        ])
                        .inc();
                    return Ok(response);
                }
                Err(e) => {
                    attempt += 1;
                    last_error = Some(e);

                    if attempt < self.max_retries {
                        let backoff_ms = BACKOFF_BASE_MS * 2_u64.pow(attempt - 1);
                        warn!(
                            "External AI call failed (attempt {}/{}), retrying in {}ms: {}",
                            attempt,
                            self.max_retries,
                            backoff_ms,
                            last_error.as_ref().unwrap()
                        );
                        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    }
                }
            }
        }

        metrics::EXTERNAL_AI_CALLS
            .with_label_values(&[self.provider.name(), self.provider.model(), "error"])
            .inc();

        Err(last_error.unwrap_or_else(|| anyhow!("Unknown error")))
    }

    /// Execute actual API call (provider-specific)
    async fn execute_api_call(&self, task: &Task) -> Result<String> {
        let start = std::time::Instant::now();

        let result = match &self.provider {
            AIProvider::OpenAI {
                api_key,
                model,
                organization,
            } => {
                self.call_openai_api(task, api_key, model, organization.as_deref())
                    .await
            }
            AIProvider::GoogleGemini { api_key, model } => {
                self.call_gemini_api(task, api_key, model).await
            }
        };

        let elapsed = start.elapsed().as_secs_f64();
        metrics::EXTERNAL_AI_LATENCY
            .with_label_values(&[self.provider.name(), self.provider.model()])
            .observe(elapsed);

        result
    }

    /// Call OpenAI API (Chat Completions)
    async fn call_openai_api(
        &self,
        task: &Task,
        api_key: &str,
        model: &str,
        organization: Option<&str>,
    ) -> Result<String> {
        #[derive(Serialize)]
        struct OpenAIRequest {
            model: String,
            messages: Vec<OpenAIMessage>,
            temperature: f64,
            max_tokens: Option<u32>,
        }

        #[derive(Serialize, Deserialize)]
        struct OpenAIMessage {
            role: String,
            content: String,
        }

        #[derive(Deserialize)]
        struct OpenAIResponse {
            choices: Vec<OpenAIChoice>,
            usage: Option<OpenAIUsage>,
        }

        #[derive(Deserialize)]
        struct OpenAIChoice {
            message: OpenAIMessage,
        }

        #[derive(Deserialize)]
        struct OpenAIUsage {
            prompt_tokens: u32,
            completion_tokens: u32,
            total_tokens: u32,
        }

        let request = OpenAIRequest {
            model: model.to_string(),
            messages: vec![
                OpenAIMessage {
                    role: "system".to_string(),
                    content: format!(
                        "You are {}, a specialized AI agent in the BIZRA system. Your capabilities: {:?}",
                        self.agent_id, self.capabilities
                    ),
                },
                OpenAIMessage {
                    role: "user".to_string(),
                    content: task.description.clone(),
                },
            ],
            temperature: 0.7,
            max_tokens: Some(2048),
        };

        let mut req_builder = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json");

        if let Some(org) = organization {
            req_builder = req_builder.header("OpenAI-Organization", org);
        }

        let response = timeout(self.timeout, req_builder.json(&request).send())
            .await
            .context("OpenAI API request timed out")?
            .context("Failed to send request to OpenAI")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("OpenAI API error ({}): {}", status, error_text));
        }

        let api_response: OpenAIResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI response")?;

        if let Some(usage) = api_response.usage {
            debug!(
                "OpenAI tokens - prompt: {}, completion: {}, total: {}",
                usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
            );
            metrics::EXTERNAL_AI_TOKENS
                .with_label_values(&["openai", model, "prompt"])
                .inc_by(usage.prompt_tokens as f64);
            metrics::EXTERNAL_AI_TOKENS
                .with_label_values(&["openai", model, "completion"])
                .inc_by(usage.completion_tokens as f64);
        }

        api_response
            .choices
            .first()
            .map(|choice| choice.message.content.clone())
            .ok_or_else(|| anyhow!("No response from OpenAI"))
    }

    /// Call Google Gemini API
    async fn call_gemini_api(&self, task: &Task, api_key: &str, model: &str) -> Result<String> {
        #[derive(Serialize)]
        struct GeminiRequest {
            contents: Vec<GeminiContent>,
        }

        #[derive(Serialize, Deserialize)]
        struct GeminiContent {
            parts: Vec<GeminiPart>,
        }

        #[derive(Serialize, Deserialize)]
        struct GeminiPart {
            text: String,
        }

        #[derive(Deserialize)]
        struct GeminiResponse {
            candidates: Vec<GeminiCandidate>,
        }

        #[derive(Deserialize)]
        struct GeminiCandidate {
            content: GeminiContent,
        }

        let request = GeminiRequest {
            contents: vec![GeminiContent {
                parts: vec![GeminiPart {
                    text: format!(
                        "You are {}, a specialized AI agent in the BIZRA system.\nCapabilities: {:?}\n\nTask: {}",
                        self.agent_id, self.capabilities, task.description
                    ),
                }],
            }],
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, api_key
        );

        let response = timeout(
            self.timeout,
            self.client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&request)
                .send(),
        )
        .await
        .context("Gemini API request timed out")?
        .context("Failed to send request to Gemini")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Gemini API error ({}): {}", status, error_text));
        }

        let api_response: GeminiResponse = response
            .json()
            .await
            .context("Failed to parse Gemini response")?;

        api_response
            .candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .map(|p| p.text.clone())
            .ok_or_else(|| anyhow!("No response from Gemini"))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AGENT TRAIT IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════

#[async_trait]
impl Agent for ExternalAIAdapter {
    async fn execute(&self, task: Task) -> Result<AgentResponse> {
        info!(
            "External AI {} executing task: {}",
            self.agent_id, task.task_id
        );

        let response_text = self.execute_with_retry(&task).await?;

        Ok(AgentResponse {
            agent_id: self.agent_id.clone(),
            task_id: task.task_id,
            content: response_text,
            confidence: 0.85, // External AIs get fixed confidence
            metadata: serde_json::json!({
                "provider": self.provider.name(),
                "model": self.provider.model(),
                "external": true,
            }),
        })
    }

    fn capabilities(&self) -> Vec<Capability> {
        self.capabilities.clone()
    }

    fn agent_card(&self) -> AgentCard {
        AgentCard {
            name: self.agent_id.clone(),
            version: "1.0.0".to_string(),
            capabilities: self.capabilities(),
            protocols: vec!["http".to_string(), "rest".to_string()],
            authentication: vec!["bearer".to_string()],
            external: true,
            provider: Some(self.provider.name().to_string()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// METRICS
// ═══════════════════════════════════════════════════════════════════════════

/// Initialize Prometheus metrics for external AI monitoring
pub fn init_metrics() {
    // Metrics are defined in src/metrics.rs and initialized there
    info!("External AI metrics initialized");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let openai = AIProvider::OpenAI {
            api_key: "test".to_string(),
            model: "gpt-4".to_string(),
            organization: None,
        };
        assert_eq!(openai.name(), "openai");
        assert_eq!(openai.model(), "gpt-4");

        let gemini = AIProvider::GoogleGemini {
            api_key: "test".to_string(),
            model: "gemini-1.5-pro".to_string(),
        };
        assert_eq!(gemini.name(), "gemini");
        assert_eq!(gemini.model(), "gemini-1.5-pro");
    }

    #[test]
    fn test_timeout_from_env() {
        std::env::set_var("EXTERNAL_AI_TIMEOUT", "60");
        std::env::set_var("OPENAI_API_KEY", "test");

        let adapter = ExternalAIAdapter::from_env_openai().unwrap();
        assert_eq!(adapter.timeout, Duration::from_secs(60));

        std::env::remove_var("EXTERNAL_AI_TIMEOUT");
        std::env::remove_var("OPENAI_API_KEY");
    }
}
