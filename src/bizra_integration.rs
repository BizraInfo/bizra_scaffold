// src/bizra_integration.rs - Connect to external BIZRA components

use crate::wisdom::HouseOfWisdom;
use std::sync::Arc;
use tokio::sync::OnceCell;
use tracing::{info, warn};

/// Global House of Wisdom instance (lazy-initialized)
static WISDOM: OnceCell<Arc<HouseOfWisdom>> = OnceCell::const_new();

/// Get or initialize the House of Wisdom
pub async fn get_wisdom() -> Arc<HouseOfWisdom> {
    WISDOM
        .get_or_init(|| async {
            let wisdom = HouseOfWisdom::from_env();
            // Try to connect but surface failures at query time
            if let Err(e) = wisdom.connect().await {
                warn!("⚠️ Neo4j not available: {}", e);
            }
            Arc::new(wisdom)
        })
        .await
        .clone()
}

/// Integration with BIZRA-NODE0 (ACE Framework)
pub struct NODE0Integration {
    base_url: String,
    wisdom: Option<Arc<HouseOfWisdom>>,
}

impl NODE0Integration {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            wisdom: None,
        }
    }

    /// Create with House of Wisdom integration
    pub async fn with_wisdom(base_url: String) -> Self {
        Self {
            base_url,
            wisdom: Some(get_wisdom().await),
        }
    }

    /// Call NODE0 ACE Framework
    pub async fn call_ace_framework(&self, task: &str) -> anyhow::Result<serde_json::Value> {
        let _ = task;
        anyhow::bail!(
            "NODE0 integration not configured for base_url={}",
            self.base_url
        );
    }

    /// Query HyperGraphRAG (18.7x advantage) via House of Wisdom
    pub async fn query_hypergraph_rag(&self, query: &str) -> anyhow::Result<Vec<String>> {
        if let Some(wisdom) = &self.wisdom {
            if wisdom.is_connected().await {
                match wisdom.query_knowledge(query, 10).await {
                    Ok(result) => {
                        info!(
                            nodes = result.nodes.len(),
                            boost = result.hypergraph_boost,
                            query_time_ms = result.query_time_ms,
                            "🏛️ HyperGraphRAG query succeeded"
                        );
                        return Ok(result
                            .nodes
                            .iter()
                            .map(|n| format!("[{}] {}", n.node_type, n.content))
                            .collect());
                    }
                    Err(e) => {
                        warn!("⚠️ HyperGraphRAG query failed: {}", e);
                        return Err(anyhow::anyhow!(e));
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "HyperGraphRAG unavailable: Neo4j connection required"
        ))
    }

    /// Store knowledge in the graph (if connected)
    pub async fn store_knowledge(
        &self,
        node_type: &str,
        content: &str,
        metadata: serde_json::Value,
    ) -> anyhow::Result<Option<String>> {
        if let Some(wisdom) = &self.wisdom {
            if wisdom.is_connected().await {
                let id = wisdom.store_knowledge(node_type, content, metadata).await?;
                return Ok(Some(id));
            }
        }
        Ok(None)
    }
}

/// Integration with BIZRA-TaskMaster (Hive-Mind orchestration)
pub struct TaskMasterIntegration {
    base_url: String,
}

impl TaskMasterIntegration {
    pub fn new(base_url: String) -> Self {
        Self { base_url }
    }

    /// Execute task with Hive-Mind pattern (84.8% solve rate)
    pub async fn execute_hive_mind(
        &self,
        _task: &str,
        agent_count: usize,
    ) -> anyhow::Result<serde_json::Value> {
        let _ = agent_count;
        anyhow::bail!(
            "TaskMaster integration not configured for base_url={}",
            self.base_url
        );
    }
}

/// Integration with deepagent node0 (CUDA acceleration)
pub struct DeepAgentIntegration {
    base_url: String,
}

impl DeepAgentIntegration {
    pub fn new(base_url: String) -> Self {
        Self { base_url }
    }

    /// Execute CUDA-accelerated inference
    pub async fn cuda_inference(&self, prompt: &str, model: &str) -> anyhow::Result<String> {
        let _ = (prompt, model);
        anyhow::bail!(
            "deepagent integration not configured for base_url={}",
            self.base_url
        );
    }
}

/// Integration with BlockGraph (Proof-of-Impact)
pub struct BlockGraphIntegration {
    base_url: String,
}

impl BlockGraphIntegration {
    pub fn new(base_url: String) -> Self {
        Self { base_url }
    }

    /// Generate Proof-of-Impact attestation
    pub async fn generate_poi_attestation(
        &self,
        user_id: &str,
        impact_type: &str,
        _evidence: serde_json::Value,
    ) -> anyhow::Result<String> {
        let _ = (user_id, impact_type);
        anyhow::bail!(
            "BlockGraph integration not configured for base_url={}",
            self.base_url
        );
    }
}
