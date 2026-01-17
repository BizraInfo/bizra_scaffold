// src/unified_memory.rs - Unified Memory Orchestration Layer
//
// PEAK MASTERPIECE v7.1: 4-Framework Integration
//
// Giants Protocol Synthesis:
// - Tulving (1972): Encoding Specificity - multi-level indexing
// - Ibn Khaldun (1377): Asabiyyah lifecycle - network centrality decay
// - Al-Ghazali (1095): Knowledge-Action Unity - preconditions→operations→postconditions
// - Mnih et al. (DQN): Prioritized Experience Replay - HRPO optimization
//
// This module unifies:
// 1. MemGovern (SovereignExperienceMemory) - 6-field experience cards
// 2. Engram (SovereignEngram) - O(1) n-gram static memory
// 3. HouseOfWisdom - Neo4j + ChromaDB hybrid retrieval
// 4. Evolution (SovereignEvolution) - DrZero HRPO feedback

use crate::engram::{SovereignEngram, SovereigntyTier};
use crate::evolution::{EvolutionState, SovereignEvolution};
use crate::experience_memory::{
    ExperienceCard, SovereignExperienceMemory, TransferContext, TransformationFunction,
};
use crate::fixed::Fixed64;
use crate::wisdom::HouseOfWisdom;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

// ============================================================================
// UNIFIED MEMORY CONFIGURATION
// ============================================================================

/// Configuration for unified memory orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedMemoryConfig {
    /// Engram tier for static memory
    pub engram_tier: SovereigntyTier,
    /// Evolution seed for reproducibility
    pub evolution_seed: u64,
    /// Whether to auto-index experiences to Engram
    pub auto_index_to_engram: bool,
    /// Whether to auto-index experiences to House of Wisdom
    pub auto_index_to_wisdom: bool,
    /// N-gram extraction window for indexing
    pub ngram_window: usize,
}

impl Default for UnifiedMemoryConfig {
    fn default() -> Self {
        Self {
            engram_tier: SovereigntyTier::T1Consumer,
            evolution_seed: 42,
            auto_index_to_engram: true,
            auto_index_to_wisdom: true,
            ngram_window: 3,
        }
    }
}

// ============================================================================
// UNIFIED SEARCH RESULT
// ============================================================================

/// Combined search result from all memory systems
#[derive(Debug, Clone)]
pub struct UnifiedSearchResult {
    /// Number of graph nodes found
    pub graph_nodes_count: usize,
    /// Number of vector results found
    pub vector_results_count: usize,
    /// Direct Engram n-gram hits
    pub engram_hits_count: usize,
    /// Total query time across all systems
    pub total_query_ms: u64,
    /// Fusion confidence score
    pub fusion_confidence: Fixed64,
}

// ============================================================================
// UNIFIED MEMORY ORCHESTRATOR
// ============================================================================

/// Unified Memory Orchestrator
///
/// Connects all 4 research frameworks into a single coherent memory system:
/// 1. **MemGovern** - Experience cards with 6-field structure
/// 2. **Engram** - O(1) n-gram static memory lookup
/// 3. **HouseOfWisdom** - Neo4j graph + ChromaDB vectors
/// 4. **Evolution** - DrZero HRPO self-improvement
pub struct UnifiedMemory {
    /// MemGovern: Experience memory (6-field cards)
    experience: Arc<SovereignExperienceMemory>,
    /// Engram: O(1) n-gram static memory
    engram: Arc<RwLock<SovereignEngram>>,
    /// HouseOfWisdom: Neo4j + ChromaDB (already has Engram integrated)
    wisdom: Arc<HouseOfWisdom>,
    /// Evolution: DrZero HRPO feedback loop
    evolution: Arc<RwLock<SovereignEvolution>>,
    /// Configuration
    config: UnifiedMemoryConfig,
}

impl UnifiedMemory {
    /// Create new unified memory system
    #[instrument(skip(config))]
    pub async fn new(config: UnifiedMemoryConfig) -> Self {
        info!("🧠 Initializing UnifiedMemory (4-Framework Integration)");

        // Initialize House of Wisdom first (shared by Experience and Unified)
        let wisdom = Arc::new(HouseOfWisdom::from_env_with_vectors().await);

        // Initialize experience memory with shared wisdom
        let experience = Arc::new(SovereignExperienceMemory::new((*wisdom).clone()));

        // Initialize Engram
        let engram = Arc::new(RwLock::new(SovereignEngram::new(config.engram_tier)));

        // Initialize Evolution
        let evolution = Arc::new(RwLock::new(SovereignEvolution::new(
            config.engram_tier,
            config.evolution_seed,
        )));

        info!(
            tier = ?config.engram_tier,
            auto_index_engram = config.auto_index_to_engram,
            auto_index_wisdom = config.auto_index_to_wisdom,
            "UnifiedMemory initialized with 4 frameworks"
        );

        Self {
            experience,
            engram,
            wisdom,
            evolution,
            config,
        }
    }

    /// Store experience and auto-index to other memory systems
    ///
    /// # MemGovern → Engram → HouseOfWisdom Pipeline
    ///
    /// 1. Store experience card in MemGovern
    /// 2. Extract n-grams and populate Engram
    /// 3. Index in House of Wisdom for graph/vector search
    #[instrument(skip(self, card, transfer_ctx, transformation))]
    pub async fn store_and_index(
        &self,
        card: ExperienceCard,
        source_repo: &str,
        source_issue: &str,
        transfer_ctx: TransferContext,
        transformation: TransformationFunction,
    ) -> anyhow::Result<String> {
        info!(
            problem_summary = %card.problem_summary[..card.problem_summary.len().min(50)],
            "Storing experience with unified indexing"
        );

        // Step 1: Store in MemGovern
        let receipt_id = self
            .experience
            .store_experience(card.clone(), source_repo, source_issue, transfer_ctx, transformation)
            .await?;

        // Step 2: Extract n-grams and populate Engram
        if self.config.auto_index_to_engram {
            let texts = self.extract_indexable_text(&card);
            let mut engram = self.engram.write().await;
            let ngrams_added = engram.ingest_ngrams(texts);
            debug!(ngrams_added = ngrams_added, "Indexed to Engram");
        }

        // Step 3: Index in House of Wisdom (already done by store_experience)
        if self.config.auto_index_to_wisdom {
            debug!("Experience indexed to House of Wisdom via MemGovern");
        }

        info!(
            receipt_id = %receipt_id,
            engram_indexed = self.config.auto_index_to_engram,
            wisdom_indexed = self.config.auto_index_to_wisdom,
            "Experience stored with unified indexing"
        );

        Ok(receipt_id)
    }

    /// Unified search across all memory systems
    ///
    /// Searches in parallel:
    /// 1. House of Wisdom (Neo4j + ChromaDB + Engram)
    /// 2. Direct Engram n-gram lookup
    #[instrument(skip(self))]
    pub async fn unified_search(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<UnifiedSearchResult> {
        let start = std::time::Instant::now();

        // Parallel search across all systems
        let (wisdom_result, engram_direct) = tokio::join!(
            self.wisdom.hybrid_search(query, limit),
            self.direct_engram_search(query, limit)
        );

        let wisdom = wisdom_result?;
        let engram_direct_hits = engram_direct;

        // Calculate fusion confidence
        let total_results = wisdom.graph_nodes.len()
            + wisdom.vector_results.len()
            + wisdom.engram_hits.len()
            + engram_direct_hits;

        let fusion_confidence = if total_results > 0 {
            // Weighted fusion based on source quality
            let graph_weight = 0.4 * wisdom.graph_nodes.len() as f64;
            let vector_weight = 0.25 * wisdom.vector_results.len() as f64;
            let engram_weight = 0.35 * (wisdom.engram_hits.len() + engram_direct_hits) as f64;

            let weighted_sum = graph_weight + vector_weight + engram_weight;
            let normalized = (weighted_sum / (total_results as f64)).min(1.0);
            Fixed64::from_f64(normalized)
        } else {
            Fixed64::ZERO
        };

        let total_query_ms = start.elapsed().as_millis() as u64;

        info!(
            query = %query,
            graph_results = wisdom.graph_nodes.len(),
            vector_results = wisdom.vector_results.len(),
            engram_hits = wisdom.engram_hits.len() + engram_direct_hits,
            fusion_confidence = fusion_confidence.to_f64(),
            query_ms = total_query_ms,
            "Unified search completed"
        );

        Ok(UnifiedSearchResult {
            graph_nodes_count: wisdom.graph_nodes.len(),
            vector_results_count: wisdom.vector_results.len(),
            engram_hits_count: wisdom.engram_hits.len() + engram_direct_hits,
            total_query_ms,
            fusion_confidence,
        })
    }

    /// Run evolution cycle and return state
    #[instrument(skip(self))]
    pub async fn evolve(&self) -> EvolutionState {
        let mut evolution = self.evolution.write().await;
        evolution.evolve_cycle()
    }

    /// Run full evolution session
    pub async fn evolve_session(&self) -> Vec<EvolutionState> {
        let mut evolution = self.evolution.write().await;
        evolution.evolve_session()
    }

    /// Get current evolution state
    pub async fn evolution_state(&self) -> EvolutionState {
        let evolution = self.evolution.read().await;
        evolution.state().clone()
    }

    /// Get solver performance from evolution
    pub async fn solver_performance(&self) -> Fixed64 {
        let evolution = self.evolution.read().await;
        evolution.solver_performance()
    }

    /// Extract indexable text from experience card
    fn extract_indexable_text(&self, card: &ExperienceCard) -> Vec<String> {
        vec![
            card.problem_summary.clone(),
            card.fix_strategy.clone(),
            card.root_cause.clone(),
            card.signals.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" "),
        ]
    }

    /// Direct Engram search bypassing House of Wisdom
    async fn direct_engram_search(&self, query: &str, limit: usize) -> usize {
        let mut engram = self.engram.write().await;
        engram.query_ngrams(query, limit).len()
    }

    // ========================================================================
    // Accessor Methods
    // ========================================================================

    /// Get reference to House of Wisdom
    pub fn wisdom(&self) -> &HouseOfWisdom {
        &self.wisdom
    }

    /// Get reference to Experience Memory
    pub fn experience(&self) -> &SovereignExperienceMemory {
        &self.experience
    }

    /// Get engram lock
    pub async fn engram(&self) -> tokio::sync::RwLockWriteGuard<'_, SovereignEngram> {
        self.engram.write().await
    }

    /// Get configuration
    pub fn config(&self) -> &UnifiedMemoryConfig {
        &self.config
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_unified_memory_creation() {
        let config = UnifiedMemoryConfig::default();
        let unified = UnifiedMemory::new(config).await;

        assert!(unified.config().auto_index_to_engram);
        assert_eq!(unified.config().engram_tier, SovereigntyTier::T1Consumer);
    }

    #[test]
    fn test_config_defaults() {
        let config = UnifiedMemoryConfig::default();
        assert_eq!(config.engram_tier, SovereigntyTier::T1Consumer);
        assert_eq!(config.evolution_seed, 42);
        assert!(config.auto_index_to_engram);
        assert!(config.auto_index_to_wisdom);
    }

    #[tokio::test]
    async fn test_evolution_integration() {
        let config = UnifiedMemoryConfig::default();
        let unified = UnifiedMemory::new(config).await;

        let state = unified.evolve().await;
        assert!(state.generation > 0);
    }
}
