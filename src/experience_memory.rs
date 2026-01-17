// src/experience_memory.rs - MemGovern × BIZRA Integration
//
// Peak Masterpiece: Sovereign Experience Memory System
// Integrates MemGovern patterns with House of Wisdom HyperGraphRAG
//
// Key innovations:
// - Dual-store architecture (ChromaDB episodic + Neo4j semantic)
// - Stage-gated retrieval with SAT validation
// - Lifecycle-aware decay (Ibn Khaldun asabiyyah model)
// - Transformation function encoding (Al-Ghazali knowledge-action unity)

use crate::fixed::Fixed64;
use crate::wisdom::{HouseOfWisdom, KnowledgeNode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, instrument, warn};

/// Experience lifecycle states (Ibn Khaldun model)
/// Connected experiences survive longer than isolated ones (asabiyyah)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExperienceLifecycle {
    /// < 7 days, single instance, unvalidated
    Nascent,
    /// 7-30 days, multiple instances, gaining confidence
    Emerging,
    /// 30-180 days, high retrieval success rate
    Mature,
    /// > 180 days, stable pattern, proven durability
    Classic,
    /// Retrieval rate dropping, dependencies outdating
    Declining,
    /// Superseded by newer patterns or deprecated APIs
    Obsolete,
}

/// Experience Card - MemGovern compatible format (6 fields)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceCard {
    /// Concise technical bug description (generalizable, no repo names)
    pub problem_summary: String,
    /// 10-18 high-signal keywords (error types, symptoms, components)
    pub signals: Vec<String>,
    /// Evidence-backed causal explanation
    pub root_cause: String,
    /// Design-level fix approach (no line-by-line)
    pub fix_strategy: String,
    /// Semantic summary of diff chunks
    pub patch_digest: String,
    /// Test plan and validation steps
    pub verification: String,
}

impl ExperienceCard {
    /// Create a new experience card
    pub fn new(
        problem_summary: String,
        signals: Vec<String>,
        root_cause: String,
        fix_strategy: String,
        patch_digest: String,
        verification: String,
    ) -> Self {
        Self {
            problem_summary,
            signals,
            root_cause,
            fix_strategy,
            patch_digest,
            verification,
        }
    }

    /// Check if the card is complete (all 6 fields populated)
    pub fn is_complete(&self) -> bool {
        !self.problem_summary.is_empty()
            && !self.signals.is_empty()
            && !self.root_cause.is_empty()
            && !self.fix_strategy.is_empty()
            && !self.patch_digest.is_empty()
            && !self.verification.is_empty()
    }

    /// Calculate signal quality score (10-18 keywords is optimal)
    pub fn signal_quality(&self) -> f64 {
        let count = self.signals.len();
        if (10..=18).contains(&count) {
            1.0
        } else if (5..10).contains(&count) {
            0.8
        } else if count > 18 && count <= 25 {
            0.85
        } else {
            0.5
        }
    }
}

/// Transfer compatibility context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TransferContext {
    pub language: String,
    pub framework: Option<String>,
    pub dependencies: HashMap<String, String>,
    pub architecture: Option<String>,
    pub test_framework: Option<String>,
}

impl TransferContext {
    /// Create context for Rust projects
    pub fn rust() -> Self {
        Self {
            language: "rust".to_string(),
            framework: None,
            dependencies: HashMap::new(),
            architecture: None,
            test_framework: Some("cargo-test".to_string()),
        }
    }

    /// Create context for Python projects
    pub fn python() -> Self {
        Self {
            language: "python".to_string(),
            framework: None,
            dependencies: HashMap::new(),
            architecture: None,
            test_framework: Some("pytest".to_string()),
        }
    }

    /// Create context for TypeScript projects
    pub fn typescript() -> Self {
        Self {
            language: "typescript".to_string(),
            framework: None,
            dependencies: HashMap::new(),
            architecture: None,
            test_framework: Some("jest".to_string()),
        }
    }
}

/// Transformation function (Al-Ghazali's knowledge-action unity)
/// Encodes not just "what" succeeded but "why" it was the right action
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TransformationFunction {
    /// What must be true for this fix to apply
    pub preconditions: Vec<String>,
    /// Abstract edit operations (not line-specific)
    pub operations: Vec<String>,
    /// What becomes true after applying the fix
    pub postconditions: Vec<String>,
}

impl TransformationFunction {
    /// Create a new transformation function
    pub fn new(
        preconditions: Vec<String>,
        operations: Vec<String>,
        postconditions: Vec<String>,
    ) -> Self {
        Self {
            preconditions,
            operations,
            postconditions,
        }
    }

    /// Check if preconditions are simple (can be validated without Z3)
    pub fn has_simple_preconditions(&self) -> bool {
        self.preconditions
            .iter()
            .all(|p| !p.contains("∀") && !p.contains("∃") && !p.contains("→") && !p.contains("∧"))
    }
}

/// SAT validation result for experiences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceSATResult {
    pub security_sentinel: bool,
    pub formal_validator: bool,
    pub ethics_guardian: bool,
    pub resource_guardian: bool,
    pub context_validator: bool,
    pub edge_sentinel: bool,
    pub consensus_score: f64,
}

impl ExperienceSATResult {
    /// Check if all validators approved
    pub fn all_approved(&self) -> bool {
        self.security_sentinel
            && self.formal_validator
            && self.ethics_guardian
            && self.resource_guardian
            && self.context_validator
            && self.edge_sentinel
    }

    /// Get list of vetoing validators
    pub fn vetoes(&self) -> Vec<&'static str> {
        let mut vetoes = Vec::new();
        if !self.security_sentinel {
            vetoes.push("security_sentinel");
        }
        if !self.formal_validator {
            vetoes.push("formal_validator");
        }
        if !self.ethics_guardian {
            vetoes.push("ethics_guardian");
        }
        if !self.resource_guardian {
            vetoes.push("resource_guardian");
        }
        if !self.context_validator {
            vetoes.push("context_validator");
        }
        if !self.edge_sentinel {
            vetoes.push("edge_sentinel");
        }
        vetoes
    }
}

/// Complete Experience Receipt (Third Fact format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceReceipt {
    /// Unique receipt ID (EXP-YYYYMMDDHHMMSS-XXXXXXXX)
    pub receipt_id: String,
    /// Source repository (owner/repo)
    pub source_repo: String,
    /// Source issue ID
    pub source_issue: String,
    /// The experience card content
    pub experience_card: ExperienceCard,
    /// Current lifecycle state
    pub lifecycle: ExperienceLifecycle,
    /// Graph centrality score (asabiyyah measure)
    pub graph_centrality: f64,
    /// Transfer compatibility context
    pub transfer_context: TransferContext,
    /// Transformation function for generalization
    pub transformation: TransformationFunction,
    /// Ihsan quality score
    pub ihsan_score: Fixed64,
    /// SAT consensus score (0.0 - 1.0)
    pub sat_consensus: f64,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last successful retrieval timestamp
    pub last_retrieval: Option<chrono::DateTime<chrono::Utc>>,
    /// Total retrieval count
    pub retrieval_count: u32,
    /// Success rate of retrievals
    pub success_rate: f64,
}

impl ExperienceReceipt {
    /// Generate a new receipt ID
    pub fn generate_id() -> String {
        format!(
            "EXP-{}-{}",
            chrono::Utc::now().format("%Y%m%d%H%M%S"),
            &uuid::Uuid::new_v4().to_string()[..8]
        )
    }

    /// Calculate the current lifecycle state based on age and activity
    pub fn calculate_lifecycle(&self) -> ExperienceLifecycle {
        let now = chrono::Utc::now();
        let age_days = (now - self.created_at).num_days();

        // Check for obsolescence first
        if self.retrieval_count > 10 && self.success_rate < 0.3 {
            return ExperienceLifecycle::Obsolete;
        }

        // Check for decline
        if let Some(last) = self.last_retrieval {
            let days_since_retrieval = (now - last).num_days();
            if days_since_retrieval > 90 && age_days > 180 {
                return ExperienceLifecycle::Declining;
            }
        }

        // Normal lifecycle progression
        match age_days {
            0..=7 => ExperienceLifecycle::Nascent,
            8..=30 => ExperienceLifecycle::Emerging,
            31..=180 => ExperienceLifecycle::Mature,
            _ => ExperienceLifecycle::Classic,
        }
    }

    /// Check if this experience is still viable for retrieval
    pub fn is_viable(&self) -> bool {
        !matches!(
            self.lifecycle,
            ExperienceLifecycle::Obsolete | ExperienceLifecycle::Declining
        )
    }
}

/// Stage-gated retrieval result
#[derive(Debug)]
pub struct RetrievalResult {
    pub experiences: Vec<RankedExperience>,
    pub query_time_ms: u64,
    pub stages_executed: Vec<String>,
    pub candidates_at_each_stage: Vec<usize>,
}

/// Ranked experience with scoring breakdown
#[derive(Debug)]
pub struct RankedExperience {
    pub experience: ExperienceReceipt,
    pub relevance_score: f64,
    pub transfer_score: f64,
    pub temporal_weight: f64,
    pub final_score: f64,
}

/// Sovereign Experience Memory System
/// Integrates MemGovern patterns with House of Wisdom HyperGraphRAG
pub struct SovereignExperienceMemory {
    wisdom: HouseOfWisdom,
    /// Ihsan quality threshold for experience storage (default 0.85)
    pub ihsan_threshold: f64,
    /// Maximum candidates for initial vector search
    pub max_vector_candidates: usize,
    /// Maximum candidates after graph filtering
    pub max_graph_candidates: usize,
}

impl SovereignExperienceMemory {
    /// Create new instance with House of Wisdom
    pub fn new(wisdom: HouseOfWisdom) -> Self {
        Self {
            wisdom,
            ihsan_threshold: 0.85,
            max_vector_candidates: 200,
            max_graph_candidates: 50,
        }
    }

    /// Create from environment with default thresholds
    pub async fn from_env() -> Self {
        let wisdom = HouseOfWisdom::from_env_with_vectors().await;
        Self::new(wisdom)
    }

    /// Calculate Ihsan score for an experience card
    #[instrument(skip(self))]
    pub fn calculate_experience_ihsan(&self, card: &ExperienceCard, _ctx: &TransferContext) -> f64 {
        let mut score = 0.0;

        // Completeness (all 6 fields populated) - 30%
        let fields = [
            !card.problem_summary.is_empty(),
            !card.signals.is_empty(),
            !card.root_cause.is_empty(),
            !card.fix_strategy.is_empty(),
            !card.patch_digest.is_empty(),
            !card.verification.is_empty(),
        ];
        let completeness = fields.iter().filter(|&&f| f).count() as f64 / 6.0;
        score += completeness * 0.3;

        // Signal quality (10-18 keywords is optimal) - 20%
        score += card.signal_quality() * 0.2;

        // Generalizability (no repo-specific paths) - 30%
        let has_specific = card.problem_summary.contains("/home/")
            || card.problem_summary.contains("C:\\")
            || card.fix_strategy.contains(".git")
            || card.root_cause.contains("my_project");
        let generalizability = if has_specific { 0.5 } else { 1.0 };
        score += generalizability * 0.3;

        // Verification quality - 20%
        let has_test_plan = card.verification.to_lowercase().contains("test")
            || card.verification.to_lowercase().contains("verify")
            || card.verification.to_lowercase().contains("check")
            || card.verification.to_lowercase().contains("assert");
        let verification_quality = if has_test_plan { 1.0 } else { 0.6 };
        score += verification_quality * 0.2;

        score
    }

    /// Calculate transfer compatibility between contexts
    #[instrument(skip(self))]
    pub fn calculate_transfer_compatibility(
        &self,
        source: &TransferContext,
        target: &TransferContext,
    ) -> f64 {
        let mut score = 1.0;

        // Hard requirement: language must match
        if source.language != target.language {
            return 0.0;
        }

        // Framework matching
        match (&source.framework, &target.framework) {
            (Some(s), Some(t)) if s != t => score *= 0.6,
            (Some(_), None) | (None, Some(_)) => score *= 0.8,
            _ => {}
        }

        // Dependency version compatibility
        for (dep, version) in &source.dependencies {
            if let Some(target_version) = target.dependencies.get(dep) {
                let src_major = version.split('.').next().unwrap_or("0");
                let tgt_major = target_version.split('.').next().unwrap_or("0");
                if src_major != tgt_major {
                    score *= 0.7;
                }
            }
        }

        // Architecture compatibility
        match (&source.architecture, &target.architecture) {
            (Some(s), Some(t)) if s != t => score *= 0.5,
            _ => {}
        }

        score
    }

    /// Calculate temporal weight using Ibn Khaldun lifecycle model
    #[instrument(skip(self))]
    pub fn calculate_temporal_weight(&self, exp: &ExperienceReceipt) -> f64 {
        let now = chrono::Utc::now();
        let age_days = (now - exp.created_at).num_days();

        // Base decay (slow for code patterns)
        let base_weight = match age_days {
            0..=30 => 1.0,
            31..=180 => 0.95,
            181..=365 => 0.85,
            366..=730 => 0.70,
            _ => 0.50,
        };

        // Recency boost from successful retrieval
        let recency_boost = if let Some(last) = exp.last_retrieval {
            let days_since = (now - last).num_days();
            match days_since {
                0..=7 => 0.2,
                8..=30 => 0.1,
                _ => 0.0,
            }
        } else {
            0.0
        };

        // Success rate multiplier
        let success_mult = 0.5 + (exp.success_rate * 0.5);

        // Network centrality bonus (asabiyyah - connected survive longer)
        let centrality_bonus = exp.graph_centrality * 0.15;

        (base_weight + recency_boost + centrality_bonus) * success_mult
    }

    /// Validate experience card with simulated SAT consensus
    #[instrument(skip(self))]
    pub async fn validate_experience(&self, card: &ExperienceCard) -> ExperienceSATResult {
        // Security Sentinel: Check for credential leakage
        let security_sentinel = !card.patch_digest.contains("api_key")
            && !card.patch_digest.contains("password")
            && !card.patch_digest.contains("secret");

        // Formal Validator: Check structural integrity
        let formal_validator = card.is_complete() && card.signal_quality() >= 0.5;

        // Ethics Guardian: Check for harmful patterns
        let ethics_guardian = !card.fix_strategy.contains("bypass")
            && !card.fix_strategy.contains("disable_security")
            && !card.fix_strategy.contains("remove_validation");

        // Resource Guardian: Check for reasonable size
        let resource_guardian = card.patch_digest.len() < 10000 && card.signals.len() <= 30;

        // Context Validator: Check for coherence
        let context_validator = !card.problem_summary.is_empty()
            && !card.root_cause.is_empty()
            && !card.fix_strategy.is_empty();

        // Edge Sentinel: Check for edge cases
        let edge_sentinel = card.verification.len() > 20;

        // Calculate consensus (weighted)
        let weights = [2.5, 1.8, 2.0, 1.2, 1.0, 0.6]; // Same as SAT weights
        let approvals = [
            security_sentinel,
            formal_validator,
            ethics_guardian,
            resource_guardian,
            context_validator,
            edge_sentinel,
        ];
        let total_weight: f64 = weights.iter().sum();
        let approved_weight: f64 = weights
            .iter()
            .zip(approvals.iter())
            .filter(|(_, &approved)| approved)
            .map(|(w, _)| w)
            .sum();

        let consensus_score = approved_weight / total_weight;

        ExperienceSATResult {
            security_sentinel,
            formal_validator,
            ethics_guardian,
            resource_guardian,
            context_validator,
            edge_sentinel,
            consensus_score,
        }
    }

    /// Store a new experience with validation
    #[instrument(skip(self, card))]
    pub async fn store_experience(
        &self,
        card: ExperienceCard,
        source_repo: &str,
        source_issue: &str,
        transfer_ctx: TransferContext,
        transformation: TransformationFunction,
    ) -> anyhow::Result<String> {
        // 1. Calculate Ihsan score
        let ihsan = self.calculate_experience_ihsan(&card, &transfer_ctx);

        if ihsan < self.ihsan_threshold {
            warn!(
                ihsan_score = ihsan,
                threshold = self.ihsan_threshold,
                "Experience quality below threshold"
            );
            return Err(anyhow::anyhow!(
                "Experience quality below threshold: {:.3} < {:.3}",
                ihsan,
                self.ihsan_threshold
            ));
        }

        // 2. SAT validation
        let sat_result = self.validate_experience(&card).await;
        if !sat_result.all_approved() {
            warn!(vetoes = ?sat_result.vetoes(), "SAT validation failed");
            return Err(anyhow::anyhow!(
                "SAT validation failed: {:?}",
                sat_result.vetoes()
            ));
        }

        // 3. Create receipt
        let receipt = ExperienceReceipt {
            receipt_id: ExperienceReceipt::generate_id(),
            source_repo: source_repo.to_string(),
            source_issue: source_issue.to_string(),
            experience_card: card.clone(),
            lifecycle: ExperienceLifecycle::Nascent,
            graph_centrality: 0.0,
            transfer_context: transfer_ctx,
            transformation,
            ihsan_score: Fixed64::from_f64(ihsan),
            sat_consensus: sat_result.consensus_score,
            created_at: chrono::Utc::now(),
            last_retrieval: None,
            retrieval_count: 0,
            success_rate: 0.0,
        };

        // 4. Store in House of Wisdom
        let content = format!(
            "{}\n\nSignals: {}\n\nRoot Cause: {}\n\nFix Strategy: {}",
            card.problem_summary,
            card.signals.join(", "),
            card.root_cause,
            card.fix_strategy
        );

        let receipt_id = receipt.receipt_id.clone();
        self.wisdom
            .store_knowledge_with_embedding("Experience", &content, serde_json::to_value(&receipt)?)
            .await?;

        info!(
            receipt_id = %receipt_id,
            ihsan_score = ihsan,
            sat_consensus = sat_result.consensus_score,
            "Experience stored successfully"
        );

        Ok(receipt_id)
    }

    /// Stage-gated retrieval with SNR optimization
    #[instrument(skip(self))]
    pub async fn retrieve(
        &self,
        problem: &str,
        current_context: &TransferContext,
        limit: usize,
    ) -> anyhow::Result<RetrievalResult> {
        let start = std::time::Instant::now();
        let mut stages = Vec::new();
        let mut candidate_counts = Vec::new();

        // Stage 1: Hybrid search (vector + graph)
        stages.push("hybrid_search".to_string());
        let hybrid_results = self
            .wisdom
            .hybrid_search(problem, self.max_vector_candidates)
            .await?;

        let initial_count = hybrid_results.graph_nodes.len() + hybrid_results.vector_results.len();
        candidate_counts.push(initial_count);

        info!(
            stage = "hybrid_search",
            graph_results = hybrid_results.graph_nodes.len(),
            vector_results = hybrid_results.vector_results.len(),
            "Stage 1 complete"
        );

        // Stage 2: Parse and filter by transfer compatibility
        stages.push("transfer_filter".to_string());
        let mut candidates: Vec<RankedExperience> = Vec::new();

        // Process graph nodes
        for node in hybrid_results.graph_nodes {
            if let Some(exp) = self.try_parse_experience(&node) {
                let transfer_score =
                    self.calculate_transfer_compatibility(&exp.transfer_context, current_context);

                if transfer_score >= 0.3 {
                    let temporal_weight = self.calculate_temporal_weight(&exp);
                    candidates.push(RankedExperience {
                        relevance_score: node.relevance_score,
                        transfer_score,
                        temporal_weight,
                        final_score: node.relevance_score * transfer_score * temporal_weight,
                        experience: exp,
                    });
                }
            }
        }

        // Process vector results
        for vr in hybrid_results.vector_results {
            if let Some(exp) = self.try_parse_experience_from_vector(&vr) {
                let transfer_score =
                    self.calculate_transfer_compatibility(&exp.transfer_context, current_context);

                if transfer_score >= 0.3 {
                    let relevance = 1.0_f64 - (vr.distance as f64);
                    let temporal_weight = self.calculate_temporal_weight(&exp);
                    candidates.push(RankedExperience {
                        relevance_score: relevance,
                        transfer_score,
                        temporal_weight,
                        final_score: relevance * transfer_score * temporal_weight,
                        experience: exp,
                    });
                }
            }
        }

        candidate_counts.push(candidates.len());
        info!(
            stage = "transfer_filter",
            candidates = candidates.len(),
            "Stage 2 complete"
        );

        // Stage 3: SNR reranking
        stages.push("snr_rerank".to_string());
        candidates.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top results
        let final_results: Vec<_> = candidates.into_iter().take(limit).collect();
        candidate_counts.push(final_results.len());

        info!(
            stage = "snr_rerank",
            final_count = final_results.len(),
            query_time_ms = start.elapsed().as_millis(),
            "Retrieval complete"
        );

        Ok(RetrievalResult {
            experiences: final_results,
            query_time_ms: start.elapsed().as_millis() as u64,
            stages_executed: stages,
            candidates_at_each_stage: candidate_counts,
        })
    }

    /// Try to parse experience from knowledge node
    fn try_parse_experience(&self, _node: &KnowledgeNode) -> Option<ExperienceReceipt> {
        // Implementation would deserialize from node metadata
        // For now, return None as placeholder
        None
    }

    /// Try to parse experience from vector search result
    fn try_parse_experience_from_vector(
        &self,
        _result: &crate::vectors::VectorSearchResult,
    ) -> Option<ExperienceReceipt> {
        // Implementation would deserialize from metadata
        // For now, return None as placeholder
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experience_card_completeness() {
        let complete_card = ExperienceCard::new(
            "NPE when calling getUserProfile".to_string(),
            vec![
                "NullPointerException".to_string(),
                "getUserProfile".to_string(),
            ],
            "User object not validated before access".to_string(),
            "Add null check before accessing user properties".to_string(),
            "Added if (user != null) check".to_string(),
            "Unit tests for null user case added".to_string(),
        );
        assert!(complete_card.is_complete());

        let incomplete_card = ExperienceCard::new(
            "".to_string(),
            vec![],
            "".to_string(),
            "".to_string(),
            "".to_string(),
            "".to_string(),
        );
        assert!(!incomplete_card.is_complete());
    }

    #[test]
    fn test_signal_quality() {
        let optimal = ExperienceCard::new(
            "test".to_string(),
            (0..12).map(|i| format!("signal_{}", i)).collect(),
            "test".to_string(),
            "test".to_string(),
            "test".to_string(),
            "test".to_string(),
        );
        assert_eq!(optimal.signal_quality(), 1.0);

        let low = ExperienceCard::new(
            "test".to_string(),
            vec!["only_one".to_string()],
            "test".to_string(),
            "test".to_string(),
            "test".to_string(),
            "test".to_string(),
        );
        assert_eq!(low.signal_quality(), 0.5);
    }

    #[test]
    fn test_transfer_context_language_mismatch() {
        let mem = SovereignExperienceMemory {
            wisdom: HouseOfWisdom::new(
                "bolt://test".to_string(),
                "test".to_string(),
                "test".to_string(),
            ),
            ihsan_threshold: 0.85,
            max_vector_candidates: 200,
            max_graph_candidates: 50,
        };

        let rust_ctx = TransferContext::rust();
        let python_ctx = TransferContext::python();

        let score = mem.calculate_transfer_compatibility(&rust_ctx, &python_ctx);
        assert_eq!(score, 0.0); // Language mismatch = incompatible
    }

    #[test]
    fn test_lifecycle_calculation() {
        let mut receipt = ExperienceReceipt {
            receipt_id: "EXP-TEST".to_string(),
            source_repo: "test/repo".to_string(),
            source_issue: "123".to_string(),
            experience_card: ExperienceCard::new(
                "test".to_string(),
                vec!["test".to_string()],
                "test".to_string(),
                "test".to_string(),
                "test".to_string(),
                "test".to_string(),
            ),
            lifecycle: ExperienceLifecycle::Nascent,
            graph_centrality: 0.5,
            transfer_context: TransferContext::rust(),
            transformation: TransformationFunction::default(),
            ihsan_score: Fixed64::from_f64(0.9),
            sat_consensus: 1.0,
            created_at: chrono::Utc::now(),
            last_retrieval: None,
            retrieval_count: 0,
            success_rate: 0.0,
        };

        assert_eq!(receipt.calculate_lifecycle(), ExperienceLifecycle::Nascent);

        // Simulate age
        receipt.created_at = chrono::Utc::now() - chrono::Duration::days(100);
        assert_eq!(receipt.calculate_lifecycle(), ExperienceLifecycle::Mature);
    }

    #[test]
    fn test_sat_validation_result() {
        let result = ExperienceSATResult {
            security_sentinel: true,
            formal_validator: true,
            ethics_guardian: true,
            resource_guardian: true,
            context_validator: true,
            edge_sentinel: true,
            consensus_score: 1.0,
        };
        assert!(result.all_approved());
        assert!(result.vetoes().is_empty());

        let partial = ExperienceSATResult {
            security_sentinel: false,
            formal_validator: true,
            ethics_guardian: false,
            resource_guardian: true,
            context_validator: true,
            edge_sentinel: true,
            consensus_score: 0.6,
        };
        assert!(!partial.all_approved());
        assert_eq!(
            partial.vetoes(),
            vec!["security_sentinel", "ethics_guardian"]
        );
    }
}
