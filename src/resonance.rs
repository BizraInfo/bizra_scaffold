// src/resonance.rs
/*
SOVEREIGN RESONANCE MESH v1.0
An autonomous self-optimizing neural-symbolic mesh that:
1. Monitors Signal-to-Noise Ratio (SNR) across Graph-of-Thought (GoT) nodes
2. Prunes low-resonance (<0.3 SNR) paths in real-time
3. Amplifies high-resonance (>0.8 SNR) pathways
4. Achieves autonomous resonance through continuous optimization
*/

use petgraph::graph::DiGraph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{info, instrument, warn};

/// Resonance metrics for a GoT node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceMetrics {
    pub node_id: String,
    pub signal_strength: f64,      // 0.0-1.0: Strength of signal (clarity)
    pub noise_level: f64,          // 0.0-1.0: Level of noise (uncertainty)
    pub resonance_score: f64,      // signal_strength / (signal_strength + noise_level + 1e-9)
    pub ihsan_correlation: f64,    // Correlation with Ihsān score
    pub pruning_threshold: f64,    // Dynamic threshold for this node
    pub last_updated: u64,         // Timestamp
    pub amplification_factor: f64, // How much to amplify if high-resonance
}

impl ResonanceMetrics {
    pub fn new(content: &str) -> Self {
        use crate::fixed::Fixed64;
        use crate::snr::SNREngine;
        use crate::types::AgentResult;
        use std::collections::HashMap;

        // Use the Elite SNR Engine to initialize metrics
        let bootstrap_result = AgentResult {
            agent_name: "resonance_init".to_string(),
            contribution: content.to_string(),
            confidence: Fixed64::from_i64(50), // Initial conservative confidence
            ihsan_score: Fixed64::ZERO,
            metadata: HashMap::new(),
            execution_time: std::time::Duration::from_secs(0),
        };

        let snr_report = SNREngine::score(&bootstrap_result);

        Self {
            node_id: uuid::Uuid::new_v4().to_string(),
            signal_strength: snr_report.signal.to_f64(),
            noise_level: snr_report.noise.to_f64(),
            resonance_score: snr_report.ratio.to_f64(),
            ihsan_correlation: snr_report.information_density.to_f64(),
            pruning_threshold: 1.0, // Default institutional threshold
            last_updated: 0,
            amplification_factor: 1.0,
        }
    }

    /// Calculate SNR: Signal-to-Noise Ratio (Normalized to 0.0-1.0)
    pub fn calculate_snr(&self) -> f64 {
        // Ratio SNR: signal / (signal + noise) for 0..1 normalization
        self.signal_strength / (self.signal_strength + self.noise_level + 1e-9)
    }

    /// Check if node should be pruned (low resonance)
    pub fn should_prune(&self) -> bool {
        self.calculate_snr() < self.pruning_threshold
    }

    /// Check if node should be amplified (high resonance)
    pub fn should_amplify(&self) -> bool {
        self.calculate_snr() > 0.8 && self.ihsan_correlation > 0.7
    }
}

/// Graph-of-Thought node with resonance data
#[derive(Debug, Clone)]
pub struct GoTNode {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub resonance: ResonanceMetrics,
    pub children: Vec<String>,
    pub parents: Vec<String>,
}

/// Sovereign Resonance Mesh - Self-Optimizing Neural-Symbolic Graph
///
/// The Resonance Mesh implements an autonomous optimization loop that:
/// 1. Monitors Signal-to-Noise Ratio (SNR) across Graph-of-Thought nodes
/// 2. Prunes low-resonance (<threshold) paths in real-time
/// 3. Amplifies high-resonance (>0.8) pathways
/// 4. Tracks the Wisdom Root for constitutional stability (Circuit 14)
///
/// # SNR Calculation
///
/// ```text
/// SNR = signal_strength / (signal_strength + noise_level + 1e-9)
/// ```
///
/// # Example
///
/// ```rust,ignore
/// let (mesh, rx) = ResonanceMesh::new(0.3, 1.2, true);
/// mesh.add_node(node).await?;
/// let stats = mesh.get_stats().await?;
/// println!("Average SNR: {}", stats.average_snr);
/// ```
#[derive(Clone)]
pub struct ResonanceMesh {
    /// Directed graph of GoT nodes weighted by resonance
    graph: Arc<RwLock<DiGraph<GoTNode, f64>>>,
    /// Node ID → Graph index mapping
    node_index: Arc<RwLock<HashMap<String, petgraph::graph::NodeIndex>>>,
    /// Channel for broadcasting resonance updates
    metrics_tx: mpsc::Sender<ResonanceUpdate>,
    /// SNR threshold below which nodes are pruned
    pruning_threshold: f64,
    /// Multiplier for amplifying high-resonance nodes
    amplification_factor: f64,
    /// Enable autonomous optimization loop
    autonomous_mode: bool,
    /// Genesis anchor for constitutional stability (Circuit 14)
    wisdom_root_id: Arc<RwLock<Option<String>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResonanceUpdate {
    pub node_id: String,
    pub old_snr: f64,
    pub new_snr: f64,
    pub action: ResonanceAction,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize)]
pub enum ResonanceAction {
    Pruned,
    Amplified,
    Maintained,
    ThresholdAdjusted(f64),
}

impl ResonanceMesh {
    /// Create a new Resonance Mesh
    pub fn new(
        initial_threshold: f64,
        amplification_factor: f64,
        autonomous_mode: bool,
    ) -> (Self, mpsc::Receiver<ResonanceUpdate>) {
        let (metrics_tx, metrics_rx) = mpsc::channel(1000);

        let mesh = Self {
            graph: Arc::new(RwLock::new(DiGraph::new())),
            node_index: Arc::new(RwLock::new(HashMap::new())),
            metrics_tx,
            pruning_threshold: initial_threshold,
            amplification_factor,
            autonomous_mode,
            wisdom_root_id: Arc::new(RwLock::new(None)),
        };

        (mesh, metrics_rx)
    }

    /// Add a node to the resonance mesh
    #[instrument(skip(self))]
    pub async fn add_node(&self, node: GoTNode) -> Result<(), ResonanceError> {
        let node_id = node.id.clone();

        {
            let mut graph = self.graph.write().await;
            let mut index = self.node_index.write().await;

            let node_idx = graph.add_node(node.clone());
            index.insert(node_id.clone(), node_idx);
        }

        info!("Node added to resonance mesh: {}", node_id);

        // Initial resonance calculation
        self.calculate_node_resonance(&node_id).await?;

        // Track as Wisdom Root if it's the first node or explicitly marked
        let mut root_id = self.wisdom_root_id.write().await;
        if root_id.is_none() || node.metadata.contains_key("is_wisdom_root") {
            *root_id = Some(node_id.clone());
            info!("🌟 Wisdom Root identified: {}", node_id);
        }

        Ok(())
    }

    /// PEAK MASTERPIECE: Winning Signal Selection
    /// Identifies the node with the highest SNR score in the mesh.
    pub async fn get_winning_signal(&self) -> Option<GoTNode> {
        let graph = self.graph.read().await;
        let mut best_node: Option<GoTNode> = None;
        let mut max_snr = -1.0;

        for node_idx in graph.node_indices() {
            let node = &graph[node_idx];
            let snr = node.resonance.calculate_snr();
            if snr > max_snr {
                max_snr = snr;
                best_node = Some(node.clone());
            }
        }
        best_node
    }

    /// Add an edge between two nodes with weight
    pub async fn add_edge(
        &self,
        from_id: &str,
        to_id: &str,
        weight: f64,
    ) -> Result<(), ResonanceError> {
        let index = self.node_index.read().await;

        let from_idx = index
            .get(from_id)
            .ok_or_else(|| ResonanceError::NodeNotFound(from_id.to_string()))?;

        let to_idx = index
            .get(to_id)
            .ok_or_else(|| ResonanceError::NodeNotFound(to_id.to_string()))?;

        let from_idx = *from_idx;
        let to_idx = *to_idx;
        drop(index); // Release read lock before write

        let mut graph = self.graph.write().await;
        graph.add_edge(from_idx, to_idx, weight);

        info!("Edge added: {} -> {} (weight: {})", from_id, to_id, weight);

        Ok(())
    }

    /// Calculate resonance for a specific node
    #[instrument(skip(self))]
    pub async fn calculate_node_resonance(&self, node_id: &str) -> Result<f64, ResonanceError> {
        let (node_idx, old_snr, signal_strength, noise_level, ihsan_correlation) = {
            let graph = self.graph.read().await;
            let index = self.node_index.read().await;

            let node_idx = index
                .get(node_id)
                .ok_or_else(|| ResonanceError::NodeNotFound(node_id.to_string()))?;

            let node = graph
                .node_weight(*node_idx)
                .expect("Resonance node weight must exist during calculation");
            let old_snr = node.resonance.calculate_snr();

            // Calculate new resonance based on:
            // 1. Signal strength from neural confidence
            // 2. Noise level from uncertainty metrics
            // 3. Ihsān correlation from FATE verification
            // 4. Network connectivity (degree centrality)

            // Note: degree_centrality is not used in signal/noise calculation here,
            // but could be integrated into calculate_signal_strength or calculate_noise_level.
            let signal_strength = self.calculate_signal_strength(node);
            let noise_level = self.calculate_noise_level(node);
            let ihsan_correlation = self.calculate_ihsan_correlation(node);

            (
                *node_idx,
                old_snr,
                signal_strength,
                noise_level,
                ihsan_correlation,
            )
        };

        let new_snr = signal_strength / (signal_strength + noise_level + 1e-9);

        // Update node resonance
        {
            let mut graph = self.graph.write().await;
            let node = graph
                .node_weight_mut(node_idx)
                .expect("Resonance node weight must exist during update");

            node.resonance.signal_strength = signal_strength;
            node.resonance.noise_level = noise_level;
            node.resonance.resonance_score = new_snr;
            node.resonance.ihsan_correlation = ihsan_correlation;
            node.resonance.last_updated = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Clock went backwards during resonance update")
                .as_secs();
        }

        // Send update
        let update = ResonanceUpdate {
            node_id: node_id.to_string(),
            old_snr,
            new_snr,
            action: ResonanceAction::Maintained,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let _ = self.metrics_tx.send(update).await;

        Ok(new_snr)
    }

    /// Autonomous optimization of the resonance mesh
    #[instrument(skip(self))]
    pub async fn optimize_resonance(&self) -> Result<OptimizationResult, ResonanceError> {
        if !self.autonomous_mode {
            return Err(ResonanceError::AutonomousModeDisabled);
        }

        let start_time = std::time::Instant::now();
        let mut pruned = 0;
        let mut amplified = 0;

        // 1. Scan all nodes for resonance (get IDs first to avoid holding lock)
        let nodes_to_check: Vec<String> = {
            let index = self.node_index.read().await;
            index.keys().cloned().collect()
        };

        for node_id in &nodes_to_check {
            // Re-calculate resonance (handles its own locking)
            let snr = self.calculate_node_resonance(node_id).await?;

            // 2. Prune low-resonance nodes (< threshold)
            if snr < self.pruning_threshold {
                self.prune_node(node_id).await?;
                pruned += 1;

                warn!("Pruned low-resonance node: {} (SNR: {:.3})", node_id, snr);
            }
            // 3. Amplify high-resonance nodes (> 0.8)
            else if snr > 0.8 {
                self.amplify_node(node_id).await?;
                amplified += 1;

                info!(
                    "Amplified high-resonance node: {} (SNR: {:.3})",
                    node_id, snr
                );
            }
        }

        // 4. Adjust global threshold based on mesh state
        let _new_threshold = self.adjust_pruning_threshold().await?;

        // 5. Monitor Wisdom Root Drift (Circuit 14)
        self.monitor_root_ihsan().await?;

        let elapsed = start_time.elapsed();

        Ok(OptimizationResult {
            pruned_nodes: pruned,
            amplified_nodes: amplified,
            threshold_adjustments: 1,
            new_pruning_threshold: _new_threshold,
            elapsed_ms: elapsed.as_millis() as u64,
            mesh_size: self.get_node_count().await,
        })
    }

    /// Prune a node from the mesh
    #[instrument(skip(self))]
    pub async fn prune_node(&self, node_id: &str) -> Result<(), ResonanceError> {
        let node_idx = {
            let mut index = self.node_index.write().await;
            index
                .remove(node_id)
                .ok_or_else(|| ResonanceError::NodeNotFound(node_id.to_string()))?
        };

        {
            let mut graph = self.graph.write().await;
            graph.remove_node(node_idx);
        }

        info!("Pruned node: {}", node_id);

        let update = ResonanceUpdate {
            node_id: node_id.to_string(),
            old_snr: 0.0, // N/A for pruning
            new_snr: 0.0,
            action: ResonanceAction::Pruned,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let _ = self.metrics_tx.send(update).await;

        Ok(())
    }

    /// Amplify a node in the mesh
    #[instrument(skip(self))]
    pub async fn amplify_node(&self, node_id: &str) -> Result<(), ResonanceError> {
        let node_idx = {
            let index = self.node_index.read().await;
            *index
                .get(node_id)
                .ok_or_else(|| ResonanceError::NodeNotFound(node_id.to_string()))?
        };

        {
            let mut graph = self.graph.write().await;
            let node = graph
                .node_weight_mut(node_idx)
                .expect("Resonance node weight must exist during amplification");

            node.resonance.signal_strength *= self.amplification_factor;
            node.resonance.noise_level *= 0.9; // Noise reduction on amplification

            // Re-normalize scores
            node.resonance.resonance_score = node.resonance.calculate_snr();
        }

        info!("Amplified node: {}", node_id);

        let update = ResonanceUpdate {
            node_id: node_id.to_string(),
            old_snr: 0.0, // N/A for simplified update
            new_snr: 0.0,
            action: ResonanceAction::Amplified,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let _ = self.metrics_tx.send(update).await;

        Ok(())
    }

    /// Adjust the global pruning threshold based on mesh health
    async fn adjust_pruning_threshold(&self) -> Result<f64, ResonanceError> {
        let stats = self.get_resonance_stats().await?;

        let mut new_threshold = self.pruning_threshold;

        // If mesh is too noisy (average SNR < 0.5), increase threshold
        if stats.average_snr < 0.5 && stats.total_nodes > 10 {
            new_threshold += 0.05;
        }
        // If mesh is extremely healthy, slightly lower threshold to allow more exploration
        else if stats.average_snr > 0.8 && stats.total_nodes < 100 {
            new_threshold -= 0.02;
        }

        Ok(new_threshold.clamp(0.1, 0.7))
    }

    /// Get the number of nodes in the mesh
    pub async fn get_node_count(&self) -> usize {
        let index = self.node_index.read().await;
        index.len()
    }

    /// Get all node IDs
    pub async fn get_all_node_ids(&self) -> Vec<String> {
        let index = self.node_index.read().await;
        index.keys().cloned().collect()
    }

    /// Get statistics for the resonance mesh
    pub async fn get_resonance_stats(&self) -> Result<ResonanceStats, ResonanceError> {
        let graph = self.graph.read().await;

        let mut total_snr = 0.0;
        let mut high_resonance_nodes = 0;
        let mut low_resonance_nodes = 0;
        let node_count = graph.node_count();

        if node_count == 0 {
            return Ok(ResonanceStats {
                total_nodes: 0,
                average_snr: 1.0,
                high_resonance_nodes: 0,
                low_resonance_nodes: 0,
                pruning_threshold: self.pruning_threshold,
                mesh_connectivity: 1.0,
                autonomous_mode: self.autonomous_mode,
            });
        }

        for node in graph.node_weights() {
            let snr = node.resonance.calculate_snr();
            total_snr += snr;
            if snr > 0.8 {
                high_resonance_nodes += 1;
            } else if snr < self.pruning_threshold {
                low_resonance_nodes += 1;
            }
        }

        let edge_count = graph.edge_count();
        let mesh_connectivity = if node_count > 1 {
            edge_count as f64 / (node_count * (node_count - 1)) as f64
        } else {
            1.0
        };

        Ok(ResonanceStats {
            total_nodes: node_count,
            average_snr: total_snr / node_count as f64,
            high_resonance_nodes,
            low_resonance_nodes,
            pruning_threshold: self.pruning_threshold,
            mesh_connectivity,
            autonomous_mode: self.autonomous_mode,
        })
    }

    /// Helper: Calculate signal strength for a node
    fn calculate_signal_strength(&self, node: &GoTNode) -> f64 {
        // Signal strength based on:
        // 1. Embedding norm (stronger embeddings = clearer signal)
        // 2. Metadata confidence scores
        // 3. Node connectivity

        let embedding_norm = self.norm(&node.embedding);
        let metadata_confidence = node
            .metadata
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let degree = (node.children.len() + node.parents.len()) as f64;
        let connectivity_factor = 1.0 - (-degree).exp(); // Sigmoid-like

        (embedding_norm * 0.4 + metadata_confidence * 0.4 + connectivity_factor * 0.2)
            .clamp(0.0, 1.0)
    }

    /// Helper: Calculate noise level for a node
    fn calculate_noise_level(&self, node: &GoTNode) -> f64 {
        // Noise level based on:
        // 1. Embedding entropy
        // 2. Contradiction detection
        // 3. Temporal decay

        let entropy = self.calculate_embedding_entropy(&node.embedding);
        let contradictions = node
            .metadata
            .get("contradiction_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let temporal_decay = self.calculate_temporal_decay(node.resonance.last_updated);

        (entropy * 0.5 + contradictions * 0.3 + temporal_decay * 0.2).clamp(0.0, 1.0)
    }

    /// Helper: Calculate Ihsān correlation
    fn calculate_ihsan_correlation(&self, node: &GoTNode) -> f64 {
        node.metadata
            .get("ihsan_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5)
    }

    /// Helper: Calculate embedding norm
    fn norm(&self, embedding: &[f32]) -> f64 {
        let sum_sq: f32 = embedding.iter().map(|&x| x * x).sum();
        (sum_sq as f64).sqrt()
    }

    /// Helper: Calculate embedding entropy
    fn calculate_embedding_entropy(&self, embedding: &[f32]) -> f64 {
        let norm = self.norm(embedding) as f32;
        if norm < 1e-9 {
            return 0.0;
        }

        let normalized: Vec<f32> = embedding.iter().map(|&x| x / norm).collect();

        // Shannon entropy
        let mut entropy = 0.0;
        for &value in &normalized {
            if value > 1e-9 {
                entropy -= (value as f64) * (value as f64).ln();
            }
        }

        entropy / (normalized.len() as f64).ln()
    }

    /// Helper: Calculate temporal decay
    fn calculate_temporal_decay(&self, last_updated: u64) -> f64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let age = now.saturating_sub(last_updated) as f64;

        // Exponential decay: 50% every 24 hours
        1.0 - (-age / 86400.0 * 0.693).exp() // 0.693 = ln(2)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResonanceStats {
    pub total_nodes: usize,
    pub average_snr: f64,
    pub high_resonance_nodes: usize,
    pub low_resonance_nodes: usize,
    pub pruning_threshold: f64,
    pub mesh_connectivity: f64,
    pub autonomous_mode: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct OptimizationResult {
    pub pruned_nodes: usize,
    pub amplified_nodes: usize,
    pub threshold_adjustments: usize,
    pub new_pruning_threshold: f64,
    pub elapsed_ms: u64,
    pub mesh_size: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum ResonanceError {
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Autonomous mode disabled")]
    AutonomousModeDisabled,

    #[error("Channel error: {0}")]
    ChannelError(String),

    #[error("Graph error: {0}")]
    GraphError(String),

    #[error("Constitutional drift detected: {0}")]
    ConstitutionalDrift(String),
}

impl ResonanceMesh {
    /// Circuit 14: Monitor Wisdom Root for Ihsān drift
    pub async fn monitor_root_ihsan(&self) -> Result<(), ResonanceError> {
        let root_id = {
            let id_lock = self.wisdom_root_id.read().await;
            id_lock.clone()
        };

        if let Some(id) = root_id {
            let snr = self.calculate_node_resonance(&id).await?;

            // Constitutional Threshold (Ihsan Floor)
            if snr < 0.5 {
                warn!("🚨 Circuit 14: Wisdom Root drift detected (SNR: {:.3}) - Triggering mesh rebirth", snr);
                self.mesh_rebirth().await?;
                return Err(ResonanceError::ConstitutionalDrift(format!(
                    "Wisdom Root {} drifted to SNR {:.3}",
                    id, snr
                )));
            }
        }
        Ok(())
    }

    /// Implement mesh rebirth logic (Constitutional Safe Mode)
    pub async fn mesh_rebirth(&self) -> Result<(), ResonanceError> {
        info!("🌀 Initiating Mesh Rebirth - Constitutional Safe Mode Active");

        // 1. Freeze mesh (implied by clearing it under write lock)
        let mut graph = self.graph.write().await;
        let mut index = self.node_index.write().await;

        // 2. Clear corrupted state
        graph.clear();
        index.clear();

        // 3. Reset threshold to conservative baseline
        // Note: pruning_threshold is not Arc<RwLock>, so we can't easily mut it here if shared
        // without wrapping it, but for now we focus on graph rebirth.

        // 4. In a real system, we would reload from TPM-measured genesis here
        info!("✅ Mesh rebirth complete - Awaiting fresh constitutional anchor");

        Ok(())
    }
}

// Implement Send + Sync for thread safety
unsafe impl Send for ResonanceMesh {}
unsafe impl Sync for ResonanceMesh {}
