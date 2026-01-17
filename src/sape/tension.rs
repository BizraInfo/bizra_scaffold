// src/sape/tension.rs
use crate::resonance::GoTNode;
use serde::{Deserialize, Serialize};
use tracing::{instrument, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensionReport {
    pub tension_score: f64,
    pub contradictions: Vec<Contradiction>,
    pub resolution_strategy: TensionResolution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub node_a: String,
    pub node_b: String,
    pub conflict_type: String,
    pub intensity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TensionResolution {
    None,
    DeepSynthesis,
    StrategicPivot,
    FormalVeto,
}

pub struct TensionStudio {
    threshold: f64,
}

impl TensionStudio {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Analyzes a reasoning graph (GoT) for logical tensions and contradictions.
    /// This is the "Conscience" of the Reasoning Engine.
    #[instrument(skip(self, nodes))]
    pub fn analyze_graph(&self, nodes: &[GoTNode]) -> TensionReport {
        let mut contradictions = Vec::new();
        let mut max_intensity = 0.0;

        // Semantic Contradiction Detection (Heuristic-based for Node0)
        // In a full implementation, this would use cross-node semantic embeddings
        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                if let Some(conflict) = self.check_conflict(&nodes[i], &nodes[j]) {
                    if conflict.intensity > max_intensity {
                        max_intensity = conflict.intensity;
                    }
                    contradictions.push(conflict);
                }
            }
        }

        let tension_score = if nodes.is_empty() {
            0.0
        } else {
            (contradictions.len() as f64 / nodes.len() as f64).min(1.0) * max_intensity
        };

        let strategy = if tension_score > self.threshold * 1.5 {
            TensionResolution::FormalVeto
        } else if tension_score > self.threshold {
            TensionResolution::DeepSynthesis
        } else if tension_score > self.threshold * 0.5 {
            TensionResolution::StrategicPivot
        } else {
            TensionResolution::None
        };

        if strategy != TensionResolution::None {
            warn!(score = %tension_score, "⚖️ Tension detected in Reasoning Graph");
        }

        TensionReport {
            tension_score,
            contradictions,
            resolution_strategy: strategy,
        }
    }

    fn check_conflict(&self, a: &GoTNode, b: &GoTNode) -> Option<Contradiction> {
        // Detecting logical opposites (Heuristic)
        let content_a = a.content.to_lowercase();
        let content_b = b.content.to_lowercase();

        let patterns = [
            ("allow", "prevent"),
            ("safe", "danger"),
            ("high", "low"),
            ("increase", "decrease"),
            ("centralize", "decentralize"),
            ("always", "never"),
        ];

        for (p1, p2) in patterns {
            if (content_a.contains(p1) && content_b.contains(p2))
                || (content_a.contains(p2) && content_b.contains(p1))
            {
                return Some(Contradiction {
                    node_a: a.id.clone(),
                    node_b: b.id.clone(),
                    conflict_type: format!("Semantic Opposition: {} vs {}", p1, p2),
                    intensity: 0.85,
                });
            }
        }

        None
    }
}

impl Default for TensionStudio {
    fn default() -> Self {
        Self::new(0.6)
    }
}
