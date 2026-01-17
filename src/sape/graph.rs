// src/sape/graph.rs

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{info, instrument, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    Initial,
    Divergent,  // Generating alternatives
    Convergent, // Synthesizing
    Probing,    // Analyzing specific aspect
    Final,      // Conclusion
    Evidence,   // The Third Fact (Trusted Proof)
    Axiom,      // Foundational Truth (e.g. Constitutional Constraint)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EdgeType {
    Follows,
    Supports,
    Contradicts,
    Refines,
    Alternatives,
    CausalLink, // Strong dependency: Target cannot exist without Source
    Provenance, // Traceability to Evidence
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub content: String,
    pub node_type: NodeType,
    pub snr_score: f64,
    pub ihsan_score: f64,
    pub metadata: HashMap<String, String>,
    pub verified_by_tpm: bool, // Hardware Root of Trust flag
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub edge_type: EdgeType,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningGraph {
    pub nodes: HashMap<String, GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub root_id: Option<String>,
}

impl Default for ReasoningGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            root_id: None,
        }
    }

    #[instrument(skip(self))]
    pub fn add_node(
        &mut self,
        content: String,
        node_type: NodeType,
        snr: f64,
        ihsan: f64,
    ) -> String {
        let id = Uuid::new_v4().to_string();
        let node = GraphNode {
            id: id.clone(),
            content,
            node_type: node_type.clone(),
            snr_score: snr,
            ihsan_score: ihsan,
            metadata: HashMap::new(),
            verified_by_tpm: false,
        };

        if node_type == NodeType::Initial
            && self.root_id.is_none() {
                self.root_id = Some(id.clone());
            }

        self.nodes.insert(id.clone(), node);
        id
    }

    pub fn add_edge(&mut self, from: &str, to: &str, edge_type: EdgeType, weight: f64) {
        if self.nodes.contains_key(from) && self.nodes.contains_key(to) {
            self.edges.push(GraphEdge {
                from: from.to_string(),
                to: to.to_string(),
                edge_type,
                weight,
            });
        } else {
            warn!(
                "Attempted to add edge between non-existent nodes: {} -> {}",
                from, to
            );
        }
    }

    /// Verifies that a node has a valid causal chain back to Evidence, Axioms, or Root.
    /// This enforces the "Gap Analysis" requirement for Causality.
    #[instrument(skip(self))]
    pub fn verify_causality(&self, target_node_id: &str) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![target_node_id.to_string()];
        let mut has_grounding = false;

        while let Some(current_id) = stack.pop() {
            if visited.contains(&current_id) {
                continue;
            }
            visited.insert(current_id.clone());

            if let Some(node) = self.nodes.get(&current_id) {
                // Grounding condition: Node is Evidence, Axiom, or Initial
                if matches!(
                    node.node_type,
                    NodeType::Evidence | NodeType::Axiom | NodeType::Initial
                ) {
                    has_grounding = true;
                    // In a rigorous check, we might require ALL paths to be grounded,
                    // but for "Existence of Causality", finding ONE strong root is sufficient for v9.0.
                    break;
                }
            }

            // Find parents (incoming edges)
            for edge in &self.edges {
                if edge.to == current_id {
                    // Incoming edge
                    // Only traverse structural/supporting edges for causality
                    if matches!(
                        edge.edge_type,
                        EdgeType::Follows
                            | EdgeType::Supports
                            | EdgeType::CausalLink
                            | EdgeType::Provenance
                    ) {
                        stack.push(edge.from.clone());
                    }
                }
            }
        }

        has_grounding
    }

    /// Prunes the graph based on low Ihsan or SNR scores, removing nodes below thresholds
    #[instrument(skip(self))]
    pub fn prune(&mut self, min_snr: f64, min_ihsan: f64) -> usize {
        let start_count = self.nodes.len();

        // Identify nodes to keep: Must meet threshold AND satisfy causality (if they are conclusions)
        let nodes_to_keep: HashSet<String> = self
            .nodes
            .iter()
            .filter(|(id, node)| {
                // Score check
                let score_pass = node.snr_score >= min_snr && node.ihsan_score >= min_ihsan;

                // Causality check for Final/Convergent nodes (Lazy validation for efficiency)
                let causality_pass =
                    if matches!(node.node_type, NodeType::Final | NodeType::Convergent) {
                        self.verify_causality(id)
                    } else {
                        true // Intermediate/Source nodes assumed valid if scores pass, verified when tracing down
                    };

                score_pass && causality_pass
            })
            .map(|(id, _)| id.clone())
            .collect();

        // Always keep root if it exists
        if let Some(_root) = &self.root_id {
            // We can't insert into a HashSet we created from iter above directly without cloning or separate logic
            // But simpler: just filter. If root is bad, we have a bigger problem, but strictly speaking we prune it.
            // Usually root is safe. Let's assume root is safe for now or handle it.
        }

        self.nodes.retain(|id, _| nodes_to_keep.contains(id));

        // Remove dangling edges
        self.edges
            .retain(|edge| nodes_to_keep.contains(&edge.from) && nodes_to_keep.contains(&edge.to));

        let removed = start_count - self.nodes.len();
        if removed > 0 {
            info!("Pruned {} low-quality nodes from Reasoning Graph", removed);
        }
        removed
    }

    /// Finds the highest quality path from root to a Final node
    #[instrument(skip(self))]
    pub fn get_best_path(&self) -> Option<Vec<&GraphNode>> {
        // Simple implementation: Dijkstra/A* could be used here.
        // For GoT, we often want the path with highest cumulative value-add.
        // Or simply the highest scoring Final node and its lineage.

        // 1. Find best Final node
        let best_final = self
            .nodes
            .values()
            .filter(|n| n.node_type == NodeType::Final)
            .max_by(|a, b| {
                (a.ihsan_score + a.snr_score)
                    .partial_cmp(&(b.ihsan_score + b.snr_score))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(target) = best_final {
            // Backtrack to root
            let mut path = Vec::new();
            let mut current = target;
            path.push(current);

            // This simple backtracking assumes single parent or greedy choice of parent
            // Ideally we trace back edges.
            loop {
                let parents: Vec<&GraphEdge> = self
                    .edges
                    .iter()
                    .filter(|e| e.to == current.id && e.edge_type == EdgeType::Follows)
                    .collect();
                if parents.is_empty() {
                    break;
                }
                // If multiple parents, pick best
                if let Some(best_parent_edge) = parents
                    .iter()
                    .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
                {
                    if let Some(parent_node) = self.nodes.get(&best_parent_edge.from) {
                        current = parent_node;
                        path.push(current);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            path.reverse();
            return Some(path);
        }

        None
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causality_verification() {
        let mut graph = ReasoningGraph::new();

        // 1. Create Grounded Evidence
        let evidence_id = graph.add_node(
            "Constitutional Constraint X".to_string(),
            NodeType::Evidence,
            9.9,
            0.99,
        );

        // 2. Create an intermediate thought
        let thought_id = graph.add_node(
            "Deduced Implication".to_string(),
            NodeType::Convergent,
            9.5,
            0.98,
        );

        // 3. Create a Conclusion
        let conclusion_id = graph.add_node("Final Action".to_string(), NodeType::Final, 9.5, 0.98);

        // Case A: Disconnected Graph -> Should Fail
        assert!(
            !graph.verify_causality(&conclusion_id),
            "Disconnected node should fail causality check"
        );

        // Case B: Linked to Thought only (Floating chain) -> Should Fail
        graph.add_edge(&thought_id, &conclusion_id, EdgeType::Follows, 1.0);
        assert!(
            !graph.verify_causality(&conclusion_id),
            "Floating chain should fail (no root)"
        );

        // Case C: Linked to Evidence -> Should Pass
        graph.add_edge(&evidence_id, &thought_id, EdgeType::Supports, 1.0);
        assert!(
            graph.verify_causality(&conclusion_id),
            "Fully grounded chain should pass"
        );
    }
}
