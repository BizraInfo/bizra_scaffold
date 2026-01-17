use crate::fixed::Fixed64;
use crate::types::AgentResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// BIZRA Sovereign Ledger - Layer 5 (Economic) & Layer 1 (Consensus)
/// Implements Pillar #1 (Graph-of-Thoughts Merkle State) and #6 (PoI Tokens).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Ledger {
    pub state_root: String,
    pub bzc_balances: HashMap<String, Fixed64>,
    pub bzt_balances: HashMap<String, Fixed64>,
    pub transactions: Vec<Transaction>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Transaction {
    pub id: String,
    pub sender: String,
    pub receiver: String,
    pub amount_bzc: Fixed64,
    pub impact_score: Fixed64,
    pub evidence_hash: String,
    pub signature: String,
}

impl Default for Ledger {
    fn default() -> Self {
        Self::new()
    }
}

impl Ledger {
    pub fn new() -> Self {
        Self {
            state_root: "genesis_root_hash".to_string(),
            bzc_balances: HashMap::new(),
            bzt_balances: HashMap::new(),
            transactions: Vec::new(),
        }
    }

    /// Record impact and update the global state root.
    pub fn record_impact(
        &mut self,
        agent: &str,
        impact_score: Fixed64,
        bzc_minted: Fixed64,
        bzt_power: Fixed64,
        evidence: &[AgentResult],
    ) -> String {
        // Calculate evidence hash (Merkle-lite)
        let mut hasher = Sha256::new();
        for res in evidence {
            hasher.update(res.contribution.as_bytes());
        }
        let evidence_hash = format!("{:x}", hasher.finalize());

        let tx = Transaction {
            id: uuid::Uuid::new_v4().to_string(),
            sender: "SYSTEM".to_string(),
            receiver: agent.to_string(),
            amount_bzc: bzc_minted,
            impact_score,
            evidence_hash,
            signature: "seal_of_ihsan".to_string(),
        };

        // Update balances
        let bzc_entry = self
            .bzc_balances
            .entry(agent.to_string())
            .or_insert(Fixed64::from_f64(0.0));
        *bzc_entry = bzc_entry.saturating_add(bzc_minted);

        let bzt_entry = self
            .bzt_balances
            .entry(agent.to_string())
            .or_insert(Fixed64::from_f64(0.0));
        *bzt_entry = bzt_entry.saturating_add(bzt_power);

        self.transactions.push(tx);

        // Update state root
        self.recalculate_root()
    }

    fn recalculate_root(&mut self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.state_root.as_bytes());
        for tx in &self.transactions {
            hasher.update(tx.id.as_bytes());
        }
        self.state_root = format!("{:x}", hasher.finalize());
        self.state_root.clone()
    }
}
