use crate::federation::PoiReceipt;
use anyhow::{bail, Result};
use std::collections::HashMap;

pub struct Sentinel {
    receipt_cache: HashMap<String, String>, // map output_hash -> node_uid
}

impl Default for Sentinel {
    fn default() -> Self {
        Self::new()
    }
}

impl Sentinel {
    pub fn new() -> Self {
        Self {
            receipt_cache: HashMap::new(),
        }
    }

    pub fn validate_receipt(&mut self, receipt: &PoiReceipt) -> Result<()> {
        // 1. Check for duplicate output hashes (Anti-cheat)
        if let Some(existing_node) = self.receipt_cache.get(&receipt.output_hash) {
            if existing_node != &receipt.node_uid.to_string() {
                bail!(
                    "DUPLICATE_OUTPUT_DETECTED: Hash {} already submitted by node {}",
                    receipt.output_hash,
                    existing_node
                );
            }
        }

        // 2. Resource correlation (Placeholder)
        // In a real system, we'd check if the CPU/Time reported aligns with work complexity

        // 3. Commit to cache
        self.receipt_cache
            .insert(receipt.output_hash.clone(), receipt.node_uid.to_string());

        Ok(())
    }
}
