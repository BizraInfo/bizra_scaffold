use sha2::{Digest, Sha256};

/// A tiny hash-chain helper.
/// Every event links to the previous digest to make tampering obvious.
#[derive(Clone, Debug)]
pub struct HashChain {
    prev: [u8; 32],
}

impl Default for HashChain {
    fn default() -> Self {
        Self { prev: [0u8; 32] }
    }
}

impl HashChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn step(&mut self, bytes: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.prev);
        hasher.update(bytes);
        let out = hasher.finalize();
        self.prev.copy_from_slice(&out);
        self.prev
    }

    pub fn head(&self) -> [u8; 32] {
        self.prev
    }
}
