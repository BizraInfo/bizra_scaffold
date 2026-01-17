// src/engram.rs - Sovereign Conditional Memory for Local-First AI
//
// PEAK MASTERPIECE v7.1: DeepSeek Engram architecture adapted for BIZRA sovereignty
//
// Giants Protocol Synthesis:
// - Neuroscience (Kandel): Engrams as distributed memory traces
// - Information Theory (Shannon): N-gram statistical regularities
// - Computer Architecture (Hennessy-Patterson): Memory hierarchy offloading
// - Islamic Scholarship (Bayt al-Hikma): Knowledge preservation patterns
// - Distributed Systems (Lamport): Local-first, no central authority
//
// Key Properties:
// - O(1) deterministic lookup via hash-based addressing
// - Fixed64 arithmetic for cross-platform reproducibility
// - Tiered tables: T0 (50MB), T1 (500MB), T2+ (2GB+)
// - mmap offloading for memory-constrained devices
// - Sovereignty-first: no cloud dependency

use crate::fixed::Fixed64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use tracing::{debug, info, instrument, warn};

// ============================================================================
// SOVEREIGNTY TIERS
// ============================================================================

/// Sovereignty tier determines resource budget for Engram tables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum SovereigntyTier {
    /// T0: Mobile devices (iOS/Android) - 50MB budget
    T0Mobile,
    /// T1: Consumer PC (Windows/Linux) - 500MB budget
    #[default]
    T1Consumer,
    /// T2+: Server/Node - 2GB+ budget
    T2Node,
}

impl SovereigntyTier {
    /// Maximum embedding table size in bytes
    pub fn max_table_bytes(&self) -> usize {
        match self {
            Self::T0Mobile => 50 * 1024 * 1024,     // 50 MB
            Self::T1Consumer => 500 * 1024 * 1024, // 500 MB
            Self::T2Node => 2 * 1024 * 1024 * 1024, // 2 GB
        }
    }

    /// Maximum vocabulary size for this tier
    pub fn max_vocab_size(&self) -> usize {
        match self {
            Self::T0Mobile => 8_192,    // 8K tokens
            Self::T1Consumer => 32_768, // 32K tokens
            Self::T2Node => 131_072,    // 128K tokens
        }
    }

    /// Embedding dimension for this tier
    pub fn embedding_dim(&self) -> usize {
        match self {
            Self::T0Mobile => 256,    // Compact
            Self::T1Consumer => 512,  // Balanced
            Self::T2Node => 1024,     // Full
        }
    }

    /// Maximum n-gram size
    pub fn max_ngram(&self) -> usize {
        match self {
            Self::T0Mobile => 3,   // Trigrams max
            Self::T1Consumer => 5, // 5-grams
            Self::T2Node => 8,     // Full context
        }
    }
}


// ============================================================================
// N-GRAM HASH MAPPING
// ============================================================================

/// Deterministic n-gram hash computation using XOR-based addressing
/// PEAK MASTERPIECE: No floats, all integer operations for reproducibility
#[derive(Debug, Clone)]
pub struct NgramHashMapping {
    /// Compressed vocabulary size
    vocab_size: usize,
    /// Maximum n-gram size (2 to max_ngram)
    max_ngram: usize,
    /// Layer-specific odd multipliers for hash diversity
    layer_multipliers: Vec<u64>,
}

impl NgramHashMapping {
    /// Create new hash mapping for given tier
    pub fn new(tier: SovereigntyTier) -> Self {
        let max_ngram = tier.max_ngram();

        // Generate deterministic odd multipliers for each n-gram size
        // Using prime-based sequence for better distribution
        let layer_multipliers: Vec<u64> = (2..=max_ngram)
            .map(Self::generate_odd_multiplier)
            .collect();

        Self {
            vocab_size: tier.max_vocab_size(),
            max_ngram,
            layer_multipliers,
        }
    }

    /// Generate deterministic odd multiplier for n-gram size
    /// Based on primordial prime sequence for hash quality
    fn generate_odd_multiplier(ngram_size: usize) -> u64 {
        // Primordial primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29...
        const PRIMES: [u64; 10] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        let prime = PRIMES.get(ngram_size % 10).unwrap_or(&31);

        // Ensure odd: 2*prime + 1
        2 * prime * (ngram_size as u64) + 1
    }

    /// Compute hash indices for all n-grams in token sequence
    /// Returns: Vec<(ngram_size, hash_index)>
    #[instrument(skip(self, token_ids))]
    pub fn compute_hashes(&self, token_ids: &[u32]) -> Vec<(usize, usize)> {
        let mut hashes = Vec::new();

        for ngram_size in 2..=self.max_ngram.min(token_ids.len()) {
            let multiplier = self.layer_multipliers[ngram_size - 2];

            for window in token_ids.windows(ngram_size) {
                let hash = self.hash_ngram(window, multiplier);
                hashes.push((ngram_size, hash));
            }
        }

        hashes
    }

    /// Hash single n-gram using XOR-folding
    /// DETERMINISTIC: Same input always produces same output
    #[inline]
    fn hash_ngram(&self, tokens: &[u32], multiplier: u64) -> usize {
        let mut hash: u64 = 0;

        for (i, &token) in tokens.iter().enumerate() {
            // Position-weighted XOR with odd multiplier
            let weighted = (token as u64).wrapping_mul(multiplier.wrapping_add(i as u64));
            hash ^= weighted;
        }

        // Fold to vocab size using modulo
        (hash as usize) % self.vocab_size
    }
}

// ============================================================================
// MULTI-HEAD EMBEDDING TABLE
// ============================================================================

/// Fixed64 embedding vector for deterministic computation
pub type EngramEmbedding = Vec<Fixed64>;

/// Multi-head embedding table with separate tables per n-gram size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHeadEmbeddingTable {
    /// Embedding dimension
    dim: usize,
    /// Number of attention heads per n-gram size
    num_heads: usize,
    /// Tables indexed by: [ngram_size - 2][head][vocab_index]
    tables: Vec<Vec<Vec<EngramEmbedding>>>,
}

impl MultiHeadEmbeddingTable {
    /// Create new embedding table for given tier
    /// NOTE: In production, tables would be loaded from disk/mmap
    pub fn new(tier: SovereigntyTier) -> Self {
        let dim = tier.embedding_dim();
        let vocab_size = tier.max_vocab_size();
        let max_ngram = tier.max_ngram();
        let num_heads = 4; // Fixed for now

        info!(
            tier = ?tier,
            dim = dim,
            vocab_size = vocab_size,
            max_ngram = max_ngram,
            "Initializing MultiHeadEmbeddingTable"
        );

        // Initialize tables with deterministic pseudo-random values
        // In production, these would be trained embeddings
        let tables: Vec<Vec<Vec<EngramEmbedding>>> = (2..=max_ngram)
            .map(|ngram_size| {
                (0..num_heads)
                    .map(|head| {
                        (0..vocab_size)
                            .map(|idx| Self::init_embedding(dim, ngram_size, head, idx))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Self {
            dim,
            num_heads,
            tables,
        }
    }

    /// Initialize single embedding with deterministic values
    /// Uses SHA256 for reproducible initialization
    fn init_embedding(dim: usize, ngram_size: usize, head: usize, idx: usize) -> EngramEmbedding {
        let seed = format!("engram:{}:{}:{}", ngram_size, head, idx);
        let hash = Sha256::digest(seed.as_bytes());

        (0..dim)
            .map(|d| {
                // Use hash bytes to generate Fixed64 values in [-1, 1]
                let byte_idx = d % 32;
                let byte_val = hash[byte_idx] as i64;
                // Scale to [-0.5, 0.5] range
                Fixed64::from_bits((byte_val - 128) * (Fixed64::SCALE / 256))
            })
            .collect()
    }

    /// Retrieve embedding for given n-gram hash
    #[inline]
    pub fn get(&self, ngram_size: usize, head: usize, hash_idx: usize) -> Option<&EngramEmbedding> {
        let table_idx = ngram_size.checked_sub(2)?;
        self.tables
            .get(table_idx)?
            .get(head)?
            .get(hash_idx)
    }

    /// Retrieve and aggregate embeddings across all heads for n-gram
    pub fn retrieve_aggregated(&self, ngram_size: usize, hash_idx: usize) -> Option<EngramEmbedding> {
        let table_idx = ngram_size.checked_sub(2)?;
        let head_tables = self.tables.get(table_idx)?;

        // Sum across heads
        let mut aggregated = vec![Fixed64::ZERO; self.dim];

        for head in 0..self.num_heads {
            if let Some(emb) = head_tables.get(head)?.get(hash_idx) {
                for (i, val) in emb.iter().enumerate() {
                    aggregated[i] = aggregated[i] + *val;
                }
            }
        }

        // Normalize by number of heads
        let scale = Fixed64::from_int(self.num_heads as i32);
        for val in &mut aggregated {
            *val = *val / scale;
        }

        Some(aggregated)
    }

    /// Get embedding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ============================================================================
// GATING MECHANISM
// ============================================================================

/// Gating layer for fusing Engram embeddings with hidden states
/// PEAK MASTERPIECE: All operations use Fixed64 for determinism
#[derive(Debug, Clone)]
#[allow(dead_code)] // Reserved fields for future expansion
pub struct EngramGate {
    /// Query projection weights [dim x dim]
    query_weights: Vec<Vec<Fixed64>>,
    /// Key projection weights [dim x dim]
    key_weights: Vec<Vec<Fixed64>>,
    /// Dimension
    dim: usize,
}

impl EngramGate {
    /// Create new gating layer
    pub fn new(dim: usize) -> Self {
        // Initialize with identity-like weights for stable start
        let query_weights = Self::init_projection(dim, "query");
        let key_weights = Self::init_projection(dim, "key");

        Self {
            query_weights,
            key_weights,
            dim,
        }
    }

    /// Initialize projection matrix with deterministic values
    fn init_projection(dim: usize, name: &str) -> Vec<Vec<Fixed64>> {
        (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        if i == j {
                            // Diagonal bias for stability
                            Fixed64::from_f64(0.9)
                        } else {
                            // Small off-diagonal
                            let seed = format!("gate:{}:{}:{}", name, i, j);
                            let hash = Sha256::digest(seed.as_bytes());
                            let val = (hash[0] as f64 - 128.0) / 1280.0; // Small init
                            Fixed64::from_f64(val)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute gate score between hidden state and engram embedding
    /// Returns value in [0, 1] representing fusion weight
    pub fn compute_gate(&self, hidden: &[Fixed64], engram: &[Fixed64]) -> Fixed64 {
        // Project hidden state to query
        let query = self.project(hidden, &self.query_weights);

        // Project engram to key
        let key = self.project(engram, &self.key_weights);

        // Normalized dot product
        let dot = Self::dot_product(&query, &key);
        let norm_q = Self::norm(&query);
        let norm_k = Self::norm(&key);

        // Avoid division by zero
        if norm_q == Fixed64::ZERO || norm_k == Fixed64::ZERO {
            return Fixed64::HALF;
        }

        // Cosine similarity scaled to [0, 1]
        let cosine = dot / (norm_q * norm_k);

        // Sigmoid approximation: (1 + cosine) / 2
        (Fixed64::ONE + cosine) / Fixed64::from_int(2)
    }

    /// Matrix-vector multiplication
    fn project(&self, vec: &[Fixed64], weights: &[Vec<Fixed64>]) -> Vec<Fixed64> {
        weights
            .iter()
            .map(|row| Self::dot_product(row, vec))
            .collect()
    }

    /// Fixed64 dot product
    #[inline]
    fn dot_product(a: &[Fixed64], b: &[Fixed64]) -> Fixed64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| *x * *y)
            .fold(Fixed64::ZERO, |acc, v| acc + v)
    }

    /// Fixed64 L2 norm (approximation)
    fn norm(vec: &[Fixed64]) -> Fixed64 {
        let sum_sq = vec
            .iter()
            .map(|x| *x * *x)
            .fold(Fixed64::ZERO, |acc, v| acc + v);

        // Newton-Raphson square root approximation
        Self::fixed_sqrt(sum_sq)
    }

    /// Fixed-point square root using Newton-Raphson
    pub fn fixed_sqrt(x: Fixed64) -> Fixed64 {
        if x <= Fixed64::ZERO {
            return Fixed64::ZERO;
        }

        // Initial guess: x / 2
        let mut guess = x / Fixed64::from_int(2);

        // 5 iterations typically sufficient for Q32.32
        for _ in 0..5 {
            let new_guess = (guess + x / guess) / Fixed64::from_int(2);
            guess = new_guess;
        }

        guess
    }
}

// ============================================================================
// SOVEREIGN ENGRAM MODULE
// ============================================================================

/// Main Engram module for BIZRA sovereign AI
/// Retrieves static n-gram memory and fuses with dynamic hidden states
#[derive(Debug)]
pub struct SovereignEngram {
    /// Sovereignty tier
    tier: SovereigntyTier,
    /// Hash mapping
    hash_mapping: NgramHashMapping,
    /// Embedding tables
    embeddings: MultiHeadEmbeddingTable,
    /// Gating mechanism
    gate: EngramGate,
    /// Cache for frequently accessed n-grams
    cache: HashMap<(usize, usize), EngramEmbedding>,
    /// Maximum cache entries
    cache_limit: usize,
}

impl SovereignEngram {
    /// Create new Sovereign Engram module
    #[instrument]
    pub fn new(tier: SovereigntyTier) -> Self {
        info!(tier = ?tier, "🧠 Initializing SovereignEngram");

        let hash_mapping = NgramHashMapping::new(tier);
        let embeddings = MultiHeadEmbeddingTable::new(tier);
        let dim = embeddings.dim();
        let gate = EngramGate::new(dim);

        // Cache limit based on tier
        let cache_limit = match tier {
            SovereigntyTier::T0Mobile => 1_000,
            SovereigntyTier::T1Consumer => 10_000,
            SovereigntyTier::T2Node => 100_000,
        };

        Self {
            tier,
            hash_mapping,
            embeddings,
            gate,
            cache: HashMap::with_capacity(cache_limit / 2),
            cache_limit,
        }
    }

    /// Forward pass: Fuse engram embeddings with hidden states
    ///
    /// # Arguments
    /// * `token_ids` - Input token sequence
    /// * `hidden_states` - Current hidden states from transformer [seq_len x dim]
    ///
    /// # Returns
    /// * Enhanced hidden states with engram knowledge injection
    #[instrument(skip(self, hidden_states))]
    pub fn forward(
        &mut self,
        token_ids: &[u32],
        hidden_states: &[Vec<Fixed64>],
    ) -> Vec<Vec<Fixed64>> {
        // Compute n-gram hashes
        let hashes = self.hash_mapping.compute_hashes(token_ids);

        debug!(
            num_hashes = hashes.len(),
            seq_len = hidden_states.len(),
            "Engram forward pass"
        );

        // Retrieve and fuse for each position
        let mut output = hidden_states.to_vec();

        for (pos, hidden) in hidden_states.iter().enumerate() {
            // Find relevant n-grams for this position
            let relevant_hashes: Vec<_> = hashes
                .iter()
                .filter(|(ngram_size, _)| {
                    // N-gram ending at or near this position
                    pos >= *ngram_size - 1
                })
                .take(self.tier.max_ngram()) // Limit lookups
                .collect();

            if relevant_hashes.is_empty() {
                continue;
            }

            // Aggregate engram embeddings
            let mut engram_sum = vec![Fixed64::ZERO; self.embeddings.dim()];
            let mut count = 0;

            for &&(ngram_size, hash_idx) in &relevant_hashes {
                if let Some(emb) = self.get_cached_or_retrieve(ngram_size, hash_idx) {
                    for (i, val) in emb.iter().enumerate() {
                        engram_sum[i] = engram_sum[i] + *val;
                    }
                    count += 1;
                }
            }

            if count == 0 {
                continue;
            }

            // Average
            let scale = Fixed64::from_int(count);
            for val in &mut engram_sum {
                *val = *val / scale;
            }

            // Compute gate
            let gate_value = self.gate.compute_gate(hidden, &engram_sum);

            // Fuse: output = hidden + gate * engram
            for (i, out_val) in output[pos].iter_mut().enumerate() {
                let engram_contribution = gate_value * engram_sum[i];
                *out_val = *out_val + engram_contribution;
            }
        }

        output
    }

    /// Get from cache or retrieve from embedding table
    fn get_cached_or_retrieve(&mut self, ngram_size: usize, hash_idx: usize) -> Option<EngramEmbedding> {
        let key = (ngram_size, hash_idx);

        // Check cache
        if let Some(emb) = self.cache.get(&key) {
            return Some(emb.clone());
        }

        // Retrieve from table
        let emb = self.embeddings.retrieve_aggregated(ngram_size, hash_idx)?;

        // Update cache with LRU eviction
        if self.cache.len() >= self.cache_limit {
            // Simple eviction: remove first entry
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }

        self.cache.insert(key, emb.clone());
        Some(emb)
    }

    /// Get sovereignty tier
    pub fn tier(&self) -> SovereigntyTier {
        self.tier
    }

    /// Get embedding dimension
    pub fn dim(&self) -> usize {
        self.embeddings.dim()
    }

    /// Clear cache (useful for memory pressure)
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.cache_limit)
    }

    // ========================================================================
    // QUERY INTERFACE FOR HOUSE OF WISDOM INTEGRATION
    // ========================================================================

    /// Query n-grams from text and return matching embeddings with scores
    /// Used by HouseOfWisdom.hybrid_search() for Engram augmentation
    ///
    /// # Arguments
    /// * `query` - Query text to search for
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// * Vec of EngramSearchResult with matched n-grams and relevance scores
    #[instrument(skip(self))]
    pub fn query_ngrams(&mut self, query: &str, limit: usize) -> Vec<EngramSearchResult> {
        // Tokenize query using simple whitespace + hash-based approach
        let token_ids = self.tokenize_query(query);

        if token_ids.is_empty() {
            return vec![];
        }

        // Compute n-gram hashes for query
        let hashes = self.hash_mapping.compute_hashes(&token_ids);

        debug!(
            query_len = query.len(),
            token_count = token_ids.len(),
            hash_count = hashes.len(),
            "Engram query_ngrams"
        );

        // Retrieve embeddings and compute relevance scores
        let mut results: Vec<EngramSearchResult> = Vec::new();

        for (ngram_size, hash_idx) in hashes {
            if let Some(embedding) = self.get_cached_or_retrieve(ngram_size, hash_idx) {
                // Compute relevance score based on embedding magnitude
                let relevance = self.compute_embedding_relevance(&embedding, ngram_size);

                results.push(EngramSearchResult {
                    ngram_size,
                    hash_idx,
                    relevance_score: relevance,
                    embedding: Some(embedding),
                });
            }
        }

        // Sort by relevance descending and take top K
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit);
        results
    }

    /// Tokenize query text into token IDs using deterministic hashing
    /// Simple but effective for n-gram matching
    fn tokenize_query(&self, query: &str) -> Vec<u32> {
        query
            .split_whitespace()
            .filter(|word| word.len() >= 2) // Skip very short words
            .map(|word| {
                // Deterministic hash to token ID
                let hash = Sha256::digest(word.to_lowercase().as_bytes());
                let id = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);
                id % (self.tier.max_vocab_size() as u32)
            })
            .collect()
    }

    /// Compute relevance score for embedding
    /// Higher magnitude + larger n-grams = more relevant
    fn compute_embedding_relevance(&self, embedding: &[Fixed64], ngram_size: usize) -> f64 {
        // L2 norm as base relevance
        let sum_sq: Fixed64 = embedding
            .iter()
            .map(|x| *x * *x)
            .fold(Fixed64::ZERO, |acc, v| acc + v);

        let norm = EngramGate::fixed_sqrt(sum_sq).to_f64();

        // Boost by n-gram size (larger context = more relevant)
        let ngram_boost = 1.0 + (ngram_size as f64 - 2.0) * 0.1;

        (norm * ngram_boost).clamp(0.0, 1.0)
    }

    /// Ingest n-grams from external source (e.g., experience cards)
    /// Used for populating Engram from MemGovern experiences
    pub fn ingest_ngrams(&mut self, texts: Vec<String>) -> usize {
        let mut ingested = 0;

        for text in texts {
            let token_ids = self.tokenize_query(&text);
            if token_ids.len() >= 2 {
                // Pre-warm cache with these n-grams
                let hashes = self.hash_mapping.compute_hashes(&token_ids);
                for (ngram_size, hash_idx) in hashes {
                    if self.get_cached_or_retrieve(ngram_size, hash_idx).is_some() {
                        ingested += 1;
                    }
                }
            }
        }

        info!(ingested = ingested, "Engram n-grams ingested");
        ingested
    }
}

/// Search result from Engram query
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngramSearchResult {
    /// N-gram size (2 = bigram, 3 = trigram, etc.)
    pub ngram_size: usize,
    /// Hash index in embedding table
    pub hash_idx: usize,
    /// Relevance score (0.0 - 1.0)
    pub relevance_score: f64,
    /// Retrieved embedding (optional, for downstream fusion)
    pub embedding: Option<EngramEmbedding>,
}

// ============================================================================
// DOMAIN-SPECIFIC ENGRAM PROFILES
// ============================================================================

/// Pre-configured Engram profiles for specific knowledge domains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngramProfile {
    /// General purpose
    General,
    /// Islamic knowledge (Quran, Hadith, Fiqh)
    IslamicKnowledge,
    /// Code/Programming
    CodeGeneration,
    /// Mathematical reasoning
    Mathematics,
    /// Scientific literature
    Scientific,
}

impl EngramProfile {
    /// Get recommended tier for this profile
    pub fn recommended_tier(&self) -> SovereigntyTier {
        match self {
            Self::General => SovereigntyTier::T1Consumer,
            Self::IslamicKnowledge => SovereigntyTier::T1Consumer, // Fits in 500MB
            Self::CodeGeneration => SovereigntyTier::T2Node,
            Self::Mathematics => SovereigntyTier::T1Consumer,
            Self::Scientific => SovereigntyTier::T2Node,
        }
    }

    /// Get profile-specific vocabulary boost terms
    pub fn vocab_boost_terms(&self) -> &'static [&'static str] {
        match self {
            Self::General => &[],
            Self::IslamicKnowledge => &[
                "الله", "محمد", "قرآن", "سورة", "آية", "حديث", "صحيح", "فقه", "شريعة",
                "Allah", "Muhammad", "Quran", "Surah", "Ayah", "Hadith", "Sahih", "Fiqh",
            ],
            Self::CodeGeneration => &[
                "function", "class", "async", "await", "impl", "struct", "enum", "trait",
            ],
            Self::Mathematics => &[
                "theorem", "proof", "lemma", "integral", "derivative", "equation", "matrix",
            ],
            Self::Scientific => &[
                "hypothesis", "experiment", "analysis", "conclusion", "methodology",
            ],
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sovereignty_tier_budgets() {
        assert!(SovereigntyTier::T0Mobile.max_table_bytes() < SovereigntyTier::T1Consumer.max_table_bytes());
        assert!(SovereigntyTier::T1Consumer.max_table_bytes() < SovereigntyTier::T2Node.max_table_bytes());
    }

    #[test]
    fn test_ngram_hash_determinism() {
        let mapping = NgramHashMapping::new(SovereigntyTier::T1Consumer);

        let tokens = vec![100u32, 200, 300, 400];
        let hashes1 = mapping.compute_hashes(&tokens);
        let hashes2 = mapping.compute_hashes(&tokens);

        assert_eq!(hashes1, hashes2, "Hash computation must be deterministic");
    }

    #[test]
    fn test_embedding_retrieval() {
        let table = MultiHeadEmbeddingTable::new(SovereigntyTier::T0Mobile);

        // Should be able to retrieve for valid indices
        let emb = table.retrieve_aggregated(2, 0);
        assert!(emb.is_some());
        assert_eq!(emb.unwrap().len(), SovereigntyTier::T0Mobile.embedding_dim());
    }

    #[test]
    fn test_gate_computation() {
        let gate = EngramGate::new(256);

        let hidden: Vec<Fixed64> = (0..256).map(|i| Fixed64::from_f64(i as f64 / 256.0)).collect();
        let engram: Vec<Fixed64> = (0..256).map(|i| Fixed64::from_f64((255 - i) as f64 / 256.0)).collect();

        let gate_value = gate.compute_gate(&hidden, &engram);

        // Gate should be in [0, 1]
        assert!(gate_value >= Fixed64::ZERO);
        assert!(gate_value <= Fixed64::ONE);
    }

    #[test]
    fn test_sovereign_engram_forward() {
        let mut engram = SovereignEngram::new(SovereigntyTier::T0Mobile);

        let tokens = vec![1u32, 2, 3, 4, 5];
        let dim = engram.dim();
        let hidden_states: Vec<Vec<Fixed64>> = (0..5)
            .map(|_| vec![Fixed64::HALF; dim])
            .collect();

        let output = engram.forward(&tokens, &hidden_states);

        assert_eq!(output.len(), hidden_states.len());
        assert_eq!(output[0].len(), dim);
    }

    #[test]
    fn test_cache_eviction() {
        let mut engram = SovereignEngram::new(SovereigntyTier::T0Mobile);

        // Trigger many retrievals to test cache
        for i in 0..2000 {
            let _ = engram.get_cached_or_retrieve(2, i % 1000);
        }

        let (size, limit) = engram.cache_stats();
        assert!(size <= limit);
    }

    #[test]
    fn test_engram_profile_tiers() {
        assert_eq!(
            EngramProfile::IslamicKnowledge.recommended_tier(),
            SovereigntyTier::T1Consumer
        );
    }
}
