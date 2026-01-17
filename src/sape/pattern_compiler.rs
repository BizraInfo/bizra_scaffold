// src/sape/pattern_compiler.rs
// Status: OPTIMIZATION_PIPELINE_V1
// SAPE Pattern Compilation for recurring symbol chains

use crate::tpm::SignerProvider;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Optimization level for pattern compilation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[derive(Default)]
pub enum OptimizationLevel {
    Minimum,
    #[default]
    Balanced,
    Aggressive,
    Maximum,
}


/// A compiled pattern representing a recurring symbol chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub id: String,
    pub symbol_chain: Vec<String>,
    pub frequency: usize,
    pub complexity: f64,
    pub optimization_level: OptimizationLevel,
    pub compiled_wasm: Option<Vec<u8>>,
    pub signature: Option<Vec<u8>>,
    pub compilation_receipt_id: Option<String>,
    pub last_used: chrono::DateTime<chrono::Utc>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Pattern {
    pub fn new(symbol_chain: Vec<String>) -> Self {
        let id = Self::generate_id(&symbol_chain);
        Pattern {
            id,
            symbol_chain,
            frequency: 1,
            complexity: 0.0,
            optimization_level: OptimizationLevel::default(),
            compiled_wasm: None,
            signature: None,
            compilation_receipt_id: None,
            last_used: chrono::Utc::now(),
            created_at: chrono::Utc::now(),
        }
    }

    fn generate_id(chain: &[String]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for symbol in chain {
            hasher.update(symbol.as_bytes());
            hasher.update(b"|");
        }
        let result = hasher.finalize();
        format!("pat_{}", hex::encode(&result[..8]))
    }

    /// Calculate optimization score for prioritization
    pub fn optimization_score(&self) -> f64 {
        let freq_score = (self.frequency as f64).log2().max(0.0);
        let complexity_score = self.complexity;
        let recency_score = {
            let age = chrono::Utc::now()
                .signed_duration_since(self.last_used)
                .num_hours() as f64;
            1.0 / (1.0 + age / 24.0) // Decay over days
        };
        (freq_score * 0.4) + (complexity_score * 0.4) + (recency_score * 0.2)
    }

    /// Check if pattern is compiled and signed
    pub fn is_compiled(&self) -> bool {
        self.compiled_wasm.is_some() && self.signature.is_some()
    }
}

/// Result of applying optimization to a thought
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub patterns_applied: usize,
    pub symbols_optimized: usize,
    pub total_symbols: usize,
    pub estimated_speedup: f64,
    pub coverage: f64,
}

impl OptimizationResult {
    pub fn none() -> Self {
        OptimizationResult {
            patterns_applied: 0,
            symbols_optimized: 0,
            total_symbols: 0,
            estimated_speedup: 1.0,
            coverage: 0.0,
        }
    }
}

/// Performance metrics for a compiled pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMetrics {
    pub compilation_time_ms: u64,
    pub estimated_speedup: f64,
    pub wasm_size_bytes: usize,
    pub applications: usize,
}

/// The Pattern Compiler - analyzes traces and compiles recurring patterns
pub struct PatternCompiler {
    patterns: RwLock<HashMap<String, Pattern>>,
    signer: Arc<dyn SignerProvider>,
    min_frequency: usize,
    min_complexity: f64,
    max_patterns: usize,
}

impl PatternCompiler {
    pub fn new(signer: Arc<dyn SignerProvider>) -> Self {
        PatternCompiler {
            patterns: RwLock::new(HashMap::new()),
            signer,
            min_frequency: 3,    // Minimum occurrences before compilation
            min_complexity: 0.5, // Minimum complexity threshold
            max_patterns: 1000,  // Maximum patterns to cache
        }
    }

    /// Initialize with existing patterns (from storage)
    pub async fn init_with_patterns(&self, patterns: Vec<Pattern>) -> Result<(), PatternError> {
        let mut lock = self.patterns.write().await;
        for pattern in patterns {
            if pattern.is_compiled() {
                // Verify signature before accepting
                if self.verify_pattern_signature(&pattern).await? {
                    lock.insert(pattern.id.clone(), pattern);
                } else {
                    warn!("Pattern {} has invalid signature, skipping", pattern.id);
                }
            }
        }
        info!("Pattern compiler initialized with {} patterns", lock.len());
        Ok(())
    }

    /// Analyze a symbol trace and identify recurring patterns
    pub async fn analyze_trace(&self, symbols: &[String]) -> Vec<Pattern> {
        let mut pattern_candidates: HashMap<Vec<String>, usize> = HashMap::new();

        // Extract sliding windows of various lengths (2-5 symbols)
        for window_size in 2..=5 {
            for window in symbols.windows(window_size) {
                let chain: Vec<String> = window.to_vec();
                *pattern_candidates.entry(chain).or_insert(0) += 1;
            }
        }

        // Filter and score candidates
        let mut patterns: Vec<Pattern> = pattern_candidates
            .into_iter()
            .filter(|(_, freq)| *freq >= self.min_frequency)
            .map(|(chain, freq)| {
                let mut pattern = Pattern::new(chain);
                pattern.frequency = freq;
                pattern.complexity = self.calculate_complexity(&pattern.symbol_chain);
                pattern.optimization_level =
                    self.determine_optimization_level(freq, pattern.complexity);
                pattern
            })
            .filter(|p| p.complexity >= self.min_complexity)
            .collect();

        // Sort by optimization potential
        patterns.sort_by(|a, b| {
            b.optimization_score()
                .partial_cmp(&a.optimization_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        patterns
    }

    /// Compile a pattern into optimized WASM
    pub async fn compile_pattern(&self, mut pattern: Pattern) -> Result<Pattern, PatternError> {
        let start = Instant::now();
        info!(
            "Compiling pattern {} (freq={}, complexity={:.2})",
            pattern.id, pattern.frequency, pattern.complexity
        );

        // Step 1: Generate WASM bytecode from symbol chain
        let wasm_bytes = self.generate_wasm(&pattern.symbol_chain, pattern.optimization_level)?;

        // Step 2: Sign the WASM module
        let signature = self
            .signer
            .sign(&wasm_bytes)
            .await
            .map_err(|e| PatternError::SigningFailed(e.to_string()))?;

        // Step 3: Update pattern
        pattern.compiled_wasm = Some(wasm_bytes.clone());
        pattern.signature = Some(signature);

        let elapsed = start.elapsed();
        info!(
            "✅ Pattern {} compiled: {} bytes in {:?}",
            pattern.id,
            wasm_bytes.len(),
            elapsed
        );

        // Store in cache
        {
            let mut lock = self.patterns.write().await;

            // Evict old patterns if at capacity
            if lock.len() >= self.max_patterns {
                self.evict_least_used(&mut lock).await;
            }

            lock.insert(pattern.id.clone(), pattern.clone());
        }

        Ok(pattern)
    }

    /// Get a compiled pattern by ID
    pub async fn get_pattern(&self, id: &str) -> Option<Pattern> {
        let lock = self.patterns.read().await;
        lock.get(id).cloned()
    }

    /// Get all compiled patterns
    pub async fn get_all_patterns(&self) -> Vec<Pattern> {
        let lock = self.patterns.read().await;
        lock.values().cloned().collect()
    }

    /// Check if a pattern should be compiled
    pub async fn should_compile(&self, pattern: &Pattern) -> bool {
        if pattern.is_compiled() {
            return false;
        }
        pattern.frequency >= self.min_frequency && pattern.complexity >= self.min_complexity
    }

    /// Generate WASM bytecode for a symbol chain
    fn generate_wasm(
        &self,
        symbols: &[String],
        opt_level: OptimizationLevel,
    ) -> Result<Vec<u8>, PatternError> {
        // Simplified WASM generation - in production this would use wasmtime/cranelift
        // For now, generate a compact representation that can be interpreted

        let mut wasm = Vec::new();

        // WASM magic number and version
        wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6D]); // \0asm
        wasm.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // version 1

        // Encode symbol chain as custom section
        let custom_section = self.encode_symbol_chain(symbols, opt_level);
        wasm.extend_from_slice(&custom_section);

        // Add minimal type section
        wasm.extend_from_slice(&[0x01, 0x04, 0x01, 0x60, 0x00, 0x00]); // empty func type

        // Add minimal function section
        wasm.extend_from_slice(&[0x03, 0x02, 0x01, 0x00]); // one function of type 0

        // Add minimal code section
        wasm.extend_from_slice(&[0x0A, 0x04, 0x01, 0x02, 0x00, 0x0B]); // empty function body

        Ok(wasm)
    }

    /// Encode symbol chain into WASM custom section
    fn encode_symbol_chain(&self, symbols: &[String], opt_level: OptimizationLevel) -> Vec<u8> {
        let mut section = Vec::new();

        // Custom section ID (0)
        section.push(0x00);

        // Section name: "bizra_pattern"
        let name = b"bizra_pattern";
        section.push(name.len() as u8);
        section.extend_from_slice(name);

        // Optimization level
        section.push(match opt_level {
            OptimizationLevel::Minimum => 0,
            OptimizationLevel::Balanced => 1,
            OptimizationLevel::Aggressive => 2,
            OptimizationLevel::Maximum => 3,
        });

        // Symbol count
        section.push(symbols.len() as u8);

        // Encode each symbol
        for symbol in symbols {
            let bytes = symbol.as_bytes();
            section.push(bytes.len() as u8);
            section.extend_from_slice(bytes);
        }

        // Prepend section size
        let size = section.len();
        let mut result = vec![size as u8];
        result.extend(section);

        result
    }

    /// Calculate complexity of a symbol chain
    fn calculate_complexity(&self, symbols: &[String]) -> f64 {
        if symbols.is_empty() {
            return 0.0;
        }

        // Factor 1: Length (longer = more complex, up to 10)
        let length_factor = (symbols.len() as f64).min(10.0) / 10.0;

        // Factor 2: Symbol diversity
        let unique: HashSet<_> = symbols.iter().collect();
        let diversity_factor = unique.len() as f64 / symbols.len() as f64;

        // Factor 3: Symbol complexity (length of individual symbols)
        let avg_symbol_len: f64 =
            symbols.iter().map(|s| s.len() as f64).sum::<f64>() / symbols.len() as f64;
        let symbol_complexity = (avg_symbol_len / 20.0).min(1.0);

        // Weighted combination
        (length_factor * 0.3) + (diversity_factor * 0.4) + (symbol_complexity * 0.3)
    }

    /// Determine optimization level based on frequency and complexity
    fn determine_optimization_level(&self, frequency: usize, complexity: f64) -> OptimizationLevel {
        let score = (frequency as f64).log2() * complexity;

        match score {
            s if s > 8.0 => OptimizationLevel::Maximum,
            s if s > 4.0 => OptimizationLevel::Aggressive,
            s if s > 2.0 => OptimizationLevel::Balanced,
            _ => OptimizationLevel::Minimum,
        }
    }

    /// Verify pattern signature
    async fn verify_pattern_signature(&self, pattern: &Pattern) -> Result<bool, PatternError> {
        match (&pattern.compiled_wasm, &pattern.signature) {
            (Some(wasm), Some(sig)) => Ok(self.signer.verify(wasm, sig)),
            _ => Ok(false),
        }
    }

    /// Evict least recently used patterns
    async fn evict_least_used(&self, patterns: &mut HashMap<String, Pattern>) {
        if patterns.len() < self.max_patterns {
            return;
        }

        // Find least recently used pattern
        if let Some((id, _)) = patterns
            .iter()
            .min_by_key(|(_, p)| p.last_used)
            .map(|(id, p)| (id.clone(), p.clone()))
        {
            patterns.remove(&id);
            debug!("Evicted pattern {} from cache", id);
        }
    }

    /// Get pattern statistics
    pub async fn get_stats(&self) -> PatternStats {
        let lock = self.patterns.read().await;

        let total = lock.len();
        let compiled = lock.values().filter(|p| p.is_compiled()).count();
        let total_frequency: usize = lock.values().map(|p| p.frequency).sum();
        let avg_complexity: f64 = if total > 0 {
            lock.values().map(|p| p.complexity).sum::<f64>() / total as f64
        } else {
            0.0
        };

        PatternStats {
            total_patterns: total,
            compiled_patterns: compiled,
            total_frequency,
            average_complexity: avg_complexity,
        }
    }
}

/// Pattern compilation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStats {
    pub total_patterns: usize,
    pub compiled_patterns: usize,
    pub total_frequency: usize,
    pub average_complexity: f64,
}

/// Pattern-related errors
#[derive(Debug, thiserror::Error)]
pub enum PatternError {
    #[error("Pattern compilation failed: {0}")]
    CompilationFailed(String),

    #[error("Signing failed: {0}")]
    SigningFailed(String),

    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    #[error("Storage error: {0}")]
    StorageError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tpm::SoftwareSigner;

    #[tokio::test]
    async fn test_pattern_analysis() {
        let signer = Arc::new(SoftwareSigner::new());
        let compiler = PatternCompiler::new(signer);

        let symbols = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
            "E".to_string(),
        ];

        let patterns = compiler.analyze_trace(&symbols).await;

        assert!(!patterns.is_empty());
        // "A", "B", "C" pattern should be found with frequency 3
        let abc_pattern = patterns
            .iter()
            .find(|p| p.symbol_chain == vec!["A", "B", "C"]);
        assert!(abc_pattern.is_some());
        assert_eq!(abc_pattern.unwrap().frequency, 3);
    }

    #[tokio::test]
    async fn test_pattern_compilation() {
        let signer = Arc::new(SoftwareSigner::new());
        let compiler = PatternCompiler::new(signer);

        let pattern = Pattern::new(vec!["TEST".to_string(), "PATTERN".to_string()]);
        let compiled = compiler.compile_pattern(pattern).await.unwrap();

        assert!(compiled.is_compiled());
        assert!(compiled.compiled_wasm.is_some());
        assert!(compiled.signature.is_some());
    }

    #[test]
    fn test_complexity_calculation() {
        let signer = Arc::new(SoftwareSigner::new());
        let compiler = PatternCompiler::new(signer);

        // Simple pattern
        let simple = vec!["A".to_string(), "A".to_string()];
        let simple_complexity = compiler.calculate_complexity(&simple);

        // Complex pattern
        let complex = vec![
            "ANALYZE".to_string(),
            "TRANSFORM".to_string(),
            "VALIDATE".to_string(),
            "EXECUTE".to_string(),
        ];
        let complex_complexity = compiler.calculate_complexity(&complex);

        assert!(complex_complexity > simple_complexity);
    }
}
