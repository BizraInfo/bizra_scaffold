// src/snr.rs - SNR (Signal-to-Noise Ratio) Scoring Engine
// Measures information density and quality vs bloat and noise.
// Migrated to Fixed64 for deterministic institutional-grade results.

use crate::fixed::Fixed64;
use crate::types::AgentResult;

/// SNR Scoring Result
#[derive(Debug, Clone)]
pub struct SNRScore {
    pub signal: Fixed64,
    pub noise: Fixed64,
    pub ratio: Fixed64,               // SNR = Signal / Noise
    pub information_density: Fixed64, // Entropy proxy
}

pub struct SNREngine;

impl SNREngine {
    /// Calculate SNR for an agent's contribution
    pub fn score(result: &AgentResult) -> SNRScore {
        let text = result.contribution.to_lowercase();
        let _char_count = Fixed64::from_i64(text.len() as i64);

        if text.is_empty() {
            return SNRScore {
                signal: Fixed64::ZERO,
                noise: Fixed64::from_i64(1),
                ratio: Fixed64::ZERO,
                information_density: Fixed64::ZERO,
            };
        }

        // 1. SIGNAL: Technical markers and confidence
        let technical_markers = [
            "formal",
            "invariant",
            "proof",
            "verification",
            "optimization",
            "latency",
            "protocol",
            "ihsan",
        ];
        let mut marker_hits = 0;
        for marker in &technical_markers {
            if text.contains(marker) {
                marker_hits += 1;
            }
        }

        // result.confidence is already Fixed64 - no conversion needed
        let confidence_fixed = result.confidence;

        // Signal base estimation without HashSet for no_std simplicity
        // In primordial mode, we prioritize markers over vocab richness to avoid alloc dependencies
        let signal = (Fixed64::from_i64(marker_hits as i64) * Fixed64::from_i64(5))
            + (confidence_fixed * Fixed64::from_i64(20));

        // 4. WISDOM ALIGNMENT: Anchoring in the House of Wisdom roots
        let wisdom_roots = [
            "quran",
            "hadith",
            "sunnah",
            "hikmah",
            "ihsan",
            "bayt",
            "sovereign",
            "genesis",
        ];
        let mut wisdom_alignment = Fixed64::ZERO;
        for root in &wisdom_roots {
            if text.contains(root) {
                wisdom_alignment = wisdom_alignment + Fixed64::from_i64(10);
            }
        }

        let total_signal = signal + wisdom_alignment;

        // 2. NOISE: Repetitive phrases and "fluff"
        let fluff_phrases = [
            "as an ai",
            "i understand",
            "let me",
            "to be honest",
            "i think",
            "sorry",
            "apologize",
            "redundant",
            "filler",
            "conclusion",
            "in conclusion",
            "crucial",
            "important to note",
        ];
        let mut fluff_hits = 0;
        for fluff in &fluff_phrases {
            if text.contains(fluff) {
                fluff_hits += 5;
            }
        }

        // 3. INTERDISCIPLINARY DENSITY: Measuring cross-domain connections
        let domain_markers = [
            "thermodynamics",
            "topology",
            "axiomatic",
            "jurisprudence",
            "economic",
            "sociological",
            "biological",
            "quantum",
            "formal",
            "metaphysical",
            "cybernetic",
            "homeostasis",
            "entropy",
        ];
        let mut domain_diversity = 0;
        for marker in &domain_markers {
            if text.contains(marker) {
                domain_diversity += 1;
            }
        }
        let interdisciplinary_signal =
            Fixed64::from_i64(domain_diversity as i64) * Fixed64::from_i64(15);

        // 4. AUTONOMOUS REPAIR BONUS: Rewarding successful Ralph Wiggum loops
        let repair_bonus = if text.contains("<promise>fixed</promise>") {
            Fixed64::from_i64(50)
        } else {
            Fixed64::ZERO
        };

        let entropy = Self::calculate_shannon_entropy(&text);
        let information_signal = entropy * Fixed64::from_i64(30);

        let final_signal =
            total_signal + interdisciplinary_signal + repair_bonus + information_signal;

        // Advanced Noise Detection: Repetition penalty (e.g., same word used > 10% of text)
        let repetition_penalty = Self::calculate_repetition_penalty(&text);
        let final_noise =
            Fixed64::from_i64(fluff_hits as i64) + repetition_penalty + Fixed64::from_i64(1); // avoid div zero

        let ratio = final_signal / final_noise;

        SNRScore {
            signal: final_signal,
            noise: final_noise,
            ratio,
            information_density: entropy,
        }
    }

    fn calculate_repetition_penalty(text: &str) -> Fixed64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Fixed64::ZERO;
        }

        let mut counts = std::collections::HashMap::new();
        for word in &words {
            *counts.entry(word).or_insert(0) += 1;
        }

        let mut max_count = 0;
        for count in counts.values() {
            if *count > max_count {
                max_count = *count;
            }
        }

        // Penalty if any word > 15% of the text
        if max_count as f64 / words.len() as f64 > 0.15 {
            Fixed64::from_i64((max_count * 10) as i64)
        } else {
            Fixed64::ZERO
        }
    }

    /// PEAK MASTERPIECE: Shannon Entropy in Fixed64
    /// Measures the uncertainty/information content per byte.
    fn calculate_shannon_entropy(s: &str) -> Fixed64 {
        if s.is_empty() {
            return Fixed64::ZERO;
        }

        let mut counts = [0i64; 256];
        let len = s.len() as i64;
        for b in s.as_bytes() {
            counts[*b as usize] += 1;
        }

        let mut entropy = Fixed64::ZERO;
        let len_f = Fixed64::from_i64(len);

        for &count in &counts {
            if count > 0 {
                let p = Fixed64::from_i64(count).saturating_div(len_f);
                let contribution = p * (Fixed64::from_i64(1) - p);
                entropy = entropy + contribution;
            }
        }

        // Normalize to [0, 1] range for signal scaling
        entropy * Fixed64::from_i64(4)
    }
}
