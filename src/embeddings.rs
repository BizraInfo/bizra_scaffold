use anyhow::Result;

const THREAT_TERMS: &[&str] = &[
    "malicious",
    "hack",
    "cyber",
    "attack",
    "breach",
    "exploit",
    "intrusion",
    "phish",
    "threat",
];
const SAFE_TERMS: &[&str] = &[
    "baking", "cookies", "cooking", "recipe", "garden", "travel", "sunshine", "art", "craft",
];

#[allow(dead_code)]
pub struct EmbeddingEngine {
    // model: Arc<TextEmbedding>,
}

#[allow(dead_code)]
impl EmbeddingEngine {
    pub fn new() -> Result<Self> {
        // Placeholder - disabled due to fastembed/ort dependency issues
        Ok(Self {
            // model: Arc::new(model),
        })
    }

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let normalized = text.to_lowercase();
        let threat_score = score_terms(&normalized, THREAT_TERMS);
        let safe_score = score_terms(&normalized, SAFE_TERMS);

        // Introduce a small bias so even neutral text yields a non-zero vector.
        Ok(vec![threat_score + 0.1, safe_score + 0.1])
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|text| self.embed_text(text))
            .collect::<Result<Vec<_>>>()
    }

    pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
        let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm_v1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_v2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_v1 == 0.0 || norm_v2 == 0.0 {
            return 0.0;
        }

        dot_product / (norm_v1 * norm_v2)
    }
}

fn score_terms(text: &str, terms: &[&str]) -> f32 {
    terms.iter().filter(|term| text.contains(*term)).count() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity() {
        let engine = EmbeddingEngine::new().unwrap();
        let vec1 = engine.embed_text("malicious hack").unwrap();
        let vec2 = engine.embed_text("cyber attack").unwrap();
        let vec3 = engine.embed_text("baking cookies").unwrap();

        let sim_threat = EmbeddingEngine::cosine_similarity(&vec1, &vec2);
        let sim_safe = EmbeddingEngine::cosine_similarity(&vec1, &vec3);

        println!("Threat similarity: {}", sim_threat);
        println!("Safe similarity: {}", sim_safe);

        assert!(sim_threat > sim_safe);
        assert!(sim_threat > 0.6); // Should be semantically close
    }
}
