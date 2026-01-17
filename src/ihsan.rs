use crate::fixed::Fixed64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, sync::OnceLock};

const IHSAN_CONSTITUTION_PATH: &str = "constitution/ihsan_v1.yaml";
const GENESIS_MANIFEST_PATH: &str = "evidence/genesis/GENESIS_MANIFEST.json";
const SEALED_GENESIS_HASH: &str =
    "0b3a9faf1e8c34c1fe5a63a6a70b9bd158eca058b755d188e11ff367602ac7a1";

// BIZRA CORE CONSTANTS (Single Source of Truth)
// Used by genesis_activation.sh to verify FFI alignment
// HARD GATE #1 FIX: Converted to Fixed64 for deterministic cross-platform consensus
// Q32.32 format: 0.4 = 0x6666_6666, 0.3 = 0x4CCC_CCCD
pub const WEIGHT_TRUTH: Fixed64 = Fixed64::from_bits(0x0000_0000_6666_6666);  // 0.4
pub const WEIGHT_LOGIC: Fixed64 = Fixed64::from_bits(0x0000_0000_4CCC_CCCD);  // 0.3
pub const WEIGHT_INTENT: Fixed64 = Fixed64::from_bits(0x0000_0000_4CCC_CCCD); // 0.3

#[derive(Debug, Deserialize)]
struct IhsanConstitutionFile {
    id: Option<String>,
    units: IhsanUnits,
    threshold_policy: Option<IhsanThresholdPolicyFile>,
    dimensions: BTreeMap<String, IhsanDimensionSpec>,
    invariants: Option<IhsanInvariants>,
}

#[derive(Debug, Deserialize)]
struct IhsanUnits {
    score_range: [f64; 2],
    threshold: f64,
}

#[derive(Debug, Deserialize)]
struct IhsanThresholdPolicyFile {
    version: Option<u32>,
    combine: Option<String>,
    default_env: Option<String>,
    thresholds_by_env: Option<BTreeMap<String, f64>>,
    thresholds_by_artifact_class: Option<BTreeMap<String, f64>>,
    normalization: Option<IhsanThresholdNormalizationFile>,
}

#[derive(Debug, Deserialize)]
struct IhsanThresholdNormalizationFile {
    env_aliases: Option<BTreeMap<String, String>>,
    artifact_class_aliases: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Deserialize)]
struct IhsanDimensionSpec {
    weight: f64,
}

#[derive(Debug, Deserialize)]
struct IhsanInvariants {
    weights_sum: Option<f64>,
    required_dimensions: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct GenesisManifest {
    genesis_hash: String,
    registry: Vec<GenesisManifestEntry>,
}

#[derive(Debug, Deserialize)]
struct GenesisManifestEntry {
    path: String,
    sha256: String,
}

#[derive(Debug, Clone)]
pub struct IhsanConstitution {
    id: String,
    threshold: f64,
    default_env: String,
    threshold_combine: ThresholdCombine,
    thresholds_by_env: BTreeMap<String, f64>,
    thresholds_by_artifact_class: BTreeMap<String, f64>,
    env_aliases: BTreeMap<String, String>,
    artifact_class_aliases: BTreeMap<String, String>,
    weights: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy)]
enum ThresholdCombine {
    Max,
    Min,
}

impl ThresholdCombine {
    fn parse(raw: Option<&str>) -> anyhow::Result<Self> {
        match raw.unwrap_or("max").trim().to_ascii_lowercase().as_str() {
            "max" => Ok(Self::Max),
            "min" => Ok(Self::Min),
            other => anyhow::bail!("unsupported threshold_policy.combine: {other}"),
        }
    }
}

impl IhsanConstitution {
    fn normalize_key(raw: &str) -> String {
        raw.trim().to_ascii_lowercase().replace(['-', ' '], "_")
    }

    fn normalize_alias_map(map: Option<BTreeMap<String, String>>) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        let Some(map) = map else {
            return out;
        };

        for (k, v) in map {
            let key = Self::normalize_key(&k);
            let val = Self::normalize_key(&v);
            if !key.is_empty() && !val.is_empty() {
                out.insert(key, val);
            }
        }
        out
    }

    fn normalize_threshold_map(
        map: Option<BTreeMap<String, f64>>,
        min: f64,
        max: f64,
    ) -> anyhow::Result<BTreeMap<String, f64>> {
        let mut out = BTreeMap::new();
        let Some(map) = map else {
            return Ok(out);
        };

        for (k, v) in map {
            let key = Self::normalize_key(&k);
            if key.is_empty() {
                continue;
            }
            if !v.is_finite() || v < min || v > max {
                anyhow::bail!("threshold out of range for '{key}': {v} (expected {min}..={max})");
            }
            out.insert(key, v);
        }
        Ok(out)
    }

    fn canonicalize(key: &str, aliases: &BTreeMap<String, String>) -> String {
        let norm = Self::normalize_key(key);
        aliases.get(&norm).cloned().unwrap_or(norm)
    }

    fn from_yaml_str(yaml: &str) -> anyhow::Result<Self> {
        let parsed: IhsanConstitutionFile = serde_yaml::from_str(yaml)?;

        let expected_sum = parsed
            .invariants
            .as_ref()
            .and_then(|i| i.weights_sum)
            .unwrap_or(1.0);

        let weights: BTreeMap<String, f64> = parsed
            .dimensions
            .iter()
            .map(|(k, v)| (k.clone(), v.weight))
            .collect();

        let sum: f64 = weights.values().sum();
        if (sum - expected_sum).abs() > 1e-9 {
            anyhow::bail!("ihsan constitution weights do not sum to {expected_sum} (got {sum})");
        }

        for (name, weight) in &weights {
            if !weight.is_finite() || *weight < 0.0 || *weight > 1.0 {
                anyhow::bail!("ihsan weight out of range for {name}: {weight}");
            }
        }

        let min = parsed.units.score_range[0];
        let max = parsed.units.score_range[1];
        if !(min <= parsed.units.threshold && parsed.units.threshold <= max) {
            anyhow::bail!(
                "ihsan threshold {} outside score_range [{}, {}]",
                parsed.units.threshold,
                min,
                max
            );
        }

        let required = parsed
            .invariants
            .as_ref()
            .and_then(|i| i.required_dimensions.clone())
            .unwrap_or_else(|| weights.keys().cloned().collect());

        for dim in required {
            if !weights.contains_key(&dim) {
                anyhow::bail!("ihsan constitution missing required dimension: {dim}");
            }
        }

        let policy = parsed.threshold_policy;
        let default_env = policy
            .as_ref()
            .and_then(|p| p.default_env.clone())
            .unwrap_or_else(|| "development".to_string());

        let threshold_combine =
            ThresholdCombine::parse(policy.as_ref().and_then(|p| p.combine.as_deref()))?;

        let thresholds_by_env = Self::normalize_threshold_map(
            policy.as_ref().and_then(|p| p.thresholds_by_env.clone()),
            min,
            max,
        )?;
        let thresholds_by_artifact_class = Self::normalize_threshold_map(
            policy
                .as_ref()
                .and_then(|p| p.thresholds_by_artifact_class.clone()),
            min,
            max,
        )?;

        let env_aliases = Self::normalize_alias_map(
            policy
                .as_ref()
                .and_then(|p| p.normalization.as_ref())
                .and_then(|n| n.env_aliases.clone()),
        );
        let artifact_class_aliases = Self::normalize_alias_map(
            policy
                .as_ref()
                .and_then(|p| p.normalization.as_ref())
                .and_then(|n| n.artifact_class_aliases.clone()),
        );

        if let Some(v) = policy.as_ref().and_then(|p| p.version) {
            if v != 1 {
                anyhow::bail!("unsupported threshold_policy.version: {v}");
            }
        }

        Ok(Self {
            id: parsed.id.unwrap_or_else(|| "ihsan".to_string()),
            threshold: parsed.units.threshold,
            default_env,
            threshold_combine,
            thresholds_by_env,
            thresholds_by_artifact_class,
            env_aliases,
            artifact_class_aliases,
            weights,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn default_env(&self) -> &str {
        &self.default_env
    }

    pub fn threshold_for(&self, env: &str, artifact_class: &str) -> f64 {
        let env_key = Self::canonicalize(env, &self.env_aliases);
        let artifact_key = Self::canonicalize(artifact_class, &self.artifact_class_aliases);

        let mut candidates: Vec<f64> = Vec::new();
        if let Some(t) = self.thresholds_by_env.get(&env_key) {
            candidates.push(*t);
        }
        if let Some(t) = self.thresholds_by_artifact_class.get(&artifact_key) {
            candidates.push(*t);
        }

        if candidates.is_empty() {
            return self.threshold;
        }

        match self.threshold_combine {
            ThresholdCombine::Max => candidates
                .into_iter()
                .fold(f64::NEG_INFINITY, |a, b| a.max(b)),
            ThresholdCombine::Min => candidates.into_iter().fold(f64::INFINITY, |a, b| a.min(b)),
        }
    }

    pub fn weights(&self) -> &BTreeMap<String, f64> {
        &self.weights
    }

    pub fn score(&self, scores: &BTreeMap<String, f64>) -> anyhow::Result<f64> {
        let min = 0.0;
        let max = 1.0;

        for (dim, weight) in &self.weights {
            let value = scores.get(dim).copied().ok_or_else(|| {
                anyhow::anyhow!("ihsan score input missing required dimension: {dim}")
            })?;
            if !value.is_finite() || value < min || value > max {
                anyhow::bail!("ihsan score input out of range for {dim}: {value}");
            }
            if !weight.is_finite() || *weight < 0.0 {
                anyhow::bail!("ihsan constitution weight invalid for {dim}: {weight}");
            }
        }

        Ok(self
            .weights
            .iter()
            .map(|(dim, w)| w * scores.get(dim).copied().unwrap_or(0.0))
            .sum())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn load_genesis_manifest() -> anyhow::Result<GenesisManifest> {
    let raw = fs::read_to_string(GENESIS_MANIFEST_PATH).map_err(|e| {
        anyhow::anyhow!(
            "failed to read genesis manifest at {}: {}",
            GENESIS_MANIFEST_PATH,
            e
        )
    })?;
    serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("invalid genesis manifest: {e}"))
}

fn enforce_genesis_seal(policy_hash: &str) -> anyhow::Result<()> {
    let manifest = load_genesis_manifest()?;
    if manifest.genesis_hash != SEALED_GENESIS_HASH {
        anyhow::bail!(
            "genesis hash mismatch: expected {} got {}",
            SEALED_GENESIS_HASH,
            manifest.genesis_hash
        );
    }

    let entry = manifest
        .registry
        .iter()
        .find(|item| item.path == IHSAN_CONSTITUTION_PATH)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "ihsan constitution entry missing from genesis manifest: {}",
                IHSAN_CONSTITUTION_PATH
            )
        })?;

    if entry.sha256 != policy_hash {
        anyhow::bail!(
            "ihsan policy hash mismatch: expected {} got {}",
            entry.sha256,
            policy_hash
        );
    }

    Ok(())
}

fn load_constitution_from_disk() -> anyhow::Result<IhsanConstitution> {
    let yaml = fs::read_to_string(IHSAN_CONSTITUTION_PATH).map_err(|e| {
        anyhow::anyhow!(
            "failed to read ihsan constitution at {}: {}",
            IHSAN_CONSTITUTION_PATH,
            e
        )
    })?;
    let policy_hash = sha256_hex(yaml.as_bytes());
    enforce_genesis_seal(&policy_hash)?;
    IhsanConstitution::from_yaml_str(&yaml)
}

pub fn constitution() -> &'static IhsanConstitution {
    static ONCE: OnceLock<IhsanConstitution> = OnceLock::new();
    ONCE.get_or_init(|| {
        // HARD GATE #2 FIX: Fail-fast with clear error message instead of expect()
        // Constitution loading failure is fatal (fail-closed security policy)
        // but we provide detailed diagnostics before termination
        load_constitution_from_disk().unwrap_or_else(|e| {
            eprintln!("❌ FATAL: Failed to load Ihsān constitution");
            eprintln!("   Error: {}", e);
            eprintln!("   Path: {}", IHSAN_CONSTITUTION_PATH);
            eprintln!("   Genesis Manifest: {}", GENESIS_MANIFEST_PATH);
            eprintln!("   Sealed Hash: {}", SEALED_GENESIS_HASH);
            eprintln!("\nThis is a fail-closed security boundary.");
            eprintln!("Constitution tampering or corruption is not permitted.");
            std::process::exit(1);
        })
    })
}

pub fn score(scores: &BTreeMap<String, f64>) -> anyhow::Result<f64> {
    constitution().score(scores)
}

/// Compute Ihsān score from individual dimension values.
/// This is the Python FFI-friendly entry point using positional args.
#[allow(clippy::too_many_arguments)] // 8 Ihsān dimensions are the API contract
pub fn compute_ihsan_score(
    correctness: f64,
    safety: f64,
    user_benefit: f64,
    efficiency: f64,
    auditability: f64,
    anti_centralization: f64,
    robustness: f64,
    adl_fairness: f64,
) -> anyhow::Result<f64> {
    let mut scores = BTreeMap::new();
    scores.insert("correctness".to_string(), correctness);
    scores.insert("safety".to_string(), safety);
    scores.insert("user_benefit".to_string(), user_benefit);
    scores.insert("efficiency".to_string(), efficiency);
    scores.insert("auditability".to_string(), auditability);
    scores.insert("anti_centralization".to_string(), anti_centralization);
    scores.insert("robustness".to_string(), robustness);
    scores.insert("adl_fairness".to_string(), adl_fairness);
    score(&scores)
}

pub fn current_env() -> String {
    // Check BIZRA_IHSAN_ENV first (set by docker-compose)
    if let Ok(v) = std::env::var("BIZRA_IHSAN_ENV") {
        if !v.trim().is_empty() {
            return v.trim().to_string();
        }
    }
    if let Ok(v) = std::env::var("BIZRA_ENV") {
        if !v.trim().is_empty() {
            return v.trim().to_string();
        }
    }
    if let Ok(v) = std::env::var("NODE_ENV") {
        if !v.trim().is_empty() {
            return v.trim().to_string();
        }
    }
    if std::env::var("CI").is_ok() {
        return "ci".to_string();
    }
    constitution().default_env().to_string()
}

pub fn should_enforce() -> bool {
    if let Ok(v) = std::env::var("BIZRA_IHSAN_ENFORCE") {
        let val = v.trim().to_ascii_lowercase();
        if matches!(val.as_str(), "1" | "true" | "yes" | "on") {
            return true;
        }
    }

    let env = current_env();
    let canonical = IhsanConstitution::canonicalize(&env, &constitution().env_aliases);
    matches!(canonical.as_str(), "ci" | "production")
}

/// IhsanDimensions: 8-dimensional quality metrics with Islamic naming
/// Maps to IhsanScore fields in thought.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IhsanDimensions {
    pub adl: Fixed64,       // Justice (correctness)
    pub amanah: Fixed64,    // Trust (safety)
    pub ihsan: Fixed64,     // Excellence (user_benefit)
    pub hikmah: Fixed64,    // Wisdom (efficiency)
    pub bayan: Fixed64,     // Clarity (auditability)
    pub tawhid: Fixed64,    // Unity (anti_centralization)
    pub sabr: Fixed64,      // Patience (robustness)
    pub mizan: Fixed64,     // Balance (fairness)
}

impl IhsanDimensions {
    /// Compute weighted total using constitutional weights
    pub fn compute_total(&self, constitution: &IhsanConstitution) -> Fixed64 {
        let weights = constitution.weights();

        let mut sum = Fixed64::ZERO;
        for (dim, weight) in weights.iter() {
            let value = match dim.as_str() {
                "correctness" => self.adl,
                "safety" => self.amanah,
                "user_benefit" => self.ihsan,
                "efficiency" => self.hikmah,
                "auditability" => self.bayan,
                "anti_centralization" => self.tawhid,
                "robustness" => self.sabr,
                "adl_fairness" => self.mizan,
                _ => Fixed64::ZERO,
            };
            sum = sum + Fixed64::from_f64(*weight) * value;
        }

        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constitution_weights_sum_to_one() {
        let sum: f64 = constitution().weights().values().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum was {sum}");
    }

    #[test]
    fn score_matches_manual_dot_product() {
        let mut scores = BTreeMap::new();
        scores.insert("correctness".to_string(), 1.0);
        scores.insert("safety".to_string(), 1.0);
        scores.insert("user_benefit".to_string(), 0.5);
        scores.insert("efficiency".to_string(), 0.25);
        scores.insert("auditability".to_string(), 0.75);
        scores.insert("anti_centralization".to_string(), 0.0);
        scores.insert("robustness".to_string(), 0.9);
        scores.insert("adl_fairness".to_string(), 0.8);

        let actual = score(&scores).expect("Score calculation should succeed with valid inputs");
        let expected = 0.20 * 1.0
            + 0.20 * 1.0
            + 0.10 * 0.5
            + 0.12 * 0.25
            + 0.12 * 0.75
            + 0.08 * 0.0
            + 0.06 * 0.9
            + 0.12 * 0.8;

        assert!(
            (actual - expected).abs() < 1e-9,
            "actual={actual} expected={expected}"
        );
    }

    #[test]
    fn threshold_policy_is_applied() {
        let c = constitution();
        assert!((c.threshold_for("production", "code") - 0.95).abs() < 1e-9);
        assert!((c.threshold_for("dev", "docs") - 0.80).abs() < 1e-9);
        assert!((c.threshold_for("ci", "docs") - 0.90).abs() < 1e-9);
        assert!((c.threshold_for("ci", "receipt") - 0.95).abs() < 1e-9);
    }
}
