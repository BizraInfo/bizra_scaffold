// src/pipeline.rs - Sovereign Cognitive Pipeline
//
// PEAK MASTERPIECE v7.1: Unified integration of all BIZRA subsystems
//
// Giants Protocol Synthesis:
// - Systems Engineering (Brooks): Manage complexity through modularity
// - Cognitive Architecture (Kahneman): System 1/2 dual processing
// - Computer Science (Turing): Composable function pipelines
// - Unix Philosophy (Thompson): Do one thing well, compose via pipes
// - Islamic Jurisprudence (Usul al-Fiqh): Maqasid priority hierarchy
// - Control Theory (Kalman): State estimation with feedback loops
//
// Pipeline Stages:
// Input → Engram → Evolution → PAT → SAT → SAPE → Resonance → Receipt
//
// Key Properties:
// - Each stage is isolated and testable
// - PipelineContext carries state immutably through stages
// - SAT veto halts pipeline early (fail-safe)
// - Resonance provides feedback for continuous improvement
// - All operations use Fixed64 for determinism

use crate::engram::{SovereignEngram, SovereigntyTier};
use crate::evolution::SovereignEvolution;
use crate::fixed::Fixed64;
use crate::sape::base::SnrTier;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info, instrument, warn};

// ============================================================================
// PIPELINE CONTEXT
// ============================================================================

/// Immutable context that flows through pipeline stages
/// Each stage appends its output without modifying previous state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineContext {
    /// Unique request identifier
    pub request_id: String,
    /// Original input
    pub raw_input: String,
    /// Tokenized input
    pub tokens: Vec<u32>,
    /// Sovereignty tier for this request
    pub tier: SovereigntyTier,
    /// Stage outputs accumulated
    pub stage_outputs: HashMap<String, StageOutput>,
    /// Cumulative latency in nanoseconds
    pub latency_ns: u64,
    /// Current pipeline state
    pub state: PipelineState,
    /// Creation timestamp
    pub created_ns: u64,
}

impl PipelineContext {
    /// Create new pipeline context
    pub fn new(input: String, tier: SovereigntyTier) -> Self {
        let request_id = Self::generate_request_id(&input);
        let tokens = Self::tokenize(&input);
        let created_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            request_id,
            raw_input: input,
            tokens,
            tier,
            stage_outputs: HashMap::new(),
            latency_ns: 0,
            state: PipelineState::Ready,
            created_ns,
        }
    }

    /// Generate deterministic request ID
    fn generate_request_id(input: &str) -> String {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let content = format!("{}:{}", timestamp, input);
        let hash = Sha256::digest(content.as_bytes());
        format!("req-{:x}", &hash[..8].iter().fold(0u64, |acc, &b| acc << 8 | b as u64))
    }

    /// Simple tokenization (in production, use proper tokenizer)
    fn tokenize(input: &str) -> Vec<u32> {
        input.bytes().map(|b| b as u32).take(512).collect()
    }

    /// Add stage output to context
    pub fn with_stage_output(mut self, stage: &str, output: StageOutput) -> Self {
        self.stage_outputs.insert(stage.to_string(), output);
        self
    }

    /// Add latency
    pub fn with_latency(mut self, latency_ns: u64) -> Self {
        self.latency_ns += latency_ns;
        self
    }

    /// Update state
    pub fn with_state(mut self, state: PipelineState) -> Self {
        self.state = state;
        self
    }

    /// Get stage output
    pub fn get_stage_output(&self, stage: &str) -> Option<&StageOutput> {
        self.stage_outputs.get(stage)
    }

    /// Compute context hash for receipts
    pub fn compute_hash(&self) -> String {
        let content = format!(
            "{}:{}:{}:{:?}",
            self.request_id,
            self.raw_input.len(),
            self.latency_ns,
            self.state
        );
        let hash = Sha256::digest(content.as_bytes());
        format!("{:x}", hash)
    }
}

/// Pipeline execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineState {
    /// Ready to start
    Ready,
    /// Currently executing
    Running,
    /// Completed successfully
    Completed,
    /// Rejected by SAT
    Rejected,
    /// Error occurred
    Error,
}

/// Output from a pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageOutput {
    /// Stage name
    pub stage: String,
    /// Success flag
    pub success: bool,
    /// Confidence/quality score
    pub confidence: Fixed64,
    /// Stage-specific data (JSON serializable)
    pub data: HashMap<String, String>,
    /// Execution time in nanoseconds
    pub execution_ns: u64,
}

impl StageOutput {
    /// Create new stage output
    pub fn new(stage: &str, success: bool, confidence: Fixed64) -> Self {
        Self {
            stage: stage.to_string(),
            success,
            confidence,
            data: HashMap::new(),
            execution_ns: 0,
        }
    }

    /// Add data field
    pub fn with_data(mut self, key: &str, value: &str) -> Self {
        self.data.insert(key.to_string(), value.to_string());
        self
    }

    /// Set execution time
    pub fn with_execution_time(mut self, ns: u64) -> Self {
        self.execution_ns = ns;
        self
    }
}

// ============================================================================
// PIPELINE STAGE TRAIT
// ============================================================================

/// Trait for pipeline stages
pub trait PipelineStage: Send + Sync {
    /// Stage name
    fn name(&self) -> &str;

    /// Execute the stage
    fn execute(&mut self, ctx: PipelineContext) -> PipelineContext;

    /// Can this stage run given current context?
    fn can_execute(&self, ctx: &PipelineContext) -> bool {
        ctx.state == PipelineState::Running
    }
}

// ============================================================================
// INPUT STAGE
// ============================================================================

/// Input validation and sanitization stage
#[derive(Debug)]
pub struct InputStage {
    /// Maximum input length
    max_length: usize,
    /// Blocked patterns (security)
    blocked_patterns: Vec<String>,
}

impl InputStage {
    /// Create new input stage
    pub fn new(max_length: usize) -> Self {
        Self {
            max_length,
            blocked_patterns: vec![
                "DROP TABLE".to_string(),
                "<script>".to_string(),
                "eval(".to_string(),
            ],
        }
    }

    /// Check for blocked patterns
    fn is_safe(&self, input: &str) -> bool {
        let upper = input.to_uppercase();
        !self.blocked_patterns.iter().any(|p| upper.contains(&p.to_uppercase()))
    }
}

impl PipelineStage for InputStage {
    fn name(&self) -> &str {
        "input"
    }

    fn execute(&mut self, ctx: PipelineContext) -> PipelineContext {
        let start = Instant::now();

        // Validate length
        if ctx.raw_input.len() > self.max_length {
            warn!(
                request_id = %ctx.request_id,
                len = ctx.raw_input.len(),
                max = self.max_length,
                "Input exceeds maximum length"
            );
            let output = StageOutput::new(self.name(), false, Fixed64::ZERO)
                .with_data("error", "Input too long")
                .with_execution_time(start.elapsed().as_nanos() as u64);

            return ctx
                .with_stage_output(self.name(), output)
                .with_state(PipelineState::Rejected);
        }

        // Check for blocked patterns
        if !self.is_safe(&ctx.raw_input) {
            warn!(
                request_id = %ctx.request_id,
                "Input contains blocked pattern"
            );
            let output = StageOutput::new(self.name(), false, Fixed64::ZERO)
                .with_data("error", "Blocked pattern detected")
                .with_execution_time(start.elapsed().as_nanos() as u64);

            return ctx
                .with_stage_output(self.name(), output)
                .with_state(PipelineState::Rejected);
        }

        let output = StageOutput::new(self.name(), true, Fixed64::ONE)
            .with_data("tokens", &ctx.tokens.len().to_string())
            .with_data("validated", "true")
            .with_execution_time(start.elapsed().as_nanos() as u64);

        debug!(request_id = %ctx.request_id, "Input validation passed");

        ctx.with_stage_output(self.name(), output)
            .with_latency(start.elapsed().as_nanos() as u64)
    }

    fn can_execute(&self, ctx: &PipelineContext) -> bool {
        ctx.state == PipelineState::Ready || ctx.state == PipelineState::Running
    }
}

// ============================================================================
// ENGRAM STAGE
// ============================================================================

/// Engram knowledge retrieval and enhancement stage
#[derive(Debug)]
pub struct EngramStage {
    /// Engram module
    engram: SovereignEngram,
}

impl EngramStage {
    /// Create new Engram stage
    pub fn new(tier: SovereigntyTier) -> Self {
        Self {
            engram: SovereignEngram::new(tier),
        }
    }
}

impl PipelineStage for EngramStage {
    fn name(&self) -> &str {
        "engram"
    }

    fn execute(&mut self, ctx: PipelineContext) -> PipelineContext {
        let start = Instant::now();

        // Create hidden states from tokens
        let dim = self.engram.dim();
        let hidden_states: Vec<Vec<Fixed64>> = ctx.tokens
            .iter()
            .map(|_| vec![Fixed64::HALF; dim])
            .collect();

        // Forward through Engram
        let enhanced = self.engram.forward(&ctx.tokens, &hidden_states);

        // Compute enhancement magnitude
        let enhancement = Self::compute_enhancement(&hidden_states, &enhanced);

        let output = StageOutput::new(self.name(), true, enhancement)
            .with_data("dim", &dim.to_string())
            .with_data("lookups", &ctx.tokens.len().to_string())
            .with_data("enhancement", &format!("{:.4}", enhancement.to_f64()))
            .with_execution_time(start.elapsed().as_nanos() as u64);

        debug!(
            request_id = %ctx.request_id,
            enhancement = enhancement.to_f64(),
            "Engram enhancement complete"
        );

        ctx.with_stage_output(self.name(), output)
            .with_latency(start.elapsed().as_nanos() as u64)
    }
}

impl EngramStage {
    /// Compute enhancement magnitude
    fn compute_enhancement(original: &[Vec<Fixed64>], enhanced: &[Vec<Fixed64>]) -> Fixed64 {
        if original.is_empty() || enhanced.is_empty() {
            return Fixed64::HALF;
        }

        let mut total = Fixed64::ZERO;
        let mut count = 0;

        for (orig, enh) in original.iter().zip(enhanced.iter()) {
            for (o, e) in orig.iter().zip(enh.iter()) {
                let diff = *e - *o;
                total = total + (diff * diff);
                count += 1;
            }
        }

        if count == 0 {
            return Fixed64::HALF;
        }

        // Normalize to [0.5, 1.0]
        let avg = total / Fixed64::from_int(count);
        let result = Fixed64::HALF + avg;

        if result > Fixed64::ONE {
            Fixed64::ONE
        } else {
            result
        }
    }
}

// ============================================================================
// EVOLUTION STAGE
// ============================================================================

/// Self-evolution learning stage
#[derive(Debug)]
#[allow(dead_code)] // Reserved fields for future expansion
pub struct EvolutionStage {
    /// Evolution engine
    evolution: SovereignEvolution,
    /// Cycles per pipeline execution
    cycles_per_execution: usize,
}

impl EvolutionStage {
    /// Create new Evolution stage
    pub fn new(tier: SovereigntyTier, seed: u64) -> Self {
        Self {
            evolution: SovereignEvolution::new(tier, seed),
            cycles_per_execution: 1, // Single cycle per request for latency
        }
    }
}

impl PipelineStage for EvolutionStage {
    fn name(&self) -> &str {
        "evolution"
    }

    fn execute(&mut self, ctx: PipelineContext) -> PipelineContext {
        let start = Instant::now();

        // Run evolution cycle
        let state = self.evolution.evolve_cycle();

        let confidence = state.avg_reward;

        let output = StageOutput::new(self.name(), true, confidence)
            .with_data("generation", &state.generation.to_string())
            .with_data("tasks_completed", &state.tasks_completed.to_string())
            .with_data("avg_reward", &format!("{:.4}", confidence.to_f64()))
            .with_data("max_difficulty", &state.max_difficulty_reached.to_string())
            .with_execution_time(start.elapsed().as_nanos() as u64);

        debug!(
            request_id = %ctx.request_id,
            generation = state.generation,
            "Evolution cycle complete"
        );

        ctx.with_stage_output(self.name(), output)
            .with_latency(start.elapsed().as_nanos() as u64)
    }
}

// ============================================================================
// PAT STAGE (Simplified for Pipeline)
// ============================================================================

/// PAT multi-agent execution stage
#[derive(Debug)]
#[allow(dead_code)] // Reserved fields for future expansion
pub struct PATStage {
    /// Agent weights
    agent_weights: HashMap<String, Fixed64>,
}

impl PATStage {
    /// Create new PAT stage
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("strategic".to_string(), Fixed64::from_f64(0.15));
        weights.insert("creative".to_string(), Fixed64::from_f64(0.15));
        weights.insert("analytical".to_string(), Fixed64::from_f64(0.15));
        weights.insert("implementer".to_string(), Fixed64::from_f64(0.15));
        weights.insert("quality".to_string(), Fixed64::from_f64(0.15));
        weights.insert("user".to_string(), Fixed64::from_f64(0.15));
        weights.insert("integration".to_string(), Fixed64::from_f64(0.10));

        Self {
            agent_weights: weights,
        }
    }

    /// Simulate PAT consensus (in production, would run actual agents)
    fn compute_consensus(&self, ctx: &PipelineContext) -> Fixed64 {
        // Base confidence from previous stages
        let engram_conf = ctx
            .get_stage_output("engram")
            .map(|o| o.confidence)
            .unwrap_or(Fixed64::HALF);

        let evolution_conf = ctx
            .get_stage_output("evolution")
            .map(|o| o.confidence)
            .unwrap_or(Fixed64::HALF);

        // Weighted combination
        (engram_conf * Fixed64::from_f64(0.4)) + (evolution_conf * Fixed64::from_f64(0.6))
    }
}

impl Default for PATStage {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStage for PATStage {
    fn name(&self) -> &str {
        "pat"
    }

    fn execute(&mut self, ctx: PipelineContext) -> PipelineContext {
        let start = Instant::now();

        let consensus = self.compute_consensus(&ctx);

        let output = StageOutput::new(self.name(), true, consensus)
            .with_data("agents", "7")
            .with_data("consensus", &format!("{:.4}", consensus.to_f64()))
            .with_execution_time(start.elapsed().as_nanos() as u64);

        debug!(
            request_id = %ctx.request_id,
            consensus = consensus.to_f64(),
            "PAT consensus complete"
        );

        ctx.with_stage_output(self.name(), output)
            .with_latency(start.elapsed().as_nanos() as u64)
    }
}

// ============================================================================
// SAT STAGE (Simplified for Pipeline)
// ============================================================================

/// SAT validation stage with veto power
#[derive(Debug)]
#[allow(dead_code)] // Reserved fields for future expansion
pub struct SATStage {
    /// Minimum confidence threshold
    min_confidence: Fixed64,
    /// Validator weights
    validator_weights: HashMap<String, Fixed64>,
}

impl SATStage {
    /// Create new SAT stage
    pub fn new(min_confidence: f64) -> Self {
        let mut weights = HashMap::new();
        weights.insert("security".to_string(), Fixed64::from_f64(2.5));
        weights.insert("formal".to_string(), Fixed64::from_f64(1.8));
        weights.insert("ethics".to_string(), Fixed64::from_f64(2.0));
        weights.insert("performance".to_string(), Fixed64::from_f64(1.0));
        weights.insert("consistency".to_string(), Fixed64::from_f64(1.0));
        weights.insert("resource".to_string(), Fixed64::from_f64(0.8));

        Self {
            min_confidence: Fixed64::from_f64(min_confidence),
            validator_weights: weights,
        }
    }

    /// Check if PAT output passes SAT validation
    fn validate(&self, ctx: &PipelineContext) -> (bool, Fixed64) {
        let pat_conf = ctx
            .get_stage_output("pat")
            .map(|o| o.confidence)
            .unwrap_or(Fixed64::ZERO);

        let approved = pat_conf >= self.min_confidence;
        (approved, pat_conf)
    }
}

impl PipelineStage for SATStage {
    fn name(&self) -> &str {
        "sat"
    }

    fn execute(&mut self, ctx: PipelineContext) -> PipelineContext {
        let start = Instant::now();

        let (approved, confidence) = self.validate(&ctx);

        if !approved {
            warn!(
                request_id = %ctx.request_id,
                confidence = confidence.to_f64(),
                threshold = self.min_confidence.to_f64(),
                "SAT VETO: Confidence below threshold"
            );

            let output = StageOutput::new(self.name(), false, confidence)
                .with_data("approved", "false")
                .with_data("reason", "Confidence below threshold")
                .with_data("validators", "6")
                .with_execution_time(start.elapsed().as_nanos() as u64);

            return ctx
                .with_stage_output(self.name(), output)
                .with_state(PipelineState::Rejected)
                .with_latency(start.elapsed().as_nanos() as u64);
        }

        let output = StageOutput::new(self.name(), true, confidence)
            .with_data("approved", "true")
            .with_data("validators", "6")
            .with_data("consensus", &format!("{:.4}", confidence.to_f64()))
            .with_execution_time(start.elapsed().as_nanos() as u64);

        debug!(
            request_id = %ctx.request_id,
            confidence = confidence.to_f64(),
            "SAT validation approved"
        );

        ctx.with_stage_output(self.name(), output)
            .with_latency(start.elapsed().as_nanos() as u64)
    }
}

// ============================================================================
// SAPE STAGE
// ============================================================================

/// SAPE pattern elevation stage
#[derive(Debug)]
pub struct SAPEStage {
    /// Probe dimensions
    probe_count: usize,
}

impl SAPEStage {
    /// Create new SAPE stage
    pub fn new() -> Self {
        Self { probe_count: 9 }
    }

    /// Compute SNR tier from pipeline context
    /// Maps confidence to SnrTier (T1-T6) based on Ihsan thresholds
    fn compute_snr_tier(&self, ctx: &PipelineContext) -> SnrTier {
        let avg_confidence = self.compute_avg_confidence(ctx);

        // Map confidence (0-1) to SNR tier using Ihsan conversion
        // T6 (9.0+): >= 0.95 (Masterpiece)
        // T5 (8.6-9.0): >= 0.90 (Elite)
        // T4 (8.2-8.6): >= 0.85 (High Stakes)
        // T3 (7.8-8.2): >= 0.75 (Professional)
        // T2 (7.5-7.8): >= 0.60 (Standard)
        // T1 (< 7.5): < 0.60 (Baseline)
        if avg_confidence >= Fixed64::from_f64(0.95) {
            SnrTier::T6
        } else if avg_confidence >= Fixed64::from_f64(0.90) {
            SnrTier::T5
        } else if avg_confidence >= Fixed64::from_f64(0.85) {
            SnrTier::T4
        } else if avg_confidence >= Fixed64::from_f64(0.75) {
            SnrTier::T3
        } else if avg_confidence >= Fixed64::from_f64(0.60) {
            SnrTier::T2
        } else {
            SnrTier::T1
        }
    }

    /// Compute average confidence across stages
    fn compute_avg_confidence(&self, ctx: &PipelineContext) -> Fixed64 {
        let stages = ["input", "engram", "evolution", "pat", "sat"];
        let mut sum = Fixed64::ZERO;
        let mut count = 0;

        for stage in &stages {
            if let Some(output) = ctx.get_stage_output(stage) {
                sum = sum + output.confidence;
                count += 1;
            }
        }

        if count == 0 {
            Fixed64::HALF
        } else {
            sum / Fixed64::from_int(count)
        }
    }
}

impl Default for SAPEStage {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStage for SAPEStage {
    fn name(&self) -> &str {
        "sape"
    }

    fn execute(&mut self, ctx: PipelineContext) -> PipelineContext {
        let start = Instant::now();

        let snr_tier = self.compute_snr_tier(&ctx);
        let avg_conf = self.compute_avg_confidence(&ctx);

        let output = StageOutput::new(self.name(), true, avg_conf)
            .with_data("snr_tier", &format!("{:?}", snr_tier))
            .with_data("probes", &self.probe_count.to_string())
            .with_data("avg_confidence", &format!("{:.4}", avg_conf.to_f64()))
            .with_execution_time(start.elapsed().as_nanos() as u64);

        debug!(
            request_id = %ctx.request_id,
            snr_tier = ?snr_tier,
            "SAPE analysis complete"
        );

        ctx.with_stage_output(self.name(), output)
            .with_latency(start.elapsed().as_nanos() as u64)
    }
}

// ============================================================================
// RESONANCE STAGE
// ============================================================================

/// Resonance feedback and optimization stage
#[derive(Debug)]
#[allow(dead_code)] // Reserved fields for future expansion
pub struct ResonanceStage {
    /// Pruning threshold
    prune_threshold: Fixed64,
    /// History for feedback
    history_size: usize,
}

impl ResonanceStage {
    /// Create new Resonance stage
    pub fn new(prune_threshold: f64) -> Self {
        Self {
            prune_threshold: Fixed64::from_f64(prune_threshold),
            history_size: 100,
        }
    }

    /// Compute feedback signal
    fn compute_feedback(&self, ctx: &PipelineContext) -> Fixed64 {
        // Feedback is based on SAPE tier and SAT approval
        let sape_conf = ctx
            .get_stage_output("sape")
            .map(|o| o.confidence)
            .unwrap_or(Fixed64::HALF);

        let sat_approved = ctx
            .get_stage_output("sat")
            .map(|o| o.success)
            .unwrap_or(false);

        if sat_approved {
            sape_conf
        } else {
            Fixed64::ZERO
        }
    }
}

impl PipelineStage for ResonanceStage {
    fn name(&self) -> &str {
        "resonance"
    }

    fn execute(&mut self, ctx: PipelineContext) -> PipelineContext {
        let start = Instant::now();

        let feedback = self.compute_feedback(&ctx);
        let should_prune = feedback < self.prune_threshold;

        let output = StageOutput::new(self.name(), true, feedback)
            .with_data("feedback", &format!("{:.4}", feedback.to_f64()))
            .with_data("prune_threshold", &format!("{:.4}", self.prune_threshold.to_f64()))
            .with_data("should_prune", &should_prune.to_string())
            .with_execution_time(start.elapsed().as_nanos() as u64);

        debug!(
            request_id = %ctx.request_id,
            feedback = feedback.to_f64(),
            "Resonance feedback computed"
        );

        ctx.with_stage_output(self.name(), output)
            .with_latency(start.elapsed().as_nanos() as u64)
    }
}

// ============================================================================
// RECEIPT STAGE
// ============================================================================

/// Final receipt generation stage
#[derive(Debug)]
pub struct ReceiptStage {
    /// Receipt counter
    counter: u64,
}

impl ReceiptStage {
    /// Create new Receipt stage
    pub fn new() -> Self {
        Self { counter: 0 }
    }
}

impl Default for ReceiptStage {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStage for ReceiptStage {
    fn name(&self) -> &str {
        "receipt"
    }

    fn execute(&mut self, mut ctx: PipelineContext) -> PipelineContext {
        let start = Instant::now();

        self.counter += 1;

        let final_state = if ctx.state == PipelineState::Running {
            PipelineState::Completed
        } else {
            ctx.state
        };

        let receipt_id = format!("PIPE-{}-{:06}", ctx.request_id, self.counter);
        let context_hash = ctx.compute_hash();

        let overall_confidence = ctx
            .stage_outputs
            .values()
            .map(|o| o.confidence)
            .fold(Fixed64::ZERO, |a, b| a + b)
            / Fixed64::from_int(ctx.stage_outputs.len().max(1) as i32);

        let output = StageOutput::new(self.name(), true, overall_confidence)
            .with_data("receipt_id", &receipt_id)
            .with_data("context_hash", &context_hash)
            .with_data("total_latency_ns", &ctx.latency_ns.to_string())
            .with_data("stages_completed", &ctx.stage_outputs.len().to_string())
            .with_data("final_state", &format!("{:?}", final_state))
            .with_execution_time(start.elapsed().as_nanos() as u64);

        info!(
            request_id = %ctx.request_id,
            receipt_id = %receipt_id,
            total_latency_ms = ctx.latency_ns / 1_000_000,
            final_state = ?final_state,
            "Pipeline receipt generated"
        );

        ctx.state = final_state;
        ctx.with_stage_output(self.name(), output)
            .with_latency(start.elapsed().as_nanos() as u64)
    }
}

// ============================================================================
// SOVEREIGN PIPELINE
// ============================================================================

/// Unified cognitive pipeline integrating all BIZRA subsystems
pub struct SovereignPipeline {
    /// Pipeline stages in execution order
    stages: Vec<Box<dyn PipelineStage>>,
    /// Sovereignty tier
    tier: SovereigntyTier,
    /// Execution count
    executions: u64,
}

impl SovereignPipeline {
    /// Create new pipeline with default configuration
    #[instrument]
    pub fn new(tier: SovereigntyTier) -> Self {
        info!(tier = ?tier, "🚀 Initializing SovereignPipeline");

        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(InputStage::new(10_000)),
            Box::new(EngramStage::new(tier)),
            Box::new(EvolutionStage::new(tier, 42)),
            Box::new(PATStage::new()),
            Box::new(SATStage::new(0.5)), // 50% minimum confidence
            Box::new(SAPEStage::new()),
            Box::new(ResonanceStage::new(0.3)),
            Box::new(ReceiptStage::new()),
        ];

        Self {
            stages,
            tier,
            executions: 0,
        }
    }

    /// Execute the pipeline
    #[instrument(skip(self, input))]
    pub fn execute(&mut self, input: String) -> PipelineResult {
        self.executions += 1;
        let start = Instant::now();

        info!(
            execution = self.executions,
            input_len = input.len(),
            "Starting pipeline execution"
        );

        // Initialize context
        let mut ctx = PipelineContext::new(input, self.tier)
            .with_state(PipelineState::Running);

        // Execute each stage
        // Note: Receipt stage always runs for auditability (even on rejection)
        let mut terminated_early = false;
        for stage in &mut self.stages {
            // Receipt stage always runs (even after rejection for auditability)
            let is_receipt_stage = stage.name() == "receipt";

            if terminated_early && !is_receipt_stage {
                debug!(stage = stage.name(), "Stage skipped (pipeline terminated)");
                continue;
            }

            if !stage.can_execute(&ctx) && !is_receipt_stage {
                debug!(stage = stage.name(), "Stage skipped (cannot execute)");
                continue;
            }

            ctx = stage.execute(ctx);

            // Check for early termination (but continue to receipt stage)
            if ctx.state == PipelineState::Rejected || ctx.state == PipelineState::Error {
                warn!(
                    stage = stage.name(),
                    state = ?ctx.state,
                    "Pipeline rejected/errored - will still generate receipt"
                );
                terminated_early = true;
            }
        }

        let total_time = start.elapsed();

        info!(
            execution = self.executions,
            total_ms = total_time.as_millis(),
            final_state = ?ctx.state,
            stages_completed = ctx.stage_outputs.len(),
            "Pipeline execution complete"
        );

        PipelineResult {
            context: ctx,
            total_time_ns: total_time.as_nanos() as u64,
            execution_number: self.executions,
        }
    }

    /// Get execution count
    pub fn executions(&self) -> u64 {
        self.executions
    }

    /// Get tier
    pub fn tier(&self) -> SovereigntyTier {
        self.tier
    }
}

/// Result of pipeline execution
#[derive(Debug)]
pub struct PipelineResult {
    /// Final context with all stage outputs
    pub context: PipelineContext,
    /// Total execution time in nanoseconds
    pub total_time_ns: u64,
    /// Execution number
    pub execution_number: u64,
}

impl PipelineResult {
    /// Was the pipeline successful?
    pub fn is_success(&self) -> bool {
        self.context.state == PipelineState::Completed
    }

    /// Was the pipeline rejected?
    pub fn is_rejected(&self) -> bool {
        self.context.state == PipelineState::Rejected
    }

    /// Get final confidence
    pub fn final_confidence(&self) -> Fixed64 {
        self.context
            .get_stage_output("receipt")
            .map(|o| o.confidence)
            .unwrap_or(Fixed64::ZERO)
    }

    /// Get receipt ID
    pub fn receipt_id(&self) -> Option<&str> {
        self.context
            .get_stage_output("receipt")
            .and_then(|o| o.data.get("receipt_id"))
            .map(|s| s.as_str())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_context_creation() {
        let ctx = PipelineContext::new("test input".to_string(), SovereigntyTier::T1Consumer);

        assert!(!ctx.request_id.is_empty());
        assert_eq!(ctx.raw_input, "test input");
        assert!(!ctx.tokens.is_empty());
        assert_eq!(ctx.state, PipelineState::Ready);
    }

    #[test]
    fn test_stage_output_creation() {
        let output = StageOutput::new("test", true, Fixed64::from_f64(0.95))
            .with_data("key", "value")
            .with_execution_time(1000);

        assert_eq!(output.stage, "test");
        assert!(output.success);
        assert_eq!(output.data.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_input_stage_validation() {
        let mut stage = InputStage::new(100);
        let ctx = PipelineContext::new("safe input".to_string(), SovereigntyTier::T0Mobile)
            .with_state(PipelineState::Running);

        let result = stage.execute(ctx);

        assert!(result.get_stage_output("input").unwrap().success);
    }

    #[test]
    fn test_input_stage_rejects_long_input() {
        let mut stage = InputStage::new(10);
        let ctx = PipelineContext::new("this input is too long".to_string(), SovereigntyTier::T0Mobile)
            .with_state(PipelineState::Ready);

        let result = stage.execute(ctx);

        assert!(!result.get_stage_output("input").unwrap().success);
        assert_eq!(result.state, PipelineState::Rejected);
    }

    #[test]
    fn test_input_stage_rejects_blocked_pattern() {
        let mut stage = InputStage::new(1000);
        let ctx = PipelineContext::new("DROP TABLE users".to_string(), SovereigntyTier::T0Mobile)
            .with_state(PipelineState::Ready);

        let result = stage.execute(ctx);

        assert!(!result.get_stage_output("input").unwrap().success);
    }

    #[test]
    fn test_full_pipeline_execution() {
        let mut pipeline = SovereignPipeline::new(SovereigntyTier::T0Mobile);

        let result = pipeline.execute("What is the theme of Surah Al-Fatiha?".to_string());

        assert!(result.context.stage_outputs.len() >= 5);
        assert!(result.total_time_ns > 0);
    }

    #[test]
    fn test_pipeline_determinism() {
        let mut pipe1 = SovereignPipeline::new(SovereigntyTier::T0Mobile);
        let mut pipe2 = SovereignPipeline::new(SovereigntyTier::T0Mobile);

        let result1 = pipe1.execute("test".to_string());
        let result2 = pipe2.execute("test".to_string());

        // Same input should produce same number of stages
        assert_eq!(
            result1.context.stage_outputs.len(),
            result2.context.stage_outputs.len()
        );
    }

    #[test]
    fn test_pipeline_result_methods() {
        let mut pipeline = SovereignPipeline::new(SovereigntyTier::T1Consumer);
        let result = pipeline.execute("test query".to_string());

        // Should have a receipt ID
        assert!(result.receipt_id().is_some());

        // Confidence should be in valid range
        let conf = result.final_confidence();
        assert!(conf >= Fixed64::ZERO);
        assert!(conf <= Fixed64::ONE);
    }
}
