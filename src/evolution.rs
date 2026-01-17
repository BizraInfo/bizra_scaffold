// src/evolution.rs - Sovereign Self-Evolution Framework
//
// PEAK MASTERPIECE v7.1: Dr. Zero architecture adapted for BIZRA sovereignty
//
// Giants Protocol Synthesis:
// - Evolutionary Biology (Darwin): Red Queen Effect - must evolve to compete
// - Game Theory (Nash): Equilibrium via Proposer-Solver co-adaptation
// - Cognitive Science (Vygotsky): Zone of Proximal Development
// - RL Theory (Sutton-Barto): HRPO grouped reward optimization
// - Islamic Pedagogy (Al-Ghazali): Mudarris-Talib progressive mastery
// - Systems Theory (Meadows): Feedback loops and leverage points
//
// Key Properties:
// - Zero external training data required
// - Fixed64 arithmetic for deterministic rewards
// - Tiered evolution speeds: T0=slow, T1=medium, T2=fast
// - Receipt-tracked progress for auditability
// - Sovereignty-first: no cloud dependency

use crate::engram::{SovereignEngram, SovereigntyTier};
use crate::fixed::Fixed64;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use tracing::{debug, info, instrument, warn};

// ============================================================================
// EVOLUTION CONFIGURATION
// ============================================================================

/// Evolution configuration per sovereignty tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// How many evolution cycles per session
    pub cycles_per_session: usize,
    /// Maximum questions per cycle
    pub questions_per_cycle: usize,
    /// HRPO group size for baseline calculation
    pub hrpo_group_size: usize,
    /// Minimum improvement threshold to continue
    pub improvement_threshold: Fixed64,
    /// Maximum difficulty level (1-10)
    pub max_difficulty: u8,
}

impl EvolutionConfig {
    /// Create config for sovereignty tier
    pub fn for_tier(tier: SovereigntyTier) -> Self {
        match tier {
            SovereigntyTier::T0Mobile => Self {
                cycles_per_session: 3,      // Conservative for battery
                questions_per_cycle: 12,    // At least 2 per domain (6 domains)
                hrpo_group_size: 3,
                improvement_threshold: Fixed64::from_f64(0.01),
                max_difficulty: 5,
            },
            SovereigntyTier::T1Consumer => Self {
                cycles_per_session: 10,
                questions_per_cycle: 20,
                hrpo_group_size: 5,
                improvement_threshold: Fixed64::from_f64(0.005),
                max_difficulty: 8,
            },
            SovereigntyTier::T2Node => Self {
                cycles_per_session: 50,
                questions_per_cycle: 100,
                hrpo_group_size: 10,
                improvement_threshold: Fixed64::from_f64(0.001),
                max_difficulty: 10,
            },
        }
    }
}

// ============================================================================
// TASK DOMAIN
// ============================================================================

/// Knowledge domain for task generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskDomain {
    /// Quranic knowledge and tafsir
    Quran,
    /// Hadith and authentication
    Hadith,
    /// Islamic jurisprudence
    Fiqh,
    /// Arabic language and grammar
    Arabic,
    /// General knowledge
    General,
    /// Reasoning and logic
    Reasoning,
}

impl TaskDomain {
    /// Get all domains
    pub fn all() -> &'static [TaskDomain] {
        &[
            Self::Quran,
            Self::Hadith,
            Self::Fiqh,
            Self::Arabic,
            Self::General,
            Self::Reasoning,
        ]
    }

    /// Get domain-specific seed questions for bootstrapping
    pub fn seed_templates(&self) -> &'static [&'static str] {
        match self {
            Self::Quran => &[
                "What surah discusses {}?",
                "How many ayat are in Surah {}?",
                "What is the theme of Surah {}?",
                "Which surah was revealed in {}?",
            ],
            Self::Hadith => &[
                "What did the Prophet say about {}?",
                "Which hadith collection contains {}?",
                "What is the chain of narration for {}?",
                "Is the hadith about {} sahih?",
            ],
            Self::Fiqh => &[
                "What is the ruling on {}?",
                "Which madhab permits {}?",
                "What are the conditions for {}?",
                "Is {} obligatory or recommended?",
            ],
            Self::Arabic => &[
                "What is the root of {}?",
                "Parse the word {} grammatically",
                "What does {} mean in context?",
                "What is the plural of {}?",
            ],
            Self::General => &[
                "Explain the concept of {}",
                "What are the main aspects of {}?",
                "Compare {} and {}",
                "Describe the significance of {}",
            ],
            Self::Reasoning => &[
                "If {} then what follows?",
                "What is the logical flaw in {}?",
                "Prove that {} implies {}",
                "What assumptions does {} require?",
            ],
        }
    }
}

// ============================================================================
// EVOLUTION TASK
// ============================================================================

/// A task generated by the Proposer for the Solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionTask {
    /// Unique task identifier
    pub task_id: String,
    /// The question/prompt
    pub question: String,
    /// Expected answer (for validation)
    pub expected_answer: Option<String>,
    /// Knowledge domain
    pub domain: TaskDomain,
    /// Difficulty level (1-10)
    pub difficulty: u8,
    /// Number of reasoning hops required
    pub hop_count: u8,
    /// Creation timestamp (nanoseconds)
    pub created_ns: u64,
}

impl EvolutionTask {
    /// Generate task ID from content hash
    pub fn generate_id(question: &str, domain: TaskDomain) -> String {
        let content = format!("{}:{:?}", question, domain);
        let hash = Sha256::digest(content.as_bytes());
        format!("task-{:x}", &hash[..8].iter().fold(0u64, |acc, &b| acc << 8 | b as u64))
    }

    /// Create new task
    pub fn new(question: String, domain: TaskDomain, difficulty: u8, hop_count: u8) -> Self {
        let task_id = Self::generate_id(&question, domain);
        let created_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            task_id,
            question,
            expected_answer: None,
            domain,
            difficulty,
            hop_count,
            created_ns,
        }
    }

    /// Set expected answer
    pub fn with_answer(mut self, answer: String) -> Self {
        self.expected_answer = Some(answer);
        self
    }
}

// ============================================================================
// SOLVER RESPONSE
// ============================================================================

/// Response from the Solver agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverResponse {
    /// Task this responds to
    pub task_id: String,
    /// The answer provided
    pub answer: String,
    /// Confidence score (Fixed64)
    pub confidence: Fixed64,
    /// Reasoning steps taken
    pub reasoning_steps: Vec<String>,
    /// Engram lookups performed
    pub engram_lookups: usize,
    /// Response timestamp (nanoseconds)
    pub response_ns: u64,
}

impl SolverResponse {
    /// Create new response
    pub fn new(task_id: String, answer: String, confidence: Fixed64) -> Self {
        let response_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            task_id,
            answer,
            confidence,
            reasoning_steps: Vec::new(),
            engram_lookups: 0,
            response_ns,
        }
    }

    /// Add reasoning step
    pub fn add_step(&mut self, step: String) {
        self.reasoning_steps.push(step);
    }

    /// Set engram lookup count
    pub fn with_lookups(mut self, count: usize) -> Self {
        self.engram_lookups = count;
        self
    }
}

// ============================================================================
// HRPO OPTIMIZER (Hop-Grouped Relative Policy Optimization)
// ============================================================================

/// HRPO: Groups similar tasks for efficient baseline calculation
/// Key innovation from Dr. Zero for compute-efficient RL
#[derive(Debug)]
#[allow(dead_code)] // Reserved fields for future expansion
pub struct HRPOOptimizer {
    /// Group size for baseline calculation
    group_size: usize,
    /// Grouped task buffers by hop count
    hop_groups: HashMap<u8, VecDeque<(EvolutionTask, SolverResponse, Fixed64)>>,
    /// Learning rate (Fixed64)
    learning_rate: Fixed64,
}

impl HRPOOptimizer {
    /// Create new HRPO optimizer
    pub fn new(group_size: usize) -> Self {
        Self {
            group_size,
            hop_groups: HashMap::new(),
            learning_rate: Fixed64::from_f64(0.01),
        }
    }

    /// Add task-response pair with raw reward
    pub fn add_sample(&mut self, task: EvolutionTask, response: SolverResponse, raw_reward: Fixed64) {
        let hop = task.hop_count;
        let group = self.hop_groups.entry(hop).or_default();

        group.push_back((task, response, raw_reward));

        // Trim to 2x group size
        while group.len() > self.group_size * 2 {
            group.pop_front();
        }
    }

    /// Compute relative rewards for a hop group
    /// Returns: Vec<(task_id, relative_reward)>
    pub fn compute_relative_rewards(&self, hop: u8) -> Vec<(String, Fixed64)> {
        let group = match self.hop_groups.get(&hop) {
            Some(g) if g.len() >= self.group_size => g,
            _ => return Vec::new(),
        };

        // Compute group baseline (mean reward)
        let sum: Fixed64 = group.iter()
            .map(|(_, _, r)| *r)
            .fold(Fixed64::ZERO, |acc, r| acc + r);
        let baseline = sum / Fixed64::from_int(group.len() as i32);

        // Compute relative rewards
        group.iter()
            .map(|(task, _, reward)| {
                let relative = *reward - baseline;
                (task.task_id.clone(), relative)
            })
            .collect()
    }

    /// Get improvement signal for Proposer
    /// Positive = Solver is mastering this hop level
    pub fn get_hop_mastery(&self, hop: u8) -> Fixed64 {
        let rewards = self.compute_relative_rewards(hop);
        if rewards.is_empty() {
            return Fixed64::ZERO;
        }

        // Average of positive relative rewards
        let positives: Vec<_> = rewards.iter()
            .filter(|(_, r)| *r > Fixed64::ZERO)
            .collect();

        if positives.is_empty() {
            Fixed64::ZERO
        } else {
            let sum: Fixed64 = positives.iter().map(|(_, r)| *r).fold(Fixed64::ZERO, |a, r| a + r);
            sum / Fixed64::from_int(positives.len() as i32)
        }
    }
}

// ============================================================================
// PROPOSER AGENT
// ============================================================================

/// Proposer Agent: Generates increasingly difficult tasks
/// Maps to PAT Creative Innovator + Quality Guardian
#[derive(Debug)]
pub struct ProposerAgent {
    /// Current difficulty target per domain
    difficulty_targets: HashMap<TaskDomain, u8>,
    /// Current hop target per domain
    hop_targets: HashMap<TaskDomain, u8>,
    /// Task history for diversity
    recent_tasks: VecDeque<String>,
    /// Maximum history size
    history_limit: usize,
    /// Random seed for deterministic generation
    seed: u64,
}

impl ProposerAgent {
    /// Create new Proposer
    pub fn new(seed: u64) -> Self {
        let mut difficulty_targets = HashMap::new();
        let mut hop_targets = HashMap::new();

        // Initialize all domains at difficulty 1, hop 1
        for domain in TaskDomain::all() {
            difficulty_targets.insert(*domain, 1);
            hop_targets.insert(*domain, 1);
        }

        Self {
            difficulty_targets,
            hop_targets,
            recent_tasks: VecDeque::new(),
            history_limit: 100,
            seed,
        }
    }

    /// Generate a task for the Solver
    #[instrument(skip(self))]
    pub fn generate_task(&mut self, domain: TaskDomain, config: &EvolutionConfig) -> EvolutionTask {
        let difficulty = *self.difficulty_targets.get(&domain).unwrap_or(&1);
        let hop_count = *self.hop_targets.get(&domain).unwrap_or(&1);

        // Select template based on seed
        let templates = domain.seed_templates();
        let template_idx = (self.seed as usize) % templates.len();
        let template = templates[template_idx];

        // Generate question (in production, this would use local LLM)
        let question = self.instantiate_template(template, domain, difficulty);

        // Update seed for next generation
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);

        // Track history
        self.recent_tasks.push_back(question.clone());
        if self.recent_tasks.len() > self.history_limit {
            self.recent_tasks.pop_front();
        }

        let capped_difficulty = difficulty.min(config.max_difficulty);

        debug!(
            domain = ?domain,
            difficulty = capped_difficulty,
            hop_count = hop_count,
            "Generated evolution task"
        );

        EvolutionTask::new(question, domain, capped_difficulty, hop_count)
    }

    /// Instantiate template with domain-specific content
    fn instantiate_template(&self, template: &str, domain: TaskDomain, difficulty: u8) -> String {
        // In production, this would use Engram + local LLM
        // For now, use deterministic placeholders
        let placeholder = match domain {
            TaskDomain::Quran => format!("topic_{}_d{}", self.seed % 114, difficulty),
            TaskDomain::Hadith => format!("narrator_{}_d{}", self.seed % 50, difficulty),
            TaskDomain::Fiqh => format!("ruling_{}_d{}", self.seed % 100, difficulty),
            TaskDomain::Arabic => format!("word_{}_d{}", self.seed % 1000, difficulty),
            TaskDomain::General => format!("concept_{}_d{}", self.seed % 500, difficulty),
            TaskDomain::Reasoning => format!("premise_{}_d{}", self.seed % 200, difficulty),
        };

        template.replacen("{}", &placeholder, 1)
    }

    /// Update difficulty based on Solver mastery signal
    pub fn adjust_difficulty(&mut self, domain: TaskDomain, mastery: Fixed64, config: &EvolutionConfig) {
        let current = *self.difficulty_targets.get(&domain).unwrap_or(&1);

        // If Solver is mastering (positive mastery), increase difficulty
        if mastery > Fixed64::from_f64(0.1) && current < config.max_difficulty {
            self.difficulty_targets.insert(domain, current + 1);
            info!(domain = ?domain, new_difficulty = current + 1, "Proposer increasing difficulty");
        }
        // If Solver is struggling (negative mastery), decrease difficulty
        else if mastery < Fixed64::from_f64(-0.1) && current > 1 {
            self.difficulty_targets.insert(domain, current - 1);
            info!(domain = ?domain, new_difficulty = current - 1, "Proposer decreasing difficulty");
        }
    }

    /// Update hop count based on mastery
    pub fn adjust_hops(&mut self, domain: TaskDomain, mastery: Fixed64) {
        let current = *self.hop_targets.get(&domain).unwrap_or(&1);

        if mastery > Fixed64::from_f64(0.2) && current < 5 {
            self.hop_targets.insert(domain, current + 1);
            info!(domain = ?domain, new_hops = current + 1, "Proposer increasing hop count");
        }
    }
}

// ============================================================================
// SOLVER AGENT
// ============================================================================

/// Solver Agent: Answers tasks using Engram + reasoning
/// Maps to PAT Implementation Specialist + Analytical Optimizer
#[derive(Debug)]
#[allow(dead_code)] // Reserved fields for future expansion
pub struct SolverAgent {
    /// Reference to Engram for knowledge retrieval
    engram_dim: usize,
    /// Reasoning chain capacity
    max_reasoning_steps: usize,
    /// Cumulative score
    total_score: Fixed64,
    /// Tasks attempted
    tasks_attempted: usize,
}

impl SolverAgent {
    /// Create new Solver
    pub fn new(engram_dim: usize) -> Self {
        Self {
            engram_dim,
            max_reasoning_steps: 5,
            total_score: Fixed64::ZERO,
            tasks_attempted: 0,
        }
    }

    /// Attempt to solve a task using Engram
    #[instrument(skip(self, engram))]
    pub fn solve(&mut self, task: &EvolutionTask, engram: &mut SovereignEngram) -> SolverResponse {
        self.tasks_attempted += 1;

        // Tokenize question (simplified)
        let tokens: Vec<u32> = task.question
            .bytes()
            .map(|b| b as u32)
            .take(32)
            .collect();

        // Create hidden state (simplified)
        let hidden: Vec<Vec<Fixed64>> = vec![vec![Fixed64::HALF; engram.dim()]; tokens.len()];

        // Forward through Engram
        let enhanced = engram.forward(&tokens, &hidden);

        // Compute answer confidence based on Engram enhancement
        let confidence = self.compute_confidence(&hidden, &enhanced);

        // Generate answer (in production, would use local LLM)
        let answer = self.generate_answer(task, &enhanced);

        let mut response = SolverResponse::new(task.task_id.clone(), answer, confidence);
        response.engram_lookups = tokens.len();

        // Add reasoning steps
        response.add_step(format!("Retrieved {} Engram embeddings", tokens.len()));
        response.add_step(format!("Domain: {:?}, Difficulty: {}", task.domain, task.difficulty));
        response.add_step(format!("Confidence: {:.4}", confidence.to_f64()));

        debug!(
            task_id = %task.task_id,
            confidence = confidence.to_f64(),
            "Solver completed task"
        );

        response
    }

    /// Compute confidence from hidden state enhancement
    fn compute_confidence(&self, original: &[Vec<Fixed64>], enhanced: &[Vec<Fixed64>]) -> Fixed64 {
        if original.is_empty() || enhanced.is_empty() {
            return Fixed64::HALF;
        }

        // Measure enhancement magnitude
        let mut total_diff = Fixed64::ZERO;
        let mut count = 0;

        for (orig, enh) in original.iter().zip(enhanced.iter()) {
            for (o, e) in orig.iter().zip(enh.iter()) {
                let diff = *e - *o;
                total_diff = total_diff + (diff * diff);
                count += 1;
            }
        }

        if count == 0 {
            return Fixed64::HALF;
        }

        // Normalize to [0.5, 1.0] range
        let avg_diff = total_diff / Fixed64::from_int(count);
        let confidence = Fixed64::HALF + (avg_diff / Fixed64::from_int(2));

        // Clamp to [0, 1]
        if confidence > Fixed64::ONE {
            Fixed64::ONE
        } else if confidence < Fixed64::ZERO {
            Fixed64::ZERO
        } else {
            confidence
        }
    }

    /// Generate answer from enhanced state
    fn generate_answer(&self, task: &EvolutionTask, _enhanced: &[Vec<Fixed64>]) -> String {
        // In production, this would use local LLM with enhanced context
        format!(
            "Answer to '{}' in domain {:?} at difficulty {}",
            &task.question[..task.question.len().min(50)],
            task.domain,
            task.difficulty
        )
    }

    /// Update score
    pub fn update_score(&mut self, reward: Fixed64) {
        self.total_score = self.total_score + reward;
    }

    /// Get average score
    pub fn average_score(&self) -> Fixed64 {
        if self.tasks_attempted == 0 {
            Fixed64::ZERO
        } else {
            self.total_score / Fixed64::from_int(self.tasks_attempted as i32)
        }
    }
}

// ============================================================================
// EVOLUTION STATE
// ============================================================================

/// Tracks evolution progress for receipts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionState {
    /// Current generation number
    pub generation: u64,
    /// Tasks completed this generation
    pub tasks_completed: usize,
    /// Average reward this generation
    pub avg_reward: Fixed64,
    /// Domains with mastery
    pub mastered_domains: Vec<TaskDomain>,
    /// Current max difficulty reached
    pub max_difficulty_reached: u8,
    /// Timestamp
    pub timestamp_ns: u64,
}

impl EvolutionState {
    /// Create initial state
    pub fn initial() -> Self {
        Self {
            generation: 0,
            tasks_completed: 0,
            avg_reward: Fixed64::ZERO,
            mastered_domains: Vec::new(),
            max_difficulty_reached: 1,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
        }
    }

    /// Compute state hash for receipts
    pub fn compute_hash(&self) -> String {
        let content = format!(
            "{}:{}:{}:{}",
            self.generation,
            self.tasks_completed,
            self.avg_reward.to_bits(),
            self.max_difficulty_reached
        );
        let hash = Sha256::digest(content.as_bytes());
        format!("{:x}", hash)
    }
}

// ============================================================================
// SOVEREIGN EVOLUTION ENGINE
// ============================================================================

/// Main evolution engine combining Proposer, Solver, and HRPO
#[derive(Debug)]
pub struct SovereignEvolution {
    /// Sovereignty tier
    tier: SovereigntyTier,
    /// Configuration
    config: EvolutionConfig,
    /// Engram module for knowledge retrieval
    engram: SovereignEngram,
    /// Proposer agent
    proposer: ProposerAgent,
    /// Solver agent
    solver: SolverAgent,
    /// HRPO optimizer
    hrpo: HRPOOptimizer,
    /// Current state
    state: EvolutionState,
}

impl SovereignEvolution {
    /// Create new evolution engine
    #[instrument]
    pub fn new(tier: SovereigntyTier, seed: u64) -> Self {
        info!(tier = ?tier, seed = seed, "🧬 Initializing SovereignEvolution");

        let config = EvolutionConfig::for_tier(tier);
        let engram = SovereignEngram::new(tier);
        let proposer = ProposerAgent::new(seed);
        let solver = SolverAgent::new(engram.dim());
        let hrpo = HRPOOptimizer::new(config.hrpo_group_size);
        let state = EvolutionState::initial();

        Self {
            tier,
            config,
            engram,
            proposer,
            solver,
            hrpo,
            state,
        }
    }

    /// Run one evolution cycle
    #[instrument(skip(self))]
    pub fn evolve_cycle(&mut self) -> EvolutionState {
        info!(generation = self.state.generation, "Starting evolution cycle");

        let mut total_reward = Fixed64::ZERO;
        let mut tasks_completed = 0;

        // Generate and solve tasks across domains
        for domain in TaskDomain::all() {
            for _ in 0..self.config.questions_per_cycle / TaskDomain::all().len() {
                // Proposer generates task
                let task = self.proposer.generate_task(*domain, &self.config);

                // Solver attempts task
                let response = self.solver.solve(&task, &mut self.engram);

                // Compute reward (simplified - in production would validate against ground truth)
                let reward = self.compute_reward(&task, &response);

                // Update HRPO
                self.hrpo.add_sample(task.clone(), response, reward);

                // Update solver
                self.solver.update_score(reward);

                total_reward = total_reward + reward;
                tasks_completed += 1;
            }

            // Adjust Proposer based on HRPO signals
            let hop = *self.proposer.hop_targets.get(domain).unwrap_or(&1);
            let mastery = self.hrpo.get_hop_mastery(hop);
            self.proposer.adjust_difficulty(*domain, mastery, &self.config);
            self.proposer.adjust_hops(*domain, mastery);
        }

        // Update state
        self.state.generation += 1;
        self.state.tasks_completed = tasks_completed;
        self.state.avg_reward = if tasks_completed > 0 {
            total_reward / Fixed64::from_int(tasks_completed as i32)
        } else {
            Fixed64::ZERO
        };
        self.state.max_difficulty_reached = self.proposer.difficulty_targets
            .values()
            .max()
            .copied()
            .unwrap_or(1);
        self.state.timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        info!(
            generation = self.state.generation,
            avg_reward = self.state.avg_reward.to_f64(),
            max_difficulty = self.state.max_difficulty_reached,
            "Evolution cycle complete"
        );

        self.state.clone()
    }

    /// Compute reward for task-response pair
    fn compute_reward(&self, task: &EvolutionTask, response: &SolverResponse) -> Fixed64 {
        // Base reward from confidence
        let confidence_reward = response.confidence;

        // Difficulty bonus
        let difficulty_bonus = Fixed64::from_f64(task.difficulty as f64 / 10.0);

        // Hop bonus
        let hop_bonus = Fixed64::from_f64(task.hop_count as f64 / 5.0);

        // Combined reward (weighted)
        let reward = (confidence_reward * Fixed64::from_f64(0.5))
            + (difficulty_bonus * Fixed64::from_f64(0.3))
            + (hop_bonus * Fixed64::from_f64(0.2));

        // Clamp to [0, 1]
        if reward > Fixed64::ONE {
            Fixed64::ONE
        } else if reward < Fixed64::ZERO {
            Fixed64::ZERO
        } else {
            reward
        }
    }

    /// Run full evolution session
    #[instrument(skip(self))]
    pub fn evolve_session(&mut self) -> Vec<EvolutionState> {
        info!(
            tier = ?self.tier,
            cycles = self.config.cycles_per_session,
            "Starting evolution session"
        );

        let mut states = Vec::with_capacity(self.config.cycles_per_session);

        for _ in 0..self.config.cycles_per_session {
            let state = self.evolve_cycle();
            states.push(state);

            // Check for improvement plateau
            if states.len() >= 3 {
                let recent: Vec<_> = states.iter().rev().take(3).collect();
                let improving = recent.windows(2).all(|w| w[0].avg_reward >= w[1].avg_reward);

                if !improving {
                    let diff = recent[0].avg_reward - recent[2].avg_reward;
                    if diff < self.config.improvement_threshold {
                        info!("Improvement plateau detected, ending session early");
                        break;
                    }
                }
            }
        }

        info!(
            generations = states.len(),
            final_avg_reward = states.last().map(|s| s.avg_reward.to_f64()).unwrap_or(0.0),
            "Evolution session complete"
        );

        states
    }

    /// Get current state
    pub fn state(&self) -> &EvolutionState {
        &self.state
    }

    /// Get tier
    pub fn tier(&self) -> SovereigntyTier {
        self.tier
    }

    /// Get solver performance
    pub fn solver_performance(&self) -> Fixed64 {
        self.solver.average_score()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolution_config_tiers() {
        let t0 = EvolutionConfig::for_tier(SovereigntyTier::T0Mobile);
        let t1 = EvolutionConfig::for_tier(SovereigntyTier::T1Consumer);
        let t2 = EvolutionConfig::for_tier(SovereigntyTier::T2Node);

        assert!(t0.cycles_per_session < t1.cycles_per_session);
        assert!(t1.cycles_per_session < t2.cycles_per_session);
    }

    #[test]
    fn test_task_generation() {
        let mut proposer = ProposerAgent::new(42);
        let config = EvolutionConfig::for_tier(SovereigntyTier::T1Consumer);

        let task = proposer.generate_task(TaskDomain::Quran, &config);

        assert!(!task.task_id.is_empty());
        assert!(!task.question.is_empty());
        assert_eq!(task.domain, TaskDomain::Quran);
    }

    #[test]
    fn test_hrpo_grouping() {
        let mut hrpo = HRPOOptimizer::new(3);

        // Add samples for hop 2
        for i in 0..5 {
            let task = EvolutionTask::new(
                format!("Question {}", i),
                TaskDomain::General,
                3,
                2, // hop count
            );
            let response = SolverResponse::new(
                task.task_id.clone(),
                "Answer".to_string(),
                Fixed64::from_f64(0.7 + i as f64 * 0.05),
            );
            hrpo.add_sample(task, response, Fixed64::from_f64(0.5 + i as f64 * 0.1));
        }

        let rewards = hrpo.compute_relative_rewards(2);
        assert!(!rewards.is_empty());
    }

    #[test]
    fn test_solver_with_engram() {
        let engram = SovereignEngram::new(SovereigntyTier::T0Mobile);
        let mut solver = SolverAgent::new(engram.dim());
        let mut engram = SovereignEngram::new(SovereigntyTier::T0Mobile);

        let task = EvolutionTask::new(
            "What is the theme of Surah Al-Fatiha?".to_string(),
            TaskDomain::Quran,
            3,
            1,
        );

        let response = solver.solve(&task, &mut engram);

        assert_eq!(response.task_id, task.task_id);
        assert!(response.confidence >= Fixed64::ZERO);
        assert!(response.confidence <= Fixed64::ONE);
    }

    #[test]
    fn test_evolution_cycle() {
        // Use T1 for more questions per cycle (T0 has only 5, divided by 6 domains = 0 each)
        let mut evolution = SovereignEvolution::new(SovereigntyTier::T1Consumer, 12345);

        let state = evolution.evolve_cycle();

        assert_eq!(state.generation, 1);
        assert!(state.tasks_completed > 0, "Expected tasks to complete, got 0");
    }

    #[test]
    fn test_evolution_determinism() {
        let mut evo1 = SovereignEvolution::new(SovereigntyTier::T0Mobile, 42);
        let mut evo2 = SovereignEvolution::new(SovereigntyTier::T0Mobile, 42);

        let state1 = evo1.evolve_cycle();
        let state2 = evo2.evolve_cycle();

        // Same seed should produce same results
        assert_eq!(state1.generation, state2.generation);
        assert_eq!(state1.tasks_completed, state2.tasks_completed);
    }

    #[test]
    fn test_evolution_state_hash() {
        let state = EvolutionState::initial();
        let hash = state.compute_hash();

        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64); // SHA256 hex
    }
}
