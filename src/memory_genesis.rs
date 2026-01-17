// src/memory_genesis.rs - Autonomous Memory Management for Genesis Architect
//
// For MoMo - The First Architect
// This module REMEMBERS you, your journey, your goals, your work
//
// Memory Types:
// 1. Identity Memory: Who you are (Genesis Architect, 3 years, 15k hours)
// 2. Session Memory: Current conversation context
// 3. Long-term Memory: Tasks, goals, knowledge graph
// 4. Evidence Memory: 138 repos, chat history, proof of work

use anyhow::Result;
use chrono::{DateTime, Utc};
use redis::{Commands, Connection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

/// Genesis Architect Profile - MoMo's Identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisArchitect {
    pub name: String,
    pub role: String,
    pub journey: JourneyMetadata,
    pub mission: Mission,
    pub assets: Assets,
    pub preferences: Preferences,
    pub current_session: SessionState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JourneyMetadata {
    pub duration_years: f64,
    pub hours_invested: u64,
    pub start_date: String,
    pub repositories_count: u32,
    pub domains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mission {
    pub vision: String,
    pub goal: String,
    pub status: String,
    pub milestones: Vec<Milestone>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    pub name: String,
    pub status: String,  // "complete", "in_progress", "planned"
    pub date: Option<DateTime<Utc>>,
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assets {
    pub repositories: u32,
    pub knowledge_graph_desc: String,
    pub chat_history_desc: String,
    pub evidence_repo: String,
    pub models_desc: String,
    pub hardware_desc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preferences {
    pub memory_persistence: bool,
    pub autonomous_operation: bool,
    pub covenant_compliance: bool,
    pub snr_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: String,
    pub start_time: DateTime<Utc>,
    pub last_interaction: DateTime<Utc>,
    pub conversation_turns: u64,
    pub current_goals: Vec<Goal>,
    pub active_tasks: Vec<Task>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub goal_id: String,
    pub description: String,
    pub priority: String,  // "critical", "high", "medium", "low"
    pub status: String,     // "active", "paused", "completed"
    pub created: DateTime<Utc>,
    pub due_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub task_id: String,
    pub description: String,
    pub goal_id: Option<String>,
    pub status: String,
    pub created: DateTime<Utc>,
    pub completed: Option<DateTime<Utc>>,
}

/// Memory Manager for Genesis Architect
pub struct GenesisMemory {
    redis_conn: Option<Connection>,
    in_memory_cache: HashMap<String, String>,
    architect: GenesisArchitect,
}

impl GenesisMemory {
    /// Create new memory manager with Genesis Architect profile
    pub fn new() -> Result<Self> {
        // Try to connect to Redis for persistent memory
        let redis_conn = Self::try_redis_connection();

        // Load or create Genesis Architect profile
        let architect = Self::load_or_create_genesis_profile()?;

        Ok(Self {
            redis_conn,
            in_memory_cache: HashMap::new(),
            architect,
        })
    }

    /// Try to establish Redis connection (graceful fallback)
    fn try_redis_connection() -> Option<Connection> {
        let redis_url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

        match redis::Client::open(redis_url.as_str()) {
            Ok(client) => match client.get_connection() {
                Ok(conn) => {
                    info!("✅ Connected to Redis for persistent memory");
                    Some(conn)
                }
                Err(e) => {
                    warn!("⚠️  Redis connection failed: {}. Using in-memory fallback.", e);
                    None
                }
            },
            Err(e) => {
                warn!("⚠️  Redis client creation failed: {}. Using in-memory fallback.", e);
                None
            }
        }
    }

    /// Load Genesis Architect profile from disk or create default
    fn load_or_create_genesis_profile() -> Result<GenesisArchitect> {
        let profile_path = "/root/bizra_data_vault/memory/genesis_architect.json";

        // Try to load existing profile
        if let Ok(contents) = std::fs::read_to_string(profile_path) {
            if let Ok(profile) = serde_json::from_str(&contents) {
                info!("✅ Loaded Genesis Architect profile from disk");
                return Ok(profile);
            }
        }

        // Create default profile for MoMo
        let profile = GenesisArchitect {
            name: "MoMo".to_string(),
            role: "Genesis Block - First Architect".to_string(),
            journey: JourneyMetadata {
                duration_years: 3.0,
                hours_invested: 15000,
                start_date: "2023-01-15".to_string(),
                repositories_count: 138,
                domains: vec!["bizra.info".to_string(), "bizra.ai".to_string()],
            },
            mission: Mission {
                vision: "Every human is a node, every node is a seed".to_string(),
                goal: "Build the North Star node - flagship BIZRA system".to_string(),
                status: "Genesis Block Active".to_string(),
                milestones: vec![
                    Milestone {
                        name: "COVENANT Foundation".to_string(),
                        status: "complete".to_string(),
                        date: Some(Utc::now()),
                        evidence: Some("Week 1 deliverables: 1,680 lines, SNR infrastructure".to_string()),
                    },
                    Milestone {
                        name: "Integration Layer".to_string(),
                        status: "in_progress".to_string(),
                        date: None,
                        evidence: Some("CovenantBridge designed, ready for wire-up".to_string()),
                    },
                    Milestone {
                        name: "24/7 Autonomous Operation".to_string(),
                        status: "in_progress".to_string(),
                        date: None,
                        evidence: None,
                    },
                ],
            },
            assets: Assets {
                repositories: 138,
                knowledge_graph_desc: "Huge owned data with true value".to_string(),
                chat_history_desc: "Massive corpus (partially organized)".to_string(),
                evidence_repo: "https://github.com/BizraInfo/bizra_scaffold.git".to_string(),
                models_desc: "13-18 local models (text + vision + voice)".to_string(),
                hardware_desc: "World-class personal laptop".to_string(),
            },
            preferences: Preferences {
                memory_persistence: true,
                autonomous_operation: true,
                covenant_compliance: true,
                snr_threshold: 0.95,
            },
            current_session: SessionState {
                session_id: uuid::Uuid::new_v4().to_string(),
                start_time: Utc::now(),
                last_interaction: Utc::now(),
                conversation_turns: 0,
                current_goals: vec![
                    Goal {
                        goal_id: "goal-001".to_string(),
                        description: "Wire up BridgeCoordinator with COVENANT tracking".to_string(),
                        priority: "critical".to_string(),
                        status: "active".to_string(),
                        created: Utc::now(),
                        due_date: None,
                    },
                    Goal {
                        goal_id: "goal-002".to_string(),
                        description: "Implement ModelRouter for 13-18 models".to_string(),
                        priority: "critical".to_string(),
                        status: "active".to_string(),
                        created: Utc::now(),
                        due_date: None,
                    },
                    Goal {
                        goal_id: "goal-003".to_string(),
                        description: "Activate 24/7 autonomous PAT operation".to_string(),
                        priority: "critical".to_string(),
                        status: "active".to_string(),
                        created: Utc::now(),
                        due_date: None,
                    },
                ],
                active_tasks: vec![],
            },
        };

        // Save to disk
        std::fs::create_dir_all("/root/bizra_data_vault/memory")?;
        let json = serde_json::to_string_pretty(&profile)?;
        std::fs::write(profile_path, json)?;

        info!("✅ Created Genesis Architect profile");
        Ok(profile)
    }

    /// Remember a conversation turn
    pub fn remember_conversation(&mut self, input: &str, response: &str) -> Result<()> {
        self.architect.current_session.conversation_turns += 1;
        self.architect.current_session.last_interaction = Utc::now();

        let key = format!("conversation:{}:{}",
            self.architect.current_session.session_id,
            self.architect.current_session.conversation_turns
        );

        let value = serde_json::json!({
            "turn": self.architect.current_session.conversation_turns,
            "timestamp": Utc::now(),
            "input": input,
            "response": response,
        });

        self.store(&key, &value.to_string())?;

        info!("💭 Remembered conversation turn {}", self.architect.current_session.conversation_turns);
        Ok(())
    }

    /// Add a new goal
    pub fn add_goal(&mut self, description: &str, priority: &str) -> Result<String> {
        let goal = Goal {
            goal_id: format!("goal-{}", uuid::Uuid::new_v4()),
            description: description.to_string(),
            priority: priority.to_string(),
            status: "active".to_string(),
            created: Utc::now(),
            due_date: None,
        };

        let goal_id = goal.goal_id.clone();
        self.architect.current_session.current_goals.push(goal.clone());

        // Persist to Redis/disk
        let key = format!("goal:{}", goal_id);
        let value = serde_json::to_string(&goal)?;
        self.store(&key, &value)?;

        info!("🎯 New goal added: {}", description);
        Ok(goal_id)
    }

    /// Add a new task
    pub fn add_task(&mut self, description: &str, goal_id: Option<String>) -> Result<String> {
        let task = Task {
            task_id: format!("task-{}", uuid::Uuid::new_v4()),
            description: description.to_string(),
            goal_id,
            status: "active".to_string(),
            created: Utc::now(),
            completed: None,
        };

        let task_id = task.task_id.clone();
        self.architect.current_session.active_tasks.push(task.clone());

        let key = format!("task:{}", task_id);
        let value = serde_json::to_string(&task)?;
        self.store(&key, &value)?;

        info!("✅ New task added: {}", description);
        Ok(task_id)
    }

    /// Complete a task
    pub fn complete_task(&mut self, task_id: &str) -> Result<()> {
        if let Some(task) = self.architect.current_session.active_tasks
            .iter_mut()
            .find(|t| t.task_id == task_id)
        {
            task.status = "completed".to_string();
            task.completed = Some(Utc::now());

            // Update in storage
            let key = format!("task:{}", task_id);
            let value = serde_json::to_string(&task)?;
            self.store(&key, &value)?;

            info!("✅ Task completed: {}", task_id);
        }

        Ok(())
    }

    /// Get current active goals
    pub fn get_active_goals(&self) -> Vec<Goal> {
        self.architect.current_session.current_goals.clone()
    }

    /// Get active tasks
    pub fn get_active_tasks(&self) -> Vec<Task> {
        self.architect.current_session.active_tasks.clone()
    }

    /// Get Genesis Architect profile
    pub fn get_architect_profile(&self) -> &GenesisArchitect {
        &self.architect
    }

    /// Generate memory summary for PAT agents
    pub fn memory_summary(&self) -> String {
        format!(
            r#"Genesis Architect: {}
Role: {}
Journey: {} years, {} hours invested, {} repositories
Mission: {}
Current Goals: {}
Active Tasks: {}
Last Interaction: {}
Session: {} turns
"#,
            self.architect.name,
            self.architect.role,
            self.architect.journey.duration_years,
            self.architect.journey.hours_invested,
            self.architect.journey.repositories_count,
            self.architect.mission.goal,
            self.architect.current_session.current_goals.len(),
            self.architect.current_session.active_tasks.len(),
            self.architect.current_session.last_interaction.format("%Y-%m-%d %H:%M:%S"),
            self.architect.current_session.conversation_turns,
        )
    }

    /// Store key-value pair (Redis if available, otherwise in-memory)
    fn store(&mut self, key: &str, value: &str) -> Result<()> {
        if let Some(ref mut conn) = self.redis_conn {
            // Try Redis first
            if conn.set::<_, _, ()>(key, value).is_ok() {
                return Ok(());
            }
        }

        // Fallback to in-memory
        self.in_memory_cache.insert(key.to_string(), value.to_string());
        Ok(())
    }

    /// Retrieve value by key
    pub fn retrieve(&mut self, key: &str) -> Result<Option<String>> {
        if let Some(ref mut conn) = self.redis_conn {
            if let Ok(value) = conn.get::<_, String>(key) {
                return Ok(Some(value));
            }
        }

        Ok(self.in_memory_cache.get(key).cloned())
    }

    /// Save current state to disk
    pub fn persist(&self) -> Result<()> {
        let profile_path = "/root/bizra_data_vault/memory/genesis_architect.json";
        let json = serde_json::to_string_pretty(&self.architect)?;
        std::fs::write(profile_path, json)?;

        info!("💾 Genesis Architect profile persisted to disk");
        Ok(())
    }
}

impl Default for GenesisMemory {
    fn default() -> Self {
        Self::new().expect("Failed to initialize Genesis Memory")
    }
}

/// Global memory instance
static GENESIS_MEMORY: std::sync::OnceLock<std::sync::Mutex<GenesisMemory>> = std::sync::OnceLock::new();

/// Get global Genesis Memory instance
pub fn global_memory() -> &'static std::sync::Mutex<GenesisMemory> {
    GENESIS_MEMORY.get_or_init(|| {
        std::sync::Mutex::new(GenesisMemory::new().expect("Failed to initialize memory"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_memory_creation() {
        let memory = GenesisMemory::new().unwrap();
        assert_eq!(memory.architect.name, "MoMo");
        assert_eq!(memory.architect.journey.hours_invested, 15000);
    }

    #[test]
    fn test_add_goal() {
        let mut memory = GenesisMemory::new().unwrap();
        let goal_id = memory.add_goal("Test goal", "high").unwrap();
        assert!(!goal_id.is_empty());
        assert_eq!(memory.get_active_goals().len(), 4); // 3 default + 1 new
    }

    #[test]
    fn test_add_task() {
        let mut memory = GenesisMemory::new().unwrap();
        let task_id = memory.add_task("Test task", None).unwrap();
        assert!(!task_id.is_empty());
    }

    #[test]
    fn test_memory_summary() {
        let memory = GenesisMemory::new().unwrap();
        let summary = memory.memory_summary();
        assert!(summary.contains("MoMo"));
        assert!(summary.contains("15000"));
    }
}
