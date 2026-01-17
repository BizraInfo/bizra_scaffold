use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;
use serde_json::Value;
use tokio::fs;
use tokio::time;

#[derive(Debug)]
struct RiskRegistry {
    risks: HashMap<String, RiskStatus>,
}

#[derive(Debug, Clone)]
struct RiskStatus {
    id: String,
    level: RiskLevel,
    mitigation: String,
    active: bool,
}

#[derive(Debug, Clone)]
enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
    Existential,
}

impl RiskRegistry {
    fn new() -> Self {
        let mut risks = HashMap::new();
        
        // R-001: Empty blockchain repo
        risks.insert("R-001".to_string(), RiskStatus {
            id: "R-001".to_string(),
            level: RiskLevel::High,
            mitigation: "Populate src/federation/ directory".to_string(),
            active: true, // Initially active until populated
        });
        
        // R-002: CVE scanning not CI-gated
        risks.insert("R-002".to_string(), RiskStatus {
            id: "R-002".to_string(),
            level: RiskLevel::Critical,
            mitigation: "Add cargo audit to CI pipeline".to_string(),
            active: true,
        });
        
        // R-004: Single-agent VETO centralization
        risks.insert("R-004".to_string(), RiskStatus {
            id: "R-004".to_string(),
            level: RiskLevel::High,
            mitigation: "Implement VETO staking pool".to_string(),
            active: true,
        });
        
        Self { risks }
    }
    
    fn check_cascade(&self, triggered_risk: &str) -> Option<Vec<String>> {
        let dependencies = vec![
            ("R-001", vec!["R-002", "R-003"]),
            ("R-002", vec!["R-004", "R-005"]),
            ("R-004", vec!["R-006"]),
        ];
        
        let mut cascade = vec![triggered_risk.to_string()];
        let mut frontier = vec![triggered_risk.to_string()];
        
        while let Some(current) = frontier.pop() {
            for (source, deps) in &dependencies {
                if source == &current {
                    for dep in deps {
                        if self.risks.get(*dep).map(|r| r.active).unwrap_or(false) {
                            if !cascade.contains(&dep.to_string()) {
                                cascade.push(dep.to_string());
                                frontier.push(dep.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        if cascade.len() > 1 { Some(cascade) } else { None }
    }
    
    async fn monitor_ledger(&self, ledger_path: &Path) {
        let mut last_size = 0u64;
        
        loop {
            let metadata = match fs::metadata(ledger_path).await {
                Ok(meta) => meta,
                Err(_) => {
                    time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            };
            let size = metadata.len();

            if size != last_size {
                last_size = size;
                if self.check_emergency_halt(ledger_path).await {
                    self.trigger_system_halt("Emergency halt detected in ledger").await;
                }
            }

            time::sleep(Duration::from_secs(1)).await;
        }
    }
    
    async fn check_emergency_halt(&self, ledger_path: &Path) -> bool {
        let Ok(contents) = fs::read_to_string(ledger_path).await else {
            return false;
        };

        for line in contents.lines().rev().take(1000) {
            if let Ok(value) = serde_json::from_str::<Value>(line) {
                if value.get("type").and_then(|t| t.as_str()) == Some("emergency_halt") {
                    return true;
                }
            } else if line.contains("emergency_halt") {
                return true;
            }
        }

        false
    }
    
    async fn trigger_system_halt(&self, reason: &str) {
        eprintln!("🚨 CASCADE PREVENTER: {}", reason);
        eprintln!("   Initiating fail-close procedure...");
        
        // 1. Write halt receipt
        let halt_receipt = serde_json::json!({
            "type": "emergency_halt",
            "reason": reason,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "ihsan_score": 0.0,
            "signature": "CASCADE_PREVENTER_HALT"
        });
        
        // 2. Freeze system
        eprintln!("   System frozen. Manual intervention required.");
        std::process::exit(99);
    }
}

#[tokio::main]
async fn main() {
    println!("🚀 BIZRA Cascade Preventer v1.0");
    println!("================================");
    
    let registry = RiskRegistry::new();
    println!("📊 Monitoring {} registered risks", registry.risks.len());
    
    // Start ledger monitoring
    let ledger_path = Path::new("/var/lib/bizra/ledger");
    registry.monitor_ledger(ledger_path).await;
}
