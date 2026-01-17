// tests/beg_gates.rs
// BIZRA Entry Gates - CI verification for Giants Protocol compliance
// These tests MUST pass before any merge to main

use meta_alpha_dual_agentic::hookchain::{
    CapabilityTier, CapabilityToken, ConsentClass, ExecutedReceipt, HookDecision, PostHookResult,
    ReceiptDraft, SATHookChain, SessionDAG,
};
use meta_alpha_dual_agentic::tpm::SoftwareSigner;
use std::sync::Arc;

// ============================================================================
// BEG-01: Capability Token Rules
// ============================================================================

mod capability_token_gates {
    use super::*;

    #[test]
    fn beg01_token_must_have_tool_id() {
        let token = CapabilityToken::new("file_read", "read", CapabilityTier::T1Consumer);
        assert!(!token.tool_id.is_empty(), "BEG-01: Token must have tool_id");
    }

    #[test]
    fn beg01_token_must_have_scope() {
        let token = CapabilityToken::new("file_read", "read", CapabilityTier::T1Consumer);
        assert!(!token.scope.is_empty(), "BEG-01: Token must have scope");
    }

    #[test]
    fn beg01_token_must_have_tier() {
        let token = CapabilityToken::new("file_read", "read", CapabilityTier::T0Mobile);
        assert_eq!(
            token.tier,
            CapabilityTier::T0Mobile,
            "BEG-01: Token must preserve tier"
        );
    }

    #[test]
    fn beg01_token_must_have_budget() {
        let token = CapabilityToken::new("file_read", "read", CapabilityTier::T1Consumer);
        assert!(
            token.budget.max_tokens > 0,
            "BEG-01: Token must have positive max_tokens"
        );
        assert!(
            token.budget.max_time_ms > 0,
            "BEG-01: Token must have positive max_time_ms"
        );
    }

    #[test]
    fn beg01_token_must_have_expiry() {
        let token = CapabilityToken::new("file_read", "read", CapabilityTier::T1Consumer);
        assert!(
            token.expiry_ns > token.created_at,
            "BEG-01: Token expiry must be after creation"
        );
    }

    #[test]
    fn beg01_unsigned_token_is_invalid() {
        let token = CapabilityToken::new("file_read", "read", CapabilityTier::T1Consumer);
        assert!(!token.is_valid(), "BEG-01: Unsigned token must be invalid");
    }

    #[tokio::test]
    async fn beg01_signed_token_is_valid() {
        let signer = Arc::new(SoftwareSigner::new());
        let mut token = CapabilityToken::new("file_read", "read", CapabilityTier::T1Consumer);
        token.sign(signer.as_ref()).await.unwrap();

        assert!(token.is_valid(), "BEG-01: Signed token must be valid");
    }

    #[test]
    fn beg01_tier_budget_defaults_are_reasonable() {
        // T0 Mobile: Strict budgets
        let t0 = CapabilityTier::T0Mobile.default_budget();
        assert!(t0.max_tokens <= 2048, "BEG-01: T0 max_tokens must be ≤2048");
        assert!(
            t0.max_time_ms <= 10000,
            "BEG-01: T0 max_time_ms must be ≤10s"
        );
        assert!(!t0.offload_allowed, "BEG-01: T0 must not allow offload");

        // T1 Consumer: Moderate budgets
        let t1 = CapabilityTier::T1Consumer.default_budget();
        assert!(t1.max_tokens >= 4096, "BEG-01: T1 max_tokens must be ≥4096");
        assert!(!t1.offload_allowed, "BEG-01: T1 must not allow offload");

        // T2 Pro: Expanded budgets
        let t2 = CapabilityTier::T2Pro.default_budget();
        assert!(
            t2.max_tokens >= 16384,
            "BEG-01: T2 max_tokens must be ≥16384"
        );
        assert!(t2.offload_allowed, "BEG-01: T2 may allow offload");
    }
}

// ============================================================================
// BEG-02: Hook Coverage Tests
// ============================================================================

mod hook_coverage_gates {
    use super::*;

    #[tokio::test]
    async fn beg02_blocked_tools_are_denied() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer, "v1.0.0");

        let blocked_tools = ["sudo", "rm", "chmod", "eval", "exec"];

        for tool in blocked_tools {
            let draft = ReceiptDraft {
                tool_id: tool.to_string(),
                input: "test".to_string(),
                capability_token: None,
                session_node: "genesis".to_string(),
                timestamp: 0,
            };

            let decision = hook_chain.pre_capability_use(&draft).await.unwrap();
            match decision {
                HookDecision::Deny { code, .. } => {
                    assert_eq!(code, "BLOCKED_TOOL", "BEG-02: {} must be blocked", tool);
                }
                _ => panic!("BEG-02: Tool '{}' must be denied", tool),
            }
        }
    }

    #[tokio::test]
    async fn beg02_sql_injection_is_denied() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer, "v1.0.0");

        let sql_injections = [
            "DROP TABLE users",
            "'; -- comment",
            "SELECT * FROM users WHERE 1=1; DROP TABLE users;",
        ];

        for injection in sql_injections {
            let draft = ReceiptDraft {
                tool_id: "text_process".to_string(),
                input: injection.to_string(),
                capability_token: None,
                session_node: "genesis".to_string(),
                timestamp: 0,
            };

            let decision = hook_chain.pre_capability_use(&draft).await.unwrap();
            match decision {
                HookDecision::Deny { code, .. } => {
                    assert_eq!(
                        code, "SECURITY_THREAT",
                        "BEG-02: SQL injection must be blocked"
                    );
                }
                _ => panic!("BEG-02: SQL injection '{}' must be denied", injection),
            }
        }
    }

    #[tokio::test]
    async fn beg02_xss_is_denied() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer, "v1.0.0");

        let draft = ReceiptDraft {
            tool_id: "html_render".to_string(),
            input: "<script>alert('xss')</script>".to_string(),
            capability_token: None,
            session_node: "genesis".to_string(),
            timestamp: 0,
        };

        let decision = hook_chain.pre_capability_use(&draft).await.unwrap();
        match decision {
            HookDecision::Deny { code, .. } => {
                assert_eq!(code, "SECURITY_THREAT", "BEG-02: XSS must be blocked");
            }
            _ => panic!("BEG-02: XSS must be denied"),
        }
    }

    #[tokio::test]
    async fn beg02_safe_input_is_allowed() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer, "v1.0.0");

        let draft = ReceiptDraft {
            tool_id: "text_process".to_string(),
            input: "Hello, please help me with my question.".to_string(),
            capability_token: None,
            session_node: "genesis".to_string(),
            timestamp: 0,
        };

        let decision = hook_chain.pre_capability_use(&draft).await.unwrap();
        match decision {
            HookDecision::Allow { .. } => (),
            _ => panic!("BEG-02: Safe input must be allowed"),
        }
    }

    #[tokio::test]
    async fn beg02_elevated_tools_require_consent() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer, "v1.0.0");

        let elevated_tools = ["file_write", "network_request", "subprocess"];

        for tool in elevated_tools {
            let draft = ReceiptDraft {
                tool_id: tool.to_string(),
                input: "safe input".to_string(),
                capability_token: None,
                session_node: "genesis".to_string(),
                timestamp: 0,
            };

            let decision = hook_chain.pre_capability_use(&draft).await.unwrap();
            match decision {
                HookDecision::Ask { .. } => (),
                _ => panic!("BEG-02: Elevated tool '{}' must require consent", tool),
            }
        }
    }
}

// ============================================================================
// BEG-03: Session DAG Evidence
// ============================================================================

mod session_dag_gates {
    use super::*;

    #[test]
    fn beg03_genesis_has_no_parent() {
        let dag = SessionDAG::new("v1.0.0");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let head = rt.block_on(dag.get_head()).unwrap();

        assert!(
            head.parent_hash.is_none(),
            "BEG-03: Genesis must have no parent"
        );
    }

    #[tokio::test]
    async fn beg03_child_links_to_parent() {
        let dag = SessionDAG::new("v1.0.0");

        let genesis = dag.get_head().await.unwrap();
        let child = dag.advance("state1", "receipts1", 0.1).await;

        assert_eq!(
            child.parent_hash,
            Some(genesis.node_hash.clone()),
            "BEG-03: Child must link to parent"
        );
    }

    #[tokio::test]
    async fn beg03_fork_preserves_state() {
        let dag = SessionDAG::new("v1.0.0");

        let head = dag.get_head().await.unwrap();
        let forked = dag.fork("experiment-1").await;

        assert_eq!(
            forked.state_root, head.state_root,
            "BEG-03: Fork must preserve state_root"
        );
        assert_eq!(
            forked.fork_id,
            Some("experiment-1".to_string()),
            "BEG-03: Fork must have fork_id"
        );
    }

    #[tokio::test]
    async fn beg03_node_hash_is_unique() {
        let dag = SessionDAG::new("v1.0.0");

        let node1 = dag.advance("state1", "receipts1", 0.1).await;
        let node2 = dag.advance("state2", "receipts2", 0.2).await;

        assert_ne!(
            node1.node_hash, node2.node_hash,
            "BEG-03: Node hashes must be unique"
        );
    }

    #[tokio::test]
    async fn beg03_impact_delta_is_recorded() {
        let dag = SessionDAG::new("v1.0.0");

        let child = dag.advance("state1", "receipts1", 0.42).await;

        assert!(
            (child.impact_delta - 0.42).abs() < f64::EPSILON,
            "BEG-03: Impact delta must be recorded"
        );
    }
}

// ============================================================================
// BEG-04: Post-Hook Evidence
// ============================================================================

mod post_hook_gates {
    use super::*;

    #[tokio::test]
    async fn beg04_successful_execution_commits() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer, "v1.0.0");

        let executed = ExecutedReceipt {
            draft: ReceiptDraft {
                tool_id: "text_process".to_string(),
                input: "Hello".to_string(),
                capability_token: None,
                session_node: "genesis".to_string(),
                timestamp: 0,
            },
            output: "Processed: Hello".to_string(),
            execution_time_ms: 100,
            tokens_used: 50,
            success: true,
            error: None,
        };

        let result = hook_chain.post_capability_use(&executed).await.unwrap();
        match result {
            PostHookResult::Commit { receipt_id, .. } => {
                assert!(
                    receipt_id.starts_with("rec_"),
                    "BEG-04: Receipt ID must have prefix"
                );
            }
            _ => panic!("BEG-04: Successful execution must commit"),
        }
    }

    #[tokio::test]
    async fn beg04_failed_execution_quarantines() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer, "v1.0.0");

        let executed = ExecutedReceipt {
            draft: ReceiptDraft {
                tool_id: "text_process".to_string(),
                input: "Hello".to_string(),
                capability_token: None,
                session_node: "genesis".to_string(),
                timestamp: 0,
            },
            output: "".to_string(),
            execution_time_ms: 100,
            tokens_used: 50,
            success: false,
            error: Some("Division by zero".to_string()),
        };

        let result = hook_chain.post_capability_use(&executed).await.unwrap();
        match result {
            PostHookResult::Quarantine { reason, .. } => {
                assert!(
                    reason.contains("failed"),
                    "BEG-04: Quarantine must explain failure"
                );
            }
            _ => panic!("BEG-04: Failed execution must quarantine"),
        }
    }

    #[tokio::test]
    async fn beg04_budget_exceeded_quarantines() {
        let signer = Arc::new(SoftwareSigner::new());
        let hook_chain = SATHookChain::new(signer.clone(), "v1.0.0");

        // Create token with small budget
        let mut token = CapabilityToken::new("text_process", "execute", CapabilityTier::T0Mobile);
        token.budget.max_tokens = 100;
        token.sign(signer.as_ref()).await.unwrap();

        let executed = ExecutedReceipt {
            draft: ReceiptDraft {
                tool_id: "text_process".to_string(),
                input: "Hello".to_string(),
                capability_token: Some(token),
                session_node: "genesis".to_string(),
                timestamp: 0,
            },
            output: "Result".to_string(),
            execution_time_ms: 100,
            tokens_used: 500, // Exceeds budget
            success: true,
            error: None,
        };

        let result = hook_chain.post_capability_use(&executed).await.unwrap();
        match result {
            PostHookResult::Quarantine { reason, .. } => {
                assert!(
                    reason.contains("budget"),
                    "BEG-04: Must explain budget exceeded"
                );
            }
            _ => panic!("BEG-04: Budget exceeded must quarantine"),
        }
    }
}

// ============================================================================
// BEG-05: Consumer-First Tier Compliance
// ============================================================================

mod consumer_tier_gates {
    use super::*;

    #[test]
    fn beg05_t0_mobile_is_most_restrictive() {
        let t0 = CapabilityTier::T0Mobile.default_budget();
        let t1 = CapabilityTier::T1Consumer.default_budget();

        assert!(
            t0.max_tokens < t1.max_tokens,
            "BEG-05: T0 must be more restrictive than T1"
        );
        assert!(
            t0.max_time_ms < t1.max_time_ms,
            "BEG-05: T0 time limit must be stricter"
        );
        assert!(
            t0.max_tool_calls < t1.max_tool_calls,
            "BEG-05: T0 tool calls must be fewer"
        );
    }

    #[test]
    fn beg05_t1_consumer_is_default() {
        let default = CapabilityTier::default();
        assert_eq!(
            default,
            CapabilityTier::T1Consumer,
            "BEG-05: Default tier must be T1Consumer"
        );
    }

    #[test]
    fn beg05_offload_only_allowed_in_higher_tiers() {
        let t0 = CapabilityTier::T0Mobile.default_budget();
        let t1 = CapabilityTier::T1Consumer.default_budget();
        let t2 = CapabilityTier::T2Pro.default_budget();
        let t3 = CapabilityTier::T3Pooled.default_budget();

        assert!(!t0.offload_allowed, "BEG-05: T0 must not allow offload");
        assert!(!t1.offload_allowed, "BEG-05: T1 must not allow offload");
        assert!(t2.offload_allowed, "BEG-05: T2 may allow offload");
        assert!(t3.offload_allowed, "BEG-05: T3 may allow offload");
    }
}
