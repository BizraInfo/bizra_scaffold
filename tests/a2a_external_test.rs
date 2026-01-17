// tests/a2a_external_test.rs - External AI Integration Tests
//
// Tests for OpenAI and Gemini integration via A2A protocol

use meta_alpha_dual_agentic::a2a::{Agent, AgentCard, Capability, Task};
use meta_alpha_dual_agentic::a2a_external::{AIProvider, ExternalAIAdapter};

#[test]
fn test_provider_names() {
    let openai = AIProvider::OpenAI {
        api_key: "test_key".to_string(),
        model: "gpt-4".to_string(),
        organization: None,
    };
    assert_eq!(openai.name(), "openai");
    assert_eq!(openai.model(), "gpt-4");

    let gemini = AIProvider::GoogleGemini {
        api_key: "test_key".to_string(),
        model: "gemini-1.5-pro".to_string(),
    };
    assert_eq!(gemini.name(), "gemini");
    assert_eq!(gemini.model(), "gemini-1.5-pro");
}

#[test]
fn test_adapter_creation() {
    let capabilities = vec![Capability::CodeGeneration, Capability::DataPipeline];

    let provider = AIProvider::OpenAI {
        api_key: "test_key".to_string(),
        model: "gpt-4".to_string(),
        organization: None,
    };

    let adapter = ExternalAIAdapter::new("test_codex".to_string(), provider, capabilities.clone());

    assert!(adapter.is_ok());
    let adapter = adapter.unwrap();
    assert_eq!(adapter.capabilities(), capabilities);
}

#[test]
fn test_agent_card() {
    let capabilities = vec![Capability::DataMining, Capability::Search];

    let provider = AIProvider::GoogleGemini {
        api_key: "test_key".to_string(),
        model: "gemini-1.5-pro".to_string(),
    };

    let adapter =
        ExternalAIAdapter::new("test_gemini".to_string(), provider, capabilities).unwrap();

    let card: AgentCard = adapter.agent_card();
    assert_eq!(card.name, "test_gemini");
    assert_eq!(card.version, "1.0.0");
    assert!(card.external);
    assert_eq!(card.provider, Some("gemini".to_string()));
}

#[test]
fn test_timeout_configuration() {
    std::env::set_var("EXTERNAL_AI_TIMEOUT", "45");
    std::env::set_var("EXTERNAL_AI_MAX_RETRIES", "5");

    let capabilities = vec![Capability::Analysis];
    let provider = AIProvider::OpenAI {
        api_key: "test_key".to_string(),
        model: "gpt-4".to_string(),
        organization: None,
    };

    let adapter = ExternalAIAdapter::new("test".to_string(), provider, capabilities);

    assert!(adapter.is_ok());

    std::env::remove_var("EXTERNAL_AI_TIMEOUT");
    std::env::remove_var("EXTERNAL_AI_MAX_RETRIES");
}

// Integration test - requires actual API keys
#[tokio::test]
#[ignore] // Ignored by default - run with `cargo test -- --ignored` when you have API keys
async fn test_openai_execution() {
    // Only run if API key is set
    if std::env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let adapter = ExternalAIAdapter::from_env_openai();
    assert!(adapter.is_ok());

    let adapter = adapter.unwrap();
    let task = Task {
        task_id: "test_001".to_string(),
        description: "Write a simple hello world function in Python".to_string(),
        context: None,
    };

    let result = adapter.execute(task).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert_eq!(response.agent_id, "codex");
    assert!(!response.content.is_empty());
    println!("OpenAI Response: {}", response.content);
}

// Integration test - requires actual API keys
#[tokio::test]
#[ignore] // Ignored by default
async fn test_gemini_execution() {
    // Only run if API key is set
    if std::env::var("GOOGLE_API_KEY").is_err() {
        return;
    }

    let adapter = ExternalAIAdapter::from_env_gemini();
    assert!(adapter.is_ok());

    let adapter = adapter.unwrap();
    let task = Task {
        task_id: "test_002".to_string(),
        description: "Explain the concept of machine learning in one paragraph".to_string(),
        context: None,
    };

    let result = adapter.execute(task).await;
    assert!(result.is_ok());

    let response = result.unwrap();
    assert_eq!(response.agent_id, "gemini");
    assert!(!response.content.is_empty());
    println!("Gemini Response: {}", response.content);
}
