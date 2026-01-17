use serde::{Deserialize, Serialize};

/// This is the language agents speak.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AgentSignal {
    Sense(Vec<u8>),
    Reason { input: String, confidence: f32 },
    Act { command: String, params: Vec<String> },
    Halt { agent_id: String, reason: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofOfInference {
    pub circuit_id: String,
    pub public_inputs: Vec<u8>,
    pub proof: Vec<u8>,
    pub timestamp_unix_ms: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attestation {
    pub node_id: String,
    pub signature: Vec<u8>,
    pub poi: ProofOfInference,
}
