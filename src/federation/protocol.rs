use crate::fixed::Fixed64;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum TrustTier {
    Bronze,
    Silver,
    Gold,
    Platinum,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum MessageType {
    EnrollReq,
    EnrollAck,
    PolicyPush,
    Heartbeat,
    PoiSubmit,
    PoiAck,
    Quarantine,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EnrollmentRequest {
    pub node_id: String,
    pub public_key: String,
    pub tpm_quote: Option<String>,
    pub hardware_manifest: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EnrollmentCertificate {
    pub node_uid: Uuid,
    pub trust_tier: TrustTier,
    pub permissions: Vec<String>,
    pub rate_limits: RateLimits,
    pub signature: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RateLimits {
    pub requests_per_sec: u32,
    pub impact_per_hour: Fixed64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Heartbeat {
    pub node_uid: Uuid,
    pub timestamp: u64,
    pub cpu_usage: f32,
    pub mem_usage: f32,
    pub nonce: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PoiReceipt {
    pub node_uid: Uuid,
    pub task_id: String,
    pub ihsan_score: Fixed64,
    pub output_hash: String,
    pub evidence_pointers: Vec<String>,
    pub signature: String,
}
