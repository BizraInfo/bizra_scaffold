// src/storage.rs
// P1-B: Persistent Receipt Store (Chain of Truth)
// Implements Atomic Append and Chain Integrity.

use crate::executor::ThoughtExecReceipt;
use async_trait::async_trait;
use std::collections::HashMap;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::warn;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Redis error: {0}")]
    Redis(String),
    #[error("Chain violation: expected prev_hash {expected}, got {actual}")]
    ChainViolation { expected: String, actual: String },
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Lock error")]
    Lock,
}

impl From<redis::RedisError> for StorageError {
    fn from(e: redis::RedisError) -> Self {
        StorageError::Redis(e.to_string())
    }
}
impl From<serde_json::Error> for StorageError {
    fn from(e: serde_json::Error) -> Self {
        StorageError::Serialization(e.to_string())
    }
}

#[async_trait]
pub trait ReceiptStore: Send + Sync {
    /// Appends a new receipt to the chain.
    /// MUST enforce that receipt.payload.prev_hash == current HEAD.
    /// Updates HEAD to receipt.receipt_hash.
    async fn append(&self, receipt: &ThoughtExecReceipt) -> Result<(), StorageError>;

    /// Returns the hash of the latest receipt (HEAD).
    /// If empty, returns "GENESIS" (or empty string/zero hash).
    async fn get_head_hash(&self) -> Result<String, StorageError>;

    /// Gets a receipt by its payload_id or receipt_hash.
    /// For this simplified store, we map receipt_hash -> Receipt.
    async fn get_receipt(&self, hash: &str) -> Result<Option<ThoughtExecReceipt>, StorageError>;
}

/// In-Memory Store for Testing
pub struct InMemoryReceiptStore {
    store: RwLock<HashMap<String, ThoughtExecReceipt>>,
    head: RwLock<String>,
}

impl Default for InMemoryReceiptStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryReceiptStore {
    pub fn new() -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
            head: RwLock::new("GENESIS".to_string()),
        }
    }
}

#[async_trait]
impl ReceiptStore for InMemoryReceiptStore {
    async fn append(&self, receipt: &ThoughtExecReceipt) -> Result<(), StorageError> {
        let mut head = self.head.write().await;
        let mut store = self.store.write().await;

        let current = head.clone();
        if receipt.payload.prev_hash != current {
            warn!(
                "Chain violation in InMemory: Expected {}, got {}",
                current, receipt.payload.prev_hash
            );
            return Err(StorageError::ChainViolation {
                expected: current,
                actual: receipt.payload.prev_hash.clone(),
            });
        }

        store.insert(receipt.receipt_hash.clone(), receipt.clone());
        *head = receipt.receipt_hash.clone();
        Ok(())
    }

    async fn get_head_hash(&self) -> Result<String, StorageError> {
        let head = self.head.read().await;
        Ok(head.clone())
    }

    async fn get_receipt(&self, hash: &str) -> Result<Option<ThoughtExecReceipt>, StorageError> {
        let store = self.store.read().await;
        Ok(store.get(hash).cloned())
    }
}

/// Redis Store for Persistence
pub struct RedisReceiptStore {
    conn_manager: redis::aio::ConnectionManager,
    key_prefix: String,
}

impl RedisReceiptStore {
    pub async fn new(url: &str) -> Result<Self, StorageError> {
        let client = redis::Client::open(url)?;
        let conn_manager = client.get_connection_manager().await?;
        Ok(Self {
            conn_manager,
            key_prefix: "bizra:chain".to_string(),
        })
    }
}

#[async_trait]
impl ReceiptStore for RedisReceiptStore {
    async fn append(&self, receipt: &ThoughtExecReceipt) -> Result<(), StorageError> {
        let mut conn = self.conn_manager.clone();
        let head_key = format!("{}:head", self.key_prefix);
        let receipt_key = format!("{}:receipt:{}", self.key_prefix, receipt.receipt_hash);

        // Atomic Check-And-Set using Lua
        let script = redis::Script::new(
            r"
            let head_key = KEYS[1]
            let receipt_key = KEYS[2]
            let new_hash = ARGV[1]
            let prev_hash = ARGV[2]
            let receipt_json = ARGV[3]

            let current_head = redis::call('GET', head_key)
            if not current_head then
                current_head = 'GENESIS'
            end

            if current_head ~= prev_hash then
                return redis::error_reply('ChainViolation: Expected ' .. current_head .. ', got ' .. prev_hash)
            end

            redis::call('SET', receipt_key, receipt_json)
            redis::call('SET', head_key, new_hash)
            return 'OK'
        ",
        );

        let json = serde_json::to_string(receipt)?;
        let result: Result<String, redis::RedisError> = script
            .key(&head_key)
            .key(&receipt_key)
            .arg(&receipt.receipt_hash)
            .arg(&receipt.payload.prev_hash)
            .arg(&json)
            .invoke_async(&mut conn)
            .await;

        match result {
            Ok(_) => Ok(()),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("ChainViolation") {
                    // Best effort parsing
                    let parts: Vec<&str> = msg.split("Expected ").collect();
                    if parts.len() > 1 {
                        let rest = parts[1];
                        let p2: Vec<&str> = rest.split(", got ").collect();
                        if p2.len() > 1 {
                            let exp = p2[0].trim();
                            let got = p2[1].trim();
                            return Err(StorageError::ChainViolation {
                                expected: exp.into(),
                                actual: got.into(),
                            });
                        }
                    }
                    return Err(StorageError::ChainViolation {
                        expected: "UNKNOWN".into(),
                        actual: receipt.payload.prev_hash.clone(),
                    });
                }
                Err(StorageError::Redis(msg))
            }
        }
    }

    async fn get_head_hash(&self) -> Result<String, StorageError> {
        let mut conn = self.conn_manager.clone();
        let head_key = format!("{}:head", self.key_prefix);
        let head: Option<String> = redis::cmd("GET")
            .arg(head_key)
            .query_async(&mut conn)
            .await?;
        Ok(head.unwrap_or_else(|| "GENESIS".to_string()))
    }

    async fn get_receipt(&self, hash: &str) -> Result<Option<ThoughtExecReceipt>, StorageError> {
        let mut conn = self.conn_manager.clone();
        let key = format!("{}:receipt:{}", self.key_prefix, hash);
        let data: Option<String> = redis::cmd("GET").arg(key).query_async(&mut conn).await?;
        match data {
            Some(s) => Ok(Some(serde_json::from_str(&s)?)),
            None => Ok(None),
        }
    }
}
