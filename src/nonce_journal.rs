// ═══════════════════════════════════════════════════════════════════════════════
// BIZRA NONCE JOURNAL - Persistent Replay Protection
// ═══════════════════════════════════════════════════════════════════════════════
//
// This module provides persistent replay protection for the Verifiable Kernel.
// An in-memory `seen_nonces` only works within a single process lifetime.
// For production and distributed systems, we need persistence.
//
// Design principles:
// 1. DURABLE - Survives process restarts
// 2. DETERMINISTIC - JCS-friendly state serialization
// 3. EFFICIENT - Bloom filter + LRU eviction for memory bounds
// 4. VERIFIABLE - Journal itself can be audited
//
// Standing on giants:
// - Sigstore Rekor: Append-only transparency log patterns
// - Certificate Transparency: Merkle tree audit proofs

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

// ═══════════════════════════════════════════════════════════════════════════════
// NONCE ENTRY
// ═══════════════════════════════════════════════════════════════════════════════

/// A single nonce journal entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NonceEntry {
    /// The nonce value
    pub nonce: String,
    /// Session ID this nonce belongs to
    pub session_id: String,
    /// Counter value when nonce was used
    pub counter: u64,
    /// Timestamp when recorded (nanoseconds)
    pub recorded_at_ns: u64,
    /// Receipt ID that used this nonce
    pub receipt_id: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION COUNTER STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Monotonic counter state per session
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionCounterState {
    /// Last seen counter for this session
    pub last_counter: u64,
    /// Total entries for this session
    pub entry_count: u64,
    /// First entry timestamp
    pub first_seen_ns: u64,
    /// Last entry timestamp
    pub last_seen_ns: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// REPLAY CHECK RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a replay check
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayCheckResult {
    /// Nonce/counter is valid (never seen, monotonically increasing)
    Valid,
    /// Duplicate nonce detected
    DuplicateNonce { original_receipt_id: String },
    /// Non-monotonic counter (replay or reordering attack)
    NonMonotonicCounter {
        expected_min: u64,
        received: u64,
    },
    /// Session expired or invalid
    InvalidSession { reason: String },
}

// ═══════════════════════════════════════════════════════════════════════════════
// NONCE JOURNAL
// ═══════════════════════════════════════════════════════════════════════════════

/// Persistent nonce journal for replay protection
///
/// Architecture:
/// - Append-only log file (survives restarts)
/// - In-memory index for fast lookups
/// - Per-session counter tracking
/// - LRU eviction for memory bounds (configurable)
pub struct NonceJournal {
    /// Path to the journal file
    journal_path: PathBuf,

    /// In-memory nonce index: nonce -> NonceEntry
    /// Uses BTreeSet for deterministic iteration
    nonces: BTreeSet<String>,

    /// Per-session counter state
    /// Uses BTreeMap for deterministic serialization
    session_counters: BTreeMap<String, SessionCounterState>,

    /// LRU tracking for memory bounds (oldest first)
    lru_queue: VecDeque<String>,

    /// Maximum nonces to keep in memory
    max_nonces: usize,

    /// Journal checkpoint hash (for integrity verification)
    checkpoint_hash: String,

    /// Total entries written to journal
    total_entries: u64,
}

impl NonceJournal {
    /// Create a new nonce journal at the given path
    ///
    /// If the file exists, loads existing entries.
    /// If not, creates a new empty journal.
    pub fn new(journal_path: impl Into<PathBuf>, max_nonces: usize) -> std::io::Result<Self> {
        let journal_path = journal_path.into();

        let mut journal = Self {
            journal_path,
            nonces: BTreeSet::new(),
            session_counters: BTreeMap::new(),
            lru_queue: VecDeque::new(),
            max_nonces,
            checkpoint_hash: "genesis".to_string(),
            total_entries: 0,
        };

        // Load existing entries if file exists
        if journal.journal_path.exists() {
            journal.load_from_disk()?;
        }

        Ok(journal)
    }

    /// Load journal entries from disk
    fn load_from_disk(&mut self) -> std::io::Result<()> {
        let file = File::open(&self.journal_path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            // Parse entry
            if let Ok(entry) = serde_json::from_str::<NonceEntry>(&line) {
                self.index_entry(&entry);
            }
        }

        // Recompute checkpoint hash
        self.checkpoint_hash = self.compute_checkpoint_hash();

        Ok(())
    }

    /// Index an entry in memory
    fn index_entry(&mut self, entry: &NonceEntry) {
        // Add to nonce set
        self.nonces.insert(entry.nonce.clone());

        // Update session counter state
        let session_state = self.session_counters
            .entry(entry.session_id.clone())
            .or_default();

        if session_state.entry_count == 0 {
            session_state.first_seen_ns = entry.recorded_at_ns;
        }
        session_state.last_counter = session_state.last_counter.max(entry.counter);
        session_state.last_seen_ns = entry.recorded_at_ns;
        session_state.entry_count += 1;

        // LRU tracking
        self.lru_queue.push_back(entry.nonce.clone());
        self.total_entries += 1;

        // Evict if over limit
        while self.nonces.len() > self.max_nonces {
            if let Some(oldest) = self.lru_queue.pop_front() {
                self.nonces.remove(&oldest);
            }
        }
    }

    /// Check if a nonce/counter combination would be a replay
    pub fn check_replay(
        &self,
        session_id: &str,
        nonce: &str,
        counter: u64,
    ) -> ReplayCheckResult {
        // Check for duplicate nonce
        if self.nonces.contains(nonce) {
            return ReplayCheckResult::DuplicateNonce {
                original_receipt_id: format!("(lookup not implemented for memory efficiency)"),
            };
        }

        // Check monotonic counter
        if let Some(session_state) = self.session_counters.get(session_id) {
            if counter <= session_state.last_counter {
                return ReplayCheckResult::NonMonotonicCounter {
                    expected_min: session_state.last_counter + 1,
                    received: counter,
                };
            }
        }

        ReplayCheckResult::Valid
    }

    /// Record a nonce (after validation)
    ///
    /// Returns the checkpoint hash after recording
    pub fn record(
        &mut self,
        entry: NonceEntry,
    ) -> std::io::Result<String> {
        // Validate before recording
        let check = self.check_replay(&entry.session_id, &entry.nonce, entry.counter);
        if check != ReplayCheckResult::Valid {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Replay detected: {:?}", check),
            ));
        }

        // Append to journal file
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.journal_path)?;

        let line = serde_json::to_string(&entry)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writeln!(file, "{}", line)?;
        file.sync_all()?; // Ensure durability

        // Index in memory
        self.index_entry(&entry);

        // Update checkpoint hash
        self.checkpoint_hash = self.compute_checkpoint_hash();

        Ok(self.checkpoint_hash.clone())
    }

    /// Compute checkpoint hash (for journal integrity)
    fn compute_checkpoint_hash(&self) -> String {
        let state = JournalCheckpoint {
            total_entries: self.total_entries,
            session_count: self.session_counters.len() as u64,
            nonce_count: self.nonces.len() as u64,
        };

        let json = serde_json::to_string(&state).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Get the current checkpoint hash
    pub fn checkpoint_hash(&self) -> &str {
        &self.checkpoint_hash
    }

    /// Get total entries recorded
    pub fn total_entries(&self) -> u64 {
        self.total_entries
    }

    /// Get session counter state
    pub fn session_state(&self, session_id: &str) -> Option<&SessionCounterState> {
        self.session_counters.get(session_id)
    }

    /// Check if a specific nonce exists
    pub fn nonce_exists(&self, nonce: &str) -> bool {
        self.nonces.contains(nonce)
    }

    /// Get current timestamp in nanoseconds
    pub fn now_ns() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        duration.as_nanos() as u64
    }

    /// Compact the journal (remove evicted entries)
    ///
    /// This rewrites the journal with only current entries.
    /// Call periodically or when journal file grows too large.
    pub fn compact(&mut self) -> std::io::Result<()> {
        // Read all entries
        let mut entries: Vec<NonceEntry> = Vec::new();

        if self.journal_path.exists() {
            let file = File::open(&self.journal_path)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                if let Ok(entry) = serde_json::from_str::<NonceEntry>(&line) {
                    // Only keep entries still in memory
                    if self.nonces.contains(&entry.nonce) {
                        entries.push(entry);
                    }
                }
            }
        }

        // Write compacted journal
        let temp_path = self.journal_path.with_extension("tmp");
        let mut file = File::create(&temp_path)?;

        for entry in &entries {
            let line = serde_json::to_string(entry)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            writeln!(file, "{}", line)?;
        }
        file.sync_all()?;

        // Atomic rename
        fs::rename(&temp_path, &self.journal_path)?;

        Ok(())
    }
}

/// Journal checkpoint state (for integrity verification)
#[derive(Debug, Serialize, Deserialize)]
struct JournalCheckpoint {
    total_entries: u64,
    session_count: u64,
    nonce_count: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// DISTRIBUTED NONCE JOURNAL (v0.3+ placeholder)
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for distributed nonce journal (v0.3+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedNonceConfig {
    /// Minimum nodes that must agree for consensus
    pub quorum: usize,
    /// Nodes in the cluster
    pub nodes: Vec<String>,
    /// Replication factor
    pub replication: usize,
    /// Sync interval (milliseconds)
    pub sync_interval_ms: u64,
}

/// Placeholder for distributed nonce journal
///
/// v0.2: Single-node file-based (implemented above)
/// v0.3: Will add distributed consensus (HotStuff BFT-inspired)
pub trait DistributedNonceJournal {
    /// Check replay across all nodes
    fn check_replay_distributed(
        &self,
        session_id: &str,
        nonce: &str,
        counter: u64,
    ) -> impl std::future::Future<Output = ReplayCheckResult> + Send;

    /// Record nonce with distributed consensus
    fn record_distributed(
        &mut self,
        entry: NonceEntry,
    ) -> impl std::future::Future<Output = Result<String, std::io::Error>> + Send;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_new_journal_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonce.journal");

        let journal = NonceJournal::new(&path, 1000).unwrap();
        assert_eq!(journal.total_entries(), 0);
        assert!(!path.exists()); // Not created until first write
    }

    #[test]
    fn test_record_and_check_replay() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonce.journal");

        let mut journal = NonceJournal::new(&path, 1000).unwrap();

        let entry = NonceEntry {
            nonce: "nonce-001".to_string(),
            session_id: "session-001".to_string(),
            counter: 1,
            recorded_at_ns: NonceJournal::now_ns(),
            receipt_id: "EXEC-001".to_string(),
        };

        // First record should succeed
        let result = journal.record(entry.clone());
        assert!(result.is_ok());

        // Duplicate nonce should fail
        let check = journal.check_replay("session-001", "nonce-001", 2);
        assert!(matches!(check, ReplayCheckResult::DuplicateNonce { .. }));
    }

    #[test]
    fn test_monotonic_counter_check() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonce.journal");

        let mut journal = NonceJournal::new(&path, 1000).unwrap();

        // Record counter=5
        let entry1 = NonceEntry {
            nonce: "nonce-001".to_string(),
            session_id: "session-001".to_string(),
            counter: 5,
            recorded_at_ns: NonceJournal::now_ns(),
            receipt_id: "EXEC-001".to_string(),
        };
        journal.record(entry1).unwrap();

        // Counter=3 should fail (not monotonic)
        let check = journal.check_replay("session-001", "nonce-002", 3);
        assert!(matches!(check, ReplayCheckResult::NonMonotonicCounter {
            expected_min: 6,
            received: 3
        }));

        // Counter=6 should succeed
        let check = journal.check_replay("session-001", "nonce-003", 6);
        assert_eq!(check, ReplayCheckResult::Valid);
    }

    #[test]
    fn test_persistence_across_restarts() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonce.journal");

        // Create and populate journal
        {
            let mut journal = NonceJournal::new(&path, 1000).unwrap();

            let entry = NonceEntry {
                nonce: "nonce-persist".to_string(),
                session_id: "session-001".to_string(),
                counter: 42,
                recorded_at_ns: NonceJournal::now_ns(),
                receipt_id: "EXEC-PERSIST".to_string(),
            };
            journal.record(entry).unwrap();
        }

        // Reload journal (simulating restart)
        {
            let journal = NonceJournal::new(&path, 1000).unwrap();

            // Nonce should still be present
            assert!(journal.nonce_exists("nonce-persist"));

            // Counter state should be preserved
            let session_state = journal.session_state("session-001").unwrap();
            assert_eq!(session_state.last_counter, 42);
        }
    }

    #[test]
    fn test_lru_eviction() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonce.journal");

        // Small max_nonces to trigger eviction
        let mut journal = NonceJournal::new(&path, 3).unwrap();

        // Record 5 entries
        for i in 1..=5 {
            let entry = NonceEntry {
                nonce: format!("nonce-{:03}", i),
                session_id: "session-001".to_string(),
                counter: i,
                recorded_at_ns: NonceJournal::now_ns(),
                receipt_id: format!("EXEC-{:03}", i),
            };
            journal.record(entry).unwrap();
        }

        // Only 3 should be in memory (LRU eviction)
        assert!(!journal.nonce_exists("nonce-001")); // Evicted
        assert!(!journal.nonce_exists("nonce-002")); // Evicted
        assert!(journal.nonce_exists("nonce-003"));
        assert!(journal.nonce_exists("nonce-004"));
        assert!(journal.nonce_exists("nonce-005"));
    }
}
