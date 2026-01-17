use serde::{Deserialize, Serialize};
use std::{fs::OpenOptions, io::Write, path::PathBuf, time::{SystemTime, UNIX_EPOCH}};

use crate::ihsan::IhsanScore;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LedgerEvent {
    pub ts_unix_ms: i64,
    pub node_id: String,
    pub kind: String,
    pub payload_json: serde_json::Value,
    pub ihsan: IhsanScore,
    pub hash: String,
}

pub trait Ledger {
    fn append(&mut self, event: LedgerEvent) -> anyhow::Result<()>;
}

#[derive(Clone, Debug)]
pub struct FileLedger {
    pub path: PathBuf,
}

impl FileLedger {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn now_ms() -> i64 {
        let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        (d.as_secs() as i64) * 1000 + (d.subsec_millis() as i64)
    }
}

impl Ledger for FileLedger {
    fn append(&mut self, event: LedgerEvent) -> anyhow::Result<()> {
        let mut f = OpenOptions::new().create(true).append(true).open(&self.path)?;
        let line = serde_json::to_string(&event)?;
        writeln!(f, "{line}")?;
        Ok(())
    }
}
