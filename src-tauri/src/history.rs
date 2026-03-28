use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub timestamp: String,
    pub text: String,
    pub model_id: String,
    pub duration_ms: u64,
    #[serde(default)]
    pub audio_duration_ms: u64,
}

pub struct History {
    entries: Mutex<Vec<HistoryEntry>>,
}

impl History {
    pub fn new() -> Self {
        let entries = Self::load_from_disk().unwrap_or_default();
        Self {
            entries: Mutex::new(entries),
        }
    }

    pub fn add(&self, text: String, model_id: String, duration_ms: u64, audio_duration_ms: u64) {
        let entry = HistoryEntry {
            timestamp: chrono::Utc::now().to_rfc3339(),
            text,
            model_id,
            duration_ms,
            audio_duration_ms,
        };
        let mut entries = self.entries.lock().unwrap();
        entries.push(entry);
        if let Err(e) = Self::save_to_disk(&entries) {
            log::error!("Failed to save history: {e}");
        }
    }

    pub fn list(&self) -> Vec<HistoryEntry> {
        self.entries.lock().unwrap().clone()
    }

    pub fn clear(&self) {
        let mut entries = self.entries.lock().unwrap();
        entries.clear();
        if let Err(e) = Self::save_to_disk(&entries) {
            log::error!("Failed to save history: {e}");
        }
    }

    fn history_path() -> Option<PathBuf> {
        #[cfg(target_os = "android")]
        {
            std::env::var_os("SCRIBE_DATA_DIR")
                .map(|d| PathBuf::from(d).join("history.json"))
        }
        #[cfg(not(target_os = "android"))]
        {
            dirs::config_dir().map(|d| d.join("scribe").join("history.json"))
        }
    }

    fn load_from_disk() -> Option<Vec<HistoryEntry>> {
        let path = Self::history_path()?;
        let data = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&data).ok()
    }

    fn save_to_disk(entries: &[HistoryEntry]) -> Result<(), String> {
        let path = Self::history_path().ok_or("no data dir")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create dir: {e}"))?;
        }
        let data = serde_json::to_string(entries).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(&path, data).map_err(|e| format!("write: {e}"))?;
        Ok(())
    }
}
