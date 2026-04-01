use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static HISTORY: OnceLock<History> = OnceLock::new();

use serde::{Deserialize, Serialize};

use crate::postprocess::PipelineStage;

/// Timing for a single transcription chunk (VAD segment or tail).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkTiming {
    pub audio_ms: u64,
    pub transcribe_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub timestamp: String,
    pub text: String,
    pub model_id: String,
    pub duration_ms: u64,
    #[serde(default)]
    pub audio_duration_ms: u64,
    #[serde(default)]
    pub postprocess_ms: u64,
    #[serde(default)]
    pub pipeline_stages: Vec<PipelineStage>,
    #[serde(default)]
    pub chunk_timings: Vec<ChunkTiming>,
    #[serde(default)]
    pub filtered_segments: u32,
    #[serde(default)]
    pub filtered_audio_ms: u64,
}

pub struct History {
    entries: Mutex<Vec<HistoryEntry>>,
}

impl History {
    pub fn init_global() -> &'static Self {
        HISTORY.get_or_init(Self::new)
    }

    pub fn global() -> &'static Self {
        HISTORY.get().expect("History not initialized")
    }

    pub fn new() -> Self {
        let entries = Self::load_from_disk().unwrap_or_default();
        Self {
            entries: Mutex::new(entries),
        }
    }

    pub fn add(
        &self,
        text: String,
        model_id: String,
        duration_ms: u64,
        audio_duration_ms: u64,
        postprocess_ms: u64,
        pipeline_stages: Vec<PipelineStage>,
        chunk_timings: Vec<ChunkTiming>,
        filtered_segments: u32,
        filtered_audio_ms: u64,
    ) {
        let entry = HistoryEntry {
            timestamp: chrono::Utc::now().to_rfc3339(),
            text,
            model_id,
            duration_ms,
            audio_duration_ms,
            postprocess_ms,
            pipeline_stages,
            chunk_timings,
            filtered_segments,
            filtered_audio_ms,
        };
        let mut entries = self.entries.lock().unwrap();
        entries.push(entry);
        if let Err(e) = Self::save_to_disk(&entries) {
            log::error!("Failed to save history: {e}");
        }
    }

    pub fn list(&self) -> Vec<HistoryEntry> {
        // Reload from disk in case of external edits.
        if let Some(entries) = Self::load_from_disk() {
            *self.entries.lock().unwrap() = entries;
        }
        self.entries.lock().unwrap().clone()
    }

    /// Export history as pretty-printed JSON string.
    pub fn export(&self) -> Result<String, String> {
        let entries = self.list();
        serde_json::to_string_pretty(&entries)
            .map_err(|e| format!("serialize: {e}"))
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
            std::env::var_os("VERBA_DATA_DIR")
                .map(|d| PathBuf::from(d).join("history.json"))
        }
        #[cfg(not(target_os = "android"))]
        {
            dirs::config_dir().map(|d| d.join("verba").join("history.json"))
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
