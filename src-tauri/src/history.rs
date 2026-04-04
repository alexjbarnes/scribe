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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry() -> HistoryEntry {
        HistoryEntry {
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            text: "Hello world.".to_string(),
            model_id: "whisper-tiny".to_string(),
            duration_ms: 150,
            audio_duration_ms: 2000,
            postprocess_ms: 5,
            pipeline_stages: vec![
                PipelineStage {
                    name: "Raw transcription".to_string(),
                    text: "hello world".to_string(),
                    changed: false,
                    duration_ms: 0,
                    grammar_score: None,
                    grammar_sentences: vec![],
                },
                PipelineStage {
                    name: "Cleanup".to_string(),
                    text: "Hello world.".to_string(),
                    changed: true,
                    duration_ms: 1,
                    grammar_score: None,
                    grammar_sentences: vec![],
                },
            ],
            chunk_timings: vec![
                ChunkTiming { audio_ms: 1500, transcribe_ms: 80 },
            ],
        }
    }

    #[test]
    fn history_entry_serialization_roundtrip() {
        let entry = sample_entry();
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: HistoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.text, "Hello world.");
        assert_eq!(deserialized.model_id, "whisper-tiny");
        assert_eq!(deserialized.duration_ms, 150);
        assert_eq!(deserialized.audio_duration_ms, 2000);
        assert_eq!(deserialized.postprocess_ms, 5);
        assert_eq!(deserialized.pipeline_stages.len(), 2);
        assert_eq!(deserialized.chunk_timings.len(), 1);
        assert_eq!(deserialized.chunk_timings[0].audio_ms, 1500);
    }

    #[test]
    fn history_entry_default_fields_on_missing_json() {
        // Older entries might lack new fields. Defaults should fill in.
        let json = r#"{"timestamp":"2026-01-01T00:00:00Z","text":"hi","model_id":"m","duration_ms":10}"#;
        let entry: HistoryEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.audio_duration_ms, 0);
        assert_eq!(entry.postprocess_ms, 0);
        assert!(entry.pipeline_stages.is_empty());
        assert!(entry.chunk_timings.is_empty());
    }

    #[test]
    fn history_add_and_list() {
        let history = History { entries: Mutex::new(vec![]) };
        history.add(
            "Test text.".into(),
            "model-1".into(),
            100, 2000, 5, vec![], vec![],
        );
        // list() tries to reload from disk, but we work with in-memory state.
        // Access entries directly for this unit test.
        let entries = history.entries.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "Test text.");
        assert_eq!(entries[0].model_id, "model-1");
    }

    #[test]
    fn history_clear() {
        let history = History { entries: Mutex::new(vec![]) };
        history.add("one".into(), "m".into(), 1, 1, 0, vec![], vec![]);
        history.add("two".into(), "m".into(), 1, 1, 0, vec![], vec![]);
        assert_eq!(history.entries.lock().unwrap().len(), 2);
        history.clear();
        assert!(history.entries.lock().unwrap().is_empty());
    }

    #[test]
    fn history_entries_serialize_as_json_array() {
        let entries = vec![sample_entry()];
        let json = serde_json::to_string_pretty(&entries).unwrap();
        let parsed: Vec<HistoryEntry> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].text, "Hello world.");
    }

    #[test]
    fn pipeline_stage_serialization() {
        let stage = PipelineStage {
            name: "Grammar (neural)".to_string(),
            text: "Corrected text.".to_string(),
            changed: true,
            duration_ms: 42,
            grammar_score: Some(0.85),
            grammar_sentences: vec![],
        };
        let json = serde_json::to_string(&stage).unwrap();
        assert!(json.contains("grammar_score"));
        let deserialized: PipelineStage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.grammar_score, Some(0.85));
    }

    #[test]
    fn pipeline_stage_skips_none_score_in_json() {
        let stage = PipelineStage {
            name: "Cleanup".to_string(),
            text: "text".to_string(),
            changed: false,
            duration_ms: 0,
            grammar_score: None,
            grammar_sentences: vec![],
        };
        let json = serde_json::to_string(&stage).unwrap();
        // grammar_score should be omitted when None (skip_serializing_if)
        assert!(!json.contains("grammar_score"));
    }

    #[test]
    fn chunk_timing_serialization() {
        let ct = ChunkTiming { audio_ms: 1234, transcribe_ms: 56 };
        let json = serde_json::to_string(&ct).unwrap();
        let parsed: ChunkTiming = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.audio_ms, 1234);
        assert_eq!(parsed.transcribe_ms, 56);
    }
}
