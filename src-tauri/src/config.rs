use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub language: String,
    pub threads: u32,
    pub ollama_url: String,
    pub ollama_model: String,
    pub output_dir: String,
    pub device_index: i32,
    pub active_engine: String,
    pub active_model_id: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            language: "en".into(),
            threads: 4,
            ollama_url: "http://localhost:11434".into(),
            ollama_model: "gemma3:4b".into(),
            output_dir: dirs::document_dir()
                .map(|d| d.join("Meetings").to_string_lossy().into_owned())
                .unwrap_or_default(),
            device_index: -1,
            active_engine: "whisper".into(),
            active_model_id: String::new(),
        }
    }
}

impl AppConfig {
    fn config_path() -> Option<PathBuf> {
        let home = dirs::home_dir()?;
        Some(home.join("Library/Application Support/scribe/config.toml"))
    }

    pub fn load() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };
        match std::fs::read_to_string(&path) {
            Ok(contents) => toml::from_str(&contents).unwrap_or_else(|e| {
                log::warn!("Bad config file, using defaults: {e}");
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let path = Self::config_path().ok_or("no home dir")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create dir: {e}"))?;
        }
        let contents = toml::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(&path, contents).map_err(|e| format!("write: {e}"))?;
        log::info!("Config saved to {}", path.display());
        Ok(())
    }
}
