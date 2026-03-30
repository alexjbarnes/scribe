use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use futures_util::StreamExt;
use serde::Serialize;
use tauri::Emitter;
use tokio::io::AsyncWriteExt;

// ── Types ──

#[derive(Clone)]
pub struct ModelFile {
    pub url: String,
    pub rel_path: String,
    pub bytes: u64,
    pub role: String,
}

#[derive(Clone)]
pub struct ModelDef {
    pub id: String,
    pub name: String,
    pub desc: String,
    pub engine: String,
    pub size: String,
    pub files: Vec<ModelFile>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub desc: String,
    pub engine: String,
    pub size: String,
    pub status: String,
    pub progress: f64,
}

// ── Manager ──

pub struct ModelManager {
    pub base_dir: PathBuf,
    alt_dirs: Vec<PathBuf>,
    registry: Vec<ModelDef>,
    progress: Mutex<HashMap<String, f64>>,
    active_model: Mutex<String>,
}

impl ModelManager {
    pub fn new() -> Result<Self, String> {
        let base_dir = Self::default_base_dir()?;
        std::fs::create_dir_all(&base_dir).map_err(|e| format!("create models dir: {e}"))?;

        let mut alt_dirs = Vec::new();
        #[cfg(target_os = "macos")]
        {
            if let Some(home) = dirs::home_dir() {
                alt_dirs.push(home.join("Library/Application Support/com.meetily.ai/models"));
            }
        }

        // Restore active model from config
        let cfg = crate::config::AppConfig::load();
        let active = if !cfg.active_model_id.is_empty() {
            cfg.active_model_id.clone()
        } else {
            String::new()
        };

        Ok(Self {
            base_dir,
            alt_dirs,
            registry: builtin_registry(),
            progress: Mutex::new(HashMap::new()),
            active_model: Mutex::new(active),
        })
    }

    fn default_base_dir() -> Result<PathBuf, String> {
        #[cfg(target_os = "android")]
        {
            std::env::var_os("VERBA_DATA_DIR")
                .map(|d| PathBuf::from(d).join("models"))
                .ok_or_else(|| "VERBA_DATA_DIR not set".into())
        }
        #[cfg(not(target_os = "android"))]
        {
            dirs::data_dir()
                .map(|d| d.join("verba").join("models"))
                .ok_or_else(|| "no data dir".into())
        }
    }

    pub fn set_active(&self, id: &str) -> Result<(), String> {
        if !self.is_downloaded(id) {
            return Err("model not downloaded".into());
        }
        *self.active_model.lock().unwrap() = id.to_string();

        // Persist to config
        let mut cfg = crate::config::AppConfig::load();
        cfg.active_model_id = id.to_string();
        if let Err(e) = cfg.save() {
            log::warn!("Failed to persist active model: {e}");
        }
        Ok(())
    }

    pub fn active_model_id(&self) -> String {
        self.active_model.lock().unwrap().clone()
    }

    pub fn list(&self) -> Vec<ModelInfo> {
        let progress = self.progress.lock().unwrap();
        let active = self.active_model.lock().unwrap();
        self.registry
            .iter()
            .map(|m| {
                let (status, prog) = if *active == m.id {
                    ("active".into(), 1.0)
                } else if let Some(&p) = progress.get(&m.id) {
                    ("downloading".into(), p)
                } else if self.is_downloaded(&m.id) {
                    ("downloaded".into(), 1.0)
                } else {
                    ("not_downloaded".into(), 0.0)
                };
                ModelInfo {
                    id: m.id.clone(),
                    name: m.name.clone(),
                    desc: m.desc.clone(),
                    engine: m.engine.clone(),
                    size: m.size.clone(),
                    status,
                    progress: prog,
                }
            })
            .collect()
    }

    pub fn find(&self, id: &str) -> Option<&ModelDef> {
        self.registry.iter().find(|m| m.id == id)
    }

    /// Build a `ModelEngine` for a downloaded model.
    pub fn model_engine(&self, id: &str) -> Option<crate::transcribe::ModelEngine> {
        let model = self.find(id)?;
        if !self.is_downloaded(id) {
            return None;
        }

        let path = |role: &str| -> Option<String> {
            self.find_file_by_role(&model.files, role)
                .map(|p| p.to_string_lossy().into_owned())
        };

        match model.engine.as_str() {
            "parakeet" => Some(crate::transcribe::ModelEngine::Transducer {
                encoder: path("encoder")?,
                decoder: path("decoder")?,
                joiner: path("joiner")?,
                tokens: path("tokens")?,
            }),
            "whisper" => Some(crate::transcribe::ModelEngine::Whisper {
                encoder: path("encoder")?,
                decoder: path("decoder")?,
                tokens: path("tokens")?,
                language: "en".into(),
            }),
            _ => None,
        }
    }

    fn find_file_by_role(&self, files: &[ModelFile], role: &str) -> Option<std::path::PathBuf> {
        files
            .iter()
            .find(|f| f.role == role)
            .and_then(|f| self.find_file(&f.rel_path))
    }

    /// Find the best downloaded model. Checks the active model first,
    /// then falls back to preferred order (Parakeet INT8, then Whisper INT8).
    pub fn first_downloaded_model(&self) -> Option<(String, crate::transcribe::ModelEngine)> {
        // Check active model first
        {
            let active = self.active_model.lock().unwrap();
            if !active.is_empty() {
                if let Some(engine) = self.model_engine(&active) {
                    return Some((active.clone(), engine));
                }
            }
        }

        let preferred = [
            "parakeet-tdt-0.6b-v3-int8",
            "parakeet-tdt-0.6b-v2-int8",
            "parakeet-tdt-0.6b-v3",
            "parakeet-tdt-0.6b-v2",
            "whisper-small.en-int8",
            "whisper-base.en-int8",
            "whisper-turbo-int8",
            "whisper-medium.en-int8",
            "whisper-large-v3-int8",
        ];
        for id in preferred {
            if let Some(engine) = self.model_engine(id) {
                return Some((id.to_string(), engine));
            }
        }
        None
    }

    /// Backwards-compatible wrapper for code that only needs Parakeet paths.
    pub fn first_downloaded_parakeet(&self) -> Option<(String, (String, String, String, String))> {
        for model in &self.registry {
            if model.engine != "parakeet" || !self.is_downloaded(&model.id) {
                continue;
            }
            if let Some(crate::transcribe::ModelEngine::Transducer { encoder, decoder, joiner, tokens }) =
                self.model_engine(&model.id)
            {
                return Some((model.id.clone(), (encoder, decoder, joiner, tokens)));
            }
        }
        None
    }

    /// Path where the Silero VAD model should live.
    pub fn vad_model_path(&self) -> PathBuf {
        self.base_dir.join("silero_vad.onnx")
    }

    /// Ensure the Silero VAD model exists on disk, writing it from the
    /// embedded binary if needed. Returns the path to the ONNX file.
    pub fn ensure_vad_model(&self) -> Result<PathBuf, String> {
        let path = self.vad_model_path();
        if path.exists() {
            return Ok(path);
        }

        log::info!("Writing embedded Silero VAD model to {}", path.display());
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, SILERO_VAD_BYTES)
            .map_err(|e| format!("VAD write: {e}"))?;
        std::fs::rename(&tmp, &path)
            .map_err(|e| format!("VAD rename: {e}"))?;

        log::info!("Silero VAD model written ({} KB)", SILERO_VAD_BYTES.len() / 1024);
        Ok(path)
    }

    /// Path where the speaker embedding model should live.
    pub fn speaker_model_path(&self) -> PathBuf {
        self.base_dir.join("speaker_embed.onnx")
    }

    /// Ensure the speaker embedding model exists on disk, writing it from
    /// the embedded binary if needed. Returns the path to the ONNX file.
    pub fn ensure_speaker_model(&self) -> Result<PathBuf, String> {
        let path = self.speaker_model_path();
        if path.exists() {
            return Ok(path);
        }

        log::info!("Writing embedded speaker embedding model to {}", path.display());
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, SPEAKER_EMBED_BYTES)
            .map_err(|e| format!("Speaker model write: {e}"))?;
        std::fs::rename(&tmp, &path)
            .map_err(|e| format!("Speaker model rename: {e}"))?;

        log::info!("Speaker embedding model written ({} KB)", SPEAKER_EMBED_BYTES.len() / 1024);
        Ok(path)
    }

    pub fn delete(&self, id: &str) -> Result<(), String> {
        let model = self.find(id).ok_or("unknown model")?;

        // Clear active if deleting the active model
        {
            let mut active = self.active_model.lock().unwrap();
            if *active == id {
                *active = String::new();
                let mut cfg = crate::config::AppConfig::load();
                cfg.active_model_id = String::new();
                let _ = cfg.save();
            }
        }

        let mut deleted = 0u32;
        for file in &model.files {
            let path = self.base_dir.join(&file.rel_path);
            if path.exists() {
                std::fs::remove_file(&path).map_err(|e| format!("delete {}: {e}", file.rel_path))?;
                deleted += 1;
            }
        }

        // Clean up empty parent directories
        for file in &model.files {
            let path = self.base_dir.join(&file.rel_path);
            if let Some(parent) = path.parent() {
                let _ = Self::remove_empty_dirs(parent, &self.base_dir);
            }
        }

        log::info!("Deleted model {id} ({deleted} files)");
        Ok(())
    }

    /// Remove empty directories up to (but not including) `stop_at`.
    fn remove_empty_dirs(dir: &std::path::Path, stop_at: &std::path::Path) -> std::io::Result<()> {
        let mut current = dir.to_path_buf();
        while current != stop_at && current.starts_with(stop_at) {
            match std::fs::read_dir(&current) {
                Ok(mut entries) => {
                    if entries.next().is_none() {
                        std::fs::remove_dir(&current)?;
                    } else {
                        break;
                    }
                }
                Err(_) => break,
            }
            match current.parent() {
                Some(p) => current = p.to_path_buf(),
                None => break,
            }
        }
        Ok(())
    }

    pub fn is_downloaded(&self, id: &str) -> bool {
        let Some(model) = self.find(id) else {
            return false;
        };
        model.files.iter().all(|f| self.find_file(&f.rel_path).is_some())
    }

    fn find_file(&self, rel_path: &str) -> Option<PathBuf> {
        let p = self.base_dir.join(rel_path);
        if p.exists() {
            return Some(p);
        }
        for dir in &self.alt_dirs {
            let p = dir.join(rel_path);
            if p.exists() {
                return Some(p);
            }
        }
        None
    }

    pub async fn download(&self, id: &str, app: &tauri::AppHandle) -> Result<(), String> {
        let model = self.find(id).ok_or("unknown model")?.clone();
        let base_dir = self.base_dir.clone();

        // Init progress
        self.progress.lock().unwrap().insert(id.to_string(), 0.0);

        let total_bytes: u64 = model.files.iter().map(|f| f.bytes).sum();
        let mut downloaded: u64 = 0;
        let client = reqwest::Client::new();

        for file in &model.files {
            let dest = base_dir.join(&file.rel_path);
            if let Some(parent) = dest.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| format!("create dir: {e}"))?;
            }

            // Skip if already exists
            if dest.exists() {
                downloaded += file.bytes;
                continue;
            }

            let resp = client
                .get(&file.url)
                .send()
                .await
                .map_err(|e| format!("HTTP request: {e}"))?;

            if !resp.status().is_success() {
                self.progress.lock().unwrap().remove(id);
                return Err(format!("HTTP {}", resp.status()));
            }

            let tmp = dest.with_extension("tmp");
            let mut out = tokio::fs::File::create(&tmp)
                .await
                .map_err(|e| format!("create file: {e}"))?;

            let mut stream = resp.bytes_stream();
            let mut last_emit = Instant::now();

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| format!("download: {e}"))?;
                out.write_all(&chunk)
                    .await
                    .map_err(|e| format!("write: {e}"))?;
                downloaded += chunk.len() as u64;

                // Throttle progress updates to 500ms
                if total_bytes > 0 && last_emit.elapsed().as_millis() > 500 {
                    let pct = downloaded as f64 / total_bytes as f64;
                    self.progress.lock().unwrap().insert(id.to_string(), pct);
                    let _ = app.emit(
                        "download-progress",
                        serde_json::json!({ "id": id, "progress": pct }),
                    );
                    last_emit = Instant::now();
                }
            }

            out.flush().await.map_err(|e| format!("flush: {e}"))?;
            drop(out);
            tokio::fs::rename(&tmp, &dest)
                .await
                .map_err(|e| format!("rename: {e}"))?;
        }

        // Done
        self.progress.lock().unwrap().remove(id);
        let _ = app.emit("download-complete", serde_json::json!({ "id": id }));
        Ok(())
    }
}

// ── Registry ──

const SILERO_VAD_BYTES: &[u8] = include_bytes!("../silero_vad.onnx");
const SPEAKER_EMBED_BYTES: &[u8] = include_bytes!("../speaker_embed.onnx");

const HF_WHISPER_BASE_EN: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base.en/resolve/main";
const HF_WHISPER_SMALL_EN: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small.en/resolve/main";
const HF_WHISPER_MEDIUM_EN: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium.en/resolve/main";
const HF_WHISPER_LARGE_V3: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-large-v3/resolve/main";
const HF_WHISPER_TURBO: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-turbo/resolve/main";
const HF_PARAKEET_V2: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2/resolve/main";
const HF_PARAKEET_V2_INT8: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main";
const HF_PARAKEET_V3: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3/resolve/main";
const HF_PARAKEET_V3_INT8: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main";

fn builtin_registry() -> Vec<ModelDef> {
    vec![
        // Whisper INT8 (recommended)
        ModelDef {
            id: "whisper-base.en-int8".into(),
            name: "Whisper Base EN INT8".into(),
            desc: "English \u{2014} fastest, ~152 MB".into(),
            engine: "whisper".into(),
            size: "~152 MB".into(),
            files: vec![
                ModelFile { url: format!("{HF_WHISPER_BASE_EN}/base.en-encoder.int8.onnx"), rel_path: "whisper/base.en-int8/encoder.int8.onnx".into(), bytes: 28_000_000, role: "encoder".into() },
                ModelFile { url: format!("{HF_WHISPER_BASE_EN}/base.en-decoder.int8.onnx"), rel_path: "whisper/base.en-int8/decoder.int8.onnx".into(), bytes: 125_000_000, role: "decoder".into() },
                ModelFile { url: format!("{HF_WHISPER_BASE_EN}/base.en-tokens.txt"), rel_path: "whisper/base.en-int8/tokens.txt".into(), bytes: 816_000, role: "tokens".into() },
            ],
        },
        ModelDef {
            id: "whisper-small.en-int8".into(),
            name: "Whisper Small EN INT8".into(),
            desc: "English \u{2014} good accuracy, fast".into(),
            engine: "whisper".into(),
            size: "~357 MB".into(),
            files: vec![
                ModelFile { url: format!("{HF_WHISPER_SMALL_EN}/small.en-encoder.int8.onnx"), rel_path: "whisper/small.en-int8/encoder.int8.onnx".into(), bytes: 107_000_000, role: "encoder".into() },
                ModelFile { url: format!("{HF_WHISPER_SMALL_EN}/small.en-decoder.int8.onnx"), rel_path: "whisper/small.en-int8/decoder.int8.onnx".into(), bytes: 250_000_000, role: "decoder".into() },
                ModelFile { url: format!("{HF_WHISPER_SMALL_EN}/small.en-tokens.txt"), rel_path: "whisper/small.en-int8/tokens.txt".into(), bytes: 816_000, role: "tokens".into() },
            ],
        },
        ModelDef {
            id: "whisper-medium.en-int8".into(),
            name: "Whisper Medium EN INT8".into(),
            desc: "English \u{2014} balanced accuracy and speed".into(),
            engine: "whisper".into(),
            size: "~945 MB".into(),
            files: vec![
                ModelFile { url: format!("{HF_WHISPER_MEDIUM_EN}/medium.en-encoder.int8.onnx"), rel_path: "whisper/medium.en-int8/encoder.int8.onnx".into(), bytes: 374_000_000, role: "encoder".into() },
                ModelFile { url: format!("{HF_WHISPER_MEDIUM_EN}/medium.en-decoder.int8.onnx"), rel_path: "whisper/medium.en-int8/decoder.int8.onnx".into(), bytes: 571_000_000, role: "decoder".into() },
                ModelFile { url: format!("{HF_WHISPER_MEDIUM_EN}/medium.en-tokens.txt"), rel_path: "whisper/medium.en-int8/tokens.txt".into(), bytes: 816_000, role: "tokens".into() },
            ],
        },
        ModelDef {
            id: "whisper-large-v3-int8".into(),
            name: "Whisper Large V3 INT8".into(),
            desc: "Multilingual \u{2014} highest accuracy".into(),
            engine: "whisper".into(),
            size: "~1.8 GB".into(),
            files: vec![
                ModelFile { url: format!("{HF_WHISPER_LARGE_V3}/large-v3-encoder.int8.onnx"), rel_path: "whisper/large-v3-int8/encoder.int8.onnx".into(), bytes: 767_000_000, role: "encoder".into() },
                ModelFile { url: format!("{HF_WHISPER_LARGE_V3}/large-v3-decoder.int8.onnx"), rel_path: "whisper/large-v3-int8/decoder.int8.onnx".into(), bytes: 1_010_000_000, role: "decoder".into() },
                ModelFile { url: format!("{HF_WHISPER_LARGE_V3}/large-v3-tokens.txt"), rel_path: "whisper/large-v3-int8/tokens.txt".into(), bytes: 797_000, role: "tokens".into() },
            ],
        },
        ModelDef {
            id: "whisper-turbo-int8".into(),
            name: "Whisper Turbo INT8".into(),
            desc: "Multilingual \u{2014} near-large accuracy, 2x faster".into(),
            engine: "whisper".into(),
            size: "~1.0 GB".into(),
            files: vec![
                ModelFile { url: format!("{HF_WHISPER_TURBO}/turbo-encoder.int8.onnx"), rel_path: "whisper/turbo-int8/encoder.int8.onnx".into(), bytes: 675_000_000, role: "encoder".into() },
                ModelFile { url: format!("{HF_WHISPER_TURBO}/turbo-decoder.int8.onnx"), rel_path: "whisper/turbo-int8/decoder.int8.onnx".into(), bytes: 361_000_000, role: "decoder".into() },
                ModelFile { url: format!("{HF_WHISPER_TURBO}/turbo-tokens.txt"), rel_path: "whisper/turbo-int8/tokens.txt".into(), bytes: 797_000, role: "tokens".into() },
            ],
        },
        // Parakeet V3
        ModelDef {
            id: "parakeet-tdt-0.6b-v3".into(),
            name: "Parakeet TDT 0.6B V3".into(),
            desc: "Multilingual \u{2014} latest, full precision".into(),
            engine: "parakeet".into(),
            size: "~2.5 GB".into(),
            files: vec![
                ModelFile { url: format!("{HF_PARAKEET_V3}/encoder.onnx"), rel_path: "parakeet/v3/encoder.onnx".into(), bytes: 42_000_000, role: "encoder".into() },
                ModelFile { url: format!("{HF_PARAKEET_V3}/encoder.weights"), rel_path: "parakeet/v3/encoder.weights".into(), bytes: 2_435_000_000, role: "encoder_weights".into() },
                ModelFile { url: format!("{HF_PARAKEET_V3}/decoder.onnx"), rel_path: "parakeet/v3/decoder.onnx".into(), bytes: 47_000_000, role: "decoder".into() },
                ModelFile { url: format!("{HF_PARAKEET_V3}/joiner.onnx"), rel_path: "parakeet/v3/joiner.onnx".into(), bytes: 25_000_000, role: "joiner".into() },
                ModelFile { url: format!("{HF_PARAKEET_V3}/tokens.txt"), rel_path: "parakeet/v3/tokens.txt".into(), bytes: 94_000, role: "tokens".into() },
            ],
        },
        // Parakeet V3 INT8
        ModelDef {
            id: "parakeet-tdt-0.6b-v3-int8".into(),
            name: "Parakeet TDT 0.6B V3 INT8".into(),
            desc: "Multilingual \u{2014} quantized, smaller download".into(),
            engine: "parakeet".into(),
            size: "~670 MB".into(),
            files: vec![
                ModelFile { url: format!("{HF_PARAKEET_V3_INT8}/encoder.int8.onnx"), rel_path: "parakeet/v3-int8/encoder.int8.onnx".into(), bytes: 652_000_000, role: "encoder".into() },
                ModelFile { url: format!("{HF_PARAKEET_V3_INT8}/decoder.int8.onnx"), rel_path: "parakeet/v3-int8/decoder.int8.onnx".into(), bytes: 12_000_000, role: "decoder".into() },
                ModelFile { url: format!("{HF_PARAKEET_V3_INT8}/joiner.int8.onnx"), rel_path: "parakeet/v3-int8/joiner.int8.onnx".into(), bytes: 6_400_000, role: "joiner".into() },
                ModelFile { url: format!("{HF_PARAKEET_V3_INT8}/tokens.txt"), rel_path: "parakeet/v3-int8/tokens.txt".into(), bytes: 94_000, role: "tokens".into() },
            ],
        },
        // Parakeet V2
        ModelDef {
            id: "parakeet-tdt-0.6b-v2".into(),
            name: "Parakeet TDT 0.6B V2".into(),
            desc: "English only \u{2014} fast, production-ready".into(),
            engine: "parakeet".into(),
            size: "~1.2 GB".into(),
            files: vec![
                ModelFile { url: format!("{HF_PARAKEET_V2}/encoder.onnx"), rel_path: "parakeet/v2/encoder.onnx".into(), bytes: 1_200_000_000, role: "encoder".into() },
                ModelFile { url: format!("{HF_PARAKEET_V2}/decoder.onnx"), rel_path: "parakeet/v2/decoder.onnx".into(), bytes: 7_000_000, role: "decoder".into() },
                ModelFile { url: format!("{HF_PARAKEET_V2}/joiner.onnx"), rel_path: "parakeet/v2/joiner.onnx".into(), bytes: 2_000_000, role: "joiner".into() },
                ModelFile { url: format!("{HF_PARAKEET_V2}/tokens.txt"), rel_path: "parakeet/v2/tokens.txt".into(), bytes: 92_000, role: "tokens".into() },
            ],
        },
        // Parakeet V2 INT8
        ModelDef {
            id: "parakeet-tdt-0.6b-v2-int8".into(),
            name: "Parakeet TDT 0.6B V2 INT8".into(),
            desc: "English only \u{2014} quantized, smallest download".into(),
            engine: "parakeet".into(),
            size: "~630 MB".into(),
            files: vec![
                ModelFile { url: format!("{HF_PARAKEET_V2_INT8}/encoder.int8.onnx"), rel_path: "parakeet/v2-int8/encoder.int8.onnx".into(), bytes: 622_000_000, role: "encoder".into() },
                ModelFile { url: format!("{HF_PARAKEET_V2_INT8}/decoder.int8.onnx"), rel_path: "parakeet/v2-int8/decoder.int8.onnx".into(), bytes: 7_000_000, role: "decoder".into() },
                ModelFile { url: format!("{HF_PARAKEET_V2_INT8}/joiner.int8.onnx"), rel_path: "parakeet/v2-int8/joiner.int8.onnx".into(), bytes: 2_000_000, role: "joiner".into() },
                ModelFile { url: format!("{HF_PARAKEET_V2_INT8}/tokens.txt"), rel_path: "parakeet/v2-int8/tokens.txt".into(), bytes: 92_000, role: "tokens".into() },
            ],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_manager(dir: &std::path::Path) -> ModelManager {
        ModelManager {
            base_dir: dir.to_path_buf(),
            alt_dirs: vec![],
            registry: builtin_registry(),
            progress: Mutex::new(HashMap::new()),
            active_model: Mutex::new(String::new()),
        }
    }

    #[test]
    fn registry_has_whisper_and_parakeet() {
        let registry = builtin_registry();
        let whisper_count = registry.iter().filter(|m| m.engine == "whisper").count();
        let parakeet_count = registry.iter().filter(|m| m.engine == "parakeet").count();
        assert!(whisper_count >= 3, "expected at least 3 whisper models");
        assert!(parakeet_count >= 2, "expected at least 2 parakeet models");
    }

    #[test]
    fn registry_ids_are_unique() {
        let registry = builtin_registry();
        let mut ids: Vec<&str> = registry.iter().map(|m| m.id.as_str()).collect();
        let original_len = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), original_len, "duplicate model IDs in registry");
    }

    #[test]
    fn find_existing_model() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(dir.path());
        assert!(mgr.find("whisper-base.en-int8").is_some());
        assert!(mgr.find("parakeet-tdt-0.6b-v3-int8").is_some());
    }

    #[test]
    fn find_nonexistent_model() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(dir.path());
        assert!(mgr.find("nonexistent-model").is_none());
    }

    #[test]
    fn not_downloaded_when_files_missing() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(dir.path());
        assert!(!mgr.is_downloaded("whisper-base.en-int8"));
        assert!(!mgr.is_downloaded("parakeet-tdt-0.6b-v2-int8"));
    }

    #[test]
    fn downloaded_when_whisper_files_exist() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(dir.path());

        let model_dir = dir.path().join("whisper/base.en-int8");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("encoder.int8.onnx"), b"fake").unwrap();
        fs::write(model_dir.join("decoder.int8.onnx"), b"fake").unwrap();
        fs::write(model_dir.join("tokens.txt"), b"fake").unwrap();

        assert!(mgr.is_downloaded("whisper-base.en-int8"));
    }

    #[test]
    fn parakeet_requires_all_files() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(dir.path());

        // Create only encoder, not decoder/joiner/tokens
        let enc_dir = dir.path().join("parakeet/v2-int8");
        fs::create_dir_all(&enc_dir).unwrap();
        fs::write(enc_dir.join("encoder.int8.onnx"), b"fake").unwrap();

        assert!(!mgr.is_downloaded("parakeet-tdt-0.6b-v2-int8"));

        // Add the rest
        fs::write(enc_dir.join("decoder.int8.onnx"), b"fake").unwrap();
        fs::write(enc_dir.join("joiner.int8.onnx"), b"fake").unwrap();
        fs::write(enc_dir.join("tokens.txt"), b"fake").unwrap();

        assert!(mgr.is_downloaded("parakeet-tdt-0.6b-v2-int8"));
    }

    #[test]
    fn model_engine_returns_whisper() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(dir.path());

        let model_dir = dir.path().join("whisper/base.en-int8");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("encoder.int8.onnx"), b"fake").unwrap();
        fs::write(model_dir.join("decoder.int8.onnx"), b"fake").unwrap();
        fs::write(model_dir.join("tokens.txt"), b"fake").unwrap();

        let engine = mgr.model_engine("whisper-base.en-int8").unwrap();
        assert!(matches!(engine, crate::transcribe::ModelEngine::Whisper { .. }));
    }

    #[test]
    fn model_engine_returns_transducer() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(dir.path());

        let enc_dir = dir.path().join("parakeet/v2-int8");
        fs::create_dir_all(&enc_dir).unwrap();
        fs::write(enc_dir.join("encoder.int8.onnx"), b"fake").unwrap();
        fs::write(enc_dir.join("decoder.int8.onnx"), b"fake").unwrap();
        fs::write(enc_dir.join("joiner.int8.onnx"), b"fake").unwrap();
        fs::write(enc_dir.join("tokens.txt"), b"fake").unwrap();

        let engine = mgr.model_engine("parakeet-tdt-0.6b-v2-int8").unwrap();
        assert!(matches!(engine, crate::transcribe::ModelEngine::Transducer { .. }));
    }

    #[test]
    fn set_active_requires_downloaded() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(dir.path());
        assert!(mgr.set_active("whisper-base.en-int8").is_err());
    }

    #[test]
    fn list_shows_all_models() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(dir.path());
        let list = mgr.list();
        assert_eq!(list.len(), builtin_registry().len());
        assert!(list.iter().all(|m| m.status == "not_downloaded"));
    }

    #[test]
    fn list_shows_downloaded_status() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(dir.path());

        let model_dir = dir.path().join("whisper/base.en-int8");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("encoder.int8.onnx"), b"fake").unwrap();
        fs::write(model_dir.join("decoder.int8.onnx"), b"fake").unwrap();
        fs::write(model_dir.join("tokens.txt"), b"fake").unwrap();

        let list = mgr.list();
        let base = list.iter().find(|m| m.id == "whisper-base.en-int8").unwrap();
        assert_eq!(base.status, "downloaded");
    }

    #[test]
    fn vad_model_path() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(dir.path());
        assert_eq!(mgr.vad_model_path(), dir.path().join("silero_vad.onnx"));
    }

    #[test]
    fn first_downloaded_parakeet_prefers_int8() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(dir.path());

        // Create v2 (non-int8)
        let v2_dir = dir.path().join("parakeet/v2");
        fs::create_dir_all(&v2_dir).unwrap();
        fs::write(v2_dir.join("encoder.onnx"), b"fake").unwrap();
        fs::write(v2_dir.join("decoder.onnx"), b"fake").unwrap();
        fs::write(v2_dir.join("joiner.onnx"), b"fake").unwrap();
        fs::write(v2_dir.join("tokens.txt"), b"fake").unwrap();

        // Create v2-int8
        let v2i_dir = dir.path().join("parakeet/v2-int8");
        fs::create_dir_all(&v2i_dir).unwrap();
        fs::write(v2i_dir.join("encoder.int8.onnx"), b"fake").unwrap();
        fs::write(v2i_dir.join("decoder.int8.onnx"), b"fake").unwrap();
        fs::write(v2i_dir.join("joiner.int8.onnx"), b"fake").unwrap();
        fs::write(v2i_dir.join("tokens.txt"), b"fake").unwrap();

        let (id, _) = mgr.first_downloaded_parakeet().unwrap();
        assert_eq!(id, "parakeet-tdt-0.6b-v2-int8");
    }
}
