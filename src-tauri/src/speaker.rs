//! Speaker verification using sherpa-onnx speaker embedding extraction.
//!
//! Extracts voice embeddings from audio segments and compares them against
//! an enrolled speaker to filter out non-target voices (TV, other people).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

const SAMPLE_RATE: i32 = 16_000;
const DEFAULT_THRESHOLD: f32 = 0.5;

pub struct SpeakerVerifier {
    extractor: SpeakerEmbeddingExtractor,
    state: Mutex<VerifierState>,
}

struct VerifierState {
    enrolled: Option<Vec<f32>>,
    threshold: f32,
}

// Safety: SpeakerEmbeddingExtractor wraps a sherpa-onnx C object that is
// thread-safe for compute operations (each call creates its own session state).
// The mutable state (enrolled embedding, threshold) is behind a Mutex.
unsafe impl Send for SpeakerVerifier {}
unsafe impl Sync for SpeakerVerifier {}

impl SpeakerVerifier {
    /// Create a new verifier from a speaker embedding ONNX model.
    pub fn new(model_path: &Path) -> Result<Self, String> {
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model_path.to_string_lossy().into_owned()),
            num_threads: 1,
            debug: false,
            provider: Some("cpu".into()),
        };

        let extractor = SpeakerEmbeddingExtractor::create(&config)
            .ok_or("failed to create speaker embedding extractor")?;

        log::info!("Speaker verifier created (embedding dim: {})", extractor.dim());

        Ok(Self {
            extractor,
            state: Mutex::new(VerifierState {
                enrolled: None,
                threshold: DEFAULT_THRESHOLD,
            }),
        })
    }

    /// Extract a speaker embedding from 16kHz mono audio samples.
    pub fn extract_embedding(&self, samples: &[f32]) -> Option<Vec<f32>> {
        let stream = self.extractor.create_stream()?;
        stream.accept_waveform(SAMPLE_RATE, samples);
        stream.input_finished();

        if !self.extractor.is_ready(&stream) {
            log::debug!("Speaker: audio too short for embedding");
            return None;
        }

        self.extractor.compute(&stream)
    }

    /// Enroll a speaker from audio samples. Extracts and stores the embedding.
    /// Multiple enrollments are averaged for better accuracy.
    pub fn enroll(&self, samples: &[f32]) -> Result<(), String> {
        let embedding = self.extract_embedding(samples)
            .ok_or("failed to extract enrollment embedding")?;

        let mut state = self.state.lock().unwrap();
        state.enrolled = match state.enrolled.take() {
            Some(existing) => {
                let averaged: Vec<f32> = existing.iter()
                    .zip(embedding.iter())
                    .map(|(a, b)| (a + b) / 2.0)
                    .collect();
                Some(averaged)
            }
            None => Some(embedding),
        };

        log::info!("Speaker enrolled (dim: {})", state.enrolled.as_ref().unwrap().len());
        Ok(())
    }

    /// Enroll from a pre-computed embedding vector (loaded from storage).
    pub fn enroll_from_embedding(&self, embedding: Vec<f32>) {
        self.state.lock().unwrap().enrolled = Some(embedding);
    }

    /// Get the enrolled embedding for persistence (cloned).
    pub fn enrolled_embedding(&self) -> Option<Vec<f32>> {
        self.state.lock().unwrap().enrolled.clone()
    }

    /// Clear enrollment (return to passthrough mode).
    pub fn clear_enrollment(&self) {
        self.state.lock().unwrap().enrolled = None;
        log::info!("Speaker enrollment cleared");
    }

    /// Check if a speaker is enrolled.
    pub fn is_enrolled(&self) -> bool {
        self.state.lock().unwrap().enrolled.is_some()
    }

    /// Set the similarity threshold (0.0 to 1.0). Higher = stricter matching.
    pub fn set_threshold(&self, threshold: f32) {
        self.state.lock().unwrap().threshold = threshold;
    }

    /// Verify whether audio samples match the enrolled speaker.
    /// Returns true if no speaker is enrolled (passthrough mode).
    pub fn verify(&self, samples: &[f32]) -> bool {
        let state = self.state.lock().unwrap();
        let Some(ref enrolled) = state.enrolled else {
            return true;
        };
        let threshold = state.threshold;
        let enrolled = enrolled.clone();
        drop(state);

        let Some(embedding) = self.extract_embedding(samples) else {
            return true;
        };

        let score = cosine_similarity(&enrolled, &embedding);
        log::debug!("Speaker: similarity = {score:.3} (threshold: {threshold:.3})");
        score >= threshold
    }

    /// Verify and return the similarity score. Returns None if extraction fails.
    pub fn verify_score(&self, samples: &[f32]) -> Option<f32> {
        let enrolled = self.state.lock().unwrap().enrolled.clone()?;
        let embedding = self.extract_embedding(samples)?;
        Some(cosine_similarity(&enrolled, &embedding))
    }

    /// Embedding dimension of the model.
    pub fn dim(&self) -> i32 {
        self.extractor.dim()
    }
}

/// Path where the speaker enrollment embedding is persisted.
fn enrollment_path() -> Option<PathBuf> {
    #[cfg(target_os = "android")]
    {
        std::env::var_os("VERBA_DATA_DIR")
            .map(|d| PathBuf::from(d).join("speaker_enrollment.bin"))
    }
    #[cfg(not(target_os = "android"))]
    {
        dirs::config_dir().map(|d| d.join("verba").join("speaker_enrollment.bin"))
    }
}

/// Save enrollment embedding to disk as raw f32 bytes.
pub fn save_enrollment(embedding: &[f32]) -> Result<(), String> {
    let path = enrollment_path().ok_or("no config dir")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create dir: {e}"))?;
    }
    let bytes: Vec<u8> = embedding.iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    std::fs::write(&path, bytes).map_err(|e| format!("write enrollment: {e}"))?;
    log::info!("Speaker enrollment saved to {}", path.display());
    Ok(())
}

/// Load enrollment embedding from disk.
pub fn load_enrollment() -> Option<Vec<f32>> {
    let path = enrollment_path()?;
    let bytes = std::fs::read(&path).ok()?;
    if bytes.len() % 4 != 0 {
        log::warn!("Speaker enrollment file has invalid size, ignoring");
        return None;
    }
    let embedding: Vec<f32> = bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    log::info!("Speaker enrollment loaded ({} dims)", embedding.len());
    Some(embedding)
}

/// Delete enrollment from disk.
pub fn delete_enrollment() -> Result<(), String> {
    let Some(path) = enrollment_path() else {
        return Ok(());
    };
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| format!("delete enrollment: {e}"))?;
        log::info!("Speaker enrollment deleted");
    }
    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut dot = 0.0f32;
    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        sum_a += x * x;
        sum_b += y * y;
    }

    let mag = (sum_a * sum_b).sqrt();
    if mag > 0.0 { dot / mag } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let score = cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let score = cosine_similarity(&a, &b);
        assert!((score + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
