use std::path::Path;

use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

const SAMPLE_RATE: i32 = 16_000;
const WINDOW_SIZE: i32 = 512; // 32ms at 16kHz

/// Wraps sherpa-onnx's Silero VAD.
///
/// sherpa-onnx's VoiceActivityDetector has a built-in onset lookback of
/// `2 * window_size + min_speech_duration_samples` (≈164ms with default
/// params). When speech is first detected, the segment is backdated to
/// include that pre-speech audio, so word beginnings are never clipped.
///
/// Do NOT add an external prefill on top of this. Audio chunks arrive in
/// 256-sample (16ms) pieces, but the VAD processes in 512-sample windows.
/// The first 16ms chunk feeds into the window accumulator and is also
/// included in the sherpa-onnx segment via its lookback. Adding that same
/// chunk to a manual prefill as well causes it to appear twice in the
/// audio fed to the transcriber, producing onset word repetition
/// (e.g. "What whatever you think" instead of "Whatever you think").
pub struct Vad {
    detector: VoiceActivityDetector,
}

/// Tunable VAD parameters.
pub struct VadParams {
    pub threshold: f32,
    pub min_silence_duration: f32,
    pub min_speech_duration: f32,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            min_silence_duration: 0.3,
            min_speech_duration: 0.1,
        }
    }
}

impl Vad {
    /// Create a new VAD with default parameters.
    pub fn new(model_path: &Path) -> Result<Self, String> {
        Self::with_params(model_path, VadParams::default())
    }

    /// Create a new VAD with custom parameters.
    pub fn with_params(model_path: &Path, params: VadParams) -> Result<Self, String> {
        let config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(model_path.to_string_lossy().into_owned()),
                threshold: params.threshold,
                min_silence_duration: params.min_silence_duration,
                min_speech_duration: params.min_speech_duration,
                window_size: WINDOW_SIZE,
                ..Default::default()
            },
            sample_rate: SAMPLE_RATE,
            num_threads: 1,
            ..Default::default()
        };

        let detector = VoiceActivityDetector::create(&config, 60.0)
            .ok_or("failed to create VAD")?;

        Ok(Self { detector })
    }

    /// Feed a chunk of 16kHz mono audio. Returns accumulated speech samples
    /// when a complete speech segment is detected (speaker stopped talking).
    /// Returns None if no complete segment is ready yet.
    pub fn accept(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        self.detector.accept_waveform(samples);

        if !self.detector.is_empty() {
            let segment = self.detector.front()?;
            let mut speech = segment.samples().to_vec();
            self.detector.pop();

            // Drain any additional segments that arrived in the same chunk
            while let Some(seg) = self.detector.front() {
                speech.extend_from_slice(seg.samples());
                self.detector.pop();
            }

            Some(speech)
        } else {
            None
        }
    }

    /// Flush any remaining speech at end of recording.
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        self.detector.flush();
        if self.detector.is_empty() {
            return None;
        }

        let mut speech = Vec::new();
        while let Some(seg) = self.detector.front() {
            speech.extend_from_slice(seg.samples());
            self.detector.pop();
        }

        if speech.is_empty() {
            None
        } else {
            Some(speech)
        }
    }

    /// Whether speech is currently being detected.
    pub fn is_speech(&self) -> bool {
        self.detector.detected()
    }

    /// Reset all state for a new recording session.
    pub fn reset(&mut self) {
        self.detector.reset();
    }
}
