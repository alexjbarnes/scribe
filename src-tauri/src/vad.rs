use std::path::Path;

use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

const SAMPLE_RATE: i32 = 16_000;
const WINDOW_SIZE: i32 = 512; // 32ms at 16kHz

/// Wraps sherpa-onnx's Silero VAD with a prefill ring buffer.
///
/// The prefill buffer keeps the last N milliseconds of audio so that when
/// speech is first detected, we can include the audio that preceded the
/// detection (avoiding clipped word beginnings).
pub struct Vad {
    detector: VoiceActivityDetector,
    prefill: PrefillBuffer,
}

/// Tunable VAD parameters.
pub struct VadParams {
    pub threshold: f32,
    pub min_silence_duration: f32,
    pub min_speech_duration: f32,
    pub prefill_ms: u32,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            min_silence_duration: 0.3,
            min_speech_duration: 0.1,
            prefill_ms: 500,
        }
    }
}

impl Vad {
    /// Create a new VAD with default parameters.
    pub fn new(model_path: &Path, prefill_ms: u32) -> Result<Self, String> {
        Self::with_params(model_path, VadParams { prefill_ms, ..Default::default() })
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

        let prefill_samples = (SAMPLE_RATE as u32 * params.prefill_ms / 1000) as usize;

        Ok(Self {
            detector,
            prefill: PrefillBuffer::new(prefill_samples),
        })
    }

    /// Feed a chunk of 16kHz mono audio. Returns accumulated speech samples
    /// when a complete speech segment is detected (speaker stopped talking).
    /// Returns None if no complete segment is ready yet.
    pub fn accept(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        self.prefill.push(samples);
        self.detector.accept_waveform(samples);

        if !self.detector.is_empty() {
            let segment = self.detector.front()?;
            let mut speech = self.prefill.drain();
            speech.extend_from_slice(segment.samples());
            self.detector.pop();

            // Drain any additional segments that arrived in the same chunk
            while let Some(seg) = self.detector.front() {
                speech.extend_from_slice(seg.samples());
                self.detector.pop();
            }

            self.prefill.clear();
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
        self.prefill.clear();
    }
}

/// Ring buffer that retains the last N samples of audio.
struct PrefillBuffer {
    buf: Vec<f32>,
    capacity: usize,
}

impl PrefillBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, samples: &[f32]) {
        if self.capacity == 0 {
            return;
        }
        self.buf.extend_from_slice(samples);
        if self.buf.len() > self.capacity {
            let excess = self.buf.len() - self.capacity;
            self.buf.drain(..excess);
        }
    }

    /// Take all buffered samples, leaving the buffer empty.
    fn drain(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.buf)
    }

    fn clear(&mut self) {
        self.buf.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_retains_last_n_samples() {
        let mut buf = PrefillBuffer::new(5);
        buf.push(&[1.0, 2.0, 3.0]);
        buf.push(&[4.0, 5.0, 6.0, 7.0]);
        // capacity=5, total pushed=7, should keep last 5
        assert_eq!(buf.drain(), vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn prefill_under_capacity() {
        let mut buf = PrefillBuffer::new(10);
        buf.push(&[1.0, 2.0]);
        buf.push(&[3.0]);
        assert_eq!(buf.drain(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn prefill_zero_capacity() {
        let mut buf = PrefillBuffer::new(0);
        buf.push(&[1.0, 2.0, 3.0]);
        assert_eq!(buf.drain(), Vec::<f32>::new());
    }

    #[test]
    fn prefill_drain_empties_buffer() {
        let mut buf = PrefillBuffer::new(5);
        buf.push(&[1.0, 2.0]);
        let _ = buf.drain();
        assert_eq!(buf.drain(), Vec::<f32>::new());
    }

    #[test]
    fn prefill_clear() {
        let mut buf = PrefillBuffer::new(5);
        buf.push(&[1.0, 2.0]);
        buf.clear();
        assert_eq!(buf.drain(), Vec::<f32>::new());
    }

    #[test]
    fn prefill_exact_capacity() {
        let mut buf = PrefillBuffer::new(3);
        buf.push(&[1.0, 2.0, 3.0]);
        assert_eq!(buf.drain(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn prefill_large_single_push() {
        let mut buf = PrefillBuffer::new(3);
        buf.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(buf.drain(), vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn prefill_ms_to_samples() {
        // 300ms at 16kHz = 4800 samples
        let samples = (16_000u32 * 300 / 1000) as usize;
        assert_eq!(samples, 4800);
    }
}
