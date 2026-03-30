//! Platform-agnostic transcription engine.
//!
//! Owns a recorder and transcriber, handles VAD streaming, background
//! segment transcription, chunk joining, post-processing, and history.
//! Platform-specific code (JNI, Tauri commands) wraps this.

use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

use crate::history::{ChunkTiming, History};
use crate::postprocess;
use crate::recorder::AudioRecorder;
use crate::speaker::SpeakerVerifier;
use crate::transcribe::Transcriber;

pub struct ChunkResult {
    pub text: String,
    pub audio_ms: u64,
    pub transcribe_ms: u64,
}

/// Tracks segments rejected by speaker verification.
pub struct SegmentConsumerResult {
    pub chunks: Vec<ChunkResult>,
    pub filtered_segments: u32,
    pub filtered_audio_ms: u64,
}

/// Final output from a transcription cycle.
pub struct TranscriptionResult {
    pub text: String,
    pub model_id: String,
    pub audio_duration_ms: u64,
    pub transcribe_ms: u64,
}

/// Intermediate state between stopping the recorder and finalizing transcription.
/// Designed to be created while holding a lock, then finalized after releasing it.
pub struct PendingTranscription {
    samples: Vec<f32>,
    transcriber: Arc<Transcriber>,
    verifier: Option<Arc<SpeakerVerifier>>,
    consumer_handle: Option<JoinHandle<SegmentConsumerResult>>,
    model_id: String,
    audio_duration_ms: u64,
}

impl PendingTranscription {
    /// Run the heavy work: wait for background segments, transcribe tail,
    /// join chunks, post-process, and record in history.
    pub fn finalize(self) -> Option<TranscriptionResult> {
        let transcribe_start = std::time::Instant::now();

        let mut filtered_segments: u32 = 0;
        let mut filtered_audio_ms: u64 = 0;

        // Wait for background segment transcription to finish
        let mut all_chunks: Vec<ChunkResult> = Vec::new();
        if let Some(handle) = self.consumer_handle {
            match handle.join() {
                Ok(result) => {
                    log::info!("Engine: got {} pre-transcribed chunks", result.chunks.len());
                    filtered_segments = result.filtered_segments;
                    filtered_audio_ms = result.filtered_audio_ms;
                    all_chunks = result.chunks;
                }
                Err(_) => log::warn!("Engine: segment consumer thread panicked"),
            }
        }

        // Transcribe remaining tail (audio after the last VAD silence boundary)
        let tail_audio_ms = (self.samples.len() as f64 / 16.0) as u64;
        let tail_speaker_ok = self.verifier.as_ref()
            .map(|v| v.verify(&self.samples))
            .unwrap_or(true);
        if !tail_speaker_ok {
            log::info!("Engine: tail rejected by speaker verification ({:.1}s)", tail_audio_ms as f64 / 1000.0);
            filtered_segments += 1;
            filtered_audio_ms += tail_audio_ms;
        }
        if !self.samples.is_empty() && tail_audio_ms > 100 && tail_speaker_ok {
            log::info!("Engine: transcribing tail ({:.1}s)", tail_audio_ms as f64 / 1000.0);
            let t = std::time::Instant::now();
            match self.transcriber.transcribe(self.samples, 16_000) {
                Ok(text) if !text.is_empty() => {
                    let transcribe_ms = t.elapsed().as_millis() as u64;
                    log::info!("Engine: tail: \"{text}\" ({transcribe_ms}ms)");
                    all_chunks.push(ChunkResult { text, audio_ms: tail_audio_ms, transcribe_ms });
                }
                Ok(_) => {}
                Err(e) => log::error!("Engine: tail transcription failed: {e}"),
            }
        }

        let total_transcribe_ms = transcribe_start.elapsed().as_millis() as u64;

        if all_chunks.is_empty() {
            log::warn!("Engine: no text produced");
            return None;
        }

        let chunk_timings: Vec<ChunkTiming> = all_chunks.iter()
            .map(|c| ChunkTiming { audio_ms: c.audio_ms, transcribe_ms: c.transcribe_ms })
            .collect();
        let all_texts: Vec<&str> = all_chunks.iter().map(|c| c.text.as_str()).collect();
        let raw_text = postprocess::join_chunks(&all_texts);

        let result = postprocess::postprocess(&raw_text);
        let full_text = result.text.clone();
        let postprocess_ms = result.total_ms;

        log::info!("Engine: final ({} chunks, {}ms audio, {}ms transcribe, {}ms postprocess, {} filtered): \"{}\"",
            all_chunks.len(), self.audio_duration_ms, total_transcribe_ms, postprocess_ms, filtered_segments,
            if full_text.len() > 60 { &full_text[..60] } else { &full_text });

        let history = History::new();
        history.add(
            full_text.clone(),
            self.model_id.clone(),
            total_transcribe_ms,
            self.audio_duration_ms,
            postprocess_ms,
            result.stages,
            chunk_timings,
            filtered_segments,
            filtered_audio_ms,
        );

        Some(TranscriptionResult {
            text: full_text,
            model_id: self.model_id,
            audio_duration_ms: self.audio_duration_ms,
            transcribe_ms: total_transcribe_ms,
        })
    }
}

pub struct Engine {
    recorder: AudioRecorder,
    transcriber: Arc<Transcriber>,
    verifier: Option<Arc<SpeakerVerifier>>,
    model_id: String,
    segment_consumer: Mutex<Option<JoinHandle<SegmentConsumerResult>>>,
    recording_start: Mutex<Option<std::time::Instant>>,
}

impl Engine {
    pub fn new(
        recorder: AudioRecorder,
        transcriber: Transcriber,
        model_id: String,
        verifier: Option<SpeakerVerifier>,
    ) -> Self {
        Self {
            recorder,
            transcriber: Arc::new(transcriber),
            verifier: verifier.map(Arc::new),
            model_id,
            segment_consumer: Mutex::new(None),
            recording_start: Mutex::new(None),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn verifier(&self) -> Option<&SpeakerVerifier> {
        self.verifier.as_deref()
    }

    pub fn verifier_arc(&self) -> Option<&Arc<SpeakerVerifier>> {
        self.verifier.as_ref()
    }

    pub fn recorder(&self) -> &AudioRecorder {
        &self.recorder
    }

    /// Replace the transcriber with a new model.
    pub fn reload_model(&mut self, transcriber: Transcriber, model_id: String) {
        self.transcriber = Arc::new(transcriber);
        self.model_id = model_id;
    }

    /// Warm up the model by running a dummy transcription.
    pub fn preload(&self) {
        log::info!("Engine: preloading model...");
        let silence = vec![0.0f32; 16000];
        match self.transcriber.transcribe(silence, 16000) {
            Ok(_) => log::info!("Engine: model preloaded"),
            Err(e) => log::warn!("Engine: preload error (expected for silence): {e}"),
        }
    }

    /// Start recording with background VAD segment transcription.
    pub fn start_streaming(&self) -> Result<(), String> {
        let seg_rx = self.recorder.start_streaming()?;
        log::info!("Engine: recording started (streaming segments)");
        *self.recording_start.lock().unwrap() = Some(std::time::Instant::now());

        let transcriber = self.transcriber.clone();
        let verifier = self.verifier.clone();
        let handle = std::thread::Builder::new()
            .name("segment-transcriber".into())
            .spawn(move || consume_segments(seg_rx, transcriber, verifier))
            .ok();
        *self.segment_consumer.lock().unwrap() = handle;

        Ok(())
    }

    /// Stop recording and extract state needed for transcription.
    /// Returns a PendingTranscription that can be finalized without holding
    /// any lock on the Engine.
    pub fn stop_recording(&self) -> Result<PendingTranscription, String> {
        let samples = self.recorder.stop()?;

        let audio_duration_ms = self.recording_start
            .lock().unwrap()
            .take()
            .map(|s| s.elapsed().as_millis() as u64)
            .unwrap_or(0);

        let transcriber = self.transcriber.clone();
        let verifier = self.verifier.clone();
        let handle = self.segment_consumer.lock().unwrap().take();
        let model_id = self.model_id.clone();

        Ok(PendingTranscription {
            samples,
            transcriber,
            verifier,
            consumer_handle: handle,
            model_id,
            audio_duration_ms,
        })
    }
}

/// Transcribe VAD speech segments as they arrive from the recorder.
fn consume_segments(
    seg_rx: mpsc::Receiver<Vec<f32>>,
    transcriber: Arc<Transcriber>,
    verifier: Option<Arc<SpeakerVerifier>>,
) -> SegmentConsumerResult {
    let mut chunks = Vec::new();
    let mut filtered_segments: u32 = 0;
    let mut filtered_audio_ms: u64 = 0;

    while let Ok(segment) = seg_rx.recv() {
        let audio_ms = (segment.len() as f64 / 16.0) as u64;
        if audio_ms < 300 {
            continue;
        }

        // Speaker verification: skip segments that don't match the enrolled speaker
        if let Some(ref v) = verifier {
            if !v.verify(&segment) {
                log::info!("Engine: segment rejected by speaker verification ({:.1}s)", audio_ms as f64 / 1000.0);
                filtered_segments += 1;
                filtered_audio_ms += audio_ms;
                continue;
            }
        }

        log::info!("Engine: transcribing segment ({:.1}s)", audio_ms as f64 / 1000.0);
        let t = std::time::Instant::now();
        match transcriber.transcribe(segment, 16_000) {
            Ok(text) if !text.is_empty() => {
                let transcribe_ms = t.elapsed().as_millis() as u64;
                log::info!("Engine: segment: \"{text}\" ({transcribe_ms}ms)");
                chunks.push(ChunkResult { text, audio_ms, transcribe_ms });
            }
            Ok(_) => {}
            Err(e) => log::warn!("Engine: segment transcription error: {e}"),
        }
    }
    log::info!("Engine: segment consumer done, {} chunks, {} filtered", chunks.len(), filtered_segments);
    SegmentConsumerResult { chunks, filtered_segments, filtered_audio_ms }
}
