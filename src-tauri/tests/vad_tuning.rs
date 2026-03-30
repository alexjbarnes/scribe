//! VAD parameter tuning tests.
//!
//! Feeds real audio through the VAD with different parameter combinations
//! and reports: number of segments, segment durations, tail size, and
//! total speech retained. Helps find the sweet spot between segment
//! frequency (more = shorter tail = faster final result) and segment
//! quality (fewer = less chunk-boundary artifacts).

use std::path::PathBuf;
use std::time::Instant;

use verba_rs_lib::vad::{Vad, VadParams};

const SAMPLE_RATE: f32 = 16_000.0;
// Feed audio in 32ms chunks (512 samples at 16kHz), matching the VAD window size
const CHUNK_SAMPLES: usize = 512;

fn vad_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("silero_vad.onnx")
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_wav(path: &std::path::Path) -> Vec<f32> {
    let reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));

    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "expected mono audio");
    assert_eq!(spec.sample_rate, 16000, "expected 16kHz audio");

    match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max)
                .collect()
        }
    }
}

struct VadResult {
    segments: Vec<f32>,      // duration in seconds of each segment
    tail_samples: usize,     // samples remaining after last segment
    total_speech_s: f32,     // total speech detected (segments + flushed)
    total_audio_s: f32,      // total audio fed in
    vad_time_ms: f64,        // wall clock time for VAD processing
}

fn run_vad(samples: &[f32], params: VadParams) -> VadResult {
    let model_path = vad_model_path();
    if !model_path.exists() {
        panic!("VAD model not found at {}. Run the app once or check the path.", model_path.display());
    }

    let mut vad = Vad::with_params(&model_path, params).expect("failed to create VAD");

    let start = Instant::now();
    let mut segments: Vec<Vec<f32>> = Vec::new();
    let mut samples_since_segment: usize = 0;

    for chunk in samples.chunks(CHUNK_SAMPLES) {
        if let Some(segment) = vad.accept(chunk) {
            segments.push(segment);
            samples_since_segment = 0;
        } else {
            samples_since_segment += chunk.len();
        }
    }

    // Flush remaining speech
    let mut flushed = false;
    if let Some(segment) = vad.flush() {
        flushed = true;
        segments.push(segment);
    }

    let vad_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let segment_durations: Vec<f32> = segments.iter()
        .map(|s| s.len() as f32 / SAMPLE_RATE)
        .collect();

    let total_speech_s: f32 = segment_durations.iter().sum();
    let total_audio_s = samples.len() as f32 / SAMPLE_RATE;

    // Tail = samples accumulated since the last segment that fired during
    // processing (before flush). This is what would need transcription
    // after the user stops recording.
    let _ = flushed;
    let tail_samples = samples_since_segment;

    VadResult {
        segments: segment_durations,
        tail_samples,
        total_speech_s,
        total_audio_s,
        vad_time_ms,
    }
}

fn print_result(label: &str, params: &VadParams, result: &VadResult) {
    let tail_s = result.tail_samples as f32 / SAMPLE_RATE;
    let mid_segments = if result.segments.is_empty() { 0 } else { result.segments.len() - 1 };

    println!("\n  {label}");
    println!("    threshold={:.2} min_silence={:.2}s min_speech={:.2}s prefill={}ms",
        params.threshold, params.min_silence_duration, params.min_speech_duration, params.prefill_ms);
    println!("    segments: {} fired during recording, {} from flush",
        mid_segments, if result.segments.len() > mid_segments { 1 } else { 0 });
    println!("    segment durations: [{}]",
        result.segments.iter().map(|d| format!("{d:.1}s")).collect::<Vec<_>>().join(", "));
    println!("    tail at stop: {tail_s:.1}s ({} samples) -- this is the post-stop transcription cost",
        result.tail_samples);
    println!("    total speech: {:.1}s / {:.1}s audio ({:.0}%)",
        result.total_speech_s, result.total_audio_s,
        result.total_speech_s / result.total_audio_s * 100.0);
    println!("    VAD processing: {:.1}ms ({:.2}x realtime)",
        result.vad_time_ms, result.vad_time_ms / (result.total_audio_s as f64 * 1000.0));
}

#[test]
fn vad_tuning_continuous_speech() {
    let wav_path = fixtures_dir().join("test_speech_16k.wav");
    if !wav_path.exists() {
        eprintln!("Skipping: no test audio at {}", wav_path.display());
        return;
    }
    let samples = load_wav(&wav_path);

    println!("\n=== Continuous speech ({:.1}s) ===", samples.len() as f32 / SAMPLE_RATE);

    let configs: Vec<(&str, VadParams)> = vec![
        ("Current defaults", VadParams {
            threshold: 0.4,
            min_silence_duration: 0.3,
            min_speech_duration: 0.1,
            prefill_ms: 300,
        }),
        ("Lower silence threshold (0.2s)", VadParams {
            min_silence_duration: 0.2,
            ..Default::default()
        }),
        ("Higher silence threshold (0.5s)", VadParams {
            min_silence_duration: 0.5,
            ..Default::default()
        }),
        ("Aggressive split (silence=0.15s)", VadParams {
            min_silence_duration: 0.15,
            ..Default::default()
        }),
        ("Sensitive detection (threshold=0.3)", VadParams {
            threshold: 0.3,
            min_silence_duration: 0.3,
            ..Default::default()
        }),
        ("Less sensitive (threshold=0.5)", VadParams {
            threshold: 0.5,
            min_silence_duration: 0.3,
            ..Default::default()
        }),
        ("Fast segments (silence=0.2, threshold=0.3)", VadParams {
            threshold: 0.3,
            min_silence_duration: 0.2,
            ..Default::default()
        }),
        ("Conservative (silence=0.5, threshold=0.5)", VadParams {
            threshold: 0.5,
            min_silence_duration: 0.5,
            ..Default::default()
        }),
    ];

    for (label, params) in &configs {
        let result = run_vad(&samples, VadParams {
            threshold: params.threshold,
            min_silence_duration: params.min_silence_duration,
            min_speech_duration: params.min_speech_duration,
            prefill_ms: params.prefill_ms,
        });
        print_result(label, params, &result);
    }
}

#[test]
fn vad_tuning_test_one() {
    let wav_path = fixtures_dir().join("test_one_16k.wav");
    if !wav_path.exists() {
        eprintln!("Skipping: no test audio at {}", wav_path.display());
        return;
    }
    let samples = load_wav(&wav_path);

    println!("\n=== Test One dictation ({:.1}s) ===", samples.len() as f32 / SAMPLE_RATE);

    let configs: Vec<(&str, VadParams)> = vec![
        ("Current defaults", VadParams::default()),
        ("Lower silence (0.2s)", VadParams {
            min_silence_duration: 0.2,
            ..Default::default()
        }),
        ("Higher silence (0.5s)", VadParams {
            min_silence_duration: 0.5,
            ..Default::default()
        }),
        ("Aggressive split (silence=0.15s)", VadParams {
            min_silence_duration: 0.15,
            ..Default::default()
        }),
        ("Sensitive detection (threshold=0.3)", VadParams {
            threshold: 0.3,
            min_silence_duration: 0.3,
            ..Default::default()
        }),
        ("Less sensitive (threshold=0.5)", VadParams {
            threshold: 0.5,
            min_silence_duration: 0.3,
            ..Default::default()
        }),
        ("Fast segments (silence=0.2, threshold=0.3)", VadParams {
            threshold: 0.3,
            min_silence_duration: 0.2,
            ..Default::default()
        }),
        ("Conservative (silence=0.5, threshold=0.5)", VadParams {
            threshold: 0.5,
            min_silence_duration: 0.5,
            ..Default::default()
        }),
    ];

    for (label, params) in &configs {
        let result = run_vad(&samples, VadParams {
            threshold: params.threshold,
            min_silence_duration: params.min_silence_duration,
            min_speech_duration: params.min_speech_duration,
            prefill_ms: params.prefill_ms,
        });
        print_result(label, params, &result);
    }
}

#[test]
fn vad_tuning_speech_with_pauses() {
    let wav_path = fixtures_dir().join("test_speech_with_pauses.wav");
    if !wav_path.exists() {
        eprintln!("Skipping: no test audio at {}", wav_path.display());
        return;
    }
    let samples = load_wav(&wav_path);

    println!("\n=== Speech with 0.5s pauses ({:.1}s) ===", samples.len() as f32 / SAMPLE_RATE);

    let configs: Vec<(&str, VadParams)> = vec![
        ("Current defaults", VadParams::default()),
        ("Lower silence (0.2s)", VadParams {
            min_silence_duration: 0.2,
            ..Default::default()
        }),
        ("Higher silence (0.5s)", VadParams {
            min_silence_duration: 0.5,
            ..Default::default()
        }),
        ("Aggressive (silence=0.15, threshold=0.3)", VadParams {
            threshold: 0.3,
            min_silence_duration: 0.15,
            ..Default::default()
        }),
    ];

    for (label, params) in &configs {
        let result = run_vad(&samples, VadParams {
            threshold: params.threshold,
            min_silence_duration: params.min_silence_duration,
            min_speech_duration: params.min_speech_duration,
            prefill_ms: params.prefill_ms,
        });
        print_result(label, params, &result);
    }
}
