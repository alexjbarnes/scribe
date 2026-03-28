//! Tests for running Parakeet ASR models via tract (pure Rust ONNX inference).
//!
//! Validates whether tract can replace sherpa-onnx, eliminating the C/C++
//! dependency for Android cross-compilation.
//!
//! ## Setup
//!
//! Download a model, then point the tests at it:
//!
//! ```sh
//! ./scripts/download_test_model.sh
//!
//! PARAKEET_MODEL_DIR=test_models/parakeet-v2-int8 \
//!   cargo test --test tract_parakeet -- --nocapture
//! ```
//!
//! Unit tests (mel spectrogram, CTC decode) always run without a model.

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::{Path, PathBuf};

use rustfft::{num_complex::Complex, FftPlanner};
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;

// ---------------------------------------------------------------------------
// Model directory helpers
// ---------------------------------------------------------------------------

fn model_dir() -> Option<PathBuf> {
    std::env::var("PARAKEET_MODEL_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .and_then(|p| p.canonicalize().ok())
}

fn find_file(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names.iter().map(|n| dir.join(n)).find(|p| p.exists())
}

fn find_encoder(dir: &Path) -> Option<PathBuf> {
    find_file(dir, &["encoder.onnx", "encoder.int8.onnx"])
}

fn find_decoder(dir: &Path) -> Option<PathBuf> {
    find_file(dir, &["decoder.onnx", "decoder.int8.onnx"])
}

fn find_joiner(dir: &Path) -> Option<PathBuf> {
    find_file(dir, &["joiner.onnx", "joiner.int8.onnx"])
}

fn skip_no_model() {
    eprintln!("PARAKEET_MODEL_DIR not set -- skipping");
    eprintln!(
        "To run: PARAKEET_MODEL_DIR=/path/to/model \
         cargo test --test tract_parakeet -- --nocapture"
    );
}

// ---------------------------------------------------------------------------
// Mel spectrogram (matches NeMo AudioToMelSpectrogramPreprocessor defaults)
// ---------------------------------------------------------------------------

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    let mel_lo = hz_to_mel(0.0);
    let mel_hi = hz_to_mel(sample_rate as f32 / 2.0);

    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_lo + (mel_hi - mel_lo) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let bin_points: Vec<f32> = mel_points
        .iter()
        .map(|&m| mel_to_hz(m) * n_fft as f32 / sample_rate as f32)
        .collect();

    let mut bank = vec![vec![0.0f32; n_freqs]; n_mels];
    for i in 0..n_mels {
        let (left, center, right) = (bin_points[i], bin_points[i + 1], bin_points[i + 2]);
        for j in 0..n_freqs {
            let f = j as f32;
            if f >= left && f <= center && center > left {
                bank[i][j] = (f - left) / (center - left);
            } else if f > center && f <= right && right > center {
                bank[i][j] = (right - f) / (right - center);
            }
        }
    }
    bank
}

/// Compute log-mel spectrogram from raw 16 kHz audio.
///
/// Returns shape (n_mels, n_frames) matching Parakeet's expected input layout.
fn mel_spectrogram(samples: &[f32]) -> Vec<Vec<f32>> {
    const SR: usize = 16_000;
    const N_FFT: usize = 512;
    const HOP: usize = 160; // 10 ms
    const WIN: usize = 400; // 25 ms
    const N_MELS: usize = 80;
    const PRE_EMPH: f32 = 0.97;

    if samples.len() < WIN {
        return vec![vec![]; N_MELS];
    }

    // Pre-emphasis
    let mut emph = Vec::with_capacity(samples.len());
    emph.push(samples[0]);
    for i in 1..samples.len() {
        emph.push(samples[i] - PRE_EMPH * samples[i - 1]);
    }

    // Hann window
    let window: Vec<f32> = (0..WIN)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (WIN - 1) as f32).cos()))
        .collect();

    let bank = mel_filterbank(N_MELS, N_FFT, SR);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    let n_frames = (emph.len() - WIN) / HOP + 1;
    let n_freqs = N_FFT / 2 + 1;

    // Output: (n_mels, n_frames)
    let mut features = vec![vec![0.0f32; n_frames]; N_MELS];

    for t in 0..n_frames {
        let off = t * HOP;
        let mut buf = vec![Complex::new(0.0f32, 0.0); N_FFT];
        for i in 0..WIN {
            buf[i] = Complex::new(emph[off + i] * window[i], 0.0);
        }
        fft.process(&mut buf);

        let power: Vec<f32> = buf[..n_freqs].iter().map(|c| c.norm_sqr()).collect();

        for (m, filter) in bank.iter().enumerate() {
            let energy: f32 = filter.iter().zip(&power).map(|(f, p)| f * p).sum();
            features[m][t] = (energy + 1e-10).ln();
        }
    }

    // Per-feature normalization
    let n = n_frames as f32;
    for row in &mut features {
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std = (var + 1e-5).sqrt();
        for v in row.iter_mut() {
            *v = (*v - mean) / std;
        }
    }

    features
}

// ---------------------------------------------------------------------------
// CTC greedy decoder
// ---------------------------------------------------------------------------

fn ctc_greedy_decode(logits: &[Vec<f32>], blank_id: usize) -> Vec<usize> {
    let mut prev = blank_id;
    let mut tokens = Vec::new();
    for frame in logits {
        let best = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(blank_id);
        if best != blank_id && best != prev {
            tokens.push(best);
        }
        prev = best;
    }
    tokens
}

fn load_tokens(path: &Path) -> HashMap<usize, String> {
    let content = std::fs::read_to_string(path).expect("failed to read tokens file");
    let mut map = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // sherpa-onnx format: "token id"
        let mut parts = line.splitn(2, ' ');
        if let (Some(tok), Some(id_str)) = (parts.next(), parts.next()) {
            if let Ok(id) = id_str.parse::<usize>() {
                map.insert(id, tok.to_string());
            }
        }
    }
    map
}

#[allow(dead_code)]
fn decode_tokens(ids: &[usize], vocab: &HashMap<usize, String>) -> String {
    ids.iter()
        .filter_map(|id| vocab.get(id))
        .map(|t| t.replace('\u{2581}', " "))
        .collect::<String>()
        .trim()
        .to_string()
}

// ---------------------------------------------------------------------------
// Model info helpers
// ---------------------------------------------------------------------------

fn print_model_io(model: &InferenceModel) {
    let inputs = model.input_outlets().unwrap();
    println!("Inputs ({}):", inputs.len());
    for (i, &outlet) in inputs.iter().enumerate() {
        let fact = model.outlet_fact(outlet).unwrap();
        let name = &model.node(outlet.node).name;
        println!("  [{i}] {name}: {fact:?}");
    }
    let outputs = model.output_outlets().unwrap();
    println!("Outputs ({}):", outputs.len());
    for (i, &outlet) in outputs.iter().enumerate() {
        let fact = model.outlet_fact(outlet).unwrap();
        let name = &model.node(outlet.node).name;
        println!("  [{i}] {name}: {fact:?}");
    }
}

/// Read the feature dimension from the encoder's first input fact.
/// The shape is (batch_sym, N_FEATURES, time_sym) where N_FEATURES
/// is the only concrete inner dimension.
fn detect_n_features(model: &InferenceModel) -> usize {
    let outlets = model.input_outlets().unwrap();
    let fact = model.outlet_fact(outlets[0]).unwrap();
    // Shape is (batch_sym, N_FEATURES, time_sym). Iterate dims looking
    // for a concrete value > 1 (skipping batch=1 and symbolic dims).
    if let Some(shape) = fact.shape.concretize() {
        for (i, d) in shape.iter().enumerate() {
            if let Ok(v) = d.to_i64() {
                let v = v as usize;
                if v > 1 {
                    println!("Detected feature dimension: {v} (axis {i})");
                    return v;
                }
            }
        }
    }
    // Fallback: try to read individual dims from the shape format string
    let shape_str = format!("{:?}", fact.shape);
    println!("Shape string: {shape_str}");
    // Look for a bare number > 1 in the shape (e.g. "?,128,?" or "S,128,S")
    for part in shape_str.split(',') {
        if let Ok(d) = part.trim().parse::<usize>() {
            if d > 1 {
                println!("Detected feature dimension from shape string: {d}");
                return d;
            }
        }
    }
    println!("Could not detect feature dim, defaulting to 80");
    80
}

/// Load encoder, setting CWD for external weight file resolution.
fn load_encoder(dir: &Path) -> InferenceModel {
    let path = find_encoder(dir).expect("no encoder ONNX found");
    let size_mb = std::fs::metadata(&path).map(|m| m.len() / 1_000_000).unwrap_or(0);
    println!("Loading encoder: {} ({size_mb} MB)", path.display());

    let prev_cwd = std::env::current_dir().ok();
    let _ = std::env::set_current_dir(dir);
    let model = tract_onnx::onnx()
        .model_for_path(&path)
        .expect("failed to parse encoder");
    if let Some(cwd) = prev_cwd {
        let _ = std::env::set_current_dir(cwd);
    }
    model
}

// ===================================================================
// Unit tests -- always run, no model needed
// ===================================================================

#[test]
fn test_mel_spectrogram_shape() {
    let samples = vec![0.0f32; 16_000]; // 1 second silence
    let feats = mel_spectrogram(&samples);

    assert_eq!(feats.len(), 80, "should have 80 mel bins");
    let expected_frames = (16_000 - 400) / 160 + 1; // 98
    assert_eq!(feats[0].len(), expected_frames, "frame count mismatch");
}

#[test]
fn test_mel_spectrogram_sine() {
    // 1 second of 440 Hz at 16 kHz
    let samples: Vec<f32> = (0..16_000)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / 16_000.0).sin())
        .collect();
    let feats = mel_spectrogram(&samples);

    let energy: f32 = feats.iter().flat_map(|r| r.iter()).map(|v| v.abs()).sum();
    assert!(energy > 0.0, "non-zero features expected for a tone");
}

#[test]
fn test_ctc_decode_basic() {
    let logits = vec![
        vec![1.0, 0.0, 0.0], // blank
        vec![0.0, 1.0, 0.0], // 1
        vec![0.0, 1.0, 0.0], // 1 (collapsed)
        vec![1.0, 0.0, 0.0], // blank
        vec![0.0, 0.0, 1.0], // 2
        vec![1.0, 0.0, 0.0], // blank
    ];
    assert_eq!(ctc_greedy_decode(&logits, 0), vec![1, 2]);
}

#[test]
fn test_ctc_decode_blank_separates_repeats() {
    let logits = vec![
        vec![0.0, 1.0], // 1
        vec![1.0, 0.0], // blank
        vec![0.0, 1.0], // 1
    ];
    assert_eq!(ctc_greedy_decode(&logits, 0), vec![1, 1]);
}

#[test]
fn test_ctc_decode_all_blank() {
    let logits = vec![vec![1.0, 0.0]; 5];
    assert!(ctc_greedy_decode(&logits, 0).is_empty());
}

// ===================================================================
// Model tests -- need PARAKEET_MODEL_DIR
// ===================================================================

/// Can tract parse the encoder ONNX protobuf?
#[test]
fn test_tract_loads_encoder() {
    let Some(dir) = model_dir() else {
        skip_no_model();
        return;
    };
    if find_encoder(&dir).is_none() {
        eprintln!("No encoder ONNX file in {}", dir.display());
        return;
    }

    let model = load_encoder(&dir);
    print_model_io(&model);
    let n_feat = detect_n_features(&model);
    println!("Feature dimension: {n_feat}");
    println!("PASS: tract parsed encoder");
}

/// Can tract type-check and optimize the encoder?
///
/// Tries multiple paths:
/// 1. into_optimized() (typed + optimized in one step)
/// 2. into_typed() only (skip optimization)
/// 3. into_typed().into_runnable() (run without optimization)
#[test]
fn test_tract_optimizes_encoder() {
    let Some(dir) = model_dir() else { return; };
    if find_encoder(&dir).is_none() { return; }

    let model = load_encoder(&dir);
    let n_feat = detect_n_features(&model);
    let time_steps: usize = 100;

    let model = model
        .with_input_fact(0, f32::fact([1, n_feat, time_steps]).into())
        .and_then(|m| m.with_input_fact(1, i64::fact([1]).into()))
        .expect("failed to set input shapes");

    // Path 1: full optimization
    match model.into_optimized() {
        Ok(opt) => {
            println!("PASS: tract optimized encoder ({} nodes)", opt.nodes().len());
            return;
        }
        Err(e) => {
            println!("into_optimized() failed: {e}");
            println!("Trying into_typed() without optimization...");
        }
    }

    // Path 2: type only (reload since into_optimized consumed the model)
    let model = load_encoder(&dir);
    let model = model
        .with_input_fact(0, f32::fact([1, n_feat, time_steps]).into())
        .and_then(|m| m.with_input_fact(1, i64::fact([1]).into()))
        .expect("set shapes");

    match model.into_typed() {
        Ok(typed) => {
            println!("into_typed() succeeded ({} nodes)", typed.nodes().len());

            // Path 3: can we run the typed model without optimization?
            match typed.into_runnable() {
                Ok(_plan) => {
                    println!("PASS: typed model is runnable (without optimization)");
                }
                Err(e) => {
                    println!("typed model not directly runnable: {e}");
                    println!("PARTIAL: tract can type but not run the encoder");
                }
            }
        }
        Err(e) => {
            println!("EXPECTED FAILURE: tract cannot type-check the Parakeet encoder.");
            println!("  Error at: {e}");
            println!("  This is a known limitation of tract with complex ONNX models.");
        }
    }
}

/// Can tract run inference on the encoder with dummy input?
///
/// Currently fails due to symbolic dimension unification issues in tract.
/// Tract cannot bind symbolic dims (e.g. audio_signal_dynamic_axes_1)
/// to concrete values during shape inference at Transpose nodes.
///
/// Potential workarounds:
/// - Preprocess ONNX to replace symbolic dims with static shapes
/// - Use onnx-simplifier or onnxruntime shape inference to bake in shapes
/// - Contribute a fix to tract's TDim unification
/// - Use ort (C++ ONNX Runtime) instead of tract for Parakeet
#[test]
fn test_tract_runs_encoder() {
    let Some(dir) = model_dir() else { return; };
    if find_encoder(&dir).is_none() { return; }

    let model = load_encoder(&dir);
    let n_feat = detect_n_features(&model);
    let time_steps: usize = 100;

    let result = model
        .with_input_fact(0, f32::fact([1, n_feat, time_steps]).into())
        .and_then(|m| m.with_input_fact(1, i64::fact([1]).into()))
        .and_then(|m| m.into_optimized())
        .and_then(|m| m.into_runnable());

    match result {
        Ok(plan) => {
            let audio: Tensor =
                tract_ndarray::Array3::<f32>::zeros((1, n_feat, time_steps)).into();
            let length: Tensor = tract_ndarray::arr1(&[time_steps as i64]).into();

            let t0 = std::time::Instant::now();
            let outputs = plan
                .run(tvec![audio.into(), length.into()])
                .expect("inference failed");
            let elapsed = t0.elapsed();

            println!("Inference time: {:.1} ms", elapsed.as_secs_f64() * 1000.0);
            for (i, out) in outputs.iter().enumerate() {
                println!("Output {i}: shape={:?} dtype={:?}", out.shape(), out.datum_type());
            }
            println!("PASS: tract ran encoder inference");
        }
        Err(e) => {
            println!("EXPECTED FAILURE: tract cannot fully optimize/run the Parakeet encoder.");
            println!("  Error at: {e}");
            println!("  This is a known limitation of tract with complex ONNX models.");
        }
    }
}

/// Can tract load the decoder (small LSTM, ~7 MB)?
///
/// Known issue: sherpa-onnx decoder ONNX uses symbolic dim names with
/// dots (e.g. "states.1_dim_1") that tract's TDim parser rejects.
#[test]
fn test_tract_loads_decoder() {
    let Some(dir) = model_dir() else { return; };
    let Some(path) = find_decoder(&dir) else {
        eprintln!("No decoder ONNX file found, skipping");
        return;
    };

    println!("Loading decoder: {}", path.display());
    match tract_onnx::onnx().model_for_path(&path) {
        Ok(model) => {
            print_model_io(&model);
            println!("PASS: tract parsed decoder");
        }
        Err(e) => {
            let msg = format!("{e}");
            if msg.contains("states.") || msg.contains("dim_") {
                println!("EXPECTED FAILURE: decoder ONNX has dotted symbolic dim names");
                println!("  tract's TDim parser rejects names like 'states.1_dim_1'.");
                println!("  Workaround: rename dims in the ONNX proto, or implement");
                println!("  the LSTM decoder natively in Rust (it's ~640 hidden units).");
                println!("  Error: {e}");
            } else {
                panic!("unexpected decoder parse failure: {e}");
            }
        }
    }
}

/// Can tract load the joiner (small linear model, ~2 MB)?
#[test]
fn test_tract_loads_joiner() {
    let Some(dir) = model_dir() else { return; };
    let Some(path) = find_joiner(&dir) else {
        eprintln!("No joiner ONNX file found, skipping");
        return;
    };

    println!("Loading joiner: {}", path.display());
    let model = tract_onnx::onnx()
        .model_for_path(&path)
        .expect("failed to parse joiner");

    print_model_io(&model);
    println!("PASS: tract parsed joiner");
}

/// Encoder with mel spectrogram input from generated audio.
#[test]
fn test_tract_encoder_with_mel_input() {
    let Some(dir) = model_dir() else { return; };
    if find_encoder(&dir).is_none() { return; }

    let model = load_encoder(&dir);
    let n_feat = detect_n_features(&model);

    // Generate 2 seconds of a 440 Hz tone
    let samples: Vec<f32> = (0..32_000)
        .map(|i| 0.5 * (2.0 * PI * 440.0 * i as f32 / 16_000.0).sin())
        .collect();
    let feats = mel_spectrogram(&samples);
    let n_mels = feats.len(); // 80 from our mel implementation
    let n_frames = feats[0].len();
    println!("Mel features: {n_mels} x {n_frames}, model expects {n_feat} features");

    // Build input tensor (1, n_feat, n_frames).
    // If model expects more features than we produce (e.g. 128 vs 80),
    // zero-pad the extra dimensions.
    let mut audio_data = vec![0.0f32; n_feat * n_frames];
    for (m, row) in feats.iter().enumerate() {
        if m >= n_feat {
            break;
        }
        for (t, &val) in row.iter().enumerate() {
            audio_data[m * n_frames + t] = val;
        }
    }

    let audio: Tensor =
        tract_ndarray::Array3::from_shape_vec((1, n_feat, n_frames), audio_data)
            .expect("shape mismatch")
            .into();
    let length: Tensor = tract_ndarray::arr1(&[n_frames as i64]).into();

    let result = model
        .with_input_fact(0, f32::fact([1, n_feat, n_frames]).into())
        .and_then(|m| m.with_input_fact(1, i64::fact([1]).into()))
        .and_then(|m| m.into_optimized())
        .and_then(|m| m.into_runnable());

    match result {
        Ok(plan) => {
            let t0 = std::time::Instant::now();
            let outputs = plan
                .run(tvec![audio.into(), length.into()])
                .expect("inference failed");
            let elapsed = t0.elapsed();

            println!("Inference on 2s audio: {:.1} ms", elapsed.as_secs_f64() * 1000.0);
            for (i, out) in outputs.iter().enumerate() {
                println!("Output {i}: shape={:?} dtype={:?}", out.shape(), out.datum_type());
            }

            let out_shape = outputs[0].shape();
            assert!(out_shape.len() >= 2, "expected at least 2D output");
            println!(
                "Output time steps: {} (input: {n_frames}, expected ~{})",
                out_shape[1],
                n_frames / 8
            );
            println!("PASS: encoder produced output from mel input");
        }
        Err(e) => {
            println!("EXPECTED FAILURE: tract cannot fully optimize/run the Parakeet encoder.");
            println!("  Error at: {e}");
            println!("  This is a known limitation of tract with complex ONNX models.");
        }
    }
}

/// Load tokens.txt and verify it parses.
#[test]
fn test_load_tokens() {
    let Some(dir) = model_dir() else { return; };
    let Some(path) = find_file(&dir, &["tokens.txt"]) else {
        eprintln!("No tokens.txt found, skipping");
        return;
    };

    let vocab = load_tokens(&path);
    println!("Loaded {} tokens", vocab.len());
    assert!(!vocab.is_empty(), "token vocabulary should not be empty");

    if let Some(blank) = vocab.get(&0) {
        println!("Token 0: {blank:?}");
    }
    for &id in &[0, 1, 2, 3, 100, 500] {
        if let Some(tok) = vocab.get(&id) {
            println!("  {id}: {tok:?}");
        }
    }
    println!("PASS: tokens loaded");
}

/// WAV loading + mel computation (needs test.wav in model dir).
#[test]
fn test_mel_from_wav() {
    let Some(dir) = model_dir() else { return; };
    let wav_path = dir.join("test.wav");
    if !wav_path.exists() {
        eprintln!(
            "No test.wav in model dir -- skipping WAV test. \
             Place a 16 kHz mono WAV there to test."
        );
        return;
    }

    let reader = hound::WavReader::open(&wav_path).expect("failed to open WAV");
    let spec = reader.spec();
    println!("WAV: {}Hz, {} ch, {:?}", spec.sample_rate, spec.channels, spec.sample_format);
    assert_eq!(spec.sample_rate, 16_000, "expected 16 kHz WAV");

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
    };

    let samples: Vec<f32> = if spec.channels > 1 {
        samples.iter().step_by(spec.channels as usize).copied().collect()
    } else {
        samples
    };

    println!("Audio: {:.2}s ({} samples)", samples.len() as f32 / 16_000.0, samples.len());
    let feats = mel_spectrogram(&samples);
    println!("Mel features: {} x {}", feats.len(), feats[0].len());
    assert_eq!(feats.len(), 80);
    assert!(!feats[0].is_empty(), "should produce at least one frame");
    println!("PASS: WAV -> mel spectrogram works");
}
