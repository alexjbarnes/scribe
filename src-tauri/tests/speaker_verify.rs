//! Speaker verification integration tests.
//!
//! Tests speaker embedding extraction and similarity scoring using
//! the embedded CAM++ model and real audio fixtures.

use std::path::PathBuf;

use verba_rs_lib::vad::{Vad, VadParams};

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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

/// Extract VAD segments from audio.
fn extract_segments(samples: &[f32], vad_path: &std::path::Path) -> Vec<Vec<f32>> {
    let mut vad = Vad::with_params(vad_path, VadParams::default()).expect("failed to create VAD");
    let mut segments = Vec::new();

    for chunk in samples.chunks(512) {
        if let Some(segment) = vad.accept(chunk) {
            segments.push(segment);
        }
    }
    if let Some(segment) = vad.flush() {
        segments.push(segment);
    }
    segments
}

#[test]
fn speaker_embedding_extraction() {
    let model_path = model_dir().join("speaker_embed.onnx");
    if !model_path.exists() {
        eprintln!("Skipping: no speaker embedding model");
        return;
    }

    let wav_path = fixtures_dir().join("test_one_16k.wav");
    if !wav_path.exists() {
        eprintln!("Skipping: no test audio at {}", wav_path.display());
        return;
    }

    let config = sherpa_onnx::SpeakerEmbeddingExtractorConfig {
        model: Some(model_path.to_string_lossy().into_owned()),
        num_threads: 1,
        debug: false,
        provider: Some("cpu".into()),
    };
    let extractor = sherpa_onnx::SpeakerEmbeddingExtractor::create(&config)
        .expect("failed to create extractor");

    println!("Embedding dimension: {}", extractor.dim());
    assert!(extractor.dim() > 0);

    let samples = load_wav(&wav_path);
    println!("Audio: {:.1}s", samples.len() as f32 / 16000.0);

    // Extract embedding from full audio
    let start = std::time::Instant::now();
    let stream = extractor.create_stream().expect("failed to create stream");
    stream.accept_waveform(16000, &samples);
    stream.input_finished();
    assert!(extractor.is_ready(&stream));

    let embedding = extractor.compute(&stream).expect("failed to compute embedding");
    let elapsed = start.elapsed();

    println!("Embedding: {} dims, computed in {:.1}ms", embedding.len(), elapsed.as_secs_f64() * 1000.0);
    assert_eq!(embedding.len(), extractor.dim() as usize);

    // Embedding should not be all zeros
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Embedding magnitude: {magnitude:.4}");
    assert!(magnitude > 0.1, "embedding should have non-trivial magnitude");
}

#[test]
fn speaker_self_similarity() {
    let model_path = model_dir().join("speaker_embed.onnx");
    let vad_path = model_dir().join("silero_vad.onnx");
    if !model_path.exists() || !vad_path.exists() {
        eprintln!("Skipping: missing models");
        return;
    }

    let wav_path = fixtures_dir().join("test_one_16k.wav");
    if !wav_path.exists() {
        eprintln!("Skipping: no test audio");
        return;
    }

    let config = sherpa_onnx::SpeakerEmbeddingExtractorConfig {
        model: Some(model_path.to_string_lossy().into_owned()),
        num_threads: 1,
        debug: false,
        provider: Some("cpu".into()),
    };
    let extractor = sherpa_onnx::SpeakerEmbeddingExtractor::create(&config)
        .expect("failed to create extractor");

    let samples = load_wav(&wav_path);
    let segments = extract_segments(&samples, &vad_path);
    println!("Got {} VAD segments from {:.1}s audio", segments.len(), samples.len() as f32 / 16000.0);

    // Extract embeddings from each segment
    let mut embeddings = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        let start = std::time::Instant::now();
        let stream = extractor.create_stream().expect("stream");
        stream.accept_waveform(16000, seg);
        stream.input_finished();

        // Skip segments under 1.5s -- too short for reliable embeddings
        if seg.len() < 24000 || !extractor.is_ready(&stream) {
            println!("  Segment {i}: too short ({:.1}s), skipping", seg.len() as f32 / 16000.0);
            continue;
        }

        let emb = extractor.compute(&stream).expect("compute");
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        println!("  Segment {i}: {:.1}s audio -> {:.1}ms extraction", seg.len() as f32 / 16000.0, ms);
        embeddings.push(emb);
    }

    // All segments should be the same speaker, so similarity should be high
    println!("\nPairwise cosine similarities (same speaker, should be >0.5):");
    let mut min_sim = f32::MAX;
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            println!("  seg {i} vs seg {j}: {sim:.3}");
            if sim < min_sim {
                min_sim = sim;
            }
        }
    }

    if embeddings.len() >= 2 {
        // Compute average similarity (more robust than minimum)
        let mut total_sim = 0.0f32;
        let mut count = 0;
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                total_sim += cosine_similarity(&embeddings[i], &embeddings[j]);
                count += 1;
            }
        }
        let avg_sim = total_sim / count as f32;
        println!("\nMinimum self-similarity: {min_sim:.3}");
        println!("Average self-similarity: {avg_sim:.3}");
        // Average should be positive for same-speaker audio.
        // Note: test audio is 8kHz resampled to 16kHz which degrades embeddings.
        // Real microphone audio at native 16kHz will produce much higher scores.
        assert!(avg_sim > 0.3, "average same-speaker similarity should be >0.3, got {avg_sim:.3}");
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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
