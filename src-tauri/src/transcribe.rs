use std::path::Path;
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::Instant;

struct Request {
    samples: Vec<f32>,
    sample_rate: i32,
    result_tx: mpsc::Sender<Result<String, String>>,
}

/// Transcription service backed by sherpa-onnx.
///
/// A dedicated worker thread owns the ONNX recognizer so the model loads
/// once and stays in memory across calls.
pub struct Transcriber {
    req_tx: mpsc::Sender<Request>,
    /// Prevents concurrent transcription calls.
    lock: Mutex<()>,
}

impl Transcriber {
    /// Validate model files and spawn the worker thread. The ONNX model
    /// is loaded lazily on the first transcription call.
    pub fn new(
        encoder: &str,
        decoder: &str,
        joiner: &str,
        tokens: &str,
    ) -> Result<Self, String> {
        for (name, path) in [
            ("encoder", encoder),
            ("decoder", decoder),
            ("joiner", joiner),
            ("tokens", tokens),
        ] {
            if !Path::new(path).exists() {
                return Err(format!("model file not found: {name} at {path}"));
            }
        }

        let model_dir = Path::new(encoder)
            .parent()
            .unwrap_or(Path::new("."))
            .to_string_lossy()
            .into_owned();

        let enc = encoder.to_string();
        let dec = decoder.to_string();
        let joi = joiner.to_string();
        let tok = tokens.to_string();

        let (req_tx, req_rx) = mpsc::channel::<Request>();

        log::info!("Transcriber ready (model dir: {model_dir})");

        std::thread::Builder::new()
            .name("transcriber".into())
            .spawn(move || {
                worker(req_rx, &model_dir, &enc, &dec, &joi, &tok);
            })
            .map_err(|e| format!("spawn transcriber thread: {e}"))?;

        Ok(Self {
            req_tx,
            lock: Mutex::new(()),
        })
    }

    pub fn transcribe(&self, samples: Vec<f32>, sample_rate: i32) -> Result<String, String> {
        let _guard = self.lock.lock().unwrap();

        let duration = samples.len() as f32 / sample_rate as f32;
        log::info!("Transcribing {duration:.1}s of audio...");

        let (result_tx, result_rx) = mpsc::channel();

        self.req_tx
            .send(Request {
                samples,
                sample_rate,
                result_tx,
            })
            .map_err(|_| "transcriber thread dead".to_string())?;

        result_rx
            .recv()
            .map_err(|_| "transcriber thread crashed".to_string())?
    }
}

fn worker(
    req_rx: mpsc::Receiver<Request>,
    model_dir: &str,
    encoder: &str,
    decoder: &str,
    joiner: &str,
    tokens: &str,
) {
    let _ = std::env::set_current_dir(model_dir);

    // Load model on first request
    let mut recognizer: Option<sherpa_onnx::OfflineRecognizer> = None;

    while let Ok(req) = req_rx.recv() {
        let total_start = Instant::now();

        // Lazy-load the recognizer
        if recognizer.is_none() {
            let load_start = Instant::now();

            let mut config = sherpa_onnx::OfflineRecognizerConfig::default();
            config.model_config.transducer.encoder = Some(encoder.to_string());
            config.model_config.transducer.decoder = Some(decoder.to_string());
            config.model_config.transducer.joiner = Some(joiner.to_string());
            config.model_config.tokens = Some(tokens.to_string());
            config.model_config.num_threads = 4;
            config.model_config.provider = Some("cpu".into());
            config.model_config.model_type = Some("nemo_transducer".into());

            match sherpa_onnx::OfflineRecognizer::create(&config) {
                Some(r) => {
                    let load_ms = load_start.elapsed().as_millis();
                    log::info!("Model loaded in {load_ms}ms");
                    recognizer = Some(r);
                }
                None => {
                    let _ = req.result_tx.send(Err("failed to create recognizer".into()));
                    continue;
                }
            }
        }

        let rec = recognizer.as_ref().unwrap();

        let decode_start = Instant::now();
        let stream = rec.create_stream();
        stream.accept_waveform(req.sample_rate, &req.samples);
        rec.decode(&stream);
        let decode_ms = decode_start.elapsed().as_millis();

        let text = stream
            .get_result()
            .map(|r| r.text.trim().to_string())
            .unwrap_or_default();

        let total_ms = total_start.elapsed().as_millis();
        let audio_duration = req.samples.len() as f32 / req.sample_rate as f32;
        let rtf = if audio_duration > 0.0 {
            total_ms as f32 / (audio_duration * 1000.0)
        } else {
            0.0
        };

        log::info!(
            "Transcription done: {total_ms}ms (decode: {decode_ms}ms), RTF: {rtf:.2}, text: \"{}\"",
            if text.len() > 80 { &text[..80] } else { &text }
        );

        let _ = req.result_tx.send(Ok(text));
    }
}
