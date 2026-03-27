use std::path::Path;
use std::sync::mpsc;
use std::sync::Mutex;

struct Request {
    samples: Vec<f32>,
    sample_rate: i32,
}

/// Transcription service backed by sherpa-onnx.
/// The recognizer lives on a dedicated thread (it's !Send).
/// Communication happens via channels, making this struct Send + Sync.
pub struct Transcriber {
    tx: mpsc::Sender<Request>,
    rx: Mutex<mpsc::Receiver<String>>,
}

impl Transcriber {
    /// Create a new transcriber for a Parakeet TDT model.
    /// Blocks until the model is loaded (can take a few seconds).
    pub fn new(
        encoder: &str,
        decoder: &str,
        joiner: &str,
        tokens: &str,
    ) -> Result<Self, String> {
        let (req_tx, req_rx) = mpsc::channel::<Request>();
        let (res_tx, res_rx) = mpsc::channel::<String>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();

        let enc = encoder.to_string();
        let dec = decoder.to_string();
        let joi = joiner.to_string();
        let tok = tokens.to_string();

        std::thread::spawn(move || {
            // ONNX Runtime resolves external data files (e.g. encoder.weights)
            // relative to CWD. Change to the model directory so it finds them.
            let model_dir = Path::new(&enc).parent().unwrap_or(Path::new("."));
            let prev_cwd = std::env::current_dir().ok();
            if let Err(e) = std::env::set_current_dir(model_dir) {
                log::warn!("Could not set CWD to model dir: {e}");
            }

            let mut config = sherpa_onnx::OfflineRecognizerConfig::default();
            config.model_config.transducer.encoder = Some(enc);
            config.model_config.transducer.decoder = Some(dec);
            config.model_config.transducer.joiner = Some(joi);
            config.model_config.tokens = Some(tok);
            config.model_config.num_threads = 4;
            config.model_config.model_type = Some("nemo_transducer".into());

            let recognizer = match sherpa_onnx::OfflineRecognizer::create(&config) {
                Some(r) => r,
                None => {
                    let _ = ready_tx.send(Err("sherpa-onnx: failed to create recognizer".into()));
                    return;
                }
            };

            // Restore CWD
            if let Some(prev) = prev_cwd {
                let _ = std::env::set_current_dir(prev);
            }

            let _ = ready_tx.send(Ok(()));
            log::info!("Transcription model loaded");

            while let Ok(req) = req_rx.recv() {
                let stream = recognizer.create_stream();
                stream.accept_waveform(req.sample_rate, &req.samples);
                recognizer.decode(&stream);
                let text = stream
                    .get_result()
                    .map(|r| r.text.trim().to_string())
                    .unwrap_or_default();
                let _ = res_tx.send(text);
            }
        });

        // Wait for model to finish loading
        ready_rx
            .recv()
            .map_err(|e| format!("transcriber thread died: {e}"))??;

        Ok(Self {
            tx: req_tx,
            rx: Mutex::new(res_rx),
        })
    }

    /// Transcribe audio samples. Blocks until result is ready.
    pub fn transcribe(&self, samples: Vec<f32>, sample_rate: i32) -> Result<String, String> {
        self.tx
            .send(Request {
                samples,
                sample_rate,
            })
            .map_err(|e| format!("send failed: {e}"))?;

        self.rx
            .lock()
            .unwrap()
            .recv()
            .map_err(|e| format!("recv failed: {e}"))
    }
}
