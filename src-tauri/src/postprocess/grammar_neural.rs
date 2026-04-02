//! Stage 4 (neural path): grammar router + t5-efficient-tiny corrector.
//!
//! Model files are embedded at compile time from `data/grammar/`. If those
//! files are absent when the crate is built the entire stage compiles out
//! and the pipeline falls back to nlprule silently.
//!
//! Router:    pszemraj/electra-small-discriminator-CoLA  (~4ms, 14MB INT8)
//! Corrector: visheratin/t5-efficient-tiny-grammar-correction (~50ms, 44MB INT8)
//!
//! To enable: place the following files in src-tauri/data/grammar/ then rebuild:
//!   cola_model_quantized.onnx     — grammar router model (ELECTRA-based)
//!   cola_tokenizer.json           — grammar router tokenizer
//!   encoder_model_quantized.onnx  — T5 encoder
//!   decoder_model_quantized.onnx  — T5 decoder (greedy, no KV cache)
//!   t5_tokenizer.json             — SentencePiece tokenizer for T5
//!
//! Generate them with:
//!   python scripts/export_cola_onnx.py --output-dir src-tauri/data/grammar/
//!   python scripts/download_t5_grammar_onnx.py --output-dir src-tauri/data/grammar/

#[cfg(grammar_neural_bundled)]
mod bundled {
    use std::sync::{Mutex, OnceLock};

    use ndarray::{s, Array2, ArrayD};
    use ort::inputs;
    use ort::session::Session;
    use ort::value::TensorRef;
    use tokenizers::Tokenizer;

    static ROUTER_MODEL_BYTES: &[u8] =
        include_bytes!("../../data/grammar/cola_model_quantized.onnx");
    static ROUTER_TOKENIZER_BYTES: &[u8] = include_bytes!("../../data/grammar/cola_tokenizer.json");
    static ENC_MODEL_BYTES: &[u8] =
        include_bytes!("../../data/grammar/encoder_model_quantized.onnx");
    static DEC_MODEL_BYTES: &[u8] =
        include_bytes!("../../data/grammar/decoder_model_quantized.onnx");
    static T5_TOKENIZER_BYTES: &[u8] = include_bytes!("../../data/grammar/t5_tokenizer.json");

    static CHECKER: OnceLock<Option<GrammarNeuralChecker>> = OnceLock::new();

    pub fn global() -> Option<&'static GrammarNeuralChecker> {
        CHECKER.get().and_then(|o| o.as_ref())
    }

    pub fn init_global() {
        if CHECKER.get().is_some() {
            return;
        }
        let checker = GrammarNeuralChecker::load()
            .map_err(|e| log::warn!("Neural grammar failed to load ({e}), using nlprule"))
            .ok();
        let _ = CHECKER.set(checker);
    }

    pub struct GrammarNeuralChecker {
        router_session: Mutex<Session>,
        router_tokenizer: Tokenizer,
        t5_encoder: Mutex<Session>,
        t5_decoder: Mutex<Session>,
        t5_tokenizer: Tokenizer,
        eos_token_id: i64,
        decoder_start_token_id: i64,
    }

    /// P(acceptable) below this threshold → route to corrector.
    const ROUTER_THRESHOLD: f32 = 0.75;

    /// Extra token headroom above encoder input length for the decode loop.
    /// Output should be close to input length; 32 covers minor expansions.
    const DECODE_HEADROOM: usize = 32;

    /// Word count above which the corrector splits on sentence boundaries
    /// before running T5. T5-tiny is unreliable on long multi-sentence input
    /// and would truncate output when encoder length exceeds ~100 tokens.
    const SENTENCE_SPLIT_THRESHOLD: usize = 30;

    static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

    fn ensure_ort_init() -> Result<(), String> {
        ORT_INIT
            .get_or_init(|| {
                if ort::init().commit() {
                    Ok(())
                } else {
                    Err("ort init failed".to_string())
                }
            })
            .as_ref()
            .map(|_| ())
            .map_err(|e| e.clone())
    }

    impl GrammarNeuralChecker {
        fn load() -> Result<Self, String> {
            ensure_ort_init()?;

            let router_session = Session::builder()
                .map_err(|e| format!("session builder: {e}"))?
                .commit_from_memory(ROUTER_MODEL_BYTES)
                .map_err(|e| format!("router model: {e}"))?;

            let router_tokenizer = Tokenizer::from_bytes(ROUTER_TOKENIZER_BYTES)
                .map_err(|e| format!("router tokenizer: {e}"))?;

            let t5_encoder = Session::builder()
                .map_err(|e| format!("session builder: {e}"))?
                .commit_from_memory(ENC_MODEL_BYTES)
                .map_err(|e| format!("t5 encoder: {e}"))?;

            let t5_decoder = Session::builder()
                .map_err(|e| format!("session builder: {e}"))?
                .commit_from_memory(DEC_MODEL_BYTES)
                .map_err(|e| format!("t5 decoder: {e}"))?;

            let t5_tokenizer = Tokenizer::from_bytes(T5_TOKENIZER_BYTES)
                .map_err(|e| format!("t5 tokenizer: {e}"))?;

            log::info!("Neural grammar checker loaded (bundled)");
            Ok(Self {
                router_session: Mutex::new(router_session),
                router_tokenizer,
                t5_encoder: Mutex::new(t5_encoder),
                t5_decoder: Mutex::new(t5_decoder),
                t5_tokenizer,
                eos_token_id: 1,           // T5 </s>
                decoder_start_token_id: 0, // T5 <pad>
            })
        }

        /// Route and correct text. Returns (corrected, min_router_score).
        /// For long texts, routes and corrects each sentence independently so
        /// T5 only runs on sentences that actually need it.
        pub fn apply(&self, text: &str) -> (String, Option<f32>) {
            if text.split_whitespace().count() > SENTENCE_SPLIT_THRESHOLD {
                let sentences = Self::split_sentences(text);
                if sentences.len() > 1 {
                    let mut parts: Vec<String> = Vec::with_capacity(sentences.len());
                    let mut min_score: Option<f32> = None;
                    for s in &sentences {
                        let (needs_correction, score) = self.route(s.as_str());
                        if let Some(p) = score {
                            min_score = Some(min_score.map_or(p, |m: f32| m.min(p)));
                        }
                        if needs_correction {
                            parts.push(self.correct(s.as_str()));
                        } else {
                            parts.push(s.clone());
                        }
                    }
                    return (parts.join(" "), min_score);
                }
            }
            let (needs_correction, score) = self.route(text);
            let corrected = if needs_correction { self.correct(text) } else { text.to_string() };
            (corrected, score)
        }

        /// Returns (needs_correction, score). Score is None on error.
        pub fn route(&self, text: &str) -> (bool, Option<f32>) {
            match self.p_acceptable(text) {
                Ok(p) => {
                    log::debug!("Grammar router p(acceptable)={p:.3} threshold={ROUTER_THRESHOLD}");
                    (p < ROUTER_THRESHOLD, Some(p))
                }
                Err(e) => {
                    log::warn!("Grammar router error: {e}");
                    (false, None)
                }
            }
        }

        fn p_acceptable(&self, text: &str) -> Result<f32, String> {
            let enc = self
                .router_tokenizer
                .encode(text, true)
                .map_err(|e| format!("router encode: {e}"))?;

            let n = enc.get_ids().len();
            let input_ids = Array2::from_shape_vec(
                (1, n),
                enc.get_ids().iter().map(|&x| x as i64).collect(),
            )
            .map_err(|e| e.to_string())?;
            let attention_mask = Array2::from_shape_vec(
                (1, n),
                enc.get_attention_mask().iter().map(|&x| x as i64).collect(),
            )
            .map_err(|e| e.to_string())?;
            let token_type_ids = Array2::from_shape_vec(
                (1, n),
                enc.get_type_ids().iter().map(|&x| x as i64).collect(),
            )
            .map_err(|e| e.to_string())?;

            let ids_ref = TensorRef::<i64>::from_array_view(&input_ids)
                .map_err(|e| format!("ids tensor: {e}"))?;
            let mask_ref = TensorRef::<i64>::from_array_view(&attention_mask)
                .map_err(|e| format!("mask tensor: {e}"))?;
            let tids_ref = TensorRef::<i64>::from_array_view(&token_type_ids)
                .map_err(|e| format!("tids tensor: {e}"))?;

            let mut session = self.router_session.lock().unwrap();
            let out = session
                .run(inputs![
                    "input_ids"      => ids_ref,
                    "attention_mask" => mask_ref,
                    "token_type_ids" => tids_ref,
                ])
                .map_err(|e| format!("router run: {e}"))?;

            // logits shape [1, 2]: index 0 = not_acceptable, 1 = acceptable
            let logits = out
                .get("logits")
                .ok_or_else(|| format!("grammar router: no 'logits' output; got: {:?}", out.keys().collect::<Vec<_>>()))?
                .try_extract_array::<f32>()
                .map_err(|e| format!("extract logits: {e}"))?;
            let l0 = logits[[0, 0]];
            let l1 = logits[[0, 1]];
            let m = l0.max(l1);
            Ok(((l1 - m).exp()) / ((l0 - m).exp() + (l1 - m).exp()))
        }

        /// Run T5 correction. Returns the corrected text, or the original on error.
        pub fn correct(&self, text: &str) -> String {
            match self.correct_inner(text) {
                Ok(s) if !s.trim().is_empty() => s,
                Ok(_) => text.to_string(),
                Err(e) => {
                    log::warn!("T5 correction failed: {e}");
                    text.to_string()
                }
            }
        }

        /// Split on sentence boundaries: `. `, `! `, `? ` followed by an
        /// uppercase letter. Keeps punctuation with its sentence.
        fn split_sentences(text: &str) -> Vec<String> {
            let mut sentences = Vec::new();
            let mut buf = String::new();
            let mut chars = text.char_indices().peekable();
            while let Some((i, ch)) = chars.next() {
                buf.push(ch);
                if matches!(ch, '.' | '!' | '?') {
                    let rest = &text[i + ch.len_utf8()..];
                    let mut rc = rest.chars();
                    if rc.next() == Some(' ') && rc.next().map_or(false, |c| c.is_uppercase()) {
                        let s = buf.trim().to_string();
                        if !s.is_empty() { sentences.push(s); }
                        buf = String::new();
                    }
                }
            }
            let tail = buf.trim().to_string();
            if !tail.is_empty() { sentences.push(tail); }
            if sentences.is_empty() { sentences.push(text.trim().to_string()); }
            sentences
        }

        fn correct_inner(&self, text: &str) -> Result<String, String> {
            let enc = self
                .t5_tokenizer
                .encode(text, true)
                .map_err(|e| format!("t5 encode: {e}"))?;

            let n = enc.get_ids().len();
            let input_ids = Array2::from_shape_vec(
                (1, n),
                enc.get_ids().iter().map(|&x| x as i64).collect(),
            )
            .map_err(|e| e.to_string())?;
            let attention_mask = Array2::<i64>::from_elem((1, n), 1);

            let ids_ref = TensorRef::<i64>::from_array_view(&input_ids)
                .map_err(|e| format!("ids tensor: {e}"))?;
            let mask_ref = TensorRef::<i64>::from_array_view(&attention_mask)
                .map_err(|e| format!("mask tensor: {e}"))?;

            let mut encoder = self.t5_encoder.lock().unwrap();
            let enc_out = encoder
                .run(inputs![
                    "input_ids"      => ids_ref,
                    "attention_mask" => mask_ref,
                ])
                .map_err(|e| format!("encoder run: {e}"))?;

            let hidden: ArrayD<f32> = enc_out
                .get("hidden_states")
                .ok_or("encoder: no 'hidden_states' output")?
                .try_extract_array::<f32>()
                .map_err(|e| format!("encoder hidden: {e}"))?
                .into_owned();

            let hidden3 = hidden
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| format!("hidden dim: {e}"))?;

            let token_ids = self.decode_greedy(&hidden3, &attention_mask)?;

            self.t5_tokenizer
                .decode(&token_ids, true)
                .map_err(|e| format!("t5 decode: {e}"))
        }

        /// Greedy decode using only decoder_model.onnx (no KV cache).
        ///
        /// Each step re-runs the full decoder over all generated tokens so far.
        /// O(n²) in token count, but for typical short sentences this stays
        /// well within the ~50ms budget on ARM64.
        fn decode_greedy(
            &self,
            hidden: &ndarray::Array3<f32>,
            encoder_mask: &Array2<i64>,
        ) -> Result<Vec<u32>, String> {
            let mut tokens: Vec<i64> = vec![self.decoder_start_token_id];
            // Allow input length + headroom. Grammar correction output should
            // be close in length to the input; 2× would allow runaway generation.
            let limit = hidden.shape()[1] + DECODE_HEADROOM;

            for _ in 0..limit {
                let seq = tokens.len();
                let dec_input = Array2::from_shape_vec((1, seq), tokens.clone())
                    .map_err(|e| e.to_string())?;

                let dec_ref = TensorRef::<i64>::from_array_view(&dec_input)
                    .map_err(|e| format!("dec ids tensor: {e}"))?;
                let mask_ref = TensorRef::<i64>::from_array_view(encoder_mask)
                    .map_err(|e| format!("enc mask tensor: {e}"))?;
                let hidden_ref = TensorRef::<f32>::from_array_view(hidden)
                    .map_err(|e| format!("hidden tensor: {e}"))?;

                let mut decoder = self.t5_decoder.lock().unwrap();
                let out = decoder
                    .run(inputs![
                        "input_ids"              => dec_ref,
                        "encoder_attention_mask" => mask_ref,
                        "encoder_hidden_states"  => hidden_ref,
                    ])
                    .map_err(|e| format!("decoder run: {e}"))?;

                // logits shape [1, seq, vocab_size] — sample from last position
                let logits = out
                    .get("logits")
                    .ok_or_else(|| format!("decoder: no 'logits' output; got: {:?}", out.keys().collect::<Vec<_>>()))?
                    .try_extract_array::<f32>()
                    .map_err(|e| format!("logits: {e}"))?;
                let next = logits
                    .slice(s![0, seq - 1, ..])
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as i64)
                    .ok_or("empty logits")?;

                if next == self.eos_token_id {
                    break;
                }
                tokens.push(next);
            }

            // Drop the decoder start token.
            Ok(tokens[1..].iter().map(|&x| x as u32).collect())
        }
    }
}

#[cfg(grammar_neural_bundled)]
pub use bundled::{global, init_global, GrammarNeuralChecker};

#[cfg(not(grammar_neural_bundled))]
pub struct GrammarNeuralChecker;

#[cfg(not(grammar_neural_bundled))]
impl GrammarNeuralChecker {
    pub fn needs_correction(&self, _text: &str) -> bool { false }
    pub fn correct(&self, text: &str) -> String { text.to_string() }
}

#[cfg(not(grammar_neural_bundled))]
pub fn global() -> Option<&'static GrammarNeuralChecker> {
    None
}

#[cfg(not(grammar_neural_bundled))]
pub fn init_global() {}
