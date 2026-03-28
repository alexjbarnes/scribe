//! JNI bridge for the Android AccessibilityService overlay.
//!
//! The accessibility service loads libscribe_rs_lib.so and calls these
//! functions directly, bypassing the Tauri command system.

use std::sync::{Arc, Mutex};

use jni::objects::{JClass, JString};
use jni::sys::{jboolean, jstring, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;

use crate::models::ModelManager;
use crate::recorder::AudioRecorder;
use crate::transcribe::Transcriber;

struct OverlayState {
    recorder: AudioRecorder,
    transcriber: Arc<Transcriber>,
    model_id: String,
    segment_consumer: Mutex<Option<std::thread::JoinHandle<Vec<String>>>>,
    recording_start: Mutex<Option<std::time::Instant>>,
}

static OVERLAY_STATE: Mutex<Option<OverlayState>> = Mutex::new(None);

/// Initialize the native pipeline: model manager, recorder, transcriber.
/// Called from ScribeAccessibilityService.onCreate().
#[no_mangle]
pub extern "system" fn Java_com_alexb151_scribe_ScribeAccessibilityService_nativeInit(
    mut env: JNIEnv,
    _class: JClass,
    data_dir: JString,
) -> jboolean {
    let _ = log::set_logger(&crate::debug_log::LOGGER);
    log::set_max_level(log::LevelFilter::Debug);

    let data_dir: String = match env.get_string(&data_dir) {
        Ok(s) => s.into(),
        Err(e) => {
            log::error!("Overlay: failed to read data_dir string: {e}");
            return JNI_FALSE;
        }
    };

    log::info!("Overlay: initializing with data dir: {data_dir}");
    std::env::set_var("SCRIBE_DATA_DIR", &data_dir);
    let _ = std::fs::create_dir_all(&data_dir);

    let mgr = match ModelManager::new() {
        Ok(m) => m,
        Err(e) => {
            log::error!("Overlay: failed to create model manager: {e}");
            return JNI_FALSE;
        }
    };

    // Ensure VAD model is available (download if needed)
    let vad_path = mgr.vad_model_path();
    let vad = if vad_path.exists() {
        log::info!("Overlay: VAD model found at {}", vad_path.display());
        Some(vad_path)
    } else {
        log::info!("Overlay: VAD model not found, downloading...");
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build();
        match rt {
            Ok(rt) => match rt.block_on(mgr.ensure_vad_model()) {
                Ok(p) => {
                    log::info!("Overlay: VAD model downloaded");
                    Some(p)
                }
                Err(e) => {
                    log::warn!("Overlay: VAD download failed, continuing without: {e}");
                    None
                }
            },
            Err(e) => {
                log::warn!("Overlay: failed to create runtime for VAD download: {e}");
                None
            }
        }
    };
    let recorder = match AudioRecorder::new(vad.as_deref()) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Overlay: failed to create recorder: {e}");
            return JNI_FALSE;
        }
    };

    // Load transcription model
    let (model_id, transcriber) = match mgr.first_downloaded_model() {
        Some((id, engine)) => {
            log::info!("Overlay: loading model: {id}");
            match Transcriber::new(engine) {
                Ok(t) => (id, Arc::new(t)),
                Err(e) => {
                    log::error!("Overlay: failed to load transcriber: {e}");
                    return JNI_FALSE;
                }
            }
        }
        None => {
            log::error!("Overlay: no transcription model downloaded");
            return JNI_FALSE;
        }
    };

    *OVERLAY_STATE.lock().unwrap() = Some(OverlayState {
        recorder,
        transcriber,
        model_id,
        segment_consumer: Mutex::new(None),
        recording_start: Mutex::new(None),
    });

    log::info!("Overlay: initialization complete");
    JNI_TRUE
}

/// Preload the transcription model by sending a tiny dummy request.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_scribe_ScribeAccessibilityService_nativePreloadModel(
    _env: JNIEnv,
    _class: JClass,
) {
    let guard = OVERLAY_STATE.lock().unwrap();
    if let Some(ref state) = *guard {
        log::info!("Overlay: preloading model...");
        let silence = vec![0.0f32; 16000];
        match state.transcriber.transcribe(silence, 16000) {
            Ok(_) => log::info!("Overlay: model preloaded"),
            Err(e) => log::warn!("Overlay: preload returned error (expected for silence): {e}"),
        }
    }
}

/// Start recording with background VAD segment transcription.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_scribe_ScribeAccessibilityService_nativeStartRecording(
    _env: JNIEnv,
    _class: JClass,
) -> jboolean {
    let guard = OVERLAY_STATE.lock().unwrap();
    let Some(ref state) = *guard else {
        log::error!("Overlay: not initialized");
        return JNI_FALSE;
    };

    // Start streaming: VAD segments arrive on seg_rx during recording
    match state.recorder.start_streaming() {
        Ok(seg_rx) => {
            log::info!("Overlay: recording started (streaming segments)");
            *state.recording_start.lock().unwrap() = Some(std::time::Instant::now());

            // Spawn background thread to transcribe segments as they arrive
            let transcriber = state.transcriber.clone();
            let handle = std::thread::Builder::new()
                .name("segment-transcriber".into())
                .spawn(move || consume_segments(seg_rx, transcriber))
                .ok();
            *state.segment_consumer.lock().unwrap() = handle;

            JNI_TRUE
        }
        Err(e) => {
            log::error!("Overlay: failed to start recording: {e}");
            JNI_FALSE
        }
    }
}

/// Transcribe VAD speech segments as they arrive from the recorder.
fn consume_segments(
    seg_rx: std::sync::mpsc::Receiver<Vec<f32>>,
    transcriber: Arc<Transcriber>,
) -> Vec<String> {
    let mut texts = Vec::new();
    while let Ok(segment) = seg_rx.recv() {
        let duration = segment.len() as f32 / 16_000.0;
        if duration < 0.3 {
            continue;
        }
        log::info!("Overlay: transcribing segment ({duration:.1}s) in background");
        match transcriber.transcribe(segment, 16_000) {
            Ok(text) if !text.is_empty() => {
                log::info!("Overlay: segment: \"{text}\"");
                texts.push(text);
            }
            Ok(_) => {}
            Err(e) => log::warn!("Overlay: segment transcription error: {e}"),
        }
    }
    log::info!("Overlay: segment consumer done, {} chunks transcribed", texts.len());
    texts
}

/// Stop recording and return all transcribed text.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_scribe_ScribeAccessibilityService_nativeStopAndTranscribe(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    let null_ptr = std::ptr::null_mut();

    // Extract what we need and release the lock quickly
    let (samples, transcriber, consumer_handle, model_id, audio_duration_ms) = {
        let guard = OVERLAY_STATE.lock().unwrap();
        let Some(ref state) = *guard else {
            log::error!("Overlay: not initialized");
            return null_ptr;
        };

        let samples = match state.recorder.stop() {
            Ok(s) => s,
            Err(e) => {
                log::error!("Overlay: failed to stop recording: {e}");
                return null_ptr;
            }
        };

        // Recording duration: from start to right after recorder.stop()
        let audio_ms = state
            .recording_start
            .lock()
            .unwrap()
            .take()
            .map(|s| s.elapsed().as_millis() as u64)
            .unwrap_or(0);

        let transcriber = state.transcriber.clone();
        let handle = state.segment_consumer.lock().unwrap().take();
        let model_id = state.model_id.clone();
        (samples, transcriber, handle, model_id, audio_ms)
    };
    // OVERLAY_STATE lock released

    let transcribe_start = std::time::Instant::now();

    // Wait for background segment transcription to finish
    let mut all_texts: Vec<String> = Vec::new();
    if let Some(handle) = consumer_handle {
        match handle.join() {
            Ok(texts) => {
                log::info!("Overlay: got {} pre-transcribed chunks", texts.len());
                all_texts = texts;
            }
            Err(_) => log::warn!("Overlay: segment consumer thread panicked"),
        }
    }

    // Transcribe remaining tail (audio after the last VAD silence boundary)
    let duration = samples.len() as f32 / 16_000.0;
    if !samples.is_empty() && duration > 0.1 {
        log::info!("Overlay: transcribing tail ({duration:.1}s)");
        match transcriber.transcribe(samples, 16_000) {
            Ok(text) if !text.is_empty() => {
                log::info!("Overlay: tail: \"{text}\"");
                all_texts.push(text);
            }
            Ok(_) => {}
            Err(e) => log::error!("Overlay: tail transcription failed: {e}"),
        }
    }

    let transcribe_ms = transcribe_start.elapsed().as_millis() as u64;

    if all_texts.is_empty() {
        log::warn!("Overlay: no text produced");
        return null_ptr;
    }

    let full_text = all_texts.join(" ");
    log::info!(
        "Overlay: final text ({} chunks, recorded {}ms, transcribed {}ms): \"{}\"",
        all_texts.len(),
        audio_duration_ms,
        transcribe_ms,
        if full_text.len() > 60 { &full_text[..60] } else { &full_text }
    );

    // Record in history
    let history = crate::history::History::new();
    history.add(full_text.clone(), model_id, transcribe_ms, audio_duration_ms);

    match env.new_string(&full_text) {
        Ok(s) => s.into_raw(),
        Err(_) => null_ptr,
    }
}

/// Log a message from Kotlin through the Rust logger so it appears in the
/// in-app debug list alongside native logs.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_scribe_ScribeAccessibilityService_nativeLog(
    mut env: JNIEnv,
    _class: JClass,
    level: jni::sys::jint,
    msg: JString,
) {
    let msg: String = match env.get_string(&msg) {
        Ok(s) => s.into(),
        Err(_) => return,
    };
    match level {
        0 => log::error!("Overlay: {msg}"),
        1 => log::warn!("Overlay: {msg}"),
        2 => log::info!("Overlay: {msg}"),
        _ => log::debug!("Overlay: {msg}"),
    }
}

/// Clean up.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_scribe_ScribeAccessibilityService_nativeDestroy(
    _env: JNIEnv,
    _class: JClass,
) {
    log::info!("Overlay: destroying");
    *OVERLAY_STATE.lock().unwrap() = None;
}
