//! JNI bridge for the Android AccessibilityService overlay.
//!
//! Thin wrapper around `Engine`. All transcription logic lives in `engine.rs`.

use std::sync::Mutex;

use jni::objects::{JClass, JString};
use jni::sys::{jboolean, jstring, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;

use crate::engine::Engine;
use crate::models::ModelManager;
use crate::recorder::AudioRecorder;
use crate::transcribe::Transcriber;

static OVERLAY: Mutex<Option<Engine>> = Mutex::new(None);

/// Initialize the native pipeline: model manager, recorder, transcriber.
/// Called from VerbaAccessibilityService.onCreate().
#[no_mangle]
pub extern "system" fn Java_com_alexb151_verba_VerbaAccessibilityService_nativeInit(
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
    std::env::set_var("VERBA_DATA_DIR", &data_dir);
    let _ = std::fs::create_dir_all(&data_dir);

    let mgr = match ModelManager::new() {
        Ok(m) => m,
        Err(e) => {
            log::error!("Overlay: failed to create model manager: {e}");
            return JNI_FALSE;
        }
    };

    let vad = match mgr.ensure_vad_model() {
        Ok(p) => {
            log::info!("Overlay: VAD model at {}", p.display());
            Some(p)
        }
        Err(e) => {
            log::warn!("Overlay: VAD setup failed, continuing without: {e}");
            None
        }
    };

    let recorder = match AudioRecorder::new(vad.as_deref()) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Overlay: failed to create recorder: {e}");
            return JNI_FALSE;
        }
    };

    let (model_id, transcriber) = match mgr.first_downloaded_model() {
        Some((id, engine)) => {
            log::info!("Overlay: loading model: {id}");
            match Transcriber::new(engine) {
                Ok(t) => (id, t),
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

    let verifier = match mgr.ensure_speaker_model() {
        Ok(p) => match crate::speaker::SpeakerVerifier::new(&p) {
            Ok(v) => Some(v),
            Err(e) => {
                log::warn!("Overlay: speaker verifier failed: {e}");
                None
            }
        },
        Err(e) => {
            log::warn!("Overlay: speaker model setup failed: {e}");
            None
        }
    };

    *OVERLAY.lock().unwrap() = Some(Engine::new(recorder, transcriber, model_id, verifier));
    log::info!("Overlay: initialization complete");
    JNI_TRUE
}

/// Preload the transcription model by sending a tiny dummy request.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_verba_VerbaAccessibilityService_nativePreloadModel(
    _env: JNIEnv,
    _class: JClass,
) {
    let guard = OVERLAY.lock().unwrap();
    if let Some(ref engine) = *guard {
        engine.preload();
    }
}

/// Reload the overlay's transcriber with the given model.
/// Called from `switch_model` Tauri command so the switch takes effect immediately.
pub fn reload_overlay_model(model_id: &str) {
    let mut guard = OVERLAY.lock().unwrap();
    let Some(ref mut engine) = *guard else {
        log::info!("Overlay: not initialized, skip model reload");
        return;
    };

    if engine.model_id() == model_id {
        return;
    }

    log::info!("Overlay: reloading model from {} to {model_id}", engine.model_id());

    let mgr = match ModelManager::new() {
        Ok(m) => m,
        Err(e) => {
            log::error!("Overlay: failed to create model manager for reload: {e}");
            return;
        }
    };

    if let Some((id, model_engine)) = mgr.first_downloaded_model() {
        log::info!("Overlay: loading model: {id}");
        match Transcriber::new(model_engine) {
            Ok(t) => {
                engine.reload_model(t, id);
                log::info!("Overlay: model reloaded");
            }
            Err(e) => {
                log::error!("Overlay: failed to reload model: {e}");
            }
        }
    }
}

/// Start recording with background VAD segment transcription.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_verba_VerbaAccessibilityService_nativeStartRecording(
    _env: JNIEnv,
    _class: JClass,
) -> jboolean {
    let guard = OVERLAY.lock().unwrap();
    let Some(ref engine) = *guard else {
        log::error!("Overlay: not initialized");
        return JNI_FALSE;
    };

    match engine.start_streaming() {
        Ok(()) => JNI_TRUE,
        Err(e) => {
            log::error!("Overlay: failed to start recording: {e}");
            JNI_FALSE
        }
    }
}

/// Stop recording and return all transcribed text.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_verba_VerbaAccessibilityService_nativeStopAndTranscribe(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    let null_ptr = std::ptr::null_mut();

    // Extract pending state while holding the lock, then release it
    let pending = {
        let guard = OVERLAY.lock().unwrap();
        let Some(ref engine) = *guard else {
            log::error!("Overlay: not initialized");
            return null_ptr;
        };
        match engine.stop_recording() {
            Ok(p) => p,
            Err(e) => {
                log::error!("Overlay: failed to stop recording: {e}");
                return null_ptr;
            }
        }
    };
    // Lock released -- heavy transcription work happens without blocking the overlay

    match pending.finalize() {
        Some(result) => match env.new_string(&result.text) {
            Ok(s) => s.into_raw(),
            Err(_) => null_ptr,
        },
        None => null_ptr,
    }
}

/// Log a message from Kotlin through the Rust logger.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_verba_VerbaAccessibilityService_nativeLog(
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
pub extern "system" fn Java_com_alexb151_verba_VerbaAccessibilityService_nativeDestroy(
    _env: JNIEnv,
    _class: JClass,
) {
    log::info!("Overlay: destroying");
    *OVERLAY.lock().unwrap() = None;
}
