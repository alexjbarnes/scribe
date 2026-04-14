//! JNI bridge for the Android AccessibilityService overlay.
//!
//! Thin wrapper around the global Engine singleton. All transcription
//! logic lives in `engine.rs`. Both the Tauri app and the IME overlay
//! share the same engine instance, so the ONNX model is loaded once.

use jni::objects::{JClass, JString};
use jni::sys::{jboolean, jstring, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;

use crate::engine;
use crate::models::ModelManager;
use crate::recorder::AudioRecorder;
use crate::transcribe::Transcriber;

/// Initialize the native pipeline if not already done.
/// Called from VerbaAccessibilityService.onCreate(). If the Tauri app
/// already initialized the engine, this is a no-op.
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

    std::env::set_var("VERBA_DATA_DIR", &data_dir);
    let _ = std::fs::create_dir_all(&data_dir);

    // Engine already initialized by Tauri app (or a prior nativeInit call)
    if engine::is_initialized() {
        log::info!("Overlay: engine already initialized, reusing");
        return JNI_TRUE;
    }
    if !engine::try_claim_init() {
        log::info!("Overlay: engine being built by another thread, waiting");
        engine::wait_until_ready();
        return JNI_TRUE;
    }

    log::info!("Overlay: initializing engine (first entry point)");

    // Initialize singletons if Tauri app hasn't run yet
    if let Err(e) = ModelManager::init_global() {
        log::error!("Overlay: failed to create model manager: {e}");
        return JNI_FALSE;
    }
    crate::history::History::init_global();
    crate::snippets::SnippetManager::init_global();
    crate::postprocess::grammar_neural::init_global();

    let mgr = ModelManager::global();

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

    let (model_id, model_engine) = match mgr.first_downloaded_model() {
        Some(pair) => pair,
        None => {
            log::error!("Overlay: no transcription model downloaded");
            return JNI_FALSE;
        }
    };

    log::info!("Overlay: loading model: {model_id}");
    let transcriber = match Transcriber::new(model_engine) {
        Ok(t) => t,
        Err(e) => {
            log::error!("Overlay: failed to load transcriber: {e}");
            return JNI_FALSE;
        }
    };

    let eng = engine::Engine::new(recorder, transcriber, model_id);
    engine::init_global(eng);

    log::info!("Overlay: initialization complete");
    JNI_TRUE
}

/// Preload the transcription model by sending a tiny dummy request.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_verba_VerbaAccessibilityService_nativePreloadModel(
    _env: JNIEnv,
    _class: JClass,
) {
    engine::with(|eng| eng.preload());
}

/// Start recording with background VAD segment transcription.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_verba_VerbaAccessibilityService_nativeStartRecording(
    _env: JNIEnv,
    _class: JClass,
) -> jboolean {
    match engine::with_mut(|eng| eng.start_streaming()) {
        Some(Ok(())) => JNI_TRUE,
        Some(Err(e)) => {
            log::error!("Overlay: failed to start recording: {e}");
            JNI_FALSE
        }
        None => {
            log::error!("Overlay: engine not initialized");
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

    // Extract pending state while holding the engine lock, then release it
    let pending = match engine::with(|eng| eng.stop_recording()) {
        Some(Ok(p)) => p,
        Some(Err(e)) => {
            log::error!("Overlay: failed to stop recording: {e}");
            return null_ptr;
        }
        None => {
            log::error!("Overlay: engine not initialized");
            return null_ptr;
        }
    };

    // Lock released -- heavy transcription work happens without blocking
    match pending.finalize() {
        Some(result) => match env.new_string(&result.text) {
            Ok(s) => s.into_raw(),
            Err(_) => null_ptr,
        },
        None => null_ptr,
    }
}

/// Stop recording and return raw transcription text (no post-processing).
/// Used for snippet trigger matching where grammar correction is unwanted.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_verba_VerbaAccessibilityService_nativeStopAndTranscribeRaw(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    let null_ptr = std::ptr::null_mut();

    let pending = match engine::with(|eng| eng.stop_recording()) {
        Some(Ok(p)) => p,
        Some(Err(e)) => {
            log::error!("Overlay: failed to stop recording: {e}");
            return null_ptr;
        }
        None => {
            log::error!("Overlay: engine not initialized");
            return null_ptr;
        }
    };

    match pending.finalize_raw() {
        Some(text) => match env.new_string(&text) {
            Ok(s) => s.into_raw(),
            Err(_) => null_ptr,
        },
        None => null_ptr,
    }
}

/// Match transcribed text against snippets. Returns the snippet body if a
/// trigger matches (exact or fuzzy), otherwise returns null.
#[no_mangle]
pub extern "system" fn Java_com_alexb151_verba_VerbaAccessibilityService_nativeMatchSnippet(
    mut env: JNIEnv,
    _class: JClass,
    text: JString,
) -> jstring {
    let text: String = match env.get_string(&text) {
        Ok(s) => s.into(),
        Err(e) => {
            log::error!("Overlay: failed to read snippet text: {e}");
            return std::ptr::null_mut();
        }
    };

    let mgr = crate::snippets::SnippetManager::global();
    match mgr.find_match(&text) {
        Some(snippet) => {
            log::info!("Overlay: snippet matched: id={}", snippet.id);
            match env.new_string(&snippet.body) {
                Ok(s) => s.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }
        None => {
            log::info!("Overlay: no snippet match for \"{}\"", text);
            std::ptr::null_mut()
        }
    }
}

/// Return all snippets as a JSON array string: [{"id":"...","triggers":["..."],"body":"..."},...]
#[no_mangle]
pub extern "system" fn Java_com_alexb151_verba_VerbaAccessibilityService_nativeListSnippets(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    let mgr = crate::snippets::SnippetManager::global();
    let list = mgr.list();
    let json = serde_json::to_string(&list).unwrap_or_else(|_| "[]".to_string());
    match env.new_string(&json) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Register an additional trigger phrase for a snippet (self-healing).
#[no_mangle]
pub extern "system" fn Java_com_alexb151_verba_VerbaAccessibilityService_nativeAddSnippetTrigger(
    mut env: JNIEnv,
    _class: JClass,
    id: JString,
    trigger: JString,
) {
    let id: String = match env.get_string(&id) {
        Ok(s) => s.into(),
        Err(_) => return,
    };
    let trigger: String = match env.get_string(&trigger) {
        Ok(s) => s.into(),
        Err(_) => return,
    };
    let mgr = crate::snippets::SnippetManager::global();
    if let Err(e) = mgr.add_trigger(&id, trigger) {
        log::warn!("Overlay: add_trigger failed: {e}");
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
    log::info!("Overlay: destroying engine");
    engine::destroy();
}
