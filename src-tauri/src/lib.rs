mod audio;
mod config;
mod debug_log;
mod engine;
mod history;
mod models;
#[cfg(desktop)]
mod paste;
mod postprocess;
mod recorder;
pub mod speaker;
mod transcribe;
pub mod vad;
#[cfg(target_os = "android")]
mod android_ime;

use tauri::{Emitter, Manager};

struct AppState {
    engine: std::sync::Mutex<Option<engine::Engine>>,
    recording: std::sync::atomic::AtomicBool,
}

#[tauri::command]
fn list_models(state: tauri::State<'_, models::ModelManager>) -> Vec<models::ModelInfo> {
    state.list()
}

#[tauri::command]
fn list_audio_devices() -> Vec<audio::AudioDevice> {
    audio::list_input_devices()
}

#[tauri::command]
fn get_config() -> config::AppConfig {
    config::AppConfig::load()
}

#[tauri::command]
fn save_config(cfg: config::AppConfig) -> Result<(), String> {
    cfg.save()
}

#[tauri::command]
fn list_history(state: tauri::State<'_, history::History>) -> Vec<history::HistoryEntry> {
    state.list()
}

#[tauri::command]
fn clear_history(state: tauri::State<'_, history::History>) {
    state.clear()
}

/// Start a short enrollment recording. Creates a temporary recorder and
/// speaker verifier, records 5 seconds, extracts and persists the embedding.
#[tauri::command]
fn enroll_speaker(app: tauri::AppHandle) -> Result<(), String> {
    let mgr = app.state::<models::ModelManager>();
    let speaker_model = mgr.ensure_speaker_model()?;
    let verifier = speaker::SpeakerVerifier::new(&speaker_model)?;

    let vad = mgr.ensure_vad_model().ok();
    let rec = recorder::AudioRecorder::new(vad.as_deref())?;

    let seg_rx = rec.start_streaming()?;

    // Record 5 seconds of audio for a stable embedding.
    std::thread::sleep(std::time::Duration::from_secs(5));

    let tail = rec.stop()?;

    // Collect all audio (VAD segments + tail)
    let mut all_samples = Vec::new();
    while let Ok(segment) = seg_rx.try_recv() {
        all_samples.extend_from_slice(&segment);
    }
    all_samples.extend_from_slice(&tail);

    // Need at least ~2 seconds of speech for a usable embedding
    if all_samples.len() < 32_000 {
        return Err("Not enough speech detected. Please speak clearly for the full duration.".into());
    }

    verifier.enroll(&all_samples)?;

    if let Some(embedding) = verifier.enrolled_embedding() {
        speaker::save_enrollment(&embedding)?;
    }

    // Update the app engine's in-memory verifier
    let state = app.state::<AppState>();
    let guard = state.engine.lock().unwrap();
    if let Some(ref eng) = *guard {
        if let Some(v) = eng.verifier() {
            if let Some(emb) = speaker::load_enrollment() {
                v.enroll_from_embedding(emb);
            }
        }
    }

    log::info!("Speaker enrolled and saved");
    Ok(())
}

#[tauri::command]
fn clear_speaker_enrollment(app: tauri::AppHandle) -> Result<(), String> {
    let state = app.state::<AppState>();
    let guard = state.engine.lock().unwrap();
    if let Some(ref eng) = *guard {
        if let Some(verifier) = eng.verifier() {
            verifier.clear_enrollment();
        }
    }
    drop(guard);
    speaker::delete_enrollment()?;
    Ok(())
}

#[tauri::command]
fn get_speaker_enrollment_status() -> bool {
    speaker::load_enrollment().is_some()
}

#[tauri::command]
fn switch_model(
    app: tauri::AppHandle,
    state: tauri::State<'_, models::ModelManager>,
    id: String,
) -> Result<(), String> {
    state.set_active(&id)?;
    log::info!("Switched to model: {id}");

    let app = app.clone();
    let id = id.clone();
    std::thread::spawn(move || {
        // Reload the app engine's transcriber
        let app_state = app.state::<AppState>();
        let mut guard = app_state.engine.lock().unwrap();
        if let Some(ref mut eng) = *guard {
            if eng.model_id() != id {
                log::info!("Reloading model to {id}");
                let mgr = match models::ModelManager::new() {
                    Ok(m) => m,
                    Err(e) => {
                        log::error!("Model manager error on reload: {e}");
                        return;
                    }
                };
                if let Some((_mid, model_engine)) = mgr.first_downloaded_model() {
                    match transcribe::Transcriber::new(model_engine) {
                        Ok(t) => eng.reload_model(t, id.clone()),
                        Err(e) => log::error!("Transcriber reload failed: {e}"),
                    }
                }
            }
        }
        drop(guard);

        // On Android, also reload the IME overlay's separate engine
        #[cfg(target_os = "android")]
        android_ime::reload_overlay_model(&id);

        let _ = app.emit("model-loaded", serde_json::json!({ "id": &id }));
    });

    Ok(())
}

#[tauri::command]
async fn download_model(app: tauri::AppHandle, id: String) -> Result<(), String> {
    let mgr = app.state::<models::ModelManager>();

    if mgr.is_downloaded(&id) {
        return Ok(());
    }

    mgr.download(&id, &app).await
}

#[tauri::command]
fn delete_model(state: tauri::State<'_, models::ModelManager>, id: String) -> Result<(), String> {
    state.delete(&id)
}

/// Initialize the Engine in the background: VAD, recorder, transcriber,
/// speaker verifier, then preload model + post-processing pipeline.
fn init_engine(app: tauri::AppHandle) {
    std::thread::Builder::new()
        .name("engine-init".into())
        .spawn(move || {
            let mgr = match models::ModelManager::new() {
                Ok(m) => m,
                Err(e) => {
                    log::error!("Engine init: failed to create model manager: {e}");
                    return;
                }
            };

            let vad = match mgr.ensure_vad_model() {
                Ok(p) => {
                    log::info!("Engine init: VAD model at {}", p.display());
                    Some(p)
                }
                Err(e) => {
                    log::warn!("Engine init: VAD setup failed: {e}");
                    None
                }
            };

            let recorder = match recorder::AudioRecorder::new(vad.as_deref()) {
                Ok(r) => r,
                Err(e) => {
                    log::error!("Engine init: failed to create recorder: {e}");
                    return;
                }
            };

            let (model_id, model_engine) = match mgr.first_downloaded_model() {
                Some(pair) => pair,
                None => {
                    log::warn!("Engine init: no model downloaded yet");
                    return;
                }
            };

            log::info!("Engine init: loading model {model_id}");
            let transcriber = match transcribe::Transcriber::new(model_engine) {
                Ok(t) => t,
                Err(e) => {
                    log::error!("Engine init: failed to create transcriber: {e}");
                    return;
                }
            };

            let verifier = match mgr.ensure_speaker_model() {
                Ok(p) => {
                    log::info!("Engine init: speaker model at {}", p.display());
                    match speaker::SpeakerVerifier::new(&p) {
                        Ok(v) => {
                            if let Some(embedding) = speaker::load_enrollment() {
                                v.enroll_from_embedding(embedding);
                                log::info!("Engine init: restored speaker enrollment");
                            }
                            Some(v)
                        }
                        Err(e) => {
                            log::warn!("Engine init: speaker verifier failed: {e}");
                            None
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Engine init: speaker model setup failed: {e}");
                    None
                }
            };

            let engine = engine::Engine::new(recorder, transcriber, model_id.clone(), verifier);
            engine.preload();

            let state = app.state::<AppState>();
            *state.engine.lock().unwrap() = Some(engine);

            log::info!("Engine init: ready (model: {model_id})");
            let _ = app.emit("engine-ready", ());
        })
        .ok();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = log::set_logger(&debug_log::LOGGER);
    log::set_max_level(log::LevelFilter::Debug);

    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init());

    #[cfg(desktop)]
    {
        builder = builder.plugin(tauri_plugin_global_shortcut::Builder::new().build());
    }

    builder.setup(|app| {
            #[cfg(target_os = "android")]
            {
                if let Ok(data_dir) = app.path().app_data_dir() {
                    std::env::set_var("VERBA_DATA_DIR", &data_dir);
                    let _ = std::fs::create_dir_all(&data_dir);
                }
            }

            let model_manager = match models::ModelManager::new() {
                Ok(m) => m,
                Err(e) => {
                    log::error!("Failed to create model manager: {e}");
                    return Ok(());
                }
            };

            debug_log::set_app_handle(app.handle().clone());

            app.manage(model_manager);
            app.manage(history::History::new());
            app.manage(AppState {
                engine: std::sync::Mutex::new(None),
                recording: std::sync::atomic::AtomicBool::new(false),
            });

            #[cfg(desktop)]
            {
                use std::sync::atomic::Ordering;
                use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
                use tauri::tray::TrayIconBuilder;
                use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

                // Alt+D: press to start recording, release to stop and paste
                let shortcut = Shortcut::new(Some(Modifiers::ALT), Code::KeyD);
                let app_handle = app.handle().clone();
                app.global_shortcut().on_shortcut(shortcut, move |_app, _shortcut, event| {
                    let state = app_handle.state::<AppState>();
                    match event.state {
                        ShortcutState::Pressed => {
                            if state.recording.load(Ordering::SeqCst) {
                                return;
                            }
                            let guard = state.engine.lock().unwrap();
                            let Some(ref engine) = *guard else {
                                log::warn!("Shortcut: engine not ready");
                                let _ = app_handle.emit("dictation-error", "Engine not ready. Wait for model to load.");
                                return;
                            };
                            match engine.start_streaming() {
                                Ok(()) => {
                                    state.recording.store(true, Ordering::SeqCst);
                                    let _ = app_handle.emit("dictation-state", "recording");
                                    log::info!("Shortcut: recording started");
                                }
                                Err(e) => {
                                    log::error!("Shortcut: failed to start: {e}");
                                    let _ = app_handle.emit("dictation-error", e.as_str());
                                }
                            }
                        }
                        ShortcutState::Released => {
                            if !state.recording.swap(false, Ordering::SeqCst) {
                                return;
                            }
                            let _ = app_handle.emit("dictation-state", "processing");
                            log::info!("Shortcut: stopping recording");

                            let pending = {
                                let guard = state.engine.lock().unwrap();
                                let Some(ref engine) = *guard else { return };
                                match engine.stop_recording() {
                                    Ok(p) => p,
                                    Err(e) => {
                                        log::error!("Shortcut: stop failed: {e}");
                                        let _ = app_handle.emit("dictation-state", "idle");
                                        return;
                                    }
                                }
                            };

                            let app_for_paste = app_handle.clone();
                            std::thread::Builder::new()
                                .name("transcribe".into())
                                .spawn(move || {
                                    match pending.finalize() {
                                        Some(result) => {
                                            log::info!("Shortcut: transcribed: \"{}\"",
                                                if result.text.len() > 60 { &result.text[..60] } else { &result.text });
                                            let _ = app_for_paste.emit("transcription-result",
                                                serde_json::json!({
                                                    "text": &result.text,
                                                    "model_id": &result.model_id,
                                                    "audio_duration_ms": result.audio_duration_ms,
                                                    "transcribe_ms": result.transcribe_ms,
                                                }));
                                            if let Err(e) = paste::paste(&result.text) {
                                                log::error!("Shortcut: paste failed: {e}");
                                            }
                                        }
                                        None => {
                                            log::warn!("Shortcut: no text produced");
                                        }
                                    }
                                    let _ = app_for_paste.emit("dictation-state", "idle");
                                })
                                .ok();
                        }
                    }
                })?;

                // System tray
                let settings =
                    MenuItem::with_id(app, "settings", "Settings...", true, None::<&str>)?;
                let sep = PredefinedMenuItem::separator(app)?;
                let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

                let menu = Menu::with_items(app, &[&settings, &sep, &quit])?;

                let icon = app.default_window_icon().cloned().expect("no app icon");
                TrayIconBuilder::new()
                    .icon(icon)
                    .menu(&menu)
                    .on_menu_event(|app, event| match event.id().as_ref() {
                        "settings" => {
                            if let Some(window) = app.get_webview_window("main") {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                        "quit" => app.exit(0),
                        _ => {}
                    })
                    .build(app)?;

                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.hide();

                    let w = window.clone();
                    window.on_window_event(move |event| {
                        if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                            api.prevent_close();
                            let _ = w.hide();
                        }
                    });
                }
            }

            // Initialize engine in background (shared path for all platforms)
            init_engine(app.handle().clone());

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_models,
            list_audio_devices,
            get_config,
            save_config,
            download_model,
            delete_model,
            switch_model,
            list_history,
            clear_history,
            enroll_speaker,
            clear_speaker_enrollment,
            get_speaker_enrollment_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
