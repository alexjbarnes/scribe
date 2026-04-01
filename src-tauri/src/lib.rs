mod audio;
mod config;
mod debug_log;
mod delivery;
mod engine;
mod history;
mod models;
#[cfg(desktop)]
mod paste;
mod postprocess;
#[cfg(desktop)]
mod sound;
mod recorder;
pub mod speaker;
mod transcribe;
pub mod vad;
#[cfg(target_os = "android")]
mod android_ime;

use tauri::{Emitter, Manager};

#[cfg(desktop)]
struct AppState {
    recording: std::sync::atomic::AtomicBool,
}

#[tauri::command]
fn list_models() -> Vec<models::ModelInfo> {
    models::ModelManager::global().list()
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
fn list_history() -> Vec<history::HistoryEntry> {
    history::History::global().list()
}

#[tauri::command]
fn clear_history() {
    history::History::global().clear()
}

#[tauri::command]
fn export_history() -> Result<String, String> {
    history::History::global().export()
}

/// Start a short enrollment recording. Creates a temporary recorder and
/// speaker verifier, records 5 seconds, extracts and persists the embedding.
#[tauri::command]
fn enroll_speaker() -> Result<(), String> {
    let mgr = models::ModelManager::global();
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

    // Update the engine's in-memory verifier
    engine::with(|eng| {
        if let Some(v) = eng.verifier() {
            if let Some(emb) = speaker::load_enrollment() {
                v.enroll_from_embedding(emb);
            }
        }
    });

    log::info!("Speaker enrolled and saved");
    Ok(())
}

#[tauri::command]
fn clear_speaker_enrollment() -> Result<(), String> {
    engine::with(|eng| {
        if let Some(verifier) = eng.verifier() {
            verifier.clear_enrollment();
        }
    });
    speaker::delete_enrollment()?;
    Ok(())
}

#[tauri::command]
fn get_speaker_enrollment_status() -> bool {
    speaker::load_enrollment().is_some()
}

#[tauri::command]
async fn switch_model(app: tauri::AppHandle, id: String) -> Result<(), String> {
    models::ModelManager::global().set_active(&id)?;
    log::info!("Switched to model: {id}");

    let id_for_thread = id.clone();
    tokio::task::spawn_blocking(move || {
        let mgr = models::ModelManager::global();
        engine::with_mut(|eng| {
            if eng.model_id() != id_for_thread {
                log::info!("Reloading model to {id_for_thread}");
                if let Some((_mid, model_engine)) = mgr.first_downloaded_model() {
                    match transcribe::Transcriber::new(model_engine) {
                        Ok(t) => eng.reload_model(t, id_for_thread.clone()),
                        Err(e) => log::error!("Transcriber reload failed: {e}"),
                    }
                }
            }
        });
    })
    .await
    .map_err(|e| format!("model reload failed: {e}"))?;

    let _ = app.emit("model-loaded", serde_json::json!({ "id": &id }));
    Ok(())
}

#[tauri::command]
async fn download_model(app: tauri::AppHandle, id: String) -> Result<(), String> {
    let mgr = models::ModelManager::global();
    if mgr.is_downloaded(&id) {
        return Ok(());
    }
    mgr.download(&id, &app).await
}

#[tauri::command]
fn delete_model(id: String) -> Result<(), String> {
    models::ModelManager::global().delete(&id)
}

/// Initialize the Engine in the background: VAD, recorder, transcriber,
/// speaker verifier, then preload model + post-processing pipeline.
/// Shared by both Tauri app and Android IME -- whichever starts first
/// creates the engine; the other is a no-op.
fn init_engine(app: tauri::AppHandle) {
    std::thread::Builder::new()
        .name("engine-init".into())
        .spawn(move || {
            if engine::is_initialized() {
                log::info!("Engine init: already initialized, skipping");
                let _ = app.emit("engine-ready", ());
                return;
            }
            if !engine::try_claim_init() {
                log::info!("Engine init: another thread is building the engine, waiting");
                engine::wait_until_ready();
                let _ = app.emit("engine-ready", ());
                return;
            }

            let mgr = models::ModelManager::global();

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

            let eng = engine::Engine::new(recorder, transcriber, model_id.clone(), verifier);
            eng.preload();
            engine::init_global(eng);

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

            if let Err(e) = models::ModelManager::init_global() {
                log::error!("Failed to create model manager: {e}");
                return Ok(());
            }
            history::History::init_global();

            debug_log::set_app_handle(app.handle().clone());

            #[cfg(desktop)]
            {
                app.manage(AppState {
                    recording: std::sync::atomic::AtomicBool::new(false),
                });
                use std::sync::atomic::Ordering;
                use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
                use tauri::tray::TrayIconBuilder;
                use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

                // Alt+D: press to start recording, release to stop and paste
                let shortcut = Shortcut::new(Some(Modifiers::ALT), Code::KeyD);
                let app_handle = app.handle().clone();
                let captured_target: std::sync::Mutex<Option<paste::PasteTarget>> = std::sync::Mutex::new(None);
                app.global_shortcut().on_shortcut(shortcut, move |_app, _shortcut, event| {
                    let state = app_handle.state::<AppState>();
                    match event.state {
                        ShortcutState::Pressed => {
                            if state.recording.load(Ordering::SeqCst) {
                                return;
                            }
                            *captured_target.lock().unwrap() = paste::capture_frontmost_app();
                            let started = engine::with(|eng| eng.start_streaming());
                            match started {
                                Some(Ok(())) => {
                                    state.recording.store(true, Ordering::SeqCst);
                                    sound::play_start();
                                    let _ = app_handle.emit("dictation-state", "recording");
                                    log::info!("Shortcut: recording started");
                                }
                                Some(Err(e)) => {
                                    log::error!("Shortcut: failed to start: {e}");
                                    let _ = app_handle.emit("dictation-error", e.as_str());
                                }
                                None => {
                                    log::warn!("Shortcut: engine not ready");
                                    let _ = app_handle.emit("dictation-error", "Engine not ready. Wait for model to load.");
                                }
                            }
                        }
                        ShortcutState::Released => {
                            if !state.recording.swap(false, Ordering::SeqCst) {
                                return;
                            }
                            sound::play_stop();
                            let _ = app_handle.emit("dictation-state", "processing");
                            log::info!("Shortcut: stopping recording");

                            let target = captured_target.lock().unwrap().take();
                            let deliver = delivery::DesktopDelivery { target };

                            let pending = match engine::with(|eng| eng.stop_recording()) {
                                Some(Ok(p)) => p,
                                Some(Err(e)) => {
                                    log::error!("Shortcut: stop failed: {e}");
                                    let _ = app_handle.emit("dictation-state", "idle");
                                    return;
                                }
                                None => return,
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
                                            use delivery::TextDelivery;
                                            match deliver.deliver(&result.text) {
                                                Ok(delivery::DeliveryResult::Inserted) => {}
                                                Ok(delivery::DeliveryResult::ClipboardOnly) => {
                                                    sound::play_error();
                                                    let _ = app_for_paste.emit("paste-fallback",
                                                        "Text copied to clipboard — paste manually");
                                                }
                                                Err(e) => {
                                                    log::error!("Shortcut: delivery failed: {e}");
                                                    sound::play_error();
                                                }
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

                // Close button hides the window instead of quitting — app keeps
                // running in the menu bar. Reopen via tray "Settings..." item.
                if let Some(window) = app.get_webview_window("main") {
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
            export_history,
            enroll_speaker,
            clear_speaker_enrollment,
            get_speaker_enrollment_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
