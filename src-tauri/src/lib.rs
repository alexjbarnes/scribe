mod audio;
mod config;
mod coordinator;
mod debug_log;
mod dictation;
mod history;
mod models;
#[cfg(desktop)]
mod paste;
mod recorder;
#[cfg(desktop)]
mod sound;
mod transcribe;
mod vad;
#[cfg(target_os = "android")]
mod android_ime;

use std::sync::Arc;

use tauri::Manager;

use coordinator::{DictationCmd, ShortcutEvent};

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

#[tauri::command]
fn switch_model(state: tauri::State<'_, models::ModelManager>, id: String) -> Result<(), String> {
    state.set_active(&id)?;
    log::info!("Switched to model: {id}");
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

#[tauri::command]
async fn start_dictation(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<dictation::DictationManager>>,
) -> Result<(), String> {
    let dm = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        dm.ensure_recorder();
        if !dm.ensure_transcriber(&app) {
            return;
        }
        dm.start();
    })
    .await
    .map_err(|e| format!("join error: {e}"))
}

#[tauri::command]
async fn stop_dictation(
    state: tauri::State<'_, Arc<dictation::DictationManager>>,
) -> Result<(), String> {
    let dm = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || dm.stop())
        .await
        .map_err(|e| format!("join error: {e}"))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = log::set_logger(&debug_log::LOGGER);
    log::set_max_level(log::LevelFilter::Debug);

    let mut builder = tauri::Builder::default();

    // Desktop: global shortcut for push-to-talk
    #[cfg(desktop)]
    {
        use tauri_plugin_global_shortcut::ShortcutState;

        builder = builder.plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_shortcut("alt+d")
                .expect("failed to register shortcut")
                .with_handler(|app, _shortcut, event| {
                    let dm = app.state::<Arc<dictation::DictationManager>>();
                    match event.state {
                        ShortcutState::Pressed => {
                            dm.on_shortcut(ShortcutEvent::Pressed);
                        }
                        ShortcutState::Released => {
                            dm.on_shortcut(ShortcutEvent::Released);
                        }
                    }
                })
                .build(),
        );
    }

    builder
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Android: set data dir env var for config/models path resolution.
            // Must happen before ModelManager::new() which reads this.
            #[cfg(target_os = "android")]
            {
                if let Ok(data_dir) = app.path().app_data_dir() {
                    std::env::set_var("SCRIBE_DATA_DIR", &data_dir);
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

            let (dictation_manager, cmd_rx) = dictation::DictationManager::new();
            let dictation_manager = Arc::new(dictation_manager);

            // Spawn the command handler thread
            let dm_for_cmds = dictation_manager.clone();
            let _ = std::thread::Builder::new()
                .name("dictation-cmds".into())
                .spawn(move || {
                    while let Ok(cmd) = cmd_rx.recv() {
                        match cmd {
                            DictationCmd::Start => dm_for_cmds.start(),
                            DictationCmd::Stop => dm_for_cmds.stop(),
                            DictationCmd::Cancel => dm_for_cmds.cancel(),
                        }
                    }
                });

            // Set the app handle for event emission (dictation + debug logs)
            dictation_manager.set_app_handle(app.handle().clone());
            debug_log::set_app_handle(app.handle().clone());

            app.manage(model_manager);
            app.manage(history::History::new());
            app.manage(dictation_manager.clone());

            // Desktop: system tray and hide-on-close
            #[cfg(desktop)]
            {
                use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
                use tauri::tray::TrayIconBuilder;

                let status = MenuItem::with_id(app, "status", "Idle", false, None::<&str>)?;
                let sep1 = PredefinedMenuItem::separator(app)?;
                let settings =
                    MenuItem::with_id(app, "settings", "Settings...", true, None::<&str>)?;
                let sep2 = PredefinedMenuItem::separator(app)?;
                let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

                let menu = Menu::with_items(app, &[&status, &sep1, &settings, &sep2, &quit])?;

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
                    // Start hidden behind tray on desktop
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

            // Background init: download VAD model, set up recorder, load ASR model.
            // On mobile, defer this until the user actually taps record to avoid
            // crashing before audio permission is granted.
            #[cfg(desktop)]
            {
                let app_handle = app.handle().clone();
                std::thread::spawn(move || {
                    background_init(&app_handle);
                });
            }

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
            start_dictation,
            stop_dictation,
            list_history,
            clear_history,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn background_init(app_handle: &tauri::AppHandle) {
    let mgr = app_handle.state::<models::ModelManager>();
    let dm = app_handle.state::<Arc<dictation::DictationManager>>();

    // Set up recorder (with VAD if model available, without if not)
    let vad_path = mgr.vad_model_path();
    let vad_path = if vad_path.exists() {
        Some(vad_path)
    } else {
        log::info!("VAD model not found, downloading...");
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build();
        match rt {
            Ok(rt) => match rt.block_on(mgr.ensure_vad_model()) {
                Ok(p) => Some(p),
                Err(e) => {
                    log::warn!("VAD download failed, continuing without: {e}");
                    None
                }
            },
            Err(e) => {
                log::warn!("Failed to create runtime for VAD download: {e}");
                None
            }
        }
    };

    match recorder::AudioRecorder::new(vad_path.as_deref()) {
        Ok(rec) => {
            dm.set_recorder(rec);
            if vad_path.is_some() {
                log::info!("Audio recorder ready (with VAD)");
            } else {
                log::info!("Audio recorder ready (no VAD)");
            }
        }
        Err(e) => log::error!("Failed to create audio recorder: {e}"),
    }

    // Load first available model
    if let Some((id, engine)) = mgr.first_downloaded_model() {
        log::info!("Loading transcription model: {id}");
        match transcribe::Transcriber::new(engine) {
            Ok(t) => {
                dm.set_transcriber(t);
                log::info!("Transcription model ready: {id}");
            }
            Err(e) => log::warn!("Failed to load transcription model: {e}"),
        }
    } else {
        log::info!("No transcription model downloaded yet");
    }
}
