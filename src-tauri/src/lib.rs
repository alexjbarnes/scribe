mod audio;
mod config;
mod dictation;
mod models;
mod sound;
mod transcribe;

use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconBuilder,
    Manager,
};
use tauri_plugin_global_shortcut::ShortcutState;

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
fn switch_model(state: tauri::State<'_, models::ModelManager>, id: String) -> Result<(), String> {
    state.set_active(&id)?;
    log::info!("Switched to model: {id}");
    Ok(())
}

#[tauri::command]
async fn download_model(app: tauri::AppHandle, id: String) -> Result<(), String> {
    let mgr = app.state::<models::ModelManager>();

    // Check not already downloaded
    if mgr.is_downloaded(&id) {
        return Ok(());
    }

    mgr.download(&id, &app).await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();

    let model_manager =
        models::ModelManager::new().expect("failed to create model manager");
    let dictation_manager = dictation::DictationManager::new();

    tauri::Builder::default()
        .manage(model_manager)
        .manage(dictation_manager)
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_shortcut("alt+d")
                .expect("failed to register shortcut")
                .with_handler(|app, _shortcut, event| {
                    let app = app.clone();
                    match event.state {
                        ShortcutState::Pressed => {
                            std::thread::spawn(move || {
                                let dm = app.state::<dictation::DictationManager>();
                                dm.start();
                            });
                        }
                        ShortcutState::Released => {
                            std::thread::spawn(move || {
                                let dm = app.state::<dictation::DictationManager>();
                                dm.stop();
                            });
                        }
                    }
                })
                .build(),
        )
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Build tray menu
            let status = MenuItem::with_id(app, "status", "Idle", false, None::<&str>)?;
            let sep1 = PredefinedMenuItem::separator(app)?;
            let settings =
                MenuItem::with_id(app, "settings", "Settings...", true, None::<&str>)?;
            let sep2 = PredefinedMenuItem::separator(app)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

            let menu = Menu::with_items(app, &[&status, &sep1, &settings, &sep2, &quit])?;

            // System tray
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

            // Hide on close instead of quitting
            if let Some(window) = app.get_webview_window("main") {
                let w = window.clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = w.hide();
                    }
                });
            }

            // Auto-load first available Parakeet model in background
            let app_handle = app.handle().clone();
            std::thread::spawn(move || {
                let mgr = app_handle.state::<models::ModelManager>();
                if let Some((id, (enc, dec, joi, tok))) = mgr.first_downloaded_parakeet() {
                    log::info!("Loading transcription model: {id}");
                    match transcribe::Transcriber::new(&enc, &dec, &joi, &tok) {
                        Ok(t) => {
                            let dm = app_handle.state::<dictation::DictationManager>();
                            dm.set_transcriber(t);
                            log::info!("Transcription model ready: {id}");
                        }
                        Err(e) => log::warn!("Failed to load transcription model: {e}"),
                    }
                } else {
                    log::info!("No Parakeet model downloaded yet");
                }
            });

            log::info!("Scribe started");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_models,
            list_audio_devices,
            get_config,
            save_config,
            download_model,
            switch_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
