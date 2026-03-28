use std::sync::mpsc;
use std::sync::Mutex;

use tauri::Emitter;

use tauri::Manager;

use crate::coordinator::{Coordinator, DictationCmd, ShortcutEvent};
use crate::models::ModelManager;
use crate::recorder::AudioRecorder;
use crate::transcribe::Transcriber;

pub struct DictationManager {
    recorder: Mutex<Option<AudioRecorder>>,
    transcriber: Mutex<Option<Transcriber>>,
    coordinator: Coordinator,
    app_handle: Mutex<Option<tauri::AppHandle>>,
}

impl DictationManager {
    pub fn new() -> (Self, mpsc::Receiver<DictationCmd>) {
        let (coordinator, cmd_rx) = Coordinator::new(30);
        let mgr = Self {
            recorder: Mutex::new(None),
            transcriber: Mutex::new(None),
            coordinator,
            app_handle: Mutex::new(None),
        };
        (mgr, cmd_rx)
    }

    pub fn set_app_handle(&self, handle: tauri::AppHandle) {
        *self.app_handle.lock().unwrap() = Some(handle);
    }

    pub fn set_recorder(&self, r: AudioRecorder) {
        *self.recorder.lock().unwrap() = Some(r);
    }

    pub fn set_transcriber(&self, t: Transcriber) {
        *self.transcriber.lock().unwrap() = Some(t);
    }

    /// Forward a shortcut event to the coordinator for debouncing.
    pub fn on_shortcut(&self, event: ShortcutEvent) {
        self.coordinator.send(event);
    }

    fn emit_state(&self, state: &str) {
        if let Some(ref handle) = *self.app_handle.lock().unwrap() {
            let _ = handle.emit("dictation-state", serde_json::json!({ "state": state }));
        }
    }

    fn emit_error(&self, msg: &str) {
        log::error!("{msg}");
        if let Some(ref handle) = *self.app_handle.lock().unwrap() {
            let _ = handle.emit("dictation-error", serde_json::json!({ "error": msg }));
        }
        self.emit_state("error");
    }

    /// Lazily create a recorder if none exists (mobile path).
    pub fn ensure_recorder(&self) {
        let mut recorder = self.recorder.lock().unwrap();
        if recorder.is_some() {
            return;
        }
        log::info!("Creating audio recorder on demand (no VAD)");
        match AudioRecorder::new(None) {
            Ok(rec) => {
                *recorder = Some(rec);
                log::info!("Audio recorder ready");
            }
            Err(e) => self.emit_error(&format!("Failed to create audio recorder: {e}")),
        }
    }

    /// Lazily load the transcription model if none exists (mobile path).
    /// Returns true if a transcriber is available.
    pub fn ensure_transcriber(&self, app_handle: &tauri::AppHandle) -> bool {
        let mut transcriber = self.transcriber.lock().unwrap();
        if transcriber.is_some() {
            return true;
        }
        let mgr = app_handle.state::<ModelManager>();
        if let Some((id, (enc, dec, joi, tok))) = mgr.first_downloaded_parakeet() {
            log::info!("Loading transcription model on demand: {id}");
            match Transcriber::new(&enc, &dec, &joi, &tok) {
                Ok(t) => {
                    *transcriber = Some(t);
                    log::info!("Transcription model ready: {id}");
                    true
                }
                Err(e) => {
                    self.emit_error(&format!("Failed to load transcription model: {e}"));
                    false
                }
            }
        } else {
            self.emit_error("No Parakeet model downloaded yet");
            false
        }
    }

    /// Handle a start command.
    pub fn start(&self) {
        let recorder = self.recorder.lock().unwrap();
        let Some(ref rec) = *recorder else {
            self.emit_error("No recorder available");
            return;
        };

        match rec.start() {
            Ok(()) => {
                #[cfg(desktop)]
                crate::sound::start_beep();
                self.emit_state("recording");
                log::info!("Recording started");
            }
            Err(e) => {
                self.emit_error(&format!("Failed to start recording: {e}"));
            }
        }
    }

    /// Handle a stop command: stop recording, transcribe, deliver result.
    pub fn stop(&self) {
        #[cfg(desktop)]
        crate::sound::stop_beep();

        let samples = {
            let recorder = self.recorder.lock().unwrap();
            match recorder.as_ref() {
                Some(rec) => match rec.stop() {
                    Ok(s) => s,
                    Err(e) => {
                        self.emit_error(&format!("Failed to stop recording: {e}"));
                        return;
                    }
                },
                None => {
                    self.emit_state("idle");
                    return;
                }
            }
        };

        let duration = samples.len() as f32 / 16_000.0;
        log::info!("Captured {:.1}s of audio", duration);

        self.emit_state("transcribing");

        let text = {
            let guard = self.transcriber.lock().unwrap();
            if let Some(ref transcriber) = *guard {
                match transcriber.transcribe(samples, 16_000) {
                    Ok(t) if !t.is_empty() => t,
                    Ok(_) => {
                        self.emit_error("No speech detected");
                        return;
                    }
                    Err(e) => {
                        self.emit_error(&format!("Transcription failed: {e}"));
                        return;
                    }
                }
            } else {
                self.emit_error("No transcriber loaded");
                return;
            }
        };

        // Emit result to frontend
        if let Some(ref handle) = *self.app_handle.lock().unwrap() {
            let _ = handle.emit(
                "transcription-result",
                serde_json::json!({ "text": &text }),
            );
        }

        // Desktop: paste into focused app
        #[cfg(desktop)]
        if let Err(e) = crate::paste::paste(&text) {
            log::error!("Paste failed: {e}");
        }

        self.emit_state("idle");
    }

    /// Handle a cancel command: stop recording, discard audio.
    pub fn cancel(&self) {
        #[cfg(desktop)]
        crate::sound::stop_beep();
        let recorder = self.recorder.lock().unwrap();
        if let Some(ref rec) = *recorder {
            let _ = rec.stop();
        }
        self.emit_state("idle");
        log::info!("Recording cancelled");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_returns_cmd_receiver() {
        let (_mgr, cmd_rx) = DictationManager::new();
        let result = cmd_rx.try_recv();
        assert!(result.is_err());
    }

    #[test]
    fn start_without_recorder_does_not_panic() {
        let (mgr, _cmd_rx) = DictationManager::new();
        mgr.start();
    }

    #[test]
    fn stop_without_recorder_does_not_panic() {
        let (mgr, _cmd_rx) = DictationManager::new();
        mgr.stop();
    }

    #[test]
    fn cancel_without_recorder_does_not_panic() {
        let (mgr, _cmd_rx) = DictationManager::new();
        mgr.cancel();
    }

    #[test]
    fn on_shortcut_sends_through_coordinator() {
        let (mgr, cmd_rx) = DictationManager::new();

        mgr.on_shortcut(ShortcutEvent::Pressed);

        let cmd = cmd_rx.recv_timeout(std::time::Duration::from_millis(200));
        assert!(matches!(cmd, Ok(DictationCmd::Start)));
    }
}
