use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::sound;
use crate::transcribe::Transcriber;

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Idle,
    Recording,
    Processing,
}

pub struct DictationManager {
    state: Mutex<State>,
    audio_buffer: Arc<Mutex<Vec<f32>>>,
    recording_active: Arc<AtomicBool>,
    sample_rate: Arc<Mutex<u32>>,
    channels: Arc<Mutex<u16>>,
    transcriber: Mutex<Option<Transcriber>>,
}

impl DictationManager {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State::Idle),
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
            recording_active: Arc::new(AtomicBool::new(false)),
            sample_rate: Arc::new(Mutex::new(48000)),
            channels: Arc::new(Mutex::new(1)),
            transcriber: Mutex::new(None),
        }
    }

    pub fn set_transcriber(&self, t: Transcriber) {
        *self.transcriber.lock().unwrap() = Some(t);
    }

    pub fn start(&self) {
        let current = *self.state.lock().unwrap();
        if current != State::Idle {
            return;
        }
        self.start_recording();
    }

    pub fn stop(&self) {
        let current = *self.state.lock().unwrap();
        if current != State::Recording {
            return;
        }
        self.stop_recording();
    }

    fn start_recording(&self) {
        sound::start_beep();

        self.audio_buffer.lock().unwrap().clear();
        self.recording_active.store(true, Ordering::SeqCst);

        let buffer = self.audio_buffer.clone();
        let active = self.recording_active.clone();
        let rate_out = self.sample_rate.clone();
        let channels_out = self.channels.clone();

        std::thread::spawn(move || {
            let host = cpal::default_host();
            let Some(device) = host.default_input_device() else {
                log::error!("No input device available");
                active.store(false, Ordering::SeqCst);
                return;
            };

            let Ok(supported) = device.default_input_config() else {
                log::error!("No input config available");
                active.store(false, Ordering::SeqCst);
                return;
            };

            let rate = supported.sample_rate().0;
            let ch = supported.channels();
            *rate_out.lock().unwrap() = rate;
            *channels_out.lock().unwrap() = ch;
            let stream_config: cpal::StreamConfig = supported.into();

            let stream = device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    buffer.lock().unwrap().extend_from_slice(data);
                },
                |err| log::error!("Input stream error: {err}"),
                None,
            );

            let Ok(stream) = stream else {
                log::error!("Failed to build input stream");
                active.store(false, Ordering::SeqCst);
                return;
            };

            if let Err(e) = stream.play() {
                log::error!("Failed to start recording: {e}");
                active.store(false, Ordering::SeqCst);
                return;
            }

            log::info!("Recording at {}Hz, {} channel(s)", rate, ch);

            while active.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(50));
            }
        });

        *self.state.lock().unwrap() = State::Recording;
        log::info!("Recording started");
    }

    fn stop_recording(&self) {
        *self.state.lock().unwrap() = State::Processing;

        // Signal recording thread to stop
        self.recording_active.store(false, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(100));

        sound::stop_beep();

        let raw_samples = self.audio_buffer.lock().unwrap().clone();
        let rate = *self.sample_rate.lock().unwrap();
        let ch = *self.channels.lock().unwrap() as usize;

        // Convert to mono if multi-channel
        let mono = if ch > 1 {
            raw_samples
                .chunks(ch)
                .map(|frame| frame.iter().sum::<f32>() / ch as f32)
                .collect::<Vec<f32>>()
        } else {
            raw_samples
        };

        let duration = mono.len() as f32 / rate as f32;
        log::info!("Recorded {:.1}s ({} mono samples @ {}Hz)", duration, mono.len(), rate);

        // Transcribe
        let text = {
            let guard = self.transcriber.lock().unwrap();
            if let Some(ref transcriber) = *guard {
                match transcriber.transcribe(mono, rate as i32) {
                    Ok(t) if !t.is_empty() => t,
                    Ok(_) => {
                        log::warn!("Transcription returned empty text");
                        "[No speech detected]".to_string()
                    }
                    Err(e) => {
                        log::error!("Transcription failed: {e}");
                        format!("[Transcription error: {e}]")
                    }
                }
            } else {
                log::warn!("No transcriber loaded");
                format!("[Recorded {:.1}s — no model loaded]", duration)
            }
        };

        if let Err(e) = paste_text(&text) {
            log::error!("Paste failed: {e}");
        }

        *self.state.lock().unwrap() = State::Idle;
    }
}

fn paste_text(text: &str) -> Result<(), String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|e| format!("clipboard: {e}"))?;
    clipboard.set_text(text).map_err(|e| format!("set text: {e}"))?;

    std::thread::sleep(Duration::from_millis(50));

    // Simulate Cmd+V via osascript (works from any thread, unlike enigo)
    std::process::Command::new("osascript")
        .arg("-e")
        .arg(r#"tell application "System Events" to keystroke "v" using command down"#)
        .output()
        .map_err(|e| format!("osascript: {e}"))?;

    log::info!("Pasted: {text}");
    Ok(())
}
