/// Audio feedback for push-to-talk hotkey events.
///
/// On macOS, plays system sounds via `afplay` (non-blocking).

/// Play a short "start recording" sound.
pub fn play_start() {
    play("Tink");
}

/// Play a "stop recording / done" sound.
pub fn play_stop() {
    play("Pop");
}

/// Play an error/fallback sound (paste failed, text on clipboard).
pub fn play_error() {
    play("Basso");
}

#[cfg(target_os = "macos")]
fn play(name: &str) {
    let path = format!("/System/Library/Sounds/{name}.aiff");
    if let Err(e) = std::process::Command::new("afplay")
        .arg(&path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        log::warn!("Failed to play sound {name}: {e}");
    }
}

#[cfg(not(target_os = "macos"))]
fn play(_name: &str) {
    // TODO: add sound feedback for Linux/Windows
}
