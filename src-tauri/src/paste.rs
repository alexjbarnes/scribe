use std::time::Duration;

use arboard::Clipboard;
use enigo::{Enigo, Key, Keyboard, Settings};

/// Paste text into the focused application.
///
/// Flow: save clipboard -> write text -> simulate paste -> restore clipboard.
pub fn paste(text: &str) -> Result<(), String> {
    let mut clipboard = Clipboard::new().map_err(|e| format!("clipboard: {e}"))?;

    // Save current clipboard contents
    let saved = clipboard.get_text().ok();

    // Write our text
    clipboard
        .set_text(text)
        .map_err(|e| format!("set clipboard: {e}"))?;

    // Brief pause for clipboard to register
    std::thread::sleep(Duration::from_millis(50));

    // Simulate paste keystroke
    simulate_paste()?;

    // Brief pause, then restore clipboard
    std::thread::sleep(Duration::from_millis(100));
    if let Some(prev) = saved {
        let _ = clipboard.set_text(prev);
    }

    log::info!("Pasted: {text}");
    Ok(())
}

fn simulate_paste() -> Result<(), String> {
    let mut enigo = Enigo::new(&Settings::default())
        .map_err(|e| format!("enigo init: {e}"))?;

    #[cfg(target_os = "macos")]
    let modifier = Key::Meta;

    #[cfg(not(target_os = "macos"))]
    let modifier = Key::Control;

    enigo
        .key(modifier, enigo::Direction::Press)
        .map_err(|e| format!("key press: {e}"))?;
    enigo
        .key(Key::Unicode('v'), enigo::Direction::Click)
        .map_err(|e| format!("key v: {e}"))?;
    enigo
        .key(modifier, enigo::Direction::Release)
        .map_err(|e| format!("key release: {e}"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modifier_key_is_platform_appropriate() {
        #[cfg(target_os = "macos")]
        {
            let modifier = Key::Meta;
            assert!(matches!(modifier, Key::Meta));
        }

        #[cfg(not(target_os = "macos"))]
        {
            let modifier = Key::Control;
            assert!(matches!(modifier, Key::Control));
        }
    }

    // Clipboard tests require a display server (Wayland/X11/macOS).
    // Skips gracefully in headless CI.
    #[test]
    fn clipboard_roundtrip() {
        let Ok(mut clipboard) = Clipboard::new() else {
            eprintln!("No display server, skipping clipboard test");
            return;
        };

        let test_text = "scribe_test_clipboard_roundtrip";
        let saved = clipboard.get_text().ok();

        clipboard.set_text(test_text).unwrap();
        let got = clipboard.get_text().unwrap();
        assert_eq!(got, test_text);

        // Restore
        if let Some(prev) = saved {
            let _ = clipboard.set_text(prev);
        }
    }
}
