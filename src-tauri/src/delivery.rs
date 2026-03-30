//! Text delivery abstraction.
//!
//! After transcription, each platform delivers text differently:
//! - macOS: Accessibility API insertion or clipboard + Cmd+V
//! - Linux: clipboard + xdotool Ctrl+V (via enigo)
//! - Windows: clipboard + SendInput Ctrl+V (via enigo)
//! - Android: text returned via JNI to the accessibility service
//!
//! New platforms implement `TextDelivery` with their own delivery
//! mechanism.

/// Outcome of a delivery attempt.
pub enum DeliveryResult {
    /// Text was inserted into the target application.
    Inserted,
    /// Text was placed on clipboard for manual paste.
    ClipboardOnly,
}

/// How transcribed text reaches the user's target application.
pub trait TextDelivery: Send + Sync {
    fn deliver(&self, text: &str) -> Result<DeliveryResult, String>;
}

/// Desktop: delegates to paste.rs (AX on macOS, enigo on Linux/Windows).
#[cfg(desktop)]
pub struct DesktopDelivery {
    pub target: Option<crate::paste::PasteTarget>,
}

#[cfg(desktop)]
impl TextDelivery for DesktopDelivery {
    fn deliver(&self, text: &str) -> Result<DeliveryResult, String> {
        match crate::paste::paste(text, self.target.as_ref())? {
            crate::paste::PasteResult::Pasted => Ok(DeliveryResult::Inserted),
            crate::paste::PasteResult::ClipboardOnly => Ok(DeliveryResult::ClipboardOnly),
        }
    }
}
