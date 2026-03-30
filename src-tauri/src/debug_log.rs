use std::sync::Mutex;

use tauri::Emitter;

#[cfg(target_os = "android")]
extern "C" {
    fn __android_log_write(
        prio: libc::c_int,
        tag: *const libc::c_char,
        text: *const libc::c_char,
    ) -> libc::c_int;
}

static APP_HANDLE: Mutex<Option<tauri::AppHandle>> = Mutex::new(None);

/// Set the app handle so log messages can be forwarded to the frontend.
pub fn set_app_handle(handle: tauri::AppHandle) {
    *APP_HANDLE.lock().unwrap() = Some(handle);
}

/// Logger that writes to the platform log (logcat on Android, stderr on
/// desktop) and emits each message to the frontend as a `log-message` event.
pub struct FrontendLogger;

impl log::Log for FrontendLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Debug
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let level = record.level().as_str();
        let target = record.target();
        let msg = format!("{}", record.args());
        let line = format!("[{level}] {target}: {msg}");

        // Platform log output
        #[cfg(target_os = "android")]
        {
            let prio: libc::c_int = match record.level() {
                log::Level::Error => 6,
                log::Level::Warn => 5,
                log::Level::Info => 4,
                log::Level::Debug => 3,
                log::Level::Trace => 2,
            };
            if let Ok(c_msg) = std::ffi::CString::new(line.as_str()) {
                unsafe {
                    __android_log_write(
                        prio,
                        b"verba\0".as_ptr() as *const libc::c_char,
                        c_msg.as_ptr(),
                    );
                }
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            eprintln!("{line}");
        }

        // Emit to frontend
        if let Ok(guard) = APP_HANDLE.try_lock() {
            if let Some(ref handle) = *guard {
                let _ = handle.emit(
                    "log-message",
                    serde_json::json!({ "level": level, "line": line }),
                );
            }
        }
    }

    fn flush(&self) {}
}

pub static LOGGER: FrontendLogger = FrontendLogger;
