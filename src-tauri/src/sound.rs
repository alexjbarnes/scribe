use std::process::Command;

/// Rising ping — dictation started
pub fn start_beep() {
    std::thread::spawn(|| {
        let _ = Command::new("afplay")
            .arg("/System/Library/Sounds/Tink.aiff")
            .spawn();
    });
}

/// Lower tone — dictation stopped
pub fn stop_beep() {
    std::thread::spawn(|| {
        let _ = Command::new("afplay")
            .arg("/System/Library/Sounds/Pop.aiff")
            .spawn();
    });
}
