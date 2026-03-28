use std::sync::mpsc;
use std::time::Instant;

/// Events that the coordinator receives from shortcut handlers.
#[derive(Debug)]
pub enum ShortcutEvent {
    Pressed,
    Released,
    Cancel,
}

/// Commands the coordinator sends to the dictation manager.
#[derive(Debug)]
pub enum DictationCmd {
    Start,
    Stop,
    Cancel,
}

/// Serializes shortcut events into dictation commands with debouncing.
///
/// Runs on a dedicated thread. All shortcut events go through the coordinator,
/// which applies debouncing and state tracking before issuing commands.
pub struct Coordinator {
    event_tx: mpsc::Sender<ShortcutEvent>,
}

impl Coordinator {
    /// Spawn the coordinator thread. Returns the coordinator handle and a
    /// receiver for dictation commands.
    pub fn new(debounce_ms: u64) -> (Self, mpsc::Receiver<DictationCmd>) {
        let (event_tx, event_rx) = mpsc::channel();
        let (cmd_tx, cmd_rx) = mpsc::channel();

        std::thread::Builder::new()
            .name("shortcut-coordinator".into())
            .spawn(move || run(event_rx, cmd_tx, debounce_ms))
            .expect("failed to spawn coordinator thread");

        (Self { event_tx }, cmd_rx)
    }

    /// Send a shortcut event to the coordinator.
    pub fn send(&self, event: ShortcutEvent) {
        let _ = self.event_tx.send(event);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Idle,
    Recording,
}

fn run(
    event_rx: mpsc::Receiver<ShortcutEvent>,
    cmd_tx: mpsc::Sender<DictationCmd>,
    debounce_ms: u64,
) {
    let mut state = State::Idle;
    let mut last_event = Instant::now() - std::time::Duration::from_millis(debounce_ms + 1);

    while let Ok(event) = event_rx.recv() {
        let now = Instant::now();

        // Debounce: ignore events that arrive too quickly after the previous one
        if now.duration_since(last_event).as_millis() < debounce_ms as u128 {
            continue;
        }
        last_event = now;

        if let Some(cmd) = next_command(&mut state, event) {
            let _ = cmd_tx.send(cmd);
        }
    }
}

fn next_command(state: &mut State, event: ShortcutEvent) -> Option<DictationCmd> {
    match (&*state, event) {
        (State::Idle, ShortcutEvent::Pressed) => {
            *state = State::Recording;
            Some(DictationCmd::Start)
        }
        (State::Recording, ShortcutEvent::Released) => {
            *state = State::Idle;
            Some(DictationCmd::Stop)
        }
        (State::Recording, ShortcutEvent::Cancel) => {
            *state = State::Idle;
            Some(DictationCmd::Cancel)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn press_while_idle_starts() {
        let mut state = State::Idle;
        let cmd = next_command(&mut state, ShortcutEvent::Pressed);
        assert!(matches!(cmd, Some(DictationCmd::Start)));
        assert_eq!(state, State::Recording);
    }

    #[test]
    fn release_while_recording_stops() {
        let mut state = State::Recording;
        let cmd = next_command(&mut state, ShortcutEvent::Released);
        assert!(matches!(cmd, Some(DictationCmd::Stop)));
        assert_eq!(state, State::Idle);
    }

    #[test]
    fn cancel_while_recording_cancels() {
        let mut state = State::Recording;
        let cmd = next_command(&mut state, ShortcutEvent::Cancel);
        assert!(matches!(cmd, Some(DictationCmd::Cancel)));
        assert_eq!(state, State::Idle);
    }

    #[test]
    fn release_while_idle_ignored() {
        let mut state = State::Idle;
        let cmd = next_command(&mut state, ShortcutEvent::Released);
        assert!(cmd.is_none());
        assert_eq!(state, State::Idle);
    }

    #[test]
    fn press_while_recording_ignored() {
        let mut state = State::Recording;
        let cmd = next_command(&mut state, ShortcutEvent::Pressed);
        assert!(cmd.is_none());
        assert_eq!(state, State::Recording);
    }

    #[test]
    fn cancel_while_idle_ignored() {
        let mut state = State::Idle;
        let cmd = next_command(&mut state, ShortcutEvent::Cancel);
        assert!(cmd.is_none());
        assert_eq!(state, State::Idle);
    }

    #[test]
    fn full_push_to_talk_cycle() {
        let mut state = State::Idle;

        let cmd = next_command(&mut state, ShortcutEvent::Pressed);
        assert!(matches!(cmd, Some(DictationCmd::Start)));

        let cmd = next_command(&mut state, ShortcutEvent::Released);
        assert!(matches!(cmd, Some(DictationCmd::Stop)));

        // Back to idle, can start again
        let cmd = next_command(&mut state, ShortcutEvent::Pressed);
        assert!(matches!(cmd, Some(DictationCmd::Start)));
    }

    #[test]
    fn debounce_filters_fast_events() {
        let (coordinator, cmd_rx) = Coordinator::new(100);

        coordinator.send(ShortcutEvent::Pressed);
        // This Released comes too fast (within 100ms debounce)
        coordinator.send(ShortcutEvent::Released);

        // Should get Start but not Stop
        let cmd = cmd_rx.recv_timeout(std::time::Duration::from_millis(200));
        assert!(matches!(cmd, Ok(DictationCmd::Start)));

        // No more commands
        let cmd = cmd_rx.recv_timeout(std::time::Duration::from_millis(200));
        assert!(cmd.is_err());
    }

    #[test]
    fn debounce_allows_slow_events() {
        let (coordinator, cmd_rx) = Coordinator::new(10);

        coordinator.send(ShortcutEvent::Pressed);
        std::thread::sleep(std::time::Duration::from_millis(50));
        coordinator.send(ShortcutEvent::Released);

        let cmd = cmd_rx.recv_timeout(std::time::Duration::from_millis(200));
        assert!(matches!(cmd, Ok(DictationCmd::Start)));

        let cmd = cmd_rx.recv_timeout(std::time::Duration::from_millis(200));
        assert!(matches!(cmd, Ok(DictationCmd::Stop)));
    }
}
