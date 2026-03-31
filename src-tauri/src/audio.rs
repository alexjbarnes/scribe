use cpal::traits::{DeviceTrait, HostTrait};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct AudioDevice {
    pub name: String,
    pub index: usize,
}

pub fn list_input_devices() -> Vec<AudioDevice> {
    let host = cpal::default_host();
    let mut devices = Vec::new();
    let mut seen_names = std::collections::HashSet::new();

    if let Ok(inputs) = host.input_devices() {
        for (i, dev) in inputs.enumerate() {
            if let Ok(name) = dev.name() {
                // Skip duplicate device names (common on Android where cpal
                // reports multiple input sources for the same physical mic).
                if seen_names.insert(name.clone()) {
                    devices.push(AudioDevice { name, index: i });
                }
            }
        }
    }

    devices
}
