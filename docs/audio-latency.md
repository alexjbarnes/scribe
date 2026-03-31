# Audio Latency

## Current state

Audio capture uses cpal 0.15 on all platforms. On Android, cpal wraps oboe internally. The recording path is:

```
Button press -> open_stream() -> cpal stream.play() -> first audio callback -> VAD -> transcriber
```

Stream-open time is ~130ms on Android (OS overhead for acquiring the audio device). First-callback latency depends on buffer size. VAD prefill buffer (500ms) captures audio before speech detection triggers, preventing clipped word beginnings.

## Implemented

**Fixed buffer size (256 frames)** in `recorder.rs open_stream()`. Queries the device's supported buffer size range and clamps to 256 frames. At 48kHz this is ~5ms per buffer, down from the OS default (often 1024-4096 frames / 20-80ms). Reduces time between `stream.play()` and first audio callback.

**VAD prefill buffer (500ms)** in `vad.rs`. Ring buffer retains last 500ms of audio so when VAD detects speech onset, the preceding audio is included. Bumped from 300ms to 500ms to better capture first words.

**Raw audio fallback** in `recorder.rs record_loop()`. When VAD produces no segments and no speech is flushed (common with single-word utterances), falls back to sending raw recorded audio to the transcriber rather than discarding it silently.

## Considered and deferred

**Stop delay (200-300ms)** - Keep recording briefly after the user releases the button to capture trailing words. Would solve end-of-speech clipping the same way prefill solves start-of-speech clipping. Trade-off: adds 200-300ms before transcription starts, which is noticeable on short (1-2s) dictations. The raw audio fallback already covers the worst case (single word, VAD misses entirely). Revisit if trailing word loss is reported as a real problem.

## Bigger changes

### Android: oboe PerformanceMode::LowLatency

The single biggest remaining win for Android. Tells oboe to use AAudio's low-latency path (MMAP mode on supported hardware). Expected to cut stream-open time from 100-130ms to 20-40ms and reduce per-buffer latency.

cpal 0.15 does not expose this. The `cpal::platform::oboe` module sets `PerformanceMode::None` by default. Options:

1. **Patch cpal's oboe backend** - Override `PerformanceMode::None` to `PerformanceMode::LowLatency` in the platform code. Smallest diff but requires a cpal fork or patch.
2. **Fork cpal** - Maintain a fork with the oboe performance mode exposed as configuration. More control, maintenance burden.
3. **Use oboe-rust directly on Android** - Keep cpal for desktop, use the `oboe` crate directly for Android behind `#[cfg(target_os = "android")]`. More code but full control over oboe settings including performance mode and sharing mode. Could also set buffer callbacks directly instead of going through cpal's abstraction.

### Android: SharingMode::Exclusive

Bypasses the Android audio mixer for lower latency. Same situation as PerformanceMode, not exposed by cpal. Could be done alongside the PerformanceMode change. Trade-off: exclusive mode blocks other apps from using the mic simultaneously. Fine for dictation, but worth noting.

### Desktop (macOS): CoreAudio exclusive mode

macOS apps can request exclusive device access via `kAudioDevicePropertyHogMode`. cpal doesn't expose this. The default shared mode on macOS is already good (sub-10ms typically), so this is low priority compared to the Android oboe work.

## Not worth doing

**Persistent stream between recordings** - Keeping the audio stream open while not recording would eliminate stream-open latency entirely. Rejected because it shows a mic-in-use indicator on Android (and macOS menu bar), could interfere with phone calls, and keeps the audio hardware powered on (battery impact).

**Thread priority for audio callback** - cpal and oboe already set appropriate thread priorities on their internal audio threads.

**Reducing VAD window size below 512 samples (32ms)** - Silero VAD expects exactly 512-sample windows at 16kHz. Changing this breaks the model.
