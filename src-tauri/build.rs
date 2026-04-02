fn main() {
    tauri_build::build();

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    // Grammar neural (CoLA router + T5 corrector) is desktop-only.
    //
    // On Android, sherpa-onnx links ORT statically into libverba_rs_lib.so.
    // Grammar neural would load a second ORT instance (libonnxruntime.so) via
    // dlopen. Two ORT copies in the same process corrupt shared global state
    // (thread pools, NUMA topology) and cause the recorder thread to crash.
    //
    // Grammar neural will be re-enabled on Android once models are delivered
    // via R2/download instead of embedded at compile time, allowing the
    // ort dependency to be replaced by a standalone inference path.
    let grammar_dir = std::path::Path::new(&manifest_dir).join("data/grammar");
    let grammar_files = [
        "cola_model_quantized.onnx",
        "cola_tokenizer.json",
        "encoder_model_quantized.onnx",
        "decoder_model_quantized.onnx",
        "t5_tokenizer.json",
    ];
    let grammar_bundled = target_os != "android"
        && grammar_files.iter().all(|f| grammar_dir.join(f).exists());
    if grammar_bundled {
        println!("cargo:rustc-cfg=grammar_neural_bundled");
    }
    for f in &grammar_files {
        println!("cargo:rerun-if-changed=data/grammar/{f}");
    }

    if target_os == "android" {
        // sherpa-onnx / ONNX Runtime C++ code requires the C++ runtime.
        // Force the linker to record libc++_shared.so as a NEEDED dependency
        // so the dynamic linker loads it and resolves C++ ABI symbols
        // (__gxx_personality_v0, operator new/delete, etc.) at load time.
        // --no-as-needed prevents the linker from dropping the dep if it
        // thinks all symbols are already resolved from the static lib.
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-lc++_shared,--as-needed");
        println!("cargo:rustc-link-lib=dylib=log");

        // sherpa-onnx's session.cc references OrtSessionOptionsAppendExecutionProvider_Nnapi
        // but the ONNX Runtime static lib doesn't include the NNAPI provider, leaving a
        // GLOBAL UNDEFINED symbol that crashes the Android dynamic linker on arm64.
        cc::Build::new().file("stubs.c").compile("stubs");

        // libonnxruntime.so is NOT bundled in the Android APK.
        // Grammar neural is desktop-only (see comment above), so ORT is never
        // loaded at runtime on Android. Omitting it saves ~25MB in the APK.
    }
}
