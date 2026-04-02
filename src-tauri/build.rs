fn main() {
    tauri_build::build();

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
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

        // Copy libonnxruntime.so into jniLibs so it gets packaged in the APK.
        // The ort Rust crate uses load-dynamic (dlopen at runtime) — it does not
        // link ORT at compile time, so this .so must be present on the device.
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let repo_root = std::path::PathBuf::from(&manifest_dir)
            .parent()
            .unwrap()
            .to_path_buf();
        let ort_so = repo_root.join(".android-deps/ort/arm64-v8a/libonnxruntime.so");
        let jni_libs = std::path::PathBuf::from(&manifest_dir)
            .join("gen/android/app/src/main/jniLibs/arm64-v8a");

        println!("cargo:rerun-if-changed={}", ort_so.display());

        if ort_so.exists() {
            std::fs::create_dir_all(&jni_libs).ok();
            if let Err(e) = std::fs::copy(&ort_so, jni_libs.join("libonnxruntime.so")) {
                println!("cargo:warning=Failed to copy libonnxruntime.so: {e}");
            }
        } else {
            println!("cargo:warning=libonnxruntime.so not found at {} — run scripts/android-build.sh --setup-only", ort_so.display());
        }
    }
}
