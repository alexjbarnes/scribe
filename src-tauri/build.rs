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
    }
}
