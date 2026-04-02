# Verba build automation
#
# Usage:
#   just setup         # first-time desktop setup: shared-ORT libs + grammar models
#   just dev           # run desktop dev server (sets SHERPA_ONNX_LIB_DIR)
#   just setup-android # first-time Android setup: sherpa-onnx, ORT, grammar models
#   just apk           # build debug APK
#   just apk-release   # build release APK
#   just test          # run Rust tests
#   just check         # cargo check
#   just clean         # clean all build artifacts

# ── Configuration ──

repo_root    := justfile_directory()
tauri_dir    := repo_root / "src-tauri"
android_dir  := tauri_dir / "gen" / "android"
jni_dir      := android_dir / "app" / "src" / "main" / "jniLibs"
keystore     := repo_root / "debug.keystore"
desktop_deps := repo_root / ".desktop-deps"

# sherpa-onnx version must match Cargo.toml's sherpa-onnx dependency
sherpa_version := "1.12.34"

# Auto-detect paths
android_home := env("ANDROID_HOME", `echo ${HOME}/Android/Sdk`)
android_ndk  := env("ANDROID_NDK_HOME", `ls -1d ${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/* 2>/dev/null | sort -V | tail -1 || echo ""`)
build_tools  := `ls -1d ${ANDROID_HOME:-$HOME/Android/Sdk}/build-tools/* 2>/dev/null | sort -V | tail -1 || echo ""`
sherpa_libs  := repo_root / ".android-deps" / "sherpa-onnx" / "install" / "lib"
strip_bin    := android_ndk / "toolchains" / "llvm" / "prebuilt" / "linux-x86_64" / "bin" / "llvm-strip"

export ANDROID_HOME := android_home
export ANDROID_NDK_HOME := android_ndk
export JAVA_HOME := env("JAVA_HOME", "/usr")

# ── Recipes ──

# First-time desktop setup: shared-ORT sherpa-onnx libs + grammar models
setup: _setup-sherpa-desktop _setup-grammar

# Run desktop dev server with shared-ORT environment
dev:
    SHERPA_ONNX_LIB_DIR="{{desktop_deps}}/sherpa-onnx/lib" npx tauri dev

# ── Desktop setup internals ──

# Download sherpa-onnx prebuilt libs and set up shared ORT (mirrors Android pattern).
# Static sherpa-onnx libs + stub libonnxruntime.a + real libonnxruntime.dylib
# so both sherpa-onnx and the ort crate share a single ORT instance.
_setup-sherpa-desktop:
    #!/usr/bin/env bash
    set -euo pipefail
    LIB_DIR="{{desktop_deps}}/sherpa-onnx/lib"
    MARKER="$LIB_DIR/.shared-ort"
    if [ -f "$MARKER" ]; then
        echo "==> Desktop sherpa-onnx libs already prepared"
        exit 0
    fi

    VERSION="{{sherpa_version}}"
    BASE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/v${VERSION}"
    CACHE="{{desktop_deps}}/cache"
    mkdir -p "$CACHE" "$LIB_DIR"

    # Detect platform
    ARCH=$(uname -m)
    OS=$(uname -s)
    if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
        STATIC_ARCHIVE="sherpa-onnx-v${VERSION}-osx-arm64-static-lib.tar.bz2"
        SHARED_ARCHIVE="sherpa-onnx-v${VERSION}-osx-arm64-shared-lib.tar.bz2"
    elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
        STATIC_ARCHIVE="sherpa-onnx-v${VERSION}-osx-x64-static-lib.tar.bz2"
        SHARED_ARCHIVE="sherpa-onnx-v${VERSION}-osx-x64-shared-lib.tar.bz2"
    elif [ "$OS" = "Linux" ] && [ "$ARCH" = "x86_64" ]; then
        STATIC_ARCHIVE="sherpa-onnx-v${VERSION}-linux-x64-static-lib.tar.bz2"
        SHARED_ARCHIVE="sherpa-onnx-v${VERSION}-linux-x64-shared-lib.tar.bz2"
    elif [ "$OS" = "Linux" ] && [ "$ARCH" = "aarch64" ]; then
        STATIC_ARCHIVE="sherpa-onnx-v${VERSION}-linux-aarch64-static-lib.tar.bz2"
        SHARED_ARCHIVE="sherpa-onnx-v${VERSION}-linux-aarch64-shared-cpu-lib.tar.bz2"
    else
        echo "ERROR: Unsupported platform: $OS $ARCH"
        exit 1
    fi

    # Download static archive (all .a files)
    if [ ! -f "$CACHE/$STATIC_ARCHIVE" ]; then
        echo "==> Downloading sherpa-onnx static libs..."
        curl -fSL "$BASE_URL/$STATIC_ARCHIVE" -o "$CACHE/$STATIC_ARCHIVE"
    fi

    # Download shared archive (libonnxruntime.dylib/.so)
    if [ ! -f "$CACHE/$SHARED_ARCHIVE" ]; then
        echo "==> Downloading sherpa-onnx shared libs (for libonnxruntime)..."
        curl -fSL "$BASE_URL/$SHARED_ARCHIVE" -o "$CACHE/$SHARED_ARCHIVE"
    fi

    # Extract static libs
    echo "==> Extracting static libs..."
    STATIC_STEM="${STATIC_ARCHIVE%.tar.bz2}"
    tar -xjf "$CACHE/$STATIC_ARCHIVE" -C "$CACHE"
    cp "$CACHE/$STATIC_STEM/lib/"*.a "$LIB_DIR/"

    # Extract shared ORT dylib
    echo "==> Extracting shared libonnxruntime..."
    SHARED_STEM="${SHARED_ARCHIVE%.tar.bz2}"
    tar -xjf "$CACHE/$SHARED_ARCHIVE" -C "$CACHE"
    if [ "$OS" = "Darwin" ]; then
        cp "$CACHE/$SHARED_STEM/lib/libonnxruntime"*.dylib "$LIB_DIR/"
    else
        cp "$CACHE/$SHARED_STEM/lib/libonnxruntime"*.so* "$LIB_DIR/"
    fi

    # Replace libonnxruntime.a with an empty stub archive.
    # sherpa-onnx-sys emits `static=onnxruntime` which expects this file,
    # but we want ORT symbols to come from the shared library only.
    echo "==> Creating stub libonnxruntime.a..."
    rm -f "$LIB_DIR/libonnxruntime.a"
    # macOS ar doesn't support creating empty archives with `ar rcs`.
    # Create a trivial .o with an empty .c file, archive it, then clean up.
    STUB_C=$(mktemp /tmp/ort_stub.XXXXXX.c)
    STUB_O="${STUB_C%.c}.o"
    : > "$STUB_C"
    cc -c "$STUB_C" -o "$STUB_O"
    ar rcs "$LIB_DIR/libonnxruntime.a" "$STUB_O"
    rm -f "$STUB_C" "$STUB_O"

    touch "$MARKER"
    echo "==> Desktop sherpa-onnx ready (shared ORT)"
    ls -lh "$LIB_DIR/libonnxruntime"*

# Export and download neural grammar models
_setup-grammar:
    #!/usr/bin/env bash
    set -euo pipefail
    GRAMMAR_DIR="{{tauri_dir}}/data/grammar"
    FILES=(cola_model_quantized.onnx cola_tokenizer.json encoder_model_quantized.onnx decoder_model_quantized.onnx t5_tokenizer.json)
    complete=true
    for f in "${FILES[@]}"; do [ -f "$GRAMMAR_DIR/$f" ] || complete=false; done
    if $complete; then
        echo "==> Grammar models already present at $GRAMMAR_DIR"
        exit 0
    fi
    echo "==> Preparing grammar models..."
    mkdir -p "$GRAMMAR_DIR"
    VENV_DIR="{{repo_root}}/.grammar-venv"
    [ -d "$VENV_DIR" ] || python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -q --upgrade pip
    "$VENV_DIR/bin/pip" install -q huggingface_hub transformers "optimum[onnxruntime]"
    echo "==> Exporting CoLA router..."
    "$VENV_DIR/bin/python" "{{repo_root}}/scripts/export_cola_onnx.py" --output-dir "$GRAMMAR_DIR"
    echo "==> Downloading T5 corrector..."
    "$VENV_DIR/bin/python" "{{repo_root}}/scripts/download_t5_grammar_onnx.py" --output-dir "$GRAMMAR_DIR"
    echo "==> Grammar models ready — rebuild to embed them"

# First-time Android setup: sherpa-onnx libs, ORT shared library, grammar models
setup-android:
    SHERPA_ONNX_LIB_DIR="{{sherpa_libs}}" {{repo_root}}/scripts/android-build.sh --setup-only

# Build debug APK (default)
apk: _ensure-keystore (_build "debug")

# Build release APK
apk-release: _ensure-keystore (_build "release")

# Run Rust library tests
test:
    cd {{tauri_dir}} && cargo test --lib

# Cargo check (fast compile check)
check:
    cd {{tauri_dir}} && cargo check

# Clean all build artifacts
clean:
    cd {{tauri_dir}} && cargo clean
    cd {{android_dir}} && ./gradlew clean

# ── Internal recipes ──

_build profile: _tauri-build _strip _repackage _sign
    @echo ""
    @echo "APK ready: {{repo_root}}/verba.apk"
    @ls -lh {{repo_root}}/verba.apk

# Build via Tauri CLI (handles Rust compilation, frontend bundling, and Gradle packaging)
_tauri-build:
    @echo "==> Building with Tauri CLI (arm64)..."
    @test -f {{sherpa_libs}}/libsherpa-onnx-c-api.a || (echo "ERROR: sherpa-onnx libs not found at {{sherpa_libs}}" && echo "Run: just setup-android" && exit 1)
    cd {{repo_root}} && SHERPA_ONNX_LIB_DIR="{{sherpa_libs}}" npx tauri android build --target aarch64 --apk

# Strip debug symbols from .so to reduce APK size
_strip:
    @echo "==> Stripping native library..."
    {{strip_bin}} --strip-unneeded {{jni_dir}}/arm64-v8a/libverba_rs_lib.so
    rm -rf {{jni_dir}}/x86_64
    @ls -lh {{jni_dir}}/arm64-v8a/libverba_rs_lib.so

# Re-run Gradle to package the stripped .so into the APK
_repackage:
    @echo "==> Repackaging APK with stripped library..."
    cd {{android_dir}} && ./gradlew assembleUniversalRelease \
        -x rustBuildArm64Release \
        -x rustBuildArmRelease \
        -x rustBuildX86_64Release \
        -x rustBuildX86Release

# Align and sign the APK
_sign:
    @echo "==> Signing APK..."
    {{build_tools}}/zipalign -f 4 \
        {{android_dir}}/app/build/outputs/apk/universal/release/app-universal-release-unsigned.apk \
        /tmp/verba-aligned.apk
    {{build_tools}}/apksigner sign \
        --ks {{keystore}} \
        --ks-pass pass:android \
        --key-pass pass:android \
        --out {{repo_root}}/verba.apk \
        /tmp/verba-aligned.apk
    rm -f /tmp/verba-aligned.apk

# Generate debug keystore if it doesn't exist
_ensure-keystore:
    @test -f {{keystore}} || ( \
        echo "==> Generating debug keystore..." && \
        keytool -genkey -v \
            -keystore {{keystore}} \
            -alias debug \
            -keyalg RSA -keysize 2048 \
            -validity 10000 \
            -storepass android \
            -keypass android \
            -dname "CN=Debug" \
    )
