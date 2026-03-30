#!/usr/bin/env bash
#
# Build Scribe for Android (arm64-v8a).
#
# NOTE: For day-to-day builds, use `just apk` instead. This script is
# primarily useful for first-time setup (--setup-only) to build and cache
# the sherpa-onnx native libraries.
#
# Usage:
#   ./scripts/android-build.sh --setup-only # build sherpa-onnx libs (first time)
#   ./scripts/android-build.sh              # debug APK (prefer `just apk`)
#   ./scripts/android-build.sh --release    # release APK (prefer `just apk-release`)
#
# Prerequisites (installed automatically where possible):
#   - Android SDK with platform 34 (ANDROID_HOME)
#   - Android NDK r28+ (ANDROID_NDK_HOME)
#   - JDK 17+
#   - Rust aarch64-linux-android target
#   - cmake, ninja-build
#
# The script builds sherpa-onnx native libraries from source on first run
# and caches them for subsequent builds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TAURI_DIR="$REPO_ROOT/src-tauri"
ANDROID_PROJECT="$TAURI_DIR/gen/android"
SHERPA_ONNX_VERSION="1.12.34"
BUILD_TYPE="debug"
SETUP_ONLY=false

# ── Parse args ──

for arg in "$@"; do
    case "$arg" in
        --release) BUILD_TYPE="release" ;;
        --setup-only) SETUP_ONLY=true ;;
        --help|-h)
            head -17 "$0" | tail -14
            exit 0
            ;;
    esac
done

# ── Helpers ──

info()  { echo "==> $*"; }
warn()  { echo "WARNING: $*" >&2; }
die()   { echo "ERROR: $*" >&2; exit 1; }

check_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$1 not found. $2"
}

# ── Step 1: Check prerequisites ──

info "Checking prerequisites..."

check_cmd rustup "Install from https://rustup.rs"
check_cmd cargo  "Install from https://rustup.rs"
check_cmd cmake  "Install with: apt install cmake / brew install cmake"
check_cmd java   "Install JDK 17+: apt install openjdk-17-jdk / brew install openjdk@17"

# Rust Android target
if ! rustup target list --installed | grep -q aarch64-linux-android; then
    info "Installing Rust target aarch64-linux-android..."
    rustup target add aarch64-linux-android
fi

# Android SDK
if [ -z "${ANDROID_HOME:-}" ]; then
    # Common locations
    for candidate in \
        "$HOME/Android/Sdk" \
        "$HOME/Library/Android/sdk" \
        "/opt/android-sdk" \
        "$HOME/.android/sdk"; do
        if [ -d "$candidate" ]; then
            export ANDROID_HOME="$candidate"
            break
        fi
    done
fi
[ -d "${ANDROID_HOME:-}" ] || die "ANDROID_HOME not set and Android SDK not found.
  Install Android Studio or set ANDROID_HOME manually.
  See: https://developer.android.com/studio"

info "Android SDK: $ANDROID_HOME"

# Android NDK
if [ -z "${ANDROID_NDK_HOME:-}" ]; then
    # Find the latest installed NDK
    NDK_DIR="$ANDROID_HOME/ndk"
    if [ -d "$NDK_DIR" ]; then
        LATEST_NDK=$(ls -1 "$NDK_DIR" 2>/dev/null | sort -V | tail -1)
        if [ -n "$LATEST_NDK" ]; then
            export ANDROID_NDK_HOME="$NDK_DIR/$LATEST_NDK"
        fi
    fi
fi
[ -d "${ANDROID_NDK_HOME:-}" ] || die "ANDROID_NDK_HOME not set and no NDK found.
  Install via: sdkmanager --install 'ndk;28.0.13004108'
  Or set ANDROID_NDK_HOME manually."

NDK_VERSION=$(basename "$ANDROID_NDK_HOME" | cut -d. -f1)
if [ "$NDK_VERSION" -lt 28 ] 2>/dev/null; then
    warn "NDK version $NDK_VERSION < 28. Google Play requires NDK 28+ for 16KB page alignment."
fi
info "Android NDK: $ANDROID_NDK_HOME"

# Tauri CLI
if ! npx tauri --version >/dev/null 2>&1; then
    die "Tauri CLI not found. Run: npm install @tauri-apps/cli"
fi

# ── Step 2: Build sherpa-onnx native libraries ──

SHERPA_ONNX_CACHE="$REPO_ROOT/.android-deps/sherpa-onnx"
SHERPA_ONNX_LIB_DIR="$SHERPA_ONNX_CACHE/lib"

if [ -d "$SHERPA_ONNX_LIB_DIR" ] && [ -f "$SHERPA_ONNX_LIB_DIR/libsherpa-onnx-c-api.a" ]; then
    info "Using cached sherpa-onnx libraries from $SHERPA_ONNX_LIB_DIR"
else
    info "Building sherpa-onnx v${SHERPA_ONNX_VERSION} for Android arm64-v8a (this takes 10-15 minutes)..."

    SHERPA_SRC="$SHERPA_ONNX_CACHE/src"
    SHERPA_BUILD="$SHERPA_ONNX_CACHE/build"

    mkdir -p "$SHERPA_ONNX_CACHE"

    # Clone or update source
    if [ -d "$SHERPA_SRC/.git" ]; then
        info "Updating sherpa-onnx source..."
        git -C "$SHERPA_SRC" fetch --depth 1 origin "v${SHERPA_ONNX_VERSION}"
        git -C "$SHERPA_SRC" checkout FETCH_HEAD
    else
        info "Cloning sherpa-onnx v${SHERPA_ONNX_VERSION}..."
        git clone --depth 1 --branch "v${SHERPA_ONNX_VERSION}" \
            https://github.com/k2-fsa/sherpa-onnx.git "$SHERPA_SRC"
    fi

    # CMake cross-compile
    TOOLCHAIN="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"
    [ -f "$TOOLCHAIN" ] || die "NDK toolchain not found at $TOOLCHAIN"

    rm -rf "$SHERPA_BUILD"
    mkdir -p "$SHERPA_BUILD"

    cmake -S "$SHERPA_SRC" -B "$SHERPA_BUILD" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-28 \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DSHERPA_ONNX_ENABLE_C_API=ON \
        -DSHERPA_ONNX_ENABLE_BINARY=OFF \
        -DSHERPA_ONNX_ENABLE_JNI=OFF \
        -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
        -DSHERPA_ONNX_ENABLE_TESTS=OFF \
        -DSHERPA_ONNX_ENABLE_CHECK=OFF \
        -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
        -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
        -DCMAKE_INSTALL_PREFIX="$SHERPA_ONNX_CACHE"

    cmake --build "$SHERPA_BUILD" --config Release -j "$(nproc 2>/dev/null || echo 4)"
    cmake --install "$SHERPA_BUILD"

    if [ ! -f "$SHERPA_ONNX_LIB_DIR/libsherpa-onnx-c-api.a" ]; then
        # Some builds put libs in lib64 or other locations
        for candidate in \
            "$SHERPA_ONNX_CACHE/lib64" \
            "$SHERPA_BUILD/lib" \
            "$SHERPA_BUILD/lib64"; do
            if [ -f "$candidate/libsherpa-onnx-c-api.a" ]; then
                mkdir -p "$SHERPA_ONNX_LIB_DIR"
                cp "$candidate"/*.a "$SHERPA_ONNX_LIB_DIR/"
                break
            fi
        done
    fi

    [ -f "$SHERPA_ONNX_LIB_DIR/libsherpa-onnx-c-api.a" ] || \
        die "sherpa-onnx build succeeded but libsherpa-onnx-c-api.a not found in expected locations."

    info "sherpa-onnx libraries built and cached at $SHERPA_ONNX_LIB_DIR"

    # Clean build dir to save space (keep source for rebuilds)
    rm -rf "$SHERPA_BUILD"
fi

export SHERPA_ONNX_LIB_DIR

# ── Step 3: Initialize Tauri Android project ──

if [ ! -d "$ANDROID_PROJECT" ]; then
    info "Initializing Tauri Android project..."
    cd "$REPO_ROOT"
    npx tauri android init
    info "Android project created at $ANDROID_PROJECT"
else
    info "Tauri Android project already exists"
fi

# ── Step 4: Configure Android permissions ──

MANIFEST="$ANDROID_PROJECT/app/src/main/AndroidManifest.xml"
if [ -f "$MANIFEST" ]; then
    # Add RECORD_AUDIO permission if not present
    if ! grep -q "RECORD_AUDIO" "$MANIFEST"; then
        info "Adding RECORD_AUDIO permission to AndroidManifest.xml..."
        sed -i 's|<application|<uses-permission android:name="android.permission.RECORD_AUDIO" />\n    <application|' "$MANIFEST"
    fi
    # Add INTERNET permission if not present (for model downloads)
    if ! grep -q "android.permission.INTERNET" "$MANIFEST"; then
        info "Adding INTERNET permission to AndroidManifest.xml..."
        sed -i 's|<application|<uses-permission android:name="android.permission.INTERNET" />\n    <application|' "$MANIFEST"
    fi
fi

# ── Step 5: Build ──

if $SETUP_ONLY; then
    info "Setup complete. Run without --setup-only to build."
    exit 0
fi

info "Building Scribe for Android ($BUILD_TYPE)..."
cd "$REPO_ROOT"

BUILD_ARGS=()
if [ "$BUILD_TYPE" = "release" ]; then
    BUILD_ARGS+=(--release)
fi

# Pass the library path and NDK linker setup through cargo config
export SHERPA_ONNX_LIB_DIR
export ANDROID_HOME
export ANDROID_NDK_HOME

npx tauri android build "${BUILD_ARGS[@]}"

# ── Done ──

if [ "$BUILD_TYPE" = "release" ]; then
    APK_DIR="$ANDROID_PROJECT/app/build/outputs/apk/universal/release"
else
    APK_DIR="$ANDROID_PROJECT/app/build/outputs/apk/universal/debug"
fi

if [ -d "$APK_DIR" ]; then
    info "APK built:"
    ls -lh "$APK_DIR"/*.apk 2>/dev/null || true
else
    info "Build complete. Check $ANDROID_PROJECT/app/build/outputs/ for APK."
fi
