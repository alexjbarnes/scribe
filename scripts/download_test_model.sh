#!/usr/bin/env bash
#
# Download a Parakeet model for running tract integration tests.
#
# Usage:
#   ./scripts/download_test_model.sh            # default: V2 INT8 (~631 MB)
#   ./scripts/download_test_model.sh v2          # V2 full precision (~1.2 GB)
#   ./scripts/download_test_model.sh v3-int8     # V3 INT8 (~670 MB)
#
# Then run tests:
#   cd src-tauri
#   PARAKEET_MODEL_DIR=../test_models/parakeet-v2-int8 \
#     cargo test --test tract_parakeet -- --nocapture

set -euo pipefail

VARIANT="${1:-v2-int8}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

case "$VARIANT" in
  v2-int8)
    BASE_URL="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main"
    OUT_DIR="$REPO_ROOT/test_models/parakeet-v2-int8"
    FILES=(
      "encoder.int8.onnx"
      "decoder.int8.onnx"
      "joiner.int8.onnx"
      "tokens.txt"
    )
    echo "Downloading Parakeet TDT 0.6B V2 INT8 (~631 MB)"
    ;;
  v2)
    BASE_URL="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2/resolve/main"
    OUT_DIR="$REPO_ROOT/test_models/parakeet-v2"
    FILES=(
      "encoder.onnx"
      "decoder.onnx"
      "joiner.onnx"
      "tokens.txt"
    )
    echo "Downloading Parakeet TDT 0.6B V2 (~1.2 GB)"
    ;;
  v3-int8)
    BASE_URL="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main"
    OUT_DIR="$REPO_ROOT/test_models/parakeet-v3-int8"
    FILES=(
      "encoder.int8.onnx"
      "decoder.int8.onnx"
      "joiner.int8.onnx"
      "tokens.txt"
    )
    echo "Downloading Parakeet TDT 0.6B V3 INT8 (~670 MB)"
    ;;
  v3)
    BASE_URL="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3/resolve/main"
    OUT_DIR="$REPO_ROOT/test_models/parakeet-v3"
    FILES=(
      "encoder.onnx"
      "encoder.weights"
      "decoder.onnx"
      "joiner.onnx"
      "tokens.txt"
    )
    echo "Downloading Parakeet TDT 0.6B V3 (~2.5 GB)"
    echo "WARNING: V3 full precision uses external weights (encoder.weights = 2.4 GB)"
    ;;
  *)
    echo "Unknown variant: $VARIANT"
    echo "Options: v2-int8 (default), v2, v3-int8, v3"
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

for file in "${FILES[@]}"; do
  dest="$OUT_DIR/$file"
  if [ -f "$dest" ]; then
    echo "  Already exists: $file"
    continue
  fi
  echo "  Downloading: $file"
  curl -L --progress-bar -o "$dest.tmp" "$BASE_URL/$file"
  mv "$dest.tmp" "$dest"
done

echo ""
echo "Done. Model saved to: $OUT_DIR"
echo ""
echo "Run tests with:"
echo "  cd src-tauri"
echo "  PARAKEET_MODEL_DIR=$OUT_DIR cargo test --test tract_parakeet -- --nocapture"
