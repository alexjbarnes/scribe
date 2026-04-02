#!/usr/bin/env python3
"""
Export pszemraj/electra-small-discriminator-CoLA to quantized ONNX.

Usage:
    python scripts/export_cola_onnx.py --output-dir /path/to/grammar/

Requires:
    pip install optimum[onnxruntime] transformers
"""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Directory to write model files")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer
    from transformers import AutoTokenizer

    model_id = "pszemraj/electra-small-discriminator-CoLA"
    print(f"Exporting {model_id} to ONNX...")

    # Export directly to ONNX via optimum — handles opset, dynamic shapes, etc.
    tmp_dir = out / "_export_tmp"
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(str(tmp_dir))

    # Quantize to INT8.
    quantizer = ORTQuantizer.from_pretrained(str(tmp_dir))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(tmp_dir / "quantized"), quantization_config=qconfig)

    # Copy the files we need to the output dir with canonical names.
    q_model = tmp_dir / "quantized" / "model_quantized.onnx"
    if not q_model.exists():
        # Fall back to unquantized if quantization produced a different name.
        q_model = next((tmp_dir / "quantized").glob("*.onnx"), None) or \
                  next(tmp_dir.glob("*.onnx"), None)

    shutil.copy(q_model, out / "cola_model_quantized.onnx")
    print(f"INT8 ONNX: cola_model_quantized.onnx ({q_model.stat().st_size // 1024}KB)")

    # Save tokenizer.json for the Rust tokenizers crate.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(out))
    tok_src = out / "tokenizer.json"
    tok_dst = out / "cola_tokenizer.json"
    if tok_src.exists():
        shutil.copy(tok_src, tok_dst)
        tok_src.unlink()
    print(f"Tokenizer: {tok_dst}")

    # Clean up temp dir.
    shutil.rmtree(tmp_dir)

    print(f"\nFiles written to {out}:")
    for f in sorted(out.iterdir()):
        if f.suffix in (".onnx", ".json"):
            print(f"  {f.name} ({f.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
