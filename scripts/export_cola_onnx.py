#!/usr/bin/env python3
"""
Export pszemraj/electra-small-discriminator-CoLA to quantized ONNX.

Usage:
    python scripts/export_cola_onnx.py --output-dir /path/to/grammar/

Requires:
    pip install transformers torch onnx onnxruntime
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

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model_id = "pszemraj/electra-small-discriminator-CoLA"
    print(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, torchscript=True)
    model.eval()

    # Save tokenizer.json for the Rust tokenizers crate.
    tokenizer.save_pretrained(str(out))
    tok_src = out / "tokenizer.json"
    tok_dst = out / "cola_tokenizer.json"
    if tok_src.exists() and tok_src != tok_dst:
        shutil.copy(tok_src, tok_dst)
    print(f"Tokenizer: {tok_dst}")

    # Export to FP32 ONNX.
    dummy = tokenizer("Hello world", return_tensors="pt")
    fp32_path = out / "cola_model.onnx"
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits":         {0: "batch"},
        },
        opset_version=14,
    )
    print(f"FP32 ONNX: {fp32_path} ({fp32_path.stat().st_size // 1024}KB)")

    # Quantize to INT8.
    q_path = out / "cola_model_quantized.onnx"
    quantize_dynamic(str(fp32_path), str(q_path), weight_type=QuantType.QInt8)
    fp32_path.unlink()
    print(f"INT8 ONNX: {q_path} ({q_path.stat().st_size // 1024}KB)")

    print(f"\nFiles written to {out}:")
    for f in sorted(out.iterdir()):
        if f.suffix in (".onnx", ".json"):
            print(f"  {f.name} ({f.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
