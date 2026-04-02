#!/usr/bin/env python3
"""
Download visheratin/t5-efficient-tiny-grammar-correction quantized ONNX files.

Places encoder_model_quantized.onnx, decoder_model_quantized.onnx, and
t5_tokenizer.json in the output directory, ready for the grammar_neural module.

Usage:
    python scripts/download_t5_grammar_onnx.py --output-dir /path/to/grammar/

Requires:
    pip install huggingface_hub transformers
"""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download, list_repo_files

    model_id = "visheratin/t5-efficient-tiny-grammar-correction"
    print(f"Listing files in {model_id}...")
    all_files = list(list_repo_files(model_id))

    onnx_files = [f for f in all_files if f.endswith(".onnx")]
    print(f"ONNX files found: {onnx_files}")

    def is_quant(f):
        return "quant" in f.lower()  # matches both "quant" and "quantized"

    # Download encoder (quantized preferred, fallback to unquantized).
    enc = next((f for f in onnx_files if "encoder" in f and is_quant(f)), None) \
       or next((f for f in onnx_files if "encoder" in f), None)
    if enc:
        local = hf_hub_download(model_id, enc)
        shutil.copy(local, out / "encoder_model_quantized.onnx")
        print(f"  {enc} -> encoder_model_quantized.onnx")
    else:
        print("ERROR: no encoder ONNX found")
        return

    # Download decoder (no 'with_past'/'init', quantized preferred).
    dec = next(
        (f for f in onnx_files if "decoder" in f and "with_past" not in f and "init" not in f and is_quant(f)),
        None,
    ) or next(
        (f for f in onnx_files if "decoder" in f and "with_past" not in f and "init" not in f),
        None,
    )
    if dec:
        local = hf_hub_download(model_id, dec)
        shutil.copy(local, out / "decoder_model_quantized.onnx")
        print(f"  {dec} -> decoder_model_quantized.onnx")
    else:
        print("ERROR: no decoder ONNX found")
        return

    # Download tokenizer files.
    for fname in ["tokenizer.json", "tokenizer_config.json", "spiece.model"]:
        if fname in all_files:
            local = hf_hub_download(model_id, fname)
            dest = out / ("t5_tokenizer.json" if fname == "tokenizer.json" else fname)
            shutil.copy(local, dest)
            print(f"  {fname} -> {dest.name}")

    if not (out / "t5_tokenizer.json").exists():
        # Fall back: generate tokenizer.json from the transformers tokenizer.
        print("tokenizer.json not found in repo — generating from transformers...")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.save_pretrained(str(out))
        if (out / "tokenizer.json").exists():
            shutil.move(str(out / "tokenizer.json"), str(out / "t5_tokenizer.json"))

    print(f"\nFiles in {out}:")
    for f in sorted(out.iterdir()):
        if f.suffix in (".onnx", ".json", ".model"):
            print(f"  {f.name} ({f.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
