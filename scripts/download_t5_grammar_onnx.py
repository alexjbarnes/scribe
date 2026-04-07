#!/usr/bin/env python3
"""
Download visheratin/t5-efficient-tiny-grammar-correction quantized ONNX files
and prepare all artifacts needed by the grammar_neural module.

Outputs (all written to --output-dir):
  encoder_model_quantized.onnx       - T5 encoder (INT8)
  decoder_with_past_quantized.onnx   - T5 decoder with KV cache inputs (INT8)
  cross_attn_kv_weights.bin          - 8x [256,256] f32 cross-attention projections
  t5_tokenizer.json                  - SentencePiece tokenizer (HF format)
  special_tokens_map.json            - EOS/UNK/PAD token mappings
  tokenizer_config.json              - Max length, padding side, special token IDs

Usage:
    python scripts/download_t5_grammar_onnx.py --output-dir src-tauri/data/grammar/

Requires:
    pip install huggingface_hub transformers onnx numpy
"""
import argparse
import shutil
import struct
from pathlib import Path

import numpy as np


def download_onnx_files(model_id: str, out: Path):
    from huggingface_hub import hf_hub_download, list_repo_files

    print(f"Listing files in {model_id}...")
    all_files = list(list_repo_files(model_id))

    onnx_files = [f for f in all_files if f.endswith(".onnx")]
    print(f"ONNX files found: {onnx_files}")

    def is_quant(f):
        return "quant" in f.lower()

    # Encoder (quantized preferred).
    enc = next((f for f in onnx_files if "encoder" in f and is_quant(f)), None) \
       or next((f for f in onnx_files if "encoder" in f), None)
    if enc:
        local = hf_hub_download(model_id, enc)
        shutil.copy(local, out / "encoder_model_quantized.onnx")
        print(f"  {enc} -> encoder_model_quantized.onnx")
    else:
        raise RuntimeError("no encoder ONNX found")

    # Decoder with past KV cache (quantized preferred).
    dec_wp = next((f for f in onnx_files if "with_past" in f and is_quant(f)), None) \
          or next((f for f in onnx_files if "with_past" in f), None)
    if dec_wp:
        local = hf_hub_download(model_id, dec_wp)
        shutil.copy(local, out / "decoder_with_past_quantized.onnx")
        print(f"  {dec_wp} -> decoder_with_past_quantized.onnx")
    else:
        raise RuntimeError(
            "no decoder_with_past ONNX found in repo. "
            "You may need to export with Optimum: "
            "ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)"
        )

    # Tokenizer files.
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        if fname in all_files:
            local = hf_hub_download(model_id, fname)
            dest = out / ("t5_tokenizer.json" if fname == "tokenizer.json" else fname)
            shutil.copy(local, dest)
            print(f"  {fname} -> {dest.name}")

    if not (out / "t5_tokenizer.json").exists():
        print("tokenizer.json not found in repo, generating from transformers...")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.save_pretrained(str(out))
        if (out / "tokenizer.json").exists():
            shutil.move(str(out / "tokenizer.json"), str(out / "t5_tokenizer.json"))


def extract_cross_attn_weights(model_path: Path, out_path: Path):
    """Extract cross-attention K/V projection weights from the decoder ONNX model.

    The decoder_with_past model has cross-attention layers that project encoder
    hidden states into keys and values. We pre-compute these projections once
    (from encoder output) rather than re-running them at every decode step.

    Extracts 8 weight matrices (4 layers x K,V): each [256, 256] f32.
    Written as contiguous little-endian floats.
    """
    import onnx

    print(f"\nExtracting cross-attention KV weights from {model_path.name}...")
    model = onnx.load(str(model_path))

    weights_by_name = {}
    for init in model.graph.initializer:
        weights_by_name[init.name] = init

    num_layers = 4
    dim = 256
    matrices = []

    for layer in range(num_layers):
        for proj in ["k", "v"]:
            # Weight naming convention in T5-efficient ONNX exports
            patterns = [
                f"decoder.block.{layer}.layer.1.EncDecAttention.{proj}.weight",
                f"model.decoder.block.{layer}.layer.1.EncDecAttention.{proj}.weight",
            ]
            found = None
            for pattern in patterns:
                if pattern in weights_by_name:
                    found = pattern
                    break

            if found is None:
                # Search by partial match
                for name in weights_by_name:
                    if (f"block.{layer}" in name
                            and "EncDecAttention" in name
                            and f".{proj}." in name
                            and "weight" in name):
                        found = name
                        break

            if found is None:
                raise RuntimeError(
                    f"Could not find cross-attn {proj} weight for layer {layer}. "
                    f"Available initializers containing 'EncDecAttention': "
                    + str([n for n in weights_by_name if "EncDecAttention" in n])
                )

            tensor = weights_by_name[found]
            arr = np.frombuffer(tensor.raw_data, dtype=np.float32).reshape(dim, dim)
            matrices.append(arr)
            print(f"  layer {layer} {proj}: {found} shape={arr.shape}")

    # Write as flat binary: 8 x [256,256] f32 little-endian
    with open(out_path, "wb") as f:
        for mat in matrices:
            f.write(mat.astype("<f4").tobytes())

    expected_bytes = num_layers * 2 * dim * dim * 4
    actual = out_path.stat().st_size
    assert actual == expected_bytes, f"Expected {expected_bytes} bytes, got {actual}"
    print(f"  -> {out_path.name} ({actual // 1024}KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare T5 grammar correction model files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write model files to (e.g. src-tauri/data/grammar/)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_id = "visheratin/t5-efficient-tiny-grammar-correction"
    download_onnx_files(model_id, out)
    extract_cross_attn_weights(
        out / "decoder_with_past_quantized.onnx",
        out / "cross_attn_kv_weights.bin",
    )

    print(f"\nFiles in {out}:")
    for f in sorted(out.iterdir()):
        if f.is_file():
            kb = f.stat().st_size / 1024
            if kb > 1024:
                print(f"  {f.name} ({kb/1024:.1f}MB)")
            else:
                print(f"  {f.name} ({kb:.0f}KB)")


if __name__ == "__main__":
    main()
