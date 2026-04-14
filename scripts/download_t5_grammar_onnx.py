#!/usr/bin/env python3
"""
Download and prepare T5 grammar correction ONNX files for the grammar_neural module.

Downloads pre-exported ONNX files from HuggingFace, quantizes to INT8, normalizes
input/output names to match what the Rust code expects, and extracts cross-attention
KV projection weights.

Outputs (all written to --output-dir, with optional --version suffix):
  encoder_model_quantized.onnx       - T5 encoder (INT8)
  decoder_with_past_quantized.onnx   - T5 decoder with KV cache inputs (INT8)
  cross_attn_kv_weights.bin          - 8x [256,256] f32 cross-attention projections
  t5_tokenizer.json                  - SentencePiece tokenizer (HF format)
  special_tokens_map.json            - EOS/UNK/PAD token mappings
  tokenizer_config.json              - Max length, padding side, special token IDs

Usage:
    python scripts/download_t5_grammar_onnx.py --output-dir src-tauri/data/grammar/
    python scripts/download_t5_grammar_onnx.py --output-dir src-tauri/data/grammar/ --version 0.0.1

Requires:
    pip install huggingface_hub onnx onnxruntime numpy
"""
import argparse
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx


MODEL_ID = "visheratin/t5-efficient-tiny-grammar-correction"

# Canonical decoder input names matching the Rust grammar_neural code.
# Rust feeds inputs by name, reads outputs by positional index.
EXPECTED_DECODER_INPUTS = {
    "input_ids",
    "encoder_attention_mask",
    "encoder_hidden_states",
    *(f"pkv_{i}" for i in range(16)),
}


def download_onnx_files(model_id: str, tmp_dir: Path):
    """Download ONNX model files from HuggingFace."""
    from huggingface_hub import hf_hub_download, list_repo_files

    print(f"Listing files in {model_id}...")
    all_files = list(list_repo_files(model_id))
    onnx_files = [f for f in all_files if f.endswith(".onnx")]
    print(f"ONNX files found: {onnx_files}")

    def is_quant(f):
        return "quant" in f.lower()

    # Encoder: prefer quantized if available in repo
    enc = next((f for f in onnx_files if "encoder" in f and is_quant(f)), None) \
       or next((f for f in onnx_files if "encoder" in f), None)
    if enc is None:
        raise RuntimeError(f"No encoder ONNX found in {onnx_files}")
    local = hf_hub_download(model_id, enc)
    enc_path = tmp_dir / "encoder.onnx"
    shutil.copy(local, enc_path)
    print(f"  {enc} -> {enc_path.name} (needs_quant={not is_quant(enc)})")

    # Decoder with past KV cache.
    # Naming conventions vary across Optimum versions:
    #   - Older exports: decoder_with_past_model.onnx
    #   - Newer exports: decoder_model.onnx (contains past KV inputs)
    # We detect the right file by checking for past KV inputs.
    dec_candidates = [
        f for f in onnx_files
        if "decoder" in f and "init" not in f
    ]
    dec_wp = None
    for candidate in dec_candidates:
        local = hf_hub_download(model_id, candidate)
        m = onnx.load(local)
        input_names = {inp.name for inp in m.graph.input}
        has_kv = any("pkv" in n or "past_key_values" in n for n in input_names)
        if has_kv:
            dec_wp = local
            print(f"  {candidate} -> decoder with past KV ({len(input_names)} inputs)")
            break
        print(f"  {candidate}: no past KV inputs, skipping")

    if dec_wp is None:
        raise RuntimeError(
            f"No decoder-with-past ONNX found in {onnx_files}. "
            f"All decoder candidates lacked past KV cache inputs."
        )
    dec_path = tmp_dir / "decoder_with_past.onnx"
    shutil.copy(dec_wp, dec_path)

    # Tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        if fname in all_files:
            local = hf_hub_download(model_id, fname)
            shutil.copy(local, tmp_dir / fname)
            print(f"  {fname}")

    return enc_path, dec_path, is_quant(enc)


def normalize_decoder_inputs(model_path: Path) -> Path:
    """Rename decoder inputs to canonical names if needed.

    Different Optimum versions produce different naming conventions:
      - Older: pkv_0, pkv_1, ..., encoder_hidden_states
      - Newer: past_key_values.0.decoder.key, ..., possibly no encoder_hidden_states

    Normalizes to: pkv_0..pkv_15 + encoder_hidden_states.
    """
    model = onnx.load(str(model_path))
    input_names = {inp.name for inp in model.graph.input}

    if input_names == EXPECTED_DECODER_INPUTS:
        print(f"  Decoder inputs already canonical")
        return model_path

    rename_map = {}

    # Map past_key_values.L.{decoder,encoder}.{key,value} -> pkv_N
    # Per-layer order: decoder_key=0, decoder_value=1, encoder_key=2, encoder_value=3
    pkv_pattern = re.compile(
        r"past_key_values\.(\d+)\.(decoder|encoder)\.(key|value)"
    )
    offset_table = {
        ("decoder", "key"): 0, ("decoder", "value"): 1,
        ("encoder", "key"): 2, ("encoder", "value"): 3,
    }
    for inp in model.graph.input:
        m = pkv_pattern.match(inp.name)
        if m:
            layer = int(m.group(1))
            idx = layer * 4 + offset_table[(m.group(2), m.group(3))]
            rename_map[inp.name] = f"pkv_{idx}"

    # Find encoder_hidden_states under alternative names
    if "encoder_hidden_states" not in input_names:
        for inp in model.graph.input:
            if inp.name in rename_map or inp.name in ("input_ids", "encoder_attention_mask"):
                continue
            dims = inp.type.tensor_type.shape.dim
            if len(dims) == 3:
                last_dim = dims[2].dim_value
                if last_dim in (256, 512, 768):
                    rename_map[inp.name] = "encoder_hidden_states"
                    print(f"  Mapped {inp.name} -> encoder_hidden_states (3D, hidden={last_dim})")
                    break

    if not rename_map:
        print(f"  No renames needed")
        return model_path

    print(f"  Renaming {len(rename_map)} nodes:")
    for old, new in sorted(rename_map.items(), key=lambda x: x[1]):
        print(f"    {old} -> {new}")

    # Apply renames everywhere in the graph
    for inp in model.graph.input:
        if inp.name in rename_map:
            inp.name = rename_map[inp.name]
    for out in model.graph.output:
        if out.name in rename_map:
            out.name = rename_map[out.name]
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in rename_map:
                node.input[i] = rename_map[name]
        for i, name in enumerate(node.output):
            if name in rename_map:
                node.output[i] = rename_map[name]
    for init in model.graph.initializer:
        if init.name in rename_map:
            init.name = rename_map[init.name]

    out_path = model_path.with_name("decoder_with_past_normalized.onnx")
    onnx.save(model, str(out_path))
    return out_path


def normalize_encoder_outputs(model_path: Path) -> Path:
    """Ensure the encoder's primary output is named 'hidden_states'."""
    model = onnx.load(str(model_path))
    output_names = {out.name for out in model.graph.output}

    if "hidden_states" in output_names:
        print(f"  Encoder output already named 'hidden_states'")
        return model_path

    rename_map = {}
    if "last_hidden_state" in output_names:
        rename_map["last_hidden_state"] = "hidden_states"

    if not rename_map:
        print(f"  Encoder outputs: {output_names} (no rename needed)")
        return model_path

    print(f"  Renaming encoder outputs: {rename_map}")
    for out in model.graph.output:
        if out.name in rename_map:
            out.name = rename_map[out.name]
    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name in rename_map:
                node.output[i] = rename_map[name]

    out_path = model_path.with_name("encoder_normalized.onnx")
    onnx.save(model, str(out_path))
    return out_path


def quantize_model(model_path: Path, output_path: Path):
    """Quantize an ONNX model to INT8 dynamic quantization."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"  Quantizing {model_path.name}...")
    quantize_dynamic(
        str(model_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )
    orig_mb = model_path.stat().st_size / (1024 * 1024)
    quant_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  {orig_mb:.1f}MB -> {quant_mb:.1f}MB")


def extract_cross_attn_weights(model_id: str, out_path: Path):
    """Extract cross-attention K/V projection weights from the PyTorch model.

    ONNX exports use opaque initializer names (onnx::MatMul_NNN), so we pull
    the weights from PyTorch where they have proper names like
    decoder.block.0.layer.1.EncDecAttention.k.weight.

    Extracts 8 weight matrices (4 layers x K,V): each [256, 256] f32.
    Written as contiguous little-endian floats.
    """
    from transformers import T5ForConditionalGeneration

    print(f"\nExtracting cross-attention KV weights from PyTorch model...")
    model = T5ForConditionalGeneration.from_pretrained(model_id)

    num_layers = 4
    dim = 256
    matrices = []

    for layer in range(num_layers):
        for proj in ["k", "v"]:
            name = f"decoder.block.{layer}.layer.1.EncDecAttention.{proj}.weight"
            param = dict(model.named_parameters()).get(name)
            if param is None:
                raise RuntimeError(f"Could not find {name} in model parameters")

            arr = param.detach().cpu().numpy()
            if arr.shape != (dim, dim):
                raise RuntimeError(f"{name}: expected ({dim},{dim}), got {arr.shape}")
            matrices.append(arr)
            print(f"  layer {layer} {proj}: {name} [{arr.shape[0]}x{arr.shape[1]}]")

    with open(out_path, "wb") as f:
        for mat in matrices:
            f.write(mat.astype("<f4").tobytes())

    expected_bytes = num_layers * 2 * dim * dim * 4
    actual = out_path.stat().st_size
    assert actual == expected_bytes, f"Expected {expected_bytes} bytes, got {actual}"
    print(f"  -> {out_path.name} ({actual // 1024}KB)")


def verify_decoder(model_path: Path):
    """Print and validate final decoder input/output names."""
    model = onnx.load(str(model_path))
    input_names = {inp.name for inp in model.graph.input}

    print(f"\nDecoder verification ({model_path.name}):")
    print(f"  Inputs ({len(model.graph.input)}):")
    for inp in model.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {dims}")

    missing = EXPECTED_DECODER_INPUTS - input_names
    extra = input_names - EXPECTED_DECODER_INPUTS
    if missing:
        print(f"\n  FAIL: missing inputs: {missing}")
        return False
    if extra:
        print(f"\n  WARNING: unexpected extra inputs: {extra}")
    print(f"\n  OK: all {len(EXPECTED_DECODER_INPUTS)} expected inputs present")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare T5 grammar correction ONNX model files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write model files to (e.g. src-tauri/data/grammar/)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version suffix for output files (e.g. 0.0.1 -> encoder_model_quantized.0.0.1.onnx)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def versioned(name: str) -> str:
        if args.version is None:
            return name
        stem, ext = name.rsplit(".", 1)
        return f"{stem}.{args.version}.{ext}"

    with tempfile.TemporaryDirectory(prefix="grammar-export-") as tmp:
        tmp_dir = Path(tmp)

        # Step 1: Download from HuggingFace
        enc_path, dec_path, enc_already_quantized = download_onnx_files(
            MODEL_ID, tmp_dir
        )

        # Step 2: Normalize input/output names
        print(f"\nNormalizing decoder inputs...")
        dec_path = normalize_decoder_inputs(dec_path)

        print(f"\nNormalizing encoder outputs...")
        enc_path = normalize_encoder_outputs(enc_path)

        # Step 3: Quantize (skip encoder if repo already had quantized version)
        print(f"\nQuantizing models...")
        enc_quant = tmp_dir / "encoder_model_quantized.onnx"
        dec_quant = tmp_dir / "decoder_with_past_quantized.onnx"

        if enc_already_quantized:
            shutil.copy(enc_path, enc_quant)
            print(f"  Encoder already quantized, copying as-is")
        else:
            quantize_model(enc_path, enc_quant)
        quantize_model(dec_path, dec_quant)

        # Step 4: Extract cross-attention weights from PyTorch model.
        # ONNX exports use opaque initializer names, so we pull from PyTorch.
        cross_attn_path = tmp_dir / "cross_attn_kv_weights.bin"
        extract_cross_attn_weights(MODEL_ID, cross_attn_path)

        # Step 5: Verify decoder has correct inputs
        if not verify_decoder(dec_quant):
            raise RuntimeError("Decoder verification failed, aborting")

        # Step 6: Copy to output directory
        print(f"\nCopying to {out}:")
        file_map = [
            (enc_quant, versioned("encoder_model_quantized.onnx")),
            (dec_quant, versioned("decoder_with_past_quantized.onnx")),
            (cross_attn_path, versioned("cross_attn_kv_weights.bin")),
        ]
        for src, dst_name in file_map:
            shutil.copy(src, out / dst_name)
            print(f"  -> {dst_name}")

        # Tokenizer
        tok_src = tmp_dir / "tokenizer.json"
        if tok_src.exists():
            shutil.copy(tok_src, out / versioned("t5_tokenizer.json"))
            print(f"  -> {versioned('t5_tokenizer.json')}")

        for fname in ["tokenizer_config.json", "special_tokens_map.json"]:
            src = tmp_dir / fname
            if src.exists():
                shutil.copy(src, out / fname)
                print(f"  -> {fname}")

    # Summary
    print(f"\nDone. Files in {out}:")
    for f in sorted(out.iterdir()):
        if f.is_file():
            kb = f.stat().st_size / 1024
            label = f"{kb/1024:.1f}MB" if kb > 1024 else f"{kb:.0f}KB"
            print(f"  {f.name} ({label})")


if __name__ == "__main__":
    main()
