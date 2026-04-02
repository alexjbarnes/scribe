#!/usr/bin/env python3
"""
Export Unbabel/gec-t5_small to ONNX INT8 and verify output quality is preserved.

Exports with Hugging Face Optimum, then runs a subset of the grammar
correction test cases through both PyTorch and ONNX to confirm they match.

Requirements:
    pip install optimum[onnxruntime] onnx

Usage:
    python3 scripts/export_gec_t5_onnx.py
    python3 scripts/export_gec_t5_onnx.py --output-dir /tmp/gec-t5-onnx
    python3 scripts/export_gec_t5_onnx.py --skip-export  # if already exported
"""

import argparse
import os
import time
from pathlib import Path

# ── Python 3.14 / Optimum 2.x compatibility fix ───────────────────────────────
#
# In Python 3.14, functools.partial wrapping a class is treated as a descriptor
# and gets bound to the instance when accessed via self.ATTR. This causes
# NormalizedConfig.__init__() to receive the T5OnnxConfig instance as a
# positional arg, landing on the `allow_new` parameter → duplicate kwargs error.
#
# Fix: access NORMALIZED_CONFIG_CLASS via type(self) to avoid descriptor binding.
def _patch_optimum_py314():
    from optimum.exporters import base as export_base

    def fixed_init(self, config, task, int_dtype="int64", float_dtype="fp32"):
        self.task = task
        self._config = config
        self._normalized_config = type(self).NORMALIZED_CONFIG_CLASS(self._config)
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype

    export_base.ExporterConfig.__init__ = fixed_init

_patch_optimum_py314()

TEST_CASES = [
    ("gec: their is a problem with the build",          "There is a problem with the build."),
    ("gec: we need to look at there approach",          "We need to look at their approach."),
    ("gec: yesterday I remove all the old log files",   "Yesterday I removed all the old log files."),
    ("gec: he walk to the office every day",            "He walks to the office every day."),
    ("gec: I should of called them earlier",            "I should have called them earlier."),
    ("gec: We deployed the new API endpoint yesterday.","We deployed the new API endpoint yesterday."),
    ("gec: The function returns a list of strings.",    "The function returns a list of strings."),
    ("gec: API endpoints need to be REST-compliant.",   "API endpoints need to be REST-compliant."),
]

MODEL_ID   = "Unbabel/gec-t5_small"
TASK       = "text2text-generation"


def export(output_dir: str):
    """Export via ORTModelForSeq2SeqLM(export=True) — simpler, avoids Optimum main_export bugs."""
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    print(f"\nExporting {MODEL_ID} to ONNX → {output_dir} ...")
    t0 = time.perf_counter()

    model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_ID, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Export done in {(time.perf_counter()-t0):.1f}s")

    # Show exported files
    for f in sorted(Path(output_dir).rglob("*.onnx")):
        mb = f.stat().st_size / 1e6
        print(f"  {f.relative_to(output_dir)}  {mb:.1f}MB")


def quantize(input_dir: str, output_dir: str):
    """Dynamic INT8 quantization on exported ONNX files."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

    for onnx_file in Path(input_dir).glob("*.onnx"):
        print(f"  Quantizing {onnx_file.name} ...")
        t0 = time.perf_counter()
        quantizer = ORTQuantizer.from_pretrained(input_dir, file_name=onnx_file.name)
        quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)
        out_path = Path(output_dir) / onnx_file.name
        mb_in  = onnx_file.stat().st_size / 1e6
        mb_out = out_path.stat().st_size / 1e6 if out_path.exists() else 0
        print(f"    {mb_in:.1f}MB → {mb_out:.1f}MB  ({(time.perf_counter()-t0):.1f}s)")


def run_pytorch(cases: list) -> list:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    print(f"\nRunning PyTorch baseline ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m   = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    pipe = pipeline("text2text-generation", model=m, tokenizer=tok)
    results = []
    for inp, _ in cases:
        t0 = time.perf_counter()
        out = pipe(inp, max_new_tokens=128)[0]["generated_text"].strip()
        ms  = (time.perf_counter() - t0) * 1000
        results.append((out, ms))
    return results


def run_onnx(cases: list, model_dir: str) -> list:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer, pipeline
    print(f"\nRunning ONNX from {model_dir} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m   = ORTModelForSeq2SeqLM.from_pretrained(model_dir)
    pipe = pipeline("text2text-generation", model=m, tokenizer=tok)
    results = []
    for inp, _ in cases:
        t0 = time.perf_counter()
        out = pipe(inp, max_new_tokens=128)[0]["generated_text"].strip()
        ms  = (time.perf_counter() - t0) * 1000
        results.append((out, ms))
    return results


def compare(pytorch_results, onnx_results):
    col = 40
    print(f"\n{'Input':<{col}}  {'PyTorch':<{col}}  {'ONNX':<{col}}  Match")
    print("─" * (col * 3 + 20))
    matches = 0
    for (inp, _), (pt_out, pt_ms), (ort_out, ort_ms) in zip(
        TEST_CASES, pytorch_results, onnx_results
    ):
        inp_s  = inp[5:][:col] + "…" if len(inp[5:]) > col else inp[5:]  # strip "gec: "
        pt_s   = pt_out[:col]  + "…" if len(pt_out)  > col else pt_out
        ort_s  = ort_out[:col] + "…" if len(ort_out) > col else ort_out
        match  = "OK" if pt_out.lower() == ort_out.lower() else "DIFF"
        if match == "OK":
            matches += 1
        print(f"  {inp_s:<{col}}  {pt_s:<{col}}  {ort_s:<{col}}  {match}")

    import statistics
    pt_ms   = [r[1] for r in pytorch_results]
    ort_ms  = [r[1] for r in onnx_results]
    print(f"\nOutput match: {matches}/{len(TEST_CASES)}")
    print(f"PyTorch median: {statistics.median(pt_ms):.0f}ms")
    print(f"ONNX    median: {statistics.median(ort_ms):.0f}ms  "
          f"({statistics.median(pt_ms)/statistics.median(ort_ms):.1f}x speedup)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/gec-t5-onnx")
    parser.add_argument("--int8-dir",   default="/tmp/gec-t5-onnx-int8")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export/quantize if already done")
    args = parser.parse_args()

    if not args.skip_export:
        export(args.output_dir)
        quantize(args.output_dir, args.int8_dir)

    print("\n── Comparing FP32 ONNX vs PyTorch ──────────────────────────────────")
    pt_results  = run_pytorch(TEST_CASES)
    ort_fp32    = run_onnx(TEST_CASES, args.output_dir)
    compare(pt_results, ort_fp32)

    int8_dir = Path(args.int8_dir)
    if int8_dir.exists() and any(int8_dir.glob("*.onnx")):
        print("\n── Comparing INT8 ONNX vs PyTorch ──────────────────────────────────")
        ort_int8 = run_onnx(TEST_CASES, args.int8_dir)
        compare(pt_results, ort_int8)
    else:
        print(f"\nINT8 dir not found or empty: {int8_dir}")


if __name__ == "__main__":
    main()
