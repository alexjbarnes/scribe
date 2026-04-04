# Fine-Tuning the Grammar Models

Guide for fine-tuning the CoLA router and T5 corrector on transcription-specific error pairs.

## Current Models

| Component | Model | Params | Format | Size |
|-----------|-------|--------|--------|------|
| Router | pszemraj/electra-small-discriminator-CoLA | 13.5M | ONNX INT8 | 14MB |
| Corrector | visheratin/t5-efficient-tiny-grammar-correction | 15.6M | ONNX INT8 | 32MB |

Both are small enough to fine-tune on a single consumer GPU in under an hour.

## Why Fine-Tune

The corrector was trained on C4_200M (web fluency rewrites), not ASR output. It does not know the error distribution of Whisper/Parakeet. Common failure modes:

- Negation inversion: "isn't working" corrected to "is working"
- Meaning drift on informal speech: "your going to" becomes "are you going to?"
- Over-correction of clean technical text: inserts articles before "API", "CI"
- Drops contractions: "I should of called" becomes "I should call" instead of "I should have called"

The router was trained on CoLA (linguistic acceptability), which is grammatical structure judgment. It misses lexical errors that are structurally valid: "your going to" parses as valid PRON VERB PARTICLE VERB.

Fine-tuning on real transcription pairs teaches both models the specific error patterns from your ASR pipeline.

## Training Data Format

### Corrector (T5)

Standard seq2seq pairs. Input is the raw transcription, target is the corrected version.

```jsonl
{"input": "your going to love this new feature", "target": "you're going to love this new feature"}
{"input": "their is a problem with the build", "target": "there is a problem with the build"}
{"input": "The PR is in review and CI is green.", "target": "The PR is in review and CI is green."}
```

Include both error cases (where input != target) and clean cases (where input == target). The model needs to learn when NOT to change text.

### Router (CoLA classifier)

Binary classification. Label 1 = acceptable (pass through), label 0 = needs correction.

```jsonl
{"text": "your going to love this new feature", "label": 0}
{"text": "you're going to love this new feature", "label": 1}
{"text": "The PR is in review and CI is green.", "label": 1}
{"text": "their is a problem with the build", "label": 0}
```

## How Many Examples

### Corrector

| Dataset size | Expected outcome |
|---|---|
| 50-100 pairs | Fixes the worst failure modes (negation inversion, contraction handling). Minimal impact on general capability. Start here. |
| 200-500 pairs | Solid coverage of common ASR error patterns. The model learns your specific domain vocabulary (technical terms, proper nouns). |
| 1000+ pairs | Diminishing returns for a 15M param model. Risk of overfitting. Only worthwhile if you expand to multiple domains or languages. |

A good initial dataset: 30 error pairs + 30 clean pass-through pairs. The pass-through cases are critical to prevent the model from becoming trigger-happy.

### Router

The router is a binary classifier, so it needs fewer examples to shift behavior.

| Dataset size | Expected outcome |
|---|---|
| 30-50 sentences | Enough to shift the decision boundary for your specific error distribution. |
| 100-200 sentences | Good coverage. Balanced 50/50 acceptable/unacceptable. |
| 500+ sentences | Overkill for a 13.5M classifier unless your domain is very specialized. |

## Collecting Training Data

### From history

Your pipeline already saves raw transcription and final output in `pipeline_stages`. Extract pairs where the Grammar (neural) stage changed text:

```python
import json

with open("corrector_test_cases.json") as f:
    cases = json.load(f)

pairs = []
for case in cases:
    stages = case.get("pipeline_stages", [])
    raw = next((s["text"] for s in stages if s["name"] == "Raw transcription"), None)
    grammar = next((s for s in stages if "Grammar" in s["name"]), None)
    if raw and grammar and grammar.get("changed"):
        pairs.append({"input": raw, "target": grammar["text"]})
    elif raw and grammar and not grammar.get("changed"):
        pairs.append({"input": raw, "target": raw})  # clean pass-through
```

### Manual correction

The most valuable data comes from cases where the current pipeline gets it wrong. When you notice a bad correction:

1. Note the raw transcription (before grammar stage)
2. Write the correct output by hand
3. Add both to the training set

Focus on failure modes you actually encounter, not hypothetical edge cases.

### From existing test suites

The scripts already contain categorized test cases:

- `scripts/test_fluency_correction.py` - 34 cases
- `scripts/test_cola_router.py` - 24 cases
- `scripts/test_full_pipeline.py` - 192 cases
- `scripts/test_pipeline_v3.py` - 270+ cases

These can be converted to training format directly.

## Fine-Tuning the Corrector

### Setup

```bash
pip install transformers datasets torch accelerate
```

### Training script

```python
#!/usr/bin/env python3
"""Fine-tune t5-efficient-tiny-grammar-correction on ASR error pairs."""

import json
from pathlib import Path
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

MODEL = "visheratin/t5-efficient-tiny-grammar-correction"
DATA_PATH = "training_data/corrector_pairs.jsonl"
OUTPUT_DIR = "finetuned/t5-grammar-corrector"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = T5ForConditionalGeneration.from_pretrained(MODEL)

# Load data
with open(DATA_PATH) as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)

# 90/10 train/eval split
split = dataset.train_test_split(test_size=0.1, seed=42)

def tokenize(example):
    inputs = tokenizer(
        example["input"],
        max_length=128,
        truncation=True,
        padding=False,
    )
    targets = tokenizer(
        example["target"],
        max_length=128,
        truncation=True,
        padding=False,
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

train_ds = split["train"].map(tokenize, remove_columns=["input", "target"])
eval_ds = split["test"].map(tokenize, remove_columns=["input", "target"])

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,          # small dataset, more epochs
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-4,          # T5 standard LR
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=10,
    fp16=True,                   # use bf16=True on Ampere+
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
```

### Training time estimates

| GPU | 100 pairs, 5 epochs | 500 pairs, 5 epochs |
|-----|---------------------|---------------------|
| RTX 3060/4060 | ~2 minutes | ~8 minutes |
| T4 (Colab free) | ~3 minutes | ~12 minutes |
| CPU only | ~15 minutes | ~60 minutes |

The model is 15M params. Training is fast regardless of hardware.

## Fine-Tuning the Router

### Training script

```python
#!/usr/bin/env python3
"""Fine-tune electra-small-discriminator-CoLA on ASR acceptability data."""

import json
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import numpy as np

MODEL = "pszemraj/electra-small-discriminator-CoLA"
DATA_PATH = "training_data/router_labels.jsonl"
OUTPUT_DIR = "finetuned/cola-router"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

with open(DATA_PATH) as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)
split = dataset.train_test_split(test_size=0.1, seed=42)

def tokenize(example):
    return tokenizer(example["text"], max_length=128, truncation=True, padding=False)

train_ds = split["train"].map(tokenize, remove_columns=["text"])
eval_ds = split["test"].map(tokenize, remove_columns=["text"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,         # classifier converges fast
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,          # BERT-style LR for classifiers
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    logging_steps=10,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
```

## Exporting to ONNX

After fine-tuning, export to ONNX INT8 using the existing scripts as reference.

### Corrector

```bash
# Use the existing export script pattern but point at finetuned model
python3 scripts/download_t5_grammar_onnx.py  # modify to load from local path
```

Or manually:

```python
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

model = ORTModelForSeq2SeqLM.from_pretrained("finetuned/t5-grammar-corrector", export=True)
model.save_pretrained("finetuned/onnx-export")

# Quantize encoder
enc_quantizer = ORTQuantizer.from_pretrained("finetuned/onnx-export", file_name="encoder_model.onnx")
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
enc_quantizer.quantize(save_dir="finetuned/onnx-quantized", quantization_config=qconfig)

# Quantize decoder
dec_quantizer = ORTQuantizer.from_pretrained("finetuned/onnx-export", file_name="decoder_model.onnx")
dec_quantizer.quantize(save_dir="finetuned/onnx-quantized", quantization_config=qconfig)
```

### Router

```bash
python3 scripts/export_cola_onnx.py --output-dir src-tauri/data/grammar/
```

Modify the script to load from `finetuned/cola-router` instead of the HuggingFace hub ID.

## Validation Before Deploying

Run the existing eval pipeline against the fine-tuned model before replacing the bundled ONNX files:

```bash
# Test corrector quality
python3 scripts/test_fluency_correction.py  # point at finetuned model

# Test full pipeline routing
just eval

# Compare before/after on your specific failure cases
```

Copy the quantized ONNX files into `src-tauri/data/grammar/` to replace the bundled models. The Rust code loads them by filename, no code changes needed.

## Risks

- **Catastrophic forgetting**: with fewer than 50 examples, the model may lose general correction ability while learning your specific fixes. Mitigate by including clean pass-through cases and keeping epochs low (3-5).
- **Overfitting**: a 15M param model can memorize 100 examples. Watch eval loss diverging from train loss. Stop early if it does.
- **Distribution shift on router**: the CoLA model was calibrated with threshold=0.75. Fine-tuning changes the output distribution. Re-run the threshold sweep after fine-tuning.
- **Tokenizer mismatch**: do not modify the tokenizer. The Rust inference code uses the same tokenizer.json. If you change the tokenizer vocabulary, the ONNX model and Rust code will produce garbage.

## Recommended Starting Point

1. Collect 30 error pairs where the current pipeline produces wrong output
2. Collect 30 clean sentences the pipeline handles correctly (pass-through cases)
3. Fine-tune the corrector for 5 epochs
4. Export to ONNX INT8
5. Run `just eval` to compare against baseline
6. If the router is the bottleneck (correct text being sent to a corrector that mangles it), fine-tune the router too with 50 labeled sentences
