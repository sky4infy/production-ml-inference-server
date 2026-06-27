"""
Run this ONCE before starting the server.
Exports DistilBERT to ONNX FP32, then applies INT8 dynamic quantization.

Usage:
    python export_onnx.py

Outputs:
    models/model_fp32.onnx       (full precision ONNX)
    models/model_int8.onnx       (INT8 quantized ONNX) <- server uses this
"""

import sys
from pathlib import Path

# ── 0. Dependency check ──────────────────────────────────────────────────────
print("Checking dependencies...")
try:
    import onnx
    print(f"  onnx {onnx.__version__} OK")
except ImportError:
    print("ERROR: pip install onnx onnxscript")
    sys.exit(1)

try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print(f"  onnxruntime {ort.__version__} OK")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"  torch {torch.__version__} OK")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# ── 1. Paths ─────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

FP32_PATH    = MODELS_DIR / "model_fp32.onnx"
CLEARED_PATH = MODELS_DIR / "model_fp32_cleared.onnx"   # intermediate
INT8_PATH    = MODELS_DIR / "model_int8.onnx"
MODEL_NAME   = "distilbert-base-uncased-finetuned-sst-2-english"

# ── 2. Load model ────────────────────────────────────────────────────────────
if FP32_PATH.exists():
    print(f"\nFP32 model already at {FP32_PATH}, skipping export.")
    fp32_mb = FP32_PATH.stat().st_size / 1024 / 1024
else:
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    dummy = tokenizer(
        "Dummy input for ONNX tracing.",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    print(f"Exporting FP32 → {FP32_PATH}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(dummy["input_ids"], dummy["attention_mask"]),
            f=str(FP32_PATH),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits":         {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

    fp32_mb = FP32_PATH.stat().st_size / 1024 / 1024
    onnx.checker.check_model(onnx.load(str(FP32_PATH)))
    print(f"  Exported OK ({fp32_mb:.1f} MB), graph validated OK")

# ── 3. INT8 quantization ─────────────────────────────────────────────────────
# onnxruntime 1.27 runs its own shape inference during quantize_dynamic, but
# it conflicts with shape annotations already baked into the ONNX graph by
# torch.onnx.export. Fix: strip those intermediate shape annotations first
# (value_info), then let onnxruntime infer them fresh from scratch.
print(f"\nPreparing model for quantization (clearing shape annotations)...")
proto = onnx.load(str(FP32_PATH))
del proto.graph.value_info[:]          # remove conflicting intermediate shapes
onnx.save(proto, str(CLEARED_PATH))
print("  Done.")

print(f"Applying INT8 dynamic quantization → {INT8_PATH}")
quantize_dynamic(
    model_input=str(CLEARED_PATH),
    model_output=str(INT8_PATH),
    weight_type=QuantType.QInt8,
)
CLEARED_PATH.unlink(missing_ok=True)   # clean up temp file

int8_mb = INT8_PATH.stat().st_size / 1024 / 1024
fp32_mb = FP32_PATH.stat().st_size / 1024 / 1024
print(f"  Done. {fp32_mb:.1f} MB → {int8_mb:.1f} MB  "
      f"({100*(1-int8_mb/fp32_mb):.0f}% smaller)")

# ── 4. Correctness check ─────────────────────────────────────────────────────
print("\nRunning correctness check on INT8 model...")
import numpy as np

if 'tokenizer' not in dir():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
sess = ort.InferenceSession(str(INT8_PATH), opts,
                            providers=["CPUExecutionProvider"])

inp = tokenizer("I absolutely loved this!",
                return_tensors="np", padding="max_length",
                max_length=128, truncation=True)
logits = sess.run(["logits"],
                  {"input_ids": inp["input_ids"],
                   "attention_mask": inp["attention_mask"]})[0][0]
label = {0: "NEGATIVE", 1: "POSITIVE"}[int(np.argmax(logits))]
print(f"  Prediction: {label}  (expected POSITIVE) OK")

print("""
Export complete. Files ready:
  models/model_fp32.onnx
  models/model_int8.onnx

Next:
  Terminal 1:  uvicorn app.main:app --reload
  Terminal 2:  python benchmark_onnx.py
""")