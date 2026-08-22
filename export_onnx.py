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
def model_size_mb(path: Path) -> float:
    """
    On-disk size of an ONNX model INCLUDING external weight files.

    Why this is not just path.stat(): torch.onnx.export writes DistilBERT's
    weights to a sidecar `model_fp32.onnx.data` (255 MB) and leaves only the
    graph structure (0.7 MB) in the .onnx itself. The old code measured just
    the .onnx, so it compared 0.7 MB against the 64.7 MB single-file INT8
    model and printed "-8578% smaller". Counting the sidecar gives the real
    figure: 256.1 MB -> 64.7 MB, i.e. 75% smaller, which is the ~4x you
    expect from float32 -> int8 weights.
    """
    total = path.stat().st_size if path.exists() else 0
    total += sum(p.stat().st_size for p in path.parent.glob(path.name + ".data"))
    total += sum(p.stat().st_size for p in path.parent.glob(path.stem + "*.weight"))
    return total / 1024 / 1024


def batch_axis_is_dynamic(path: Path) -> bool:
    """
    True if the graph keeps batch as a symbolic dim on its logits output.

    Why this check exists: a graph traced from a batch-1 example silently
    specializes the batch axis to the literal 1 (see the dummy input below).
    Such a graph loads fine, passes a batch-1 smoke test, and then throws
    "MatMul dimension mismatch" on every batch>=2 call. The previous version
    of this script gated re-export on `FP32_PATH.exists()` alone, so once a
    bad graph landed on disk it was never regenerated and the bug survived
    into every later phase.
    """
    try:
        out = onnx.load(str(path), load_external_data=False).graph.output[0]
    except Exception:
        return False
    # dim_param is non-empty for symbolic dims ('batch_size'); a baked-in
    # constant leaves dim_param empty and sets dim_value instead.
    return out.type.tensor_type.shape.dim[0].dim_param != ""


reuse_fp32 = FP32_PATH.exists() and batch_axis_is_dynamic(FP32_PATH)

if FP32_PATH.exists() and not reuse_fp32:
    print(f"\nFound {FP32_PATH}, but its batch axis is hardcoded to 1 "
          f"(batch>=2 would crash) — re-exporting.")

if reuse_fp32:
    print(f"\nFP32 model already at {FP32_PATH} (batch axis dynamic), skipping export.")
    fp32_mb = model_size_mb(FP32_PATH)
else:
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    # TWO rows, not one — this is load-bearing, not cosmetic.
    #
    # torch.onnx.export defaults to dynamo=True in torch 2.9+, which routes
    # through torch.export. torch.export SPECIALIZES any dimension whose
    # example value is 1, because size-1 dims carry broadcasting semantics.
    # With a single-string dummy, dim 0 was 1, so the `batch_size` entry in
    # dynamic_axes below was silently ignored: the literal 1 got baked into
    # the attention out_lin reshapes as Concat([1], Shape(mask)[1:2], [-1]).
    # At batch=2 that reshapes (2,128,12,64) to (1,128,1536) and the next
    # MatMul against a (768,768) weight fails with a dimension mismatch.
    # sequence_length escaped the same fate only because 128 != 1.
    dummy = tokenizer(
        ["Dummy input for ONNX tracing.",
         "Second row so the batch axis is not size 1 and stays dynamic."],
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
            # 18, not 14. Requesting 14 was a silent no-op: the dynamo
            # exporter emits opset 18 natively, then onnx's version converter
            # fails to downgrade with "No Previous Version of
            # LayerNormalization exists" and the graph stays at 18 anyway.
            # Stating 18 matches what actually ships and drops the error spam.
            opset_version=18,
            do_constant_folding=True,
        )

    fp32_mb = model_size_mb(FP32_PATH)
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

int8_mb = model_size_mb(INT8_PATH)
fp32_mb = model_size_mb(FP32_PATH)
print(f"  Done. {fp32_mb:.1f} MB → {int8_mb:.1f} MB  "
      f"({100*(1-int8_mb/fp32_mb):.0f}% smaller, {fp32_mb/int8_mb:.1f}x)")

# ── 4. Correctness check ─────────────────────────────────────────────────────
# The old check ran ONE sentence at batch 1 and printed "OK". That is exactly
# the blind spot that let a batch-broken graph ship: batch 1 is the only size
# a batch-specialized graph can run. This version exercises batch>=2 and
# exits non-zero on failure, so a bad artifact can never pass silently again.
print("\nRunning correctness check on INT8 model...")
import numpy as np

if 'tokenizer' not in dir():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
sess = ort.InferenceSession(str(INT8_PATH), opts,
                            providers=["CPUExecutionProvider"])

ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
failures = []

# 4a. declared output shape — batch must still be symbolic after quantization
out_dims = [(d.dim_param or d.dim_value)
            for d in onnx.load(str(INT8_PATH)).graph.output[0].type.tensor_type.shape.dim]
print(f"  logits output shape: {out_dims}")
if not isinstance(out_dims[0], str):
    failures.append(f"batch axis is baked to {out_dims[0]} instead of symbolic")

# 4b. real predictions at batch 1 and batch 4, using the same dynamic padding
#     app/model_onnx.py uses at serving time
CASES = [
    ("I absolutely loved this!",             "POSITIVE"),
    ("This was a complete waste of money.",  "NEGATIVE"),
    ("An absolute masterpiece, wonderful.",  "POSITIVE"),
    ("Terrible, I want my money back.",      "NEGATIVE"),
]

for size in (1, 4):
    texts, expected = zip(*CASES[:size])
    inp = tokenizer(list(texts), return_tensors="np",
                    padding=True, max_length=128, truncation=True)
    try:
        logits = sess.run(["logits"], {"input_ids": inp["input_ids"],
                                       "attention_mask": inp["attention_mask"]})[0]
    except Exception as exc:
        failures.append(f"batch={size} raised {type(exc).__name__}: "
                        f"{str(exc).splitlines()[0][:110]}")
        print(f"  batch={size}: FAILED")
        continue

    if logits.shape[0] != size:
        failures.append(f"batch={size} returned {logits.shape[0]} rows")

    got = [ID2LABEL[int(np.argmax(row))] for row in logits]
    wrong = [(t, g, e) for t, g, e in zip(texts, got, expected) if g != e]
    status = "OK" if not wrong else f"WRONG LABELS: {wrong}"
    if wrong:
        failures.append(f"batch={size} {status}")
    print(f"  batch={size}: shape={logits.shape}  labels={got}  {status}")

if failures:
    print("\nEXPORT FAILED VALIDATION:")
    for f in failures:
        print(f"  - {f}")
    print("\nThe artifacts on disk are not safe to serve. Not proceeding.")
    sys.exit(1)

print("  All checks passed.")

print("""
Export complete. Files ready:
  models/model_fp32.onnx
  models/model_int8.onnx

Next:
  Terminal 1:  uvicorn app.main:app --reload
  Terminal 2:  python benchmark_onnx.py
""")