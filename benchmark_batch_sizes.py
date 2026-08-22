"""
benchmark_batch_sizes.py

Compares PyTorch, ONNX INT8, and ONNX FP32 predict_batch() speed and
correctness across batch sizes, in-process (no HTTP, no batcher).

--- What was wrong with the previous version ---

1. It reported only TOTAL batch latency, so the key fact was invisible:
   per-item cost on CPU is flat (17.8 / 16.4 / 17.0 / 15.8 ms at batch
   1/2/4/8). Printed as totals, "126ms at batch 8" looks alarming; printed
   per item, it correctly reads as "batching buys ~1.1x on CPU, not 8x".

2. It had no FP32 ONNX arm, so it could not answer the actual question —
   is INT8 quantization helping? (It is: ~2.5x over FP32.)

3. It did not report sequence length. The two backends were padding
   differently (ONNX to a fixed 128, PyTorch dynamically to ~13), so the
   comparison was measuring ~10x different amounts of work and attributing
   the gap to the runtime. Both pad dynamically now; this prints the token
   count so any future divergence is visible rather than silent.

4. Its notes asserted a "DynamicQuantizeMatMul dimension-mismatch bug" as
   the confirmed cause of the batch>=2 failure. That was wrong — the FP32
   graph failed at the same node. See export_onnx.py for the real cause.

Run from the project root (same folder as app/):
    python benchmark_batch_sizes.py
"""

import statistics
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ort.set_default_logger_severity(4)   # keep ORT's own error spam out of the table

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
FP32_PATH  = "models/model_fp32.onnx"

SAMPLE_TEXTS = [
    "I absolutely loved this movie, best film all year.",
    "This was a complete waste of my time and money.",
    "The acting was fine but the plot made no sense.",
    "An absolute masterpiece, I was moved to tears.",
    "Terrible. I want my two hours back.",
    "Pretty average, nothing special but not bad either.",
    "One of the best experiences I've had in a long time.",
    "I regret watching this, deeply disappointing.",
]

BATCH_SIZES = [1, 2, 4, 8, 16]

# 15, not 5. This is a hybrid CPU (i5-12450H: 4 performance cores + 4
# efficiency cores). A 2-thread inference job can be scheduled onto 2 P-cores,
# 2 E-cores, or one of each, and the E-cores are roughly half the speed — so
# identical work measured ~16ms, ~32ms and ~36ms across runs during
# development. Five repeats was not enough to stabilise the median, which is
# how the same PyTorch batch-1 call was recorded as both 17.8ms and 31.7ms in
# different phases. More repeats plus a printed spread makes the error bar
# visible instead of letting scheduler luck look like a real regression.
REPEATS = 15


def bench_fn(fn, texts, repeats=REPEATS):
    """Returns (median_ms, spread_pct, error). error is None on success.

    spread_pct is (max-min)/median — on this hybrid CPU anything above ~30%
    means core placement, not the code under test, dominated the measurement.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        try:
            fn(texts)
        except Exception as exc:
            return None, None, exc
        times.append((time.perf_counter() - start) * 1000)
    med = statistics.median(times)
    return med, 100 * (max(times) - min(times)) / med, None


def make_fp32_predict():
    """
    ONNX FP32 batch predict, built here rather than in app/model_onnx.py.

    The app only ever serves INT8, so keeping the FP32 session local to the
    benchmark avoids adding a second code path to production for the sake of
    a measurement. Mirrors model_onnx.py's session options and dynamic
    padding exactly so the only variable is weight precision.
    """
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(FP32_PATH, opts, providers=["CPUExecutionProvider"])

    def predict(texts):
        inp = tok(texts, return_tensors="np", padding=True,
                  max_length=128, truncation=True)
        return sess.run(["logits"], {"input_ids": inp["input_ids"],
                                     "attention_mask": inp["attention_mask"]})[0]
    return predict


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("=" * 92)
    print("Loading PyTorch backend...")
    from app.model import predict_batch as pytorch_predict_batch
    pytorch_predict_batch(["warmup"])

    print("Loading ONNX INT8 backend...")
    onnx_int8 = None
    try:
        from app.model_onnx import predict_batch as onnx_int8_batch
        onnx_int8_batch(["warmup"])
        onnx_int8 = onnx_int8_batch
    except Exception as exc:
        print(f"  ONNX INT8 unavailable: {exc}")

    print("Loading ONNX FP32 backend (for the quantization comparison)...")
    onnx_fp32 = None
    try:
        onnx_fp32 = make_fp32_predict()
        onnx_fp32(["warmup"])
    except Exception as exc:
        print(f"  ONNX FP32 unavailable: {exc}")

    print("=" * 92)
    print("Total = whole batch. Per-item = total / batch_size (this is the number")
    print("that tells you whether batching is actually buying you anything).")
    print("=" * 92)
    hdr = (f"{'Batch':<7}{'tok':<6}"
           f"{'PyTorch':<9}{'/item':<8}"
           f"{'INT8':<9}{'/item':<8}{'±%':<6}"
           f"{'FP32':<9}{'/item':<8}"
           f"{'INT8 vs PT':<12}{'INT8 vs FP32'}")
    print(hdr)
    print("-" * 92)

    rows = []
    for size in BATCH_SIZES:
        texts = (SAMPLE_TEXTS * ((size // len(SAMPLE_TEXTS)) + 1))[:size]
        # Actual padded length this batch will run at — pads to the longest
        # member, so it grows slightly with batch size on mixed-length input.
        n_tok = tok(texts, padding=True, max_length=128,
                    truncation=True, return_tensors="np")["input_ids"].shape[1]

        pt_ms, _, pt_err        = bench_fn(pytorch_predict_batch, texts)
        i8_ms, i8_spread, i8_err = (bench_fn(onnx_int8, texts) if onnx_int8
                                    else (None, None, "n/a"))
        f32_ms, _, _            = (bench_fn(onnx_fp32, texts) if onnx_fp32
                                   else (None, None, "n/a"))

        def cell(ms):
            return f"{ms:<9.1f}" if ms is not None else f"{'FAIL':<9}"

        def per_item(ms):
            return f"{ms/size:<8.1f}" if ms is not None else f"{'-':<8}"

        spread  = f"{i8_spread:<6.0f}" if i8_spread is not None else f"{'-':<6}"
        vs_pt   = f"{pt_ms/i8_ms:<12.2f}" if (pt_ms and i8_ms) else f"{'-':<12}"
        vs_fp32 = f"{f32_ms/i8_ms:.2f}x"  if (f32_ms and i8_ms) else "-"

        print(f"{size:<7}{n_tok:<6}"
              f"{cell(pt_ms)}{per_item(pt_ms)}"
              f"{cell(i8_ms)}{per_item(i8_ms)}{spread}"
              f"{cell(f32_ms)}{per_item(f32_ms)}"
              f"{vs_pt}{vs_fp32}")

        if i8_err not in (None, "n/a"):
            print(f"       ONNX INT8 error: {str(i8_err).splitlines()[0][:78]}")
        if pt_err is not None:
            print(f"       PyTorch error:   {str(pt_err).splitlines()[0][:78]}")
        rows.append((size, pt_ms, i8_ms, f32_ms))

    print("=" * 92)

    # ── Correctness: batched output must equal unbatched output ──────────────
    # A graph with a specialized batch axis can also produce silently WRONG
    # rows rather than raising, so agreement matters as much as not crashing.
    if onnx_int8:
        singles = [onnx_int8([t])[0] for t in SAMPLE_TEXTS]
        batched = onnx_int8(SAMPLE_TEXTS)
        mismatch = [(i, s, b) for i, (s, b) in enumerate(zip(singles, batched))
                    if s["label"] != b["label"]]
        drift = max(abs(s["score"] - b["score"]) for s, b in zip(singles, batched))
        print(f"Correctness: batch-of-8 vs 8 singles -> "
              f"{'ALL LABELS MATCH' if not mismatch else f'MISMATCH {mismatch}'}"
              f", max score drift {drift:.4f}")

    # ── Interpretation, computed rather than asserted ────────────────────────
    print("\nWhat these numbers mean:")
    b1 = next((r for r in rows if r[0] == 1), None)

    # Find the batch size with the LOWEST per-item cost rather than assuming
    # the largest batch is best. On CPU it often is not: per-item cost is flat
    # up to the point where the batch stops fitting in cache / starts fighting
    # the 2-thread limit, then it gets worse. Reporting batch-1 vs largest-batch
    # as a "gain" printed 0.56x on this box — a regression labelled as a gain.
    per_item = [(r[0], r[2] / r[0]) for r in rows if r[2]]
    if b1 and b1[2] and per_item:
        best_size, best_per = min(per_item, key=lambda x: x[1])
        worst_size, worst_per = max(per_item, key=lambda x: x[1])
        per1 = b1[2]
        print(f"- Best per-item cost (ONNX INT8): {best_per:.1f} ms/item at "
              f"batch {best_size}, vs {per1:.1f} ms/item at batch 1 "
              f"= {per1/best_per:.2f}x")
        print(f"- Worst per-item cost: {worst_per:.1f} ms/item at batch {worst_size}. "
              f"Batching past the optimum COSTS throughput on CPU.")
        print("  On CPU the best case is expected to be small. A CPU GEMM is")
        print("  already compute-bound at batch 1, so there is no kernel-launch")
        print("  overhead to amortize the way there is on a GPU. Batching here")
        print("  amortizes tokenizer and graph overhead only, and past the")
        print("  optimum the larger GEMM spills cache and loses more than that.")
        print(f"  -> set MAX_BATCH_SIZE in app/batcher.py near {best_size}, not higher.")
    if b1 and b1[1] and b1[2]:
        print(f"- ONNX INT8 vs PyTorch at batch 1: {b1[1]:.1f} -> {b1[2]:.1f} ms "
              f"({b1[1]/b1[2]:.2f}x)")
    if b1 and b1[3] and b1[2]:
        print(f"- INT8 vs FP32 (same graph, same padding): {b1[3]:.1f} -> {b1[2]:.1f} ms "
              f"({b1[3]/b1[2]:.2f}x) — this isolates the quantization win")
    print("- 'tok' is the padded sequence length actually fed to the model.")
    print("  If the backends ever show different token counts for the same")
    print("  batch, the comparison is invalid — that was the phase 5 bug.")
    print("- '±%' is (max-min)/median over the repeats. This is a hybrid CPU")
    print("  (i5-12450H, 4 P-cores + 4 E-cores); above ~30% means OS core")
    print("  placement moved the number more than the code did, so treat small")
    print("  differences between adjacent rows as noise.")


if __name__ == "__main__":
    main()
