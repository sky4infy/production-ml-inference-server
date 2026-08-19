"""
benchmark_batch_sizes.py

Compares PyTorch vs ONNX INT8 predict_batch() speed AND correctness across
batch sizes 1, 2, 4, 8. This is different from benchmark_onnx.py (which only
ever sends batch_size=1 through the live server). This script calls the
model layer directly, in-process, so it can push real batches through and
show exactly where/if ONNX breaks.

Run from the project root (same folder as app/):
    python benchmark_batch_sizes.py
"""

import time
import statistics

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

BATCH_SIZES = [1, 2, 4, 8]
REPEATS = 5  # runs per batch size, to get a stable median


def bench_fn(fn, texts, repeats=REPEATS):
    """Runs fn(texts) `repeats` times, returns (median_ms, error) tuple.
    error is None on success, or the exception on failure."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        try:
            fn(texts)
        except Exception as exc:
            return None, exc
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), None


def main():
    print("=" * 60)
    print("Loading PyTorch backend...")
    from app.model import predict_batch as pytorch_predict_batch
    pytorch_predict_batch(["warmup"])  # trigger lazy load, exclude from timing

    print("Loading ONNX backend...")
    onnx_available = True
    try:
        from app.model_onnx import predict_batch as onnx_predict_batch
        onnx_predict_batch(["warmup"])
    except Exception as exc:
        onnx_available = False
        print(f"  ONNX backend unavailable: {exc}")

    print("=" * 60)
    print(f"{'Batch':<8}{'PyTorch (ms)':<18}{'ONNX (ms)':<18}{'ONNX status'}")
    print("-" * 60)

    for size in BATCH_SIZES:
        texts = (SAMPLE_TEXTS * ((size // len(SAMPLE_TEXTS)) + 1))[:size]

        pt_ms, pt_err = bench_fn(pytorch_predict_batch, texts)
        pt_str = f"{pt_ms:.1f}" if pt_ms is not None else f"ERROR: {pt_err}"

        if onnx_available:
            onnx_ms, onnx_err = bench_fn(onnx_predict_batch, texts)
            if onnx_ms is not None:
                onnx_str = f"{onnx_ms:.1f}"
                status = "OK"
            else:
                onnx_str = "FAILED"
                status = str(onnx_err).splitlines()[0][:60]
        else:
            onnx_str = "N/A"
            status = "backend not loaded"

        print(f"{size:<8}{pt_str:<18}{onnx_str:<18}{status}")

    print("=" * 60)
    print("Notes:")
    print("- Times are median of", REPEATS, "runs, model already warmed up.")
    print("- If ONNX FAILED at batch >= 2, this confirms the")
    print("  DynamicQuantizeMatMul dimension-mismatch bug for batched")
    print("  inference on this onnxruntime/model export combination.")
    print("- PyTorch numbers here are directly comparable to your")
    print("  earlier phase 2/3 findings (18ms single-request baseline).")


if __name__ == "__main__":
    main()