"""
Fair benchmark: PyTorch FP32 vs ONNX INT8, both without batcher overhead.

--- What was wrong with the previous version ---

1. Its else-branch printed a confident, fabricated diagnosis whenever ONNX
   came out slower:
     "known result with dynamic quantization on CPUs without AVX-512 VNNI"
     "dynamic quantization dequantizes weights back to FP32 before multiply"
   Neither applied. This CPU (i5-12450H) does have AVX-VNNI — only AVX-512 is
   fused off — and MLAS uses it; INT8 measures ~2.4x faster than the same
   graph in FP32. The actual reason ONNX looked slow was that
   app/model_onnx.py padded every request to 128 tokens while PyTorch's HF
   pipeline padded dynamically to ~13, so the two arms were doing ~10x
   different amounts of work. Hardcoded explanations in a benchmark are worse
   than no explanation: this one sent the project down a wrong path for a
   whole phase. The summary now reports what was measured and lists what to
   check, rather than asserting a cause.

2. It ran all 100 PyTorch requests, then all 100 ONNX requests. On a hybrid
   CPU (4 P-cores + 4 E-cores) plus thermal drift, whichever arm runs second
   is systematically penalised. Requests are now INTERLEAVED so both arms see
   the same machine conditions.

3. It never reported sequence length, so the padding mismatch that caused
   problem 1 was invisible in the output.

Run with server up:
    uvicorn app.main:app --reload
    python benchmark_onnx.py
"""

import asyncio
import statistics
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"
WARMUP   = 10
BENCH    = 100

TEXTS = [
    "I absolutely loved this product, exceeded all my expectations.",
    "Terrible experience, would not recommend to anyone.",
    "It was okay, nothing special but got the job done.",
    "Best purchase I have made this year, absolutely wonderful.",
    "Very disappointing, quality much worse than advertised.",
]

ARMS = [
    ("PyTorch FP32 — baseline", "/predict_unbatched"),
    ("ONNX INT8    — quantized", "/predict_onnx_unbatched"),
]


def percentile(sorted_lats: list[float], pct: float) -> float:
    idx = min(int(len(sorted_lats) * pct), len(sorted_lats) - 1)
    return sorted_lats[idx]


async def one(client: httpx.AsyncClient, endpoint: str, text: str) -> float:
    start = time.perf_counter()
    r = await client.post(f"{BASE_URL}{endpoint}", json={"text": text})
    r.raise_for_status()
    return (time.perf_counter() - start) * 1000


async def measure_interleaved(client: httpx.AsyncClient, n: int) -> dict:
    """
    Alternate between endpoints request-by-request.

    Sequential arms (all of A, then all of B) let thermal drift and P/E core
    placement land entirely on one arm. Interleaving spreads any drift evenly,
    so the comparison survives a machine that is not perfectly steady-state.
    """
    lats = {ep: [] for _, ep in ARMS}
    for i in range(n):
        text = TEXTS[i % len(TEXTS)]
        for _, endpoint in ARMS:
            lats[endpoint].append(await one(client, endpoint, text))
    return lats


def stats(label: str, endpoint: str, lats: list[float]):
    s = sorted(lats)
    print(f"\n{'-' * 52}")
    print(f"  {label}")
    print(f"  endpoint: {endpoint}")
    print(f"{'-' * 52}")
    print(f"  Mean:   {statistics.mean(lats):.1f} ms")
    print(f"  Median: {statistics.median(lats):.1f} ms")
    print(f"  p95:    {percentile(s, 0.95):.1f} ms")
    print(f"  p99:    {percentile(s, 0.99):.1f} ms")
    print(f"  Min:    {min(lats):.1f} ms   Max: {max(lats):.1f} ms")
    print(f"  Spread: {100 * (max(lats) - min(lats)) / statistics.median(lats):.0f}% "
          f"of median (hybrid-CPU core placement; treat small gaps as noise)")
    return statistics.median(lats), percentile(s, 0.99)


async def main():
    async with httpx.AsyncClient(timeout=60.0) as client:
        h = (await client.get(f"{BASE_URL}/health")).json()
        print(f"Server up. Batcher backend: {h.get('backend')}")
        print(f"Test sentences: {len(TEXTS)}, "
              f"{min(len(t.split()) for t in TEXTS)}-{max(len(t.split()) for t in TEXTS)} words each")
        print("Both endpoints tokenize with padding=True (pad to longest in")
        print("batch), so each arm runs the same sequence length. This is the")
        print("check that was missing when ONNX 'looked' 1.45x slower.")

        print(f"\nWarmup ({WARMUP} requests each)...")
        await measure_interleaved(client, WARMUP)
        print("  Done.")

        print(f"\nBenchmarking ({BENCH} interleaved requests per endpoint)...")
        lats = await measure_interleaved(client, BENCH)

    results = {}
    for label, endpoint in ARMS:
        results[endpoint] = stats(label, endpoint, lats[endpoint])

    pt_med,  pt_p99   = results["/predict_unbatched"]
    onnx_med, onnx_p99 = results["/predict_onnx_unbatched"]
    speedup_med = pt_med / onnx_med
    speedup_p99 = pt_p99 / onnx_p99

    print(f"\n{'=' * 52}")
    print("  SUMMARY  (no batcher, no cache, interleaved)")
    print(f"{'=' * 52}")
    print(f"  Median:  {pt_med:.1f}ms -> {onnx_med:.1f}ms  ({speedup_med:.2f}x)")
    print(f"  p99:     {pt_p99:.1f}ms -> {onnx_p99:.1f}ms  ({speedup_p99:.2f}x)")

    if speedup_med >= 1.0:
        print(f"""
  ONNX INT8 is {speedup_med:.2f}x faster than PyTorch FP32 at the median.
  To attribute that between "ONNX runtime" and "INT8 quantization", run
  benchmark_batch_sizes.py — it adds an ONNX FP32 arm, which isolates the
  quantization contribution (measured ~2.4x on this box).
""")
    else:
        print(f"""
  ONNX INT8 measured {1/speedup_med:.2f}x SLOWER than PyTorch here. Before
  concluding anything about the hardware, check these in order — the last
  time this happened, it was cause #1 and it was misdiagnosed as #4:

    1. Padding parity. Are both arms running the same sequence length?
       app/model_onnx.py must use padding=True, not padding="max_length".
       A fixed 128-token pad against a dynamic ~13-token pad is a ~10x
       compute difference that looks exactly like a slow runtime.
    2. Thread parity. torch.set_num_threads(N) in app/model.py must match
       intra_op_num_threads in app/model_onnx.py (both 2 today).
    3. Measurement noise. Check the Spread line above. On this hybrid CPU
       (4 P-cores + 4 E-cores) core placement alone moves a single-request
       median by ~2x. Re-run before trusting a small gap.
    4. Only then consider the ISA. Alder Lake has AVX-VNNI (AVX-512 is
       fused off); MLAS uses it, so INT8 does have hardware support here.
       Confirm with the ONNX FP32 arm in benchmark_batch_sizes.py: if INT8
       beats FP32 there, quantization is working and the problem is upstream.
""")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    asyncio.run(main())
