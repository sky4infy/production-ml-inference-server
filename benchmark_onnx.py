"""
Fair benchmark: PyTorch FP32 vs ONNX INT8, both without batcher overhead.
Sequential requests, 100 measured after 10 warmup.

Run with server up:
    uvicorn app.main:app --reload
    python benchmark_onnx.py
"""

import asyncio
import statistics
import time
import httpx

BASE_URL   = "http://127.0.0.1:8000"
WARMUP     = 10
BENCH      = 100

TEXTS = [
    "I absolutely loved this product, exceeded all my expectations.",
    "Terrible experience, would not recommend to anyone.",
    "It was okay, nothing special but got the job done.",
    "Best purchase I have made this year, absolutely wonderful.",
    "Very disappointing, quality much worse than advertised.",
]


async def measure(client: httpx.AsyncClient, endpoint: str, n: int) -> list[float]:
    latencies = []
    for i in range(n):
        t = TEXTS[i % len(TEXTS)]
        start = time.perf_counter()
        r = await client.post(f"{BASE_URL}{endpoint}", json={"text": t})
        r.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def stats(label: str, endpoint: str, lats: list[float]):
    s = sorted(lats)
    p = lambda pct: s[int(len(s) * pct)]
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"  endpoint: {endpoint}")
    print(f"{'─'*50}")
    print(f"  Mean:   {statistics.mean(lats):.1f} ms")
    print(f"  Median: {statistics.median(lats):.1f} ms")
    print(f"  p95:    {p(0.95):.1f} ms")
    print(f"  p99:    {p(0.99):.1f} ms")
    print(f"  Min:    {min(lats):.1f} ms   Max: {max(lats):.1f} ms")
    return statistics.median(lats), p(0.99)


async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{BASE_URL}/health")
        h = r.json()
        print(f"Server up. Backend: {h.get('backend', 'unknown')}")

        print(f"\nWarmup ({WARMUP} requests each)...")
        await measure(client, "/predict_unbatched", WARMUP)
        await measure(client, "/predict_onnx_unbatched", WARMUP)
        print("  Done.")

        print(f"\nBenchmarking ({BENCH} sequential requests each)...")
        pytorch_lats = await measure(client, "/predict_unbatched", BENCH)
        onnx_lats    = await measure(client, "/predict_onnx_unbatched", BENCH)

    pt_med,  pt_p99  = stats("PyTorch FP32 — baseline", "/predict_unbatched",       pytorch_lats)
    onnx_med, onnx_p99 = stats("ONNX INT8  — quantized", "/predict_onnx_unbatched", onnx_lats)

    speedup_med = pt_med  / onnx_med
    speedup_p99 = pt_p99  / onnx_p99

    print(f"\n{'═'*50}")
    print("  SUMMARY  (fair comparison — no batcher on either side)")
    print(f"{'═'*50}")
    print(f"  Median:  {pt_med:.0f}ms → {onnx_med:.0f}ms  ({speedup_med:.2f}x)")
    print(f"  p99:     {pt_p99:.0f}ms → {onnx_p99:.0f}ms  ({speedup_p99:.2f}x)")

    if speedup_med >= 1.0:
        print(f"""
  RESUME BULLET:
  "Exported DistilBERT to ONNX and applied INT8 dynamic quantization,
   reducing median inference latency from {pt_med:.0f}ms to {onnx_med:.0f}ms
   ({speedup_med:.1f}x faster); p99 improved from {pt_p99:.0f}ms to {onnx_p99:.0f}ms"
""")
    else:
        print(f"""
  NOTE: ONNX INT8 is {1/speedup_med:.1f}x SLOWER than PyTorch on this CPU.
  This is a known result with dynamic quantization on CPUs without AVX-512
  VNNI support (most consumer Intel i5/i7 without 11th gen+).
  Dynamic quantization dequantizes weights back to FP32 before multiply —
  it only wins when memory bandwidth is the bottleneck, not compute.

  Your honest resume bullet is about the ARCHITECTURE, not the number:
  "Implemented ONNX export pipeline and INT8 dynamic quantization;
   benchmarked on CPU (Intel i5) and documented that dynamic quantization
   trades memory footprint for latency on CPU without VNNI support —
   identified static quantization or GPU deployment as next steps"

  This answer in an interview is MORE impressive than a fake speedup.
""")
    print(f"{'═'*50}\n")


if __name__ == "__main__":
    asyncio.run(main())