"""
load_test_nocache.py

Same shape as load_test.py, but generates a UNIQUE text for every single
request instead of reusing a small fixed sample list. That guarantees
every request is a Redis cache miss, so /predict's numbers reflect
batching + model inference only — not a mix of cache hits and misses.

Run this alongside your normal load_test.py to get two clean comparisons:
  - load_test.py         -> batching + cache together (realistic traffic)
  - load_test_nocache.py -> batching alone (cache bypassed)

Usage:
    python load_test_nocache.py
"""

import asyncio
import time
import uuid

import httpx

BASE_URL    = "http://127.0.0.1:8000"
NUM_REQUESTS = 200
CONCURRENCY  = 50


def unique_text(i: int) -> str:
    # A random uuid per request guarantees a distinct SHA-256 cache key
    # every time, so app/cache.py will never find an existing entry.
    return f"Review number {i} — {uuid.uuid4()} — this product was okay I guess."


async def fire_request(client: httpx.AsyncClient, endpoint: str, text: str,
                        semaphore: asyncio.Semaphore) -> float:
    async with semaphore:
        start = time.perf_counter()
        response = await client.post(f"{BASE_URL}{endpoint}", json={"text": text})
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        return elapsed_ms


async def run_load_test(endpoint: str, label: str):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    texts = [unique_text(i) for i in range(NUM_REQUESTS)]

    async with httpx.AsyncClient(timeout=30.0) as client:
        start = time.perf_counter()
        tasks = [fire_request(client, endpoint, texts[i], semaphore)
                 for i in range(NUM_REQUESTS)]
        latencies = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]
    avg = sum(latencies) / n

    print(f"\n--- {label} ({endpoint}) — UNIQUE TEXTS, CACHE BYPASSED ---")
    print(f"Total requests:   {n}")
    print(f"Concurrency:      {CONCURRENCY}")
    print(f"Total time:       {total_time:.2f}s")
    print(f"Throughput:       {n / total_time:.1f} req/sec")
    print(f"Avg latency:      {avg:.1f}ms")
    print(f"p50 latency:      {p50:.1f}ms")
    print(f"p95 latency:      {p95:.1f}ms")
    print(f"p99 latency:      {p99:.1f}ms")


async def main():
    # /predict_unbatched: PyTorch, no batcher, no cache — true baseline.
    await run_load_test("/predict_unbatched", "WITHOUT batching")

    # /predict: PyTorch batcher + Redis cache — but every text is unique,
    # so every request is a guaranteed cache miss. This isolates the
    # batching effect from the caching effect.
    await run_load_test("/predict", "WITH batching (cache bypassed)")


if __name__ == "__main__":
    asyncio.run(main())