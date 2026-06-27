"""
Fires many CONCURRENT requests at both endpoints and compares throughput.
This is what actually proves batching helps - test_requests.py only sent
requests one at a time, which never exercises the batching logic at all
(the batcher only groups requests that arrive close together).

Run this AFTER starting the server: uvicorn app.main:app --reload

Usage:
    python load_test.py
"""

import asyncio
import time
import httpx

BASE_URL = "http://127.0.0.1:8000"
NUM_REQUESTS = 200
CONCURRENCY = 50  # how many requests are "in flight" at once

SAMPLE_TEXTS = [
    "I absolutely loved this movie, best one this year.",
    "This was a complete waste of my time.",
    "The food was okay, nothing special.",
    "Amazing service, will definitely come back!",
    "I'm so disappointed with this product.",
]


async def fire_request(client: httpx.AsyncClient, endpoint: str, semaphore: asyncio.Semaphore):
    text = SAMPLE_TEXTS[0]
    async with semaphore:
        start = time.perf_counter()
        response = await client.post(f"{BASE_URL}{endpoint}", json={"text": text})
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        return elapsed_ms


async def run_load_test(endpoint: str, label: str):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=30.0) as client:
        start = time.perf_counter()
        tasks = [fire_request(client, endpoint, semaphore) for _ in range(NUM_REQUESTS)]
        latencies = await asyncio.gather(*tasks)
        total_s = time.perf_counter() - start

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]

    print(f"\n--- {label} ({endpoint}) ---")
    print(f"Total requests:   {NUM_REQUESTS}")
    print(f"Concurrency:      {CONCURRENCY}")
    print(f"Total time:       {total_s:.2f}s")
    print(f"Throughput:       {NUM_REQUESTS / total_s:.1f} req/sec")
    print(f"Avg latency:      {sum(latencies) / len(latencies):.1f}ms")
    print(f"p50 latency:      {p50:.1f}ms")
    print(f"p95 latency:      {p95:.1f}ms")


async def main():
    await run_load_test("/predict_unbatched", "WITHOUT batching")
    await run_load_test("/predict", "WITH batching")


if __name__ == "__main__":
    asyncio.run(main())