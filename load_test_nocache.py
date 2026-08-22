"""
load_test_nocache.py

Batching in isolation: every request is a guaranteed Redis MISS, so /predict's
numbers reflect batcher + model only, with no cache hits mixed in.

--- What was wrong with the previous version ---

1. unique_text() embedded a uuid4:
       f"Review number {i} — {uuid.uuid4()} — this product was okay I guess."
   That guarantees a distinct cache key, but BERT's wordpiece tokenizer splits
   a raw uuid into 27 subword tokens on its own. Total: 42 tokens per request
   versus 13 for the realistic reviews in load_test.py — 3.2x more compute per
   request than the traffic it claimed to simulate. That inflated every
   latency number here and made this file's results non-comparable with
   load_test.py's, even though the docstring invited exactly that comparison.

   Fixed by making texts unique with a short trailing index instead. An
   integer costs 1-2 tokens, so uniqueness no longer distorts the workload.

2. It compared /predict (batched) against /predict_unbatched and labelled the
   difference "the batching effect". But /predict_unbatched is a *sync* def,
   so FastAPI runs it in the 40-worker anyio threadpool, while the old batcher
   ran exactly one batch at a time. That is 40-way parallelism versus 1-way —
   the 3.1x "batching is slower" result was a scheduling artifact. The batcher
   now runs up to 4 concurrent batches (see app/batcher.py), so this
   comparison finally measures batching rather than thread-pool width.

3. No ONNX arm, and percentile indices could run off the end of the list.

Usage:
    python load_test_nocache.py
"""

import asyncio
import time

import httpx

BASE_URL     = "http://127.0.0.1:8000"
NUM_REQUESTS = 200
CONCURRENCY  = 50

# See load_test.py. Arms running back-to-back let one arm's detached cache
# writes and in-flight batches land inside the next arm's timed window.
SETTLE_S = 3.0

# Same sentences as load_test.py so the two scripts' numbers are comparable.
TEMPLATES = [
    "I absolutely loved this movie, best one this year.",
    "This was a complete waste of my time.",
    "The food was okay, nothing special.",
    "Amazing service, will definitely come back!",
    "I'm so disappointed with this product.",
]


def unique_text(i: int) -> str:
    """Unique per request (distinct SHA-256 cache key -> guaranteed miss)
    while staying the length of a real review. The trailing index costs 1-2
    tokens; the uuid it replaced cost 27."""
    return f"{TEMPLATES[i % len(TEMPLATES)]} (review {i})"


def percentile(sorted_lats: list[float], pct: float) -> float:
    idx = min(int(len(sorted_lats) * pct), len(sorted_lats) - 1)
    return sorted_lats[idx]


async def fire_request(client: httpx.AsyncClient, endpoint: str, text: str,
                       semaphore: asyncio.Semaphore) -> tuple[float, float]:
    """Returns (client_ms, server_ms) — see load_test.py for why both."""
    async with semaphore:
        start = time.perf_counter()
        response = await client.post(f"{BASE_URL}{endpoint}", json={"text": text})
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        return elapsed_ms, response.json()["latency_ms"]


async def run_load_test(endpoint: str, label: str, note: str = ""):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    # Fresh offset per arm so no arm can benefit from another arm's cache
    # writes. Without this the second endpoint tested would see hits.
    offset = abs(hash(endpoint)) % 100000
    texts = [unique_text(offset + i) for i in range(NUM_REQUESTS)]

    async with httpx.AsyncClient(timeout=120.0) as client:
        start = time.perf_counter()
        tasks = [fire_request(client, endpoint, texts[i], semaphore)
                 for i in range(NUM_REQUESTS)]
        pairs = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

    latencies = [c for c, _ in pairs]
    server    = [s for _, s in pairs]
    s   = sorted(latencies)
    ssv = sorted(server)
    thr = NUM_REQUESTS / total_time

    print(f"\n--- {label} ({endpoint}) — UNIQUE TEXTS, CACHE BYPASSED ---")
    if note:
        print(f"    {note}")
    print(f"Total requests:   {NUM_REQUESTS}")
    print(f"Concurrency:      {CONCURRENCY}")
    print(f"Total time:       {total_time:.2f}s")
    print(f"Throughput:       {thr:.1f} req/sec")
    print(f"  client-observed (includes queueing behind the other 49):")
    print(f"    avg {sum(latencies)/len(latencies):7.1f}ms   "
          f"p50 {percentile(s, 0.50):7.1f}ms   "
          f"p95 {percentile(s, 0.95):7.1f}ms   "
          f"p99 {percentile(s, 0.99):7.1f}ms")
    print(f"  server-side handler only (the actual cost of the work):")
    print(f"    avg {sum(server)/len(server):7.2f}ms   "
          f"p50 {percentile(ssv, 0.50):7.2f}ms   "
          f"p95 {percentile(ssv, 0.95):7.2f}ms   "
          f"p99 {percentile(ssv, 0.99):7.2f}ms")
    print(f"  Little's law: {CONCURRENCY}/{thr:.0f} = "
          f"{1000*CONCURRENCY/thr:.0f}ms expected client latency from queueing "
          f"alone (observed p50 {percentile(s, 0.50):.0f}ms)")
    return thr, percentile(ssv, 0.50), percentile(ssv, 0.99)


async def settle(seconds: float = SETTLE_S):
    print(f"\n  ...settling {seconds:.0f}s so this arm's tail does not land in the next one")
    await asyncio.sleep(seconds)


async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        h = (await client.get(f"{BASE_URL}/health")).json()
    print(f"Server up. Batcher backend: {h.get('backend')}")

    pt, pt_p50, pt_p99 = await run_load_test(
        "/predict_unbatched", "PyTorch, NO batcher",
        "sync endpoint -> 40-thread anyio pool. The parallelism baseline.")
    await settle()
    onnx, onnx_p50, onnx_p99 = await run_load_test(
        "/predict_onnx_unbatched", "ONNX INT8, NO batcher",
        "same 40-thread pool, so this isolates runtime from scheduling.")
    await settle()
    batched, b_p50, b_p99 = await run_load_test(
        "/predict", "ONNX INT8 + batcher",
        "up to 4 concurrent batches of 8, cache bypassed by unique texts.")

    print(f"\n{'=' * 76}")
    print(f"  SUMMARY (cache bypassed){'':<8}req/sec   srv p50    srv p99")
    print(f"{'=' * 76}")
    print(f"  PyTorch    no batcher : {pt:8.1f}  {pt_p50:8.2f}ms {pt_p99:8.2f}ms")
    print(f"  ONNX INT8  no batcher : {onnx:8.1f}  {onnx_p50:8.2f}ms {onnx_p99:8.2f}ms")
    print(f"  ONNX INT8  + batcher  : {batched:8.1f}  {b_p50:8.2f}ms {b_p99:8.2f}ms")
    print()

    # Verdict computed from the measurement, not asserted in advance. An
    # earlier version of this block claimed "batching should look better at
    # p99" — it does not on this box, and a hardcoded expectation like that is
    # exactly what let three wrong conclusions survive phases 3-5.
    thr_ratio = batched / onnx
    print(f"  VERDICT: batcher throughput {thr_ratio:.2f}x, "
          f"p50 {b_p50/onnx_p50:.2f}x, p99 {b_p99/onnx_p99:.2f}x "
          f"(>1.0 on latency = worse)")
    if thr_ratio < 1.15 and b_p99 > onnx_p99:
        print("  -> Batching is NOT paying for itself on this hardware. Throughput")
        print("     is a wash and tail latency is worse. That is the expected CPU")
        print("     result: a CPU GEMM is already compute-bound at batch 1, so")
        print("     there is no kernel-launch overhead to amortize the way there")
        print("     is on a GPU, while the 20ms batch window is pure added delay.")
        print("     benchmark_batch_sizes.py shows per-item cost roughly flat to")
        print("     batch 8 and clearly WORSE at batch 16, which agrees.")
        print("     What the batcher still gives you is a BOUNDED number of")
        print("     in-flight model calls (4 x 2 threads = 8 on 8 cores) instead")
        print("     of 40 anyio threads x 2 = 80 oversubscribing them. That is a")
        print("     memory- and thread-safety property, not a speedup. On a GPU")
        print("     the same code is where batching would pay.")
    elif thr_ratio >= 1.15:
        print(f"  -> Batching is helping: {thr_ratio:.2f}x throughput.")
    else:
        print("  -> Throughput a wash, tail latency no worse. Batching is neutral")
        print("     here and worth keeping for its bounded-concurrency property.")
    print()
    print("  If the batcher row is dramatically WORSE (~0.3x throughput), check")
    print("  max_concurrent_batches in app/main.py is >1. It was effectively 1")
    print("  until phase 6, which is what made batching look 3.1x slower.")
    print(f"{'=' * 76}")


if __name__ == "__main__":
    asyncio.run(main())
