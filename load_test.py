"""
Concurrent load test — throughput and tail latency across all three paths.

--- What was wrong with the previous version ---

1. Line 31 was `text = SAMPLE_TEXTS[0]` — every one of the 200 requests sent
   the IDENTICAL text, so they all hashed to the same Redis key. After the
   first request, /predict served 199 straight cache hits while
   /predict_unbatched (which has no cache) ran real inference every time.
   The script then printed that as "WITHOUT batching vs WITH batching" and
   the docstring claimed it "actually proves batching helps". It proved
   nothing about batching: it measured Redis lookups against model forward
   passes. The other four SAMPLE_TEXTS were dead code.

2. It never touched /predict_onnx_unbatched, so the ONNX path was never
   measured under concurrency at all.

3. Percentile indices could run off the end of the list for small
   NUM_REQUESTS (int(n * 0.95) == n when n is small).

This version keeps the cache-hit measurement (it is a legitimate and useful
number) but LABELS it honestly, rotates the sample texts so it is not a
single-key microbenchmark, and adds the ONNX arm. For a clean
batching-only comparison with the cache bypassed, use load_test_nocache.py.

Run this AFTER starting the server: uvicorn app.main:app --reload

Usage:
    python load_test.py
"""

import asyncio
import time

import httpx

BASE_URL     = "http://127.0.0.1:8000"
NUM_REQUESTS = 200
CONCURRENCY  = 50   # how many requests are "in flight" at once

# Quiet period between arms. The arms used to run back-to-back, which let one
# arm's trailing work land inside the next arm's timed window: /predict fires
# its cache write as a detached task (main.py: create_task(to_thread(cache.set)))
# and the batcher can have up to 4 batches still in flight when the last
# response goes out. That contaminated the arm that ran second — the warm-cache
# arm measured 81 req/s directly after the cold arm, and 266 req/s when run on
# its own. Same code, same cache state, 3.3x apart purely from bleed-through.
SETTLE_S = 3.0

SAMPLE_TEXTS = [
    "I absolutely loved this movie, best one this year.",
    "This was a complete waste of my time.",
    "The food was okay, nothing special.",
    "Amazing service, will definitely come back!",
    "I'm so disappointed with this product.",
]


def percentile(sorted_lats: list[float], pct: float) -> float:
    """Clamped index so pct=1.0 cannot walk off the end of the list."""
    idx = min(int(len(sorted_lats) * pct), len(sorted_lats) - 1)
    return sorted_lats[idx]


async def fire_request(client: httpx.AsyncClient, endpoint: str, text: str,
                       semaphore: asyncio.Semaphore) -> tuple[float, float]:
    """Returns (client_ms, server_ms).

    Both numbers matter and they are not the same thing. client_ms includes
    time queued behind the other in-flight requests; server_ms is what the
    handler itself took. At 50-concurrency against a single-process uvicorn
    they diverge by ~100x on cache hits (1ms of work, 130ms of queueing), and
    reporting only client_ms makes a fast path look slow.
    """
    async with semaphore:
        start = time.perf_counter()
        response = await client.post(f"{BASE_URL}{endpoint}", json={"text": text})
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        return elapsed_ms, response.json()["latency_ms"]


async def run_load_test(endpoint: str, label: str, note: str = "",
                        warm_cache: bool = False):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    # Rotate the sample texts instead of hammering SAMPLE_TEXTS[0]. With 5
    # texts and a warm cache /predict still hits on almost everything, which
    # is the point of this script — but it is no longer a single-key test.
    texts = [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(NUM_REQUESTS)]

    async with httpx.AsyncClient(timeout=60.0) as client:
        if warm_cache:
            # Populate the cache BEFORE the timed window.
            #
            # Without this the arm measures a cold cache: all 200 requests are
            # released at once, so the first ~CONCURRENCY of them miss
            # simultaneously and queue through the batcher, and only the
            # remainder hit. That produced "Production (cached): 0.64x vs
            # uncached ONNX" — which reads as "the cache makes it slower" and
            # is purely an artifact of measuring cache population as though it
            # were cache steady state. One sequential request per distinct text
            # is enough; there are only len(SAMPLE_TEXTS) keys.
            for t in SAMPLE_TEXTS:
                await client.post(f"{BASE_URL}{endpoint}", json={"text": t})

        start = time.perf_counter()
        tasks = [fire_request(client, endpoint, texts[i], semaphore)
                 for i in range(NUM_REQUESTS)]
        pairs = await asyncio.gather(*tasks)
        total_s = time.perf_counter() - start

    latencies = [c for c, _ in pairs]
    server    = [s for _, s in pairs]
    s   = sorted(latencies)
    ssv = sorted(server)
    thr = NUM_REQUESTS / total_s

    print(f"\n--- {label} ({endpoint}) ---")
    if note:
        print(f"    {note}")
    print(f"Total requests:   {NUM_REQUESTS}")
    print(f"Concurrency:      {CONCURRENCY}")
    print(f"Total time:       {total_s:.2f}s")
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
    # Little's law sanity check. If client p50 is approximately
    # CONCURRENCY/throughput, the client latency is queueing delay and says
    # nothing about how fast the code is -- read the server-side row instead.
    print(f"  Little's law: {CONCURRENCY}/{thr:.0f} = "
          f"{1000*CONCURRENCY/thr:.0f}ms expected client latency from queueing "
          f"alone (observed p50 {percentile(s, 0.50):.0f}ms)")
    return thr, percentile(ssv, 0.50)


async def settle(seconds: float = SETTLE_S):
    """Let detached cache writes and in-flight batches drain before the next
    arm starts timing. See SETTLE_S."""
    print(f"\n  ...settling {seconds:.0f}s so this arm's tail does not land in the next one")
    await asyncio.sleep(seconds)


async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        h = (await client.get(f"{BASE_URL}/health")).json()
    print(f"Server up. Batcher backend: {h.get('backend')}")

    pt, pt_srv = await run_load_test(
        "/predict_unbatched", "PyTorch, no batcher, no cache",
        "baseline: sync endpoint, so FastAPI runs it in the 40-thread anyio pool")
    await settle()
    onnx, onnx_srv = await run_load_test(
        "/predict_onnx_unbatched", "ONNX INT8, no batcher, no cache",
        "same threadpool as above — isolates the runtime change")
    await settle()
    cold, cold_srv = await run_load_test(
        "/predict", "Production path: cache + batcher, COLD cache",
        "all 200 released at once against an empty cache, so the first ~50 miss")
    await settle()
    warm, warm_srv = await run_load_test(
        "/predict", "Production path: cache + batcher, WARM cache",
        "5 keys pre-populated, so this is steady-state cache-hit throughput",
        warm_cache=True)

    print(f"\n{'=' * 72}")
    print(f"  SUMMARY{'':<19}req/sec    server-side p50")
    print(f"{'=' * 72}")
    print(f"  PyTorch    uncached   : {pt:8.1f}   {pt_srv:8.2f}ms")
    print(f"  ONNX INT8  uncached   : {onnx:8.1f}   {onnx_srv:8.2f}ms")
    print(f"  Production cold cache : {cold:8.1f}   {cold_srv:8.2f}ms")
    print(f"  Production warm cache : {warm:8.1f}   {warm_srv:8.2f}ms")
    print()
    print("  Read the SERVER-SIDE column for how fast each path is. The req/sec")
    print("  column saturates on single-process uvicorn — one event loop doing")
    print("  HTTP parsing and pydantic validation for all 50 connections — so")
    print("  once the handler drops below a few ms, throughput stops tracking it")
    print("  and the client-observed latency is almost entirely queueing.")
    print()
    print(f"  Do NOT quote {pt_srv/onnx_srv:.0f}x as the ONNX speedup. Under 50-way")
    print("  concurrency a handler's own measured time includes being descheduled:")
    print("  40 anyio threads x 2 intra-op threads oversubscribe 8 cores, and")
    print("  PyTorch degrades far worse than ONNX there. That ratio is a")
    print("  SATURATION figure. The per-call speedup is 2.56x — measure it with")
    print("  benchmark_onnx.py, which is sequential and interleaved.")
    print()
    print("  Rows 3 and 4 are CACHE results, not batching results — only 5")
    print("  distinct texts. For batching in isolation: load_test_nocache.py")
    print(f"{'=' * 72}")



if __name__ == "__main__":
    asyncio.run(main())
