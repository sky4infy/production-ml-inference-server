"""
Verifies Redis caching is working correctly.

Why the thresholds are RELATIVE, not absolute:
  The old version flagged a cache hit as `ms < 5` and failed the whole test
  if the average hit was slower than 5ms. But `ms` is measured end-to-end
  over HTTP from a separate process — loopback RTT plus requests/urllib3
  overhead alone exceeds 5ms on Windows, so that threshold was unreachable no
  matter how healthy Redis was. It reported "✗ Cache may not be working" on a
  run where /health showed 158 hits at a 74.9% hit rate.

  A cache test can only honestly measure hit latency against *this machine's*
  miss latency. So we measure a miss first, then require hits to be
  meaningfully faster than it, and cross-check the hit counter from /health.

Run with server up:
    python test_cache.py
"""

import time
import uuid

import requests

BASE = "http://127.0.0.1:8000"

# Hits must be at least this much faster than a measured miss to pass.
HIT_SPEEDUP_REQUIRED = 1.5

TEXTS = [
    "I absolutely loved this product!",
    "This was a complete waste of money.",
    "Pretty good overall, would recommend.",
]


def cache_counters() -> tuple[int, int]:
    c = requests.get(f"{BASE}/health").json().get("cache", {})
    return c.get("hits", 0), c.get("misses", 0)


def send(text: str, label: str, miss_baseline: float | None = None) -> float:
    start = time.perf_counter()
    r = requests.post(f"{BASE}/predict", json={"text": text})
    ms = (time.perf_counter() - start) * 1000
    r.raise_for_status()
    d = r.json()
    if miss_baseline is None:
        tag = ""
    elif ms < miss_baseline / HIT_SPEEDUP_REQUIRED:
        tag = "CACHE HIT"
    else:
        tag = "model ran"
    print(f"  {label:12s} {ms:6.1f}ms  {d['label']:8s} ({d['score']:.4f})  {tag}")
    return ms


print("=" * 60)
print("Cache Verification")
print("=" * 60)

# ── Warm the server first ────────────────────────────────────────────────────
# The very first request after boot pays lazy model load + ONNX session init.
# The old script folded that one-off cost into its "miss" number, inflating
# the apparent cache win. Warm with a throwaway unique text so we neither
# measure startup nor seed the key we are about to test.
requests.post(f"{BASE}/predict", json={"text": f"warmup {uuid.uuid4()}"})

# ── Establish a real miss baseline on this machine ───────────────────────────
print("\nCalibrating: measuring uncached (miss) latency on 3 unique texts")
miss_samples = [send(f"Calibration text {uuid.uuid4()}", f"miss {i+1}")
                for i in range(3)]
miss_baseline = sum(miss_samples) / len(miss_samples)
print(f"\n  Miss baseline (avg of 3): {miss_baseline:.1f}ms")

# ── Test 1: Same text sent 5 times ──────────────────────────────────────────
print(f"\nTest 1: Same text x 5 (first miss, rest hits)")
text = f"{TEXTS[0]} {uuid.uuid4()}"   # unique per run so run #2 isn't pre-warmed
hits_before, misses_before = cache_counters()

latencies = [send(text, f"Request {i+1}", miss_baseline) for i in range(5)]

hits_after, misses_after = cache_counters()
avg_after_first = sum(latencies[1:]) / len(latencies[1:])
speedup = miss_baseline / avg_after_first

print(f"\n  First request (miss):   {latencies[0]:.1f}ms")
print(f"  Avg requests 2-5 (hit): {avg_after_first:.1f}ms")
print(f"  Miss baseline:          {miss_baseline:.1f}ms")
print(f"  Hit speedup:            {speedup:.2f}x  (need >= {HIT_SPEEDUP_REQUIRED}x)")
print(f"  /health hits delta:     +{hits_after - hits_before}  (expect +4)")

latency_ok = speedup >= HIT_SPEEDUP_REQUIRED
counter_ok = (hits_after - hits_before) >= 4

if latency_ok and counter_ok:
    print("  PASS - cache is working (both latency and hit counter agree)")
elif counter_ok and not latency_ok:
    print(f"  PASS (weak) - hit counter confirms {hits_after - hits_before} hits, "
          f"but HTTP overhead is masking the latency win on this machine")
else:
    print("  FAIL - hit counter did not increase; check the Redis connection")

# ── Test 2: Multiple different texts ────────────────────────────────────────
print(f"\nTest 2: Three different texts x 2 each")
for t in TEXTS:
    t = f"{t} {uuid.uuid4()}"
    print(f"\n  Text: '{t[:40]}...'")
    send(t, "  1st call", miss_baseline)
    send(t, "  2nd call", miss_baseline)

# ── Health check — show cache stats ─────────────────────────────────────────
print(f"\n{'=' * 60}")
print("Cache stats from /health:")
h = requests.get(f"{BASE}/health").json()
c = h.get("cache", {})
print(f"  Connected : {c.get('connected')}")
print(f"  Hits      : {c.get('hits')}")
print(f"  Misses    : {c.get('misses')}")
print(f"  Hit rate  : {c.get('hit_rate')}%")
print(f"  Backend   : {h.get('backend')}")
print("=" * 60)
