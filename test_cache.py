"""
Verifies Redis caching is working correctly.

What to expect:
  Request 1: normal latency (~20-50ms) — cache miss, model runs
  Request 2-5: <5ms — cache hit, model never runs
  /health shows increasing hit count

Run with server up:
    python test_cache.py
"""

import time
import requests

BASE = "http://127.0.0.1:8000"

TEXTS = [
    "I absolutely loved this product!",
    "This was a complete waste of money.",
    "Pretty good overall, would recommend.",
]


def send(text: str, label: str) -> float:
    start = time.perf_counter()
    r = requests.post(f"{BASE}/predict", json={"text": text})
    ms = (time.perf_counter() - start) * 1000
    d = r.json()
    hit = "CACHE HIT ⚡" if ms < 5 else "model ran"
    print(f"  {label:12s} {ms:6.1f}ms  {d['label']:8s} ({d['score']:.4f})  {hit}")
    return ms


print("=" * 60)
print("PHASE 4 — Redis Cache Verification")
print("=" * 60)

# ── Test 1: Same text sent 5 times ──────────────────────────────────────────
print(f"\nTest 1: Same text × 5 (first miss, rest hits)")
text = TEXTS[0]
latencies = []
for i in range(5):
    ms = send(text, f"Request {i+1}")
    latencies.append(ms)

avg_after_first = sum(latencies[1:]) / len(latencies[1:])
print(f"\n  First request (miss):  {latencies[0]:.1f}ms")
print(f"  Avg requests 2-5 (hit): {avg_after_first:.1f}ms")
if avg_after_first < 5:
    print("  ✓ Cache is working correctly")
else:
    print("  ✗ Cache may not be working — check Redis connection")

# ── Test 2: Multiple different texts ────────────────────────────────────────
print(f"\nTest 2: Three different texts × 2 each")
for text in TEXTS:
    print(f"\n  Text: '{text[:40]}...' " if len(text) > 40 else f"\n  Text: '{text}'")
    send(text, "  1st call")
    send(text, "  2nd call")

# ── Health check — show cache stats ─────────────────────────────────────────
print(f"\n{'=' * 60}")
print("Cache stats from /health:")
r = requests.get(f"{BASE}/health")
h = r.json()
c = h.get("cache", {})
print(f"  Connected : {c.get('connected')}")
print(f"  Hits      : {c.get('hits')}")
print(f"  Misses    : {c.get('misses')}")
print(f"  Hit rate  : {c.get('hit_rate')}%")
print(f"  Backend   : {h.get('backend')}")
print("=" * 60)