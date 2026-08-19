"""
Custom Prometheus metrics — Phase 5.

All metric objects are defined here and imported wherever they are used.
This file has NO imports from the rest of the app — no circular imports possible.

Naming conventions followed:
- Counters:   snake_case, no _total suffix (prometheus_client adds it automatically)
- Histograms: snake_case, _seconds suffix for time, no suffix for counts
- Gauges:     snake_case, describes current state

These metrics appear at /metrics alongside the auto-instrumented ones from
prometheus-fastapi-instrumentator. In Grafana, prefix all queries with
"ml_" to find custom metrics quickly.
"""

from prometheus_client import Counter, Histogram

# ── Cache metrics ────────────────────────────────────────────────────────────
cache_hits = Counter(
    "ml_cache_hits",
    "Total number of Redis cache hits — requests served without model inference"
)

cache_misses = Counter(
    "ml_cache_misses",
    "Total number of Redis cache misses — requests that required model inference"
)

# ── Batcher metrics ──────────────────────────────────────────────────────────
batch_size_histogram = Histogram(
    "ml_batch_size",
    "Distribution of batch sizes sent to the model in one forward pass",
    buckets=[1, 2, 4, 8, 16, 32]
    # Buckets chosen to match batch_size=8 cap and show distribution clearly.
    # A p95 of 8 means batches consistently fill to cap — useful tuning signal.
)

model_inference_seconds = Histogram(
    "ml_model_inference_seconds",
    "Raw model inference time per batch (wall clock, excludes queue wait time)",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5]
    # Buckets match observed range: ~20-50ms on Intel i5.
    # 0.005 = 5ms lower bound, 0.5 = 500ms upper bound for outliers.
)