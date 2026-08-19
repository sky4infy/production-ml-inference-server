"""
Phase 5 — ML Inference Server with Prometheus monitoring.

What's new:
- prometheus-fastapi-instrumentator auto-instruments all endpoints:
  request count, latency histograms, status code breakdown → /metrics
- Custom metrics (cache hits, batch size, inference time) imported from
  app/metrics.py and recorded in cache.py and batcher.py
- /metrics endpoint exposed for Prometheus to scrape every 15 seconds

Everything else (ONNX backend, Redis cache, batcher) unchanged from phase 4.

--- Phase 5 debugging fix (post-Prometheus) ---
Benchmarking found the quantized ONNX INT8 model throws a
DynamicQuantizeMatMul dimension-mismatch error for any batch size >= 2
(works fine at batch=1). Since the batcher groups concurrent requests
into batches of up to 8, sending it through ONNX crashed every batched
request under load. ONNX INT8 was already measured slower than PyTorch
for single requests on this CPU (no AVX-512 VNNI), so PyTorch is used
for the batcher. ONNX INT8 remains available at /predict_onnx_unbatched
for single-request comparison — see README for full writeup.
"""

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# ── Backend selection ────────────────────────────────────────────────────────
ONNX_AVAILABLE = Path("models/model_int8.onnx").exists()

if ONNX_AVAILABLE:
    from app.model_onnx import (predict as onnx_predict,
                                 predict_batch as onnx_predict_batch,
                                 get_classifier as onnx_load)
    print("[startup] ONNX INT8 backend detected")
else:
    from app.model import (predict as onnx_predict,
                            predict_batch as onnx_predict_batch,
                            get_classifier as onnx_load)
    print("[startup] ONNX model not found - using PyTorch backend")

# predict_batch imported explicitly for the batcher — ONNX INT8 fails with
# a DynamicQuantizeMatMul dimension mismatch for batch_size >= 2, so the
# batcher must always use PyTorch regardless of ONNX_AVAILABLE.
from app.model import (predict as pytorch_predict,
                        predict_batch as pytorch_predict_batch,
                        get_classifier as pytorch_load)
from app.batcher import InferenceBatcher
from app import cache

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="ML Inference Server", version="0.5.0")

# Auto-instrument all endpoints — adds request count, latency histograms,
# and status code breakdown to /metrics with 4 lines of code.
# Must be called after app = FastAPI(...) and before first request.
Instrumentator().instrument(app).expose(app)

batcher: InferenceBatcher | None = None


# ── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float
    latency_ms: float


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global batcher
    pytorch_load()
    onnx_load()
    batcher = InferenceBatcher(
        predict_fn=pytorch_predict_batch,  # ← FIXED: was onnx_predict_batch
        batch_size=8,
        timeout_ms=20,
    )
    batcher.start()

    print("[startup] /predict using PyTorch + batching + Redis cache "
          "(ONNX INT8 batch>=2 crashes with DynamicQuantizeMatMul error — see README)")

    s = cache.stats()
    if s["connected"]:
        print("[startup] Redis connected OK")
    else:
        print("[startup] WARNING: Redis not reachable — cache disabled")

    print("[startup] Prometheus metrics available at http://127.0.0.1:8000/metrics")


@app.on_event("shutdown")
async def shutdown():
    if batcher:
        await batcher.stop()


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"  : "ok",
        "version" : "0.5.0",
        "backend" : "pytorch (batcher) + onnx_int8 (unbatched only)" if ONNX_AVAILABLE else "pytorch",
        "cache"   : cache.stats(),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Production path: Cache → Batcher (PyTorch) → Cache write → Response."""
    start = time.perf_counter()

    cached = await asyncio.to_thread(cache.get, request.text)
    if cached:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {**cached, "latency_ms": round(elapsed_ms, 2)}

    try:
        result = await batcher.predict(request.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    asyncio.create_task(asyncio.to_thread(cache.set, request.text, result))

    elapsed_ms = (time.perf_counter() - start) * 1000
    return {**result, "latency_ms": round(elapsed_ms, 2)}


@app.post("/predict_unbatched", response_model=PredictResponse)
def predict_unbatched(request: PredictRequest):
    """PyTorch FP32, no batcher, no cache. Baseline only."""
    start = time.perf_counter()
    result = pytorch_predict(request.text)
    return {**result, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}


@app.post("/predict_onnx_unbatched", response_model=PredictResponse)
def predict_onnx_unbatched(request: PredictRequest):
    """ONNX INT8, no batcher, no cache. Single-request only — batching
    this backend crashes (DynamicQuantizeMatMul dimension mismatch)."""
    start = time.perf_counter()
    result = onnx_predict(request.text)
    return {**result, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}