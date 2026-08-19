"""
Phase 5 — ML Inference Server with Prometheus monitoring.

What's new:
- prometheus-fastapi-instrumentator auto-instruments all endpoints:
  request count, latency histograms, status code breakdown → /metrics
- Custom metrics (cache hits, batch size, inference time) imported from
  app/metrics.py and recorded in cache.py and batcher.py
- /metrics endpoint exposed for Prometheus to scrape every 15 seconds

Everything else (ONNX backend, Redis cache, batcher) unchanged from phase 4.
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

from app.model import predict as pytorch_predict, get_classifier as pytorch_load
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
        predict_fn=onnx_predict_batch,
        batch_size=8,
        timeout_ms=20,
    )
    batcher.start()

    backend = "ONNX INT8" if ONNX_AVAILABLE else "PyTorch"
    print(f"[startup] /predict using {backend} + batching + Redis cache")

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
        "backend" : "onnx_int8" if ONNX_AVAILABLE else "pytorch",
        "cache"   : cache.stats(),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Production path: Cache → Batcher → Cache write → Response."""
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
    """ONNX INT8, no batcher, no cache. Direct inference benchmarking."""
    start = time.perf_counter()
    result = onnx_predict(request.text)
    return {**result, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}