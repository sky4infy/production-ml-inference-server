"""
Phase 4 — ML Inference Server with Redis caching.

Request flow for /predict:
  1. Check Redis cache
     → HIT:  return immediately (<1ms, model never runs)
     → MISS: continue
  2. Send to batcher (ONNX INT8 + dynamic batching)
  3. Store result in cache (fire-and-forget, doesn't delay response)
  4. Return result

All other endpoints are unchanged from phase 3.
"""

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Backend selection (same as phase 3) ─────────────────────────────────────
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

app = FastAPI(title="ML Inference Server", version="0.4.0")

batcher: InferenceBatcher | None = None


# ── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float
    latency_ms: float


# ── Lifecycle ────────────────────────────────────────────────────────────────
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

    # Log Redis connectivity at startup so you know immediately if it's reachable
    s = cache.stats()
    if s["connected"]:
        print("[startup] Redis connected OK")
    else:
        print("[startup] WARNING: Redis not reachable — cache disabled, predictions still work")


@app.on_event("shutdown")
async def shutdown():
    if batcher:
        await batcher.stop()


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"  : "ok",
        "backend" : "onnx_int8" if ONNX_AVAILABLE else "pytorch",
        "cache"   : cache.stats(),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Production path: Cache → Batcher → Cache write → Response.

    asyncio.to_thread() is used for cache.get and cache.set because
    redis-py is synchronous. Calling blocking I/O directly in an async
    function would freeze the entire event loop.
    """
    start = time.perf_counter()

    # ── Step 1: Cache check ──────────────────────────────────────────────────
    cached = await asyncio.to_thread(cache.get, request.text)
    if cached:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {**cached, "latency_ms": round(elapsed_ms, 2)}

    # ── Step 2: Cache miss → run model via batcher ───────────────────────────
    try:
        result = await batcher.predict(request.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── Step 3: Store in cache (fire-and-forget) ─────────────────────────────
    # create_task schedules the cache write without awaiting it,
    # so the response goes back to the client immediately.
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
    """ONNX INT8, no batcher, no cache. For direct inference benchmarking."""
    start = time.perf_counter()
    result = onnx_predict(request.text)
    return {**result, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}