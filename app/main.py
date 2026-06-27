"""
Phase 3 API: auto-detects ONNX INT8 model if exported, falls back to PyTorch.

Endpoints:
  /predict_unbatched      PyTorch FP32, no batcher  (original baseline)
  /predict_onnx_unbatched ONNX INT8,  no batcher   (fair inference comparison)
  /predict                ONNX INT8 + batcher       (full production path)
"""

import time
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

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
    print("[startup] ONNX model not found - using PyTorch for all endpoints")

from app.model import predict as pytorch_predict, get_classifier as pytorch_load
from app.batcher import InferenceBatcher

app = FastAPI(title="ML Inference Server", version="0.3.0")
batcher: InferenceBatcher | None = None


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float
    latency_ms: float


@app.on_event("startup")
async def startup():
    global batcher
    pytorch_load()
    onnx_load()
    batcher = InferenceBatcher(predict_fn=onnx_predict_batch, batch_size=8, timeout_ms=20)
    batcher.start()
    backend = "ONNX INT8" if ONNX_AVAILABLE else "PyTorch"
    print(f"[startup] /predict using {backend} + batching")
    print(f"[startup] /predict_onnx_unbatched using {backend} directly")


@app.on_event("shutdown")
async def shutdown():
    if batcher:
        await batcher.stop()


@app.get("/health")
def health():
    return {"status": "ok", "backend": "onnx_int8" if ONNX_AVAILABLE else "pytorch"}


@app.post("/predict_unbatched", response_model=PredictResponse)
def predict_unbatched(request: PredictRequest):
    """PyTorch FP32, no batcher. Always the baseline."""
    start = time.perf_counter()
    result = pytorch_predict(request.text)
    return {**result, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}


@app.post("/predict_onnx_unbatched", response_model=PredictResponse)
def predict_onnx_unbatched(request: PredictRequest):
    """ONNX INT8, no batcher. Fair apples-to-apples vs /predict_unbatched."""
    start = time.perf_counter()
    result = onnx_predict(request.text)
    return {**result, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """ONNX INT8 + dynamic batching. Full production path."""
    start = time.perf_counter()
    result = await batcher.predict(request.text)
    return {**result, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}