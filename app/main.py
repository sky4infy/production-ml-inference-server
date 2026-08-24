"""
Phase 6 — ML Inference Server: ONNX batching fixed, benchmarks corrected.

--- What phase 6 changed and why ---

Phase 5 concluded that "ONNX INT8 crashes for batch>=2 because of INT8
quantization" and that "ONNX INT8 is slower than PyTorch on this CPU (no
AVX-512 VNNI)". Both conclusions were wrong. Measured:

1. The batch>=2 crash was NOT quantization. The unquantized FP32 graph fails
   identically at the same node (node_MatMul_206, matmul_helper.h:61).
   Real cause: export_onnx.py traced the model with a batch-1 dummy input,
   and torch.onnx.export defaults to dynamo=True, which routes through
   torch.export — and torch.export specializes any dim whose example value
   is 1. So dynamic_axes={"input_ids": {0: "batch_size"}} was silently
   ignored and batch=1 was baked into the graph. Fixed in export_onnx.py by
   tracing with a 2-row dummy.

2. ONNX INT8 was never slower. app/model_onnx.py padded every request to
   max_length=128 while app/model.py's HF pipeline pads dynamically (~13
   tokens for a typical review) — so ONNX was doing ~10x the compute and the
   benchmark was comparing two different workloads. With padding fixed,
   measured over HTTP with the arms interleaved: PyTorch 20.9ms -> INT8
   8.2ms (2.56x median, 4.46x at p99), and INT8 vs the same graph in FP32
   is 2.36-2.52x across every batch size. INT8 quantization was working the
   whole time. AVX-VNNI is present on this Alder Lake CPU; only AVX-512 is
   fused off.

3. The batcher serialized everything (one batch in flight), so batching
   measured 3.1x slower than unbatched under load. Fixed in app/batcher.py;
   batcher vs no-batcher is now 1.00x throughput. Batching still does not
   pay on CPU — see the README — but it no longer *costs* 3x.

4. Fixing the harness exposed a real bug it had been hiding: app/cache.py
   built a new Redis client, and therefore a new connection pool and TCP
   connection, on every get() and set(). 9.58ms -> 0.31ms per operation once
   the client is reused; a cache hit went 14.6ms -> 0.85ms end to end. The
   cache had been slower than the 8.2ms model it was caching.

The batcher now uses ONNX INT8, chosen at startup by an actual batch probe
rather than a hardcoded assumption — see _onnx_batch_is_safe().

--- Phase 7a ---

A circuit breaker guards the model path. Five consecutive failed batches and
/predict stops calling the backend entirely, rejecting at admission instead of
queueing more doomed work; after a 30s cooldown one probe request tests
recovery. Cache hits are unaffected, because the gate lives inside
batcher.predict() and /predict checks Redis first — so a dead model still serves
everything it has already answered. State is in /health and at
ml_circuit_breaker_state. See app/circuit_breaker.py.
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

# PyTorch stays imported unconditionally: /predict_unbatched uses it as the
# baseline arm in every benchmark, and it is the fallback if the ONNX batch
# probe below fails.
from app.model import (predict as pytorch_predict,
                        predict_batch as pytorch_predict_batch,
                        get_classifier as pytorch_load)
from app.batcher import InferenceBatcher
from app.circuit_breaker import CircuitBreaker
from app import cache

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="ML Inference Server", version="0.7.0")

# Auto-instrument all endpoints — adds request count, latency histograms,
# and status code breakdown to /metrics with 4 lines of code.
# Must be called after app = FastAPI(...) and before first request.
Instrumentator().instrument(app).expose(app)

batcher: InferenceBatcher | None = None
BATCHER_BACKEND = "unknown"   # resolved at startup by the batch probe


def _onnx_batch_is_safe() -> bool:
    """
    Probe the ONNX session with a real batch of 2 before trusting it.

    Why probe instead of hardcode: phase 5 pinned the batcher to PyTorch with
    a comment asserting ONNX could not batch. That assertion was true of the
    graph on disk at the time, but not of ONNX or quantization in general — a
    graph exported from a batch-1 example specializes its batch axis and
    throws "MatMul dimension mismatch" on batch>=2 while still passing a
    batch-1 smoke test. Measuring at startup means a stale artifact degrades
    to PyTorch with a clear message instead of 500-ing every batched request,
    and a correctly exported one is picked up automatically with no code edit.
    """
    if not ONNX_AVAILABLE:
        return False
    try:
        out = onnx_predict_batch(["probe row one", "probe row two"])
        return len(out) == 2
    except Exception as exc:
        print(f"[startup] ONNX batch probe FAILED — {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:130]}")
        print("[startup] The ONNX graph on disk likely has a hardcoded batch "
              "axis. Delete models/model_fp32.onnx* and re-run "
              "`python export_onnx.py`.")
        return False


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
    global batcher, BATCHER_BACKEND
    pytorch_load()
    onnx_load()

    # Pick the batcher backend by measurement, not assumption.
    if _onnx_batch_is_safe():
        predict_fn, BATCHER_BACKEND = onnx_predict_batch, "onnx_int8"
    else:
        predict_fn, BATCHER_BACKEND = pytorch_predict_batch, "pytorch (ONNX batch probe failed)"

    batcher = InferenceBatcher(
        predict_fn=predict_fn,
        batch_size=8,
        timeout_ms=20,
        max_concurrent_batches=4,   # was effectively 1 — see app/batcher.py
        # Constructed here rather than inside InferenceBatcher so the breaker's
        # tuning sits next to the batcher's, which is where a reader looks for
        # it. 5 counts consecutive failed *batches*, not requests — one batch is
        # one predict_fn call, so at batch_size=8 that is up to 40 failed
        # requests before the breaker opens.
        circuit_breaker=CircuitBreaker(failure_threshold=5, cooldown_seconds=30),
    )
    batcher.start()

    print(f"[startup] /predict = {BATCHER_BACKEND} + batching "
          f"(<=4 concurrent batches) + Redis cache + circuit breaker")

    s = cache.stats()
    if s["connected"]:
        print("[startup] Redis connected OK")
    else:
        print("[startup] WARNING: Redis not reachable — cache disabled")

    # Path only, not a full URL. This used to print
    # "http://127.0.0.1:8000/metrics", which is wrong from inside a container:
    # 127.0.0.1 there is the container, and Prometheus reaches this over the
    # compose network as http://app:8000/metrics while a developer reaches it
    # via the published port. The path is the part that is true everywhere.
    print("[startup] Prometheus metrics exposed at /metrics")


@app.on_event("shutdown")
async def shutdown():
    if batcher:
        await batcher.stop()


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"  : "ok",
        "version" : "0.7.0",
        # Reports what the batcher actually resolved to at startup rather than
        # a hardcoded string. The old value claimed ONNX was "unbatched only",
        # which stopped being true once the export was fixed.
        "backend" : f"{BATCHER_BACKEND} (batcher)",
        "onnx_available": ONNX_AVAILABLE,
        "cache"   : cache.stats(),
        # status() is sync and side-effect-free apart from applying the
        # time-based OPEN -> HALF_OPEN transition, so polling this endpoint
        # cannot consume the single recovery probe.
        "circuit_breaker": batcher.circuit_breaker.status() if batcher else {},
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Production path: Cache → Batcher (ONNX INT8) → Cache write → Response."""
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
    """ONNX INT8, no batcher, no cache. The fair single-request comparison
    against /predict_unbatched — both now pad dynamically, so they measure
    the same amount of work."""
    start = time.perf_counter()
    result = onnx_predict(request.text)
    return {**result, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}