# Phase 7 — container image for the inference server.
#
# Two stages, and the split is doing real work rather than following a habit:
# the builder owns the ONNX export (which needs torch, ~260 MB of Hugging Face
# download, and about a minute of quantization) and the runtime carries only the
# 64.7 MB INT8 graph that app/model_onnx.py actually loads. The 256 MB FP32
# graph and its .onnx.data sidecar stay behind in the builder.
#
# The export runs at BUILD time on purpose. models/ is gitignored, so a clean
# clone has no graph — and app/main.py:53 handles that by falling back to
# PyTorch *silently*. Baking the export in means a fresh clone gets a working
# INT8 image, and export_onnx.py's non-zero exit turns the phase 3 and phase 6
# export bugs (baked batch axis, batch>=2 MatMul failure) into build failures
# instead of runtime 500s.

# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: builder — install deps, export and validate the ONNX graph
# ═════════════════════════════════════════════════════════════════════════════
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

WORKDIR /build

# Install into a venv rather than system site-packages so the runtime stage can
# copy one self-contained directory and run no pip at all.
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch FIRST, from the CPU index, explicitly.
#
# `pip install torch` on Linux resolves to the CUDA build: roughly 2.5 GB of
# nvidia-* wheels, for a model that runs on two CPU threads
# (torch.set_num_threads(2) in app/model.py). The CPU wheel is a fraction of
# that. This has to be a separate step from the lock file below, because both
# indexes publish 2.12.1 and letting pip choose between them is not
# deterministic. Installing it here first means the lock file's `torch==2.12.1`
# is already satisfied and pip leaves it alone.
RUN pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

# Export FP32 -> quantize INT8 -> validate. This is the build gate.
#
# export_onnx.py is standalone (it imports nothing from app/) and writes to a
# relative models/ dir, so WORKDIR puts the output at /build/models. It exits
# non-zero if the logits batch axis came out baked instead of symbolic, or if
# batch 1 and batch 4 disagree with the expected labels — the exact failure that
# shipped undetected in phase 3 and was misdiagnosed as a quantization bug in
# phase 5. It also warms HF_HOME as a side effect, which the runtime stage
# inherits.
COPY export_onnx.py .
RUN python export_onnx.py

# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: runtime — the served application
# ═════════════════════════════════════════════════════════════════════════════
FROM python:3.13-slim AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    REDIS_URL=redis://redis:6379 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# WORKDIR /app is load-bearing, not cosmetic. app/main.py:53 and
# app/model_onnx.py:19 both resolve the graph as a RELATIVE path
# (Path("models") / "model_int8.onnx"). Launch from anywhere else and the
# existence check fails, ONNX_AVAILABLE goes False, and the server comes up
# happily on the 2.56x-slower PyTorch backend with no error anywhere. Keep the
# graph at /app/models/ and the process started from /app.
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# The Hugging Face snapshot, baked in. Both backends load by model name —
# app/model.py:33 (pipeline) and app/model_onnx.py:40 (tokenizer) — so without
# this the container would pull ~260 MB from the hub on every cold start.
COPY --from=builder /opt/hf /opt/hf

# Only the INT8 graph. app/model_onnx.py never opens the FP32 one; that is
# benchmark_batch_sizes.py's comparison arm, and it runs from the host.
COPY --from=builder /build/models/model_int8.onnx ./models/model_int8.onnx

COPY app/ ./app/

# Non-root. HF_HOME needs to be writable, not just readable: transformers and
# huggingface_hub take file locks inside the cache dir even on a pure read path,
# and as a non-root user without this chown that surfaces as a permission error
# during startup rather than at import.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/hf
USER appuser

EXPOSE 8000

# python -c rather than curl, which python:*-slim does not ship and which would
# cost an apt layer to add.
#
# start-period is generous because startup is genuinely slow: app/main.py's
# startup hook loads the torch pipeline, builds the ORT session, and then runs
# _onnx_batch_is_safe() — a real 2-row forward pass — before the app serves
# anything. Failing the healthcheck during that window would restart-loop a
# container that was about to become healthy.
HEALTHCHECK --interval=15s --timeout=3s --start-period=90s --retries=3 \
    CMD python -c "import sys, urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

# One worker, deliberately.
#
# The batcher is in-process (app/batcher.py), so N workers means N independent
# batchers each filling at 1/N the request rate, and N x Semaphore(4) x 2
# intra-op threads competing for 12 — which is the oversubscription phase 6
# just finished removing. Multiple workers would also split the Prometheus
# registry, so /metrics would report one arbitrary worker instead of the server
# and every counter in the phase 5 writeup would silently become a fraction.
#
# One worker keeps the process identical to the one every number in phases 5
# and 6 was measured on. The known cost is the event-loop ceiling the README
# documents: HTTP parsing and pydantic validation for all connections on one
# thread. Scaling that out needs PROMETHEUS_MULTIPROC_DIR and a batcher that
# does not assume a single process — a later phase, measured, not assumed.
#
# No --reload: it watches the filesystem and forks a reloader for nothing here.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
