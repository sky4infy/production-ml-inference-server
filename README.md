# ML Inference Server

A FastAPI server that wraps a DistilBERT sentiment classifier and serves
predictions over HTTP: request batching, ONNX INT8 quantization, Redis
caching, a circuit breaker on the model path, and Prometheus/Grafana metrics
— each phase measured before the next was added. Phase 7b puts the whole
stack behind one `docker compose up`.

```bash
docker compose up -d --build
```

Phases 1–2 below are the original build notes. **Phase 6 documents four
findings from phases 3–5 that turned out to be measurement artifacts**, and
what the harness now prints so they cannot recur; read it before trusting
any number in the earlier sections. Phase 5 gets the same treatment for the
Grafana dashboard, where four of the five panel values turn out to be
properties of the PromQL query rather than of the server.

## Architecture

```
HTTP request
     │
     ▼
FastAPI  (one uvicorn worker, async)
     │
     ├─► GET  /health    backend, cache and breaker status
     ├─► GET  /metrics   Prometheus scrape target
     │
     │  POST /predict
     ▼
Redis cache-aside   SHA-256 key, 1h TTL
     │
     ├─► hit ─────────────────────────────────► ~0.7 ms, model never touched
     │
     ▼ miss
Circuit breaker gate      CLOSED     admit
  checked before the      OPEN       503 in 0.015 ms, model never called
  queue, not around       HALF_OPEN  exactly one probe admitted
  the model call
     │
     ▼
Request batcher  (asyncio.Queue)
  batch_size=8, 20 ms collection window, ≤4 concurrent batches
     │
     ▼
ONNX Runtime INT8 — DistilBERT SST-2
  backend resolved at startup by a real batch-of-2 probe;
  PyTorch FP32 if that probe fails
     │
     ▼
observes ml_batch_size, ml_model_inference_seconds
     │
     ▼
/metrics ──► Prometheus (15 s scrape) ──► Grafana

POST /predict_unbatched       PyTorch FP32 — no cache, no batcher, no breaker
POST /predict_onnx_unbatched  ONNX INT8   — no cache, no batcher, no breaker
     documented baseline arms for the benchmarks below, not serving paths
```

| Layer | Choice |
|---|---|
| API | FastAPI + uvicorn, one worker deliberately |
| Model | `distilbert-base-uncased-finetuned-sst-2-english` |
| Inference | ONNX Runtime INT8 dynamic quantization, PyTorch FP32 fallback |
| Cache | Redis 7, cache-aside, SHA-256 keys, 1 h TTL, singleton client |
| Reliability | consecutive-failure circuit breaker, checked at admission |
| Observability | `prometheus-fastapi-instrumentator` + 6 custom metrics, Grafana |
| Packaging | Docker Compose — `app`, `redis`, `prometheus`, `grafana` |
| Python | 3.13 — host venv and image both, so the container matches the process every number was measured on |

## Project structure

```
.
├── app/
│   ├── main.py              FastAPI app — startup backend probe, routes, /health
│   ├── model.py             PyTorch FP32 backend — baseline and fallback
│   ├── model_onnx.py        ONNX INT8 backend, pads to batch-longest
│   ├── batcher.py           InferenceBatcher — queue, 20 ms window, Semaphore(4)
│   ├── circuit_breaker.py   admission gate — CLOSED / OPEN / HALF_OPEN
│   ├── cache.py             Redis cache-aside, module-level singleton client
│   └── metrics.py           the 6 custom Prometheus metrics
├── models/                  gitignored — built by export_onnx.py, or at image build
│   ├── model_fp32.onnx      (+ model_fp32.onnx.data sidecar, 256 MB)
│   └── model_int8.onnx      64.7 MB — the only graph that ships
├── export_onnx.py           export + INT8 quantize; validates the batch axis
├── benchmark_onnx.py        ONNX vs PyTorch over HTTP, interleaved arms
├── benchmark_batch_sizes.py in-process per-item cost by batch size
├── load_test.py             4 arms — PyTorch, ONNX, cold cache, warm cache
├── load_test_nocache.py     batching in isolation, cache bypassed
├── test_requests.py         phase 1 smoke test
├── test_cache.py            hit rate + speedup, calibrated per machine
├── test_circuit_breaker.py  state machine — unit, integration, live
├── Dockerfile               builder exports the graph; runtime is 3.13-slim
├── docker-compose.yml       app, redis, prometheus, grafana on one network
├── prometheus.yml           scrapes app:8000/metrics every 15 s
├── docs/
│   └── grafana-dashboard.png  the 5 panels under sustained load
├── requirements.txt         unpinned, for host work
└── requirements.lock.txt    pinned — what the image actually installs
```

## Phase 1: baseline

Get it running and measured before adding batching, caching, or metrics.

## 1. Set up a virtual environment

A virtual environment keeps this project's packages separate from anything
else on your machine, so installs here can't break other projects.

```bash
# from inside the ml-inference-server folder
python -m venv venv

# activate it (do this every time you open a new terminal for this project)
# macOS / Linux:
source venv/bin/activate
# Windows (cmd):
venv\Scripts\activate.bat
# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

You'll know it worked because your terminal prompt will show `(venv)` at
the start of the line. Everything below assumes it's active.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

This pulls in PyTorch and Transformers, so it's a few hundred MB and may
take a few minutes on the first run. Only needs to happen once per venv.

## 3. Run the server

```bash
uvicorn app.main:app --reload
```

First startup will download the model (`distilbert-base-uncased-finetuned-sst-2-english`,
~260MB) from Hugging Face the very first time — after that it's cached
locally and startup is fast. Leave this terminal running.

## 4. Test it

In a **second** terminal (with the same venv activated), check the health
endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Then send a real prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I really loved this product!"}'
```

You should get back something like:
```json
{"label": "POSITIVE", "score": 0.9998, "latency_ms": 42.1}
```

Or run the provided test script to send several requests and see a
baseline average latency:

```bash
pip install requests   # if not already installed
python test_requests.py
```

**Save the average latency number it prints** — that's your "before"
number. Once batching is added in phase 2, you'll re-run this same script
under load and compare.

## What's next (phase 2)

Right now every request blocks on the model individually — that's the
baseline we want to improve on. Phase 2 adds:
- Async request handling so the server can accept many requests at once
- A request batcher that groups requests arriving within a short window
  and runs them through the model together
- A proper load test (many concurrent requests) to produce a real
  throughput number, not just single-request latency

Don't move to phase 2 until this phase is running cleanly and you
understand what each file does — ask if anything in `model.py` or
`main.py` is unclear before we add complexity on top of it.

## Phase 2: batching and async

New files: `app/batcher.py` (the batching logic itself, with detailed
comments — read this one closely) and `load_test.py` (concurrent load
test, replaces `test_requests.py` which only sent one request at a time
and so never actually exercised batching).

The server now exposes two prediction endpoints on the same running
instance, so you can compare them fairly:
- `POST /predict_unbatched` — the original phase 1 behavior, one model
  call per request, kept specifically as a baseline.
- `POST /predict` — new batched, async behavior. Requests that arrive
  within a 20ms window (up to 8 at a time) get grouped into a single
  model call.

### Running it

1. Reinstall dependencies (added `httpx` for the load test):
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server same as before:
   ```bash
   uvicorn app.main:app --reload
   ```
3. In a second terminal, run the load test:
   ```bash
   python load_test.py
   ```

This sends 200 requests at 50-concurrency to each endpoint in turn and
prints throughput (req/sec), average latency, and p50/p95 latency for
both.

> **Correction (see Phase 6).** The advice above — "the req/sec difference
> between the two is your real before-vs-after batching number" — was wrong
> as written, because `load_test.py` sent the *same* text for all 200
> requests. Once caching arrived in Phase 4, `/predict` served 199 cache
> hits while `/predict_unbatched` ran 200 real forward passes, so the
> "batching" number was actually measuring Redis against DistilBERT. Both
> the script and this claim are fixed; for a batching-only number use
> `load_test_nocache.py`.

## Phase 3: ONNX export

`export_onnx.py` converts the PyTorch model to ONNX and writes two graphs
to `models/`:

| File | Size | What it is |
|---|---|---|
| `model_fp32.onnx` (+ `.onnx.data`) | 256.1 MB | direct export, full precision |
| `model_int8.onnx` | 64.7 MB | dynamically quantized, 4.0x smaller |

```bash
python export_onnx.py
```

Two things about this script are worth understanding before you trust its
output, because both bit this project:

**The batch axis must actually be dynamic.** `dynamic_axes` is a request,
not a guarantee. `torch.export` specializes any dimension whose example
value is `1`, because a size-1 dim carries broadcasting semantics it is not
allowed to discard. Exporting with a single-row dummy input therefore bakes
`batch=1` into the attention reshapes even though `dynamic_axes` says
otherwise — and the export succeeds silently. The graph then works
perfectly at batch 1 and dies at batch 2 inside `MatMul`. The dummy input
here has **two rows** for exactly this reason, and the script verifies the
result three ways: the `logits` batch dim must be symbolic (`dim_param`,
not `dim_value`), and both batch 1 and batch 4 must run and return the
right number of correctly-labelled rows. It exits non-zero if any check
fails, and it re-exports rather than reusing an existing `model_fp32.onnx`
whose batch axis is baked.

**Size has to include the sidecar.** A >2 GB-capable export writes weights
to `model_fp32.onnx.data` and leaves a 0.7 MB stub, so `stat().st_size` on
the `.onnx` alone reports the quantized model as *8578% larger*. The
`model_size_mb()` helper sums both files.

## Phase 4: Redis caching

Identical text is common in real traffic, and DistilBERT is deterministic,
so repeated inputs never need a second forward pass. `app/cache.py` keys on
`sha256(text)` with a 1-hour TTL.

```bash
docker compose up -d redis     # was: docker run -d -p 6379:6379 redis:alpine
python test_cache.py
```

Measured on the 5-text rotation: **~73% hit rate**. After the Phase 6
connection-pool fix a hit costs **0.85 ms** server-side against **8.2 ms**
for an ONNX INT8 forward pass — about 10x. Before that fix a hit cost
**14.6 ms**, i.e. *more than the model it was caching*; see Phase 6. The
cache fails open: if Redis is unreachable every request goes to the model.

`test_cache.py` originally decided hit-vs-miss with `latency_ms < 5`. That
threshold is unreachable over HTTP loopback on Windows regardless of
whether the cache works, so the script reported FAIL on a run with 158
hits and a 74.9% hit rate. It now measures a miss baseline from unique
texts and requires the hits to be **1.5x faster relative to that
baseline**, cross-checked against the server's own hit/miss counters from
`/health`. Current result: **13.0x, PASS**.

## Phase 5: Prometheus + Grafana

`/metrics` exposes the HTTP metrics that `prometheus-fastapi-instrumentator`
adds automatically, plus four custom ones from `app/metrics.py`:

| Metric | Type | Observed at |
|---|---|---|
| `ml_cache_hits` / `ml_cache_misses` | Counter | `app/cache.py:79`, `:81`, `:84` |
| `ml_batch_size` | Histogram, buckets `1,2,4,8,16,32` | `app/batcher.py:101` |
| `ml_model_inference_seconds` | Histogram, buckets 5 ms–500 ms | `app/batcher.py:106` |

The last two are observed **inside the batcher, once per batch**. So they
describe batches rather than requests, and they are blind to everything the
cache served and to both unbatched endpoints. (Phase 7a adds two more,
`ml_circuit_breaker_state` and `ml_circuit_breaker_rejections`.)

```bash
docker compose up -d prometheus grafana
# Grafana at localhost:3000 (admin/admin), Prometheus at localhost:9090
```

Prometheus runs in Docker and the server did not, so `prometheus.yml` targeted
`host.docker.internal:8000` rather than `localhost` — inside a container
`localhost` is the container. (Phase 7b moved the server into Docker too, so the
target is now the compose service name `app:8000`; the old line is kept
commented in `prometheus.yml` for host-run development.)
Scrape interval 15 s, retention 7 d.
Target health over the retained window: **511 scrapes, up, mean scrape
duration 8.97 ms** (max 185 ms, on the cold first scrape). Serving `/metrics`
costs the server 5.11 ms, or 0.03% of a 15-second interval.

### The session behind these numbers

Two `load_test.py` runs, ~01:31 and ~01:37. Each run is four arms of 200
requests at concurrency 50, plus five cache-priming requests. Read straight
off the counters:

| Handler | Requests | Mean latency |
|---|---|---|
| `/predict` | 810 | 22.98 ms |
| `/predict_onnx_unbatched` | 400 | 68.47 ms |
| `/predict_unbatched` | 400 | 522.51 ms |
| `/metrics` | 81 | 5.11 ms |
| `/health` | 2 | 3.91 ms |
| **total** | **1693 — every one 2xx** | 150.87 ms |

Pooled across all 1693 requests (`http_request_duration_highr_seconds`, which
has fine buckets but no `handler` label): p50 31.0 ms, p95 646.9 ms, p99
1344.5 ms. That spread is the four arms sitting on top of each other, not
variance within any one path.

Two counters carry the whole batching-and-caching story:

- 810 `/predict` requests → **50 cache misses → 7 model forward passes.**
  116 requests per forward pass.
- Run 1's cold arm released 200 requests at once against an empty cache, so
  the first 50 — exactly `CONCURRENCY` — missed before any cache write
  landed. Those 50 items became the 7 batches. Run 2 missed zero times, so
  the batcher never ran again.

Every batching and inference number on the dashboard therefore rests on
**7 observations, formed from one burst of 50 requests.**

### Reading the panels

The five panel values and what each is actually made of. The queries below
reproduce the plotted values exactly.

**Cache Hit Rate — panel 87.65%, session 93.8%.** The query is
`rate(ml_cache_hits_total[5m]) / (rate(ml_cache_hits_total[5m]) + rate(ml_cache_misses_total[5m])) * 100`.
That 5-minute window contained all 50 misses but only 355 of the 760 hits:
355/405 = 87.65%. Cumulative for the session it is 760/810 = **93.8%**, and
the model ran for **6.2%** of `/predict` requests, not the 12.3% the windowed
figure implies. Both are correct; they answer different questions, and a rate
window straddling a cold start answers neither cleanly. Neither contradicts
Phase 4's ~73% — that is `test_cache.py`'s 5-text rotation, a different
traffic shape from a 200-request burst over 5 keys.

**Batch Size p95 = 7.80 is arithmetic, not measurement.** All 7 batches
landed in the `(4, 8]` bucket, and `histogram_quantile` interpolates linearly
inside it: `4 + 4 × 0.95 = 7.8`. It prints 7.80 if every batch held 5 items
and 7.80 if every batch held 8 — once every observation lands in one bucket,
the value is fixed by the bucket edges `[1,2,4,8,16,32]` and the percentile
alone. What the histogram does support: 7 batches, 50 items, **mean 7.14**,
none of them ≤ 4. For batch size, `ml_batch_size_sum / ml_batch_size_count` is
a real number; the p95 of a six-bucket histogram is not.

**Model Inference p99 = 74.4 ms is a bucket edge, and it is per batch.** Same
mechanism: 4 of the 7 observations fall in `(25, 50] ms` and 3 in
`(50, 75] ms`, so `0.05 + 0.025 × (6.93 − 4)/3 = 0.0744`. The p99 of 7 samples
sits 97.7% of the way through the highest occupied bucket by construction —
it is the bucket edge, not the slowest call. The measurable
figure is the mean — 343.6 ms / 7 = **49.1 ms per batch** — and at 7.14 items
per batch that is **6.9 ms per item**, consistent with the 8.2 ms
single-request INT8 number below, not with a 74 ms model call. Quoting 74 ms
as the model's per-request cost overstates it by ~10x.

**Latency Percentiles: the per-handler histogram has three buckets.**
`prometheus-fastapi-instrumentator` defaults to
`latency_lowr_buckets = (0.1, 0.5, 1)` for `http_request_duration_seconds`
(`instrumentation.py:201`); the fine buckets go to
`http_request_duration_highr_seconds`, which has no `handler` label. So every
per-handler percentile is an interpolation between 0 and 100 ms, 100 and
500 ms, or 500 ms and 1 s. At the instant the screenshots were taken:

| Handler | p50 | p95 | p99 | mean |
|---|---|---|---|---|
| `/metrics` | 50 ms | 95 ms | 99 ms | 5.11 ms |
| `/predict` | 50 ms | 95 ms | 99 ms | 22.98 ms |
| `/predict_unbatched` | 369 ms | 902 ms | 980 ms | 522.51 ms |

`/predict` and `/metrics` plot **identical lines** — 50/95/99 ms — because
both finish inside the first bucket, so the histogram cannot tell a 1 ms
cache hit from a 99 ms one. Any reading of this panel that has `/metrics`
"near 0 ms" while `/predict` shows cache-plus-inference is reporting a
difference the buckets cannot represent: the two series are drawn on top of
each other. `/predict_unbatched`'s 902/980 ms are interpolations inside one
500 ms-wide bucket — all the histogram knows is "between 0.5 s and 1 s." Over
the full session 50 of its 400 requests (12.5%) exceeded 1 s, putting its
session p95 and p99 past the last finite bucket, where `histogram_quantile`
can only return 1.0.

**Request Rate 1.75 req/s, against ~96 req/s actual.**
`rate(http_requests_total{handler="/predict_unbatched"}[1m])` peaks at
**1.748** at 01:31:30 — the plotted value. That arm sent 200 requests; at
522.51 ms mean latency and concurrency 50, Little's law puts real throughput
at 50 / 0.5225 ≈ **96 req/s**, so the burst lasted about 2 seconds. `rate()`
divides the counter increase by the **window**, not by the burst, so a
2-second burst inside a 60-second window is averaged down ~55x. The 15 s
scrape interval compounds it: a 2 s burst is a single counter jump, and no
rate window can recover its shape. Peak throughput has to come from the load
harness, which times it directly. (`histogram_quantile` is not the fix — it
does not apply to counters.)

**The flat lines are the rate window, not stability.** Every panel in the
phase 5 screenshots (taken during that session, not the sustained-load
capture further down) holds one value dead flat for five minutes and is
empty for the other 5h 55m. That is a 5-minute rate window republishing a
single scrape's delta at every step until the delta ages out. There is one
measurement per panel, so a flat line here cannot show that the model is
not degrading, that there is no memory leak, or that there is no thermal
throttling. Phase 6's E-core finding — one identical PyTorch call recorded
at 17.8 ms in one phase and 31.7 ms in another — is exactly the drift this
chart is too sparse to see.

### What the dashboard did establish

None of the above is an argument against the dashboard. Four of its findings
survive scrutiny, and they are the ones that come from counters rather than
from quantiles:

- **1693 requests, 1693 2xx.** No silent failures on any path, including the
  ONNX batch path Phase 6 had to fix.
- **810 requests served by 7 forward passes.** The cache and the batcher
  compose: the cache removed 93.8% of the work, the batcher folded what was
  left into 7 calls. Measured in production traffic, not inferred from a
  benchmark.
- **The batcher fills under burst load** — mean 7.14 against a cap of 8, from
  50 requests released simultaneously. It says nothing about steady state,
  where Phase 6 measured batching as a 1.00x wash on this CPU.
- **The target is scrapeable and cheap**, and `up` goes to 0 the moment the
  server stops — the one signal here that needs no interpretation.

### What to change before trusting the panels again

1. Widen `latency_lowr_buckets`. The default three cannot separate any two
   paths in this project; a cache hit and a saturated PyTorch call are three
   orders of magnitude apart. Roughly
   `(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5)`.
2. Add buckets between 1 and 8 to `ml_batch_size`, or drop the p95 and plot
   `ml_batch_size_sum / ml_batch_size_count`.
3. Plot `ml_model_inference_seconds_sum / ml_batch_size_sum` for per-item
   cost, since the histogram observes batches.
4. Drive load for longer than the rate window before reading the rate panel,
   or read throughput from the harness.
5. Put `ml_batch_size_count` and `ml_model_inference_seconds_count` on the
   dashboard beside every percentile derived from them. A p99 over 7 samples
   should look wrong at a glance.

### The dashboard, re-captured under sustained load

![Grafana dashboard under sustained load](docs/grafana-dashboard.png)

This is the same five panels, captured against the Phase 7b compose stack
(note `instance="app:8000"`) with recommendation 4 above actually applied:
load ran continuously for longer than the widest rate window instead of in
two bursts, so the lines carry shape rather than republishing one scrape's
delta. Read off it directly — model inference p99 climbing 0.25 s → 0.39 s
as the queue fills, batch size p95 pinned at 7.61–7.65 against a cap of 8,
`/predict` steady at ~13.5 req/s alongside ~9 req/s on each unbatched
endpoint.

Three caveats that matter more than the picture:

- **The cache hit rate is a property of the harness, not of production.**
  The load alternates `load_test_nocache.py` (unique texts, guaranteed
  misses) with `load_test.py` (five rotating texts, near-total hits), so
  60–83% is the mix I chose in order to keep the model panels fed. An
  earlier attempt at this screenshot left three panels empty for exactly
  this reason: `load_test.py` alone primes the cache on its first loop and
  every subsequent loop is served from Redis, so the model saw **9 forward
  passes against 6886 requests** and there was nothing to plot.
- **The model-inference drop at the right edge is the cache, not
  recovery.** Cache hit rate rises into the 80s over the same interval; the
  batcher was simply handling less work per batch.
- **Request Rate was blank until a dashboard bug was fixed.** Panel 1
  carried a saved `hideSeriesFrom` override pinned to
  `instance="host.docker.internal:8000"`. Containerizing the app in Phase
  7b changed that label to `app:8000`, so an exclude-all-but-that-instance
  override hid every series from the plot while the legend still listed
  them — a panel that looks like "no data" and is really "no data
  *matching a filter you cannot see*". Worth noting as a class of bug:
  moving a deployment renames labels, and dashboards store label matchers.

## Phase 6: fixing the benchmarks — four wrong conclusions

Phases 3–5 produced findings that were measurement artifacts, and fixing
the harness exposed a real performance bug that the broken harness had been
hiding. They are documented here because the artifacts are more instructive
than the fixes, and because several had already been written into code
comments as settled fact.

### "ONNX INT8 is 1.45x *slower* than PyTorch"

**Real cause: padding mismatch, not the runtime.** `app/model_onnx.py`
tokenized with `padding="max_length", max_length=128`. The HF pipeline in
`app/model.py` pads dynamically to the longest input in the batch — about
13 tokens for a one-sentence review. Every ONNX request was doing ~10x the
work of the PyTorch request it was being compared against.

Measured, same machine, same 2 threads:

| | 128 tokens (padded) | 13 tokens (dynamic) |
|---|---|---|
| PyTorch FP32 | 65.8 ms | 16.4 ms |
| ONNX FP32 | 66.0 ms | 10.9 ms |
| ONNX INT8 | 25.8 ms | 4.9 ms |

One config value was 81% of ONNX's latency. Fixed by switching `_tokenize`
to `padding=True`; `MAX_SEQ_LEN` is now a truncation cap only. Verified
after the fix:

- `benchmark_onnx.py` (sequential, interleaved, over HTTP): **20.9 ms →
  8.2 ms = 2.56x** median, **43.9 → 9.8 ms = 4.46x** at p99.
- `benchmark_batch_sizes.py` (in-process, batch 1): PyTorch 18.5 ms → INT8
  4.6 ms = **3.99x**; and against the *same graph* in FP32, 11.7 → 4.6 ms =
  **2.52x**. That second ratio is the quantization win in isolation, which
  the old benchmark had no arm to measure. It holds at **2.36–2.52x** across
  every batch size.

The explanation that had been committed for the original result — "a known
result on CPUs without AVX-512 VNNI; dynamic quantization dequantizes
weights back to FP32 before the multiply" — was false on both counts. This
CPU (i5-12450H) has AVX-VNNI; only AVX-512 is fused off. MLAS uses it.
`benchmark_onnx.py` no longer asserts a cause; it prints an ordered
checklist of what to rule out.

### "ONNX batching crashes, so quantization must be buggy"

**Real cause: the specialized batch axis from Phase 3, in the FP32 graph
too.** `predict_batch` with 2+ texts failed in `matmul_helper.h:61`. It was
attributed to a `DynamicQuantizeMatMul` fusion bug, but the FP32 graph
fails at the same node, so quantization was never involved. The size-1 dim
had been folded into a `Concat([1], Shape(mask)[1:2], [-1])` feeding the
attention output reshape, so batch 2 reshaped `(2,128,12,64)` into
`(1,128,1536)` and the following `MatMul` against `(768,768)` had nowhere
to go. Fixed in `export_onnx.py` (two-row dummy, `opset_version=18` —
`14` was a silent no-op — plus the symbolic-dim assertion). `app/main.py`
now probes a 2-row batch at startup and falls back to PyTorch with
remediation instructions rather than serving a graph that will fail on the
first real batch. `/health` reports which backend actually won the probe.

Verified: batch-of-8 agrees with 8 separate single calls on all labels, max
score drift 0.0432.

### "Batching is slower on this hardware"

Two separate things, and only one was real.

**Artifact:** the 3.1x throughput drop (58.7 → 18.8 req/s) was not
batching. `app/batcher.py` did `await self._run_batch(batch)` inline in its
worker loop, so exactly one batch was ever in flight, while
`/predict_unbatched` — a sync `def` — got FastAPI's 40-worker anyio
threadpool. That is 40-way concurrency against 1-way, labelled "the
batching effect". The batcher now dispatches with `create_task` under a
`Semaphore(4)`. After the fix `load_test_nocache.py` measures batcher vs
no-batcher at **313.2 vs 314.6 req/s = 1.00x**. The regression is gone.

**Real, and it survives the fix:** batching still buys nothing on this CPU.
Throughput is a wash and tail latency is *worse* (server-side p99 107 ms
batched vs 61 ms unbatched, since the 20 ms batch window is pure added
delay). Per-item INT8 cost is 4.6 ms at batch 1, 3.0 ms at batch 4, 3.1 ms
at batch 8, and 5.7 ms at batch 16 — flat, then worse once the larger GEMM
stops fitting cache. A CPU GEMM is already compute-bound at batch 1, so
there is no kernel-launch overhead to amortize the way there is on a GPU.

So the original conclusion was right by accident, for the wrong reason and
at the wrong magnitude (1.0x, not 0.32x). What the batcher still provides
is a *bounded* number of in-flight model calls — 4 batches × 2 intra-op
threads = 8 threads on 8 cores, instead of 40 anyio threads × 2 = 80
oversubscribing them. That is a thread- and memory-safety property, not a
speedup. On a GPU the same code is where batching would pay.
`load_test_nocache.py` now computes this verdict from the measurement
instead of asserting an expectation.

### The bug the broken benchmarks were hiding: Redis with no connection pool

Fixing the harness surfaced a genuine performance bug that the noise had
concealed. `app/cache.py`'s `_client()` was:

```python
def _client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=1)
```

called on *every* `get()` and `set()`. `from_url()` builds a new client and
a new `ConnectionPool`, so each cache operation opened a fresh TCP
connection, completed the handshake, ran one command, and discarded it:

| | median |
|---|---|
| new client per `get` | 9.58 ms |
| one reused client | 0.31 ms |

**31x.** End to end a cache hit went from **14.6 ms to 0.85 ms** server-side
(17x). Since an INT8 inference is 8.2 ms, the cache had been *slower than
the model it was caching* — every hit was a pessimization, and
`load_test.py` faithfully reported the cached path at 0.68x the uncached
one. The client is now a lazily-built module-level singleton;
`ConnectionPool` is thread-safe, which matters because `main.py` calls these
through `asyncio.to_thread`.

This is why the "cache makes it slower" reading was worth chasing rather
than explaining away: it was true, and for a fixable reason.

### Also fixed in the test scripts

- **`load_test.py`** — sent `SAMPLE_TEXTS[0]` for all 200 requests (4 of 5
  samples were dead code). After Phase 4 that meant `/predict` served 199
  cache hits while `/predict_unbatched` ran 200 forward passes, and the
  result was printed as "before vs after batching". Now rotates the texts,
  adds an ONNX arm, and separates cold-cache from warm-cache arms.
- **Client vs server latency** — the load tests reported only
  client-observed latency. At 50-concurrency against single-process uvicorn
  that is almost entirely queueing: cache hits measured 132 ms at the
  client and 1.06 ms in the handler, a 125x gap. Both are now printed,
  alongside a Little's-law line (`concurrency / throughput`) that shows when
  the client number is just queueing delay. This also means throughput
  saturates on the event loop — HTTP parsing and pydantic validation for 50
  connections on one thread — so once a handler drops below a few ms,
  req/sec stops tracking how fast the code is.
- **Arms bleeding into each other** — the arms ran back-to-back, so one
  arm's detached `cache.set` tasks and in-flight batches landed inside the
  next arm's timed window. The warm-cache arm measured **81 req/s**
  immediately after the cold arm and **266 req/s** run on its own: same
  code, same cache state, 3.3x apart. There is now a 3-second settle
  between arms.
- **Saturation ratios quoted as speedups** — with server-side latency now
  printed, PyTorch vs ONNX under 50-way load reads as ~52x. That is a
  *saturation* figure: a handler's own measured time includes being
  descheduled, and PyTorch degrades far worse than ONNX under 80 threads on
  8 cores. The per-call speedup is 2.56x. The summary says so explicitly
  rather than letting the bigger number be quoted.
- **`load_test_nocache.py`** — made texts unique with a `uuid4()`, which
  the wordpiece tokenizer splits into 27 subword tokens. 42 tokens/request
  versus 13 for a real review: 3.2x the intended workload, and
  non-comparable with `load_test.py` even though the docstring invited the
  comparison. Uniqueness now comes from a trailing index (1–2 tokens).
- **`benchmark_onnx.py`** — ran all of arm A then all of arm B, so thermal
  drift and P/E-core placement landed entirely on whichever ran second.
  Arms are now interleaved request-by-request.
- **Repeat count** — `REPEATS` 5 → 15. This is a hybrid CPU (4 P-cores +
  4 E-cores); a 2-thread job landing on E-cores runs at roughly half
  speed, which is how one identical PyTorch call was recorded as 17.8 ms
  in one phase and 31.7 ms in another. The benchmarks now print a `±%`
  spread column so scheduler luck is visible instead of being mistaken for
  a regression.
- **Export size math** — a >2 GB-capable export writes weights to a
  `.onnx.data` sidecar, so measuring the `.onnx` stub alone reported the
  quantized model as *8578% larger*. Now sums both files.
- **Percentile indices** — `int(n * 0.99)` could index off the end for
  small `n`; clamped in all three load tests.
- **Hardcoded interpretations** — every "what this means" block that
  asserted a cause now computes its verdict from the run, including the
  optimal batch size and whether batching paid off.

### The lesson

Almost every one of these was a *comparison* bug, not a model bug. Padding
mismatch, cache-vs-inference, 1-way-vs-40-way concurrency, queueing delay
read as service time, arms bleeding into each other, and E-core placement
each produced a plausible number that survived because nothing in the
harness printed the thing that would have contradicted it. Two of them had
been written up as hardware limitations.

The one real bug — Redis reconnecting on every operation, a 31x cost — was
invisible for two phases *because* the harness was noisy enough to absorb
it. Fixing measurement first is what made it findable.

The benchmarks now print sequence length, per-item cost, measurement
spread, server-side *and* client-side latency, a queueing sanity check, and
which code path actually served each request. Where a script used to assert
a cause, it lists what to check.

The Grafana dashboard in Phase 5 is the same failure mode with a different
mechanism. Nothing there is mismeasured — the counters are exact — but four
of the five panels report a *quantile*, and a quantile inherits the shape of
its bucket layout and its rate window. 7.80 batch-p95 and 74.4 ms
inference-p99 are both bucket-edge arithmetic over 7 samples; the 50/95/99 ms
latency lines are the first bucket's edges; 1.75 req/s is a 2-second burst
divided by a 60-second window. The numbers that held up — 1693/1693 2xx, 810
requests served by 7 forward passes, 93.8% cumulative hit rate — are all
counters and ratios of counters. A dashboard makes a system observable; it
does not make its own readings true.

### Current numbers, all verified on this box

| Path | server-side p50 | note |
|---|---|---|
| PyTorch FP32, single request | 20.9 ms | baseline |
| ONNX INT8, single request | 8.2 ms | 2.56x vs PyTorch |
| ONNX INT8, best batch (4) | 3.0 ms/item | vs 4.6 at batch 1 |
| Cache hit | 0.85 ms | ~10x vs inference |
| Model size | 256.1 → 64.7 MB | 4.0x smaller |

From the Grafana session (two `load_test.py` runs, 1693 requests, all 2xx) —
counter-derived figures only, since the quantile panels are bucket artifacts:

| Measure | Value | Source |
|---|---|---|
| Cache hit rate, cumulative | 93.8% | 760 hits / 810 `/predict` |
| Forward passes for 810 requests | 7 | `ml_batch_size_count` |
| Mean batch size | 7.14 / cap 8 | `ml_batch_size_sum / _count` |
| Model inference, per batch | 49.1 ms | `_sum / _count`, 7 batches |
| Model inference, per item | 6.9 ms | 49.1 / 7.14 |
| `/predict` mean, end to end | 22.98 ms | 810 requests |
| `/predict_onnx_unbatched` mean | 68.47 ms | 400 requests, 50-way load |
| `/predict_unbatched` mean | 522.51 ms | 400 requests, 50-way load |
| Errors | 0 / 1693 | `http_requests_total` by status |

## Phase 7a: circuit breaker

When the model backend starts failing for real — a dead ONNX session, OOM, a
graph swapped out from under the process — the worst thing the server can do is
keep feeding it work. Every doomed request still costs a queue slot, a batch
collection window, a semaphore permit and a threadpool hop before it fails, so a
broken backend turns into a growing backlog of requests that are all going to
error anyway.

`app/circuit_breaker.py` stops that. Five consecutive failed batches and the
server stops calling the backend at all; after a 30-second cooldown, exactly one
request is let through to find out whether it recovered.

```
CLOSED ──5 consecutive failed batches──► OPEN ──30s elapsed──► HALF_OPEN
   ▲                                       ▲                      │
   └───────────── probe succeeds ──────────┴── probe fails ────────┘
```

### The check is at admission, not around the model call

Both build plans specify the textbook shape — `breaker.call(fn, *args)` wrapping
the protected call, which puts the check inside `InferenceBatcher._run_batch()`.
That call site cannot deliver what the same documents ask for a few paragraphs
later: `/predict` returning 503 "immediately (not after a full batch timeout)".
By the time `_run_batch()` runs, the request has already sat in the queue and
paid up to the 20 ms collection window. A rejection that costs 20 ms and still
lets the queue grow is not fast-failing.

So the breaker is a gate, not a wrapper, and permission and outcome are separate
calls because they happen at different points in a request's life:

- `predict()` in `app/batcher.py` calls `check()` **before** `queue.put()`
- `_run_batch()` calls `record_success()` / `record_failure()` once the outcome
  is actually known

Measured cost of a rejection on that path: **0.015 ms**, against the 20 ms window
it would have paid at the other call site. That assertion lives in the test suite
rather than in this file, because it is the one measurement that would catch the
check drifting back into `_run_batch()`.

### HALF_OPEN admits exactly one request

The plan doc's state diagram says "let ONE request through as a probe" and its
code does not do that: once the `state` property flips OPEN→HALF_OPEN, every
subsequent caller sees a non-OPEN state and proceeds. At any real request rate
that stampedes a backend which has just been given one chance. The fix is an
explicit in-flight token, claimed by whichever request becomes the probe.

`_refresh()` — the time-based OPEN→HALF_OPEN flip — is deliberately called by
`status()` as well, but `status()` never claims the token. Otherwise polling
`/health` would consume the one probe the cooldown just earned, and the backend
would never actually get tested.

Because everyone else is rejected while HALF_OPEN, `_collect_batch()` finds a
single item: the probe batch is naturally size 1, risking one request rather
than eight.

### The threshold counts batches, not requests

One batch is one `predict_fn` call, so `failure_threshold=5` is five consecutive
failed *batches* — up to 40 failed requests at `batch_size=8`. Counting
per-request would open the breaker on one bad batch, which is too twitchy for a
batching server.

Two limits worth stating rather than engineering around:

1. **It is a consecutive-failure breaker.** Any success resets the count, so a
   backend failing 50% of the time will never trip it. Catching that needs a
   rolling error-rate window — a different design, and not what either build plan
   specifies.
2. **`/predict_unbatched` and `/predict_onnx_unbatched` stay unprotected.** They
   bypass the batcher by design; they are the documented baseline arms for the
   benchmarks above, not serving paths.

### Cached predictions survive an open breaker

`/predict` is cache-then-batcher, and the gate lives inside `batcher.predict()`,
so cache hits never reach it. When the model is down the server keeps answering
everything it has already answered, and only misses get shed.

### No lock

`check()`, `record_success()` and `record_failure()` all run on the single
asyncio event loop and none of them contains an `await`, so each runs to
completion without yielding and they are already atomic with respect to each
other. The plan doc carries an `asyncio.Lock`; it implies a race that cannot
happen. The one real await on this path — `asyncio.to_thread(predict_fn, ...)` —
is *before* the bookkeeping, not inside it.

Relatedly, `asyncio.CancelledError` is a `BaseException`, so `_run_batch()`'s
`except Exception` does not catch it, and `stop()` cancelling the worker at
shutdown is therefore not recorded as a model failure. That is load-bearing,
which is why there is a comment there telling the next reader not to widen it.

### Nothing in `/predict` changed

`CircuitOpenError` subclasses `RuntimeError`, and `/predict` already had

```python
try:
    result = await batcher.predict(request.text)
except RuntimeError as e:
    raise HTTPException(status_code=503, detail=str(e))
```

so the 503 path needed no edit at all — the plan doc lists it as a modification
and it isn't one. Being its own class still lets tests tell a breaker rejection
apart from a genuine batcher error.

### Testing it without editing production code

The plan doc's testing note concedes its own approach doesn't work: "the model
won't actually fail on ordinary bad input — temporarily set `failure_threshold=1`
and manually raise Exception in `_run_batch`, then revert." That needs hand edits
to shipping code and proves nothing repeatable.

`test_circuit_breaker.py` injects a failing `predict_fn` instead — no server, no
temporary edits, and no permanent test hook that could trip in production:

```bash
python test_circuit_breaker.py
```

1. **Unit** — the state machine against a fast breaker (`failure_threshold=2`,
   `cooldown_seconds=0.5`): opens at threshold, rejects while OPEN, cooldown
   yields HALF_OPEN, `status()` polling does not consume the probe, a second
   concurrent caller is rejected while the probe is in flight, probe success
   closes it, probe failure reopens it and restarts the cooldown.
2. **Integration** — a real `InferenceBatcher` with a `predict_fn` that raises.
   Drives `.predict()` until it opens, asserts the backend was called exactly
   twice and not once more after that, and times the rejection against the 20 ms
   window. Then swaps in a working `predict_fn` and confirms exactly one of two
   concurrent requests becomes the probe and closes the breaker.
3. **Live** — `/health`, `/predict` and `/metrics` against a running server;
   skipped with a printed note if port 8000 is not answering, so the script is
   useful either way.

`/health` gains the breaker's state, and it is on the metrics path too:

```json
"circuit_breaker": {"state": "closed", "failure_count": 0, "threshold": 5,
                    "cooldown_seconds": 30, "rejections": 0}
```

```
ml_circuit_breaker_state 0.0              # 0=closed, 1=half_open, 2=open
ml_circuit_breaker_rejections_total 0.0
```

The gauge exists because `/health` only reports when something polls it: a
breaker that opened and recovered between two polls leaves no trace there, which
is exactly the class of invisible event phase 5 existed to fix. `rejections` is
kept separate from the 503s in `http_requests_total` — a request rejected here
never reached the model, so it is load shed on purpose rather than a server
error. A Grafana panel for the gauge is not wired up; that belongs with the
provisioning-as-code work still open from 7b.

### What was not re-measured

The breaker adds one branch per uncached request, and `load_test_nocache.py`
cannot resolve that on this box. Two host runs after the change, plus the
container:

| Run | ONNX INT8, no batcher | ONNX INT8 + batcher | verdict |
|---|---|---|---|
| host 1 | 108.0 req/s | 150.4 req/s | 1.39x |
| host 2 | 124.7 req/s | 87.8 req/s | 0.70x |
| container | 91.9 req/s | 110.5 req/s | 1.20x |

The batched arm moves 87.8 → 150.4 across two runs of the same script at N=200,
so the noise floor here is far wider than a branch. The unbatched arm carries no
breaker code at all and swings by the same order, which is the isolation
argument that matters: whatever is moving these numbers is not the breaker.

All three rows are also well below the 313.2 / 314.6 req/s recorded in phase 6
above, on code whose ONNX path is unchanged by this phase — so that gap is
machine state, not this change, and none of these numbers are a re-baseline.
Re-baselining stays deferred, as it was in 7b.

## Phase 7b: containerization

`docker compose up -d --build` starts the whole stack. Before this phase, running
the project meant three separate things — `docker run` for Redis, a partial
compose file for Prometheus and Grafana, and `uvicorn` on the host — and the
server being outside Docker is the only reason `prometheus.yml` had to scrape
`host.docker.internal:8000`. It now targets the compose service name `app:8000`.

| Service | Port | Image |
|---|---|---|
| `app` | 8000 | built from `Dockerfile` — 975 MB |
| `redis` | 6379 | `redis:7-alpine` |
| `prometheus` | 9090 | `prom/prometheus:v2.51.0` |
| `grafana` | 3000 | `grafana/grafana:10.4.0` |

### The ONNX export runs during `docker build`

`models/` is gitignored, so a clean clone has no graph — and what the server does
about that is the actual problem. `app/main.py:53` is

```python
ONNX_AVAILABLE = Path("models/model_int8.onnx").exists()
```

A missing graph is therefore not an error. The server starts, reports healthy,
and serves every request on the PyTorch backend at 2.56x the latency, with
nothing anywhere saying so. The same happens if the process is launched from the
wrong working directory, because that path is relative — which is why the
runtime stage pins `WORKDIR /app` and keeps the graph at `/app/models/`.

So the builder stage runs `export_onnx.py` and the runtime stage copies the
result out. Two things follow:

- **The export's own checks become build failures.** The script requires the
  logits batch axis to be symbolic and runs real batch-1 and batch-4
  predictions, exiting non-zero on either. The phase 3 bug — a graph specialized
  to batch 1, which passes a batch-1 smoke test and dies at batch 2 — can no
  longer be built into an image. From this build:

  ```
  logits output shape: ['batch_size', 2]
  batch=1: shape=(1, 2)  labels=['POSITIVE']  OK
  batch=4: shape=(4, 2)  labels=['POSITIVE', 'NEGATIVE', 'POSITIVE', 'NEGATIVE']  OK
  Done. 256.2 MB → 64.7 MB  (75% smaller, 4.0x)
  ```

- **Only the INT8 graph ships.** `app/model_onnx.py` never opens the FP32 one —
  that is `benchmark_batch_sizes.py`'s comparison arm — so 256 MB of
  `model_fp32.onnx.data` stays behind in the builder. The trade-off is that that
  benchmark's FP32 arm cannot run inside the container.

`.dockerignore` excludes `models/` alongside `venv/`, so a local build takes the
same path a clean clone does instead of silently picking up the host's graphs.
The Hugging Face snapshot is baked in at `HF_HOME=/opt/hf` with
`HF_HUB_OFFLINE=1`, so startup does no network I/O and a cache-layout mistake
fails loudly rather than re-downloading 260 MB per container start.

### torch, installed twice, in that order

`pip install torch` on Linux resolves to the CUDA build: roughly 2.5 GB of
`nvidia-*` wheels for a model capped at two threads by `torch.set_num_threads(2)`.
The Dockerfile installs `torch==2.12.1` from `download.pytorch.org/whl/cpu`
**before** the lock file, as its own step, because both indexes publish 2.12.1
and letting pip choose between them is not deterministic. The lock file then
finds torch already satisfied and leaves it alone. The container confirms it:
`torch 2.12.1+cpu`.

torch cannot just be dropped to shrink the image. `app/main.py` imports
`app.model` unconditionally — it backs `/predict_unbatched`, the baseline arm in
every benchmark here, and it is the fallback when the batch probe fails.

Versions are pinned in `requirements.lock.txt`; `requirements.txt` stays unpinned
for host work. This matters because the phase 6 findings are properties of
specific versions rather than of the code: the padding fix rests on tokenizer
behaviour, the batch-axis bug is `torch.onnx.export`'s `dynamo=True` default, and
the `quantize_dynamic` shape-inference conflict is onnxruntime 1.27.

### One worker, deliberately

The container runs a single uvicorn worker. The batcher is in-process, so N
workers means N independent batchers each filling at 1/N the request rate, and
N × `Semaphore(4)` × 2 intra-op threads competing for 12 — which is the
oversubscription phase 6 just removed. Multiple workers would also split the
Prometheus registry, so `/metrics` would describe one arbitrary worker and every
counter in the phase 5 writeup would silently become a fraction.

The known cost is the event-loop ceiling phase 6 measured: HTTP parsing and
pydantic validation for every connection on one thread, which is why req/sec
stops tracking code speed once a handler drops below a few ms. Scaling that out
needs `PROMETHEUS_MULTIPROC_DIR` and a batcher that does not assume one process.
Both are deferred rather than guessed at.

### The startup log is the verification step

All three ways this can fail quietly show up here and nowhere else — each of them
serves traffic successfully, so a 200 from `/health` proves nothing:

```
[startup] ONNX INT8 backend detected                 <- relative path resolved
[startup] /predict = onnx_int8 + batching (<=4 concurrent batches) + Redis cache
          + circuit breaker                          <- 2-row batch probe passed
[startup] Redis connected OK                         <- reached redis:6379
```

`ONNX model not found` means the graph or the working directory is wrong.
`ONNX batch probe FAILED` means the graph is batch-specialized. `WARNING: Redis
not reachable` means the cache is failing open and every request is going to the
model — the phase 4 regression, in a form that looks like a healthy server.

`/health` then reports `"backend": "onnx_int8 (batcher)"`, and the metrics
pipeline can be checked end to end: three identical POSTs to `/predict` produced
`ml_cache_misses_total 1`, `ml_batch_size_count 1`, `ml_cache_hits_total 2` in
Prometheus, with the target at `http://app:8000/metrics`, `up`.

### Two caveats on numbers from the container

Nothing in this phase was re-baselined, and the earlier sections' figures are
host measurements. Two reasons not to compare them directly:

1. **Docker Desktop on Windows runs under WSL2**, a VM with its own scheduling
   and network stack. Server-side, a cache hit in the container measures 0.70 ms
   against 0.85 ms on the host, and a cold `/predict` 32.6 ms — the same order,
   but not the same machine.
2. **The published-port hop is not free.** `test_cache.py` from the host passes
   (6.42x hit speedup, hit counters agreeing) but reports hits at ~7 ms
   client-side for what the handler itself measured at 0.7 ms. That gap is the
   port forward, and it is exactly the client-vs-server confusion phase 6 added
   both numbers to make visible.

Re-baselining inside the container, and parameterizing the five test scripts'
hardcoded `127.0.0.1:8000`, are not part of this phase.

### Grafana state now survives a teardown

The pre-phase-7 Grafana container had no volume, so the phase 5 dashboard lived
in its writable layer and any `docker rm` would have destroyed it. Compose now
mounts a named `grafana-data` volume; the dashboard and its datasource were
exported over the API and re-imported, and verified to survive
`docker compose down && docker compose up -d`. Provisioning them from files —
and applying phase 5's five bucket and rate-window fixes — is still open.

### Running everything

```bash
docker compose up -d --build
docker compose logs -f app        # watch the backend probe resolve
```

Grafana at `localhost:3000` (admin/admin), Prometheus at `localhost:9090`,
the server at `localhost:8000`.

The benchmarks and load tests still run from the host against the published
port, in the venv:

```bash
python benchmark_batch_sizes.py   # in-process: PyTorch vs INT8 vs FP32, per batch size
python benchmark_onnx.py          # over HTTP, interleaved, no batcher, no cache
python load_test.py               # concurrency: uncached, cold cache, warm cache
python load_test_nocache.py       # batching in isolation, cache bypassed
python test_cache.py              # cache hit rate + relative speedup
python test_circuit_breaker.py    # breaker state machine; last section needs the server
```

`benchmark_batch_sizes.py` runs the model in-process rather than over HTTP, so it
needs the venv and both graphs on the host — including the FP32 one the container
image omits. For a fast edit loop on the server itself, `uvicorn app.main:app
--reload` on the host still works; switch `prometheus.yml` to the
`host.docker.internal:8000` target kept commented there.

The dashboard panels only carry signal while load is running, and a full
four-arm `load_test.py` run is over in about 30 seconds — two or three
scrapes, of which 9 s is the deliberate `SETTLE_S` idling between arms. That
is shorter than the 1-minute rate window on the request-rate panel and far
shorter than the 5-minute window on the rest. To read anything time-varying
off Grafana, loop the load test for longer than the widest rate window
first.

