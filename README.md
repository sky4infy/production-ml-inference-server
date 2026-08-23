# ML Inference Server

A FastAPI server that wraps a DistilBERT sentiment classifier and serves
predictions over HTTP: request batching, ONNX INT8 quantization, Redis
caching, and Prometheus/Grafana metrics — each phase measured before the
next was added.

Phases 1–2 below are the original build notes. **Phase 6 documents four
findings from phases 3–5 that turned out to be measurement artifacts**, and
what the harness now prints so they cannot recur; read it before trusting
any number in the earlier sections. Phase 5 gets the same treatment for the
Grafana dashboard, where four of the five panel values turn out to be
properties of the PromQL query rather than of the server.

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
docker run -d -p 6379:6379 redis:alpine
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
cache served and to both unbatched endpoints.

```bash
docker compose -f docker-compose.monitoring.yml up -d
# Grafana at localhost:3000 (admin/admin), Prometheus at localhost:9090
```

Prometheus runs in Docker and the server does not, so `prometheus.yml`
targets `host.docker.internal:8000` rather than `localhost` — inside a
container `localhost` is the container. Scrape interval 15 s, retention 7 d.
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
screenshots holds one value dead flat for five minutes and is empty for the
other 5h 55m. That is a 5-minute rate window republishing a single scrape's
delta at every step until the delta ages out. There is one measurement per
panel, so a flat line here cannot show that the model is not degrading, that
there is no memory leak, or that there is no thermal throttling. Phase 6's
E-core finding — one identical PyTorch call recorded at 17.8 ms in one phase
and 31.7 ms in another — is exactly the drift this chart is too sparse to
see.

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

### Running everything

```bash
docker run -d -p 6379:6379 --name redis-ml redis:alpine
docker compose -f docker-compose.monitoring.yml up -d
uvicorn app.main:app --reload
```

```bash
python benchmark_batch_sizes.py   # in-process: PyTorch vs INT8 vs FP32, per batch size
python benchmark_onnx.py          # over HTTP, interleaved, no batcher, no cache
python load_test.py               # concurrency: uncached, cold cache, warm cache
python load_test_nocache.py       # batching in isolation, cache bypassed
python test_cache.py              # cache hit rate + relative speedup
```

The dashboard panels only carry signal while load is running, and a full
four-arm `load_test.py` run is over in about 30 seconds — two or three
scrapes, of which 9 s is the deliberate `SETTLE_S` idling between arms. That
is shorter than the 1-minute rate window on the request-rate panel and far
shorter than the 5-minute window on the rest. To read anything time-varying
off Grafana, loop the load test for longer than the widest rate window
first.

