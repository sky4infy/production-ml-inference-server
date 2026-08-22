# ML Inference Server

A FastAPI server that wraps a DistilBERT sentiment classifier and serves
predictions over HTTP: request batching, ONNX INT8 quantization, Redis
caching, and Prometheus/Grafana metrics — each phase measured before the
next was added.

Phases 1–2 below are the original build notes. **Phase 6 documents three
findings from phases 3–5 that turned out to be measurement artifacts**, and
what the harness now prints so they cannot recur; read it before trusting
any number in the earlier sections.

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

`/metrics` exposes request counts, latency histograms, cache hit/miss
counters, and a `batch_size_histogram`.

```bash
docker compose up -d          # prometheus + grafana
# Grafana at localhost:3000, Prometheus at localhost:9090
```

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

### Current numbers, all verified on this box

| Path | server-side p50 | note |
|---|---|---|
| PyTorch FP32, single request | 20.9 ms | baseline |
| ONNX INT8, single request | 8.2 ms | 2.56x vs PyTorch |
| ONNX INT8, best batch (4) | 3.0 ms/item | vs 4.6 at batch 1 |
| Cache hit | 0.85 ms | ~10x vs inference |
| Model size | 256.1 → 64.7 MB | 4.0x smaller |

### Running everything

```bash
uvicorn app.main:app --reload
```

```bash
python benchmark_batch_sizes.py   # in-process: PyTorch vs INT8 vs FP32, per batch size
python benchmark_onnx.py          # over HTTP, interleaved, no batcher, no cache
python load_test.py               # concurrency: uncached, cold cache, warm cache
python load_test_nocache.py       # batching in isolation, cache bypassed
python test_cache.py              # cache hit rate + relative speedup
```

