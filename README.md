# ML Inference Server — Phase 1...

A FastAPI server that wraps a sentiment classification model and serves
predictions over HTTP. This is the foundation phase: get it running and
measured before adding batching, caching, or metrics.

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
both. **The req/sec difference between the two is your real "before vs
after batching" number** — write it down, it's the centerpiece of your
resume bullet.

If the numbers come back close to each other rather than clearly
different, that's worth flagging rather than ignoring — it usually means
the batch window or batch size needs tuning, not that something is
broken. Come back with the actual numbers either way and we'll look at
them together.
