"""
Dynamic request batcher — Phase 5 update.

Added: records batch_size and model_inference_seconds as Prometheus
histograms on every batch. Everything else unchanged from Phase 4.
"""

import asyncio
import time

from app.metrics import batch_size_histogram, model_inference_seconds


class InferenceBatcher:
    def __init__(self, predict_fn, batch_size: int = 8, timeout_ms: int = 20):
        self.predict_fn   = predict_fn
        self.batch_size   = batch_size
        self.timeout_s    = timeout_ms / 1000
        self.queue        = asyncio.Queue()
        self._worker_task = None

    def start(self):
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()

    async def predict(self, text: str) -> dict:
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((text, future))
        return await future

    async def _worker_loop(self):
        while True:
            batch = []

            text, future = await self.queue.get()
            batch.append((text, future))

            deadline = time.monotonic() + self.timeout_s
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    text, future = await asyncio.wait_for(
                        self.queue.get(), timeout=remaining
                    )
                    batch.append((text, future))
                except asyncio.TimeoutError:
                    break

            await self._run_batch(batch)

    async def _run_batch(self, batch: list):
        texts = [text for text, _ in batch]

        # ── NEW: record batch size ───────────────────────────────────────────
        batch_size_histogram.observe(len(batch))

        # ── NEW: time the model call ─────────────────────────────────────────
        t_start = time.perf_counter()
        try:
            results = await asyncio.to_thread(self.predict_fn, texts)
            model_inference_seconds.observe(time.perf_counter() - t_start)
        except Exception as exc:
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return

        for (_, future), result in zip(batch, results):
            if not future.done():
                future.set_result(result)