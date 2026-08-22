"""
Dynamic request batcher.

Phase 6 fix — batch dispatch is now CONCURRENT.

The bug this fixes: _worker_loop used to `await self._run_batch(batch)`
inline, so the collector blocked until the model returned. That meant exactly
ONE batch was ever in flight. With batch_size=8 and 2 torch threads, that is
2 of 12 cores busy, and every queued request waits behind the current batch.

Meanwhile /predict_unbatched is a *sync* def, so FastAPI runs it in the anyio
threadpool (40 workers) and saturates all 12 cores. The load test was
therefore comparing 40-way parallelism against 1-way, and "batching" measured
3.1x SLOWER (18.8 vs 58.7 req/sec) — a scheduling artifact, not a property of
batching.

Now: the collector dispatches each batch as its own task and only blocks when
max_concurrent_batches are already running, which is real backpressure
instead of a hard serial bottleneck.
"""

import asyncio
import time

from app.metrics import batch_size_histogram, model_inference_seconds


class InferenceBatcher:
    def __init__(self, predict_fn, batch_size: int = 8, timeout_ms: int = 20,
                 max_concurrent_batches: int = 4):
        self.predict_fn = predict_fn
        self.batch_size = batch_size
        self.timeout_s  = timeout_ms / 1000
        self.queue      = asyncio.Queue()

        # How many batches may occupy the inference threadpool at once.
        # Each batch call uses ~2 intra-op threads (torch.set_num_threads(2) /
        # intra_op_num_threads=2), so 4 concurrent batches ~= 8 busy threads
        # on this 12-core box: enough to saturate it without thrashing.
        # Raising this past cpu_count/2 trades latency for no extra throughput.
        self.max_concurrent_batches = max_concurrent_batches

        self._sem         = asyncio.Semaphore(max_concurrent_batches)
        self._inflight    = set()
        self._worker_task = None

    def start(self):
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
        # Let batches already handed to the threadpool finish, so in-flight
        # requests get a real answer instead of a cancelled future on shutdown.
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)

    async def predict(self, text: str) -> dict:
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((text, future))
        return await future

    async def _collect_batch(self) -> list:
        """Block for the first item, then top up until full or timeout."""
        batch = [await self.queue.get()]

        deadline = time.monotonic() + self.timeout_s
        while len(batch) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.queue.get(),
                                                    timeout=remaining))
            except asyncio.TimeoutError:
                break
        return batch

    async def _worker_loop(self):
        while True:
            batch = await self._collect_batch()

            # Acquire BEFORE dispatch: if the threadpool is already full this
            # blocks the collector, which is what applies backpressure. The
            # alternative (spawn unconditionally) would let a burst create
            # unbounded tasks and blow out p99.
            await self._sem.acquire()

            task = asyncio.create_task(self._run_batch(batch))
            self._inflight.add(task)
            task.add_done_callback(self._on_batch_done)

    def _on_batch_done(self, task: asyncio.Task):
        self._inflight.discard(task)
        self._sem.release()

    async def _run_batch(self, batch: list):
        texts   = [text for text, _ in batch]
        futures = [future for _, future in batch]

        batch_size_histogram.observe(len(batch))

        t_start = time.perf_counter()
        try:
            results = await asyncio.to_thread(self.predict_fn, texts)
            model_inference_seconds.observe(time.perf_counter() - t_start)

            if len(results) != len(futures):
                raise RuntimeError(
                    f"predict_fn returned {len(results)} results for "
                    f"{len(futures)} inputs"
                )
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as exc:
            for future in futures:
                if not future.done():
                    future.set_exception(exc)
        finally:
            # Belt and braces: a future left pending here would hang its
            # request until the client timeout. The old code could do exactly
            # that if anything outside the try block raised.
            for future in futures:
                if not future.done():
                    future.set_exception(
                        RuntimeError("batch completed without producing a result")
                    )
