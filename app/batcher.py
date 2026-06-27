"""
The batcher. This is the piece that turns N separate requests into 1
model call, while still giving each caller back only their own result.

Two asyncio concepts make this work, both worth understanding (you'll
likely get asked about them in an interview if you mention batching):

1. asyncio.Queue - a thread-safe-for-asyncio queue. Incoming requests are
   dropped in here; the worker loop below pulls them out.

2. asyncio.Future - a "promise" for a result that doesn't exist yet. Each
   caller gets their own Future and awaits it. The worker fills in that
   exact Future once the batch comes back, which is what lets caller A
   get only caller A's result even though A, B, and C's text all went
   through the model together.
"""

import asyncio
import time


class InferenceBatcher:
    def __init__(self, predict_fn, batch_size: int = 8, timeout_ms: int = 20):
        """
        predict_fn: a function taking list[str] -> list[dict], i.e.
                    model.predict_batch. Kept as a parameter (not imported
                    directly) so this class doesn't need to know anything
                    about sentiment analysis specifically - it would work
                    for any batchable model.
        batch_size: max requests to group into one model call.
        timeout_ms: max time to wait for a batch to fill up before running
                    it anyway, even if it's not full.
        """
        self.predict_fn = predict_fn
        self.batch_size = batch_size
        self.timeout_s = timeout_ms / 1000
        self.queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def start(self):
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()

    async def predict(self, text: str) -> dict:
        """
        Called by the API endpoint. Drops the request in the queue and
        waits for the worker loop to fill in the result. From the
        caller's point of view this looks just like calling the model
        directly - the batching is invisible to them.
        """
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((text, future))
        return await future

    async def _worker_loop(self):
        """
        Runs forever in the background. Each iteration: wait for the
        first request (however long that takes), then keep collecting
        more until either batch_size is hit or timeout_s has elapsed
        since that first request arrived - whichever comes first.
        """
        while True:
            batch = []

            # Block here until at least one request shows up.
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

        # IMPORTANT: predict_fn is a blocking, CPU-bound call (it's
        # running a real model). If we called it directly here with a
        # plain `await`, it would freeze the entire event loop - no other
        # request could even be ACCEPTED while it runs, which defeats the
        # whole point of being async. asyncio.to_thread runs it on a
        # separate thread so the event loop stays free to keep accepting
        # and queueing new requests while this batch is being computed.
        try:
            results = await asyncio.to_thread(self.predict_fn, texts)
        except Exception as exc:
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return

        for (_, future), result in zip(batch, results):
            if not future.done():
                future.set_result(result)