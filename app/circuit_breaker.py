"""
Circuit breaker for the model inference path — Phase 7a.

--- What this is for ---

When the model backend starts failing for real — a dead ONNX session, OOM, a
graph that got swapped out from under the process — the worst thing the server
can do is keep feeding it work. Every doomed request still costs a queue slot, a
batch collection window, a semaphore permit and a threadpool hop before it
fails, so a broken backend turns into a growing backlog of requests that are all
going to 503 anyway.

The breaker short-circuits that: once the backend has failed enough times in a
row, stop calling it at all and reject immediately. After a cooldown, let exactly
ONE request through to find out whether it recovered.

    CLOSED ──5 consecutive failed batches──► OPEN ──30s elapsed──► HALF_OPEN
       ▲                                       ▲                      │
       └───────────── probe succeeds ──────────┴── probe fails ────────┘

--- Why the check is at admission, not around the model call ---

The textbook shape for this is a wrapper — `breaker.call(fn, *args)` around the
protected call. That would put the check inside InferenceBatcher._run_batch(),
which is too late to be useful here: by then the request has already sat in the
queue and paid up to timeout_ms (20ms) in the collection window. "Fast-fail"
that costs 20ms and still grows the queue is not fast-failing.

So this class is a gate, not a wrapper. app/batcher.py calls check() in
predict() before queue.put(), and record_success()/record_failure() in
_run_batch() once the outcome is actually known. Permission and outcome are
separate calls because they happen at different points in the request's life.

--- Why there is no lock ---

check(), record_success() and record_failure() all run on the single asyncio
event loop and none of them contain an `await`, so each runs to completion
without yielding and they are already atomic with respect to each other. An
asyncio.Lock here would imply a race that cannot happen. The one real await on
this path — asyncio.to_thread(predict_fn, ...) in _run_batch — happens *before*
the bookkeeping, not inside it.

--- Known limits, stated rather than engineered around ---

1. The counter is CONSECUTIVE failures: any success resets it to zero. That means
   a backend failing 50% of the time will never trip this breaker. Catching that
   needs a rolling error-rate window, which is a bigger design and is not what
   the build plan specifies.
2. The threshold counts BATCHES, not requests. One batch is one predict_fn call,
   so failure_threshold=5 is five consecutive failed batches — up to 40 failed
   requests at batch_size=8. Counting per-request would open the breaker on a
   single bad batch, which is too twitchy for a batching server.
3. A probe cancelled mid-flight leaves the probe token held. The only thing that
   cancels a batch here is InferenceBatcher.stop() at shutdown, where the process
   is going away regardless, so this is not worth a timeout mechanism.
"""

import time
from enum import Enum

from app.metrics import circuit_breaker_rejections, circuit_breaker_state


class State(Enum):
    CLOSED    = "closed"      # normal operation, failures counted
    OPEN      = "open"        # backend presumed down, everything rejected
    HALF_OPEN = "half_open"   # cooldown elapsed, one probe allowed through


# Gauge encoding. Prometheus gauges hold a float, so the state has to be
# projected onto numbers; ordering them by severity means a Grafana panel reads
# correctly as "higher is worse" without any value mapping.
_GAUGE_VALUE = {State.CLOSED: 0, State.HALF_OPEN: 1, State.OPEN: 2}


class CircuitOpenError(RuntimeError):
    """
    Raised by check() when the breaker will not admit a request.

    Subclassing RuntimeError is deliberate and load-bearing: app/main.py's
    /predict already wraps `await batcher.predict(...)` in
    `except RuntimeError -> HTTPException(503)`, so the 503 path needs no change
    at all. Being its own class still lets tests and callers tell a breaker
    rejection apart from a genuine batcher error.
    """


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds  = cooldown_seconds

        self._state           = State.CLOSED
        self._failure_count   = 0
        self._rejections      = 0
        # monotonic, not time.time(): the cooldown is a duration, and wall clock
        # can step backwards under an NTP correction or a DST change.
        self._opened_at       = 0.0
        self._probe_in_flight = False

        circuit_breaker_state.set(_GAUGE_VALUE[State.CLOSED])

    # ── State ────────────────────────────────────────────────────────────────
    def _set_state(self, new: State) -> None:
        """Single place that assigns state, so the gauge cannot drift from it."""
        self._state = new
        circuit_breaker_state.set(_GAUGE_VALUE[new])

    def _refresh(self) -> State:
        """
        Apply the time-based OPEN -> HALF_OPEN transition.

        Idempotent, and called by both check() and status(). Deliberately does
        NOT claim the probe token — otherwise polling /health would consume the
        one probe the cooldown just earned, and the backend would never actually
        get tested.
        """
        if (self._state is State.OPEN
                and time.monotonic() - self._opened_at >= self.cooldown_seconds):
            self._set_state(State.HALF_OPEN)
        return self._state

    @property
    def state(self) -> State:
        return self._refresh()

    # ── Gate ─────────────────────────────────────────────────────────────────
    def check(self) -> None:
        """
        Admit or reject a request. Raises CircuitOpenError to reject.

        Called before the request is queued, so a rejection costs a branch rather
        than a batch window.
        """
        state = self._refresh()

        if state is State.CLOSED:
            return

        if state is State.HALF_OPEN and not self._probe_in_flight:
            # This request becomes the probe. Claiming the token here is what
            # makes HALF_OPEN mean "one request", not "the floodgates reopen".
            self._probe_in_flight = True
            return

        self._rejections += 1
        circuit_breaker_rejections.inc()

        if state is State.HALF_OPEN:
            raise CircuitOpenError(
                "Circuit breaker HALF_OPEN — a recovery probe is already in "
                "flight, try again shortly"
            )

        wait = max(0.0, self.cooldown_seconds - (time.monotonic() - self._opened_at))
        raise CircuitOpenError(
            f"Circuit breaker OPEN — model backend failed "
            f"{self._failure_count} times in a row; retrying in {wait:.0f}s"
        )

    # ── Outcome ──────────────────────────────────────────────────────────────
    def record_success(self) -> None:
        """A backend call returned cleanly. Closes the breaker unconditionally."""
        self._failure_count   = 0
        self._probe_in_flight = False
        if self._state is not State.CLOSED:
            print(f"[circuit_breaker] {self._state.value} -> closed "
                  f"(backend recovered)")
            self._set_state(State.CLOSED)

    def record_failure(self) -> None:
        """A backend call raised. Opens the breaker once the threshold is hit."""
        self._failure_count  += 1
        was_probe             = self._probe_in_flight
        self._probe_in_flight = False

        # A failed probe goes straight back to OPEN and restarts the cooldown,
        # regardless of the count — HALF_OPEN exists precisely to test one
        # request, and it just failed.
        if was_probe or self._failure_count >= self.failure_threshold:
            if self._state is not State.OPEN:
                print(f"[circuit_breaker] {self._state.value} -> OPEN after "
                      f"{self._failure_count} consecutive failures; rejecting "
                      f"for {self.cooldown_seconds}s")
            self._set_state(State.OPEN)
            self._opened_at = time.monotonic()

    # ── Introspection ────────────────────────────────────────────────────────
    def status(self) -> dict:
        """Shape reported by /health. Safe to call from a sync endpoint."""
        state = self._refresh()
        out = {
            "state":            state.value,
            "failure_count":    self._failure_count,
            "threshold":        self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "rejections":       self._rejections,
        }
        if state is State.OPEN:
            out["seconds_until_probe"] = round(
                max(0.0, self.cooldown_seconds
                    - (time.monotonic() - self._opened_at)), 1)
        elif state is State.HALF_OPEN:
            out["probe_in_flight"] = self._probe_in_flight
        return out
