"""
Verifies the Phase 7a circuit breaker.

Why this test is in-process instead of over HTTP:

  The build plan's own testing note conceded that its approach doesn't work —
  "the model won't actually fail on ordinary bad input, so a real test requires
  temporarily setting failure_threshold=1 and forcing an exception inside
  _run_batch, then reverting." That needs hand edits to production code before
  every run, proves nothing repeatably, and the alternative — leaving a
  fault-injection hook in the server — is a switch that can be flipped by
  accident in production.

  Neither is necessary. InferenceBatcher takes predict_fn and circuit_breaker as
  constructor arguments, so this file builds a real batcher around a deliberately
  exploding predict_fn and a fast breaker. That exercises the actual integration
  — the real _worker_loop, the real _run_batch, the real gate in predict() — with
  no server, no model load, and nothing to revert afterwards.

The one assertion worth pointing at: section 2 measures how long a rejection
takes. The gate is in predict() before queue.put(), so a rejection should cost
microseconds. If someone later moves the check into _run_batch (which is where
the textbook `breaker.call(...)` wrapper would put it), rejections start costing
the full 20ms collection window and that timing assertion fails. It is there to
catch exactly that regression, not to measure performance.

Run:
    python test_circuit_breaker.py          # sections 1 and 2 always run
                                            # section 3 runs if :8000 answers
"""

import asyncio
import sys
import time

from prometheus_client import REGISTRY

from app.batcher import InferenceBatcher
from app.circuit_breaker import CircuitBreaker, CircuitOpenError, State

BASE = "http://127.0.0.1:8000"

# Fast breaker for tests: 2 failures to open, half-second cooldown. The
# production values (5 / 30s) would make this script take minutes.
THRESHOLD = 2
COOLDOWN  = 0.5

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def gauge() -> float | None:
    return REGISTRY.get_sample_value("ml_circuit_breaker_state")


def rejections() -> float | None:
    return REGISTRY.get_sample_value("ml_circuit_breaker_rejections_total")


def raised_open(fn) -> bool:
    """True if fn() raised CircuitOpenError."""
    try:
        fn()
        return False
    except CircuitOpenError:
        return True


# ═════════════════════════════════════════════════════════════════════════════
# Section 1 — the state machine, in isolation
# ═════════════════════════════════════════════════════════════════════════════
def section_1_state_machine() -> None:
    print("\n" + "=" * 62)
    print("Section 1: state machine")
    print("=" * 62)

    cb = CircuitBreaker(failure_threshold=THRESHOLD, cooldown_seconds=COOLDOWN)

    check("starts CLOSED", cb.state is State.CLOSED)
    check("CLOSED admits requests", not raised_open(cb.check))
    check("gauge reads 0 when closed", gauge() == 0, f"gauge={gauge()}")

    # ── Consecutive, not cumulative ─────────────────────────────────────────
    cb.record_failure()
    check(f"1 failure of {THRESHOLD} stays CLOSED", cb.state is State.CLOSED,
          f"count={cb.status()['failure_count']}")
    cb.record_success()
    check("a success resets the counter", cb.status()["failure_count"] == 0)
    cb.record_failure()
    check("failure after reset is counted as the first, not the second",
          cb.state is State.CLOSED)
    cb.record_success()

    # ── Opening ─────────────────────────────────────────────────────────────
    for _ in range(THRESHOLD):
        cb.record_failure()
    check(f"{THRESHOLD} consecutive failures open it", cb.state is State.OPEN)
    check("gauge reads 2 when open", gauge() == 2, f"gauge={gauge()}")

    before = rejections()
    check("OPEN rejects requests", raised_open(cb.check))
    check("a rejection increments ml_circuit_breaker_rejections_total",
          rejections() == before + 1, f"{before} -> {rejections()}")
    check("status() reports seconds_until_probe while open",
          "seconds_until_probe" in cb.status(),
          str(cb.status().get("seconds_until_probe")))

    # ── status() must not consume the probe ─────────────────────────────────
    check("still OPEN before the cooldown elapses", cb.state is State.OPEN)
    time.sleep(COOLDOWN + 0.05)

    for _ in range(3):
        cb.status()          # polling /health repeatedly must not burn the probe
    check("cooldown transitions OPEN -> HALF_OPEN", cb.state is State.HALF_OPEN)
    check("gauge reads 1 when half-open", gauge() == 1, f"gauge={gauge()}")
    check("status() polling did not claim the probe",
          cb.status().get("probe_in_flight") is False)

    # ── Exactly one probe ───────────────────────────────────────────────────
    check("HALF_OPEN admits the first request as the probe",
          not raised_open(cb.check))
    check("HALF_OPEN rejects a second request while the probe is in flight",
          raised_open(cb.check))
    check("and a third", raised_open(cb.check))

    # ── Failed probe reopens and restarts the cooldown ──────────────────────
    cb.record_failure()
    check("a failed probe returns to OPEN", cb.state is State.OPEN)
    check("failed probe restarts the cooldown",
          cb.status()["seconds_until_probe"] > 0)

    # ── Successful probe closes it ──────────────────────────────────────────
    time.sleep(COOLDOWN + 0.05)
    check("cooldown elapses again", cb.state is State.HALF_OPEN)
    cb.check()
    cb.record_success()
    check("a successful probe closes it", cb.state is State.CLOSED)
    check("and zeroes the failure count", cb.status()["failure_count"] == 0)
    check("gauge back to 0", gauge() == 0, f"gauge={gauge()}")
    check("CLOSED admits again", not raised_open(cb.check))

    check("CircuitOpenError is a RuntimeError (so /predict returns 503, "
          "not 500)", issubclass(CircuitOpenError, RuntimeError))


# ═════════════════════════════════════════════════════════════════════════════
# Section 2 — a real InferenceBatcher around a failing backend
# ═════════════════════════════════════════════════════════════════════════════
class Exploder:
    """Stands in for a dead ONNX session. Counts how often it was called."""
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        raise RuntimeError("simulated backend failure")


def working(texts):
    return [{"label": "POSITIVE", "score": 0.99} for _ in texts]


async def section_2_integration() -> None:
    print("\n" + "=" * 62)
    print("Section 2: real batcher, failing backend")
    print("=" * 62)

    exploder = Exploder()
    cb = CircuitBreaker(failure_threshold=THRESHOLD, cooldown_seconds=COOLDOWN)
    batcher = InferenceBatcher(
        predict_fn=exploder,
        batch_size=8,
        timeout_ms=20,
        max_concurrent_batches=4,
        circuit_breaker=cb,
    )
    batcher.start()

    try:
        # Sequential, not concurrent: concurrent calls would be collected into
        # ONE batch and count as a single backend failure. THRESHOLD distinct
        # batches need THRESHOLD separate round trips.
        for i in range(THRESHOLD):
            try:
                await batcher.predict(f"failing request {i}")
                check(f"request {i} propagated the backend failure", False,
                      "it returned a result instead of raising")
            except CircuitOpenError:
                check(f"request {i} propagated the backend failure", False,
                      "rejected by the breaker too early")
            except RuntimeError as exc:
                check(f"request {i} propagated the backend failure",
                      "simulated backend failure" in str(exc))

        check(f"breaker opened after {THRESHOLD} failed batches",
              cb.state is State.OPEN, f"state={cb.status()['state']}")
        check(f"backend was called exactly {THRESHOLD} times",
              exploder.calls == THRESHOLD, f"calls={exploder.calls}")

        # ── The assertion that pins the gate to admission ───────────────────
        calls_before = exploder.calls
        t0 = time.perf_counter()
        rejected = False
        try:
            await batcher.predict("this should be rejected")
        except CircuitOpenError:
            rejected = True
        elapsed_ms = (time.perf_counter() - t0) * 1000

        check("open breaker rejects with CircuitOpenError", rejected)
        check("rejection did not reach the backend",
              exploder.calls == calls_before, f"calls={exploder.calls}")
        check("rejection is faster than the 20ms collection window "
              "(gate is at admission, not at the model call)",
              elapsed_ms < 5.0, f"{elapsed_ms:.3f}ms")

        # ── Recovery ────────────────────────────────────────────────────────
        batcher.predict_fn = working
        await asyncio.sleep(COOLDOWN + 0.05)
        check("cooldown moved it to HALF_OPEN", cb.state is State.HALF_OPEN)

        # Two at once: the first claims the probe, the second must be shed. This
        # is the stampede the build plan's version would have allowed.
        results = await asyncio.gather(
            batcher.predict("probe"),
            batcher.predict("should be shed"),
            return_exceptions=True,
        )
        succeeded = [r for r in results if isinstance(r, dict)]
        shed      = [r for r in results if isinstance(r, CircuitOpenError)]
        check("exactly one of two concurrent requests became the probe",
              len(succeeded) == 1, f"{len(succeeded)} succeeded")
        check("the other was shed, not queued",
              len(shed) == 1, f"{len(shed)} shed")

        check("successful probe closed the breaker", cb.state is State.CLOSED,
              f"state={cb.status()['state']}")

        out = await batcher.predict("normal traffic again")
        check("traffic flows again once closed", out["label"] == "POSITIVE")
    finally:
        await batcher.stop()


# ═════════════════════════════════════════════════════════════════════════════
# Section 3 — the live server, if one is up
# ═════════════════════════════════════════════════════════════════════════════
def section_3_live() -> None:
    print("\n" + "=" * 62)
    print("Section 3: live server")
    print("=" * 62)

    try:
        import requests
        health = requests.get(f"{BASE}/health", timeout=3).json()
    except Exception as exc:
        print(f"  SKIP  no server on {BASE} ({type(exc).__name__}) — sections 1 "
              f"and 2 cover the logic; start the stack with "
              f"`docker compose up -d --build` to run this one")
        return

    cbs = health.get("circuit_breaker")
    check("/health reports a circuit_breaker block", isinstance(cbs, dict),
          str(cbs))
    if isinstance(cbs, dict):
        check("breaker is closed on a healthy server",
              cbs.get("state") == "closed", f"state={cbs.get('state')}")
        check("threshold is the production value 5", cbs.get("threshold") == 5,
              f"threshold={cbs.get('threshold')}")
        check("cooldown is the production value 30s",
              cbs.get("cooldown_seconds") == 30,
              f"cooldown={cbs.get('cooldown_seconds')}")
        check("no rejections on a healthy server", cbs.get("rejections") == 0,
              f"rejections={cbs.get('rejections')}")

    r = requests.post(f"{BASE}/predict", json={"text": "still serving traffic"},
                      timeout=30)
    check("/predict still returns 200 with the breaker in the path",
          r.status_code == 200, f"status={r.status_code}")

    metrics = requests.get(f"{BASE}/metrics", timeout=5).text
    lines = [ln for ln in metrics.splitlines()
             if ln.startswith("ml_circuit_breaker")]
    check("ml_circuit_breaker_state is exposed at /metrics",
          any(ln.startswith("ml_circuit_breaker_state ") for ln in lines))
    check("ml_circuit_breaker_rejections_total is exposed at /metrics",
          any(ln.startswith("ml_circuit_breaker_rejections_total") for ln in lines))
    for ln in lines:
        print(f"        {ln}")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 62)
    print("Circuit Breaker Verification — Phase 7a")
    print("=" * 62)
    print(f"Test breaker: failure_threshold={THRESHOLD}, "
          f"cooldown_seconds={COOLDOWN}")
    print("Note: all CircuitBreaker instances share the one process-wide "
          "Prometheus gauge,\nso gauge assertions below read whichever breaker "
          "transitioned last. There is\nexactly one breaker in the server, "
          "where this is not a concern.")

    section_1_state_machine()
    asyncio.run(section_2_integration())
    section_3_live()

    print("\n" + "=" * 62)
    if _failures:
        print(f"{len(_failures)} FAILED:")
        for f in _failures:
            print(f"  - {f}")
        print("=" * 62)
        sys.exit(1)
    print("All checks passed.")
    print("=" * 62)
