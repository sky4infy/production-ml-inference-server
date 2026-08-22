"""
Redis cache layer.

--- Phase 6 fix: reuse one client instead of building a new one per call ---

_client() used to be `return redis.Redis.from_url(...)`, called fresh on every
get() and set(). from_url() constructs a new Redis client AND a new
ConnectionPool, so each cache operation opened a brand-new TCP connection, did
the handshake, ran one command, and threw the connection away. Measured on this
box:

    new client per get :  9.58 ms
    reused client      :  0.31 ms      (31x)

An ONNX INT8 inference here costs 8.2 ms. So the cache was *slower than the
model it was caching* — every hit was a pessimization, and load_test.py duly
reported the cached path at 0.68x the uncached path. The connection pool is the
entire point of a redis-py client; it has to outlive the request.

The client is now a module-level singleton, built lazily. redis-py's
ConnectionPool is thread-safe, which matters because main.py calls these through
asyncio.to_thread. Reconnecting after a Redis restart is the pool's job, so the
fail-open behaviour below is unchanged: any exception is swallowed and counted
as a miss.
"""

import hashlib
import json
import os
import threading

import redis
from app.metrics import cache_hits, cache_misses

REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL        = 3600
KEY_PREFIX = "ml:pred:"

_client_singleton: redis.Redis | None = None
_client_lock = threading.Lock()


def _client() -> redis.Redis:
    """One process-wide client, so the connection pool is actually reused.

    max_connections is sized above the default asyncio.to_thread executor width
    (min(32, cpu_count+4)) so concurrent lookups don't serialize on the pool.
    health_check_interval lets the pool quietly replace a connection Redis has
    dropped instead of surfacing it as a miss.
    """
    global _client_singleton
    if _client_singleton is None:
        with _client_lock:
            if _client_singleton is None:          # re-check under the lock
                _client_singleton = redis.Redis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_timeout=1,
                    socket_connect_timeout=1,
                    max_connections=64,
                    health_check_interval=30,
                )
    return _client_singleton


def _make_key(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{KEY_PREFIX}{digest}"


def get(text: str) -> dict | None:
    """
    Returns cached result if found, None if miss or Redis down.
    Increments the appropriate Prometheus counter on every call.
    """
    try:
        raw = _client().get(_make_key(text))
        if raw:
            cache_hits.inc()
            return json.loads(raw)
        cache_misses.inc()
        return None
    except Exception:
        cache_misses.inc()         # Redis down counts as a miss - fail open
        return None


def set(text: str, result: dict) -> None:
    """Stores result in cache with TTL. Silently ignores failures."""
    try:
        _client().setex(_make_key(text), TTL, json.dumps(result))
    except Exception:
        pass


def stats() -> dict:
    """Returns cache stats for /health endpoint.

    NOTE: keyspace_hits/misses are Redis-SERVER-WIDE counters, not this app's —
    any other client touching this Redis instance moves them. They are still
    valid measured as a *delta* across a test window (which is how
    test_cache.py uses them), but don't read the absolute numbers as the
    application's hit rate. The app's own per-process counters are the
    ml_cache_hits / ml_cache_misses Prometheus metrics on /metrics.
    """
    try:
        info = _client().info("stats")
        hits   = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total  = hits + misses
        return {
            "connected" : True,
            "hits"      : hits,
            "misses"    : misses,
            "hit_rate"  : round(hits / total * 100, 1) if total > 0 else 0.0,
        }
    except Exception:
        return {"connected": False, "hits": 0, "misses": 0, "hit_rate": 0.0}
