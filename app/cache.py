"""
Redis cache layer — Phase 5 update.

Added: increments ml_cache_hits and ml_cache_misses Prometheus counters
on every get() call. Everything else unchanged from Phase 4.
"""

import hashlib
import json
import os

import redis
from app.metrics import cache_hits, cache_misses

REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL        = 3600
KEY_PREFIX = "ml:pred:"


def _client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=1)


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
            cache_hits.inc()       # ← NEW: increment hit counter
            return json.loads(raw)
        cache_misses.inc()         # ← NEW: increment miss counter
        return None
    except Exception:
        cache_misses.inc()         # ← Redis down counts as a miss
        return None


def set(text: str, result: dict) -> None:
    """Stores result in cache with TTL. Silently ignores failures."""
    try:
        _client().setex(_make_key(text), TTL, json.dumps(result))
    except Exception:
        pass


def stats() -> dict:
    """Returns cache stats for /health endpoint."""
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