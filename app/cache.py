"""
Redis cache layer — Phase 4.

How it works:
- Input text is hashed with SHA-256 to make a short, fixed-length cache key
- Prediction result is stored as JSON with a 1-hour TTL
- On Redis failure (down, timeout), always returns None (cache miss)
  so the prediction path is NEVER broken by cache errors

This file is synchronous (redis-py default).
All calls from async endpoints go through asyncio.to_thread().
"""

import hashlib
import json
import os

import redis

REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL        = 3600          # 1 hour in seconds
KEY_PREFIX = "ml:pred:"   # namespace prefix — isolates our keys from others


def _client() -> redis.Redis:
    """Create a Redis client from URL. Called fresh each time to avoid
    stale connections — redis-py handles connection pooling internally."""
    return redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=1)


def _make_key(text: str) -> str:
    """SHA-256 hash of input text, prefixed.
    Why hash: raw text as a key is slow, variable-length, and leaks data.
    SHA-256 is deterministic — same text always gives same key."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{KEY_PREFIX}{digest}"


def get(text: str) -> dict | None:
    """
    Returns cached result dict if found, None if miss or Redis is down.
    Never raises — cache errors are always treated as misses.
    """
    try:
        raw = _client().get(_make_key(text))
        if raw:
            return json.loads(raw)
        return None
    except Exception:
        return None   # Redis down or timeout → treat as miss


def set(text: str, result: dict) -> None:
    """
    Stores result in cache with TTL.
    Silently ignores failures — a failed cache write is never a prediction error.
    """
    try:
        _client().setex(_make_key(text), TTL, json.dumps(result))
    except Exception:
        pass


def stats() -> dict:
    """
    Returns cache statistics for the /health endpoint.
    hit_rate is a rolling counter from Redis INFO — resets on Redis restart.
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