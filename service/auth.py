"""API-key authentication + per-key rate limiting, and the append-only
chain-of-custody audit log.

Kept dependency-light (in-memory token buckets, a local JSONL log) since
the blueprint's Extended-tier platform features ("API keys, rate
limiting, usage metering", "chain of custody... audit log") are cross-
cutting service concerns, not ML -- swap the in-memory pieces for Redis
+ a real datastore at production scale without touching the pipeline.
"""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Header, HTTPException, Request


# --- API keys ---------------------------------------------------------

def _load_valid_keys() -> set[str]:
    raw = os.environ.get("FORGELENS_API_KEYS", "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    return keys


_VALID_KEYS = _load_valid_keys()
AUTH_DISABLED = len(_VALID_KEYS) == 0  # dev-mode default: no keys configured


def _hash_key(key: str) -> str:
    """Never log or store the raw key -- only its hash, for audit
    correlation without credential exposure."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class TokenBucket:
    capacity: float
    refill_per_sec: float
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)

    def __post_init__(self):
        self.tokens = self.capacity

    def try_consume(self, cost: float = 1.0) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        self.last_refill = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class RateLimiter:
    def __init__(self, capacity: float = 30.0, refill_per_sec: float = 0.5):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(self.capacity, self.refill_per_sec)
                self._buckets[key] = bucket
            return bucket.try_consume(1.0)


_rate_limiter = RateLimiter(
    capacity=float(os.environ.get("FORGELENS_RATE_CAPACITY", 30)),
    refill_per_sec=float(os.environ.get("FORGELENS_RATE_PER_SEC", 0.5)),
)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency: validates the API key and enforces the
    per-key rate limit. Returns the key's hash for downstream audit
    logging (never the raw key)."""
    if AUTH_DISABLED:
        return "dev-mode"

    if not x_api_key or x_api_key not in _VALID_KEYS:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")

    key_hash = _hash_key(x_api_key)
    if not _rate_limiter.allow(key_hash):
        raise HTTPException(status_code=429,
                             detail="Rate limit exceeded. Slow down and retry shortly.")
    return key_hash


# --- Chain-of-custody audit log ---------------------------------------

_AUDIT_LOG_PATH = Path(os.environ.get("FORGELENS_AUDIT_LOG", "runs/audit_log.jsonl"))
_audit_lock = threading.Lock()


def record_audit_event(event: str, sha256: str | None, key_hash: str,
                        detail: dict | None = None):
    """Appends a single audit record. Append-only by construction (no
    update/delete path) so the log itself can serve as part of a chain-
    of-custody record: hash on receipt, log every analysis with
    timestamp and model version, per the blueprint's ethics section.
    """
    _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "event": event,
        "sha256": sha256,
        "api_key_hash": key_hash,
        "detail": detail or {},
    }
    with _audit_lock:
        with open(_AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
