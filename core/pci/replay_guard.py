"""
BIZRA PCI Replay Guard
═══════════════════════════════════════════════════════════════════════════════
Replay resistance via nonce tracking and timestamp validation.

PROTOCOL.md Section 7: Replay Resistance
- Nonce: 32 bytes, never reused
- Seen-nonce cache: TTL 120 seconds
- Timestamp window: ±120 seconds

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from core.pci.reject_codes import RejectCode

# Default configuration per PROTOCOL.md Section 7
DEFAULT_TTL_SECONDS = 120
DEFAULT_MAX_CLOCK_SKEW_SECONDS = 120
DEFAULT_MAX_CACHE_SIZE = 100_000


@dataclass(frozen=True)
class NonceEntry:
    """Record of a seen nonce with expiry time."""

    nonce: str
    envelope_digest: str
    seen_at: float  # Monotonic time
    expires_at: float  # Monotonic time


class ReplayGuard:
    """
    Replay resistance guard per PROTOCOL.md Section 7.

    Provides:
    - Nonce tracking with TTL-based expiration
    - Timestamp freshness validation
    - Thread-safe operations
    - LRU eviction when capacity exceeded
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_clock_skew_seconds: float = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
    ):
        """
        Initialize replay guard.

        Args:
            ttl_seconds: Time-to-live for seen nonces
            max_clock_skew_seconds: Maximum allowed clock skew
            max_cache_size: Maximum nonces to track (LRU eviction)
        """
        self._ttl = ttl_seconds
        self._max_skew = max_clock_skew_seconds
        self._max_size = max_cache_size

        # Nonce cache: nonce -> NonceEntry
        self._seen: Dict[str, NonceEntry] = {}

        # Lock for thread safety
        self._lock = threading.Lock()

        # Statistics
        self._total_checks = 0
        self._replays_blocked = 0
        self._stale_blocked = 0
        self._future_blocked = 0

    def check_timestamp(self, timestamp: datetime) -> Tuple[bool, Optional[RejectCode]]:
        """
        Validate timestamp freshness.

        Args:
            timestamp: Envelope timestamp (must be timezone-aware UTC)

        Returns:
            (True, None) if valid
            (False, RejectCode) if invalid
        """
        now = datetime.now(timezone.utc)
        delta = (timestamp - now).total_seconds()

        if delta > self._max_skew:
            # Timestamp too far in future
            self._future_blocked += 1
            return (False, RejectCode.REJECT_TIMESTAMP_FUTURE)

        if delta < -self._max_skew:
            # Timestamp too far in past
            self._stale_blocked += 1
            return (False, RejectCode.REJECT_TIMESTAMP_STALE)

        return (True, None)

    def check_nonce(
        self,
        nonce: str,
        envelope_digest: str,
    ) -> Tuple[bool, Optional[RejectCode]]:
        """
        Check if nonce has been seen before.

        Args:
            nonce: Hex-encoded 32-byte nonce
            envelope_digest: Digest of the envelope

        Returns:
            (True, None) if nonce is fresh
            (False, RejectCode.REJECT_NONCE_REPLAY) if seen
        """
        self._total_checks += 1
        now_mono = time.monotonic()

        with self._lock:
            # Evict expired entries
            self._evict_expired(now_mono)

            # Check if nonce already seen
            if nonce in self._seen:
                self._replays_blocked += 1
                return (False, RejectCode.REJECT_NONCE_REPLAY)

            # Evict oldest if at capacity
            if len(self._seen) >= self._max_size:
                self._evict_oldest()

            # Record this nonce
            self._seen[nonce] = NonceEntry(
                nonce=nonce,
                envelope_digest=envelope_digest,
                seen_at=now_mono,
                expires_at=now_mono + self._ttl,
            )

        return (True, None)

    def check(
        self,
        timestamp: datetime,
        nonce: str,
        envelope_digest: str,
    ) -> Tuple[bool, Optional[RejectCode]]:
        """
        Full replay check: timestamp freshness + nonce uniqueness.

        Args:
            timestamp: Envelope timestamp
            nonce: Hex-encoded 32-byte nonce
            envelope_digest: Digest of the envelope

        Returns:
            (True, None) if valid
            (False, RejectCode) if replay detected
        """
        # Check timestamp first (cheaper)
        ts_ok, ts_code = self.check_timestamp(timestamp)
        if not ts_ok:
            return (False, ts_code)

        # Check nonce
        return self.check_nonce(nonce, envelope_digest)

    def _evict_expired(self, now_mono: float) -> int:
        """
        Remove expired nonces. Must be called with lock held.

        Returns:
            Number of entries evicted
        """
        expired = [
            nonce for nonce, entry in self._seen.items() if entry.expires_at <= now_mono
        ]
        for nonce in expired:
            del self._seen[nonce]
        return len(expired)

    def _evict_oldest(self) -> None:
        """Remove oldest entry. Must be called with lock held."""
        if not self._seen:
            return

        oldest_nonce = min(self._seen.keys(), key=lambda n: self._seen[n].seen_at)
        del self._seen[oldest_nonce]

    def clear(self) -> None:
        """Clear all cached nonces."""
        with self._lock:
            self._seen.clear()

    def stats(self) -> Dict[str, int]:
        """Get statistics about replay guard operation."""
        return {
            "total_checks": self._total_checks,
            "replays_blocked": self._replays_blocked,
            "stale_blocked": self._stale_blocked,
            "future_blocked": self._future_blocked,
            "cache_size": len(self._seen),
            "cache_capacity": self._max_size,
        }


# Global singleton for shared replay guard
_global_guard: Optional[ReplayGuard] = None
_global_lock = threading.Lock()


def get_replay_guard() -> ReplayGuard:
    """Get the global replay guard singleton."""
    global _global_guard
    if _global_guard is None:
        with _global_lock:
            if _global_guard is None:
                _global_guard = ReplayGuard()
    return _global_guard


def reset_replay_guard() -> None:
    """Reset the global replay guard (for testing)."""
    global _global_guard
    with _global_lock:
        _global_guard = None


__all__ = [
    "ReplayGuard",
    "NonceEntry",
    "get_replay_guard",
    "reset_replay_guard",
    "DEFAULT_TTL_SECONDS",
    "DEFAULT_MAX_CLOCK_SKEW_SECONDS",
    "DEFAULT_MAX_CACHE_SIZE",
]
