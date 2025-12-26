"""
BIZRA AEON OMEGA - Memory Management & Leak Prevention
Bounded Collections, LRU Eviction, and Resource Cleanup

Addresses CVE-FIX-002: Unbounded history accumulation
Fixes identified memory leaks in:
- L2 compression_history
- Circuit breaker call_history
- Evaluation history
- Event listeners

Author: BIZRA Performance Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import gc
import logging
import sys
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger("bizra.memory")


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")
KT = TypeVar("KT")
VT = TypeVar("VT")


# =============================================================================
# BOUNDED COLLECTIONS - PREVENTS UNBOUNDED GROWTH
# =============================================================================


class BoundedList(Generic[T]):
    """
    A list with a maximum size that automatically evicts oldest items.

    Thread-safe replacement for unbounded lists like compression_history.
    Uses deque internally for O(1) append and popleft.

    Example:
        # Before (memory leak):
        self.compression_history: List[float] = []
        self.compression_history.append(ratio)  # UNBOUNDED!

        # After (bounded):
        self.compression_history = BoundedList[float](max_size=1000)
        self.compression_history.append(ratio)  # Auto-evicts when full
    """

    def __init__(
        self,
        max_size: int = 1000,
        eviction_callback: Optional[Callable[[T], None]] = None,
    ):
        """
        Initialize bounded list.

        Args:
            max_size: Maximum number of items to retain
            eviction_callback: Optional callback when item is evicted
        """
        if max_size < 1:
            raise ValueError("max_size must be at least 1")

        self._max_size = max_size
        self._data: Deque[T] = deque(maxlen=max_size)
        self._eviction_callback = eviction_callback
        self._lock = threading.RLock()
        self._total_added = 0
        self._total_evicted = 0

    def append(self, item: T) -> Optional[T]:
        """
        Append item, evicting oldest if at capacity.

        Returns: Evicted item if any, else None
        """
        with self._lock:
            evicted = None
            if len(self._data) == self._max_size:
                evicted = self._data[0]
                self._total_evicted += 1

            self._data.append(item)
            self._total_added += 1

            if evicted is not None and self._eviction_callback:
                try:
                    self._eviction_callback(evicted)
                except Exception as e:
                    logger.warning(f"Eviction callback error: {e}")

            return evicted

    def extend(self, items: List[T]) -> List[T]:
        """Extend with multiple items, returning all evicted items."""
        evicted = []
        with self._lock:
            for item in items:
                e = self.append(item)
                if e is not None:
                    evicted.append(e)
        return evicted

    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        with self._lock:
            return iter(list(self._data))

    def __getitem__(self, index: int) -> T:
        with self._lock:
            return self._data[index]

    def __bool__(self) -> bool:
        return len(self._data) > 0

    def to_list(self) -> List[T]:
        """Get a copy of all items as a list."""
        with self._lock:
            return list(self._data)

    def last_n(self, n: int) -> List[T]:
        """Get the last n items."""
        with self._lock:
            if n >= len(self._data):
                return list(self._data)
            return list(self._data)[-n:]

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        with self._lock:
            return {
                "current_size": len(self._data),
                "max_size": self._max_size,
                "total_added": self._total_added,
                "total_evicted": self._total_evicted,
                "utilization_percent": int(100 * len(self._data) / self._max_size),
            }


class LRUCache(Generic[KT, VT]):
    """
    Thread-safe LRU cache with optional TTL and size-based eviction.

    Replacement for unbounded dicts that grow over time.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[float] = None,
        on_evict: Optional[Callable[[KT, VT], None]] = None,
    ):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            ttl_seconds: Optional time-to-live for entries
            on_evict: Optional callback on eviction
        """
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._on_evict = on_evict
        self._cache: OrderedDict[KT, Tuple[VT, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: KT, default: Optional[VT] = None) -> Optional[VT]:
        """Get item, updating LRU order."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return default

            value, timestamp = self._cache[key]

            # Check TTL
            if self._ttl and time.time() - timestamp > self._ttl:
                self._evict(key)
                self._misses += 1
                return default

            # Move to end (most recent)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: KT, value: VT) -> Optional[Tuple[KT, VT]]:
        """
        Put item in cache.

        Returns: (evicted_key, evicted_value) if eviction occurred
        """
        with self._lock:
            evicted = None

            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    # Evict oldest
                    oldest_key = next(iter(self._cache))
                    oldest_value, _ = self._cache[oldest_key]
                    evicted = (oldest_key, oldest_value)
                    self._evict(oldest_key)

            self._cache[key] = (value, time.time())
            return evicted

    def remove(self, key: KT) -> Optional[VT]:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                value, _ = self._cache.pop(key)
                return value
            return None

    def _evict(self, key: KT) -> None:
        """Evict a specific key."""
        if key in self._cache:
            value, _ = self._cache.pop(key)
            if self._on_evict:
                try:
                    self._on_evict(key, value)
                except Exception as e:
                    logger.warning(f"Eviction callback error: {e}")

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        if not self._ttl:
            return 0

        removed = 0
        now = time.time()

        with self._lock:
            expired_keys = [
                k for k, (_, ts) in self._cache.items() if now - ts > self._ttl
            ]
            for key in expired_keys:
                self._evict(key)
                removed += 1

        return removed

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: KT) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            if self._ttl:
                _, ts = self._cache[key]
                if time.time() - ts > self._ttl:
                    self._evict(key)
                    return False
            return True

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(1, total),
                "ttl_seconds": self._ttl,
            }


class TTLDict(Generic[KT, VT]):
    """
    Dictionary with automatic expiration of entries.

    Useful for caches that should automatically clean up stale data.
    """

    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        cleanup_interval: float = 60.0,
        max_size: Optional[int] = None,
    ):
        """
        Initialize TTL dictionary.

        Args:
            ttl_seconds: Time-to-live for entries
            cleanup_interval: How often to run cleanup (seconds)
            max_size: Optional maximum size (uses LRU eviction)
        """
        self._ttl = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._max_size = max_size
        self._data: Dict[KT, Tuple[VT, float]] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._access_order: List[KT] = []

    def __setitem__(self, key: KT, value: VT) -> None:
        with self._lock:
            self._maybe_cleanup()

            if self._max_size and len(self._data) >= self._max_size:
                if key not in self._data:
                    # Evict oldest
                    if self._access_order:
                        oldest = self._access_order.pop(0)
                        self._data.pop(oldest, None)

            self._data[key] = (value, time.time())

            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    def __getitem__(self, key: KT) -> VT:
        with self._lock:
            if key not in self._data:
                raise KeyError(key)

            value, timestamp = self._data[key]

            if time.time() - timestamp > self._ttl:
                self._remove(key)
                raise KeyError(key)

            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            return value

    def get(self, key: KT, default: Optional[VT] = None) -> Optional[VT]:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: KT) -> bool:
        with self._lock:
            if key not in self._data:
                return False
            _, timestamp = self._data[key]
            if time.time() - timestamp > self._ttl:
                self._remove(key)
                return False
            return True

    def __len__(self) -> int:
        with self._lock:
            self._maybe_cleanup()
            return len(self._data)

    def _remove(self, key: KT) -> None:
        self._data.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)

    def _maybe_cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        expired = [k for k, (_, ts) in self._data.items() if now - ts > self._ttl]
        for key in expired:
            self._remove(key)

        if expired:
            logger.debug(f"TTLDict cleanup: removed {len(expired)} expired entries")


# =============================================================================
# SLIDING WINDOW STATISTICS
# =============================================================================


class SlidingWindowStats:
    """
    Calculate statistics over a sliding time window.

    Efficient for metrics like "average over last 5 minutes".
    Automatically drops old samples, preventing memory growth.
    """

    def __init__(self, window_seconds: float = 300.0, max_samples: int = 10000):
        """
        Initialize sliding window.

        Args:
            window_seconds: Time window for statistics
            max_samples: Maximum samples to retain (memory bound)
        """
        self._window_seconds = window_seconds
        self._max_samples = max_samples
        self._samples: Deque[Tuple[float, float]] = deque(maxlen=max_samples)
        self._lock = threading.RLock()

    def record(self, value: float) -> None:
        """Record a sample with current timestamp."""
        with self._lock:
            self._samples.append((time.time(), value))

    def _get_window_samples(self) -> List[float]:
        """Get samples within the time window."""
        cutoff = time.time() - self._window_seconds
        return [v for ts, v in self._samples if ts >= cutoff]

    def mean(self) -> Optional[float]:
        """Get mean of samples in window."""
        with self._lock:
            samples = self._get_window_samples()
            if not samples:
                return None
            return sum(samples) / len(samples)

    def min(self) -> Optional[float]:
        """Get minimum of samples in window."""
        with self._lock:
            samples = self._get_window_samples()
            return min(samples) if samples else None

    def max(self) -> Optional[float]:
        """Get maximum of samples in window."""
        with self._lock:
            samples = self._get_window_samples()
            return max(samples) if samples else None

    def count(self) -> int:
        """Get count of samples in window."""
        with self._lock:
            return len(self._get_window_samples())

    def percentile(self, p: float) -> Optional[float]:
        """Get p-th percentile (0-100) of samples in window."""
        with self._lock:
            samples = self._get_window_samples()
            if not samples:
                return None
            samples.sort()
            idx = int(len(samples) * p / 100)
            return samples[min(idx, len(samples) - 1)]

    def stats(self) -> Dict[str, Optional[float]]:
        """Get all statistics."""
        with self._lock:
            samples = self._get_window_samples()
            if not samples:
                return {
                    "count": 0,
                    "mean": None,
                    "min": None,
                    "max": None,
                    "p50": None,
                    "p95": None,
                    "p99": None,
                }

            samples.sort()
            n = len(samples)

            return {
                "count": n,
                "mean": sum(samples) / n,
                "min": samples[0],
                "max": samples[-1],
                "p50": samples[n // 2],
                "p95": samples[int(n * 0.95)],
                "p99": samples[int(n * 0.99)],
            }


# =============================================================================
# RESOURCE TRACKER - CLEANUP MANAGEMENT
# =============================================================================


@dataclass
class ResourceInfo:
    """Information about a tracked resource."""

    resource_id: str
    resource_type: str
    created_at: datetime
    cleanup_func: Optional[Callable[[], None]]
    weak_ref: Optional[weakref.ref]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourceTracker:
    """
    Track and manage cleanup of resources.

    Prevents resource leaks by:
    1. Tracking all created resources
    2. Automatic cleanup on program exit
    3. Periodic cleanup of weak-referenced resources
    4. Explicit cleanup API
    """

    _instance: Optional[ResourceTracker] = None
    _lock = threading.Lock()

    def __new__(cls) -> ResourceTracker:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._resources: Dict[str, ResourceInfo] = {}
        self._resource_lock = threading.RLock()
        self._cleanup_registered = False
        self._initialized = True

    def track(
        self,
        resource_id: str,
        resource_type: str,
        cleanup_func: Optional[Callable[[], None]] = None,
        weak_ref: Optional[Any] = None,
        **metadata,
    ) -> None:
        """
        Track a resource for cleanup.

        Args:
            resource_id: Unique identifier for the resource
            resource_type: Type of resource (e.g., "connection", "file")
            cleanup_func: Optional cleanup function to call
            weak_ref: Optional object to create weak reference to
            **metadata: Additional metadata about the resource
        """
        with self._resource_lock:
            ref = weakref.ref(weak_ref) if weak_ref else None

            self._resources[resource_id] = ResourceInfo(
                resource_id=resource_id,
                resource_type=resource_type,
                created_at=datetime.now(timezone.utc),
                cleanup_func=cleanup_func,
                weak_ref=ref,
                metadata=metadata,
            )

            if not self._cleanup_registered:
                import atexit

                atexit.register(self.cleanup_all)
                self._cleanup_registered = True

        logger.debug(f"Tracking resource: {resource_id} ({resource_type})")

    def untrack(self, resource_id: str) -> bool:
        """Remove a resource from tracking. Returns True if found."""
        with self._resource_lock:
            if resource_id in self._resources:
                del self._resources[resource_id]
                logger.debug(f"Untracked resource: {resource_id}")
                return True
            return False

    def cleanup(self, resource_id: str) -> bool:
        """Cleanup a specific resource. Returns True if cleaned up."""
        with self._resource_lock:
            if resource_id not in self._resources:
                return False

            info = self._resources[resource_id]

            if info.cleanup_func:
                try:
                    info.cleanup_func()
                    logger.debug(f"Cleaned up resource: {resource_id}")
                except Exception as e:
                    logger.warning(f"Cleanup error for {resource_id}: {e}")

            del self._resources[resource_id]
            return True

    def cleanup_by_type(self, resource_type: str) -> int:
        """Cleanup all resources of a given type. Returns count cleaned."""
        with self._resource_lock:
            to_cleanup = [
                rid
                for rid, info in self._resources.items()
                if info.resource_type == resource_type
            ]

        cleaned = 0
        for rid in to_cleanup:
            if self.cleanup(rid):
                cleaned += 1

        return cleaned

    def cleanup_dead_refs(self) -> int:
        """Cleanup resources whose weak references are dead."""
        with self._resource_lock:
            dead = [
                rid
                for rid, info in self._resources.items()
                if info.weak_ref and info.weak_ref() is None
            ]

        cleaned = 0
        for rid in dead:
            if self.cleanup(rid):
                cleaned += 1

        if cleaned:
            logger.info(f"Cleaned up {cleaned} dead-referenced resources")

        return cleaned

    def cleanup_all(self) -> int:
        """Cleanup all tracked resources."""
        with self._resource_lock:
            all_ids = list(self._resources.keys())

        cleaned = 0
        for rid in all_ids:
            if self.cleanup(rid):
                cleaned += 1

        logger.info(f"Cleaned up {cleaned} resources on shutdown")
        return cleaned

    def get_stats(self) -> Dict[str, Any]:
        """Get resource tracking statistics."""
        with self._resource_lock:
            by_type: Dict[str, int] = {}
            for info in self._resources.values():
                by_type[info.resource_type] = by_type.get(info.resource_type, 0) + 1

            return {
                "total_tracked": len(self._resources),
                "by_type": by_type,
                "resource_ids": list(self._resources.keys()),
            }


# =============================================================================
# MEMORY MONITOR
# =============================================================================


class MemoryMonitor:
    """
    Monitor memory usage and trigger cleanup when thresholds exceeded.
    """

    def __init__(
        self,
        warning_threshold_mb: float = 500.0,
        critical_threshold_mb: float = 1000.0,
        check_interval_seconds: float = 30.0,
    ):
        self._warning_threshold = warning_threshold_mb * 1024 * 1024
        self._critical_threshold = critical_threshold_mb * 1024 * 1024
        self._check_interval = check_interval_seconds
        self._callbacks: List[Callable[[str, float], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register_callback(self, callback: Callable[[str, float], None]) -> None:
        """
        Register callback for memory alerts.

        Callback receives: (level: "warning"|"critical", usage_mb: float)
        """
        self._callbacks.append(callback)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        import psutil

        process = psutil.Process()
        mem = process.memory_info()

        return {
            "rss_mb": mem.rss / 1024 / 1024,
            "vms_mb": mem.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }

    def check_memory(self) -> Optional[str]:
        """Check memory and return alert level if threshold exceeded."""
        try:
            usage = self.get_memory_usage()
            rss = usage["rss_mb"] * 1024 * 1024

            if rss >= self._critical_threshold:
                level = "critical"
            elif rss >= self._warning_threshold:
                level = "warning"
            else:
                return None

            for callback in self._callbacks:
                try:
                    callback(level, usage["rss_mb"])
                except Exception as e:
                    logger.warning(f"Memory callback error: {e}")

            return level

        except ImportError:
            logger.debug("psutil not available for memory monitoring")
            return None

    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection and return stats."""
        collected = {}
        for gen in range(3):
            collected[f"gen{gen}"] = gc.collect(gen)
        return collected

    def start_monitoring(self) -> None:
        """Start background memory monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="bizra-memory-monitor"
        )
        self._thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            level = self.check_memory()
            if level == "critical":
                logger.warning("Critical memory usage - forcing GC")
                self.force_gc()
                ResourceTracker().cleanup_dead_refs()

            time.sleep(self._check_interval)


# =============================================================================
# ASYNC CLEANUP UTILITIES
# =============================================================================


class AsyncResourceManager:
    """
    Async context manager for resource cleanup.

    Example:
        async with AsyncResourceManager() as mgr:
            conn = await create_connection()
            mgr.register(conn.close)
            # conn.close() called automatically on exit
    """

    def __init__(self):
        self._cleanup_funcs: List[Callable] = []
        self._async_cleanup_funcs: List[Callable] = []

    async def __aenter__(self) -> AsyncResourceManager:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Run async cleanup functions
        for func in reversed(self._async_cleanup_funcs):
            try:
                await func()
            except Exception as e:
                logger.warning(f"Async cleanup error: {e}")

        # Run sync cleanup functions
        for func in reversed(self._cleanup_funcs):
            try:
                func()
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")

    def register(self, cleanup_func: Callable) -> None:
        """Register a sync cleanup function."""
        self._cleanup_funcs.append(cleanup_func)

    def register_async(self, cleanup_func: Callable) -> None:
        """Register an async cleanup function."""
        self._async_cleanup_funcs.append(cleanup_func)


def bounded(max_size: int = 1000):
    """
    Decorator to replace list attributes with bounded versions.

    Example:
        class MyClass:
            @bounded(max_size=500)
            def history(self):
                return []  # Will be replaced with BoundedList
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self):
            attr_name = f"_bounded_{func.__name__}"
            if not hasattr(self, attr_name):
                setattr(self, attr_name, BoundedList(max_size=max_size))
            return getattr(self, attr_name)

        return property(wrapper)

    return decorator


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_bounded_history(max_size: int = 1000) -> BoundedList[float]:
    """Create a bounded history list for metrics."""
    return BoundedList[float](max_size=max_size)


def create_lru_cache(
    max_size: int = 1000, ttl_seconds: Optional[float] = 3600.0
) -> LRUCache:
    """Create an LRU cache with optional TTL."""
    return LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)


def create_sliding_stats(
    window_seconds: float = 300.0, max_samples: int = 10000
) -> SlidingWindowStats:
    """Create a sliding window statistics tracker."""
    return SlidingWindowStats(window_seconds=window_seconds, max_samples=max_samples)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Bounded Collections
    "BoundedList",
    "LRUCache",
    "TTLDict",
    # Statistics
    "SlidingWindowStats",
    # Resource Management
    "ResourceInfo",
    "ResourceTracker",
    "MemoryMonitor",
    "AsyncResourceManager",
    # Decorators
    "bounded",
    # Factory Functions
    "create_bounded_history",
    "create_lru_cache",
    "create_sliding_stats",
]
