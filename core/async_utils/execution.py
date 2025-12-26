"""
BIZRA AEON OMEGA - Async Execution Utilities
Non-blocking Execution Patterns for CPU-bound Operations

Addresses CVE-FIX-003: Event loop blocking
Fixes identified in:
- Synchronous crypto operations in async contexts
- File I/O blocking event loop
- CPU-intensive calculations without offloading

Author: BIZRA Performance Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import os
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger("bizra.async_utils")

# Type variables
T = TypeVar("T")
P = ParamSpec("P")


# =============================================================================
# PYTHON 3.10 COMPATIBILITY: asyncio.timeout shim
# =============================================================================

import sys

if sys.version_info >= (3, 11):
    # Python 3.11+ has native asyncio.timeout
    from asyncio import timeout as async_timeout
else:
    # Python 3.10: Provide a compatible implementation
    from contextlib import asynccontextmanager as _acm

    @_acm
    async def async_timeout(delay: float):
        """
        Compatibility shim for asyncio.timeout (added in Python 3.11).

        Uses asyncio.wait_for semantics to provide timeout functionality.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + delay

        # Create a task that will be cancelled on timeout
        class _TimeoutContext:
            def __init__(self):
                self._task = asyncio.current_task()
                self._cancelled = False
                self._timeout_handle = None

            def _on_timeout(self):
                self._cancelled = True
                if self._task:
                    self._task.cancel()

        ctx = _TimeoutContext()
        ctx._timeout_handle = loop.call_at(deadline, ctx._on_timeout)

        try:
            yield
        except asyncio.CancelledError:
            if ctx._cancelled:
                raise asyncio.TimeoutError()
            raise
        finally:
            if ctx._timeout_handle:
                ctx._timeout_handle.cancel()


# =============================================================================
# THREAD POOL EXECUTOR MANAGEMENT
# =============================================================================


class ExecutorPool:
    """
    Managed pool of executors for different workload types.

    Separates CPU-bound, I/O-bound, and priority workloads
    to prevent blocking and ensure fair scheduling.
    """

    _instance: Optional[ExecutorPool] = None
    _lock = threading.Lock()

    def __new__(cls) -> ExecutorPool:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Calculate optimal worker counts
        cpu_count = os.cpu_count() or 4

        # CPU-bound: Use exactly CPU count to avoid context switching
        self._cpu_workers = max(2, cpu_count)

        # I/O-bound: Higher count since threads spend time waiting
        self._io_workers = max(8, cpu_count * 4)

        # Priority: Small pool for urgent tasks
        self._priority_workers = max(2, cpu_count // 2)

        self._cpu_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._io_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._priority_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._process_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

        self._shutdown = False
        self._initialized = True

        logger.info(
            f"ExecutorPool initialized: "
            f"CPU={self._cpu_workers}, IO={self._io_workers}, Priority={self._priority_workers}"
        )

    @property
    def cpu_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Executor for CPU-bound operations."""
        if self._cpu_executor is None:
            self._cpu_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._cpu_workers, thread_name_prefix="bizra-cpu"
            )
        return self._cpu_executor

    @property
    def io_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Executor for I/O-bound operations."""
        if self._io_executor is None:
            self._io_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._io_workers, thread_name_prefix="bizra-io"
            )
        return self._io_executor

    @property
    def priority_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Executor for high-priority tasks."""
        if self._priority_executor is None:
            self._priority_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._priority_workers, thread_name_prefix="bizra-priority"
            )
        return self._priority_executor

    @property
    def process_executor(self) -> concurrent.futures.ProcessPoolExecutor:
        """Executor for truly CPU-intensive work that benefits from multiprocessing."""
        if self._process_executor is None:
            self._process_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=max(2, self._cpu_workers - 1)
            )
        return self._process_executor

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown all executors."""
        self._shutdown = True

        if self._cpu_executor:
            self._cpu_executor.shutdown(wait=wait)
        if self._io_executor:
            self._io_executor.shutdown(wait=wait)
        if self._priority_executor:
            self._priority_executor.shutdown(wait=wait)
        if self._process_executor:
            self._process_executor.shutdown(wait=wait)

        logger.info("ExecutorPool shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""

        def get_executor_stats(
            executor: Optional[concurrent.futures.ThreadPoolExecutor],
        ) -> Dict:
            if executor is None:
                return {"active": False}
            return {
                "active": True,
                "max_workers": executor._max_workers,
                # Note: _work_queue is internal but useful for monitoring
            }

        return {
            "cpu": get_executor_stats(self._cpu_executor),
            "io": get_executor_stats(self._io_executor),
            "priority": get_executor_stats(self._priority_executor),
            "process": {"active": self._process_executor is not None},
            "shutdown": self._shutdown,
        }


# =============================================================================
# ASYNC EXECUTION HELPERS
# =============================================================================


async def run_in_executor(
    func: Callable[P, T], *args: P.args, executor_type: str = "cpu", **kwargs: P.kwargs
) -> T:
    """
    Run a synchronous function in an executor without blocking event loop.

    Args:
        func: Synchronous function to execute
        *args: Positional arguments
        executor_type: "cpu", "io", or "priority"
        **kwargs: Keyword arguments

    Returns:
        Function result

    Example:
        # Before (blocking):
        result = expensive_crypto_operation(data)

        # After (non-blocking):
        result = await run_in_executor(expensive_crypto_operation, data)
    """
    loop = asyncio.get_running_loop()
    pool = ExecutorPool()

    if executor_type == "io":
        executor = pool.io_executor
    elif executor_type == "priority":
        executor = pool.priority_executor
    else:
        executor = pool.cpu_executor

    # Create partial with kwargs
    if kwargs:
        func = functools.partial(func, **kwargs)

    return await loop.run_in_executor(executor, func, *args)


async def run_cpu_bound(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Shorthand for running CPU-bound operations."""
    return await run_in_executor(func, *args, executor_type="cpu", **kwargs)


async def run_io_bound(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Shorthand for running I/O-bound operations."""
    return await run_in_executor(func, *args, executor_type="io", **kwargs)


async def run_in_process(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """
    Run function in separate process for true parallelism.

    Use for CPU-intensive work that needs to bypass the GIL.
    Note: func and args must be picklable.
    """
    loop = asyncio.get_running_loop()
    pool = ExecutorPool()

    if kwargs:
        func = functools.partial(func, **kwargs)

    return await loop.run_in_executor(pool.process_executor, func, *args)


# =============================================================================
# DECORATORS FOR NON-BLOCKING EXECUTION
# =============================================================================


def async_cpu_bound(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """
    Decorator to make a synchronous function non-blocking for CPU-bound work.

    Example:
        @async_cpu_bound
        def hash_password(password: str) -> str:
            return bcrypt.hashpw(password, bcrypt.gensalt())

        # Now it's awaitable and won't block
        hashed = await hash_password("secret")
    """

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return await run_cpu_bound(func, *args, **kwargs)

    return wrapper


def async_io_bound(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """
    Decorator to make a synchronous function non-blocking for I/O-bound work.

    Example:
        @async_io_bound
        def read_large_file(path: str) -> bytes:
            with open(path, "rb") as f:
                return f.read()

        # Now it's awaitable and won't block
        data = await read_large_file("/path/to/file")
    """

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return await run_io_bound(func, *args, **kwargs)

    return wrapper


def timeout_async(seconds: float):
    """
    Decorator to add timeout to async functions.

    Example:
        @timeout_async(5.0)
        async def fetch_data():
            ...  # Raises asyncio.TimeoutError if > 5 seconds
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


# =============================================================================
# BATCH ASYNC EXECUTION
# =============================================================================


@dataclass
class BatchResult(Generic[T]):
    """Result of a batch operation."""

    successful: List[T]
    failed: List[Tuple[int, Exception]]
    total: int
    success_rate: float


async def gather_with_concurrency(
    coros: List[Coroutine[Any, Any, T]],
    max_concurrent: int = 10,
    return_exceptions: bool = False,
    timeout: Optional[float] = None,
) -> List[Union[T, Exception]]:
    """
    Execute coroutines with concurrency limit and optional timeout.

    Prevents overwhelming resources with too many concurrent operations.

    Args:
        coros: List of coroutines to execute
        max_concurrent: Maximum simultaneous executions
        return_exceptions: If True, exceptions are returned instead of raised
        timeout: Optional overall timeout in seconds

    Returns:
        List of results (or exceptions if return_exceptions=True)
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_coro(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    gather_coro = asyncio.gather(
        *[limited_coro(c) for c in coros], return_exceptions=return_exceptions
    )

    if timeout is not None:
        return await asyncio.wait_for(gather_coro, timeout=timeout)
    return await gather_coro


async def batch_process(
    items: List[Any],
    processor: Callable[[Any], Awaitable[T]],
    batch_size: int = 100,
    max_concurrent: int = 10,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> BatchResult[T]:
    """
    Process items in batches with progress tracking.

    Args:
        items: Items to process
        processor: Async function to process each item
        batch_size: Number of items per batch
        max_concurrent: Max concurrent processing
        on_progress: Optional callback (completed, total)

    Returns:
        BatchResult with successes and failures
    """
    successful: List[T] = []
    failed: List[Tuple[int, Exception]] = []
    total = len(items)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = items[batch_start:batch_end]

        # Create coroutines for batch
        coros = [processor(item) for item in batch]

        # Execute with concurrency limit
        results = await gather_with_concurrency(
            coros, max_concurrent=max_concurrent, return_exceptions=True
        )

        # Process results
        for i, result in enumerate(results):
            idx = batch_start + i
            if isinstance(result, Exception):
                failed.append((idx, result))
            else:
                successful.append(result)

        # Progress callback
        if on_progress:
            on_progress(batch_end, total)

    return BatchResult(
        successful=successful,
        failed=failed,
        total=total,
        success_rate=len(successful) / max(1, total),
    )


# =============================================================================
# ASYNC RETRY UTILITIES
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 0.1
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[type, ...] = (Exception,)


async def retry_async(
    func: Callable[[], Awaitable[T]], config: Optional[RetryConfig] = None
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry (no arguments)
        config: Retry configuration

    Returns:
        Result from successful execution

    Raises:
        Last exception if all retries exhausted
    """
    cfg = config or RetryConfig()
    last_exception: Optional[Exception] = None

    for attempt in range(cfg.max_attempts):
        try:
            return await func()
        except cfg.retryable_exceptions as e:
            last_exception = e

            if attempt == cfg.max_attempts - 1:
                break

            # Calculate delay with exponential backoff
            delay = min(
                cfg.initial_delay * (cfg.exponential_base**attempt), cfg.max_delay
            )

            # Add jitter to prevent thundering herd
            if cfg.jitter:
                import random

                delay *= 0.5 + random.random()

            logger.warning(
                f"Retry {attempt + 1}/{cfg.max_attempts} after {delay:.2f}s: {e}"
            )
            await asyncio.sleep(delay)

    raise last_exception  # type: ignore


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator to add retry behavior to async functions.

    Example:
        @with_retry(RetryConfig(max_attempts=5))
        async def fetch_data():
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await retry_async(lambda: func(*args, **kwargs), config)

        return wrapper

    return decorator


# =============================================================================
# ASYNC CONTEXT MANAGERS
# =============================================================================


@asynccontextmanager
async def timeout_context(seconds: float):
    """
    Async context manager with timeout.

    Example:
        async with timeout_context(5.0):
            await slow_operation()  # Raises TimeoutError if > 5s
    """
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {seconds}s")
        raise


@asynccontextmanager
async def cancelation_scope():
    """
    Async context manager that handles cancellation gracefully.

    Example:
        async with cancelation_scope():
            await interruptible_operation()
    """
    try:
        yield
    except asyncio.CancelledError:
        logger.debug("Operation was cancelled")
        raise


# =============================================================================
# ASYNC RATE LIMITING
# =============================================================================


class AsyncRateLimiter:
    """
    Token bucket rate limiter for async operations.

    Prevents overwhelming external services or internal components.
    """

    def __init__(
        self,
        rate: float,
        burst: int = 1,
        time_func: Callable[[], float] = time.monotonic,
    ):
        """
        Initialize rate limiter.

        Args:
            rate: Tokens per second
            burst: Maximum burst size
            time_func: Time function (for testing)
        """
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_update = time_func()
        self._time_func = time_func
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.

        Returns: Time waited in seconds
        """
        async with self._lock:
            now = self._time_func()

            # Replenish tokens
            elapsed = now - self._last_update
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_update = now

            # Check if we have enough tokens
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            # Calculate wait time
            deficit = tokens - self._tokens
            wait_time = deficit / self._rate

            await asyncio.sleep(wait_time)
            self._tokens = 0
            self._last_update = self._time_func()

            return wait_time

    async def __aenter__(self) -> float:
        return await self.acquire()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


# =============================================================================
# ASYNC DEBOUNCING AND THROTTLING
# =============================================================================


class AsyncDebouncer:
    """
    Debounce rapid calls to an async function.

    Only executes after calls stop for the specified delay.
    """

    def __init__(self, delay: float):
        self._delay = delay
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[[], Awaitable[T]]) -> Optional[T]:
        """
        Debounced call. Returns result only if executed.
        """
        async with self._lock:
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

            async def delayed_call() -> T:
                await asyncio.sleep(self._delay)
                return await func()

            self._task = asyncio.create_task(delayed_call())

        try:
            return await self._task
        except asyncio.CancelledError:
            return None


class AsyncThrottler:
    """
    Throttle calls to ensure minimum time between executions.
    """

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """
        Throttled call. Waits if called too soon after previous call.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call

            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)

            result = await func()
            self._last_call = time.monotonic()
            return result


# =============================================================================
# GRACEFUL SHUTDOWN
# =============================================================================


class GracefulShutdown:
    """
    Handle graceful shutdown of async applications.

    Ensures all tasks complete before exit.
    """

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout
        self._shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []

    def register_task(self, task: asyncio.Task) -> None:
        """Register a task for graceful shutdown."""
        self._tasks.append(task)

    async def shutdown(self) -> None:
        """
        Initiate graceful shutdown.

        Cancels all registered tasks and waits for completion.
        """
        logger.info("Initiating graceful shutdown...")
        self._shutdown_event.set()

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks with timeout
        if self._tasks:
            await asyncio.wait(
                self._tasks, timeout=self._timeout, return_when=asyncio.ALL_COMPLETED
            )

        # Shutdown executor pool
        ExecutorPool().shutdown(wait=False)

        logger.info("Graceful shutdown complete")

    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown_event.is_set()

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()


def setup_signal_handlers(shutdown: GracefulShutdown) -> None:
    """
    Setup signal handlers for graceful shutdown.

    Call this at application startup.
    """
    loop = asyncio.get_running_loop()

    def handler(sig):
        logger.info(f"Received signal {sig}")
        loop.create_task(shutdown.shutdown())

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, handler, sig)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Executor Management
    "ExecutorPool",
    # Execution Helpers
    "run_in_executor",
    "run_cpu_bound",
    "run_io_bound",
    "run_in_process",
    # Decorators
    "async_cpu_bound",
    "async_io_bound",
    "timeout_async",
    "with_retry",
    # Batch Processing
    "BatchResult",
    "gather_with_concurrency",
    "batch_process",
    # Retry
    "RetryConfig",
    "retry_async",
    # Context Managers
    "timeout_context",
    "cancelation_scope",
    # Rate Limiting
    "AsyncRateLimiter",
    "AsyncDebouncer",
    "AsyncThrottler",
    # Shutdown
    "GracefulShutdown",
    "setup_signal_handlers",
]
