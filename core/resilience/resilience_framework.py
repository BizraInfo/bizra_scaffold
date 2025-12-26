"""
BIZRA Resilience Framework
===========================
Production-Grade Fault Tolerance with Ihsan-Aware Circuit Breakers

This module implements comprehensive resilience patterns for the BIZRA
system, ensuring graceful degradation and recovery under failures:

- Circuit Breakers (Ihsan-weighted failure detection)
- Bulkheads (Resource isolation)
- Adaptive Retry (Exponential backoff with jitter)
- Rate Limiting (Token bucket algorithm)
- Backpressure (Load shedding)
- Timeout Management
- Health Checks

Design Philosophy:
    FAIL-CLOSED: When in doubt, protect the system
    Ihsan violations are weighted as CRITICAL failures

Author: BIZRA Genesis Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
import random
import logging
import math
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger("bizra.resilience")

# Python 3.10 compatibility: use shim for asyncio.timeout
from core.async_utils.execution import async_timeout

# ============================================================================
# TIMEOUT CONSTANTS
# ============================================================================

HEALTH_CHECK_TIMEOUT_SECONDS = 10.0  # Max time for individual health check
ALL_HEALTH_CHECKS_TIMEOUT_SECONDS = 30.0  # Max time for all health checks

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

T = TypeVar("T")
AsyncFunc = Callable[..., Awaitable[T]]

# ============================================================================
# ENUMERATIONS
# ============================================================================


class CircuitState(Enum):
    """Circuit breaker state machine."""
    
    CLOSED = auto()     # Normal operation
    OPEN = auto()       # Rejecting requests
    HALF_OPEN = auto()  # Testing recovery


class FailureType(Enum):
    """Types of failures."""
    
    TIMEOUT = auto()
    EXCEPTION = auto()
    IHSAN_VIOLATION = auto()
    RATE_LIMITED = auto()
    CIRCUIT_OPEN = auto()
    RESOURCE_EXHAUSTED = auto()


class BulkheadType(Enum):
    """Types of bulkhead isolation."""
    
    SEMAPHORE = auto()   # Limit concurrent calls
    THREAD_POOL = auto() # Separate thread pool
    QUEUE = auto()       # Bounded queue


class RetryStrategy(Enum):
    """Retry strategies."""
    
    CONSTANT = auto()      # Fixed delay
    LINEAR = auto()        # Linearly increasing
    EXPONENTIAL = auto()   # Exponential backoff
    DECORRELATED = auto()  # Decorrelated jitter


# ============================================================================
# FAILURE TRACKING
# ============================================================================


@dataclass
class FailureRecord:
    """Record of a failure occurrence."""
    
    failure_id: str
    failure_type: FailureType
    timestamp: datetime
    message: str
    ihsan_score: Optional[float]
    recovery_time_ms: Optional[float]
    
    @staticmethod
    def create(
        failure_type: FailureType,
        message: str,
        ihsan_score: Optional[float] = None,
    ) -> FailureRecord:
        return FailureRecord(
            failure_id=f"fail_{secrets.token_hex(8)}",
            failure_type=failure_type,
            timestamp=datetime.now(timezone.utc),
            message=message,
            ihsan_score=ihsan_score,
            recovery_time_ms=None,
        )


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    
    failure_threshold: int = 5           # Failures to open
    success_threshold: int = 3           # Successes to close
    timeout_seconds: float = 30.0        # Open state duration
    half_open_max_calls: int = 3         # Max calls in half-open
    
    # Ihsan-specific
    ihsan_failure_weight: float = 3.0    # Weight for Ihsan violations
    ihsan_threshold: float = 0.95        # Ihsan score threshold
    
    # Sliding window
    window_size_seconds: float = 60.0    # Time window for failure rate
    failure_rate_threshold: float = 0.5  # Failure rate to open


class CircuitBreaker:
    """
    Ihsan-aware circuit breaker.
    
    Implements the circuit breaker pattern with special
    handling for Ihsan violations (weighted as critical).
    
    State Machine:
        CLOSED -> OPEN (on failure threshold)
        OPEN -> HALF_OPEN (after timeout)
        HALF_OPEN -> CLOSED (on success threshold)
        HALF_OPEN -> OPEN (on any failure)
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0.0  # Float for weighted counting
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        
        # Sliding window for failure rate
        self._recent_results: Deque[Tuple[datetime, bool]] = deque()
        
        self._lock = asyncio.Lock()
        
        # Metrics
        self._total_calls = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_rejections = 0
        self._ihsan_failures = 0
        # Bounded state change history to prevent memory growth
        self._state_changes: Deque[Tuple[datetime, CircuitState, CircuitState]] = deque(maxlen=1000)
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN
    
    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        ihsan_score: float = 1.0,
    ) -> T:
        """
        Execute operation through circuit breaker.
        
        Args:
            operation: Async function to execute
            ihsan_score: Ihsan score for the operation
        
        Returns:
            Operation result
        
        Raises:
            CircuitOpenError: If circuit is open
        """
        async with self._lock:
            self._total_calls += 1
            
            # Clean old results from sliding window
            self._clean_sliding_window()
            
            # Check Ihsan before proceeding
            if ihsan_score < self.config.ihsan_threshold:
                await self._record_failure(
                    FailureType.IHSAN_VIOLATION,
                    weight=self.config.ihsan_failure_weight,
                )
                raise IhsanCircuitViolation(
                    f"Ihsan {ihsan_score:.4f} < {self.config.ihsan_threshold}"
                )
            
            # State-based behavior
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self._total_rejections += 1
                    raise CircuitOpenError(
                        self.name,
                        self._time_until_reset(),
                    )
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._total_rejections += 1
                    raise CircuitOpenError(self.name, 0.0)
                self._half_open_calls += 1
        
        # Execute operation (outside lock)
        try:
            result = await operation()
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(FailureType.EXCEPTION)
            raise
    
    async def _record_success(self) -> None:
        """Record successful execution."""
        async with self._lock:
            self._total_successes += 1
            self._recent_results.append((datetime.now(timezone.utc), True))
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 0.5)
    
    async def _record_failure(
        self,
        failure_type: FailureType,
        weight: float = 1.0,
    ) -> None:
        """Record failed execution."""
        async with self._lock:
            self._total_failures += 1
            self._last_failure_time = datetime.now(timezone.utc)
            self._recent_results.append((datetime.now(timezone.utc), False))
            
            if failure_type == FailureType.IHSAN_VIOLATION:
                self._ihsan_failures += 1
            
            self._failure_count += weight
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                # Check threshold or failure rate
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                elif self._get_failure_rate() >= self.config.failure_rate_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def _clean_sliding_window(self) -> None:
        """Remove old results from sliding window."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.config.window_size_seconds
        )
        while self._recent_results and self._recent_results[0][0] < cutoff:
            self._recent_results.popleft()
    
    def _get_failure_rate(self) -> float:
        """Calculate current failure rate."""
        if not self._recent_results:
            return 0.0
        
        failures = sum(1 for _, success in self._recent_results if not success)
        return failures / len(self._recent_results)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to try reset."""
        if self._last_failure_time is None:
            return True
        
        elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_seconds
    
    def _time_until_reset(self) -> float:
        """Time until circuit may close."""
        if self._last_failure_time is None:
            return 0.0
        
        elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
        return max(0.0, self.config.timeout_seconds - elapsed)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            self._state_changes.append((
                datetime.now(timezone.utc),
                old_state,
                new_state,
            ))
            
            logger.info(
                f"Circuit '{self.name}': {old_state.name} -> {new_state.name}"
            )
            
            if new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._success_count = 0
                self._half_open_calls = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self._state.name,
            "total_calls": self._total_calls,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "total_rejections": self._total_rejections,
            "ihsan_failures": self._ihsan_failures,
            "failure_count": self._failure_count,
            "failure_rate": self._get_failure_rate(),
            "state_changes": len(self._state_changes),
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{name}' is open. Retry after {retry_after:.1f}s")


class IhsanCircuitViolation(Exception):
    """Raised when Ihsan violation triggers circuit."""
    pass


# ============================================================================
# BULKHEAD
# ============================================================================


class Bulkhead:
    """
    Bulkhead pattern for resource isolation.
    
    Limits concurrent executions to prevent
    resource exhaustion and cascading failures.
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_wait_seconds: float = 5.0,
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_wait_seconds = max_wait_seconds
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._waiting_count = 0
        
        # Metrics
        self._total_acquired = 0
        self._total_rejected = 0
        self._total_timeout = 0
        self._max_active = 0
        self._max_waiting = 0
    
    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a slot in the bulkhead.
        
        Raises:
            BulkheadFullError: If bulkhead is full and timeout exceeded
        """
        self._waiting_count += 1
        self._max_waiting = max(self._max_waiting, self._waiting_count)
        
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.max_wait_seconds,
            )
            
            self._waiting_count -= 1
            self._active_count += 1
            self._total_acquired += 1
            self._max_active = max(self._max_active, self._active_count)
            
            try:
                yield
            finally:
                self._active_count -= 1
                self._semaphore.release()
                
        except asyncio.TimeoutError:
            self._waiting_count -= 1
            self._total_timeout += 1
            self._total_rejected += 1
            raise BulkheadFullError(self.name, self.max_concurrent)
    
    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        """Execute operation within bulkhead."""
        async with self.acquire():
            return await operation()
    
    @property
    def active_count(self) -> int:
        return self._active_count
    
    @property
    def available_slots(self) -> int:
        return self.max_concurrent - self._active_count
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "active_count": self._active_count,
            "waiting_count": self._waiting_count,
            "available_slots": self.available_slots,
            "total_acquired": self._total_acquired,
            "total_rejected": self._total_rejected,
            "total_timeout": self._total_timeout,
            "max_active": self._max_active,
            "max_waiting": self._max_waiting,
        }


class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""
    
    def __init__(self, name: str, capacity: int):
        self.name = name
        self.capacity = capacity
        super().__init__(f"Bulkhead '{name}' at capacity ({capacity})")


# ============================================================================
# RETRY POLICY
# ============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    base_delay_ms: float = 100.0
    max_delay_ms: float = 10000.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter_factor: float = 0.2
    
    # Ihsan-specific
    retry_on_ihsan_violation: bool = False
    
    # Exception filtering
    retryable_exceptions: Tuple[type, ...] = (Exception,)
    non_retryable_exceptions: Tuple[type, ...] = ()


class RetryPolicy:
    """
    Adaptive retry policy with multiple strategies.
    
    Strategies:
    - CONSTANT: Fixed delay between retries
    - LINEAR: Linearly increasing delay
    - EXPONENTIAL: Exponential backoff
    - DECORRELATED: Decorrelated jitter (AWS style)
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        
        # Metrics
        self._total_attempts = 0
        self._total_retries = 0
        self._total_successes = 0
        self._total_exhausted = 0
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        strategy = self.config.strategy
        base = self.config.base_delay_ms
        max_delay = self.config.max_delay_ms
        
        if strategy == RetryStrategy.CONSTANT:
            delay = base
        elif strategy == RetryStrategy.LINEAR:
            delay = base * attempt
        elif strategy == RetryStrategy.EXPONENTIAL:
            delay = base * (2 ** (attempt - 1))
        elif strategy == RetryStrategy.DECORRELATED:
            # AWS-style decorrelated jitter
            delay = min(max_delay, random.uniform(base, base * 3 * attempt))
        else:
            delay = base
        
        # Apply jitter
        jitter = delay * self.config.jitter_factor
        delay += random.uniform(-jitter, jitter)
        
        # Clamp to max
        return min(delay, max_delay)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if retry should be attempted."""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check non-retryable first
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False
        
        # Check Ihsan violations
        if isinstance(exception, IhsanCircuitViolation):
            return self.config.retry_on_ihsan_violation
        
        # Check retryable
        return isinstance(exception, self.config.retryable_exceptions)
    
    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        """Execute operation with retry."""
        last_exception: Optional[Exception] = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            self._total_attempts += 1
            
            try:
                result = await operation()
                self._total_successes += 1
                return result
                
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    break
                
                self._total_retries += 1
                delay_ms = self.calculate_delay(attempt)
                
                logger.debug(
                    f"Retry attempt {attempt}/{self.config.max_attempts}, "
                    f"delay={delay_ms:.0f}ms"
                )
                
                await asyncio.sleep(delay_ms / 1000.0)
        
        self._total_exhausted += 1
        raise last_exception or RuntimeError("Retry exhausted")
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "max_attempts": self.config.max_attempts,
            "strategy": self.config.strategy.name,
            "total_attempts": self._total_attempts,
            "total_retries": self._total_retries,
            "total_successes": self._total_successes,
            "total_exhausted": self._total_exhausted,
            "success_rate": self._total_successes / max(1, self._total_attempts),
        }


# ============================================================================
# RATE LIMITER
# ============================================================================


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Implements the token bucket algorithm for
    smooth rate limiting with burst handling.
    """
    
    def __init__(
        self,
        name: str,
        rate_per_second: float = 100.0,
        burst_size: int = 10,
    ):
        self.name = name
        self.rate_per_second = rate_per_second
        self.burst_size = burst_size
        
        self._tokens = float(burst_size)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
        
        # Metrics
        self._total_acquired = 0
        self._total_rejected = 0
        self._total_waited = 0
    
    async def acquire(self, tokens: int = 1, wait: bool = True) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            wait: If True, wait for tokens; if False, return immediately
        
        Returns:
            True if tokens acquired, False if not (when wait=False)
        """
        async with self._lock:
            self._refill()
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_acquired += tokens
                return True
            
            if not wait:
                self._total_rejected += tokens
                return False
            
            # Calculate wait time
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self.rate_per_second
            self._total_waited += 1
        
        # Wait outside lock
        await asyncio.sleep(wait_time)
        
        async with self._lock:
            self._refill()
            self._tokens -= tokens
            self._total_acquired += tokens
            return True
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        
        self._tokens = min(
            self.burst_size,
            self._tokens + elapsed * self.rate_per_second,
        )
    
    @asynccontextmanager
    async def limit(self, tokens: int = 1):
        """Context manager for rate limiting."""
        await self.acquire(tokens)
        yield
    
    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        tokens: int = 1,
    ) -> T:
        """Execute operation with rate limiting."""
        await self.acquire(tokens)
        return await operation()
    
    @property
    def available_tokens(self) -> float:
        return self._tokens
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rate_per_second": self.rate_per_second,
            "burst_size": self.burst_size,
            "available_tokens": self._tokens,
            "total_acquired": self._total_acquired,
            "total_rejected": self._total_rejected,
            "total_waited": self._total_waited,
        }


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, name: str, limit: float):
        self.name = name
        self.limit = limit
        super().__init__(f"Rate limit '{name}' exceeded ({limit}/s)")


# ============================================================================
# TIMEOUT
# ============================================================================


class Timeout:
    """
    Timeout wrapper for async operations.
    
    Provides consistent timeout handling with metrics.
    """
    
    def __init__(
        self,
        name: str,
        timeout_seconds: float = 30.0,
    ):
        self.name = name
        self.timeout_seconds = timeout_seconds
        
        # Metrics
        self._total_calls = 0
        self._total_timeouts = 0
        self._total_successes = 0
    
    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        timeout: Optional[float] = None,
    ) -> T:
        """Execute operation with timeout."""
        timeout = timeout or self.timeout_seconds
        self._total_calls += 1
        
        try:
            result = await asyncio.wait_for(operation(), timeout=timeout)
            self._total_successes += 1
            return result
        except asyncio.TimeoutError:
            self._total_timeouts += 1
            raise TimeoutError(
                f"Operation '{self.name}' timed out after {timeout}s"
            )
    
    @asynccontextmanager
    async def scope(self, timeout: Optional[float] = None):
        """Context manager for timeout scope."""
        timeout = timeout or self.timeout_seconds
        self._total_calls += 1
        
        try:
            async with async_timeout(timeout):
                yield
            self._total_successes += 1
        except asyncio.TimeoutError:
            self._total_timeouts += 1
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timeout_seconds": self.timeout_seconds,
            "total_calls": self._total_calls,
            "total_timeouts": self._total_timeouts,
            "total_successes": self._total_successes,
            "timeout_rate": self._total_timeouts / max(1, self._total_calls),
        }


# ============================================================================
# RESILIENCE POLICY (COMPOSITE)
# ============================================================================


class ResiliencePolicy:
    """
    Composite resilience policy combining multiple patterns.
    
    Order of execution:
    1. Rate Limiter (if configured)
    2. Bulkhead (if configured)
    3. Timeout
    4. Circuit Breaker
    5. Retry
    """
    
    def __init__(
        self,
        name: str,
        circuit_breaker: Optional[CircuitBreaker] = None,
        bulkhead: Optional[Bulkhead] = None,
        retry: Optional[RetryPolicy] = None,
        rate_limiter: Optional[RateLimiter] = None,
        timeout: Optional[Timeout] = None,
    ):
        self.name = name
        self.circuit_breaker = circuit_breaker
        self.bulkhead = bulkhead
        self.retry = retry
        self.rate_limiter = rate_limiter
        self.timeout = timeout
    
    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        ihsan_score: float = 0.95,
    ) -> T:
        """Execute operation through all configured policies."""
        
        async def wrapped() -> T:
            # Timeout wrapper
            if self.timeout:
                return await self.timeout.execute(operation)
            return await operation()
        
        async def with_circuit_breaker() -> T:
            if self.circuit_breaker:
                return await self.circuit_breaker.execute(wrapped, ihsan_score)
            return await wrapped()
        
        async def with_retry() -> T:
            if self.retry:
                return await self.retry.execute(with_circuit_breaker)
            return await with_circuit_breaker()
        
        async def with_bulkhead() -> T:
            if self.bulkhead:
                return await self.bulkhead.execute(with_retry)
            return await with_retry()
        
        async def with_rate_limit() -> T:
            if self.rate_limiter:
                return await self.rate_limiter.execute(with_bulkhead)
            return await with_bulkhead()
        
        return await with_rate_limit()
    
    def get_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"name": self.name}
        
        if self.circuit_breaker:
            metrics["circuit_breaker"] = self.circuit_breaker.get_metrics()
        if self.bulkhead:
            metrics["bulkhead"] = self.bulkhead.get_metrics()
        if self.retry:
            metrics["retry"] = self.retry.get_metrics()
        if self.rate_limiter:
            metrics["rate_limiter"] = self.rate_limiter.get_metrics()
        if self.timeout:
            metrics["timeout"] = self.timeout.get_metrics()
        
        return metrics


# ============================================================================
# HEALTH CHECK
# ============================================================================


@dataclass
class HealthStatus:
    """Health check result."""
    
    name: str
    healthy: bool
    message: str
    timestamp: datetime
    latency_ms: float
    details: Dict[str, Any]


class HealthChecker:
    """
    Health check manager for resilience components.
    
    Aggregates health from multiple sources.
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], Awaitable[HealthStatus]]] = {}
        self._results: Dict[str, HealthStatus] = {}
        self._lock = asyncio.Lock()
    
    def register(
        self,
        name: str,
        check: Callable[[], Awaitable[HealthStatus]],
    ) -> None:
        """Register a health check."""
        self._checks[name] = check
    
    async def check(self, name: str, timeout: Optional[float] = None) -> HealthStatus:
        """Run a specific health check with timeout.
        
        Args:
            name: Health check name
            timeout: Max seconds for check. Defaults to HEALTH_CHECK_TIMEOUT_SECONDS.
        """
        if name not in self._checks:
            return HealthStatus(
                name=name,
                healthy=False,
                message="Check not found",
                timestamp=datetime.now(timezone.utc),
                latency_ms=0.0,
                details={},
            )
        
        effective_timeout = timeout or HEALTH_CHECK_TIMEOUT_SECONDS
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._checks[name](),
                timeout=effective_timeout
            )
            result.latency_ms = (time.perf_counter() - start) * 1000
        except asyncio.TimeoutError:
            result = HealthStatus(
                name=name,
                healthy=False,
                message=f"Health check timed out after {effective_timeout}s",
                timestamp=datetime.now(timezone.utc),
                latency_ms=(time.perf_counter() - start) * 1000,
                details={"error": "timeout"},
            )
        except Exception as e:
            result = HealthStatus(
                name=name,
                healthy=False,
                message=str(e),
                timestamp=datetime.now(timezone.utc),
                latency_ms=(time.perf_counter() - start) * 1000,
                details={"error": str(e)},
            )
        
        async with self._lock:
            self._results[name] = result
        
        return result
    
    async def check_all(self, timeout: Optional[float] = None) -> Dict[str, HealthStatus]:
        """Run all health checks with overall timeout.
        
        Args:
            timeout: Max seconds for all checks. Defaults to ALL_HEALTH_CHECKS_TIMEOUT_SECONDS.
        """
        effective_timeout = timeout or ALL_HEALTH_CHECKS_TIMEOUT_SECONDS
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[
                    self.check(name) for name in self._checks
                ]),
                timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"check_all timed out after {effective_timeout}s"
            )
            # Return partial results from cache + timeout status for missing
            results = []
            for name in self._checks:
                if name in self._results:
                    results.append(self._results[name])
                else:
                    results.append(HealthStatus(
                        name=name,
                        healthy=False,
                        message="Check timed out",
                        timestamp=datetime.now(timezone.utc),
                        latency_ms=effective_timeout * 1000,
                        details={"error": "batch_timeout"},
                    ))
        return {r.name: r for r in results}
    
    async def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        results = await self.check_all()
        return all(r.healthy for r in results.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        return {
            "checks": len(self._checks),
            "healthy": sum(1 for r in self._results.values() if r.healthy),
            "unhealthy": sum(1 for r in self._results.values() if not r.healthy),
            "results": {
                name: {
                    "healthy": r.healthy,
                    "message": r.message,
                    "latency_ms": r.latency_ms,
                }
                for name, r in self._results.items()
            },
        }


# ============================================================================
# DECORATORS
# ============================================================================


def with_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
):
    """Decorator to wrap function with circuit breaker."""
    cb = CircuitBreaker(name, config)
    
    def decorator(func: AsyncFunc[T]) -> AsyncFunc[T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            ihsan = kwargs.pop("ihsan_score", 0.95)
            return await cb.execute(
                lambda: func(*args, **kwargs),
                ihsan,
            )
        return wrapper
    
    return decorator


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator to wrap function with retry."""
    policy = RetryPolicy(config)
    
    def decorator(func: AsyncFunc[T]) -> AsyncFunc[T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await policy.execute(lambda: func(*args, **kwargs))
        return wrapper
    
    return decorator


def with_timeout(seconds: float):
    """Decorator to wrap function with timeout."""
    def decorator(func: AsyncFunc[T]) -> AsyncFunc[T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=seconds,
            )
        return wrapper
    
    return decorator


# ============================================================================
# RESILIENCE REGISTRY
# ============================================================================


class ResilienceRegistry:
    """
    Central registry for resilience components.
    
    Provides unified access and monitoring.
    """
    
    _instance: Optional[ResilienceRegistry] = None
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._bulkheads: Dict[str, Bulkhead] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._policies: Dict[str, ResiliencePolicy] = {}
        self._health_checker = HealthChecker()
    
    @classmethod
    def get_instance(cls) -> ResilienceRegistry:
        if cls._instance is None:
            cls._instance = ResilienceRegistry()
        return cls._instance
    
    def get_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name, config)
        return self._circuit_breakers[name]
    
    def get_bulkhead(
        self,
        name: str,
        max_concurrent: int = 10,
    ) -> Bulkhead:
        """Get or create a bulkhead."""
        if name not in self._bulkheads:
            self._bulkheads[name] = Bulkhead(name, max_concurrent)
        return self._bulkheads[name]
    
    def get_rate_limiter(
        self,
        name: str,
        rate: float = 100.0,
    ) -> RateLimiter:
        """Get or create a rate limiter."""
        if name not in self._rate_limiters:
            self._rate_limiters[name] = RateLimiter(name, rate)
        return self._rate_limiters[name]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics from all components."""
        return {
            "circuit_breakers": {
                name: cb.get_metrics()
                for name, cb in self._circuit_breakers.items()
            },
            "bulkheads": {
                name: b.get_metrics()
                for name, b in self._bulkheads.items()
            },
            "rate_limiters": {
                name: rl.get_metrics()
                for name, rl in self._rate_limiters.items()
            },
            "health": self._health_checker.get_summary(),
        }


# ============================================================================
# DEMO
# ============================================================================


async def demo_resilience():
    """Demonstrate resilience framework capabilities."""
    print("=" * 70)
    print("BIZRA RESILIENCE FRAMEWORK DEMO")
    print("=" * 70)
    
    # 1. Circuit Breaker
    print("\n1. Circuit Breaker Demo")
    print("-" * 40)
    
    cb = CircuitBreaker("demo_cb", CircuitBreakerConfig(failure_threshold=3))
    
    async def unreliable_op(fail: bool = False):
        if fail:
            raise ValueError("Simulated failure")
        return "success"
    
    # Normal operations
    for i in range(3):
        result = await cb.execute(lambda: unreliable_op(False))
        print(f"  Call {i+1}: {result}")
    
    # Trigger failures
    for i in range(4):
        try:
            await cb.execute(lambda: unreliable_op(True))
        except (ValueError, CircuitOpenError) as e:
            print(f"  Failure {i+1}: {type(e).__name__}")
    
    print(f"  Circuit state: {cb.state.name}")
    
    # 2. Bulkhead
    print("\n2. Bulkhead Demo")
    print("-" * 40)
    
    bulkhead = Bulkhead("demo_bulkhead", max_concurrent=3, max_wait_seconds=1.0)
    
    async def slow_op():
        await asyncio.sleep(0.5)
        return "done"
    
    # Start multiple concurrent operations
    tasks = [bulkhead.execute(slow_op) for _ in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successes = sum(1 for r in results if r == "done")
    failures = sum(1 for r in results if isinstance(r, Exception))
    print(f"  Successes: {successes}, Failures: {failures}")
    
    # 3. Retry
    print("\n3. Retry Policy Demo")
    print("-" * 40)
    
    retry = RetryPolicy(RetryConfig(
        max_attempts=3,
        base_delay_ms=100,
        strategy=RetryStrategy.EXPONENTIAL,
    ))
    
    attempt_count = 0
    
    async def eventually_succeeds():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError("Not yet")
        return "finally!"
    
    result = await retry.execute(eventually_succeeds)
    print(f"  Succeeded after {attempt_count} attempts: {result}")
    
    # 4. Rate Limiter
    print("\n4. Rate Limiter Demo")
    print("-" * 40)
    
    limiter = RateLimiter("demo_limiter", rate_per_second=10.0, burst_size=5)
    
    start = time.perf_counter()
    for i in range(10):
        await limiter.acquire()
    elapsed = time.perf_counter() - start
    
    print(f"  10 requests completed in {elapsed:.2f}s")
    print(f"  Effective rate: {10/elapsed:.1f}/s")
    
    # 5. Composite Policy
    print("\n5. Composite Resilience Policy Demo")
    print("-" * 40)
    
    policy = ResiliencePolicy(
        name="demo_policy",
        circuit_breaker=CircuitBreaker("policy_cb"),
        retry=RetryPolicy(),
        timeout=Timeout("policy_timeout", timeout_seconds=5.0),
    )
    
    async def protected_op():
        return "protected result"
    
    result = await policy.execute(protected_op, ihsan_score=0.99)
    print(f"  Result: {result}")
    
    # 6. Metrics
    print("\n6. Metrics Summary")
    print("-" * 40)
    
    registry = ResilienceRegistry.get_instance()
    registry._circuit_breakers["demo_cb"] = cb
    
    metrics = registry.get_all_metrics()
    print(f"  Circuit breakers: {len(metrics['circuit_breakers'])}")
    
    print("\n" + "=" * 70)
    print("RESILIENCE FRAMEWORK DEMO COMPLETE")
    print("=" * 70)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "CircuitState",
    "FailureType",
    "BulkheadType",
    "RetryStrategy",
    # Failure
    "FailureRecord",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "IhsanCircuitViolation",
    # Bulkhead
    "Bulkhead",
    "BulkheadFullError",
    # Retry
    "RetryPolicy",
    "RetryConfig",
    # Rate Limiter
    "RateLimiter",
    "RateLimitExceeded",
    # Timeout
    "Timeout",
    # Composite
    "ResiliencePolicy",
    # Health
    "HealthStatus",
    "HealthChecker",
    # Decorators
    "with_circuit_breaker",
    "with_retry",
    "with_timeout",
    # Registry
    "ResilienceRegistry",
]


if __name__ == "__main__":
    asyncio.run(demo_resilience())
