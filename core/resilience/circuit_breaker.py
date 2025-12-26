"""
BIZRA AEON OMEGA - Circuit Breaker Resilience Pattern
═══════════════════════════════════════════════════════════════════════════════
Production-Grade | Failure Isolation | Self-Healing

Implements the circuit breaker pattern for graceful degradation:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests fail fast
- HALF_OPEN: Testing if service recovered

References:
- Martin Fowler's Circuit Breaker pattern
- Netflix Hystrix design principles
- Microsoft Azure resilience patterns
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing fast
    HALF_OPEN = auto()  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    timeout_seconds: float = 30.0  # Time before attempting recovery
    half_open_max_calls: int = 3  # Max calls in half-open state

    # Failure rate based (alternative to count-based)
    failure_rate_threshold: float = 0.5  # 50% failure rate
    minimum_calls: int = 10  # Min calls before rate calculation

    # Slow call handling
    slow_call_threshold_ms: float = 1000.0  # Calls slower than this are "slow"
    slow_call_rate_threshold: float = 0.5  # 50% slow calls opens circuit


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    slow_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    @property
    def slow_call_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.slow_calls / self.total_calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "slow_calls": self.slow_calls,
            "failure_rate": round(self.failure_rate, 4),
            "slow_call_rate": round(self.slow_call_rate, 4),
            "state_changes": self.state_changes,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_success_time": (
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker rejects a call."""

    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is OPEN."""

    def __init__(self, circuit_name: str, time_until_retry: float):
        self.circuit_name = circuit_name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit '{circuit_name}' is OPEN. Retry in {time_until_retry:.1f}s"
        )


class CircuitBreaker:
    """
    Production-grade circuit breaker implementation.

    Usage:
        breaker = CircuitBreaker("database")

        @breaker.protect
        async def query_database(query: str):
            return await db.execute(query)

        # Or manual usage:
        async with breaker:
            result = await risky_operation()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._metrics = CircuitMetrics()
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

        # Sliding window for rate-based calculation
        self._call_history: deque = deque(maxlen=100)

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def metrics(self) -> CircuitMetrics:
        return self._metrics

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN

    async def _check_state_transition(self) -> None:
        """Check if state should transition."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.timeout_seconds:
                        self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state
        self._metrics.state_changes += 1
        self._metrics.last_state_change = datetime.now(timezone.utc)

        # Reset counters
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
            self._failure_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_calls = 0

        logger.info(
            f"Circuit '{self.name}' transitioned: {old_state.name} -> {new_state.name}"
        )

        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    async def _record_success(self, duration_ms: float) -> None:
        """Record a successful call."""
        async with self._lock:
            self._metrics.total_calls += 1
            self._metrics.successful_calls += 1
            self._metrics.last_success_time = datetime.now(timezone.utc)

            if duration_ms > self.config.slow_call_threshold_ms:
                self._metrics.slow_calls += 1

            self._call_history.append(("success", duration_ms))

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    async def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self._metrics.total_calls += 1
            self._metrics.failed_calls += 1
            self._metrics.last_failure_time = datetime.now(timezone.utc)
            self._last_failure_time = time.time()

            self._call_history.append(("failure", 0))

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1

                # Check thresholds
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                elif self._should_open_by_rate():
                    self._transition_to(CircuitState.OPEN)

    def _should_open_by_rate(self) -> bool:
        """Check if failure rate exceeds threshold."""
        if len(self._call_history) < self.config.minimum_calls:
            return False

        failures = sum(1 for call, _ in self._call_history if call == "failure")
        rate = failures / len(self._call_history)

        return rate >= self.config.failure_rate_threshold

    async def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        await self._check_state_transition()

        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed < self.config.timeout_seconds:
                    self._metrics.rejected_calls += 1
                    raise CircuitOpenError(
                        self.name, self.config.timeout_seconds - elapsed
                    )
            # Timeout elapsed, transition to half-open
            async with self._lock:
                self._transition_to(CircuitState.HALF_OPEN)

        if self._state == CircuitState.HALF_OPEN:
            async with self._lock:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._metrics.rejected_calls += 1
                    raise CircuitOpenError(self.name, 1.0)
                self._half_open_calls += 1

        return True

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function through the circuit breaker."""
        await self._can_execute()

        start = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            duration_ms = (time.perf_counter() - start) * 1000
            await self._record_success(duration_ms)
            return result

        except Exception as e:
            await self._record_failure(e)

            # Try fallback if available
            if self.fallback is not None:
                if asyncio.iscoroutinefunction(self.fallback):
                    return await self.fallback(*args, **kwargs)
                return self.fallback(*args, **kwargs)

            raise

    def protect(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to protect a function with circuit breaker."""
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.execute(func, *args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.get_event_loop().run_until_complete(
                    self.execute(func, *args, **kwargs)
                )

            return sync_wrapper

    async def __aenter__(self):
        await self._can_execute()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self._record_success(0)
        else:
            await self._record_failure(exc_val)
        return False  # Don't suppress exception

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._call_history.clear()
        logger.info(f"Circuit '{self.name}' manually reset")


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Usage:
        registry = CircuitBreakerRegistry()

        db_breaker = registry.get_or_create("database")
        api_breaker = registry.get_or_create("external_api")

        all_metrics = registry.get_all_metrics()
    """

    def __init__(self, default_config: Optional[CircuitBreakerConfig] = None):
        self.default_config = default_config or CircuitBreakerConfig()
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def get_or_create(
        self, name: str, config: Optional[CircuitBreakerConfig] = None, **kwargs
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name, config=config or self.default_config, **kwargs
            )
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all circuit breakers."""
        return {
            name: {"state": breaker.state.name, **breaker.metrics.to_dict()}
            for name, breaker in self._breakers.items()
        }

    def get_open_circuits(self) -> List[str]:
        """Get names of all open circuits."""
        return [name for name, breaker in self._breakers.items() if breaker.is_open]

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA-SPECIFIC CIRCUIT BREAKERS
# ═══════════════════════════════════════════════════════════════════════════════

# Global registry for BIZRA services
bizra_circuit_registry = CircuitBreakerRegistry(
    default_config=CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=3,
        timeout_seconds=30.0,
        slow_call_threshold_ms=500.0,
    )
)


def get_verification_circuit() -> CircuitBreaker:
    """Get circuit breaker for verification engine."""
    return bizra_circuit_registry.get_or_create(
        "verification_engine",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=15.0,
            slow_call_threshold_ms=200.0,
        ),
    )


def get_value_oracle_circuit() -> CircuitBreaker:
    """Get circuit breaker for value oracle."""
    return bizra_circuit_registry.get_or_create(
        "value_oracle",
        config=CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=30.0,
            slow_call_threshold_ms=100.0,
        ),
    )


def get_ethics_circuit() -> CircuitBreaker:
    """Get circuit breaker for ethics engine."""
    return bizra_circuit_registry.get_or_create(
        "ethics_engine",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=60.0,  # Ethics is critical, longer timeout
            slow_call_threshold_ms=500.0,
        ),
    )


def get_quantum_security_circuit() -> CircuitBreaker:
    """Get circuit breaker for quantum security operations."""
    return bizra_circuit_registry.get_or_create(
        "quantum_security",
        config=CircuitBreakerConfig(
            failure_threshold=2,  # Security is critical
            timeout_seconds=60.0,
            slow_call_threshold_ms=1000.0,
        ),
    )
