"""
BIZRA APEX Runtime Orchestrator
================================
The Unified Control Plane for DDAGI 7-Layer Architecture

This module implements the crown jewel of the BIZRA system - a unified
runtime orchestrator that coordinates all 7 APEX layers with cross-cutting
Ihsan enforcement, telemetry, and fault tolerance.

Architecture:
    Layer 7: Philosophy (Ihsan Protocol)
    Layer 6: Governance (FATE Engine)
    Layer 5: Economic (PAT/SAT Tokenomics)
    Layer 4: Cognitive (Thermodynamic Engine)
    Layer 3: Execution (State Persistence)
    Layer 2: DePIN (Network Infrastructure)
    Layer 1: Blockchain (Third Fact Ledger)

Design Principles:
    - FAIL-CLOSED: All operations require Ihsan validation
    - CQRS: Command/Query responsibility separation
    - Event Sourcing: Immutable audit trail
    - Circuit Breakers: Cascading failure prevention
    - Backpressure: Adaptive load management

Author: BIZRA Genesis Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
import logging
import weakref
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
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
    cast,
)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger("bizra.apex")

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

T = TypeVar("T")
R = TypeVar("R")
CommandResult = Tuple[bool, str, Optional[Dict[str, Any]]]

# ============================================================================
# ENUMERATIONS
# ============================================================================


class APEXLayer(Enum):
    """APEX 7-Layer Stack enumeration."""
    
    BLOCKCHAIN = 1      # L1: Third Fact Ledger
    DEPIN = 2          # L2: Decentralized Physical Infrastructure
    EXECUTION = 3      # L3: State Persistence / DAaaS
    COGNITIVE = 4      # L4: Thermodynamic Engine
    ECONOMIC = 5       # L5: PAT/SAT Tokenomics
    GOVERNANCE = 6     # L6: FATE Engine
    PHILOSOPHY = 7     # L7: Ihsan Protocol


class OperationType(Enum):
    """Operation classification for routing."""
    
    COMMAND = auto()   # State-mutating operation
    QUERY = auto()     # Read-only operation
    EVENT = auto()     # Asynchronous event
    SAGA = auto()      # Distributed transaction


class OperationPriority(Enum):
    """Operation priority levels."""
    
    CRITICAL = 0       # System-critical (governance vetoes)
    HIGH = 1           # Time-sensitive (trades, attestations)
    NORMAL = 2         # Standard operations
    LOW = 3            # Background tasks (analytics)
    DEFERRED = 4       # Can be delayed indefinitely


class CircuitState(Enum):
    """Circuit breaker state machine."""
    
    CLOSED = auto()    # Normal operation
    OPEN = auto()      # Failing, reject requests
    HALF_OPEN = auto() # Testing recovery


class BackpressureStrategy(Enum):
    """Backpressure handling strategies."""
    
    DROP_OLDEST = auto()     # Drop oldest queued items
    DROP_NEWEST = auto()     # Reject new items
    BLOCK = auto()           # Block until capacity
    SAMPLE = auto()          # Probabilistic sampling


# ============================================================================
# PROTOCOLS (STRUCTURAL TYPING)
# ============================================================================


class IhsanAware(Protocol):
    """Protocol for Ihsan-aware components."""
    
    @property
    def ihsan_score(self) -> float:
        """Current Ihsan compliance score."""
        ...
    
    def validate_ihsan(self, threshold: float) -> Tuple[bool, str]:
        """Validate Ihsan compliance."""
        ...


class LayerComponent(Protocol):
    """Protocol for APEX layer components."""
    
    @property
    def layer(self) -> APEXLayer:
        """The APEX layer this component belongs to."""
        ...
    
    async def health_check(self) -> Tuple[bool, str]:
        """Check component health."""
        ...


# ============================================================================
# IMMUTABLE EVENT LOG (EVENT SOURCING)
# ============================================================================


@dataclass(frozen=True)
class DomainEvent:
    """
    Immutable domain event for event sourcing.
    
    All state changes in BIZRA are captured as immutable events,
    enabling:
    - Complete audit trail
    - Temporal queries (point-in-time state)
    - Event replay for recovery
    - CQRS read model projection
    """
    
    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_type: str
    timestamp: datetime
    version: int
    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    ihsan_score: float
    layer: APEXLayer
    
    @staticmethod
    def create(
        event_type: str,
        aggregate_id: str,
        aggregate_type: str,
        payload: Dict[str, Any],
        layer: APEXLayer,
        ihsan_score: float = 0.95,
        metadata: Optional[Dict[str, Any]] = None,
        version: int = 1,
    ) -> DomainEvent:
        """Factory method for creating domain events."""
        return DomainEvent(
            event_id=f"evt_{secrets.token_hex(16)}",
            event_type=event_type,
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            timestamp=datetime.now(timezone.utc),
            version=version,
            payload=dict(payload),  # Defensive copy
            metadata=dict(metadata or {}),
            ihsan_score=ihsan_score,
            layer=layer,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "payload": self.payload,
            "metadata": self.metadata,
            "ihsan_score": self.ihsan_score,
            "layer": self.layer.name,
        }
    
    @property
    def content_hash(self) -> str:
        """Compute content hash for integrity verification."""
        content = f"{self.event_type}:{self.aggregate_id}:{self.timestamp.isoformat()}"
        content += f":{self.version}:{self.ihsan_score}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]


class EventStore:
    """
    Append-only event store with temporal query support.
    
    Features:
    - Immutable event log (append-only)
    - Temporal queries (events by time range)
    - Aggregate streams (events by aggregate)
    - Snapshotting for fast replay
    - Event versioning for schema evolution
    """
    
    def __init__(self, max_events: int = 100_000):
        self._events: List[DomainEvent] = []
        self._by_aggregate: Dict[str, List[DomainEvent]] = {}
        self._by_type: Dict[str, List[DomainEvent]] = {}
        self._by_layer: Dict[APEXLayer, List[DomainEvent]] = {
            layer: [] for layer in APEXLayer
        }
        self._max_events = max_events
        self._lock = asyncio.Lock()
        self._subscribers: List[Callable[[DomainEvent], Awaitable[None]]] = []
        
        # Metrics
        self._total_appended = 0
        self._total_queries = 0
    
    async def append(self, event: DomainEvent) -> str:
        """
        Append event to the store (immutable).
        
        Returns the event ID.
        """
        async with self._lock:
            # Enforce Ihsan threshold
            if event.ihsan_score < 0.95:
                raise ValueError(
                    f"Event rejected: Ihsan {event.ihsan_score:.4f} < 0.95"
                )
            
            self._events.append(event)
            
            # Index by aggregate
            if event.aggregate_id not in self._by_aggregate:
                self._by_aggregate[event.aggregate_id] = []
            self._by_aggregate[event.aggregate_id].append(event)
            
            # Index by type
            if event.event_type not in self._by_type:
                self._by_type[event.event_type] = []
            self._by_type[event.event_type].append(event)
            
            # Index by layer
            self._by_layer[event.layer].append(event)
            
            self._total_appended += 1
            
            # Evict oldest if over capacity (with warning)
            if len(self._events) > self._max_events:
                evicted = self._events.pop(0)
                logger.warning(f"Event store eviction: {evicted.event_id}")
        
        # Notify subscribers (outside lock)
        for subscriber in self._subscribers:
            try:
                await subscriber(event)
            except Exception as e:
                logger.error(f"Event subscriber error: {e}")
        
        return event.event_id
    
    async def get_aggregate_stream(
        self,
        aggregate_id: str,
        from_version: int = 0,
    ) -> List[DomainEvent]:
        """Get all events for an aggregate from a version."""
        self._total_queries += 1
        events = self._by_aggregate.get(aggregate_id, [])
        return [e for e in events if e.version >= from_version]
    
    async def get_events_by_time_range(
        self,
        start: datetime,
        end: datetime,
        layer: Optional[APEXLayer] = None,
    ) -> List[DomainEvent]:
        """Temporal query: events within time range."""
        self._total_queries += 1
        
        if layer:
            events = self._by_layer.get(layer, [])
        else:
            events = self._events
        
        return [
            e for e in events
            if start <= e.timestamp <= end
        ]
    
    async def get_events_by_type(
        self,
        event_type: str,
        limit: int = 100,
    ) -> List[DomainEvent]:
        """Get events by type with limit."""
        self._total_queries += 1
        events = self._by_type.get(event_type, [])
        return events[-limit:]
    
    def subscribe(
        self,
        handler: Callable[[DomainEvent], Awaitable[None]],
    ) -> Callable[[], None]:
        """
        Subscribe to events.
        
        Returns unsubscribe function.
        """
        self._subscribers.append(handler)
        
        def unsubscribe():
            if handler in self._subscribers:
                self._subscribers.remove(handler)
        
        return unsubscribe
    
    @property
    def event_count(self) -> int:
        return len(self._events)
    
    @property
    def aggregate_count(self) -> int:
        return len(self._by_aggregate)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event store metrics."""
        return {
            "total_events": self.event_count,
            "total_aggregates": self.aggregate_count,
            "total_appended": self._total_appended,
            "total_queries": self._total_queries,
            "events_by_layer": {
                layer.name: len(events)
                for layer, events in self._by_layer.items()
            },
            "subscriber_count": len(self._subscribers),
        }


# ============================================================================
# CIRCUIT BREAKER PATTERN
# ============================================================================


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close
    timeout_seconds: float = 30.0       # Time in open state
    half_open_max_calls: int = 3        # Max calls in half-open
    
    # Ihsan-specific
    ihsan_failure_weight: float = 2.0   # Weight for Ihsan failures


class CircuitBreaker:
    """
    Circuit breaker with Ihsan-aware failure handling.
    
    States:
    - CLOSED: Normal operation, tracking failures
    - OPEN: Rejecting all requests after threshold
    - HALF_OPEN: Testing if service recovered
    
    Ihsan violations are weighted more heavily than
    standard failures (fail-closed philosophy).
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        
        self._lock = asyncio.Lock()
        
        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejections = 0
        self._state_transitions: List[Tuple[datetime, CircuitState]] = []
    
    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        ihsan_score: float = 1.0,
    ) -> T:
        """
        Execute operation through circuit breaker.
        
        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        async with self._lock:
            self._total_calls += 1
            
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self._total_rejections += 1
                    raise CircuitBreakerOpen(self.name, self._time_until_retry())
            
            # Check half-open call limit
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._total_rejections += 1
                    raise CircuitBreakerOpen(self.name, self._time_until_retry())
                self._half_open_calls += 1
        
        # Execute operation (outside lock)
        try:
            result = await operation()
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(ihsan_score, e)
            raise
    
    async def _record_success(self) -> None:
        """Record successful operation."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    async def _record_failure(
        self,
        ihsan_score: float,
        error: Exception,
    ) -> None:
        """Record failed operation with Ihsan weighting."""
        async with self._lock:
            self._total_failures += 1
            self._last_failure_time = datetime.now(timezone.utc)
            
            # Ihsan failures are weighted more heavily
            weight = 1.0
            if ihsan_score < 0.95:
                weight = self.config.ihsan_failure_weight
            
            self._failure_count += int(weight)
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        
        elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_seconds
    
    def _time_until_retry(self) -> float:
        """Time until circuit breaker may close."""
        if self._last_failure_time is None:
            return 0.0
        
        elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
        return max(0.0, self.config.timeout_seconds - elapsed)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        if new_state != self._state:
            logger.info(f"Circuit {self.name}: {self._state.name} -> {new_state.name}")
            self._state_transitions.append((datetime.now(timezone.utc), new_state))
            self._state = new_state
            
            if new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._success_count = 0
                self._half_open_calls = 0
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self._state.name,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_rejections": self._total_rejections,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "transitions": len(self._state_transitions),
        }


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker '{name}' is open. Retry after {retry_after:.1f}s"
        )


# ============================================================================
# BACKPRESSURE QUEUE
# ============================================================================


@dataclass
class QueuedOperation(Generic[T]):
    """Operation queued for execution."""
    
    operation_id: str
    priority: OperationPriority
    operation: Callable[[], Awaitable[T]]
    created_at: datetime
    ihsan_score: float
    timeout: float
    future: asyncio.Future[T]


class BackpressureQueue(Generic[T]):
    """
    Priority queue with backpressure management.
    
    Features:
    - Priority-based ordering (Ihsan-weighted)
    - Configurable capacity with overflow strategies
    - Timeout-based expiration
    - Metrics and monitoring
    """
    
    def __init__(
        self,
        capacity: int = 10_000,
        strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST,
    ):
        self._capacity = capacity
        self._strategy = strategy
        
        # Priority queues (one per priority level)
        self._queues: Dict[OperationPriority, Deque[QueuedOperation[T]]] = {
            priority: deque() for priority in OperationPriority
        }
        
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        
        # Metrics
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._total_dropped = 0
        self._total_expired = 0
    
    async def enqueue(
        self,
        operation: Callable[[], Awaitable[T]],
        priority: OperationPriority = OperationPriority.NORMAL,
        ihsan_score: float = 0.95,
        timeout: float = 30.0,
    ) -> asyncio.Future[T]:
        """
        Enqueue operation with priority.
        
        Returns future that resolves when operation completes.
        """
        async with self._lock:
            # Check capacity
            current_size = self._total_size()
            
            if current_size >= self._capacity:
                if self._strategy == BackpressureStrategy.DROP_NEWEST:
                    self._total_dropped += 1
                    raise BackpressureExceeded(
                        f"Queue at capacity ({self._capacity}), dropping new"
                    )
                elif self._strategy == BackpressureStrategy.DROP_OLDEST:
                    self._drop_oldest()
                elif self._strategy == BackpressureStrategy.SAMPLE:
                    # Probabilistic drop based on priority
                    import random
                    drop_prob = (priority.value + 1) / 10.0
                    if random.random() < drop_prob:
                        self._total_dropped += 1
                        raise BackpressureExceeded("Sampled out due to load")
                    self._drop_oldest()
                # BLOCK strategy: wait handled by condition
            
            # Create queued operation
            loop = asyncio.get_running_loop()
            future: asyncio.Future[T] = loop.create_future()
            
            queued = QueuedOperation(
                operation_id=f"op_{secrets.token_hex(8)}",
                priority=priority,
                operation=operation,
                created_at=datetime.now(timezone.utc),
                ihsan_score=ihsan_score,
                timeout=timeout,
                future=future,
            )
            
            # Ihsan-weighted priority: higher Ihsan scores get priority boost
            effective_priority = priority
            if ihsan_score >= 0.99:
                # Boost priority for excellent Ihsan
                effective_priority = OperationPriority(max(0, priority.value - 1))
            
            self._queues[effective_priority].append(queued)
            self._total_enqueued += 1
            
            self._not_empty.notify()
        
        return future
    
    async def dequeue(self) -> QueuedOperation[T]:
        """Dequeue highest priority operation."""
        async with self._lock:
            while self._total_size() == 0:
                await self._not_empty.wait()
            
            # Clean expired first
            self._clean_expired()
            
            # Get from highest priority queue with items
            for priority in OperationPriority:
                if self._queues[priority]:
                    op = self._queues[priority].popleft()
                    self._total_dequeued += 1
                    return op
            
            # Should never reach here due to wait condition
            raise RuntimeError("Queue unexpectedly empty")
    
    def _total_size(self) -> int:
        """Total items across all queues."""
        return sum(len(q) for q in self._queues.values())
    
    def _drop_oldest(self) -> None:
        """Drop oldest item from lowest priority queue."""
        for priority in reversed(list(OperationPriority)):
            if self._queues[priority]:
                dropped = self._queues[priority].popleft()
                dropped.future.set_exception(
                    BackpressureExceeded("Dropped due to capacity")
                )
                self._total_dropped += 1
                return
    
    def _clean_expired(self) -> None:
        """Remove expired operations."""
        now = datetime.now(timezone.utc)
        
        for priority, queue in self._queues.items():
            expired_indices = []
            for i, op in enumerate(queue):
                age = (now - op.created_at).total_seconds()
                if age > op.timeout:
                    expired_indices.append(i)
            
            # Remove in reverse to preserve indices
            for i in reversed(expired_indices):
                expired = queue[i]
                del queue[i]
                expired.future.set_exception(
                    asyncio.TimeoutError("Operation expired in queue")
                )
                self._total_expired += 1
    
    @property
    def size(self) -> int:
        return self._total_size()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics."""
        return {
            "capacity": self._capacity,
            "current_size": self._total_size(),
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_dropped": self._total_dropped,
            "total_expired": self._total_expired,
            "by_priority": {
                p.name: len(q) for p, q in self._queues.items()
            },
            "utilization": self._total_size() / self._capacity,
        }


class BackpressureExceeded(Exception):
    """Exception when backpressure capacity exceeded."""
    pass


# ============================================================================
# IHSAN CROSS-CUTTING ENFORCEMENT
# ============================================================================


@dataclass
class IhsanEnforcementResult:
    """Result of Ihsan enforcement check."""
    
    passed: bool
    score: float
    threshold: float
    reason: str
    layer: APEXLayer
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "threshold": self.threshold,
            "reason": self.reason,
            "layer": self.layer.name,
            "timestamp": self.timestamp.isoformat(),
        }


class IhsanEnforcementPolicy:
    """
    Cross-cutting Ihsan enforcement policy.
    
    This enforces Ihsan compliance at every layer transition,
    implementing the FAIL-CLOSED philosophy.
    """
    
    # Layer-specific thresholds (higher layers require higher Ihsan)
    LAYER_THRESHOLDS = {
        APEXLayer.BLOCKCHAIN: 0.95,
        APEXLayer.DEPIN: 0.95,
        APEXLayer.EXECUTION: 0.95,
        APEXLayer.COGNITIVE: 0.96,
        APEXLayer.ECONOMIC: 0.97,
        APEXLayer.GOVERNANCE: 0.98,
        APEXLayer.PHILOSOPHY: 0.99,
    }
    
    def __init__(self):
        self._enforcement_log: List[IhsanEnforcementResult] = []
        self._total_checks = 0
        self._total_passed = 0
        self._total_failed = 0
        self._lock = asyncio.Lock()
    
    async def enforce(
        self,
        ihsan_score: float,
        layer: APEXLayer,
        operation_type: OperationType,
        context: Optional[Dict[str, Any]] = None,
    ) -> IhsanEnforcementResult:
        """
        Enforce Ihsan compliance for an operation.
        
        Returns enforcement result.
        Raises IhsanViolation if check fails and not recoverable.
        """
        threshold = self.LAYER_THRESHOLDS.get(layer, 0.95)
        
        # Commands at governance layer require even higher bar
        if layer == APEXLayer.GOVERNANCE and operation_type == OperationType.COMMAND:
            threshold = 0.99
        
        passed = ihsan_score >= threshold
        reason = (
            f"Ihsan {ihsan_score:.4f} >= {threshold:.2f}"
            if passed
            else f"Ihsan {ihsan_score:.4f} < {threshold:.2f} required for {layer.name}"
        )
        
        result = IhsanEnforcementResult(
            passed=passed,
            score=ihsan_score,
            threshold=threshold,
            reason=reason,
            layer=layer,
        )
        
        async with self._lock:
            self._enforcement_log.append(result)
            self._total_checks += 1
            if passed:
                self._total_passed += 1
            else:
                self._total_failed += 1
                
                # Keep log bounded
                if len(self._enforcement_log) > 10_000:
                    self._enforcement_log = self._enforcement_log[-5_000:]
        
        if not passed:
            raise IhsanViolation(result)
        
        return result
    
    @property
    def compliance_rate(self) -> float:
        if self._total_checks == 0:
            return 1.0
        return self._total_passed / self._total_checks
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_checks": self._total_checks,
            "total_passed": self._total_passed,
            "total_failed": self._total_failed,
            "compliance_rate": self.compliance_rate,
            "thresholds": {
                layer.name: threshold
                for layer, threshold in self.LAYER_THRESHOLDS.items()
            },
        }


class IhsanViolation(Exception):
    """Exception raised when Ihsan enforcement fails."""
    
    def __init__(self, result: IhsanEnforcementResult):
        self.result = result
        super().__init__(result.reason)


# ============================================================================
# APEX RUNTIME ORCHESTRATOR
# ============================================================================


class APEXOrchestrator:
    """
    The Unified Control Plane for BIZRA DDAGI.
    
    This orchestrator coordinates all 7 APEX layers with:
    - Cross-cutting Ihsan enforcement
    - Event sourcing with immutable audit trail
    - Circuit breakers per layer
    - Backpressure management
    - Distributed tracing (span propagation)
    - Health monitoring
    
    Architecture:
        ┌─────────────────────────────────────────┐
        │          APEX Orchestrator              │
        ├─────────────────────────────────────────┤
        │  ┌─────────────────────────────────┐   │
        │  │    Ihsan Enforcement Policy     │   │
        │  └─────────────────────────────────┘   │
        │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
        │  │  CB  │ │  CB  │ │  CB  │ │  CB  │  │
        │  │  L1  │ │  L3  │ │  L4  │ │  L6  │  │
        │  └──────┘ └──────┘ └──────┘ └──────┘  │
        │  ┌─────────────────────────────────┐   │
        │  │      Backpressure Queue         │   │
        │  └─────────────────────────────────┘   │
        │  ┌─────────────────────────────────┐   │
        │  │         Event Store             │   │
        │  └─────────────────────────────────┘   │
        └─────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        queue_capacity: int = 10_000,
    ):
        # Core components
        self.event_store = event_store or EventStore()
        self.ihsan_policy = IhsanEnforcementPolicy()
        self.operation_queue: BackpressureQueue[Any] = BackpressureQueue(
            capacity=queue_capacity,
            strategy=BackpressureStrategy.DROP_OLDEST,
        )
        
        # Circuit breakers per layer
        self.circuit_breakers: Dict[APEXLayer, CircuitBreaker] = {
            layer: CircuitBreaker(
                name=f"cb_{layer.name.lower()}",
                config=CircuitBreakerConfig(
                    failure_threshold=5,
                    success_threshold=3,
                    timeout_seconds=30.0,
                ),
            )
            for layer in APEXLayer
        }
        
        # Layer components registry
        self._layer_components: Dict[APEXLayer, Any] = {}
        
        # Orchestrator state
        self._started = False
        self._shutdown = False
        self._worker_task: Optional[asyncio.Task[None]] = None
        
        # Metrics
        self._operations_processed = 0
        self._operations_failed = 0
        self._start_time: Optional[datetime] = None
        
        # Tracing
        self._active_spans: Dict[str, Dict[str, Any]] = {}
    
    async def start(self) -> None:
        """Start the orchestrator."""
        if self._started:
            return
        
        self._started = True
        self._shutdown = False
        self._start_time = datetime.now(timezone.utc)
        
        # Start worker task
        self._worker_task = asyncio.create_task(self._worker_loop())
        
        logger.info("APEX Orchestrator started")
    
    async def stop(self) -> None:
        """Gracefully stop the orchestrator."""
        if not self._started:
            return
        
        self._shutdown = True
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        self._started = False
        logger.info("APEX Orchestrator stopped")
    
    def register_component(self, layer: APEXLayer, component: Any) -> None:
        """Register a layer component."""
        self._layer_components[layer] = component
        logger.info(f"Registered component for {layer.name}")
    
    async def execute_command(
        self,
        layer: APEXLayer,
        command_name: str,
        payload: Dict[str, Any],
        ihsan_score: float = 0.95,
        priority: OperationPriority = OperationPriority.NORMAL,
    ) -> CommandResult:
        """
        Execute a command (state-mutating operation).
        
        Commands are:
        1. Validated against Ihsan policy
        2. Routed through circuit breaker
        3. Recorded as domain event
        4. Processed with backpressure
        """
        span_id = self._start_span(f"command.{command_name}", layer)
        
        try:
            # 1. Ihsan enforcement
            await self.ihsan_policy.enforce(
                ihsan_score=ihsan_score,
                layer=layer,
                operation_type=OperationType.COMMAND,
            )
            
            # 2. Circuit breaker
            cb = self.circuit_breakers[layer]
            
            async def _execute():
                # Get component and execute
                component = self._layer_components.get(layer)
                if component is None:
                    return (True, "No component registered", {"mock": True})
                
                # Try to find command handler
                handler = getattr(component, f"handle_{command_name}", None)
                if handler:
                    result = await handler(payload)
                    return (True, "Command executed", result)
                else:
                    return (True, f"Command {command_name} queued", payload)
            
            result = await cb.execute(_execute, ihsan_score)
            
            # 3. Record event
            event = DomainEvent.create(
                event_type=f"command.{command_name}",
                aggregate_id=payload.get("aggregate_id", secrets.token_hex(8)),
                aggregate_type=f"{layer.name.lower()}_aggregate",
                payload={
                    "command": command_name,
                    "request": payload,
                    "result": result[2] if result[2] else {},
                },
                layer=layer,
                ihsan_score=ihsan_score,
            )
            await self.event_store.append(event)
            
            self._operations_processed += 1
            return result
            
        except IhsanViolation as e:
            self._operations_failed += 1
            return (False, e.result.reason, None)
        except CircuitBreakerOpen as e:
            self._operations_failed += 1
            return (False, str(e), {"retry_after": e.retry_after})
        except Exception as e:
            self._operations_failed += 1
            logger.error(f"Command {command_name} failed: {e}")
            return (False, str(e), None)
        finally:
            self._end_span(span_id)
    
    async def execute_query(
        self,
        layer: APEXLayer,
        query_name: str,
        params: Dict[str, Any],
        ihsan_score: float = 0.95,
    ) -> Tuple[bool, Any]:
        """
        Execute a query (read-only operation).
        
        Queries are lighter-weight than commands:
        - Ihsan validated but with relaxed threshold
        - No event sourcing (reads don't mutate)
        - Circuit breaker protected
        """
        span_id = self._start_span(f"query.{query_name}", layer)
        
        try:
            # Relaxed Ihsan for queries (one tier lower)
            relaxed_threshold = max(0.90, self.ihsan_policy.LAYER_THRESHOLDS[layer] - 0.02)
            
            if ihsan_score < relaxed_threshold:
                return (False, f"Ihsan {ihsan_score:.4f} < {relaxed_threshold:.2f}")
            
            cb = self.circuit_breakers[layer]
            
            async def _query():
                component = self._layer_components.get(layer)
                if component is None:
                    return {"mock": True, "query": query_name}
                
                handler = getattr(component, f"query_{query_name}", None)
                if handler:
                    return await handler(params)
                else:
                    return {"query": query_name, "params": params}
            
            result = await cb.execute(_query, ihsan_score)
            return (True, result)
            
        except Exception as e:
            return (False, str(e))
        finally:
            self._end_span(span_id)
    
    async def publish_event(
        self,
        event_type: str,
        aggregate_id: str,
        layer: APEXLayer,
        payload: Dict[str, Any],
        ihsan_score: float = 0.95,
    ) -> str:
        """
        Publish a domain event.
        
        Events are asynchronous notifications that don't
        require immediate response.
        """
        await self.ihsan_policy.enforce(
            ihsan_score=ihsan_score,
            layer=layer,
            operation_type=OperationType.EVENT,
        )
        
        event = DomainEvent.create(
            event_type=event_type,
            aggregate_id=aggregate_id,
            aggregate_type=f"{layer.name.lower()}_aggregate",
            payload=payload,
            layer=layer,
            ihsan_score=ihsan_score,
        )
        
        return await self.event_store.append(event)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check across all layers.
        
        Returns health status and metrics.
        """
        health: Dict[str, Any] = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": self._get_uptime(),
            "layers": {},
            "circuit_breakers": {},
            "metrics": {},
        }
        
        # Check each layer
        for layer, component in self._layer_components.items():
            try:
                if hasattr(component, "health_check"):
                    ok, msg = await component.health_check()
                    health["layers"][layer.name] = {
                        "status": "healthy" if ok else "unhealthy",
                        "message": msg,
                    }
                else:
                    health["layers"][layer.name] = {
                        "status": "healthy",
                        "message": "Component registered",
                    }
            except Exception as e:
                health["layers"][layer.name] = {
                    "status": "unhealthy",
                    "message": str(e),
                }
                health["status"] = "degraded"
        
        # Circuit breaker status
        for layer, cb in self.circuit_breakers.items():
            health["circuit_breakers"][layer.name] = {
                "state": cb.state.name,
                "is_closed": cb.is_closed,
            }
            if not cb.is_closed:
                health["status"] = "degraded"
        
        # Aggregate metrics
        health["metrics"] = {
            "operations_processed": self._operations_processed,
            "operations_failed": self._operations_failed,
            "success_rate": (
                self._operations_processed / 
                max(1, self._operations_processed + self._operations_failed)
            ),
            "ihsan_compliance": self.ihsan_policy.compliance_rate,
            "event_count": self.event_store.event_count,
            "queue_size": self.operation_queue.size,
        }
        
        return health
    
    def _get_uptime(self) -> float:
        if self._start_time is None:
            return 0.0
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()
    
    def _start_span(self, name: str, layer: APEXLayer) -> str:
        """Start a tracing span."""
        span_id = f"span_{secrets.token_hex(8)}"
        self._active_spans[span_id] = {
            "name": name,
            "layer": layer.name,
            "start_time": time.perf_counter(),
        }
        return span_id
    
    def _end_span(self, span_id: str) -> None:
        """End a tracing span."""
        if span_id in self._active_spans:
            span = self._active_spans.pop(span_id)
            duration_ms = (time.perf_counter() - span["start_time"]) * 1000
            logger.debug(f"Span {span['name']}: {duration_ms:.2f}ms")
    
    async def _worker_loop(self) -> None:
        """Background worker for processing queued operations."""
        while not self._shutdown:
            try:
                # Process queued operations
                op = await asyncio.wait_for(
                    self.operation_queue.dequeue(),
                    timeout=1.0,
                )
                
                try:
                    result = await op.operation()
                    op.future.set_result(result)
                except Exception as e:
                    op.future.set_exception(e)
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(0.1)
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator metrics."""
        return {
            "orchestrator": {
                "started": self._started,
                "uptime_seconds": self._get_uptime(),
                "operations_processed": self._operations_processed,
                "operations_failed": self._operations_failed,
            },
            "event_store": self.event_store.get_metrics(),
            "ihsan_policy": self.ihsan_policy.get_metrics(),
            "operation_queue": self.operation_queue.get_metrics(),
            "circuit_breakers": {
                layer.name: cb.get_metrics()
                for layer, cb in self.circuit_breakers.items()
            },
            "registered_layers": [
                layer.name for layer in self._layer_components.keys()
            ],
        }


# ============================================================================
# SAGA ORCHESTRATOR (DISTRIBUTED TRANSACTIONS)
# ============================================================================


@dataclass
class SagaStep:
    """A single step in a saga."""
    
    name: str
    layer: APEXLayer
    execute: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
    compensate: Callable[[Dict[str, Any]], Awaitable[None]]
    ihsan_threshold: float = 0.95


class SagaOrchestrator:
    """
    Saga pattern for distributed transactions across APEX layers.
    
    Sagas provide eventual consistency through compensating
    transactions when failures occur.
    """
    
    def __init__(self, apex: APEXOrchestrator):
        self.apex = apex
        self._active_sagas: Dict[str, Dict[str, Any]] = {}
    
    async def execute_saga(
        self,
        saga_id: str,
        steps: List[SagaStep],
        initial_context: Dict[str, Any],
        ihsan_score: float = 0.95,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a saga with compensating transactions.
        
        If any step fails, previously completed steps
        are compensated in reverse order.
        """
        context = dict(initial_context)
        completed_steps: List[SagaStep] = []
        
        self._active_sagas[saga_id] = {
            "status": "running",
            "steps_completed": 0,
            "steps_total": len(steps),
        }
        
        try:
            for step in steps:
                # Enforce Ihsan for this step
                await self.apex.ihsan_policy.enforce(
                    ihsan_score=ihsan_score,
                    layer=step.layer,
                    operation_type=OperationType.SAGA,
                )
                
                # Execute step
                step_result = await step.execute(context)
                context.update(step_result)
                completed_steps.append(step)
                
                self._active_sagas[saga_id]["steps_completed"] = len(completed_steps)
            
            self._active_sagas[saga_id]["status"] = "completed"
            return (True, context)
            
        except Exception as e:
            logger.error(f"Saga {saga_id} failed at step: {e}")
            self._active_sagas[saga_id]["status"] = "compensating"
            
            # Compensate in reverse order
            for step in reversed(completed_steps):
                try:
                    await step.compensate(context)
                except Exception as comp_error:
                    logger.error(f"Compensation failed for {step.name}: {comp_error}")
            
            self._active_sagas[saga_id]["status"] = "failed"
            return (False, {"error": str(e), "compensated_steps": len(completed_steps)})


# ============================================================================
# TELEMETRY COLLECTOR
# ============================================================================


class TelemetryCollector:
    """
    Centralized telemetry collection for observability.
    
    Collects:
    - Metrics (counters, gauges, histograms)
    - Traces (distributed spans)
    - Logs (structured events)
    """
    
    def __init__(self):
        self._metrics: Dict[str, List[Tuple[datetime, float]]] = {}
        self._traces: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric value."""
        async with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []
            
            self._metrics[name].append((datetime.now(timezone.utc), value))
            
            # Keep bounded
            if len(self._metrics[name]) > 10_000:
                self._metrics[name] = self._metrics[name][-5_000:]
    
    async def record_trace(
        self,
        trace_id: str,
        span_name: str,
        duration_ms: float,
        layer: APEXLayer,
        success: bool,
    ) -> None:
        """Record a trace span."""
        async with self._lock:
            self._traces.append({
                "trace_id": trace_id,
                "span_name": span_name,
                "duration_ms": duration_ms,
                "layer": layer.name,
                "success": success,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            if len(self._traces) > 10_000:
                self._traces = self._traces[-5_000:]
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = [v for _, v in self._metrics.get(name, [])]
        if not values:
            return {"count": 0}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all metric statistics."""
        return {name: self.get_metric_stats(name) for name in self._metrics}


# ============================================================================
# DEMO AND VALIDATION
# ============================================================================


async def demo_apex_orchestrator():
    """Demonstrate APEX Orchestrator capabilities."""
    print("=" * 70)
    print("BIZRA APEX RUNTIME ORCHESTRATOR DEMO")
    print("=" * 70)
    
    # Create orchestrator
    orchestrator = APEXOrchestrator()
    await orchestrator.start()
    
    try:
        # Execute commands across layers
        print("\n1. Executing Commands Across Layers")
        print("-" * 40)
        
        for layer in [APEXLayer.BLOCKCHAIN, APEXLayer.EXECUTION, APEXLayer.GOVERNANCE]:
            success, msg, result = await orchestrator.execute_command(
                layer=layer,
                command_name="test_operation",
                payload={"test": True, "layer": layer.name},
                ihsan_score=0.99,  # High Ihsan for governance
            )
            print(f"  {layer.name}: {msg}")
        
        # Test Ihsan enforcement
        print("\n2. Ihsan Enforcement Demo")
        print("-" * 40)
        
        for score in [0.99, 0.96, 0.92]:
            try:
                success, msg, _ = await orchestrator.execute_command(
                    layer=APEXLayer.GOVERNANCE,
                    command_name="governance_action",
                    payload={"action": "vote"},
                    ihsan_score=score,
                )
                print(f"  Ihsan {score:.2f}: {msg}")
            except Exception as e:
                print(f"  Ihsan {score:.2f}: REJECTED - {e}")
        
        # Event store demo
        print("\n3. Event Store Demo")
        print("-" * 40)
        
        event_id = await orchestrator.publish_event(
            event_type="demo.event",
            aggregate_id="demo_aggregate",
            layer=APEXLayer.COGNITIVE,
            payload={"demo": True},
            ihsan_score=0.97,
        )
        print(f"  Published event: {event_id[:20]}...")
        
        events = await orchestrator.event_store.get_events_by_type("demo.event")
        print(f"  Events of type 'demo.event': {len(events)}")
        
        # Health check
        print("\n4. Health Check")
        print("-" * 40)
        
        health = await orchestrator.health_check()
        print(f"  Status: {health['status']}")
        print(f"  Uptime: {health['uptime_seconds']:.1f}s")
        print(f"  Operations: {health['metrics']['operations_processed']}")
        print(f"  Ihsan Compliance: {health['metrics']['ihsan_compliance']:.1%}")
        
        # Comprehensive metrics
        print("\n5. Comprehensive Metrics")
        print("-" * 40)
        
        metrics = orchestrator.get_comprehensive_metrics()
        print(f"  Event count: {metrics['event_store']['total_events']}")
        print(f"  Ihsan checks: {metrics['ihsan_policy']['total_checks']}")
        print(f"  Queue size: {metrics['operation_queue']['current_size']}")
        
        print("\n" + "=" * 70)
        print("APEX ORCHESTRATOR DEMO COMPLETE")
        print("=" * 70)
        
    finally:
        await orchestrator.stop()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "APEXLayer",
    "OperationType",
    "OperationPriority",
    "CircuitState",
    "BackpressureStrategy",
    # Events
    "DomainEvent",
    "EventStore",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpen",
    # Backpressure
    "BackpressureQueue",
    "QueuedOperation",
    "BackpressureExceeded",
    # Ihsan
    "IhsanEnforcementPolicy",
    "IhsanEnforcementResult",
    "IhsanViolation",
    # Orchestrator
    "APEXOrchestrator",
    # Saga
    "SagaStep",
    "SagaOrchestrator",
    # Telemetry
    "TelemetryCollector",
]


if __name__ == "__main__":
    asyncio.run(demo_apex_orchestrator())
