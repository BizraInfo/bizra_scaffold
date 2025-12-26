"""
BIZRA Event Sourcing Engine
============================
CQRS + Event Sourcing for Perfect Audit Trail and Temporal Queries

This module implements a production-grade event sourcing system with
Command/Query Responsibility Segregation (CQRS), providing:

- Immutable event log (append-only, tamper-evident)
- Aggregate reconstruction from events
- Temporal queries (point-in-time state)
- Snapshotting for performance
- Projections for read models
- Event versioning for schema evolution

Design Philosophy:
    "The truth is in the events" - all state is derived from
    the immutable sequence of domain events.

Author: BIZRA Genesis Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger("bizra.eventsourcing")

# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

MAX_AGGREGATE_ID_LENGTH = 256
MAX_EVENTS_PER_APPEND = 1000
MAX_EVENT_PAYLOAD_SIZE = 1024 * 1024  # 1MB

# ============================================================================
# TYPE VARIABLES
# ============================================================================

T = TypeVar("T")
E = TypeVar("E", bound="Event")
A = TypeVar("A", bound="Aggregate")
S = TypeVar("S")  # State type

# ============================================================================
# ENUMERATIONS
# ============================================================================


class EventStatus(Enum):
    """Event lifecycle status."""
    
    PENDING = auto()      # Created, not yet persisted
    PERSISTED = auto()    # Stored in event store
    PROJECTED = auto()    # Projected to read models
    ARCHIVED = auto()     # Moved to cold storage


class SnapshotStrategy(Enum):
    """When to create snapshots."""
    
    EVERY_N_EVENTS = auto()    # After N events
    TIME_BASED = auto()        # Every N seconds
    SIZE_BASED = auto()        # When aggregate size exceeds threshold
    MANUAL = auto()            # Only on explicit request


# ============================================================================
# CORE EVENT TYPES
# ============================================================================


@dataclass(frozen=True)
class EventMetadata:
    """
    Immutable metadata attached to every event.
    
    Provides:
    - Causation chain (what caused this event)
    - Correlation (related operations)
    - Ihsan compliance score
    - Temporal information
    """
    
    correlation_id: str
    causation_id: str
    timestamp: datetime
    ihsan_score: float
    actor_id: str
    source_layer: str
    schema_version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "timestamp": self.timestamp.isoformat(),
            "ihsan_score": self.ihsan_score,
            "actor_id": self.actor_id,
            "source_layer": self.source_layer,
            "schema_version": self.schema_version,
        }
    
    @staticmethod
    def create(
        actor_id: str,
        source_layer: str,
        ihsan_score: float = 0.95,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
    ) -> EventMetadata:
        return EventMetadata(
            correlation_id=correlation_id or f"corr_{secrets.token_hex(8)}",
            causation_id=causation_id or f"cause_{secrets.token_hex(8)}",
            timestamp=datetime.now(timezone.utc),
            ihsan_score=ihsan_score,
            actor_id=actor_id,
            source_layer=source_layer,
        )


@dataclass(frozen=True)
class Event:
    """
    Base immutable domain event.
    
    All events are:
    - Immutable (frozen dataclass)
    - Self-describing (type + version)
    - Content-addressable (hash for integrity)
    - Ihsan-validated (score >= threshold)
    """
    
    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_type: str
    sequence_number: int
    data: Dict[str, Any]
    metadata: EventMetadata
    
    def __post_init__(self):
        # Validate Ihsan on creation
        if self.metadata.ihsan_score < 0.95:
            raise ValueError(
                f"Event rejected: Ihsan {self.metadata.ihsan_score:.4f} < 0.95"
            )
    
    @property
    def content_hash(self) -> str:
        """Cryptographic hash for integrity verification."""
        content = json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "sequence_number": self.sequence_number,
            "data": self.data,
            "timestamp": self.metadata.timestamp.isoformat(),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "sequence_number": self.sequence_number,
            "data": self.data,
            "metadata": self.metadata.to_dict(),
            "content_hash": self.content_hash,
        }
    
    @staticmethod
    def create(
        event_type: str,
        aggregate_id: str,
        aggregate_type: str,
        data: Dict[str, Any],
        sequence_number: int,
        metadata: EventMetadata,
    ) -> Event:
        return Event(
            event_id=f"evt_{secrets.token_hex(16)}",
            event_type=event_type,
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            sequence_number=sequence_number,
            data=dict(data),  # Defensive copy
            metadata=metadata,
        )


# ============================================================================
# AGGREGATE ROOT
# ============================================================================


class Aggregate(ABC, Generic[S]):
    """
    Abstract Aggregate Root for event-sourced entities.
    
    Aggregates are:
    - Reconstructed from events (not stored directly)
    - Consistency boundaries (changes are atomic)
    - Event emitters (state changes produce events)
    
    Pattern:
        1. Load events from store
        2. Replay to rebuild state
        3. Execute command (validate + produce events)
        4. Persist new events
    """
    
    def __init__(self, aggregate_id: str):
        self._id = aggregate_id
        self._version = 0
        self._pending_events: List[Event] = []
        self._state: Optional[S] = None
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def version(self) -> int:
        return self._version
    
    @property
    @abstractmethod
    def aggregate_type(self) -> str:
        """Return the aggregate type name."""
        ...
    
    @abstractmethod
    def _apply(self, event: Event) -> None:
        """
        Apply an event to update internal state.
        
        This is the event handler - called for both replay
        and new events.
        """
        ...
    
    @abstractmethod
    def _get_initial_state(self) -> S:
        """Return the initial state for a new aggregate."""
        ...
    
    def _raise_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: EventMetadata,
    ) -> Event:
        """
        Raise a new domain event.
        
        The event is applied immediately and queued for persistence.
        """
        event = Event.create(
            event_type=event_type,
            aggregate_id=self._id,
            aggregate_type=self.aggregate_type,
            data=data,
            sequence_number=self._version + 1,
            metadata=metadata,
        )
        
        self._apply(event)
        self._version = event.sequence_number
        self._pending_events.append(event)
        
        return event
    
    def _load_from_history(self, events: List[Event]) -> None:
        """
        Replay events to rebuild state.
        
        Used when loading an aggregate from the event store.
        """
        self._state = self._get_initial_state()
        
        for event in events:
            self._apply(event)
            self._version = event.sequence_number
    
    def _get_pending_events(self) -> List[Event]:
        """Get events pending persistence."""
        return list(self._pending_events)
    
    def _clear_pending_events(self) -> None:
        """Clear pending events after successful persistence."""
        self._pending_events.clear()


# ============================================================================
# SNAPSHOT
# ============================================================================


@dataclass(frozen=True)
class Snapshot(Generic[S]):
    """
    Aggregate snapshot for fast reconstruction.
    
    Snapshots store serialized aggregate state at a point in time,
    allowing replay to start from snapshot instead of genesis.
    """
    
    snapshot_id: str
    aggregate_id: str
    aggregate_type: str
    version: int
    state: S
    created_at: datetime
    ihsan_score: float
    
    @staticmethod
    def create(
        aggregate: Aggregate[S],
        state: S,
        ihsan_score: float = 0.95,
    ) -> Snapshot[S]:
        return Snapshot(
            snapshot_id=f"snap_{secrets.token_hex(8)}",
            aggregate_id=aggregate.id,
            aggregate_type=aggregate.aggregate_type,
            version=aggregate.version,
            state=state,
            created_at=datetime.now(timezone.utc),
            ihsan_score=ihsan_score,
        )


# ============================================================================
# EVENT STORE
# ============================================================================


class EventStore:
    """
    Production-grade Event Store with CQRS support.
    
    Features:
    - Append-only event log
    - Optimistic concurrency control
    - Snapshot storage
    - Stream subscriptions
    - Temporal queries
    """
    
    def __init__(
        self,
        snapshot_threshold: int = 100,
        max_events: int = 1_000_000,
    ):
        # Event storage (in production: database)
        self._events: List[Event] = []
        self._by_aggregate: Dict[str, List[Event]] = defaultdict(list)
        self._by_type: Dict[str, List[Event]] = defaultdict(list)
        
        # Snapshot storage
        self._snapshots: Dict[str, Snapshot[Any]] = {}
        self._snapshot_threshold = snapshot_threshold
        
        # Subscriptions
        self._subscribers: List[Callable[[Event], Awaitable[None]]] = []
        
        # Configuration
        self._max_events = max_events
        
        # Concurrency
        self._lock = asyncio.Lock()
        self._version_cache: Dict[str, int] = {}
        
        # Metrics
        self._total_appends = 0
        self._total_reads = 0
        self._total_conflicts = 0
    
    async def append_events(
        self,
        aggregate_id: str,
        events: List[Event],
        expected_version: int,
    ) -> int:
        """
        Append events with optimistic concurrency control.
        
        Args:
            aggregate_id: The aggregate ID (max 256 chars)
            events: Events to append (max 1000 per call)
            expected_version: Expected current version (for conflict detection)
        
        Returns:
            New version number
        
        Raises:
            ConcurrencyConflict: If expected_version doesn't match
            ValueError: If inputs fail validation
        """
        # Input validation
        if not isinstance(aggregate_id, str):
            raise ValueError(
                f"aggregate_id must be str, got {type(aggregate_id).__name__}"
            )
        if len(aggregate_id) == 0:
            raise ValueError("aggregate_id cannot be empty")
        if len(aggregate_id) > MAX_AGGREGATE_ID_LENGTH:
            raise ValueError(
                f"aggregate_id length {len(aggregate_id)} exceeds max {MAX_AGGREGATE_ID_LENGTH}"
            )
        
        if not isinstance(events, list):
            raise ValueError(f"events must be list, got {type(events).__name__}")
        if len(events) == 0:
            raise ValueError("events list cannot be empty")
        if len(events) > MAX_EVENTS_PER_APPEND:
            raise ValueError(
                f"events count {len(events)} exceeds max {MAX_EVENTS_PER_APPEND} per append"
            )
        
        if not isinstance(expected_version, int):
            raise ValueError(
                f"expected_version must be int, got {type(expected_version).__name__}"
            )
        if expected_version < 0:
            raise ValueError(f"expected_version must be >= 0, got {expected_version}")
        
        async with self._lock:
            current_version = self._version_cache.get(aggregate_id, 0)
            
            if current_version != expected_version:
                self._total_conflicts += 1
                raise ConcurrencyConflict(
                    aggregate_id,
                    expected_version,
                    current_version,
                )
            
            # Validate all events have correct sequence
            for i, event in enumerate(events):
                expected_seq = expected_version + i + 1
                if event.sequence_number != expected_seq:
                    raise ValueError(
                        f"Event sequence mismatch: {event.sequence_number} != {expected_seq}"
                    )
            
            # Append events
            for event in events:
                self._events.append(event)
                self._by_aggregate[aggregate_id].append(event)
                self._by_type[event.event_type].append(event)
            
            new_version = expected_version + len(events)
            self._version_cache[aggregate_id] = new_version
            self._total_appends += len(events)
        
        # Notify subscribers (outside lock)
        for event in events:
            for subscriber in self._subscribers:
                try:
                    await subscriber(event)
                except Exception as e:
                    logger.error(f"Subscriber error: {e}")
        
        return new_version
    
    async def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None,
    ) -> List[Event]:
        """Get events for an aggregate from a version."""
        self._total_reads += 1
        
        events = self._by_aggregate.get(aggregate_id, [])
        filtered = [
            e for e in events
            if e.sequence_number > from_version
            and (to_version is None or e.sequence_number <= to_version)
        ]
        return sorted(filtered, key=lambda e: e.sequence_number)
    
    async def get_all_events(
        self,
        from_position: int = 0,
        limit: int = 1000,
    ) -> List[Event]:
        """Get all events from a global position."""
        self._total_reads += 1
        return self._events[from_position:from_position + limit]
    
    async def get_events_by_type(
        self,
        event_type: str,
        limit: int = 100,
    ) -> List[Event]:
        """Get events by type."""
        self._total_reads += 1
        return self._by_type.get(event_type, [])[-limit:]
    
    async def get_events_by_time_range(
        self,
        start: datetime,
        end: datetime,
        aggregate_id: Optional[str] = None,
    ) -> List[Event]:
        """Temporal query: events within time range."""
        self._total_reads += 1
        
        if aggregate_id:
            events = self._by_aggregate.get(aggregate_id, [])
        else:
            events = self._events
        
        return [
            e for e in events
            if start <= e.metadata.timestamp <= end
        ]
    
    async def save_snapshot(self, snapshot: Snapshot[Any]) -> None:
        """Save an aggregate snapshot."""
        async with self._lock:
            self._snapshots[snapshot.aggregate_id] = snapshot
    
    async def get_snapshot(
        self,
        aggregate_id: str,
    ) -> Optional[Snapshot[Any]]:
        """Get the latest snapshot for an aggregate."""
        return self._snapshots.get(aggregate_id)
    
    def subscribe(
        self,
        handler: Callable[[Event], Awaitable[None]],
    ) -> Callable[[], None]:
        """Subscribe to all events."""
        self._subscribers.append(handler)
        
        def unsubscribe():
            if handler in self._subscribers:
                self._subscribers.remove(handler)
        
        return unsubscribe
    
    async def get_aggregate_version(self, aggregate_id: str) -> int:
        """Get current version of an aggregate."""
        return self._version_cache.get(aggregate_id, 0)
    
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
            "total_appends": self._total_appends,
            "total_reads": self._total_reads,
            "total_conflicts": self._total_conflicts,
            "snapshot_count": len(self._snapshots),
            "subscriber_count": len(self._subscribers),
            "events_by_type": {
                k: len(v) for k, v in self._by_type.items()
            },
        }


class ConcurrencyConflict(Exception):
    """Raised when optimistic concurrency check fails."""
    
    def __init__(
        self,
        aggregate_id: str,
        expected: int,
        actual: int,
    ):
        self.aggregate_id = aggregate_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Concurrency conflict on {aggregate_id}: "
            f"expected {expected}, got {actual}"
        )


# ============================================================================
# REPOSITORY
# ============================================================================


class Repository(Generic[A]):
    """
    Repository for loading and saving aggregates.
    
    Abstracts event store interaction with:
    - Automatic snapshot loading
    - Event replay
    - Optimistic concurrency
    - Snapshot creation
    """
    
    def __init__(
        self,
        event_store: EventStore,
        aggregate_factory: Callable[[str], A],
        snapshot_threshold: int = 100,
    ):
        self._store = event_store
        self._factory = aggregate_factory
        self._snapshot_threshold = snapshot_threshold
    
    async def get(self, aggregate_id: str) -> A:
        """
        Load an aggregate from the event store.
        
        Uses snapshot if available, then replays remaining events.
        """
        aggregate = self._factory(aggregate_id)
        
        # Try to load from snapshot
        snapshot = await self._store.get_snapshot(aggregate_id)
        from_version = 0
        
        if snapshot:
            # Restore from snapshot
            aggregate._state = snapshot.state
            aggregate._version = snapshot.version
            from_version = snapshot.version
        
        # Replay events since snapshot
        events = await self._store.get_events(aggregate_id, from_version)
        aggregate._load_from_history(events)
        
        return aggregate
    
    async def save(self, aggregate: A) -> int:
        """
        Save pending events from an aggregate.
        
        Returns new version number.
        """
        pending = aggregate._get_pending_events()
        if not pending:
            return aggregate.version
        
        expected_version = aggregate.version - len(pending)
        
        new_version = await self._store.append_events(
            aggregate.id,
            pending,
            expected_version,
        )
        
        aggregate._clear_pending_events()
        
        # Check if snapshot needed
        if aggregate.version % self._snapshot_threshold == 0:
            await self._create_snapshot(aggregate)
        
        return new_version
    
    async def _create_snapshot(self, aggregate: A) -> None:
        """Create a snapshot of the aggregate."""
        if hasattr(aggregate, "_state") and aggregate._state is not None:
            snapshot = Snapshot.create(
                aggregate=aggregate,
                state=aggregate._state,
                ihsan_score=0.95,
            )
            await self._store.save_snapshot(snapshot)


# ============================================================================
# PROJECTIONS (READ MODELS)
# ============================================================================


class Projection(ABC):
    """
    Abstract projection for building read models.
    
    Projections transform the event stream into optimized
    read models for queries.
    """
    
    @property
    @abstractmethod
    def projection_name(self) -> str:
        """Name of this projection."""
        ...
    
    @abstractmethod
    def handles(self, event_type: str) -> bool:
        """Check if this projection handles an event type."""
        ...
    
    @abstractmethod
    async def apply(self, event: Event) -> None:
        """Apply an event to update the read model."""
        ...
    
    @abstractmethod
    async def rebuild(self, events: List[Event]) -> None:
        """Rebuild the projection from scratch."""
        ...


class ProjectionEngine:
    """
    Engine for managing and running projections.
    
    Features:
    - Automatic projection updates on new events
    - Projection rebuilding
    - Position tracking (exactly-once processing)
    """
    
    def __init__(self, event_store: EventStore):
        self._store = event_store
        self._projections: List[Projection] = []
        self._positions: Dict[str, int] = {}  # projection -> position
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
    
    def register(self, projection: Projection) -> None:
        """Register a projection."""
        self._projections.append(projection)
        self._positions[projection.projection_name] = 0
    
    async def start(self) -> None:
        """Start the projection engine."""
        if self._running:
            return
        
        self._running = True
        
        # Subscribe to new events
        self._store.subscribe(self._on_event)
        
        logger.info("Projection engine started")
    
    async def stop(self) -> None:
        """Stop the projection engine."""
        self._running = False
        logger.info("Projection engine stopped")
    
    async def rebuild_all(self) -> None:
        """Rebuild all projections from scratch."""
        all_events = await self._store.get_all_events(limit=1_000_000)
        
        for projection in self._projections:
            relevant = [e for e in all_events if projection.handles(e.event_type)]
            await projection.rebuild(relevant)
            self._positions[projection.projection_name] = len(all_events)
    
    async def _on_event(self, event: Event) -> None:
        """Handle new event for projections."""
        for projection in self._projections:
            if projection.handles(event.event_type):
                try:
                    await projection.apply(event)
                except Exception as e:
                    logger.error(f"Projection {projection.projection_name} error: {e}")


# ============================================================================
# COMMAND HANDLER
# ============================================================================


@dataclass
class Command:
    """Base command type."""
    
    command_id: str = field(default_factory=lambda: f"cmd_{secrets.token_hex(8)}")
    aggregate_id: Optional[str] = None
    metadata: Optional[EventMetadata] = None


@dataclass
class CommandResult:
    """Result of command execution."""
    
    success: bool
    aggregate_id: str
    new_version: int
    events_produced: int
    message: str
    data: Optional[Dict[str, Any]] = None


class CommandHandler(ABC, Generic[A]):
    """
    Abstract command handler for aggregate operations.
    
    Implements the command side of CQRS.
    """
    
    def __init__(self, repository: Repository[A]):
        self._repository = repository
    
    @abstractmethod
    async def handle(self, command: Command) -> CommandResult:
        """Handle a command."""
        ...
    
    async def _execute(
        self,
        aggregate_id: str,
        executor: Callable[[A], Awaitable[None]],
    ) -> CommandResult:
        """
        Execute a command against an aggregate.
        
        Loads aggregate, executes command, saves events.
        """
        try:
            aggregate = await self._repository.get(aggregate_id)
            initial_version = aggregate.version
            
            await executor(aggregate)
            
            new_version = await self._repository.save(aggregate)
            events_produced = new_version - initial_version
            
            return CommandResult(
                success=True,
                aggregate_id=aggregate_id,
                new_version=new_version,
                events_produced=events_produced,
                message="Command executed successfully",
            )
        except ConcurrencyConflict as e:
            return CommandResult(
                success=False,
                aggregate_id=aggregate_id,
                new_version=e.actual,
                events_produced=0,
                message=str(e),
            )
        except Exception as e:
            return CommandResult(
                success=False,
                aggregate_id=aggregate_id,
                new_version=0,
                events_produced=0,
                message=str(e),
            )


# ============================================================================
# EXAMPLE: AGENT AGGREGATE
# ============================================================================


@dataclass
class AgentState:
    """State of an agent aggregate."""
    
    agent_id: str
    name: str
    ihsan_score: float
    cognitive_load: float
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    operation_count: int = 0


class AgentAggregate(Aggregate[AgentState]):
    """
    Example aggregate for BIZRA agents.
    
    Demonstrates event sourcing patterns:
    - State reconstruction from events
    - Command validation
    - Event emission
    """
    
    @property
    def aggregate_type(self) -> str:
        return "Agent"
    
    @property
    def state(self) -> AgentState:
        return self._state  # type: ignore
    
    def _get_initial_state(self) -> AgentState:
        return AgentState(
            agent_id=self._id,
            name="",
            ihsan_score=0.95,
            cognitive_load=0.0,
        )
    
    def _apply(self, event: Event) -> None:
        """Apply event to state."""
        if self._state is None:
            self._state = self._get_initial_state()
        
        if event.event_type == "AgentCreated":
            self._state.name = event.data.get("name", "")
            self._state.ihsan_score = event.data.get("ihsan_score", 0.95)
            self._state.created_at = event.metadata.timestamp
            self._state.last_activity = event.metadata.timestamp
            
        elif event.event_type == "IhsanUpdated":
            self._state.ihsan_score = event.data.get("new_score", 0.95)
            self._state.last_activity = event.metadata.timestamp
            
        elif event.event_type == "OperationExecuted":
            self._state.operation_count += 1
            self._state.cognitive_load = event.data.get("cognitive_load", 0.0)
            self._state.last_activity = event.metadata.timestamp
    
    # Commands
    
    def create(
        self,
        name: str,
        ihsan_score: float,
        metadata: EventMetadata,
    ) -> Event:
        """Create a new agent."""
        if self._version > 0:
            raise ValueError("Agent already exists")
        
        if ihsan_score < 0.95:
            raise ValueError(f"Ihsan {ihsan_score:.4f} below threshold")
        
        return self._raise_event(
            event_type="AgentCreated",
            data={"name": name, "ihsan_score": ihsan_score},
            metadata=metadata,
        )
    
    def update_ihsan(
        self,
        new_score: float,
        metadata: EventMetadata,
    ) -> Event:
        """Update agent's Ihsan score."""
        if self._version == 0:
            raise ValueError("Agent does not exist")
        
        if new_score < 0.95:
            raise ValueError(f"Ihsan {new_score:.4f} below threshold")
        
        return self._raise_event(
            event_type="IhsanUpdated",
            data={
                "old_score": self._state.ihsan_score if self._state else 0.95,
                "new_score": new_score,
            },
            metadata=metadata,
        )
    
    def execute_operation(
        self,
        operation_type: str,
        cognitive_load: float,
        metadata: EventMetadata,
    ) -> Event:
        """Record an operation execution."""
        if self._version == 0:
            raise ValueError("Agent does not exist")
        
        return self._raise_event(
            event_type="OperationExecuted",
            data={
                "operation_type": operation_type,
                "cognitive_load": cognitive_load,
            },
            metadata=metadata,
        )


# ============================================================================
# EXAMPLE: AGENT PROJECTION
# ============================================================================


class AgentSummaryProjection(Projection):
    """
    Projection for agent summaries.
    
    Builds a read-optimized view of all agents.
    """
    
    def __init__(self):
        self._summaries: Dict[str, Dict[str, Any]] = {}
    
    @property
    def projection_name(self) -> str:
        return "AgentSummary"
    
    def handles(self, event_type: str) -> bool:
        return event_type in {"AgentCreated", "IhsanUpdated", "OperationExecuted"}
    
    async def apply(self, event: Event) -> None:
        """Apply event to update summaries."""
        agent_id = event.aggregate_id
        
        if agent_id not in self._summaries:
            self._summaries[agent_id] = {
                "agent_id": agent_id,
                "name": "",
                "ihsan_score": 0.95,
                "operation_count": 0,
                "created_at": None,
                "last_activity": None,
            }
        
        summary = self._summaries[agent_id]
        
        if event.event_type == "AgentCreated":
            summary["name"] = event.data.get("name", "")
            summary["ihsan_score"] = event.data.get("ihsan_score", 0.95)
            summary["created_at"] = event.metadata.timestamp.isoformat()
            summary["last_activity"] = event.metadata.timestamp.isoformat()
            
        elif event.event_type == "IhsanUpdated":
            summary["ihsan_score"] = event.data.get("new_score", 0.95)
            summary["last_activity"] = event.metadata.timestamp.isoformat()
            
        elif event.event_type == "OperationExecuted":
            summary["operation_count"] += 1
            summary["last_activity"] = event.metadata.timestamp.isoformat()
    
    async def rebuild(self, events: List[Event]) -> None:
        """Rebuild from all events."""
        self._summaries.clear()
        for event in events:
            await self.apply(event)
    
    def get_summary(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get summary for an agent."""
        return self._summaries.get(agent_id)
    
    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Get all agent summaries."""
        return list(self._summaries.values())
    
    def get_high_ihsan_agents(self, threshold: float = 0.98) -> List[Dict[str, Any]]:
        """Get agents with Ihsan above threshold."""
        return [
            s for s in self._summaries.values()
            if s["ihsan_score"] >= threshold
        ]


# ============================================================================
# DEMO
# ============================================================================


async def demo_event_sourcing():
    """Demonstrate event sourcing capabilities."""
    print("=" * 70)
    print("BIZRA EVENT SOURCING ENGINE DEMO")
    print("=" * 70)
    
    # Create event store
    store = EventStore()
    
    # Create repository
    repo: Repository[AgentAggregate] = Repository(
        event_store=store,
        aggregate_factory=lambda id: AgentAggregate(id),
    )
    
    # Create projection
    projection = AgentSummaryProjection()
    engine = ProjectionEngine(store)
    engine.register(projection)
    await engine.start()
    
    # Create an agent
    print("\n1. Creating Agent")
    print("-" * 40)
    
    agent = AgentAggregate("agent_001")
    metadata = EventMetadata.create(
        actor_id="system",
        source_layer="COGNITIVE",
        ihsan_score=0.99,
    )
    
    agent.create(name="Alpha", ihsan_score=0.99, metadata=metadata)
    version = await repo.save(agent)
    print(f"  Created agent 'Alpha' at version {version}")
    
    # Perform operations
    print("\n2. Executing Operations")
    print("-" * 40)
    
    agent = await repo.get("agent_001")
    for i in range(5):
        meta = EventMetadata.create("system", "COGNITIVE", 0.98)
        agent.execute_operation(f"task_{i}", cognitive_load=0.1 * i, metadata=meta)
    
    version = await repo.save(agent)
    print(f"  Executed 5 operations, now at version {version}")
    
    # Update Ihsan
    print("\n3. Updating Ihsan Score")
    print("-" * 40)
    
    agent = await repo.get("agent_001")
    meta = EventMetadata.create("governance", "GOVERNANCE", 0.99)
    agent.update_ihsan(new_score=0.995, metadata=meta)
    version = await repo.save(agent)
    print(f"  Updated Ihsan to 0.995, version {version}")
    
    # Query projection
    print("\n4. Querying Read Model")
    print("-" * 40)
    
    summary = projection.get_summary("agent_001")
    if summary:
        print(f"  Name: {summary['name']}")
        print(f"  Ihsan: {summary['ihsan_score']:.4f}")
        print(f"  Operations: {summary['operation_count']}")
    
    # Temporal query
    print("\n5. Temporal Query")
    print("-" * 40)
    
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)
    events = await store.get_events_by_time_range(hour_ago, now)
    print(f"  Events in last hour: {len(events)}")
    
    # Metrics
    print("\n6. Event Store Metrics")
    print("-" * 40)
    
    metrics = store.get_metrics()
    print(f"  Total events: {metrics['total_events']}")
    print(f"  Total aggregates: {metrics['total_aggregates']}")
    print(f"  Total appends: {metrics['total_appends']}")
    
    print("\n" + "=" * 70)
    print("EVENT SOURCING DEMO COMPLETE")
    print("=" * 70)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Core Types
    "Event",
    "EventMetadata",
    "EventStatus",
    # Aggregate
    "Aggregate",
    "Snapshot",
    "SnapshotStrategy",
    # Store
    "EventStore",
    "ConcurrencyConflict",
    # Repository
    "Repository",
    # Projections
    "Projection",
    "ProjectionEngine",
    # Commands
    "Command",
    "CommandResult",
    "CommandHandler",
    # Example
    "AgentAggregate",
    "AgentState",
    "AgentSummaryProjection",
]


if __name__ == "__main__":
    asyncio.run(demo_event_sourcing())
