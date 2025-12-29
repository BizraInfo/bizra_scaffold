"""
BIZRA Genesis Event Protocol
════════════════════════════════════════════════════════════════════════════════
The nervous system of the Genesis Node.

Decouples the Orchestrator (Brain) from the Terminal (Eyes) via a typed
event protocol. This enables:

1. REAL-TIME OBSERVABILITY: Terminal can visualize thoughts as they are born
2. LOOSE COUPLING: Orchestrator doesn't need to know about the UI
3. REPLAY: Events can be persisted and replayed for debugging
4. DISTRIBUTED: Events can be sent over network for remote monitoring

"The nervous system carries signals from the brain to the body.
 Without it, consciousness cannot manifest in the physical world."

DESIGN PHILOSOPHY:
────────────────────────────────────────────────────────────────────────────────
- Every significant state change emits an event
- Events are immutable data structures (frozen dataclass)
- Listeners are async to avoid blocking the orchestrator
- Protocol-based for flexible implementation

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger("bizra.genesis.events")


# =============================================================================
# EVENT TYPES
# =============================================================================


class GenesisEventType(Enum):
    """
    Event types emitted by the Genesis Orchestrator.
    
    Organized by processing phase:
    - SYSTEM_*: Lifecycle events
    - LENS_*: Interdisciplinary analysis
    - WISDOM_*: Giants Protocol seeding
    - THOUGHT_*: Graph of Thoughts expansion
    - SNR_*: Signal-to-Noise gating
    - CRYSTAL_*: Wisdom crystallization
    - ATTEST_*: Genesis binding
    - ORACLE_*: External oracle verification
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # System Lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    SYSTEM_START = auto()
    SYSTEM_IDLE = auto()
    SYSTEM_ERROR = auto()
    SYSTEM_SHUTDOWN = auto()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Interdisciplinary Lens Analysis
    # ─────────────────────────────────────────────────────────────────────────
    LENS_ANALYSIS_START = auto()
    LENS_ACTIVATED = auto()
    LENS_INSIGHT_GENERATED = auto()
    LENS_SYNTHESIS_START = auto()
    LENS_SYNTHESIS_COMPLETE = auto()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Giants Protocol / Wisdom Seeding
    # ─────────────────────────────────────────────────────────────────────────
    WISDOM_SEEDING_START = auto()
    WISDOM_SEED_LOADED = auto()
    WISDOM_SEEDING_COMPLETE = auto()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Graph of Thoughts Expansion
    # ─────────────────────────────────────────────────────────────────────────
    THOUGHT_EXPANSION_START = auto()
    THOUGHT_NODE_CREATED = auto()
    THOUGHT_NODE_PRUNED = auto()
    THOUGHT_BEAM_COMPLETE = auto()
    THOUGHT_EXPANSION_COMPLETE = auto()
    
    # ─────────────────────────────────────────────────────────────────────────
    # SNR Gating
    # ─────────────────────────────────────────────────────────────────────────
    SNR_GATE_START = auto()
    SNR_SCORE_COMPUTED = auto()
    SNR_GATE_PASSED = auto()
    SNR_GATE_REJECTED = auto()
    SNR_GATE_COMPLETE = auto()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Crystallization
    # ─────────────────────────────────────────────────────────────────────────
    CRYSTAL_START = auto()
    CRYSTAL_INSIGHT_ADDED = auto()
    CRYSTAL_COMPLETE = auto()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Genesis Attestation
    # ─────────────────────────────────────────────────────────────────────────
    ATTEST_START = auto()
    ATTEST_HASH_COMPUTED = auto()
    ATTEST_BOUND_TO_NODE = auto()
    ATTEST_COMPLETE = auto()
    
    # ─────────────────────────────────────────────────────────────────────────
    # External Oracle
    # ─────────────────────────────────────────────────────────────────────────
    ORACLE_CHECK_START = auto()
    ORACLE_VERDICT = auto()
    ORACLE_DRIFT_DETECTED = auto()


# =============================================================================
# EVENT DATACLASS
# =============================================================================


@dataclass(frozen=True)
class GenesisEvent:
    """
    Immutable event emitted by the Genesis Orchestrator.
    
    Attributes:
        type: The event type from GenesisEventType enum
        phase: Human-readable phase name (e.g., "Lens Analysis")
        progress: Overall processing progress (0.0 to 1.0)
        data: Event-specific payload
        timestamp: When the event occurred (UTC)
        correlation_id: Links events in the same processing run
    
    Design Notes:
    - Frozen for immutability (safe to pass between threads)
    - Progress clamped to [0.0, 1.0] in post_init
    - Serializable to dict for persistence/network
    """
    
    type: GenesisEventType
    phase: str
    progress: float
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate and clamp progress to valid range."""
        if not 0.0 <= self.progress <= 1.0:
            # Use object.__setattr__ since frozen
            object.__setattr__(self, 'progress', max(0.0, min(1.0, self.progress)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.name,
            "phase": self.phase,
            "progress": self.progress,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenesisEvent":
        """Create from dictionary."""
        return cls(
            type=GenesisEventType[data["type"]],
            phase=data["phase"],
            progress=data["progress"],
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
        )
    
    def with_progress(self, new_progress: float) -> "GenesisEvent":
        """Create copy with updated progress."""
        return GenesisEvent(
            type=self.type,
            phase=self.phase,
            progress=new_progress,
            data=self.data,
            timestamp=self.timestamp,
            correlation_id=self.correlation_id,
        )


# =============================================================================
# LISTENER PROTOCOL
# =============================================================================


@runtime_checkable
class GenesisEventListener(Protocol):
    """
    Protocol for event listeners.
    
    Implement this to receive events from the orchestrator.
    The async signature ensures listeners don't block the main loop.
    """
    
    async def on_genesis_event(self, event: GenesisEvent) -> None:
        """
        Called when an event is emitted.
        
        Args:
            event: The emitted event
        """
        ...


# =============================================================================
# EVENT BUS
# =============================================================================


class GenesisEventBus:
    """
    Event bus for distributing events to listeners.
    
    Features:
    - Multiple listeners (global or type-specific)
    - Async event delivery
    - Optional event history for replay/debugging
    - Error isolation (one listener failure doesn't break others)
    
    Usage:
        bus = GenesisEventBus(keep_history=True)
        bus.add_listener(my_listener)
        await bus.emit(GenesisEvent(...))
    """
    
    def __init__(
        self,
        keep_history: bool = False,
        max_history: int = 1000,
    ):
        """
        Initialize the event bus.
        
        Args:
            keep_history: Whether to store events for replay
            max_history: Maximum events to keep in history
        """
        self._listeners: List[GenesisEventListener] = []
        self._type_listeners: Dict[GenesisEventType, List[GenesisEventListener]] = {}
        self._keep_history = keep_history
        self._max_history = max_history
        self._history: List[GenesisEvent] = []
    
    def add_listener(
        self,
        listener: GenesisEventListener,
        event_types: Optional[List[GenesisEventType]] = None,
    ) -> None:
        """
        Add a listener for events.
        
        Args:
            listener: The listener to add
            event_types: Optional list of specific event types to listen for.
                        If None, listens to all events.
        """
        if event_types is None:
            self._listeners.append(listener)
        else:
            for event_type in event_types:
                if event_type not in self._type_listeners:
                    self._type_listeners[event_type] = []
                self._type_listeners[event_type].append(listener)
    
    def remove_listener(self, listener: GenesisEventListener) -> None:
        """Remove a listener from all registrations."""
        if listener in self._listeners:
            self._listeners.remove(listener)
        
        for listeners in self._type_listeners.values():
            if listener in listeners:
                listeners.remove(listener)
    
    async def emit(self, event: GenesisEvent) -> None:
        """
        Emit an event to all relevant listeners.
        
        Args:
            event: The event to emit
        """
        # Store in history
        if self._keep_history:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        
        # Notify global listeners
        for listener in self._listeners:
            try:
                await listener.on_genesis_event(event)
            except Exception as e:
                logger.warning(f"Global listener error: {e}")
        
        # Notify type-specific listeners
        type_listeners = self._type_listeners.get(event.type, [])
        for listener in type_listeners:
            try:
                await listener.on_genesis_event(event)
            except Exception as e:
                logger.warning(f"Type listener error for {event.type.name}: {e}")
    
    def get_history(
        self,
        event_type: Optional[GenesisEventType] = None,
        limit: int = 100,
    ) -> List[GenesisEvent]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type (None for all)
            limit: Maximum events to return
            
        Returns:
            List of events (newest first)
        """
        if not self._keep_history:
            return []
        
        events = self._history
        if event_type is not None:
            events = [e for e in events if e.type == event_type]
        
        return list(reversed(events[-limit:]))
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()
    
    @property
    def listener_count(self) -> int:
        """Get total number of listeners."""
        type_count = sum(len(listeners) for listeners in self._type_listeners.values())
        return len(self._listeners) + type_count
    
    @property
    def history_size(self) -> int:
        """Get current history size."""
        return len(self._history)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_event(
    event_type: GenesisEventType,
    phase: str,
    progress: float,
    correlation_id: Optional[str] = None,
    **data: Any,
) -> GenesisEvent:
    """
    Factory function for creating events.
    
    Args:
        event_type: The event type
        phase: Human-readable phase name
        progress: Processing progress (0.0 to 1.0)
        correlation_id: Optional correlation ID for this run
        **data: Additional event data
        
    Returns:
        A new GenesisEvent
    """
    return GenesisEvent(
        type=event_type,
        phase=phase,
        progress=progress,
        data=dict(data),
        correlation_id=correlation_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Factories for Common Events
# ─────────────────────────────────────────────────────────────────────────────


def system_start(problem: str, correlation_id: Optional[str] = None) -> GenesisEvent:
    """Create SYSTEM_START event."""
    return create_event(
        GenesisEventType.SYSTEM_START,
        "Ignition",
        0.0,
        correlation_id=correlation_id,
        problem=problem,
    )


def lens_activated(
    lens_name: str,
    confidence: float,
    progress: float,
    correlation_id: Optional[str] = None,
) -> GenesisEvent:
    """Create LENS_ACTIVATED event."""
    return create_event(
        GenesisEventType.LENS_ACTIVATED,
        "Lens Analysis",
        progress,
        correlation_id=correlation_id,
        lens=lens_name,
        confidence=confidence,
    )


def wisdom_seed_loaded(
    wisdom_id: str,
    title: str,
    snr: float,
    progress: float,
    correlation_id: Optional[str] = None,
) -> GenesisEvent:
    """Create WISDOM_SEED_LOADED event."""
    return create_event(
        GenesisEventType.WISDOM_SEED_LOADED,
        "Giants Protocol",
        progress,
        correlation_id=correlation_id,
        wisdom_id=wisdom_id,
        title=title,
        snr=snr,
    )


def thought_created(
    content: str,
    snr: float,
    ihsan: float,
    depth: int,
    node_id: str,
    progress: float,
    correlation_id: Optional[str] = None,
) -> GenesisEvent:
    """Create THOUGHT_NODE_CREATED event."""
    return create_event(
        GenesisEventType.THOUGHT_NODE_CREATED,
        f"Depth {depth}",
        progress,
        correlation_id=correlation_id,
        content=content,
        snr=snr,
        ihsan=ihsan,
        depth=depth,
        node_id=node_id,
    )


def thought_pruned(
    node_id: str,
    snr: float,
    reason: str,
    progress: float,
    correlation_id: Optional[str] = None,
) -> GenesisEvent:
    """Create THOUGHT_NODE_PRUNED event."""
    return create_event(
        GenesisEventType.THOUGHT_NODE_PRUNED,
        "SNR Pruning",
        progress,
        correlation_id=correlation_id,
        node_id=node_id,
        snr=snr,
        reason=reason,
    )


def snr_computed(
    score: float,
    ihsan: float,
    passed: bool,
    progress: float,
    correlation_id: Optional[str] = None,
) -> GenesisEvent:
    """Create SNR_SCORE_COMPUTED event."""
    level = "HIGH" if score >= 0.80 else "MEDIUM" if score >= 0.50 else "LOW"
    return create_event(
        GenesisEventType.SNR_SCORE_COMPUTED,
        "SNR Gating",
        progress,
        correlation_id=correlation_id,
        snr=score,
        ihsan=ihsan,
        passed=passed,
        level=level,
    )


def crystal_added(
    wisdom_id: str,
    title: str,
    snr: float,
    progress: float,
    correlation_id: Optional[str] = None,
) -> GenesisEvent:
    """Create CRYSTAL_INSIGHT_ADDED event."""
    return create_event(
        GenesisEventType.CRYSTAL_INSIGHT_ADDED,
        "Crystallization",
        progress,
        correlation_id=correlation_id,
        wisdom_id=wisdom_id,
        title=title,
        snr=snr,
    )


def attestation_complete(
    attestation_hash: str,
    node_id: Optional[str],
    correlation_id: Optional[str] = None,
) -> GenesisEvent:
    """Create ATTEST_COMPLETE event."""
    return create_event(
        GenesisEventType.ATTEST_COMPLETE,
        "Binding",
        1.0,
        correlation_id=correlation_id,
        hash=attestation_hash,
        node_id=node_id,
    )


def oracle_drift(
    expected: float,
    actual: float,
    drift_percent: float,
    correlation_id: Optional[str] = None,
) -> GenesisEvent:
    """Create ORACLE_DRIFT_DETECTED event."""
    return create_event(
        GenesisEventType.ORACLE_DRIFT_DETECTED,
        "Oracle Alert",
        1.0,
        correlation_id=correlation_id,
        expected=expected,
        actual=actual,
        drift_percent=drift_percent,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core types
    "GenesisEventType",
    "GenesisEvent",
    "GenesisEventListener",
    "GenesisEventBus",
    # Factory functions
    "create_event",
    "system_start",
    "lens_activated",
    "wisdom_seed_loaded",
    "thought_created",
    "thought_pruned",
    "snr_computed",
    "crystal_added",
    "attestation_complete",
    "oracle_drift",
]
