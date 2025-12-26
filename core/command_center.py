r"""
BIZRA AEON OMEGA - Node Zero Command Center
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Pattern: Unified System Orchestration & Control Plane

The Command Center serves as the single point of control for the entire BIZRA
ecosystem, integrating:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                    NODE ZERO COMMAND CENTER                              │
  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌─────────────┐  │
  │  │ Data Lake     │ │ Graph of      │ │ Event         │ │ Health      │  │
  │  │ Watcher       │ │ Thoughts      │ │ Sourcing      │ │ Monitor     │  │
  │  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └──────┬──────┘  │
  │          │                 │                 │                │         │
  │          └────────────┬────┴─────────────────┴────────────────┘         │
  │                       ▼                                                  │
  │              ┌────────────────────┐                                      │
  │              │  UNIFIED BUS       │──────► Event Stream                  │
  │              │  (Command/Query)   │                                      │
  │              └────────────────────┘                                      │
  │                       │                                                  │
  │          ┌────────────┼────────────┐                                     │
  │          ▼            ▼            ▼                                     │
  │   ┌────────────┐ ┌──────────┐ ┌──────────┐                              │
  │   │ SNR Scorer │ │ Ihsan    │ │ APEX     │                              │
  │   │            │ │ Enforcer │ │ Router   │                              │
  │   └────────────┘ └──────────┘ └──────────┘                              │
  └─────────────────────────────────────────────────────────────────────────┘

Key Features:
  1. Unified Command Bus: All operations routed through single interface
  2. Health Aggregation: Real-time status from all subsystems
  3. Knowledge Bridge: Data lake → Graph of Thoughts semantic indexing
  4. Event-Driven: All state changes emit events for observability
  5. SNR-Weighted Routing: High-signal operations get priority
  6. Ihsan Enforcement: All operations respect IM ≥ 0.95 threshold

BIZRA SOT Compliance:
  - Section 3 (Invariants): IM ≥ 0.95 enforced on all commands
  - Section 7 (Evidence Policy): All commands logged with evidence
  - Section 8 (Change Control): Version-tracked operations

Author: BIZRA Genesis Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════


class SubsystemType(Enum):
    """BIZRA subsystem classification."""
    DATA_LAKE = auto()          # Data Lake Watcher
    KNOWLEDGE_GRAPH = auto()    # Graph of Thoughts
    EVENT_STORE = auto()        # Event Sourcing Engine
    VERIFICATION = auto()       # Tiered Verification
    ETHICS = auto()             # Consequential Ethics
    VALUE_ORACLE = auto()       # Pluralistic Value Oracle
    SNR_SCORER = auto()         # Signal-to-Noise Scorer
    APEX = auto()               # APEX Orchestrator


class CommandType(Enum):
    """Command classification for routing."""
    QUERY = auto()              # Read-only, no side effects
    MUTATION = auto()           # State-changing command
    SCAN = auto()               # Data lake scan operation
    VERIFY = auto()             # Verification request
    SCORE = auto()              # SNR scoring request
    HEALTH_CHECK = auto()       # System health query
    SYNC = auto()               # Synchronization command


class HealthLevel(Enum):
    """Subsystem health levels."""
    OPTIMAL = auto()            # 100% operational
    HEALTHY = auto()            # >95% operational
    DEGRADED = auto()           # 70-95% operational
    IMPAIRED = auto()           # 50-70% operational
    CRITICAL = auto()           # <50% operational
    OFFLINE = auto()            # Not responding


class Priority(Enum):
    """Command priority levels."""
    CRITICAL = 0                # Immediate execution
    HIGH = 1                    # Priority queue
    NORMAL = 2                  # Standard queue
    LOW = 3                     # Background execution
    DEFERRED = 4                # Execute when idle


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CommandId:
    """Immutable command identifier."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value[:8]


@dataclass
class Command:
    """
    Base command structure for the Command Center.
    
    All operations are expressed as commands for:
    - Unified logging and audit trail
    - SNR-weighted priority routing
    - Ihsan compliance checking
    - Event emission on completion
    """
    
    id: CommandId
    command_type: CommandType
    target_subsystem: SubsystemType
    payload: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    
    # Metadata
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    actor: str = "system"
    
    # SNR weighting
    signal_strength: float = 0.5
    
    # Ihsan compliance
    requires_ihsan_check: bool = True
    min_ihsan_threshold: float = 0.95
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize command for logging/transmission."""
        return {
            "id": str(self.id),
            "type": self.command_type.name,
            "target": self.target_subsystem.name,
            "payload": self.payload,
            "priority": self.priority.name,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "actor": self.actor,
            "signal_strength": self.signal_strength,
            "requires_ihsan_check": self.requires_ihsan_check,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CommandResult:
    """Result of command execution."""
    
    command_id: CommandId
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Metrics
    execution_time_ms: float = 0.0
    ihsan_score: float = 1.0
    snr_score: float = 0.5
    
    # Provenance
    executed_by: SubsystemType = SubsystemType.APEX
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize result."""
        return {
            "command_id": str(self.command_id),
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "executed_by": self.executed_by.name,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SubsystemStatus:
    """Status report from a subsystem."""
    
    subsystem: SubsystemType
    health: HealthLevel
    uptime_seconds: float
    last_heartbeat: datetime
    
    # Metrics
    operations_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    
    # Resource usage
    memory_mb: float = 0.0
    
    # Custom metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.operations_count == 0:
            return 0.0
        return self.error_count / self.operations_count
    
    @property
    def is_healthy(self) -> bool:
        """Check if subsystem is healthy."""
        return self.health in (HealthLevel.OPTIMAL, HealthLevel.HEALTHY)


@dataclass
class SystemHealthReport:
    """Aggregated health report for entire system."""
    
    timestamp: datetime
    overall_health: HealthLevel
    subsystems: Dict[SubsystemType, SubsystemStatus]
    
    # Aggregate metrics
    total_operations: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    
    # Ihsan compliance
    ihsan_score: float = 1.0
    ihsan_violations: int = 0
    
    # Alerts
    alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize report."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_health": self.overall_health.name,
            "subsystems": {
                k.name: {
                    "health": v.health.name,
                    "uptime_seconds": v.uptime_seconds,
                    "operations_count": v.operations_count,
                    "error_rate": v.error_rate,
                }
                for k, v in self.subsystems.items()
            },
            "total_operations": self.total_operations,
            "total_errors": self.total_errors,
            "avg_latency_ms": self.avg_latency_ms,
            "ihsan_score": self.ihsan_score,
            "ihsan_violations": self.ihsan_violations,
            "alerts": self.alerts,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════


class Subsystem(Protocol):
    """Protocol for BIZRA subsystems."""
    
    @property
    def subsystem_type(self) -> SubsystemType:
        """Return subsystem type identifier."""
        ...
    
    async def health_check(self) -> SubsystemStatus:
        """Return current health status."""
        ...
    
    async def execute(self, command: Command) -> CommandResult:
        """Execute a command."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT BUS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Event:
    """Event emitted by the Command Center."""
    
    id: str
    event_type: str
    source: SubsystemType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Provenance
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None


class EventBus:
    """
    Central event bus for inter-subsystem communication.
    
    Implements publish-subscribe pattern with:
    - Topic-based routing
    - Event persistence (in-memory buffer)
    - Async delivery
    """
    
    def __init__(self, buffer_size: int = 10000):
        self._subscribers: Dict[str, List[Callable[[Event], Awaitable[None]]]] = defaultdict(list)
        self._event_buffer: Deque[Event] = deque(maxlen=buffer_size)
        self._lock = asyncio.Lock()
        self._event_count = 0
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Awaitable[None]],
    ) -> None:
        """Subscribe to events of a specific type."""
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler to event type: {event_type}")
    
    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Awaitable[None]],
    ) -> None:
        """Unsubscribe from events."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
    
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        async with self._lock:
            self._event_buffer.append(event)
            self._event_count += 1
        
        # Notify subscribers
        handlers = self._subscribers.get(event.event_type, [])
        handlers.extend(self._subscribers.get("*", []))  # Wildcard subscribers
        
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def get_recent_events(self, count: int = 100) -> List[Event]:
        """Get recent events from buffer."""
        return list(self._event_buffer)[-count:]
    
    @property
    def event_count(self) -> int:
        """Total events published."""
        return self._event_count


# ═══════════════════════════════════════════════════════════════════════════════
# IHSAN ENFORCER
# ═══════════════════════════════════════════════════════════════════════════════


class IhsanEnforcer:
    """
    Ihsan compliance enforcement layer.
    
    Ensures all operations respect the Ihsan threshold (IM ≥ 0.95)
    as specified in BIZRA SOT Section 3.
    """
    
    DEFAULT_THRESHOLD = 0.95
    
    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._violation_count = 0
        self._check_count = 0
        self._violations: Deque[Dict[str, Any]] = deque(maxlen=100)
    
    async def check_compliance(
        self,
        command: Command,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, float, str]:
        """
        Check if command meets Ihsan compliance.
        
        Returns:
            (compliant, score, reason)
        """
        self._check_count += 1
        
        # Skip check if not required
        if not command.requires_ihsan_check:
            return True, 1.0, "Check not required"
        
        # Compute Ihsan score based on command properties
        score = self._compute_ihsan_score(command, context)
        threshold = command.min_ihsan_threshold
        
        if score >= threshold:
            return True, score, "Compliant"
        else:
            self._violation_count += 1
            self._violations.append({
                "command_id": str(command.id),
                "score": score,
                "threshold": threshold,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return False, score, f"Ihsan score {score:.3f} below threshold {threshold:.3f}"
    
    def _compute_ihsan_score(
        self,
        command: Command,
        context: Optional[Dict[str, Any]],
    ) -> float:
        """
        Compute Ihsan score for a command.
        
        Components:
        - Signal strength (weighted)
        - Actor reputation
        - Command integrity
        """
        score = 0.0
        
        # Signal strength contributes 40%
        score += command.signal_strength * 0.4
        
        # Priority alignment contributes 20%
        priority_scores = {
            Priority.CRITICAL: 1.0,
            Priority.HIGH: 0.95,
            Priority.NORMAL: 0.90,
            Priority.LOW: 0.85,
            Priority.DEFERRED: 0.80,
        }
        score += priority_scores.get(command.priority, 0.85) * 0.2
        
        # Command completeness contributes 20%
        completeness = 1.0 if command.payload else 0.8
        if command.correlation_id:
            completeness += 0.1
        score += min(completeness, 1.0) * 0.2
        
        # Temporal validity contributes 20%
        if command.expires_at:
            if datetime.now(timezone.utc) > command.expires_at:
                score += 0.0  # Expired
            else:
                score += 0.2
        else:
            score += 0.2  # No expiry = always valid
        
        return min(score, 1.0)
    
    @property
    def compliance_rate(self) -> float:
        """Calculate overall compliance rate."""
        if self._check_count == 0:
            return 1.0
        return 1.0 - (self._violation_count / self._check_count)
    
    def get_violations(self) -> List[Dict[str, Any]]:
        """Get recent violations."""
        return list(self._violations)


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND ROUTER
# ═══════════════════════════════════════════════════════════════════════════════


class CommandRouter:
    """
    SNR-weighted command router.
    
    Routes commands to appropriate subsystems with:
    - Priority-based queuing
    - SNR-weighted scheduling
    - Load balancing
    """
    
    def __init__(self):
        self._queues: Dict[Priority, Deque[Command]] = {
            p: deque() for p in Priority
        }
        self._routing_table: Dict[SubsystemType, Subsystem] = {}
        self._stats = {
            "routed": 0,
            "failed": 0,
        }
    
    def register_subsystem(self, subsystem: Subsystem) -> None:
        """Register a subsystem for command routing."""
        self._routing_table[subsystem.subsystem_type] = subsystem
        logger.info(f"Registered subsystem: {subsystem.subsystem_type.name}")
    
    def unregister_subsystem(self, subsystem_type: SubsystemType) -> None:
        """Unregister a subsystem."""
        if subsystem_type in self._routing_table:
            del self._routing_table[subsystem_type]
    
    async def route(self, command: Command) -> CommandResult:
        """Route command to target subsystem."""
        target = command.target_subsystem
        
        if target not in self._routing_table:
            self._stats["failed"] += 1
            return CommandResult(
                command_id=command.id,
                success=False,
                error=f"Subsystem {target.name} not registered",
            )
        
        subsystem = self._routing_table[target]
        
        try:
            start_time = time.time()
            result = await subsystem.execute(command)
            result.execution_time_ms = (time.time() - start_time) * 1000
            self._stats["routed"] += 1
            return result
        except Exception as e:
            self._stats["failed"] += 1
            logger.error(f"Command routing failed: {e}")
            return CommandResult(
                command_id=command.id,
                success=False,
                error=str(e),
            )
    
    def get_registered_subsystems(self) -> List[SubsystemType]:
        """List all registered subsystems."""
        return list(self._routing_table.keys())
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get routing statistics."""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════════


class HealthAggregator:
    """
    Aggregates health status from all subsystems.
    
    Provides:
    - Real-time health monitoring
    - Alert generation
    - Historical metrics
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self._subsystem_status: Dict[SubsystemType, SubsystemStatus] = {}
        self._alerts: Deque[str] = deque(maxlen=100)
        self._history: Deque[SystemHealthReport] = deque(maxlen=1000)
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def update_status(self, status: SubsystemStatus) -> None:
        """Update status for a subsystem."""
        old_status = self._subsystem_status.get(status.subsystem)
        self._subsystem_status[status.subsystem] = status
        
        # Generate alerts on health degradation
        if old_status and old_status.health != status.health:
            if status.health.value > old_status.health.value:  # Degraded
                alert = f"[{datetime.now(timezone.utc).isoformat()}] {status.subsystem.name} degraded: {old_status.health.name} → {status.health.name}"
                self._alerts.append(alert)
                logger.warning(alert)
    
    def get_report(self) -> SystemHealthReport:
        """Generate aggregated health report."""
        now = datetime.now(timezone.utc)
        
        # Determine overall health
        if not self._subsystem_status:
            overall = HealthLevel.OFFLINE
        else:
            health_values = [s.health.value for s in self._subsystem_status.values()]
            avg_health = sum(health_values) / len(health_values)
            
            if avg_health <= 1:
                overall = HealthLevel.OPTIMAL
            elif avg_health <= 2:
                overall = HealthLevel.HEALTHY
            elif avg_health <= 3:
                overall = HealthLevel.DEGRADED
            elif avg_health <= 4:
                overall = HealthLevel.IMPAIRED
            else:
                overall = HealthLevel.CRITICAL
        
        # Aggregate metrics
        total_ops = sum(s.operations_count for s in self._subsystem_status.values())
        total_errors = sum(s.error_count for s in self._subsystem_status.values())
        latencies = [s.avg_latency_ms for s in self._subsystem_status.values() if s.avg_latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        report = SystemHealthReport(
            timestamp=now,
            overall_health=overall,
            subsystems=self._subsystem_status.copy(),
            total_operations=total_ops,
            total_errors=total_errors,
            avg_latency_ms=avg_latency,
            alerts=list(self._alerts)[-10:],  # Last 10 alerts
        )
        
        self._history.append(report)
        return report
    
    def get_history(self, count: int = 100) -> List[SystemHealthReport]:
        """Get historical reports."""
        return list(self._history)[-count:]


# ═══════════════════════════════════════════════════════════════════════════════
# NODE ZERO COMMAND CENTER
# ═══════════════════════════════════════════════════════════════════════════════


class NodeZeroCommandCenter:
    """
    The unified control plane for BIZRA ecosystem.
    
    Integrates:
    - Command Bus (CQRS pattern)
    - Event Bus (Event-driven architecture)
    - Health Aggregation (Real-time monitoring)
    - Ihsan Enforcement (Ethical compliance)
    - SNR Routing (Signal-weighted scheduling)
    
    Usage:
        center = NodeZeroCommandCenter()
        
        # Register subsystems
        center.register_subsystem(data_lake_subsystem)
        center.register_subsystem(knowledge_graph_subsystem)
        
        # Execute commands
        result = await center.execute(command)
        
        # Get system health
        health = center.get_health_report()
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        ihsan_threshold: float = 0.95,
        enable_event_logging: bool = True,
    ):
        """
        Initialize the Command Center.
        
        Args:
            ihsan_threshold: Minimum Ihsan score for commands
            enable_event_logging: Whether to log all events
        """
        # Core components
        self.event_bus = EventBus()
        self.ihsan_enforcer = IhsanEnforcer(threshold=ihsan_threshold)
        self.router = CommandRouter()
        self.health_aggregator = HealthAggregator()
        
        # Configuration
        self.enable_event_logging = enable_event_logging
        
        # State
        self._started_at = datetime.now(timezone.utc)
        self._command_count = 0
        self._error_count = 0
        
        # Command history for audit
        self._command_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        
        logger.info(f"Node Zero Command Center v{self.VERSION} initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUBSYSTEM MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def register_subsystem(self, subsystem: Subsystem) -> None:
        """Register a subsystem with the Command Center."""
        self.router.register_subsystem(subsystem)
        
        # Subscribe to health updates
        async def health_handler(event: Event):
            if event.event_type == "health_update":
                status = SubsystemStatus(**event.payload)
                self.health_aggregator.update_status(status)
        
        self.event_bus.subscribe(f"health.{subsystem.subsystem_type.name}", health_handler)
    
    def unregister_subsystem(self, subsystem_type: SubsystemType) -> None:
        """Unregister a subsystem."""
        self.router.unregister_subsystem(subsystem_type)
    
    def list_subsystems(self) -> List[SubsystemType]:
        """List all registered subsystems."""
        return self.router.get_registered_subsystems()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMMAND EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def execute(self, command: Command) -> CommandResult:
        """
        Execute a command through the Command Center.
        
        Pipeline:
        1. Ihsan compliance check
        2. SNR-weighted routing
        3. Subsystem execution
        4. Event emission
        5. Audit logging
        """
        self._command_count += 1
        start_time = time.time()
        
        # Step 1: Ihsan compliance check
        compliant, ihsan_score, reason = await self.ihsan_enforcer.check_compliance(command)
        
        if not compliant:
            self._error_count += 1
            result = CommandResult(
                command_id=command.id,
                success=False,
                error=f"Ihsan compliance failed: {reason}",
                ihsan_score=ihsan_score,
            )
            await self._log_command(command, result)
            return result
        
        # Step 2: Route to subsystem
        result = await self.router.route(command)
        result.ihsan_score = ihsan_score
        
        if not result.success:
            self._error_count += 1
        
        # Step 3: Emit event
        if self.enable_event_logging:
            event = Event(
                id=str(uuid.uuid4()),
                event_type="command_executed",
                source=command.target_subsystem,
                payload={
                    "command": command.to_dict(),
                    "result": result.to_dict(),
                },
                correlation_id=command.correlation_id,
                causation_id=str(command.id),
            )
            await self.event_bus.publish(event)
        
        # Step 4: Audit log
        await self._log_command(command, result)
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
    
    async def execute_batch(
        self,
        commands: List[Command],
        parallel: bool = False,
    ) -> List[CommandResult]:
        """
        Execute multiple commands.
        
        Args:
            commands: List of commands to execute
            parallel: Whether to execute in parallel
        """
        if parallel:
            return await asyncio.gather(*[self.execute(cmd) for cmd in commands])
        else:
            results = []
            for cmd in commands:
                results.append(await self.execute(cmd))
            return results
    
    async def _log_command(self, command: Command, result: CommandResult) -> None:
        """Log command execution for audit trail."""
        entry = {
            "command": command.to_dict(),
            "result": result.to_dict(),
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        self._command_history.append(entry)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HEALTH & MONITORING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_health_report(self) -> SystemHealthReport:
        """Get current system health report."""
        report = self.health_aggregator.get_report()
        
        # Add command center metrics
        report.ihsan_score = self.ihsan_enforcer.compliance_rate
        report.ihsan_violations = self.ihsan_enforcer._violation_count
        
        return report
    
    async def check_all_subsystems(self) -> Dict[SubsystemType, SubsystemStatus]:
        """Check health of all registered subsystems."""
        results = {}
        
        for subsystem_type in self.router.get_registered_subsystems():
            subsystem = self.router._routing_table[subsystem_type]
            try:
                status = await subsystem.health_check()
                self.health_aggregator.update_status(status)
                results[subsystem_type] = status
            except Exception as e:
                logger.error(f"Health check failed for {subsystem_type.name}: {e}")
                results[subsystem_type] = SubsystemStatus(
                    subsystem=subsystem_type,
                    health=HealthLevel.OFFLINE,
                    uptime_seconds=0,
                    last_heartbeat=datetime.now(timezone.utc),
                )
        
        return results
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_command(
        self,
        command_type: CommandType,
        target: SubsystemType,
        payload: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        signal_strength: float = 0.5,
    ) -> Command:
        """Factory method for creating commands."""
        return Command(
            id=CommandId(),
            command_type=command_type,
            target_subsystem=target,
            payload=payload,
            priority=priority,
            signal_strength=signal_strength,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Command Center statistics."""
        uptime = (datetime.now(timezone.utc) - self._started_at).total_seconds()
        
        return {
            "version": self.VERSION,
            "started_at": self._started_at.isoformat(),
            "uptime_seconds": uptime,
            "registered_subsystems": len(self.router.get_registered_subsystems()),
            "total_commands": self._command_count,
            "total_errors": self._error_count,
            "error_rate": self._error_count / max(1, self._command_count),
            "ihsan_compliance_rate": self.ihsan_enforcer.compliance_rate,
            "events_published": self.event_bus.event_count,
            "routing_stats": self.router.stats,
        }
    
    def get_audit_log(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent command audit log."""
        return list(self._command_history)[-count:]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEXT MANAGER
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def __aenter__(self) -> "NodeZeroCommandCenter":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # Log shutdown
        logger.info(f"Command Center shutting down. Stats: {self.get_statistics()}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM ADAPTERS
# ═══════════════════════════════════════════════════════════════════════════════


class DataLakeSubsystem:
    """
    Adapter to expose DataLakeWatcher as a Command Center subsystem.
    """
    
    def __init__(self, watcher: Optional[Any] = None):
        """
        Initialize with optional watcher instance.
        
        Args:
            watcher: DataLakeWatcher instance (lazy loaded if None)
        """
        self._watcher = watcher
        self._started_at = datetime.now(timezone.utc)
        self._op_count = 0
        self._error_count = 0
    
    @property
    def subsystem_type(self) -> SubsystemType:
        return SubsystemType.DATA_LAKE
    
    async def health_check(self) -> SubsystemStatus:
        """Return current health status."""
        uptime = (datetime.now(timezone.utc) - self._started_at).total_seconds()
        
        # Check watcher state
        if self._watcher is None:
            health = HealthLevel.DEGRADED
        elif hasattr(self._watcher, 'state'):
            from core.data_lake_watcher import WatcherState
            state_health = {
                WatcherState.IDLE: HealthLevel.HEALTHY,
                WatcherState.WATCHING: HealthLevel.OPTIMAL,
                WatcherState.SCANNING: HealthLevel.OPTIMAL,
                WatcherState.ERROR: HealthLevel.IMPAIRED,
                WatcherState.PAUSED: HealthLevel.DEGRADED,
            }
            health = state_health.get(self._watcher.state, HealthLevel.HEALTHY)
        else:
            health = HealthLevel.HEALTHY
        
        return SubsystemStatus(
            subsystem=SubsystemType.DATA_LAKE,
            health=health,
            uptime_seconds=uptime,
            last_heartbeat=datetime.now(timezone.utc),
            operations_count=self._op_count,
            error_count=self._error_count,
            metrics={
                "assets": len(self._watcher.assets) if self._watcher else 0,
                "watched_paths": len(self._watcher.watched_paths) if self._watcher else 0,
            },
        )
    
    async def execute(self, command: Command) -> CommandResult:
        """Execute a command."""
        self._op_count += 1
        
        if self._watcher is None:
            self._error_count += 1
            return CommandResult(
                command_id=command.id,
                success=False,
                error="DataLakeWatcher not initialized",
            )
        
        try:
            action = command.payload.get("action", "status")
            
            if action == "scan":
                changes = await self._watcher.scan_all()
                return CommandResult(
                    command_id=command.id,
                    success=True,
                    data={
                        "changes": len(changes),
                        "assets": len(self._watcher.assets),
                    },
                )
            
            elif action == "verify":
                report = self._watcher.verify_manifest()
                return CommandResult(
                    command_id=command.id,
                    success=True,
                    data=report,
                )
            
            elif action == "status":
                summary = self._watcher.get_summary()
                return CommandResult(
                    command_id=command.id,
                    success=True,
                    data=summary,
                )
            
            elif action == "score":
                distribution = await self._watcher.score_all_assets()
                return CommandResult(
                    command_id=command.id,
                    success=True,
                    data={"distribution": distribution},
                )
            
            else:
                return CommandResult(
                    command_id=command.id,
                    success=False,
                    error=f"Unknown action: {action}",
                )
        
        except Exception as e:
            self._error_count += 1
            return CommandResult(
                command_id=command.id,
                success=False,
                error=str(e),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def create_command_center(
    with_data_lake: bool = True,
    ihsan_threshold: float = 0.95,
) -> NodeZeroCommandCenter:
    """
    Factory function to create a fully configured Command Center.
    
    Args:
        with_data_lake: Whether to register DataLake subsystem
        ihsan_threshold: Ihsan compliance threshold
    
    Returns:
        Configured NodeZeroCommandCenter instance
    """
    center = NodeZeroCommandCenter(ihsan_threshold=ihsan_threshold)
    
    if with_data_lake:
        try:
            from core.data_lake_watcher import create_default_watcher
            watcher = create_default_watcher()
            center.register_subsystem(DataLakeSubsystem(watcher))
        except ImportError:
            logger.warning("DataLakeWatcher not available")
    
    return center


async def run_command_center_cli():
    """CLI entry point for the Command Center."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA Node Zero Command Center")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--health", action="store_true", help="Show health report")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    center = create_command_center()
    
    print("=" * 70)
    print(" BIZRA NODE ZERO COMMAND CENTER")
    print("=" * 70)
    print(f" Version: {NodeZeroCommandCenter.VERSION}")
    print(f" Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    print()
    
    if args.health:
        await center.check_all_subsystems()
        report = center.get_health_report()
        print(f"Overall Health: {report.overall_health.name}")
        print(f"Ihsan Compliance: {report.ihsan_score:.1%}")
        print()
        print("Subsystems:")
        for ss_type, status in report.subsystems.items():
            print(f"  {ss_type.name}: {status.health.name}")
    
    if args.stats:
        stats = center.get_statistics()
        print(json.dumps(stats, indent=2))
    
    if not any([args.status, args.health, args.stats]):
        # Default: show status
        print(f"Registered Subsystems: {len(center.list_subsystems())}")
        for ss in center.list_subsystems():
            print(f"  • {ss.name}")
        print()
        stats = center.get_statistics()
        print(f"Total Commands: {stats['total_commands']}")
        print(f"Error Rate: {stats['error_rate']:.1%}")
        print(f"Ihsan Compliance: {stats['ihsan_compliance_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(run_command_center_cli())
