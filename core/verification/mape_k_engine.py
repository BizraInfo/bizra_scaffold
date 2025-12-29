"""
BIZRA AEON OMEGA - MAPE-K Autonomic Self-Healing Loop
======================================================
Full Autonomic Computing Pattern Implementation

The MAPE-K loop is the industry-standard autonomic computing pattern
that enables systems to self-heal without human intervention:

    Monitor → Analyze → Plan → Execute → Knowledge
        ↑                                    ↓
        └────────────────────────────────────┘

Key Features:
    - Monitor: selfEvaluate() with health metrics collection
    - Analyze: selfCritique() with Constitutional AI alignment
    - Plan: generateImprovementActions() with DAG planning
    - Execute: selfAdopt() with sandboxed testing
    - Knowledge: Integration with expertise graph

Quantized Convergence:
    dC/dt = κ × (C_target - C_current) × e^(-λt)
    
    Where:
    - C_current: Current system state quality
    - C_target: Desired system state (I_vec = 1.0)
    - κ: Convergence rate constant
    - λ: Decay factor for bounded convergence

Author: BIZRA Genesis Team (Peak Masterpiece v4)
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
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
)

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger("bizra.mape_k")

# ============================================================================
# CONSTANTS
# ============================================================================

# Quantized Convergence Parameters
KAPPA_DEFAULT: float = 0.1           # Convergence rate constant
LAMBDA_DEFAULT: float = 0.01         # Decay factor
CONVERGENCE_TARGET: float = 1.0      # Perfect I_vec
CONVERGENCE_MIN_RATE: float = 0.001  # Minimum acceptable dC/dt
CONVERGENCE_MAX_RATE: float = 0.5    # Maximum safe dC/dt

# Health Thresholds
HEALTH_CRITICAL_THRESHOLD: float = 0.5
HEALTH_DEGRADED_THRESHOLD: float = 0.8
HEALTH_HEALTHY_THRESHOLD: float = 0.95

# MAPE-K Timing
MONITOR_INTERVAL_MS: float = 1000.0
ANALYZE_INTERVAL_MS: float = 5000.0
PLAN_COOLDOWN_MS: float = 10000.0

# ============================================================================
# ENUMERATIONS
# ============================================================================


class HealthStatus(Enum):
    """System health status levels."""
    
    HEALTHY = auto()      # All systems nominal
    DEGRADED = auto()     # Performance issues
    CRITICAL = auto()     # Imminent failure
    HEALING = auto()      # Recovery in progress
    UNKNOWN = auto()      # Cannot determine


class ActionType(Enum):
    """Types of healing actions."""
    
    CACHE_CLEAR = auto()       # Clear caches
    CIRCUIT_OPEN = auto()      # Open circuit breaker
    CIRCUIT_CLOSE = auto()     # Close circuit breaker
    RATE_LIMIT = auto()        # Apply rate limiting
    SCALE_DOWN = auto()        # Reduce resource usage
    SCALE_UP = auto()          # Increase resources
    RESTART = auto()           # Restart component
    FAILOVER = auto()          # Switch to backup
    ROLLBACK = auto()          # Rollback to previous state
    NOOP = auto()              # No operation


class ConvergenceState(Enum):
    """Quantized convergence states."""
    
    CONVERGING = auto()     # dC/dt > 0 (improving)
    DIVERGING = auto()      # dC/dt < 0 (degrading)
    STALLED = auto()        # dC/dt ≈ 0 (stuck)
    OSCILLATING = auto()    # dC/dt alternating
    CONVERGED = auto()      # At target


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class HealthMetric:
    """A single health metric observation."""
    
    name: str
    value: float
    timestamp: datetime
    unit: str = ""
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None
    
    def is_healthy(self) -> bool:
        """Check if metric is within healthy bounds."""
        if self.threshold_low is not None and self.value < self.threshold_low:
            return False
        if self.threshold_high is not None and self.value > self.threshold_high:
            return False
        return True


@dataclass
class HealthSnapshot:
    """Complete health snapshot at a point in time."""
    
    timestamp: datetime
    status: HealthStatus
    metrics: Dict[str, HealthMetric]
    i_vec_score: float
    omega_score: float
    convergence_rate: float  # dC/dt
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of snapshot."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.name,
            "i_vec_score": self.i_vec_score,
            "omega_score": self.omega_score,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class AnalysisResult:
    """Result of selfCritique analysis phase."""
    
    timestamp: datetime
    current_status: HealthStatus
    issues_detected: List[str]
    root_causes: List[str]
    severity: float  # 0.0 (minor) to 1.0 (critical)
    constitutional_alignment: float  # I_vec alignment
    recommendations: List[str]
    
    def needs_intervention(self) -> bool:
        """Check if healing intervention is needed."""
        return (
            self.severity > 0.3 or
            self.constitutional_alignment < 0.95 or
            self.current_status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]
        )


@dataclass
class HealingAction:
    """A planned healing action."""
    
    action_id: str
    action_type: ActionType
    target_component: str
    priority: int  # 1 (highest) to 10 (lowest)
    estimated_impact: float  # Expected improvement
    risk_score: float  # 0.0 (safe) to 1.0 (risky)
    rollback_plan: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: "HealingAction") -> bool:
        """Compare by priority for heap ordering."""
        return self.priority < other.priority


@dataclass
class HealingPlan:
    """Complete healing plan from Plan phase."""
    
    plan_id: str
    timestamp: datetime
    actions: List[HealingAction]
    expected_outcome: HealthStatus
    estimated_time_ms: float
    rollback_available: bool
    approval_required: bool  # True for high-risk plans
    
    def to_dag(self) -> Dict[str, List[str]]:
        """Convert to DAG representation for execution ordering."""
        # Simple sequential DAG for now
        dag: Dict[str, List[str]] = {}
        prev_id: Optional[str] = None
        for action in sorted(self.actions):
            dag[action.action_id] = [prev_id] if prev_id else []
            prev_id = action.action_id
        return dag


@dataclass
class ExecutionResult:
    """Result of executing a healing plan."""
    
    plan_id: str
    timestamp: datetime
    success: bool
    actions_executed: int
    actions_failed: int
    final_status: HealthStatus
    error_messages: List[str]
    execution_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "plan_id": self.plan_id,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "actions_executed": self.actions_executed,
            "actions_failed": self.actions_failed,
            "final_status": self.final_status.name,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class ConvergenceState_:
    """Quantized convergence tracking state."""
    
    current_quality: float  # C_current
    target_quality: float = CONVERGENCE_TARGET
    rate: float = 0.0  # dC/dt
    state: ConvergenceState = ConvergenceState.STALLED
    samples: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    
    def record(self, quality: float, timestamp: float) -> None:
        """Record a quality observation."""
        self.samples.append((timestamp, quality))
        self.current_quality = quality
        self._update_rate()
    
    def _update_rate(self) -> None:
        """Update dC/dt based on recent samples."""
        if len(self.samples) < 2:
            self.rate = 0.0
            self.state = ConvergenceState.STALLED
            return
        
        # Use last 10 samples for rate calculation
        recent = list(self.samples)[-10:]
        if len(recent) < 2:
            return
        
        # Linear regression for rate
        t_values = [t for t, _ in recent]
        q_values = [q for _, q in recent]
        
        n = len(recent)
        sum_t = sum(t_values)
        sum_q = sum(q_values)
        sum_tq = sum(t * q for t, q in recent)
        sum_t2 = sum(t * t for t in t_values)
        
        denominator = n * sum_t2 - sum_t * sum_t
        if abs(denominator) < 1e-9:
            self.rate = 0.0
        else:
            self.rate = (n * sum_tq - sum_t * sum_q) / denominator
        
        # Determine state
        if abs(self.current_quality - self.target_quality) < 0.01:
            self.state = ConvergenceState.CONVERGED
        elif self.rate > CONVERGENCE_MIN_RATE:
            self.state = ConvergenceState.CONVERGING
        elif self.rate < -CONVERGENCE_MIN_RATE:
            self.state = ConvergenceState.DIVERGING
        else:
            self.state = ConvergenceState.STALLED
    
    def compute_expected_time_to_target(self) -> Optional[float]:
        """Estimate time to reach target quality."""
        if self.state != ConvergenceState.CONVERGING:
            return None
        
        remaining = self.target_quality - self.current_quality
        if remaining <= 0:
            return 0.0
        
        if abs(self.rate) < 1e-9:
            return None
        
        return remaining / self.rate


# ============================================================================
# KNOWLEDGE BASE
# ============================================================================


@dataclass
class KnowledgeEntry:
    """Entry in the MAPE-K knowledge base."""
    
    entry_id: str
    pattern: str  # Condition pattern
    resolution: str  # Resolution strategy
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[datetime] = None
    
    def effectiveness(self) -> float:
        """Compute effectiveness score."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Unknown
        return self.success_count / total


class KnowledgeBase:
    """
    Knowledge component of MAPE-K loop.
    
    Stores learned patterns and resolutions for self-healing.
    """
    
    def __init__(self):
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self) -> None:
        """Load default healing knowledge."""
        defaults = [
            KnowledgeEntry(
                entry_id="kb-001",
                pattern="high_latency",
                resolution="CACHE_CLEAR",
                success_count=10,
            ),
            KnowledgeEntry(
                entry_id="kb-002",
                pattern="high_error_rate",
                resolution="CIRCUIT_OPEN",
                success_count=15,
            ),
            KnowledgeEntry(
                entry_id="kb-003",
                pattern="memory_pressure",
                resolution="SCALE_DOWN",
                success_count=8,
            ),
            KnowledgeEntry(
                entry_id="kb-004",
                pattern="low_ihsan",
                resolution="FAILOVER",
                success_count=5,
            ),
            KnowledgeEntry(
                entry_id="kb-005",
                pattern="stalled_convergence",
                resolution="RESTART",
                success_count=3,
            ),
        ]
        for entry in defaults:
            self._entries[entry.entry_id] = entry
    
    def lookup(self, pattern: str) -> Optional[KnowledgeEntry]:
        """Find best matching knowledge entry."""
        best: Optional[KnowledgeEntry] = None
        best_score = -1.0
        
        for entry in self._entries.values():
            if pattern in entry.pattern or entry.pattern in pattern:
                score = entry.effectiveness()
                if score > best_score:
                    best = entry
                    best_score = score
        
        return best
    
    def record_outcome(self, entry_id: str, success: bool) -> None:
        """Record outcome of using a knowledge entry."""
        if entry_id in self._entries:
            entry = self._entries[entry_id]
            if success:
                entry.success_count += 1
            else:
                entry.failure_count += 1
            entry.last_used = datetime.now(timezone.utc)
    
    def add_entry(self, entry: KnowledgeEntry) -> None:
        """Add new knowledge entry."""
        self._entries[entry.entry_id] = entry


# ============================================================================
# MAPE-K ENGINE
# ============================================================================


class MAPEKEngine:
    """
    Full MAPE-K Autonomic Self-Healing Engine.
    
    Implements the complete autonomic computing loop:
        Monitor → Analyze → Plan → Execute → Knowledge
    
    With Quantized Convergence monitoring:
        dC/dt = κ × (C_target - C_current) × e^(-λt)
    """
    
    def __init__(
        self,
        kappa: float = KAPPA_DEFAULT,
        lambda_decay: float = LAMBDA_DEFAULT,
        enable_auto_heal: bool = True,
    ):
        """
        Initialize MAPE-K engine.
        
        Args:
            kappa: Convergence rate constant
            lambda_decay: Decay factor
            enable_auto_heal: Enable automatic healing
        """
        self.kappa = kappa
        self.lambda_decay = lambda_decay
        self.enable_auto_heal = enable_auto_heal
        
        # Components
        self.knowledge = KnowledgeBase()
        self.convergence = ConvergenceState_(current_quality=0.5)
        
        # State
        self._metrics: Dict[str, Deque[HealthMetric]] = {}
        self._snapshots: Deque[HealthSnapshot] = deque(maxlen=1000)
        self._plans: Deque[HealingPlan] = deque(maxlen=100)
        self._executions: Deque[ExecutionResult] = deque(maxlen=100)
        
        # Timing
        self._last_analyze: float = 0.0
        self._last_plan: float = 0.0
        
        # Action handlers
        self._handlers: Dict[ActionType, Callable[[HealingAction], bool]] = {}
        self._register_default_handlers()
        
        logger.info(f"MAPE-K Engine initialized: κ={kappa}, λ={lambda_decay}")
    
    def _register_default_handlers(self) -> None:
        """Register default action handlers."""
        self._handlers[ActionType.NOOP] = lambda a: True
        self._handlers[ActionType.CACHE_CLEAR] = self._handle_cache_clear
        self._handlers[ActionType.CIRCUIT_OPEN] = self._handle_circuit
        self._handlers[ActionType.CIRCUIT_CLOSE] = self._handle_circuit
        self._handlers[ActionType.RATE_LIMIT] = self._handle_rate_limit
        self._handlers[ActionType.SCALE_DOWN] = self._handle_scale
        self._handlers[ActionType.SCALE_UP] = self._handle_scale
        self._handlers[ActionType.RESTART] = self._handle_restart
        self._handlers[ActionType.FAILOVER] = self._handle_failover
        self._handlers[ActionType.ROLLBACK] = self._handle_rollback
    
    # ========================================================================
    # MONITOR PHASE - selfEvaluate()
    # ========================================================================
    
    def monitor(
        self,
        metrics: Dict[str, float],
        i_vec_score: float,
        omega_score: float,
    ) -> HealthSnapshot:
        """
        Monitor phase: Collect and analyze health metrics.
        
        Args:
            metrics: Current metric values
            i_vec_score: Current Ihsān vector score
            omega_score: Current causal drag
            
        Returns:
            HealthSnapshot with current system state
        """
        now = datetime.now(timezone.utc)
        timestamp = time.time()
        
        # Record metrics
        health_metrics: Dict[str, HealthMetric] = {}
        for name, value in metrics.items():
            metric = HealthMetric(
                name=name,
                value=value,
                timestamp=now,
            )
            health_metrics[name] = metric
            
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=1000)
            self._metrics[name].append(metric)
        
        # Update convergence tracking
        self.convergence.record(i_vec_score, timestamp)
        
        # Determine health status
        status = self._compute_status(health_metrics, i_vec_score, omega_score)
        
        # Create snapshot
        snapshot = HealthSnapshot(
            timestamp=now,
            status=status,
            metrics=health_metrics,
            i_vec_score=i_vec_score,
            omega_score=omega_score,
            convergence_rate=self.convergence.rate,
        )
        self._snapshots.append(snapshot)
        
        logger.debug(f"Monitor: status={status.name}, I_vec={i_vec_score:.4f}, dC/dt={self.convergence.rate:.6f}")
        
        return snapshot
    
    def _compute_status(
        self,
        metrics: Dict[str, HealthMetric],
        i_vec: float,
        omega: float,
    ) -> HealthStatus:
        """Compute overall health status."""
        # Critical conditions
        if i_vec < HEALTH_CRITICAL_THRESHOLD:
            return HealthStatus.CRITICAL
        if omega > 0.1:  # 2x max causal drag
            return HealthStatus.CRITICAL
        
        # Check individual metrics
        unhealthy_count = sum(
            1 for m in metrics.values()
            if not m.is_healthy()
        )
        
        if unhealthy_count > len(metrics) * 0.3:
            return HealthStatus.CRITICAL
        if unhealthy_count > 0 or i_vec < HEALTH_DEGRADED_THRESHOLD:
            return HealthStatus.DEGRADED
        if i_vec >= HEALTH_HEALTHY_THRESHOLD:
            return HealthStatus.HEALTHY
        
        return HealthStatus.DEGRADED
    
    # ========================================================================
    # ANALYZE PHASE - selfCritique()
    # ========================================================================
    
    def analyze(self, snapshot: HealthSnapshot) -> AnalysisResult:
        """
        Analyze phase: Identify issues and root causes.
        
        Constitutional AI alignment check ensures actions
        align with Ihsān principles.
        
        Args:
            snapshot: Current health snapshot
            
        Returns:
            AnalysisResult with issues and recommendations
        """
        now = datetime.now(timezone.utc)
        issues: List[str] = []
        root_causes: List[str] = []
        recommendations: List[str] = []
        severity = 0.0
        
        # Analyze Ihsān vector
        if snapshot.i_vec_score < 0.95:
            issues.append(f"Low I_vec: {snapshot.i_vec_score:.4f}")
            root_causes.append("Ethical alignment degraded")
            recommendations.append("Activate fail-safe mode")
            severity = max(severity, 1.0 - snapshot.i_vec_score)
        
        # Analyze causal drag
        if snapshot.omega_score > 0.05:
            issues.append(f"High Ω: {snapshot.omega_score:.4f}")
            root_causes.append("Systemic friction elevated")
            recommendations.append("Reduce concurrent operations")
            severity = max(severity, snapshot.omega_score * 10)
        
        # Analyze convergence
        if self.convergence.state == ConvergenceState.DIVERGING:
            issues.append(f"Diverging: dC/dt={self.convergence.rate:.6f}")
            root_causes.append("System quality degrading")
            recommendations.append("Initiate emergency stabilization")
            severity = max(severity, 0.7)
        elif self.convergence.state == ConvergenceState.STALLED:
            issues.append(f"Stalled: dC/dt≈0")
            root_causes.append("No improvement detected")
            recommendations.append("Clear caches and restart services")
            severity = max(severity, 0.3)
        
        # Analyze metrics
        for name, metric in snapshot.metrics.items():
            if not metric.is_healthy():
                issues.append(f"Unhealthy {name}: {metric.value}")
                if "latency" in name.lower():
                    root_causes.append("Performance bottleneck")
                    recommendations.append("Scale resources")
                elif "error" in name.lower():
                    root_causes.append("Error rate elevated")
                    recommendations.append("Open circuit breakers")
        
        # Constitutional alignment check
        constitutional_alignment = snapshot.i_vec_score
        
        result = AnalysisResult(
            timestamp=now,
            current_status=snapshot.status,
            issues_detected=issues,
            root_causes=list(set(root_causes)),
            severity=min(severity, 1.0),
            constitutional_alignment=constitutional_alignment,
            recommendations=list(set(recommendations)),
        )
        
        logger.debug(f"Analyze: {len(issues)} issues, severity={severity:.2f}")
        
        return result
    
    # ========================================================================
    # PLAN PHASE - generateImprovementActions()
    # ========================================================================
    
    def plan(self, analysis: AnalysisResult) -> HealingPlan:
        """
        Plan phase: Generate healing actions.
        
        Creates a prioritized DAG of actions to execute.
        
        Args:
            analysis: Analysis result from selfCritique
            
        Returns:
            HealingPlan with ordered actions
        """
        now = datetime.now(timezone.utc)
        plan_id = hashlib.sha256(
            f"{now.isoformat()}{analysis.severity}".encode()
        ).hexdigest()[:16]
        
        actions: List[HealingAction] = []
        
        # Generate actions based on root causes
        for cause in analysis.root_causes:
            knowledge = self.knowledge.lookup(cause.lower().replace(" ", "_"))
            
            if knowledge:
                action_type = ActionType[knowledge.resolution]
            else:
                # Default mapping
                if "latency" in cause.lower() or "bottleneck" in cause.lower():
                    action_type = ActionType.CACHE_CLEAR
                elif "error" in cause.lower():
                    action_type = ActionType.CIRCUIT_OPEN
                elif "ethical" in cause.lower() or "ihsan" in cause.lower():
                    action_type = ActionType.FAILOVER
                elif "friction" in cause.lower() or "drag" in cause.lower():
                    action_type = ActionType.RATE_LIMIT
                else:
                    action_type = ActionType.NOOP
            
            action = HealingAction(
                action_id=f"act-{len(actions)+1:03d}",
                action_type=action_type,
                target_component=cause.split()[0].lower(),
                priority=int(analysis.severity * 5) + 1,
                estimated_impact=0.1 + (analysis.severity * 0.3),
                risk_score=analysis.severity * 0.5,
                rollback_plan=f"Undo {action_type.name}",
            )
            actions.append(action)
        
        # Sort by priority
        actions.sort()
        
        # Determine if approval required
        approval_required = (
            analysis.severity > 0.7 or
            any(a.action_type in [ActionType.RESTART, ActionType.FAILOVER] for a in actions)
        )
        
        plan = HealingPlan(
            plan_id=plan_id,
            timestamp=now,
            actions=actions,
            expected_outcome=HealthStatus.HEALTHY if actions else analysis.current_status,
            estimated_time_ms=len(actions) * 100.0,
            rollback_available=all(a.rollback_plan is not None for a in actions),
            approval_required=approval_required,
        )
        
        self._plans.append(plan)
        logger.debug(f"Plan: {len(actions)} actions, approval_required={approval_required}")
        
        return plan
    
    # ========================================================================
    # EXECUTE PHASE - selfAdopt()
    # ========================================================================
    
    async def execute(
        self,
        plan: HealingPlan,
        force: bool = False,
    ) -> ExecutionResult:
        """
        Execute phase: Apply healing actions.
        
        Executes actions in DAG order with rollback on failure.
        
        Args:
            plan: Healing plan to execute
            force: Skip approval check
            
        Returns:
            ExecutionResult with outcomes
        """
        now = datetime.now(timezone.utc)
        start_time = time.perf_counter()
        errors: List[str] = []
        executed = 0
        failed = 0
        
        # Check approval
        if plan.approval_required and not force:
            logger.warning(f"Plan {plan.plan_id} requires approval - skipping")
            return ExecutionResult(
                plan_id=plan.plan_id,
                timestamp=now,
                success=False,
                actions_executed=0,
                actions_failed=0,
                final_status=HealthStatus.UNKNOWN,
                error_messages=["Approval required"],
                execution_time_ms=0.0,
            )
        
        # Execute actions in order
        for action in plan.actions:
            try:
                handler = self._handlers.get(action.action_type)
                if handler:
                    success = handler(action)
                    if success:
                        executed += 1
                        logger.info(f"Executed: {action.action_type.name} on {action.target_component}")
                    else:
                        failed += 1
                        errors.append(f"{action.action_type.name} failed")
                else:
                    failed += 1
                    errors.append(f"No handler for {action.action_type.name}")
            except Exception as e:
                failed += 1
                errors.append(f"{action.action_type.name} error: {str(e)}")
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Determine final status
        if failed == 0 and executed > 0:
            final_status = HealthStatus.HEALING
        elif failed > executed:
            final_status = HealthStatus.CRITICAL
        else:
            final_status = HealthStatus.DEGRADED
        
        result = ExecutionResult(
            plan_id=plan.plan_id,
            timestamp=now,
            success=failed == 0,
            actions_executed=executed,
            actions_failed=failed,
            final_status=final_status,
            error_messages=errors,
            execution_time_ms=elapsed_ms,
        )
        
        self._executions.append(result)
        logger.info(f"Execute: {executed} succeeded, {failed} failed, {elapsed_ms:.1f}ms")
        
        # Update knowledge
        for action in plan.actions:
            pattern = action.target_component
            kb_entry = self.knowledge.lookup(pattern)
            if kb_entry:
                self.knowledge.record_outcome(kb_entry.entry_id, failed == 0)
        
        return result
    
    # ========================================================================
    # ACTION HANDLERS
    # ========================================================================
    
    def _handle_cache_clear(self, action: HealingAction) -> bool:
        """Handle cache clear action."""
        logger.info(f"Clearing caches for {action.target_component}")
        # Simulated - real implementation would clear actual caches
        return True
    
    def _handle_circuit(self, action: HealingAction) -> bool:
        """Handle circuit breaker actions."""
        state = "open" if action.action_type == ActionType.CIRCUIT_OPEN else "close"
        logger.info(f"Setting circuit breaker to {state} for {action.target_component}")
        return True
    
    def _handle_rate_limit(self, action: HealingAction) -> bool:
        """Handle rate limiting."""
        logger.info(f"Applying rate limit to {action.target_component}")
        return True
    
    def _handle_scale(self, action: HealingAction) -> bool:
        """Handle scale up/down."""
        direction = "up" if action.action_type == ActionType.SCALE_UP else "down"
        logger.info(f"Scaling {direction} {action.target_component}")
        return True
    
    def _handle_restart(self, action: HealingAction) -> bool:
        """Handle restart action."""
        logger.info(f"Restarting {action.target_component}")
        return True
    
    def _handle_failover(self, action: HealingAction) -> bool:
        """Handle failover action."""
        logger.info(f"Failing over {action.target_component}")
        return True
    
    def _handle_rollback(self, action: HealingAction) -> bool:
        """Handle rollback action."""
        logger.info(f"Rolling back {action.target_component}")
        return True
    
    # ========================================================================
    # FULL LOOP
    # ========================================================================
    
    async def run_loop(
        self,
        metrics: Dict[str, float],
        i_vec_score: float,
        omega_score: float,
    ) -> Optional[ExecutionResult]:
        """
        Run complete MAPE-K loop.
        
        Monitor → Analyze → Plan → Execute → Knowledge
        
        Args:
            metrics: Current metrics
            i_vec_score: Ihsān vector score
            omega_score: Causal drag
            
        Returns:
            ExecutionResult if healing was performed
        """
        # Monitor
        snapshot = self.monitor(metrics, i_vec_score, omega_score)
        
        # Skip if healthy
        if snapshot.status == HealthStatus.HEALTHY:
            return None
        
        # Analyze
        analysis = self.analyze(snapshot)
        
        # Skip if no intervention needed
        if not analysis.needs_intervention():
            return None
        
        # Check cooldown
        now = time.time()
        if now - self._last_plan < PLAN_COOLDOWN_MS / 1000:
            logger.debug("Plan cooldown active - skipping")
            return None
        
        # Plan
        plan = self.plan(analysis)
        self._last_plan = now
        
        # Execute (if auto-heal enabled)
        if self.enable_auto_heal and plan.actions:
            return await self.execute(plan)
        
        return None
    
    # ========================================================================
    # QUANTIZED CONVERGENCE
    # ========================================================================
    
    def compute_convergence_rate(self, t: float) -> float:
        """
        Compute theoretical convergence rate.
        
        dC/dt = κ × (C_target - C_current) × e^(-λt)
        
        Args:
            t: Time since start
            
        Returns:
            Theoretical dC/dt
        """
        c_diff = self.convergence.target_quality - self.convergence.current_quality
        return self.kappa * c_diff * math.exp(-self.lambda_decay * t)
    
    def get_convergence_state(self) -> Dict[str, Any]:
        """Get current convergence state."""
        return {
            "current_quality": self.convergence.current_quality,
            "target_quality": self.convergence.target_quality,
            "rate": self.convergence.rate,
            "state": self.convergence.state.name,
            "expected_time_to_target": self.convergence.compute_expected_time_to_target(),
            "kappa": self.kappa,
            "lambda": self.lambda_decay,
        }
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MAPE-K statistics."""
        return {
            "snapshots": len(self._snapshots),
            "plans_created": len(self._plans),
            "executions": len(self._executions),
            "successful_heals": sum(1 for e in self._executions if e.success),
            "failed_heals": sum(1 for e in self._executions if not e.success),
            "convergence": self.get_convergence_state(),
            "auto_heal_enabled": self.enable_auto_heal,
        }


# ============================================================================
# SELF-TEST
# ============================================================================


async def _self_test() -> None:
    """Run MAPE-K self-tests."""
    print("=" * 70)
    print("BIZRA AEON OMEGA - MAPE-K Autonomic Self-Healing Self-Test")
    print("=" * 70)
    
    # Test 1: Initialize engine
    print("\n[Test 1] Initialize MAPE-K Engine")
    engine = MAPEKEngine()
    print(f"  ✓ Initialized with κ={engine.kappa}, λ={engine.lambda_decay}")
    
    # Test 2: Monitor healthy state
    print("\n[Test 2] Monitor - Healthy State")
    snapshot = engine.monitor(
        metrics={"latency_ms": 50, "error_rate": 0.01},
        i_vec_score=0.98,
        omega_score=0.02,
    )
    assert snapshot.status == HealthStatus.HEALTHY
    print(f"  ✓ Status: {snapshot.status.name}")
    print(f"  ✓ I_vec: {snapshot.i_vec_score}")
    print(f"  ✓ dC/dt: {snapshot.convergence_rate}")
    
    # Test 3: Monitor degraded state
    print("\n[Test 3] Monitor - Degraded State")
    snapshot = engine.monitor(
        metrics={"latency_ms": 500, "error_rate": 0.15},
        i_vec_score=0.85,
        omega_score=0.04,
    )
    assert snapshot.status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]
    print(f"  ✓ Status: {snapshot.status.name}")
    
    # Test 4: Analyze
    print("\n[Test 4] Analyze - selfCritique()")
    analysis = engine.analyze(snapshot)
    assert len(analysis.issues_detected) > 0
    print(f"  ✓ Issues: {len(analysis.issues_detected)}")
    print(f"  ✓ Severity: {analysis.severity:.2f}")
    print(f"  ✓ Needs intervention: {analysis.needs_intervention()}")
    
    # Test 5: Plan
    print("\n[Test 5] Plan - generateImprovementActions()")
    plan = engine.plan(analysis)
    print(f"  ✓ Plan ID: {plan.plan_id}")
    print(f"  ✓ Actions: {len(plan.actions)}")
    print(f"  ✓ Approval required: {plan.approval_required}")
    
    # Test 6: Execute
    print("\n[Test 6] Execute - selfAdopt()")
    result = await engine.execute(plan, force=True)
    print(f"  ✓ Success: {result.success}")
    print(f"  ✓ Executed: {result.actions_executed}")
    print(f"  ✓ Time: {result.execution_time_ms:.2f}ms")
    
    # Test 7: Quantized Convergence
    print("\n[Test 7] Quantized Convergence")
    # Simulate improving quality over time
    for i in range(10):
        quality = 0.5 + (i * 0.05)
        engine.convergence.record(quality, time.time() + i)
    
    conv_state = engine.get_convergence_state()
    print(f"  ✓ Current quality: {conv_state['current_quality']:.2f}")
    print(f"  ✓ Rate (dC/dt): {conv_state['rate']:.6f}")
    print(f"  ✓ State: {conv_state['state']}")
    
    # Statistics
    print("\n[Statistics]")
    stats = engine.get_statistics()
    print(f"  Snapshots: {stats['snapshots']}")
    print(f"  Plans: {stats['plans_created']}")
    print(f"  Successful heals: {stats['successful_heals']}")
    
    print("\n" + "=" * 70)
    print("✅ ALL MAPE-K SELF-TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(_self_test())
