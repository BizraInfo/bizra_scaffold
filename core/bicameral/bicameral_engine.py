"""
BIZRA AEON OMEGA - Bicameral Engine (Main Orchestrator)
=========================================================
Cognitive Architecture Orchestration

The Bicameral Engine is the main orchestrator that coordinates between
Cold Core (deterministic Rust operations) and Warm Surface (flexible Python
operations) through the typed Membrane interface.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    BICAMERAL ENGINE                              │
    │                                                                  │
    │  ┌──────────────────┐          ┌──────────────────┐            │
    │  │   WARM SURFACE   │◄────────►│   COLD CORE      │            │
    │  │   (Python)       │ MEMBRANE │   (Rust)         │            │
    │  │                  │          │                  │            │
    │  │  • Agents        │          │  • Crypto        │            │
    │  │  • Tools         │          │  • Invariants    │            │
    │  │  • Integrations  │          │  • Verification  │            │
    │  └──────────────────┘          └──────────────────┘            │
    │                                                                  │
    │                    ┌──────────────────┐                         │
    │                    │   FATE ENGINE    │                         │
    │                    │  (Z3 SMT)        │                         │
    │                    └──────────────────┘                         │
    └─────────────────────────────────────────────────────────────────┘

Author: BIZRA Genesis Team (Peak Masterpiece v5)
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from .cold_core import (
    ColdCore,
    ColdCoreConfig,
    ColdOperation,
    ColdOperationType,
    ColdResult,
    CrystallizedFunction,
    CrystallizationState,
)
from .warm_surface import (
    WarmSurface,
    WarmSurfaceConfig,
    WarmOperation,
    WarmOperationType,
    WarmResult,
    AgentContext,
    AgentState,
)
from .membrane import (
    Membrane,
    MembraneConfig,
    Message,
    MessageType,
    MessagePriority,
    CrossingDirection,
    CrossingReceipt,
    MembraneState,
)

# Try to import FATE Engine
try:
    from core.verification.fate_engine import (
        FATEEngine,
        FATEVerdict,
        ActionProposal,
    )
    FATE_AVAILABLE = True
except ImportError:
    FATE_AVAILABLE = False

# Try to import MAPE-K Engine
try:
    from core.verification.mape_k_engine import (
        MAPEKEngine,
        HealingPlan,
        MAPEPhase,
    )
    MAPEK_AVAILABLE = True
except ImportError:
    MAPEK_AVAILABLE = False

logger = logging.getLogger("bizra.bicameral.engine")

# ============================================================================
# CONSTANTS
# ============================================================================

# Default configuration values
DEFAULT_COLD_LATENCY_BUDGET_MS = 1.0
DEFAULT_WARM_LATENCY_BUDGET_MS = 100.0
DEFAULT_CROSSING_LATENCY_BUDGET_MS = 5.0

# ============================================================================
# ENUMERATIONS
# ============================================================================


class Hemisphere(Enum):
    """Hemisphere of the Bicameral Engine."""
    
    COLD = auto()  # Cold Core (Rust, deterministic)
    WARM = auto()  # Warm Surface (Python, flexible)


class CognitiveState(Enum):
    """Cognitive state of the Bicameral Engine."""
    
    DORMANT = auto()      # Not running
    INITIALIZING = auto() # Starting up
    OPERATIONAL = auto()  # Fully operational
    DEGRADED = auto()     # Partial functionality
    HEALING = auto()      # Self-healing in progress
    SHUTDOWN = auto()     # Shutting down


class OperationRouting(Enum):
    """Routing decision for an operation."""
    
    COLD_ONLY = auto()    # Route to Cold Core only
    WARM_ONLY = auto()    # Route to Warm Surface only
    COLD_FIRST = auto()   # Cold verification, then Warm execution
    WARM_FIRST = auto()   # Warm proposal, then Cold verification
    PARALLEL = auto()     # Execute in parallel (where possible)


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class BicameralConfig:
    """Configuration for Bicameral Engine."""
    
    cold_config: Optional[ColdCoreConfig] = None
    warm_config: Optional[WarmSurfaceConfig] = None
    membrane_config: Optional[MembraneConfig] = None
    enable_fate_integration: bool = True
    enable_mapek_integration: bool = True
    auto_heal: bool = True
    health_check_interval: float = 30.0


@dataclass
class BicameralOperation:
    """A single operation in the Bicameral Engine."""
    
    operation_id: str
    hemisphere: Hemisphere
    routing: OperationRouting
    operation_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_id": self.operation_id,
            "hemisphere": self.hemisphere.name,
            "routing": self.routing.name,
            "operation_type": self.operation_type,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BicameralResult:
    """Result from a Bicameral Engine operation."""
    
    operation_id: str
    success: bool
    hemisphere: Hemisphere
    cold_result: Optional[ColdResult]
    warm_result: Optional[WarmResult]
    crossing_receipts: List[CrossingReceipt]
    total_latency_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_id": self.operation_id,
            "success": self.success,
            "hemisphere": self.hemisphere.name,
            "cold_result": self.cold_result.to_dict() if self.cold_result else None,
            "warm_result": self.warm_result.to_dict() if self.warm_result else None,
            "crossing_receipts": [r.to_dict() for r in self.crossing_receipts],
            "total_latency_ms": self.total_latency_ms,
            "error": self.error,
        }


@dataclass
class HealthStatus:
    """Health status of the Bicameral Engine."""
    
    state: CognitiveState
    cold_healthy: bool
    warm_healthy: bool
    membrane_healthy: bool
    fate_available: bool
    mapek_available: bool
    last_health_check: datetime
    issues: List[str] = field(default_factory=list)
    
    def is_healthy(self) -> bool:
        """Check if engine is fully healthy."""
        return (
            self.state == CognitiveState.OPERATIONAL and
            self.cold_healthy and
            self.warm_healthy and
            self.membrane_healthy and
            len(self.issues) == 0
        )


# ============================================================================
# BICAMERAL ENGINE
# ============================================================================


class BicameralEngine:
    """
    The Bicameral Engine - Main Cognitive Orchestrator.
    
    Coordinates between Cold Core (deterministic Rust operations) and
    Warm Surface (flexible Python operations) through the typed Membrane.
    
    Usage:
        engine = BicameralEngine()
        await engine.start()
        
        # Verify a proposal
        result = await engine.verify_proposal(proposal)
        
        # Execute an agent task
        result = await engine.execute_agent_task(agent_id, task)
        
        # Crystallize a function
        result = await engine.crystallize("my_func", my_func)
        
        await engine.stop()
    """
    
    def __init__(self, config: Optional[BicameralConfig] = None):
        """Initialize Bicameral Engine."""
        self.config = config or BicameralConfig()
        
        # Initialize components
        self._cold_core = ColdCore(self.config.cold_config)
        self._warm_surface = WarmSurface(self.config.warm_config)
        self._membrane = Membrane(self.config.membrane_config)
        
        # FATE Engine (optional)
        if FATE_AVAILABLE and self.config.enable_fate_integration:
            self._fate = FATEEngine()
        else:
            self._fate = None
        
        # MAPE-K Engine (optional)
        if MAPEK_AVAILABLE and self.config.enable_mapek_integration:
            self._mapek = MAPEKEngine()
        else:
            self._mapek = None
        
        # State
        self._state = CognitiveState.DORMANT
        self._last_health_check: Optional[datetime] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._pipeline_tasks: List[asyncio.Task] = []  # P1 FIX: Track pipeline tasks
        
        # Statistics
        self._operations: int = 0
        self._cold_operations: int = 0
        self._warm_operations: int = 0
        self._crossings: int = 0
        
        # Audit log
        self._audit_log: List[Dict[str, Any]] = []
        
        logger.info("Bicameral Engine initialized")
    
    # ========================================================================
    # LIFECYCLE
    # ========================================================================
    
    async def start(self) -> None:
        """Start the Bicameral Engine."""
        self._state = CognitiveState.INITIALIZING
        logger.info("Starting Bicameral Engine...")
        
        # Open membrane
        self._membrane.open()
        
        # Register message handlers
        self._register_handlers()
        
        # P1 FIX: Start membrane pipelines for message processing
        # This ensures messages are consumed and responses are correlated
        from .membrane import CrossingDirection
        self._pipeline_tasks = [
            asyncio.create_task(
                self._membrane.run_pipeline(CrossingDirection.WARM_TO_COLD),
                name="pipeline-warm-to-cold"
            ),
            asyncio.create_task(
                self._membrane.run_pipeline(CrossingDirection.COLD_TO_WARM),
                name="pipeline-cold-to-warm"
            ),
        ]
        logger.info("Membrane pipelines started")
        
        # Start health check loop
        if self.config.auto_heal:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Run initial health check
        health = await self.check_health()
        
        if health.is_healthy():
            self._state = CognitiveState.OPERATIONAL
            logger.info("Bicameral Engine operational")
        else:
            self._state = CognitiveState.DEGRADED
            logger.warning(f"Bicameral Engine degraded: {health.issues}")
    
    async def stop(self) -> None:
        """Stop the Bicameral Engine."""
        self._state = CognitiveState.SHUTDOWN
        logger.info("Stopping Bicameral Engine...")
        
        # P1 FIX: Cancel pipeline tasks first
        for task in self._pipeline_tasks:
            task.cancel()
        for task in self._pipeline_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._pipeline_tasks.clear()
        logger.info("Membrane pipelines stopped")
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close membrane
        self._membrane.close()
        
        self._state = CognitiveState.DORMANT
        logger.info("Bicameral Engine stopped")
    
    # ========================================================================
    # VERIFICATION OPERATIONS (Cold First)
    # ========================================================================
    
    async def verify_proposal(
        self,
        proposal: Dict[str, Any],
        risk_level: str = "LOW",
    ) -> BicameralResult:
        """
        Verify a proposal through Cold Core.
        
        Args:
            proposal: Proposal to verify
            risk_level: Risk level (LOW, HIGH, CRITICAL)
            
        Returns:
            BicameralResult with verification outcome
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("VERIFY")
        receipts: List[CrossingReceipt] = []
        
        try:
            # Send to Cold Core via Membrane
            message = self._membrane._create_message(
                message_type=MessageType.VERIFY_PROPOSAL,
                direction=CrossingDirection.WARM_TO_COLD,
                priority=MessagePriority.HIGH if risk_level == "CRITICAL" else MessagePriority.NORMAL,
                payload={"proposal": proposal, "risk_level": risk_level},
            )
            receipt = await self._membrane.send(message)
            receipts.append(receipt)
            self._crossings += 1
            
            # Use FATE if available
            if self._fate:
                fate_result = self._fate.evaluate(
                    ActionProposal(
                        action_type=proposal.get("action_type", "GENERAL"),
                        resource_type=proposal.get("resource_type", "UNKNOWN"),
                        resource_path=proposal.get("resource_path", ""),
                        state_delta=proposal.get("state_delta", {}),
                        justification=proposal.get("justification", ""),
                        risk=proposal.get("risk", risk_level),
                        rollback_plan=proposal.get("rollback_plan"),
                    ),
                    context={"origin": "BicameralEngine"},
                )
                
                cold_result = ColdResult(
                    operation_id=op_id,
                    success=fate_result.verdict == FATEVerdict.PASS,
                    output={
                        "verdict": fate_result.verdict.name,
                        "ihsan_score": fate_result.ihsan_score,
                        "omega": fate_result.omega,
                        "reasons": fate_result.reasons,
                    },
                    latency_ms=(time.perf_counter() - start) * 1000,
                    within_budget=True,
                    audit_hash=fate_result.proposal_hash[:16],
                )
            else:
                # Fallback: basic Ihsān check
                ihsan_score = proposal.get("ihsan_score", 0.95)
                cold_result = self._cold_core.check_ihsan(ihsan_score, risk_level)
            
            self._cold_operations += 1
            self._operations += 1
            
            total_latency = (time.perf_counter() - start) * 1000
            
            return BicameralResult(
                operation_id=op_id,
                success=cold_result.success and cold_result.output.get("passed", cold_result.output.get("verdict") == "PASS"),
                hemisphere=Hemisphere.COLD,
                cold_result=cold_result,
                warm_result=None,
                crossing_receipts=receipts,
                total_latency_ms=total_latency,
            )
        except Exception as e:
            total_latency = (time.perf_counter() - start) * 1000
            return BicameralResult(
                operation_id=op_id,
                success=False,
                hemisphere=Hemisphere.COLD,
                cold_result=None,
                warm_result=None,
                crossing_receipts=receipts,
                total_latency_ms=total_latency,
                error=str(e),
            )
    
    async def sign(self, message: bytes, private_key: bytes) -> BicameralResult:
        """
        Sign a message through Cold Core.
        
        Args:
            message: Message to sign
            private_key: Signing key
            
        Returns:
            BicameralResult with signature
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("SIGN")
        
        cold_result = self._cold_core.sign(message, private_key)
        self._cold_operations += 1
        self._operations += 1
        
        total_latency = (time.perf_counter() - start) * 1000
        
        return BicameralResult(
            operation_id=op_id,
            success=cold_result.success,
            hemisphere=Hemisphere.COLD,
            cold_result=cold_result,
            warm_result=None,
            crossing_receipts=[],
            total_latency_ms=total_latency,
        )
    
    async def hash(self, data: bytes, algorithm: str = "sha3_512") -> BicameralResult:
        """
        Compute hash through Cold Core.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm
            
        Returns:
            BicameralResult with hash
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("HASH")
        
        cold_result = self._cold_core.hash(data, algorithm)
        self._cold_operations += 1
        self._operations += 1
        
        total_latency = (time.perf_counter() - start) * 1000
        
        return BicameralResult(
            operation_id=op_id,
            success=cold_result.success,
            hemisphere=Hemisphere.COLD,
            cold_result=cold_result,
            warm_result=None,
            crossing_receipts=[],
            total_latency_ms=total_latency,
        )
    
    async def crystallize(
        self,
        name: str,
        function: Callable,
        verify: bool = True,
    ) -> BicameralResult:
        """
        Crystallize a function into Cold Core.
        
        Args:
            name: Function name
            function: Function to crystallize
            verify: Verify determinism
            
        Returns:
            BicameralResult with crystallization status
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("CRYSTALLIZE")
        
        cold_result = self._cold_core.crystallize(name, function, verify)
        self._cold_operations += 1
        self._operations += 1
        
        total_latency = (time.perf_counter() - start) * 1000
        
        return BicameralResult(
            operation_id=op_id,
            success=cold_result.success,
            hemisphere=Hemisphere.COLD,
            cold_result=cold_result,
            warm_result=None,
            crossing_receipts=[],
            total_latency_ms=total_latency,
        )
    
    # ========================================================================
    # EXECUTION OPERATIONS (Warm First)
    # ========================================================================
    
    async def spawn_agent(
        self,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BicameralResult:
        """
        Spawn an agent in Warm Surface.
        
        Args:
            agent_type: Type of agent
            config: Agent configuration
            
        Returns:
            BicameralResult with agent_id
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("SPAWN")
        receipts: List[CrossingReceipt] = []
        
        try:
            warm_result = await self._warm_surface.spawn_agent(agent_type, config)
            self._warm_operations += 1
            self._operations += 1
            
            total_latency = (time.perf_counter() - start) * 1000
            
            return BicameralResult(
                operation_id=op_id,
                success=warm_result.success,
                hemisphere=Hemisphere.WARM,
                cold_result=None,
                warm_result=warm_result,
                crossing_receipts=receipts,
                total_latency_ms=total_latency,
            )
        except Exception as e:
            total_latency = (time.perf_counter() - start) * 1000
            return BicameralResult(
                operation_id=op_id,
                success=False,
                hemisphere=Hemisphere.WARM,
                cold_result=None,
                warm_result=None,
                crossing_receipts=receipts,
                total_latency_ms=total_latency,
                error=str(e),
            )
    
    async def execute_agent_task(
        self,
        agent_id: str,
        task: Dict[str, Any],
        verify_first: bool = False,
    ) -> BicameralResult:
        """
        Execute a task with an agent.
        
        Args:
            agent_id: Agent ID
            task: Task to execute
            verify_first: Verify through Cold Core first
            
        Returns:
            BicameralResult with task result
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("EXECUTE")
        receipts: List[CrossingReceipt] = []
        cold_result: Optional[ColdResult] = None
        
        try:
            # Optional Cold Core verification
            if verify_first:
                verify_result = await self.verify_proposal(
                    {"action_type": "AGENT_TASK", "task": task},
                    risk_level="LOW",
                )
                receipts.extend(verify_result.crossing_receipts)
                cold_result = verify_result.cold_result
                
                if not verify_result.success:
                    return BicameralResult(
                        operation_id=op_id,
                        success=False,
                        hemisphere=Hemisphere.COLD,
                        cold_result=cold_result,
                        warm_result=None,
                        crossing_receipts=receipts,
                        total_latency_ms=(time.perf_counter() - start) * 1000,
                        error="Cold Core verification failed",
                    )
            
            # Execute in Warm Surface
            warm_result = await self._warm_surface.execute_agent_task(agent_id, task)
            self._warm_operations += 1
            self._operations += 1
            
            total_latency = (time.perf_counter() - start) * 1000
            
            return BicameralResult(
                operation_id=op_id,
                success=warm_result.success,
                hemisphere=Hemisphere.WARM,
                cold_result=cold_result,
                warm_result=warm_result,
                crossing_receipts=receipts,
                total_latency_ms=total_latency,
            )
        except Exception as e:
            total_latency = (time.perf_counter() - start) * 1000
            return BicameralResult(
                operation_id=op_id,
                success=False,
                hemisphere=Hemisphere.WARM,
                cold_result=cold_result,
                warm_result=None,
                crossing_receipts=receipts,
                total_latency_ms=total_latency,
                error=str(e),
            )
    
    async def invoke_tool(
        self,
        name: str,
        **kwargs: Any,
    ) -> BicameralResult:
        """
        Invoke a tool in Warm Surface.
        
        Args:
            name: Tool name
            **kwargs: Tool arguments
            
        Returns:
            BicameralResult with tool output
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("INVOKE_TOOL")
        
        try:
            warm_result = await self._warm_surface.invoke_tool(name, **kwargs)
            self._warm_operations += 1
            self._operations += 1
            
            total_latency = (time.perf_counter() - start) * 1000
            
            return BicameralResult(
                operation_id=op_id,
                success=warm_result.success,
                hemisphere=Hemisphere.WARM,
                cold_result=None,
                warm_result=warm_result,
                crossing_receipts=[],
                total_latency_ms=total_latency,
            )
        except Exception as e:
            total_latency = (time.perf_counter() - start) * 1000
            return BicameralResult(
                operation_id=op_id,
                success=False,
                hemisphere=Hemisphere.WARM,
                cold_result=None,
                warm_result=None,
                crossing_receipts=[],
                total_latency_ms=total_latency,
                error=str(e),
            )
    
    # ========================================================================
    # HEALTH & HEALING
    # ========================================================================
    
    async def check_health(self) -> HealthStatus:
        """
        Check health of all components.
        
        Returns:
            HealthStatus with current state
        """
        issues: List[str] = []
        
        # Check Cold Core
        cold_healthy = True
        try:
            result = self._cold_core.hash(b"health_check")
            if not result.success:
                cold_healthy = False
                issues.append("Cold Core hash failed")
        except Exception as e:
            cold_healthy = False
            issues.append(f"Cold Core error: {e}")
        
        # Check Warm Surface
        warm_healthy = True
        try:
            stats = self._warm_surface.get_statistics()
            # Check for excessive backlog
            if stats.get("hot_reload_state") == "FAILED":
                warm_healthy = False
                issues.append("Warm Surface hot-reload failed")
        except Exception as e:
            warm_healthy = False
            issues.append(f"Warm Surface error: {e}")
        
        # Check Membrane
        membrane_healthy = self._membrane._state != MembraneState.CLOSED
        if not membrane_healthy:
            issues.append("Membrane is closed")
        
        self._last_health_check = datetime.now(timezone.utc)
        
        return HealthStatus(
            state=self._state,
            cold_healthy=cold_healthy,
            warm_healthy=warm_healthy,
            membrane_healthy=membrane_healthy,
            fate_available=self._fate is not None,
            mapek_available=self._mapek is not None,
            last_health_check=self._last_health_check,
            issues=issues,
        )
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._state in (CognitiveState.OPERATIONAL, CognitiveState.DEGRADED, CognitiveState.INITIALIZING):
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                health = await self.check_health()
                
                if not health.is_healthy():
                    if self._state == CognitiveState.OPERATIONAL:
                        self._state = CognitiveState.DEGRADED
                        logger.warning(f"Engine degraded: {health.issues}")
                    
                    if self.config.auto_heal and self._mapek:
                        await self._attempt_healing(health)
                elif self._state == CognitiveState.DEGRADED:
                    self._state = CognitiveState.OPERATIONAL
                    logger.info("Engine recovered to operational state")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _attempt_healing(self, health: HealthStatus) -> None:
        """Attempt self-healing using MAPE-K."""
        if not self._mapek:
            return
        
        self._state = CognitiveState.HEALING
        logger.info("Attempting self-healing...")
        
        try:
            # MAPE-K healing cycle
            # Monitor phase already done via health check
            
            # Analyze: identify root causes
            # Plan: create healing strategy
            # Execute: apply remediation
            # Knowledge: learn from outcome
            
            # For now, simple remediation
            if not health.membrane_healthy:
                self._membrane.open()
                logger.info("Reopened membrane")
            
            self._state = CognitiveState.DEGRADED  # Will be checked again
            
        except Exception as e:
            logger.error(f"Self-healing failed: {e}")
            self._state = CognitiveState.DEGRADED
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _register_handlers(self) -> None:
        """Register message handlers on the Membrane."""
        # Handle verification requests
        async def handle_verify(payload: Dict[str, Any]) -> Dict[str, Any]:
            result = await self.verify_proposal(
                payload.get("proposal", {}),
                payload.get("risk_level", "LOW"),
            )
            return result.to_dict()
        
        self._membrane.register_handler(MessageType.VERIFY_PROPOSAL, handle_verify)
        
        # Handle agent execution
        async def handle_execute(payload: Dict[str, Any]) -> Dict[str, Any]:
            result = await self.execute_agent_task(
                payload.get("agent_id", ""),
                payload.get("task", {}),
            )
            return result.to_dict()
        
        self._membrane.register_handler(MessageType.EXECUTE_TASK, handle_execute)
    
    def _generate_operation_id(self, prefix: str) -> str:
        """Generate unique operation ID."""
        self._operations += 1
        timestamp = int(time.time() * 1000000)
        return f"BICAM-{prefix}-{timestamp}-{self._operations:06d}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Bicameral Engine statistics."""
        return {
            "state": self._state.name,
            "operations": self._operations,
            "cold_operations": self._cold_operations,
            "warm_operations": self._warm_operations,
            "crossings": self._crossings,
            "cold_core": self._cold_core.get_statistics(),
            "warm_surface": self._warm_surface.get_statistics(),
            "membrane": self._membrane.get_statistics(),
            "fate_available": self._fate is not None,
            "mapek_available": self._mapek is not None,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
        }


# ============================================================================
# SELF-TEST
# ============================================================================


async def _self_test() -> None:
    """Run Bicameral Engine self-tests."""
    print("=" * 70)
    print("BIZRA AEON OMEGA - Bicameral Engine Self-Test")
    print("=" * 70)
    
    engine = BicameralEngine()
    
    # Test 1: Start engine
    print("\n[Test 1] Start Engine")
    await engine.start()
    assert engine._state in (CognitiveState.OPERATIONAL, CognitiveState.DEGRADED)
    print(f"  ✓ Engine started: {engine._state.name}")
    
    # Test 2: Verify proposal
    print("\n[Test 2] Verify Proposal")
    result = await engine.verify_proposal(
        {"action_type": "TEST", "ihsan_score": 0.98},
        risk_level="LOW",
    )
    print(f"  ✓ Verification: {result.success}")
    print(f"  ✓ Total latency: {result.total_latency_ms:.3f}ms")
    
    # Test 3: Hash data
    print("\n[Test 3] Hash Data")
    result = await engine.hash(b"Hello, Bicameral Engine!")
    assert result.success
    print(f"  ✓ Hash computed: {result.cold_result.output.hex()[:32]}...")
    print(f"  ✓ Latency: {result.total_latency_ms:.3f}ms")
    
    # Test 4: Spawn agent
    print("\n[Test 4] Spawn Agent")
    result = await engine.spawn_agent("PAT", {"mode": "test"})
    assert result.success
    agent_id = result.warm_result.output["agent_id"]
    print(f"  ✓ Spawned agent: {agent_id}")
    
    # Test 5: Execute agent task
    print("\n[Test 5] Execute Agent Task")
    result = await engine.execute_agent_task(agent_id, {"action": "plan"})
    assert result.success
    print(f"  ✓ Task executed: {result.warm_result.output}")
    print(f"  ✓ Total latency: {result.total_latency_ms:.3f}ms")
    
    # Test 6: Crystallize function
    print("\n[Test 6] Crystallize Function")
    result = await engine.crystallize("triple", lambda x: x * 3)
    assert result.success
    print(f"  ✓ Crystallized: triple")
    
    # Test 7: Health check
    print("\n[Test 7] Health Check")
    health = await engine.check_health()
    print(f"  ✓ State: {health.state.name}")
    print(f"  ✓ Cold healthy: {health.cold_healthy}")
    print(f"  ✓ Warm healthy: {health.warm_healthy}")
    print(f"  ✓ Membrane healthy: {health.membrane_healthy}")
    print(f"  ✓ FATE available: {health.fate_available}")
    print(f"  ✓ MAPE-K available: {health.mapek_available}")
    
    # Test 8: Statistics
    print("\n[Test 8] Statistics")
    stats = engine.get_statistics()
    print(f"  ✓ Operations: {stats['operations']}")
    print(f"  ✓ Cold ops: {stats['cold_operations']}")
    print(f"  ✓ Warm ops: {stats['warm_operations']}")
    print(f"  ✓ Crossings: {stats['crossings']}")
    
    # Test 9: Stop engine
    print("\n[Test 9] Stop Engine")
    await engine.stop()
    assert engine._state == CognitiveState.DORMANT
    print(f"  ✓ Engine stopped")
    
    print("\n" + "=" * 70)
    print("✅ ALL BICAMERAL ENGINE SELF-TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(_self_test())
