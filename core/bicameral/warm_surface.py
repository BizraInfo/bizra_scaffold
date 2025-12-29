"""
BIZRA AEON OMEGA - Warm Surface (Flexible Python Layer)
=========================================================
Hot-Reloadable Agent Orchestration

The Warm Surface is the "neocortex" of the Bicameral Engine - handling
high-level, flexible operations that benefit from Python's dynamic nature.

Characteristics:
    - Hot-reloadable without affecting Cold Core
    - Flexible, adaptable behavior
    - Rich integrations with external systems
    - Agent orchestration (PAT/SAT)
    - UI and API interactions

Operations Handled:
    - Agent execution (PAT/SAT)
    - Tool invocation
    - External API calls
    - User interaction
    - State management

Author: BIZRA Genesis Team (Peak Masterpiece v5)
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
)

logger = logging.getLogger("bizra.bicameral.warm_surface")

# ============================================================================
# CONSTANTS
# ============================================================================

# Warm Surface latency tolerance (10-100ms is acceptable)
WARM_SURFACE_LATENCY_BUDGET_MS = 100.0

# Maximum concurrent operations
MAX_CONCURRENT_OPERATIONS = 50

# Hot-reload interval (check for changes every N seconds)
HOT_RELOAD_CHECK_INTERVAL = 5.0

# ============================================================================
# ENUMERATIONS
# ============================================================================


class WarmOperationType(Enum):
    """Types of operations handled by Warm Surface."""
    
    # Agent operations
    AGENT_EXECUTE = auto()
    AGENT_SPAWN = auto()
    AGENT_TERMINATE = auto()
    
    # Tool operations
    TOOL_INVOKE = auto()
    TOOL_REGISTER = auto()
    TOOL_UNREGISTER = auto()
    
    # Integration operations
    API_CALL = auto()
    EVENT_EMIT = auto()
    EVENT_SUBSCRIBE = auto()
    
    # State operations
    STATE_READ = auto()
    STATE_WRITE = auto()
    STATE_SNAPSHOT = auto()


class AgentState(Enum):
    """State of an agent in the Warm Surface."""
    
    IDLE = auto()         # Awaiting tasks
    EXECUTING = auto()    # Processing a task
    SUSPENDED = auto()    # Temporarily paused
    TERMINATED = auto()   # Stopped permanently


class HotReloadState(Enum):
    """State of hot-reload system."""
    
    STABLE = auto()       # No pending changes
    PENDING = auto()      # Changes detected
    RELOADING = auto()    # Reload in progress
    FAILED = auto()       # Reload failed


# ============================================================================
# PROTOCOLS
# ============================================================================

T = TypeVar("T")


class AgentProtocol(Protocol):
    """Protocol for agents in Warm Surface."""
    
    agent_id: str
    state: AgentState
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task."""
        ...
    
    async def terminate(self) -> None:
        """Terminate the agent."""
        ...


class ToolProtocol(Protocol):
    """Protocol for tools in Warm Surface."""
    
    tool_id: str
    name: str
    
    async def invoke(self, **kwargs: Any) -> Any:
        """Invoke the tool."""
        ...


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class WarmSurfaceConfig:
    """Configuration for Warm Surface."""
    
    latency_budget_ms: float = WARM_SURFACE_LATENCY_BUDGET_MS
    max_concurrent_ops: int = MAX_CONCURRENT_OPERATIONS
    enable_hot_reload: bool = True
    hot_reload_interval: float = HOT_RELOAD_CHECK_INTERVAL
    enable_audit_logging: bool = True


@dataclass
class WarmOperation:
    """A single Warm Surface operation."""
    
    operation_id: str
    operation_type: WarmOperationType
    agent_id: Optional[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type.name,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "parameters": self.parameters,
        }


@dataclass
class WarmResult:
    """Result from a Warm Surface operation."""
    
    operation_id: str
    success: bool
    output: Any
    latency_ms: float
    within_budget: bool
    audit_hash: str
    agent_id: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_id": self.operation_id,
            "success": self.success,
            "agent_id": self.agent_id,
            "latency_ms": self.latency_ms,
            "within_budget": self.within_budget,
            "audit_hash": self.audit_hash,
            "error": self.error,
        }


@dataclass
class AgentContext:
    """Context for agent execution."""
    
    agent_id: str
    task_id: str
    session_id: str
    parent_agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "parent_agent_id": self.parent_agent_id,
            "metadata": self.metadata,
        }


@dataclass
class WarmAgentRecord:
    """Record of an agent in Warm Surface."""
    
    agent_id: str
    agent_type: str
    state: AgentState
    spawned_at: datetime
    last_active: datetime
    execution_count: int = 0
    total_latency_ms: float = 0.0
    
    def average_latency_ms(self) -> float:
        """Compute average execution latency."""
        if self.execution_count == 0:
            return 0.0
        return self.total_latency_ms / self.execution_count


@dataclass
class ToolRecord:
    """Record of a registered tool."""
    
    tool_id: str
    name: str
    description: str
    registered_at: datetime
    invocation_count: int = 0
    total_latency_ms: float = 0.0
    
    def average_latency_ms(self) -> float:
        """Compute average invocation latency."""
        if self.invocation_count == 0:
            return 0.0
        return self.total_latency_ms / self.invocation_count


# ============================================================================
# WARM SURFACE IMPLEMENTATION
# ============================================================================


class WarmSurface:
    """
    The Warm Surface - Flexible Python Layer.
    
    Handles all operations requiring:
    - Flexibility and adaptability
    - Hot-reloadability
    - External integrations
    - Agent orchestration
    
    Usage:
        surface = WarmSurface()
        
        # Spawn an agent
        agent_id = await surface.spawn_agent("PAT", config={})
        
        # Execute a task
        result = await surface.execute_agent_task(agent_id, {"action": "plan"})
        
        # Register a tool
        await surface.register_tool("calculator", my_calculator)
        
        # Invoke a tool
        result = await surface.invoke_tool("calculator", x=5, y=3)
    """
    
    def __init__(self, config: Optional[WarmSurfaceConfig] = None):
        """Initialize Warm Surface."""
        self.config = config or WarmSurfaceConfig()
        
        # Agent registry
        self._agents: Dict[str, Tuple[WarmAgentRecord, Any]] = {}
        
        # Tool registry
        self._tools: Dict[str, Tuple[ToolRecord, Callable]] = {}
        
        # Event subscribers
        self._event_subscribers: Dict[str, List[Callable]] = {}
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_ops)
        
        # Hot-reload state
        self._hot_reload_state = HotReloadState.STABLE
        self._watched_modules: Set[str] = set()
        
        # Audit log
        self._audit_log: List[Dict[str, Any]] = []
        
        # Statistics
        self._operations: int = 0
        self._within_budget: int = 0
        
        logger.info("Warm Surface initialized")
    
    # ========================================================================
    # AGENT OPERATIONS
    # ========================================================================
    
    async def spawn_agent(
        self,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None,
        parent_agent_id: Optional[str] = None,
    ) -> WarmResult:
        """
        Spawn a new agent.
        
        Args:
            agent_type: Type of agent (PAT, SAT, etc.)
            config: Agent configuration
            parent_agent_id: Parent agent ID (for hierarchical agents)
            
        Returns:
            WarmResult with agent_id
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("SPAWN")
        
        async with self._semaphore:
            try:
                # Generate agent ID
                agent_id = hashlib.sha256(
                    f"{agent_type}:{time.time()}:{self._operations}".encode()
                ).hexdigest()[:16]
                
                # Create agent record
                now = datetime.now(timezone.utc)
                record = WarmAgentRecord(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    state=AgentState.IDLE,
                    spawned_at=now,
                    last_active=now,
                )
                
                # Create agent instance (placeholder)
                agent_instance = self._create_agent_instance(agent_type, config or {})
                
                self._agents[agent_id] = (record, agent_instance)
                
                latency = (time.perf_counter() - start) * 1000
                within_budget = latency <= self.config.latency_budget_ms
                
                self._record_operation(op_id, WarmOperationType.AGENT_SPAWN, agent_id, latency)
                
                logger.info(f"Spawned agent: {agent_type} (id={agent_id})")
                
                return WarmResult(
                    operation_id=op_id,
                    success=True,
                    output={"agent_id": agent_id, "agent_type": agent_type},
                    latency_ms=latency,
                    within_budget=within_budget,
                    audit_hash=self._hash(agent_id),
                    agent_id=agent_id,
                )
            except Exception as e:
                latency = (time.perf_counter() - start) * 1000
                return WarmResult(
                    operation_id=op_id,
                    success=False,
                    output=None,
                    latency_ms=latency,
                    within_budget=False,
                    audit_hash="",
                    error=str(e),
                )
    
    async def execute_agent_task(
        self,
        agent_id: str,
        task: Dict[str, Any],
        context: Optional[AgentContext] = None,
    ) -> WarmResult:
        """
        Execute a task with an agent.
        
        Args:
            agent_id: Agent ID
            task: Task to execute
            context: Execution context
            
        Returns:
            WarmResult with task result
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("EXECUTE")
        
        if agent_id not in self._agents:
            return WarmResult(
                operation_id=op_id,
                success=False,
                output=None,
                latency_ms=0,
                within_budget=True,
                audit_hash="",
                error=f"Agent not found: {agent_id}",
            )
        
        async with self._semaphore:
            record: Optional[Any] = None  # P2 FIX: Pre-initialize to avoid UnboundLocalError
            try:
                record, agent_instance = self._agents[agent_id]
                
                # Update state
                record.state = AgentState.EXECUTING
                record.last_active = datetime.now(timezone.utc)
                
                # Execute task
                if asyncio.iscoroutinefunction(getattr(agent_instance, "execute", None)):
                    result = await agent_instance.execute(task)
                elif callable(getattr(agent_instance, "execute", None)):
                    result = agent_instance.execute(task)
                else:
                    # Default execution for placeholder agents
                    result = {"status": "executed", "task": task}
                
                # Update statistics
                record.state = AgentState.IDLE
                record.execution_count += 1
                exec_latency = (time.perf_counter() - start) * 1000
                record.total_latency_ms += exec_latency
                
                within_budget = exec_latency <= self.config.latency_budget_ms
                
                self._record_operation(op_id, WarmOperationType.AGENT_EXECUTE, agent_id, exec_latency)
                
                return WarmResult(
                    operation_id=op_id,
                    success=True,
                    output=result,
                    latency_ms=exec_latency,
                    within_budget=within_budget,
                    audit_hash=self._hash(str(result)),
                    agent_id=agent_id,
                )
            except Exception as e:
                # P2 FIX: Only reset state if record was successfully retrieved
                if record is not None:
                    record.state = AgentState.IDLE
                latency = (time.perf_counter() - start) * 1000
                return WarmResult(
                    operation_id=op_id,
                    success=False,
                    output=None,
                    latency_ms=latency,
                    within_budget=False,
                    audit_hash="",
                    agent_id=agent_id,
                    error=str(e),
                )
    
    async def terminate_agent(self, agent_id: str) -> WarmResult:
        """
        Terminate an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            WarmResult with termination status
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("TERMINATE")
        
        if agent_id not in self._agents:
            return WarmResult(
                operation_id=op_id,
                success=False,
                output=None,
                latency_ms=0,
                within_budget=True,
                audit_hash="",
                error=f"Agent not found: {agent_id}",
            )
        
        try:
            record, agent_instance = self._agents[agent_id]
            
            # Terminate agent
            if asyncio.iscoroutinefunction(getattr(agent_instance, "terminate", None)):
                await agent_instance.terminate()
            elif callable(getattr(agent_instance, "terminate", None)):
                agent_instance.terminate()
            
            record.state = AgentState.TERMINATED
            
            latency = (time.perf_counter() - start) * 1000
            within_budget = latency <= self.config.latency_budget_ms
            
            self._record_operation(op_id, WarmOperationType.AGENT_TERMINATE, agent_id, latency)
            
            logger.info(f"Terminated agent: {agent_id}")
            
            return WarmResult(
                operation_id=op_id,
                success=True,
                output={"agent_id": agent_id, "state": "TERMINATED"},
                latency_ms=latency,
                within_budget=within_budget,
                audit_hash=self._hash(agent_id),
                agent_id=agent_id,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return WarmResult(
                operation_id=op_id,
                success=False,
                output=None,
                latency_ms=latency,
                within_budget=False,
                audit_hash="",
                agent_id=agent_id,
                error=str(e),
            )
    
    # ========================================================================
    # TOOL OPERATIONS
    # ========================================================================
    
    async def register_tool(
        self,
        name: str,
        handler: Callable,
        description: str = "",
    ) -> WarmResult:
        """
        Register a tool.
        
        Args:
            name: Tool name
            handler: Tool handler function
            description: Tool description
            
        Returns:
            WarmResult with registration status
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("REGISTER_TOOL")
        
        try:
            # Generate tool ID
            tool_id = hashlib.sha256(f"tool:{name}".encode()).hexdigest()[:16]
            
            # Create tool record
            record = ToolRecord(
                tool_id=tool_id,
                name=name,
                description=description,
                registered_at=datetime.now(timezone.utc),
            )
            
            self._tools[name] = (record, handler)
            
            latency = (time.perf_counter() - start) * 1000
            within_budget = latency <= self.config.latency_budget_ms
            
            self._record_operation(op_id, WarmOperationType.TOOL_REGISTER, None, latency)
            
            logger.info(f"Registered tool: {name} (id={tool_id})")
            
            return WarmResult(
                operation_id=op_id,
                success=True,
                output={"tool_id": tool_id, "name": name},
                latency_ms=latency,
                within_budget=within_budget,
                audit_hash=self._hash(tool_id),
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return WarmResult(
                operation_id=op_id,
                success=False,
                output=None,
                latency_ms=latency,
                within_budget=False,
                audit_hash="",
                error=str(e),
            )
    
    async def invoke_tool(self, name: str, **kwargs: Any) -> WarmResult:
        """
        Invoke a registered tool.
        
        Args:
            name: Tool name
            **kwargs: Tool arguments
            
        Returns:
            WarmResult with tool output
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("INVOKE_TOOL")
        
        if name not in self._tools:
            return WarmResult(
                operation_id=op_id,
                success=False,
                output=None,
                latency_ms=0,
                within_budget=True,
                audit_hash="",
                error=f"Tool not found: {name}",
            )
        
        async with self._semaphore:
            try:
                record, handler = self._tools[name]
                
                # Invoke handler
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**kwargs)
                else:
                    result = handler(**kwargs)
                
                # Update statistics
                record.invocation_count += 1
                invoke_latency = (time.perf_counter() - start) * 1000
                record.total_latency_ms += invoke_latency
                
                within_budget = invoke_latency <= self.config.latency_budget_ms
                
                self._record_operation(op_id, WarmOperationType.TOOL_INVOKE, None, invoke_latency)
                
                return WarmResult(
                    operation_id=op_id,
                    success=True,
                    output=result,
                    latency_ms=invoke_latency,
                    within_budget=within_budget,
                    audit_hash=self._hash(str(result)),
                )
            except Exception as e:
                latency = (time.perf_counter() - start) * 1000
                return WarmResult(
                    operation_id=op_id,
                    success=False,
                    output=None,
                    latency_ms=latency,
                    within_budget=False,
                    audit_hash="",
                    error=str(e),
                )
    
    # ========================================================================
    # EVENT SYSTEM
    # ========================================================================
    
    def subscribe(self, event_type: str, handler: Callable) -> str:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Event type to subscribe to
            handler: Event handler function
            
        Returns:
            Subscription ID
        """
        if event_type not in self._event_subscribers:
            self._event_subscribers[event_type] = []
        
        sub_id = hashlib.sha256(
            f"sub:{event_type}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        self._event_subscribers[event_type].append(handler)
        
        logger.debug(f"Subscribed to event: {event_type} (sub_id={sub_id})")
        
        return sub_id
    
    async def emit(self, event_type: str, data: Dict[str, Any]) -> WarmResult:
        """
        Emit an event.
        
        Args:
            event_type: Event type
            data: Event data
            
        Returns:
            WarmResult with emission status
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("EMIT")
        
        try:
            handlers = self._event_subscribers.get(event_type, [])
            
            for handler in handlers:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            
            latency = (time.perf_counter() - start) * 1000
            within_budget = latency <= self.config.latency_budget_ms
            
            self._record_operation(op_id, WarmOperationType.EVENT_EMIT, None, latency)
            
            return WarmResult(
                operation_id=op_id,
                success=True,
                output={"event_type": event_type, "handlers_called": len(handlers)},
                latency_ms=latency,
                within_budget=within_budget,
                audit_hash=self._hash(event_type),
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return WarmResult(
                operation_id=op_id,
                success=False,
                output=None,
                latency_ms=latency,
                within_budget=False,
                audit_hash="",
                error=str(e),
            )
    
    # ========================================================================
    # HOT-RELOAD
    # ========================================================================
    
    def watch_module(self, module_name: str) -> None:
        """
        Watch a module for hot-reload.
        
        Args:
            module_name: Module name to watch
        """
        self._watched_modules.add(module_name)
        logger.info(f"Watching module for hot-reload: {module_name}")
    
    async def hot_reload(self) -> WarmResult:
        """
        Perform hot-reload of watched modules.
        
        Returns:
            WarmResult with reload status
        """
        start = time.perf_counter()
        op_id = self._generate_operation_id("HOT_RELOAD")
        
        if not self.config.enable_hot_reload:
            return WarmResult(
                operation_id=op_id,
                success=False,
                output=None,
                latency_ms=0,
                within_budget=True,
                audit_hash="",
                error="Hot-reload is disabled",
            )
        
        self._hot_reload_state = HotReloadState.RELOADING
        reloaded = []
        errors = []
        
        try:
            for module_name in self._watched_modules:
                try:
                    module = importlib.import_module(module_name)
                    importlib.reload(module)
                    reloaded.append(module_name)
                except Exception as e:
                    errors.append(f"{module_name}: {e}")
            
            if errors:
                self._hot_reload_state = HotReloadState.FAILED
            else:
                self._hot_reload_state = HotReloadState.STABLE
            
            latency = (time.perf_counter() - start) * 1000
            
            logger.info(f"Hot-reload complete: reloaded={len(reloaded)}, errors={len(errors)}")
            
            return WarmResult(
                operation_id=op_id,
                success=len(errors) == 0,
                output={"reloaded": reloaded, "errors": errors},
                latency_ms=latency,
                within_budget=True,
                audit_hash=self._hash(str(reloaded)),
            )
        except Exception as e:
            self._hot_reload_state = HotReloadState.FAILED
            latency = (time.perf_counter() - start) * 1000
            return WarmResult(
                operation_id=op_id,
                success=False,
                output=None,
                latency_ms=latency,
                within_budget=False,
                audit_hash="",
                error=str(e),
            )
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _create_agent_instance(
        self,
        agent_type: str,
        config: Dict[str, Any],
    ) -> Any:
        """Create an agent instance (placeholder)."""
        # This would be replaced with actual agent instantiation
        class PlaceholderAgent:
            def __init__(self, agent_type: str, config: Dict[str, Any]):
                self.agent_type = agent_type
                self.config = config
            
            def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {"agent_type": self.agent_type, "task": task, "status": "executed"}
            
            def terminate(self) -> None:
                pass
        
        return PlaceholderAgent(agent_type, config)
    
    def _generate_operation_id(self, prefix: str) -> str:
        """Generate unique operation ID."""
        self._operations += 1
        timestamp = int(time.time() * 1000000)
        return f"{prefix}-{timestamp}-{self._operations:06d}"
    
    def _hash(self, data: str) -> str:
        """Compute hash of data."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _record_operation(
        self,
        op_id: str,
        op_type: WarmOperationType,
        agent_id: Optional[str],
        latency_ms: float,
    ) -> None:
        """Record operation in audit log."""
        if latency_ms <= self.config.latency_budget_ms:
            self._within_budget += 1
        
        if self.config.enable_audit_logging:
            self._audit_log.append({
                "operation_id": op_id,
                "operation_type": op_type.name,
                "agent_id": agent_id,
                "latency_ms": latency_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            # Keep last 10000 entries
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-10000:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Warm Surface statistics."""
        return {
            "operations": self._operations,
            "within_budget": self._within_budget,
            "budget_compliance_rate": self._within_budget / max(self._operations, 1),
            "active_agents": len([a for a, _ in self._agents.values() if a.state != AgentState.TERMINATED]),
            "total_agents": len(self._agents),
            "registered_tools": len(self._tools),
            "hot_reload_state": self._hot_reload_state.name,
            "watched_modules": list(self._watched_modules),
        }
    
    def get_agents(self) -> List[Dict[str, Any]]:
        """Get list of agents."""
        return [
            {
                "agent_id": record.agent_id,
                "agent_type": record.agent_type,
                "state": record.state.name,
                "spawned_at": record.spawned_at.isoformat(),
                "last_active": record.last_active.isoformat(),
                "execution_count": record.execution_count,
                "average_latency_ms": record.average_latency_ms(),
            }
            for record, _ in self._agents.values()
        ]
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of registered tools."""
        return [
            {
                "tool_id": record.tool_id,
                "name": record.name,
                "description": record.description,
                "registered_at": record.registered_at.isoformat(),
                "invocation_count": record.invocation_count,
                "average_latency_ms": record.average_latency_ms(),
            }
            for record, _ in self._tools.values()
        ]


# ============================================================================
# SELF-TEST
# ============================================================================


async def _self_test() -> None:
    """Run Warm Surface self-tests."""
    print("=" * 70)
    print("BIZRA AEON OMEGA - Warm Surface Self-Test")
    print("=" * 70)
    
    surface = WarmSurface()
    
    # Test 1: Spawn agent
    print("\n[Test 1] Spawn Agent")
    result = await surface.spawn_agent("PAT", config={"mode": "test"})
    assert result.success, f"Spawn failed: {result.error}"
    agent_id = result.output["agent_id"]
    print(f"  ✓ Spawned PAT agent (id={agent_id})")
    print(f"  ✓ Latency: {result.latency_ms:.3f}ms")
    
    # Test 2: Execute agent task
    print("\n[Test 2] Execute Agent Task")
    result = await surface.execute_agent_task(agent_id, {"action": "plan", "goal": "test"})
    assert result.success, f"Execute failed: {result.error}"
    print(f"  ✓ Task executed: {result.output}")
    print(f"  ✓ Latency: {result.latency_ms:.3f}ms")
    
    # Test 3: Register tool
    print("\n[Test 3] Register Tool")
    async def calculator(x: int, y: int) -> int:
        return x + y
    result = await surface.register_tool("calculator", calculator, "Adds two numbers")
    assert result.success, f"Register failed: {result.error}"
    print(f"  ✓ Registered tool: calculator")
    
    # Test 4: Invoke tool
    print("\n[Test 4] Invoke Tool")
    result = await surface.invoke_tool("calculator", x=5, y=3)
    assert result.success and result.output == 8
    print(f"  ✓ calculator(5, 3) = {result.output}")
    print(f"  ✓ Latency: {result.latency_ms:.3f}ms")
    
    # Test 5: Event system
    print("\n[Test 5] Event System")
    events_received = []
    def handler(data: Dict[str, Any]) -> None:
        events_received.append(data)
    surface.subscribe("test_event", handler)
    result = await surface.emit("test_event", {"message": "hello"})
    assert result.success and len(events_received) == 1
    print(f"  ✓ Event emitted and received: {events_received[0]}")
    
    # Test 6: Terminate agent
    print("\n[Test 6] Terminate Agent")
    result = await surface.terminate_agent(agent_id)
    assert result.success
    print(f"  ✓ Agent terminated: {agent_id}")
    
    # Test 7: Statistics
    print("\n[Test 7] Statistics")
    stats = surface.get_statistics()
    print(f"  ✓ Operations: {stats['operations']}")
    print(f"  ✓ Budget compliance: {stats['budget_compliance_rate']*100:.1f}%")
    print(f"  ✓ Active agents: {stats['active_agents']}")
    print(f"  ✓ Registered tools: {stats['registered_tools']}")
    
    print("\n" + "=" * 70)
    print("✅ ALL WARM SURFACE SELF-TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(_self_test())
