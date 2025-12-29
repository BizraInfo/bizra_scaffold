"""
BIZRA AEON OMEGA - Bicameral Cognitive Architecture
====================================================
Cold Core (Rust) + Warm Surface (Python) Separation

Peak Masterpiece v5: The Bicameral Engine implements a dual-hemisphere
cognitive architecture inspired by Julian Jaynes' bicameral mind theory,
adapted for sovereign AI systems.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      WARM SURFACE (Python)                       │
    │  • Agents (PAT/SAT), UI, Integrations                           │
    │  • Flexible, high-level orchestration                           │
    │  • Latency: 10-100ms                                            │
    ├─────────────────────────────────────────────────────────────────┤
    │                 MEMBRANE (Typed Message Passing)                 │
    │  • Protocol Buffers / JSON-RPC                                  │
    │  • Serialization boundary                                       │
    │  • Audit logging                                                │
    ├─────────────────────────────────────────────────────────────────┤
    │                       COLD CORE (Rust)                          │
    │  • Cryptography, Invariants, Formal Verification               │
    │  • Deterministic, formally verified                            │
    │  • Latency: <1ms                                                │
    └─────────────────────────────────────────────────────────────────┘

Key Principles:
    1. Cold Core is immutable after crystallization
    2. All state mutations flow through the membrane
    3. Warm Surface can be hot-reloaded without affecting Cold Core
    4. Deterministic behavior guaranteed by Cold Core

Author: BIZRA Genesis Team (Peak Masterpiece v5)
Version: 1.0.0
License: MIT
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    # Cold Core
    "ColdCore",
    "ColdCoreConfig",
    "ColdOperation",
    "ColdOperationType",
    "ColdResult",
    "CrystallizedFunction",
    "CrystallizationState",
    "COLD_CORE_AVAILABLE",
    # Warm Surface
    "WarmSurface",
    "WarmSurfaceConfig",
    "WarmOperation",
    "WarmOperationType",
    "WarmResult",
    "AgentContext",
    "AgentState",
    "HotReloadState",
    # Membrane
    "Membrane",
    "MembraneConfig",
    "Message",
    "MessageType",
    "MessagePriority",
    "CrossingDirection",
    "CrossingReceipt",
    "MembraneState",
    "MembraneException",
    "MembraneClosedException",
    "MessageTooLargeException",
    "QueueFullException",
    "QueueThrottledException",
    # Bicameral Engine
    "BicameralEngine",
    "BicameralConfig",
    "BicameralOperation",
    "BicameralResult",
    "Hemisphere",
    "CognitiveState",
    "OperationRouting",
    "HealthStatus",
]


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name in (
        "ColdCore", "ColdCoreConfig", "ColdOperation", "ColdOperationType",
        "ColdResult", "CrystallizedFunction", "CrystallizationState", "COLD_CORE_AVAILABLE",
    ):
        from .cold_core import (
            ColdCore, ColdCoreConfig, ColdOperation, ColdOperationType,
            ColdResult, CrystallizedFunction, CrystallizationState, COLD_CORE_AVAILABLE,
        )
        return locals()[name]
    
    if name in (
        "WarmSurface", "WarmSurfaceConfig", "WarmOperation", "WarmOperationType",
        "WarmResult", "AgentContext", "AgentState", "HotReloadState",
    ):
        from .warm_surface import (
            WarmSurface, WarmSurfaceConfig, WarmOperation, WarmOperationType,
            WarmResult, AgentContext, AgentState, HotReloadState,
        )
        return locals()[name]
    
    if name in (
        "Membrane", "MembraneConfig", "Message", "MessageType", "MessagePriority",
        "CrossingDirection", "CrossingReceipt", "MembraneState", "MembraneException",
        "MembraneClosedException", "MessageTooLargeException", "QueueFullException",
        "QueueThrottledException",
    ):
        from .membrane import (
            Membrane, MembraneConfig, Message, MessageType, MessagePriority,
            CrossingDirection, CrossingReceipt, MembraneState, MembraneException,
            MembraneClosedException, MessageTooLargeException, QueueFullException,
            QueueThrottledException,
        )
        return locals()[name]
    
    if name in (
        "BicameralEngine", "BicameralConfig", "BicameralOperation", "BicameralResult",
        "Hemisphere", "CognitiveState", "OperationRouting", "HealthStatus",
    ):
        from .bicameral_engine import (
            BicameralEngine, BicameralConfig, BicameralOperation, BicameralResult,
            Hemisphere, CognitiveState, OperationRouting, HealthStatus,
        )
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
