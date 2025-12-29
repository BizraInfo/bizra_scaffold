"""
BIZRA AEON OMEGA - Isolation Package
=====================================
Firecracker MicroVM and Sandbox Isolation

This package provides secure isolation for code execution through
Firecracker MicroVMs and lightweight sandboxing.

Components:
    - FirecrackerOrchestrator: MicroVM lifecycle management
    - SandboxEngine: Lightweight isolation for code execution
    - ResourceLimiter: CPU/Memory/IO resource control
    - SnapshotManager: VM snapshot and restore

Author: BIZRA Genesis Team (Peak Masterpiece v5)
Version: 1.0.0
License: MIT
"""

from .firecracker_orchestrator import (
    FirecrackerOrchestrator,
    FirecrackerConfig,
    MicroVM,
    VMState,
    VMSpec,
    BootResult,
    ExecutionResult,
    ResourceLimits,
)

from .sandbox_engine import (
    SandboxEngine,
    SandboxConfig,
    Sandbox,
    SandboxState,
    SandboxResult,
    IsolationLevel,
    CodeExecution,
    ExecutionReceipt,
)

__all__ = [
    # Firecracker
    "FirecrackerOrchestrator",
    "FirecrackerConfig",
    "MicroVM",
    "VMState",
    "VMSpec",
    "BootResult",
    "ExecutionResult",
    "ResourceLimits",
    # Sandbox
    "SandboxEngine",
    "SandboxConfig",
    "Sandbox",
    "SandboxState",
    "SandboxResult",
    "IsolationLevel",
    "CodeExecution",
    "ExecutionReceipt",
]
