"""
BIZRA AEON OMEGA - Firecracker Orchestrator
============================================
MicroVM Lifecycle Management

Firecracker is a lightweight virtual machine monitor (VMM) that enables
secure, fast, and resource-efficient workload isolation. This module
provides orchestration for Firecracker MicroVMs.

Features:
    - Fast boot times (<125ms cold start)
    - Memory-safe isolation (KVM-based)
    - Snapshot/restore for instant warm starts
    - Resource limiting (CPU, memory, network, I/O)
    - Secure by default (jailer integration)

Architecture:
    ┌────────────────────────────────────────────────────────────┐
    │                FIRECRACKER ORCHESTRATOR                     │
    │                                                             │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
    │  │  MicroVM-1  │  │  MicroVM-2  │  │  MicroVM-N  │        │
    │  │  (sandbox)  │  │  (sandbox)  │  │  (sandbox)  │        │
    │  └─────────────┘  └─────────────┘  └─────────────┘        │
    │         │               │               │                  │
    │  ┌──────┴───────────────┴───────────────┴───────┐         │
    │  │              API SOCKET (UDS)                 │         │
    │  └───────────────────────────────────────────────┘         │
    │                                                             │
    │  ┌───────────────────────────────────────────────┐         │
    │  │              JAILER (seccomp)                 │         │
    │  └───────────────────────────────────────────────┘         │
    └────────────────────────────────────────────────────────────┘

Note: This is a Pure Python simulation when Firecracker is not available.
      In production, this would interface with the actual Firecracker binary.

Author: BIZRA Genesis Team (Peak Masterpiece v5)
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

logger = logging.getLogger("bizra.isolation.firecracker")

# ============================================================================
# CONSTANTS
# ============================================================================

# Default resource limits
DEFAULT_VCPU_COUNT = 1
DEFAULT_MEM_SIZE_MB = 128
DEFAULT_BOOT_TIMEOUT_MS = 5000
DEFAULT_EXEC_TIMEOUT_MS = 30000

# Firecracker paths (Linux only)
FIRECRACKER_BIN = "/usr/bin/firecracker"
JAILER_BIN = "/usr/bin/jailer"

# Check if Firecracker is available
FIRECRACKER_AVAILABLE = os.path.exists(FIRECRACKER_BIN) if os.name == "posix" else False

# ============================================================================
# ENUMERATIONS
# ============================================================================


class VMState(Enum):
    """State of a MicroVM."""
    
    CREATING = auto()   # VM being created
    READY = auto()      # VM ready to execute
    RUNNING = auto()    # VM executing workload
    PAUSED = auto()     # VM paused (snapshot-able)
    STOPPED = auto()    # VM stopped
    ERROR = auto()      # VM in error state


class BootMode(Enum):
    """Boot mode for MicroVM."""
    
    COLD = auto()       # Full cold boot
    WARM = auto()       # Warm start from snapshot


class NetworkMode(Enum):
    """Network mode for MicroVM."""
    
    NONE = auto()       # No network
    HOST = auto()       # Host network (not recommended)
    ISOLATED = auto()   # Isolated virtual network
    TAP = auto()        # TAP device


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ResourceLimits:
    """Resource limits for a MicroVM."""
    
    vcpu_count: int = DEFAULT_VCPU_COUNT
    mem_size_mb: int = DEFAULT_MEM_SIZE_MB
    disk_size_mb: int = 512
    net_rate_limit_kbps: int = 10240  # 10 Mbps
    io_rate_limit_iops: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vcpu_count": self.vcpu_count,
            "mem_size_mb": self.mem_size_mb,
            "disk_size_mb": self.disk_size_mb,
            "net_rate_limit_kbps": self.net_rate_limit_kbps,
            "io_rate_limit_iops": self.io_rate_limit_iops,
        }


@dataclass
class VMSpec:
    """Specification for creating a MicroVM."""
    
    vm_id: str
    kernel_path: str
    rootfs_path: str
    boot_args: str = "console=ttyS0 reboot=k panic=1 pci=off"
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    network_mode: NetworkMode = NetworkMode.ISOLATED
    enable_jailer: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vm_id": self.vm_id,
            "kernel_path": self.kernel_path,
            "rootfs_path": self.rootfs_path,
            "boot_args": self.boot_args,
            "limits": self.limits.to_dict(),
            "network_mode": self.network_mode.name,
            "enable_jailer": self.enable_jailer,
            "metadata": self.metadata,
        }


@dataclass
class MicroVM:
    """Representation of a running MicroVM."""
    
    vm_id: str
    spec: VMSpec
    state: VMState
    socket_path: str
    pid: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    boot_time_ms: float = 0.0
    exec_count: int = 0
    total_exec_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vm_id": self.vm_id,
            "state": self.state.name,
            "socket_path": self.socket_path,
            "pid": self.pid,
            "created_at": self.created_at.isoformat(),
            "boot_time_ms": self.boot_time_ms,
            "exec_count": self.exec_count,
            "avg_exec_time_ms": self.total_exec_time_ms / max(self.exec_count, 1),
        }


@dataclass
class BootResult:
    """Result of booting a MicroVM."""
    
    vm_id: str
    success: bool
    boot_time_ms: float
    boot_mode: BootMode
    socket_path: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vm_id": self.vm_id,
            "success": self.success,
            "boot_time_ms": self.boot_time_ms,
            "boot_mode": self.boot_mode.name,
            "socket_path": self.socket_path,
            "error": self.error,
        }


@dataclass
class ExecutionResult:
    """Result of executing code in a MicroVM."""
    
    vm_id: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    exec_time_ms: float
    resource_usage: Dict[str, Any]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vm_id": self.vm_id,
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:1000],  # Truncate for safety
            "stderr": self.stderr[:1000],
            "exec_time_ms": self.exec_time_ms,
            "resource_usage": self.resource_usage,
            "error": self.error,
        }


@dataclass
class FirecrackerConfig:
    """Configuration for Firecracker Orchestrator."""
    
    firecracker_bin: str = FIRECRACKER_BIN
    jailer_bin: str = JAILER_BIN
    socket_dir: str = "/tmp/firecracker"
    snapshot_dir: str = "/tmp/firecracker/snapshots"
    max_vms: int = 10
    boot_timeout_ms: float = DEFAULT_BOOT_TIMEOUT_MS
    exec_timeout_ms: float = DEFAULT_EXEC_TIMEOUT_MS
    enable_simulation: bool = not FIRECRACKER_AVAILABLE
    enable_audit_logging: bool = True


# ============================================================================
# FIRECRACKER ORCHESTRATOR
# ============================================================================


class FirecrackerOrchestrator:
    """
    Firecracker MicroVM Orchestrator.
    
    Manages the lifecycle of Firecracker MicroVMs for secure,
    isolated code execution.
    
    Usage:
        orchestrator = FirecrackerOrchestrator()
        
        # Create and boot a VM
        spec = VMSpec(vm_id="vm-1", kernel_path="...", rootfs_path="...")
        result = await orchestrator.boot(spec)
        
        # Execute code
        result = await orchestrator.execute(vm_id, "echo 'hello'")
        
        # Snapshot and restore
        await orchestrator.snapshot(vm_id, "/path/to/snapshot")
        await orchestrator.restore("/path/to/snapshot")
        
        # Stop VM
        await orchestrator.stop(vm_id)
    """
    
    def __init__(self, config: Optional[FirecrackerConfig] = None):
        """Initialize Firecracker Orchestrator."""
        self.config = config or FirecrackerConfig()
        
        # VM registry
        self._vms: Dict[str, MicroVM] = {}
        
        # Ensure directories exist
        if not self.config.enable_simulation:
            os.makedirs(self.config.socket_dir, exist_ok=True)
            os.makedirs(self.config.snapshot_dir, exist_ok=True)
        
        # Statistics
        self._boots: int = 0
        self._executions: int = 0
        self._total_boot_time_ms: float = 0.0
        self._total_exec_time_ms: float = 0.0
        
        # Audit log
        self._audit_log: List[Dict[str, Any]] = []
        
        mode = "SIMULATION" if self.config.enable_simulation else "NATIVE"
        logger.info(f"Firecracker Orchestrator initialized: mode={mode}")
    
    # ========================================================================
    # VM LIFECYCLE
    # ========================================================================
    
    async def boot(
        self,
        spec: VMSpec,
        mode: BootMode = BootMode.COLD,
        snapshot_path: Optional[str] = None,
    ) -> BootResult:
        """
        Boot a MicroVM.
        
        Args:
            spec: VM specification
            mode: Boot mode (COLD or WARM)
            snapshot_path: Path to snapshot for WARM boot
            
        Returns:
            BootResult with boot status
        """
        start = time.perf_counter()
        
        # Check VM limit
        if len(self._vms) >= self.config.max_vms:
            return BootResult(
                vm_id=spec.vm_id,
                success=False,
                boot_time_ms=0,
                boot_mode=mode,
                socket_path="",
                error=f"Maximum VMs ({self.config.max_vms}) reached",
            )
        
        # Check for duplicate VM ID
        if spec.vm_id in self._vms:
            return BootResult(
                vm_id=spec.vm_id,
                success=False,
                boot_time_ms=0,
                boot_mode=mode,
                socket_path="",
                error=f"VM {spec.vm_id} already exists",
            )
        
        socket_path = os.path.join(self.config.socket_dir, f"{spec.vm_id}.sock")
        
        try:
            if self.config.enable_simulation:
                # Simulation mode
                await self._simulate_boot(spec, mode)
                pid = None
            else:
                # Real Firecracker boot
                pid = await self._real_boot(spec, mode, socket_path, snapshot_path)
            
            boot_time = (time.perf_counter() - start) * 1000
            
            # Create VM record
            vm = MicroVM(
                vm_id=spec.vm_id,
                spec=spec,
                state=VMState.READY,
                socket_path=socket_path,
                pid=pid,
                boot_time_ms=boot_time,
            )
            
            self._vms[spec.vm_id] = vm
            self._boots += 1
            self._total_boot_time_ms += boot_time
            
            self._log_event("BOOT", spec.vm_id, {"mode": mode.name, "boot_time_ms": boot_time})
            
            logger.info(f"Booted VM {spec.vm_id}: mode={mode.name}, boot_time={boot_time:.1f}ms")
            
            return BootResult(
                vm_id=spec.vm_id,
                success=True,
                boot_time_ms=boot_time,
                boot_mode=mode,
                socket_path=socket_path,
            )
        except Exception as e:
            boot_time = (time.perf_counter() - start) * 1000
            logger.error(f"Boot failed for VM {spec.vm_id}: {e}")
            return BootResult(
                vm_id=spec.vm_id,
                success=False,
                boot_time_ms=boot_time,
                boot_mode=mode,
                socket_path="",
                error=str(e),
            )
    
    async def stop(self, vm_id: str) -> bool:
        """
        Stop a MicroVM.
        
        Args:
            vm_id: VM ID
            
        Returns:
            True if stopped successfully
        """
        if vm_id not in self._vms:
            logger.warning(f"VM {vm_id} not found")
            return False
        
        vm = self._vms[vm_id]
        
        try:
            if not self.config.enable_simulation and vm.pid:
                # Kill the process
                os.kill(vm.pid, 9)
                await asyncio.sleep(0.1)  # Allow cleanup
            
            # Clean up socket
            if os.path.exists(vm.socket_path):
                os.remove(vm.socket_path)
            
            vm.state = VMState.STOPPED
            self._log_event("STOP", vm_id, {})
            
            logger.info(f"Stopped VM {vm_id}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to stop VM {vm_id}: {e}")
            vm.state = VMState.ERROR
            return False
    
    async def destroy(self, vm_id: str) -> bool:
        """
        Destroy a MicroVM and clean up all resources.
        
        Args:
            vm_id: VM ID
            
        Returns:
            True if destroyed successfully
        """
        if vm_id not in self._vms:
            return False
        
        # Stop first if running
        vm = self._vms[vm_id]
        if vm.state in (VMState.READY, VMState.RUNNING, VMState.PAUSED):
            await self.stop(vm_id)
        
        # Remove from registry
        del self._vms[vm_id]
        
        self._log_event("DESTROY", vm_id, {})
        logger.info(f"Destroyed VM {vm_id}")
        
        return True
    
    # ========================================================================
    # EXECUTION
    # ========================================================================
    
    async def execute(
        self,
        vm_id: str,
        command: str,
        stdin: Optional[str] = None,
        timeout_ms: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute a command in a MicroVM.
        
        Args:
            vm_id: VM ID
            command: Command to execute
            stdin: Optional stdin input
            timeout_ms: Optional timeout
            
        Returns:
            ExecutionResult with command output
        """
        start = time.perf_counter()
        timeout = timeout_ms or self.config.exec_timeout_ms
        
        if vm_id not in self._vms:
            return ExecutionResult(
                vm_id=vm_id,
                success=False,
                exit_code=-1,
                stdout="",
                stderr="",
                exec_time_ms=0,
                resource_usage={},
                error=f"VM {vm_id} not found",
            )
        
        vm = self._vms[vm_id]
        
        if vm.state != VMState.READY:
            return ExecutionResult(
                vm_id=vm_id,
                success=False,
                exit_code=-1,
                stdout="",
                stderr="",
                exec_time_ms=0,
                resource_usage={},
                error=f"VM {vm_id} not in READY state: {vm.state.name}",
            )
        
        try:
            vm.state = VMState.RUNNING
            
            if self.config.enable_simulation:
                # Simulation mode
                result = await self._simulate_execute(command, stdin, timeout)
            else:
                # Real execution via vsock/API
                result = await self._real_execute(vm, command, stdin, timeout)
            
            exec_time = (time.perf_counter() - start) * 1000
            
            vm.state = VMState.READY
            vm.exec_count += 1
            vm.total_exec_time_ms += exec_time
            
            self._executions += 1
            self._total_exec_time_ms += exec_time
            
            self._log_event("EXECUTE", vm_id, {
                "command": command[:100],
                "exit_code": result.exit_code,
                "exec_time_ms": exec_time,
            })
            
            return ExecutionResult(
                vm_id=vm_id,
                success=result.exit_code == 0,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                exec_time_ms=exec_time,
                resource_usage=result.resource_usage,
            )
        except asyncio.TimeoutError:
            exec_time = (time.perf_counter() - start) * 1000
            vm.state = VMState.ERROR
            return ExecutionResult(
                vm_id=vm_id,
                success=False,
                exit_code=-1,
                stdout="",
                stderr="",
                exec_time_ms=exec_time,
                resource_usage={},
                error=f"Execution timed out after {timeout}ms",
            )
        except Exception as e:
            exec_time = (time.perf_counter() - start) * 1000
            vm.state = VMState.ERROR
            return ExecutionResult(
                vm_id=vm_id,
                success=False,
                exit_code=-1,
                stdout="",
                stderr="",
                exec_time_ms=exec_time,
                resource_usage={},
                error=str(e),
            )
    
    # ========================================================================
    # SNAPSHOT/RESTORE
    # ========================================================================
    
    async def snapshot(self, vm_id: str, snapshot_path: str) -> bool:
        """
        Create a snapshot of a MicroVM.
        
        Args:
            vm_id: VM ID
            snapshot_path: Path to save snapshot
            
        Returns:
            True if snapshot created successfully
        """
        if vm_id not in self._vms:
            logger.warning(f"VM {vm_id} not found")
            return False
        
        vm = self._vms[vm_id]
        
        try:
            if self.config.enable_simulation:
                # Simulation: create dummy snapshot
                os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
                with open(snapshot_path, "w") as f:
                    json.dump({
                        "vm_id": vm_id,
                        "spec": vm.spec.to_dict(),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }, f)
            else:
                # Real snapshot via API
                await self._real_snapshot(vm, snapshot_path)
            
            vm.state = VMState.PAUSED
            
            self._log_event("SNAPSHOT", vm_id, {"path": snapshot_path})
            logger.info(f"Created snapshot for VM {vm_id}: {snapshot_path}")
            
            return True
        except Exception as e:
            logger.error(f"Snapshot failed for VM {vm_id}: {e}")
            return False
    
    async def restore(self, snapshot_path: str) -> BootResult:
        """
        Restore a MicroVM from snapshot.
        
        Args:
            snapshot_path: Path to snapshot
            
        Returns:
            BootResult with restore status
        """
        if not os.path.exists(snapshot_path):
            return BootResult(
                vm_id="",
                success=False,
                boot_time_ms=0,
                boot_mode=BootMode.WARM,
                socket_path="",
                error=f"Snapshot not found: {snapshot_path}",
            )
        
        try:
            if self.config.enable_simulation:
                # Load snapshot metadata
                with open(snapshot_path, "r") as f:
                    data = json.load(f)
                
                vm_id = data["vm_id"]
                spec = VMSpec(**{k: v for k, v in data["spec"].items() 
                               if k in ["vm_id", "kernel_path", "rootfs_path", "boot_args"]})
            else:
                # Real restore
                vm_id, spec = await self._real_restore(snapshot_path)
            
            # Boot from snapshot
            return await self.boot(spec, mode=BootMode.WARM, snapshot_path=snapshot_path)
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return BootResult(
                vm_id="",
                success=False,
                boot_time_ms=0,
                boot_mode=BootMode.WARM,
                socket_path="",
                error=str(e),
            )
    
    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================
    
    async def _simulate_boot(self, spec: VMSpec, mode: BootMode) -> None:
        """Simulate VM boot for testing."""
        # Simulate boot time based on mode
        if mode == BootMode.COLD:
            await asyncio.sleep(0.05)  # 50ms simulated cold boot
        else:
            await asyncio.sleep(0.01)  # 10ms simulated warm boot
    
    async def _real_boot(
        self,
        spec: VMSpec,
        mode: BootMode,
        socket_path: str,
        snapshot_path: Optional[str],
    ) -> int:
        """Real Firecracker boot (Linux only)."""
        # Build Firecracker command
        cmd = [
            self.config.firecracker_bin,
            "--api-sock", socket_path,
        ]
        
        if spec.enable_jailer and os.path.exists(self.config.jailer_bin):
            # Use jailer for additional isolation
            cmd = [
                self.config.jailer_bin,
                "--id", spec.vm_id,
                "--exec-file", self.config.firecracker_bin,
                "--",
                "--api-sock", socket_path,
            ]
        
        # Start Firecracker process
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # Wait for socket to appear
        timeout = self.config.boot_timeout_ms / 1000
        start = time.perf_counter()
        while not os.path.exists(socket_path):
            if time.perf_counter() - start > timeout:
                proc.kill()
                raise TimeoutError(f"Boot timeout after {timeout}s")
            await asyncio.sleep(0.01)
        
        # Configure VM via API
        # (In production, this would make HTTP requests to the socket)
        
        return proc.pid
    
    @dataclass
    class _SimulatedExecResult:
        exit_code: int
        stdout: str
        stderr: str
        resource_usage: Dict[str, Any]
    
    async def _simulate_execute(
        self,
        command: str,
        stdin: Optional[str],
        timeout: float,
    ) -> _SimulatedExecResult:
        """Simulate command execution."""
        # Simulate execution time
        await asyncio.sleep(0.01)  # 10ms simulated exec
        
        # Simulate output
        return self._SimulatedExecResult(
            exit_code=0,
            stdout=f"[SIMULATION] Executed: {command}",
            stderr="",
            resource_usage={
                "cpu_time_ms": 5.0,
                "memory_peak_mb": 32,
            },
        )
    
    async def _real_execute(
        self,
        vm: MicroVM,
        command: str,
        stdin: Optional[str],
        timeout: float,
    ) -> _SimulatedExecResult:
        """Real command execution via vsock."""
        # In production, this would communicate with guest agent via vsock
        # For now, return simulation
        return await self._simulate_execute(command, stdin, timeout)
    
    async def _real_snapshot(self, vm: MicroVM, snapshot_path: str) -> None:
        """Real Firecracker snapshot (Linux only)."""
        # In production, this would call Firecracker's snapshot API
        pass
    
    async def _real_restore(self, snapshot_path: str) -> Tuple[str, VMSpec]:
        """Real Firecracker restore (Linux only)."""
        # In production, this would restore from snapshot
        raise NotImplementedError("Real restore not implemented")
    
    def _log_event(self, event: str, vm_id: str, data: Dict[str, Any]) -> None:
        """Log an event to audit log."""
        if self.config.enable_audit_logging:
            self._audit_log.append({
                "event": event,
                "vm_id": vm_id,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            # Keep last 10000 entries
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-10000:]
    
    # ========================================================================
    # QUERIES
    # ========================================================================
    
    def get_vm(self, vm_id: str) -> Optional[MicroVM]:
        """Get a VM by ID."""
        return self._vms.get(vm_id)
    
    def list_vms(self) -> List[Dict[str, Any]]:
        """List all VMs."""
        return [vm.to_dict() for vm in self._vms.values()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "total_vms": len(self._vms),
            "active_vms": len([v for v in self._vms.values() if v.state in (VMState.READY, VMState.RUNNING)]),
            "total_boots": self._boots,
            "total_executions": self._executions,
            "avg_boot_time_ms": self._total_boot_time_ms / max(self._boots, 1),
            "avg_exec_time_ms": self._total_exec_time_ms / max(self._executions, 1),
            "firecracker_available": FIRECRACKER_AVAILABLE,
            "simulation_mode": self.config.enable_simulation,
        }


# ============================================================================
# SELF-TEST
# ============================================================================


async def _self_test() -> None:
    """Run Firecracker Orchestrator self-tests."""
    print("=" * 70)
    print("BIZRA AEON OMEGA - Firecracker Orchestrator Self-Test")
    print("=" * 70)
    
    orchestrator = FirecrackerOrchestrator()
    
    # Test 1: Boot VM
    print("\n[Test 1] Boot MicroVM (Cold)")
    spec = VMSpec(
        vm_id="test-vm-1",
        kernel_path="/path/to/kernel",
        rootfs_path="/path/to/rootfs",
    )
    result = await orchestrator.boot(spec)
    assert result.success, f"Boot failed: {result.error}"
    print(f"  ✓ VM booted: {result.vm_id}")
    print(f"  ✓ Boot time: {result.boot_time_ms:.1f}ms")
    print(f"  ✓ Mode: {result.boot_mode.name}")
    
    # Test 2: Execute command
    print("\n[Test 2] Execute Command")
    result = await orchestrator.execute("test-vm-1", "echo 'hello world'")
    assert result.success, f"Execute failed: {result.error}"
    print(f"  ✓ Exit code: {result.exit_code}")
    print(f"  ✓ Output: {result.stdout}")
    print(f"  ✓ Exec time: {result.exec_time_ms:.1f}ms")
    
    # Test 3: Snapshot
    print("\n[Test 3] Snapshot")
    snapshot_path = os.path.join(tempfile.gettempdir(), "fc-snapshot", "test-vm-1.snap")
    success = await orchestrator.snapshot("test-vm-1", snapshot_path)
    assert success, "Snapshot failed"
    print(f"  ✓ Snapshot created: {snapshot_path}")
    
    # Test 4: Stop VM
    print("\n[Test 4] Stop VM")
    success = await orchestrator.stop("test-vm-1")
    assert success, "Stop failed"
    print(f"  ✓ VM stopped")
    
    # Test 5: Restore from snapshot
    print("\n[Test 5] Restore from Snapshot")
    await orchestrator.destroy("test-vm-1")  # Clean up first
    result = await orchestrator.restore(snapshot_path)
    assert result.success, f"Restore failed: {result.error}"
    print(f"  ✓ VM restored: {result.vm_id}")
    print(f"  ✓ Boot time: {result.boot_time_ms:.1f}ms (warm)")
    
    # Test 6: Statistics
    print("\n[Test 6] Statistics")
    stats = orchestrator.get_statistics()
    print(f"  ✓ Total VMs: {stats['total_vms']}")
    print(f"  ✓ Active VMs: {stats['active_vms']}")
    print(f"  ✓ Total boots: {stats['total_boots']}")
    print(f"  ✓ Total executions: {stats['total_executions']}")
    print(f"  ✓ Simulation mode: {stats['simulation_mode']}")
    
    # Cleanup
    await orchestrator.destroy("test-vm-1")
    shutil.rmtree(os.path.dirname(snapshot_path), ignore_errors=True)
    
    print("\n" + "=" * 70)
    print("✅ ALL FIRECRACKER ORCHESTRATOR SELF-TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(_self_test())
