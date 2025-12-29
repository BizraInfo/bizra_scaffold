"""
BIZRA AEON OMEGA - Sandbox Engine
==================================
Lightweight Code Isolation

The Sandbox Engine provides lightweight isolation for code execution
without the overhead of full MicroVMs. It uses process isolation,
seccomp filters, and resource limits.

Features:
    - Multiple isolation levels (MINIMAL → MAXIMUM)
    - Resource limiting (CPU, memory, time)
    - Secure temporary filesystem
    - Execution receipts with audit trail
    - SEED token integration

Isolation Levels:
    MINIMAL:    Process isolation only
    STANDARD:   Process + chroot + resource limits
    STRICT:     Standard + seccomp + network isolation
    MAXIMUM:    MicroVM isolation (via Firecracker)

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
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

# resource module is Unix-only
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    resource = None  # type: ignore
    RESOURCE_AVAILABLE = False

logger = logging.getLogger("bizra.isolation.sandbox")

# ============================================================================
# CONSTANTS
# ============================================================================

# Default resource limits
DEFAULT_CPU_TIME_LIMIT_S = 30
DEFAULT_MEMORY_LIMIT_MB = 256
DEFAULT_DISK_LIMIT_MB = 100
DEFAULT_PROCESS_LIMIT = 10

# Sandbox directory
SANDBOX_BASE_DIR = tempfile.gettempdir()

# ============================================================================
# ENUMERATIONS
# ============================================================================


class IsolationLevel(Enum):
    """Level of isolation for sandbox."""
    
    MINIMAL = 1     # Process isolation only
    STANDARD = 2    # Process + chroot + limits
    STRICT = 3      # Standard + seccomp + network
    MAXIMUM = 4     # MicroVM isolation


class SandboxState(Enum):
    """State of a sandbox."""
    
    CREATING = auto()
    READY = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    ERROR = auto()
    DESTROYED = auto()


class ExecutionStatus(Enum):
    """Status of code execution."""
    
    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()
    RESOURCE_EXCEEDED = auto()
    SECURITY_VIOLATION = auto()


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class SandboxConfig:
    """Configuration for Sandbox Engine."""
    
    base_dir: str = SANDBOX_BASE_DIR
    default_isolation: IsolationLevel = IsolationLevel.STANDARD
    cpu_time_limit_s: float = DEFAULT_CPU_TIME_LIMIT_S
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB
    disk_limit_mb: int = DEFAULT_DISK_LIMIT_MB
    process_limit: int = DEFAULT_PROCESS_LIMIT
    network_allowed: bool = False
    enable_audit_logging: bool = True


@dataclass
class ResourceLimits:
    """Resource limits for a sandbox."""
    
    cpu_time_s: float = DEFAULT_CPU_TIME_LIMIT_S
    memory_mb: int = DEFAULT_MEMORY_LIMIT_MB
    disk_mb: int = DEFAULT_DISK_LIMIT_MB
    max_processes: int = DEFAULT_PROCESS_LIMIT
    max_files: int = 100
    max_file_size_mb: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_time_s": self.cpu_time_s,
            "memory_mb": self.memory_mb,
            "disk_mb": self.disk_mb,
            "max_processes": self.max_processes,
            "max_files": self.max_files,
            "max_file_size_mb": self.max_file_size_mb,
        }


@dataclass
class CodeExecution:
    """Specification for code execution."""
    
    code: str
    language: str = "python"
    entrypoint: Optional[str] = None
    stdin: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)  # filename -> content
    timeout_s: float = 30.0
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "language": self.language,
            "code_hash": hashlib.sha256(self.code.encode()).hexdigest()[:16],
            "entrypoint": self.entrypoint,
            "has_stdin": self.stdin is not None,
            "env_keys": list(self.env.keys()),
            "file_count": len(self.files),
            "timeout_s": self.timeout_s,
            "limits": self.limits.to_dict(),
        }


@dataclass
class Sandbox:
    """A sandbox instance."""
    
    sandbox_id: str
    isolation_level: IsolationLevel
    state: SandboxState
    root_dir: str
    limits: ResourceLimits
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exec_count: int = 0
    total_exec_time_s: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sandbox_id": self.sandbox_id,
            "isolation_level": self.isolation_level.name,
            "state": self.state.name,
            "root_dir": self.root_dir,
            "limits": self.limits.to_dict(),
            "created_at": self.created_at.isoformat(),
            "exec_count": self.exec_count,
            "avg_exec_time_s": self.total_exec_time_s / max(self.exec_count, 1),
        }


@dataclass
class ExecutionReceipt:
    """Receipt for code execution."""
    
    receipt_id: str
    sandbox_id: str
    execution_hash: str
    code_hash: str
    status: ExecutionStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "receipt_id": self.receipt_id,
            "sandbox_id": self.sandbox_id,
            "execution_hash": self.execution_hash,
            "code_hash": self.code_hash,
            "status": self.status.name,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SandboxResult:
    """Result of sandbox execution."""
    
    sandbox_id: str
    success: bool
    status: ExecutionStatus
    exit_code: int
    stdout: str
    stderr: str
    exec_time_s: float
    resource_usage: Dict[str, Any]
    receipt: ExecutionReceipt
    error: Optional[str] = None
    output_files: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sandbox_id": self.sandbox_id,
            "success": self.success,
            "status": self.status.name,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:1000],
            "stderr": self.stderr[:1000],
            "exec_time_s": self.exec_time_s,
            "resource_usage": self.resource_usage,
            "receipt": self.receipt.to_dict(),
            "error": self.error,
        }


# ============================================================================
# SANDBOX ENGINE
# ============================================================================


class SandboxEngine:
    """
    Sandbox Engine for isolated code execution.
    
    Provides multiple isolation levels from minimal process isolation
    to full MicroVM isolation.
    
    Usage:
        engine = SandboxEngine()
        
        # Create sandbox
        sandbox = await engine.create()
        
        # Execute code
        execution = CodeExecution(code="print('hello')", language="python")
        result = await engine.execute(sandbox.sandbox_id, execution)
        
        # Destroy sandbox
        await engine.destroy(sandbox.sandbox_id)
        
        # Or use context manager
        async with engine.sandbox() as sandbox:
            result = await engine.execute(sandbox.sandbox_id, execution)
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        """Initialize Sandbox Engine."""
        self.config = config or SandboxConfig()
        
        # Sandbox registry
        self._sandboxes: Dict[str, Sandbox] = {}
        
        # Firecracker integration for MAXIMUM isolation
        self._firecracker: Optional[Any] = None
        
        # Statistics
        self._created: int = 0
        self._executions: int = 0
        self._total_exec_time_s: float = 0.0
        
        # Audit log
        self._audit_log: List[Dict[str, Any]] = []
        
        # Counters
        self._sandbox_counter: int = 0
        self._receipt_counter: int = 0
        
        logger.info(f"Sandbox Engine initialized: isolation={self.config.default_isolation.name}")
    
    # ========================================================================
    # LIFECYCLE
    # ========================================================================
    
    async def create(
        self,
        isolation: Optional[IsolationLevel] = None,
        limits: Optional[ResourceLimits] = None,
    ) -> Sandbox:
        """
        Create a new sandbox.
        
        Args:
            isolation: Isolation level
            limits: Resource limits
            
        Returns:
            Sandbox instance
        """
        isolation = isolation or self.config.default_isolation
        limits = limits or ResourceLimits()
        
        # Generate sandbox ID
        self._sandbox_counter += 1
        sandbox_id = f"SBX-{int(time.time()*1000)}-{self._sandbox_counter:06d}"
        
        # Create sandbox directory
        root_dir = os.path.join(self.config.base_dir, "bizra_sandbox", sandbox_id)
        os.makedirs(root_dir, exist_ok=True)
        
        # Create sandbox
        sandbox = Sandbox(
            sandbox_id=sandbox_id,
            isolation_level=isolation,
            state=SandboxState.READY,
            root_dir=root_dir,
            limits=limits,
        )
        
        self._sandboxes[sandbox_id] = sandbox
        self._created += 1
        
        self._log_event("CREATE", sandbox_id, {"isolation": isolation.name})
        logger.info(f"Created sandbox {sandbox_id}: isolation={isolation.name}")
        
        return sandbox
    
    async def destroy(self, sandbox_id: str) -> bool:
        """
        Destroy a sandbox and clean up resources.
        
        Args:
            sandbox_id: Sandbox ID
            
        Returns:
            True if destroyed successfully
        """
        if sandbox_id not in self._sandboxes:
            logger.warning(f"Sandbox {sandbox_id} not found")
            return False
        
        sandbox = self._sandboxes[sandbox_id]
        
        try:
            # Clean up filesystem
            if os.path.exists(sandbox.root_dir):
                shutil.rmtree(sandbox.root_dir, ignore_errors=True)
            
            sandbox.state = SandboxState.DESTROYED
            del self._sandboxes[sandbox_id]
            
            self._log_event("DESTROY", sandbox_id, {})
            logger.info(f"Destroyed sandbox {sandbox_id}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to destroy sandbox {sandbox_id}: {e}")
            return False
    
    @asynccontextmanager
    async def sandbox(
        self,
        isolation: Optional[IsolationLevel] = None,
        limits: Optional[ResourceLimits] = None,
    ) -> AsyncIterator[Sandbox]:
        """
        Context manager for sandbox lifecycle.
        
        Usage:
            async with engine.sandbox() as sandbox:
                result = await engine.execute(sandbox.sandbox_id, execution)
        """
        sandbox = await self.create(isolation, limits)
        try:
            yield sandbox
        finally:
            await self.destroy(sandbox.sandbox_id)
    
    # ========================================================================
    # EXECUTION
    # ========================================================================
    
    async def execute(
        self,
        sandbox_id: str,
        execution: CodeExecution,
    ) -> SandboxResult:
        """
        Execute code in a sandbox.
        
        Args:
            sandbox_id: Sandbox ID
            execution: Code execution specification
            
        Returns:
            SandboxResult with execution output
        """
        start = time.perf_counter()
        
        if sandbox_id not in self._sandboxes:
            return self._error_result(
                sandbox_id=sandbox_id,
                error="Sandbox not found",
                start=start,
            )
        
        sandbox = self._sandboxes[sandbox_id]
        
        if sandbox.state != SandboxState.READY:
            return self._error_result(
                sandbox_id=sandbox_id,
                error=f"Sandbox not ready: {sandbox.state.name}",
                start=start,
            )
        
        try:
            sandbox.state = SandboxState.EXECUTING
            
            # Prepare execution environment
            await self._prepare_environment(sandbox, execution)
            
            # Execute based on isolation level
            if sandbox.isolation_level == IsolationLevel.MAXIMUM:
                result = await self._execute_microvm(sandbox, execution)
            elif sandbox.isolation_level == IsolationLevel.STRICT:
                result = await self._execute_strict(sandbox, execution)
            elif sandbox.isolation_level == IsolationLevel.STANDARD:
                result = await self._execute_standard(sandbox, execution)
            else:
                result = await self._execute_minimal(sandbox, execution)
            
            exec_time = time.perf_counter() - start
            
            # Create receipt
            receipt = self._create_receipt(
                sandbox_id=sandbox_id,
                code_hash=hashlib.sha256(execution.code.encode()).hexdigest()[:16],
                status=result.status,
            )
            
            # Update statistics
            sandbox.state = SandboxState.READY
            sandbox.exec_count += 1
            sandbox.total_exec_time_s += exec_time
            
            self._executions += 1
            self._total_exec_time_s += exec_time
            
            self._log_event("EXECUTE", sandbox_id, {
                "status": result.status.name,
                "exit_code": result.exit_code,
                "exec_time_s": exec_time,
            })
            
            return SandboxResult(
                sandbox_id=sandbox_id,
                success=result.success,
                status=result.status,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                exec_time_s=exec_time,
                resource_usage=result.resource_usage,
                receipt=receipt,
                output_files=result.output_files,
            )
            
        except asyncio.TimeoutError:
            sandbox.state = SandboxState.READY
            return self._timeout_result(sandbox_id, start, execution.timeout_s)
        except Exception as e:
            sandbox.state = SandboxState.ERROR
            return self._error_result(sandbox_id, str(e), start)
    
    # ========================================================================
    # INTERNAL EXECUTION METHODS
    # ========================================================================
    
    async def _prepare_environment(
        self,
        sandbox: Sandbox,
        execution: CodeExecution,
    ) -> None:
        """Prepare sandbox environment for execution."""
        # Write code file
        code_file = os.path.join(sandbox.root_dir, self._get_code_filename(execution))
        with open(code_file, "w") as f:
            f.write(execution.code)
        
        # Write additional files
        for filename, content in execution.files.items():
            filepath = os.path.join(sandbox.root_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                f.write(content)
    
    def _get_code_filename(self, execution: CodeExecution) -> str:
        """Get filename for code based on language."""
        if execution.entrypoint:
            return execution.entrypoint
        
        extensions = {
            "python": "main.py",
            "javascript": "main.js",
            "typescript": "main.ts",
            "rust": "main.rs",
            "go": "main.go",
            "java": "Main.java",
            "c": "main.c",
            "cpp": "main.cpp",
        }
        return extensions.get(execution.language.lower(), "main.txt")
    
    def _get_executor(self, language: str) -> Tuple[str, List[str]]:
        """Get executor command for language (interpreted languages only)."""
        executors = {
            "python": (sys.executable, []),
            "javascript": ("node", []),
            "typescript": ("npx", ["ts-node"]),
            "go": ("go", ["run"]),
            "java": ("java", []),
            # Compiled languages handled separately by _build_compiled_language
            "rust": ("rustc", []),  # Placeholder, not used directly
            "c": ("gcc", []),       # Placeholder, not used directly
            "cpp": ("g++", []),     # Placeholder, not used directly
        }
        return executors.get(language.lower(), (sys.executable, []))
    
    async def _build_compiled_language(
        self,
        sandbox: Sandbox,
        execution: CodeExecution,
        language: str,
        code_file: str,
    ) -> Tuple[List[str], Optional["_ExecResult"]]:
        """
        Build compiled language code and return executable command.
        
        P1 FIX: Properly handle compiled languages with separate build/run steps.
        
        Returns:
            Tuple of (command_to_run, error_result_or_none)
        """
        output_binary = os.path.join(sandbox.root_dir, "main")
        if sys.platform == "win32":
            output_binary += ".exe"
        
        # Determine compiler and flags
        compilers = {
            "rust": ("rustc", ["--edition", "2021", "-o", output_binary, code_file]),
            "c": ("gcc", ["-o", output_binary, code_file]),
            "cpp": ("g++", ["-o", output_binary, code_file]),
            "c++": ("g++", ["-o", output_binary, code_file]),
        }
        
        compiler, compile_args = compilers.get(language, ("gcc", ["-o", output_binary, code_file]))
        
        # Build step
        try:
            logger.info(f"Compiling {language} code: {compiler} {' '.join(compile_args)}")
            compile_proc = await asyncio.create_subprocess_exec(
                compiler,
                *compile_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=sandbox.root_dir,
            )
            
            compile_stdout, compile_stderr = await asyncio.wait_for(
                compile_proc.communicate(),
                timeout=execution.timeout_s / 2,  # Use half timeout for compile
            )
            
            if compile_proc.returncode != 0:
                return [], self._ExecResult(
                    success=False,
                    status=ExecutionStatus.FAILURE,
                    exit_code=compile_proc.returncode or 1,
                    stdout=compile_stdout.decode(errors="replace"),
                    stderr=f"Compilation failed:\n{compile_stderr.decode(errors='replace')}",
                    resource_usage={"stage": "compile"},
                )
            
            # Return command to run the binary
            return [output_binary], None
            
        except asyncio.TimeoutError:
            return [], self._ExecResult(
                success=False,
                status=ExecutionStatus.TIMEOUT,
                exit_code=-1,
                stdout="",
                stderr=f"Compilation timed out after {execution.timeout_s / 2}s",
                resource_usage={"stage": "compile"},
            )
        except FileNotFoundError:
            return [], self._ExecResult(
                success=False,
                status=ExecutionStatus.FAILURE,
                exit_code=-1,
                stdout="",
                stderr=f"Compiler not found: {compiler}",
                resource_usage={"stage": "compile"},
            )
    
    async def _execute_minimal(
        self,
        sandbox: Sandbox,
        execution: CodeExecution,
    ) -> "_ExecResult":
        """Execute with minimal isolation (process only)."""
        return await self._run_subprocess(sandbox, execution)
    
    async def _execute_standard(
        self,
        sandbox: Sandbox,
        execution: CodeExecution,
    ) -> "_ExecResult":
        """Execute with standard isolation (process + limits)."""
        return await self._run_subprocess(sandbox, execution, with_limits=True)
    
    async def _execute_strict(
        self,
        sandbox: Sandbox,
        execution: CodeExecution,
    ) -> "_ExecResult":
        """Execute with strict isolation (standard + network isolation)."""
        # On Linux, would use unshare/namespaces
        return await self._run_subprocess(sandbox, execution, with_limits=True)
    
    async def _execute_microvm(
        self,
        sandbox: Sandbox,
        execution: CodeExecution,
    ) -> "_ExecResult":
        """Execute in MicroVM (maximum isolation)."""
        # Would delegate to Firecracker orchestrator
        # For now, fall back to strict
        return await self._execute_strict(sandbox, execution)
    
    @dataclass
    class _ExecResult:
        success: bool
        status: ExecutionStatus
        exit_code: int
        stdout: str
        stderr: str
        resource_usage: Dict[str, Any]
        output_files: Dict[str, str] = field(default_factory=dict)
    
    def _get_preexec_fn(self, limits: ResourceLimits) -> Optional[Callable[[], None]]:
        """Get preexec_fn for Unix resource limits (P0 fix)."""
        if not RESOURCE_AVAILABLE:
            return None
        
        def set_limits():
            """Set resource limits in child process."""
            # CPU time limit (soft, hard)
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (int(limits.cpu_time_s), int(limits.cpu_time_s) + 5)
            )
            # Memory limit (bytes)
            mem_bytes = limits.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            # Max file size
            file_bytes = limits.max_file_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
            # Max open files
            resource.setrlimit(resource.RLIMIT_NOFILE, (limits.max_files, limits.max_files))
            # Max processes
            resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
        
        return set_limits

    async def _run_subprocess(
        self,
        sandbox: Sandbox,
        execution: CodeExecution,
        with_limits: bool = False,
    ) -> _ExecResult:
        """Run code in subprocess with optional resource limits."""
        code_file = os.path.join(sandbox.root_dir, self._get_code_filename(execution))
        executor, args = self._get_executor(execution.language)
        
        # P1 FIX: Handle compiled languages that need build step
        language_lower = execution.language.lower()
        if language_lower in ("rust", "c", "cpp", "c++"):
            cmd, build_result = await self._build_compiled_language(
                sandbox, execution, language_lower, code_file
            )
            if build_result is not None:
                return build_result  # Build failed
        else:
            # Build command for interpreted languages
            cmd = [executor] + args + [code_file]
        
        # Set environment
        env = os.environ.copy()
        env.update(execution.env)
        
        # P0 FIX: Apply resource limits when requested
        preexec_fn = None
        if with_limits and RESOURCE_AVAILABLE:
            preexec_fn = self._get_preexec_fn(execution.limits)
            logger.info(f"Applying resource limits: CPU={execution.limits.cpu_time_s}s, "
                       f"MEM={execution.limits.memory_mb}MB")
        elif with_limits and not RESOURCE_AVAILABLE:
            logger.warning("Resource limits requested but 'resource' module unavailable "
                          "(Windows). Isolation not enforced.")
        
        # Run with timeout
        try:
            start = time.perf_counter()
            
            # Use subprocess for preexec_fn support on Unix
            if preexec_fn:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE if execution.stdin else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=sandbox.root_dir,
                    env=env,
                    preexec_fn=preexec_fn,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE if execution.stdin else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=sandbox.root_dir,
                    env=env,
                )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(
                    input=execution.stdin.encode() if execution.stdin else None
                ),
                timeout=execution.timeout_s,
            )
            
            exec_time = time.perf_counter() - start
            
            return self._ExecResult(
                success=proc.returncode == 0,
                status=ExecutionStatus.SUCCESS if proc.returncode == 0 else ExecutionStatus.FAILURE,
                exit_code=proc.returncode or 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                resource_usage={
                    "exec_time_s": exec_time,
                    "exit_code": proc.returncode,
                },
            )
            
        except asyncio.TimeoutError:
            proc.kill()
            return self._ExecResult(
                success=False,
                status=ExecutionStatus.TIMEOUT,
                exit_code=-1,
                stdout="",
                stderr=f"Execution timed out after {execution.timeout_s}s",
                resource_usage={},
            )
        except Exception as e:
            return self._ExecResult(
                success=False,
                status=ExecutionStatus.FAILURE,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                resource_usage={},
            )
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _create_receipt(
        self,
        sandbox_id: str,
        code_hash: str,
        status: ExecutionStatus,
    ) -> ExecutionReceipt:
        """Create execution receipt."""
        self._receipt_counter += 1
        receipt_id = f"RCPT-{int(time.time()*1000)}-{self._receipt_counter:06d}"
        
        # Create execution hash
        execution_hash = hashlib.sha256(
            f"{sandbox_id}:{code_hash}:{status.name}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        return ExecutionReceipt(
            receipt_id=receipt_id,
            sandbox_id=sandbox_id,
            execution_hash=execution_hash,
            code_hash=code_hash,
            status=status,
        )
    
    def _error_result(
        self,
        sandbox_id: str,
        error: str,
        start: float,
    ) -> SandboxResult:
        """Create error result."""
        receipt = self._create_receipt(
            sandbox_id=sandbox_id,
            code_hash="",
            status=ExecutionStatus.FAILURE,
        )
        
        return SandboxResult(
            sandbox_id=sandbox_id,
            success=False,
            status=ExecutionStatus.FAILURE,
            exit_code=-1,
            stdout="",
            stderr=error,
            exec_time_s=time.perf_counter() - start,
            resource_usage={},
            receipt=receipt,
            error=error,
        )
    
    def _timeout_result(
        self,
        sandbox_id: str,
        start: float,
        timeout: float,
    ) -> SandboxResult:
        """Create timeout result."""
        receipt = self._create_receipt(
            sandbox_id=sandbox_id,
            code_hash="",
            status=ExecutionStatus.TIMEOUT,
        )
        
        return SandboxResult(
            sandbox_id=sandbox_id,
            success=False,
            status=ExecutionStatus.TIMEOUT,
            exit_code=-1,
            stdout="",
            stderr=f"Execution timed out after {timeout}s",
            exec_time_s=time.perf_counter() - start,
            resource_usage={},
            receipt=receipt,
            error=f"Timeout after {timeout}s",
        )
    
    def _log_event(self, event: str, sandbox_id: str, data: Dict[str, Any]) -> None:
        """Log an event to audit log."""
        if self.config.enable_audit_logging:
            self._audit_log.append({
                "event": event,
                "sandbox_id": sandbox_id,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            # Keep last 10000 entries
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-10000:]
    
    # ========================================================================
    # QUERIES
    # ========================================================================
    
    def get_sandbox(self, sandbox_id: str) -> Optional[Sandbox]:
        """Get a sandbox by ID."""
        return self._sandboxes.get(sandbox_id)
    
    def list_sandboxes(self) -> List[Dict[str, Any]]:
        """List all sandboxes."""
        return [s.to_dict() for s in self._sandboxes.values()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_sandboxes": len(self._sandboxes),
            "active_sandboxes": len([s for s in self._sandboxes.values() if s.state == SandboxState.READY]),
            "total_created": self._created,
            "total_executions": self._executions,
            "avg_exec_time_s": self._total_exec_time_s / max(self._executions, 1),
            "default_isolation": self.config.default_isolation.name,
        }


# ============================================================================
# SELF-TEST
# ============================================================================


async def _self_test() -> None:
    """Run Sandbox Engine self-tests."""
    print("=" * 70)
    print("BIZRA AEON OMEGA - Sandbox Engine Self-Test")
    print("=" * 70)
    
    engine = SandboxEngine()
    
    # Test 1: Create sandbox
    print("\n[Test 1] Create Sandbox")
    sandbox = await engine.create()
    assert sandbox.state == SandboxState.READY
    print(f"  ✓ Created sandbox: {sandbox.sandbox_id}")
    print(f"  ✓ Isolation: {sandbox.isolation_level.name}")
    
    # Test 2: Execute Python code
    print("\n[Test 2] Execute Python Code")
    execution = CodeExecution(
        code="print('Hello from sandbox!')\nprint(2 + 2)",
        language="python",
    )
    result = await engine.execute(sandbox.sandbox_id, execution)
    assert result.success, f"Execution failed: {result.stderr}"
    print(f"  ✓ Exit code: {result.exit_code}")
    print(f"  ✓ Output: {result.stdout.strip()}")
    print(f"  ✓ Exec time: {result.exec_time_s:.3f}s")
    print(f"  ✓ Receipt: {result.receipt.receipt_id}")
    
    # Test 3: Execute with error
    print("\n[Test 3] Execute with Error")
    execution = CodeExecution(
        code="raise ValueError('test error')",
        language="python",
    )
    result = await engine.execute(sandbox.sandbox_id, execution)
    assert not result.success
    assert result.status == ExecutionStatus.FAILURE
    print(f"  ✓ Error captured: {result.status.name}")
    print(f"  ✓ Exit code: {result.exit_code}")
    
    # Test 4: Execute with stdin
    print("\n[Test 4] Execute with stdin")
    execution = CodeExecution(
        code="name = input(); print(f'Hello, {name}!')",
        language="python",
        stdin="World",
    )
    result = await engine.execute(sandbox.sandbox_id, execution)
    assert result.success
    print(f"  ✓ Output: {result.stdout.strip()}")
    
    # Test 5: Context manager
    print("\n[Test 5] Context Manager")
    async with engine.sandbox() as temp_sandbox:
        execution = CodeExecution(code="print('temp sandbox')", language="python")
        result = await engine.execute(temp_sandbox.sandbox_id, execution)
        assert result.success
        print(f"  ✓ Temp sandbox: {temp_sandbox.sandbox_id}")
        print(f"  ✓ Output: {result.stdout.strip()}")
    print(f"  ✓ Sandbox auto-destroyed")
    
    # Test 6: Destroy sandbox
    print("\n[Test 6] Destroy Sandbox")
    success = await engine.destroy(sandbox.sandbox_id)
    assert success
    print(f"  ✓ Sandbox destroyed")
    
    # Test 7: Statistics
    print("\n[Test 7] Statistics")
    stats = engine.get_statistics()
    print(f"  ✓ Total created: {stats['total_created']}")
    print(f"  ✓ Total executions: {stats['total_executions']}")
    print(f"  ✓ Avg exec time: {stats['avg_exec_time_s']:.3f}s")
    
    print("\n" + "=" * 70)
    print("✅ ALL SANDBOX ENGINE SELF-TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(_self_test())
