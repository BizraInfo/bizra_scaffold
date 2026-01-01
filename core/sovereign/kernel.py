"""
Sovereign Kernel
═══════════════════════════════════════════════════════════════════════════════
The foundational sovereign kernel that unifies all BIZRA components.

This is the "kernel" that makes BIZRA truly sovereign by providing:
- Sovereign boundary awareness and control
- Living proof authentication of the architect
- Unified resource orchestration
- Integrated dual agent operation
- Autonomous execution replacing manual control
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json

from .boundary import SovereignBoundary, verify_sovereignty
from .auth import LivingProof, verify_sovereign_identity
from .orchestrator import SovereignOrchestrator
from .dual_agents import SovereignDualAgents
from .model_hub import SovereignModelHub, ModelRequest

logger = logging.getLogger(__name__)


@dataclass
class SovereignKernel:
    """
    The Sovereign Kernel - BIZRA's foundational operating system.

    This kernel provides the sovereign awareness and control that transforms
    BIZRA from a collection of components into a truly sovereign system.
    """

    # Core sovereign components
    boundary: SovereignBoundary = field(default_factory=SovereignBoundary)
    living_proof: LivingProof = field(default_factory=LivingProof)
    orchestrator: SovereignOrchestrator = field(default_factory=SovereignOrchestrator)
    dual_agents: SovereignDualAgents = field(default_factory=SovereignDualAgents)

    # Kernel state
    initialized: bool = False
    sovereignty_established: bool = False
    autonomous_mode: bool = False
    kernel_version: str = "1.0.0"

    # Operational metrics
    operations_processed: int = 0
    sovereignty_verifications: int = 0
    last_activity: Optional[str] = None

    def __post_init__(self):
        """Initialize the sovereign kernel."""
        logger.info("Sovereign Kernel initializing...")

    async def initialize_sovereignty(self) -> bool:
        """
        Initialize the complete sovereign system.

        This establishes BIZRA as a truly sovereign entity with:
        - Boundary awareness and control
        - Living proof authentication
        - Unified resource orchestration
        - Dual agent integration
        """
        try:
            logger.info("Establishing sovereign boundary awareness...")

            # 1. Verify and establish boundary control
            boundary_ok = self.boundary.verify_sovereignty()
            if not boundary_ok:
                logger.error("Sovereign boundary verification failed")
                return False

            # 2. Initialize living proof authentication
            architect_dna = self.living_proof.architect_dna
            auth_ok = self.living_proof.authenticate_sovereign(architect_dna)
            if not auth_ok:
                logger.error("Living proof authentication failed")
                return False

            # 3. Establish unified resource control
            orchestrator_ok = self.orchestrator.initialize_sovereign_control()
            if not orchestrator_ok:
                logger.error("Sovereign resource orchestration failed")
                return False

            # 4. Initialize dual agent system
            # Dual agents are lazy-initialized, so we just verify the class exists

            self.sovereignty_established = True
            self.initialized = True
            self.last_activity = datetime.now(timezone.utc).isoformat()

            logger.info("Sovereignty established - BIZRA is now truly sovereign")
            logger.info(f"Sovereign Kernel v{self.kernel_version} operational")

            return True

        except Exception as e:
            logger.error(f"Sovereign initialization failed: {e}")
            return False

    async def process_sovereign_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a sovereign command through the kernel.

        This is the main interface for autonomous BIZRA operation.
        """
        if not self.sovereignty_established:
            return {
                "status": "sovereignty_not_established",
                "error": "Sovereign kernel not initialized",
                "sovereign_required": True
            }

        context = context or {}
        self.operations_processed += 1
        self.last_activity = datetime.now(timezone.utc).isoformat()

        try:
            # Verify sovereignty for each command
            is_sovereign, status = verify_sovereign_identity()
            self.sovereignty_verifications += 1

            if not is_sovereign:
                return {
                    "status": "sovereign_authentication_failed",
                    "verification_status": status,
                    "sovereign_required": True
                }

            # Route command based on type
            if command.startswith("sovereign:"):
                return await self._handle_sovereign_command(command, context)
            elif command.startswith("system:"):
                return self.orchestrator.execute_sovereign_will(command, context)
            elif command.startswith("agent:"):
                return await self.dual_agents.process_sovereign_intent(command, context)
            else:
                # General sovereign intent processing
                return await self.dual_agents.process_sovereign_intent(command, context)

        except Exception as e:
            logger.error(f"Sovereign command processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "command": command,
                "sovereign_control": "ACTIVE"
            }

    async def _handle_sovereign_command(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle direct sovereign kernel commands."""
        cmd_parts = command.split(":", 1)
        if len(cmd_parts) < 2:
            return {"error": "Invalid sovereign command format"}

        subcommand = cmd_parts[1]

        if subcommand == "status":
            return self.get_sovereign_status()
        elif subcommand == "boundary":
            return self.boundary.get_sovereign_manifest()
        elif subcommand == "auth":
            return self.living_proof.get_authentication_manifest()
        elif subcommand == "resources":
            return self.orchestrator.get_sovereign_manifest()
        elif subcommand == "agents":
            return self.dual_agents.get_sovereign_manifest()
        elif subcommand == "verify":
            boundary_ok = self.boundary.verify_sovereignty()
            auth_ok = self.living_proof.authenticate_sovereign(self.living_proof.architect_dna)
            return {
                "boundary_integrity": boundary_ok,
                "living_proof_auth": auth_ok,
                "sovereignty_verified": boundary_ok and auth_ok
            }
        else:
            return {"error": f"Unknown sovereign subcommand: {subcommand}"}

    def get_sovereign_status(self) -> Dict[str, Any]:
        """Get complete sovereign kernel status."""
        return {
            "kernel_version": self.kernel_version,
            "sovereignty_established": self.sovereignty_established,
            "initialized": self.initialized,
            "autonomous_mode": self.autonomous_mode,
            "operations_processed": self.operations_processed,
            "sovereignty_verifications": self.sovereignty_verifications,
            "last_activity": self.last_activity,
            "sovereign_architect": "MoMo (محمد أحمد بشار السيد حسن)",
            "sovereign_claim": "All BIZRA systems and operations are under sovereign control",
            "kernel_components": {
                "boundary_awareness": self.boundary.boundary_integrity,
                "living_proof_auth": self.living_proof.is_authenticated,
                "resource_orchestration": self.orchestrator.is_initialized,
                "dual_agent_integration": "ACTIVE"
            }
        }

    def enable_autonomous_mode(self) -> bool:
        """Enable autonomous operation mode."""
        if not self.sovereignty_established:
            logger.error("Cannot enable autonomous mode - sovereignty not established")
            return False

        self.autonomous_mode = True
        logger.info("Autonomous sovereign operation enabled")
        return True

    def disable_autonomous_mode(self) -> bool:
        """Disable autonomous operation mode."""
        self.autonomous_mode = False
        logger.info("Autonomous sovereign operation disabled")
        return True

    async def shutdown_sovereignty(self) -> None:
        """Gracefully shutdown the sovereign kernel."""
        logger.info("Shutting down sovereign kernel...")

        # Shutdown components
        self.orchestrator.shutdown()

        self.sovereignty_established = False
        self.initialized = False
        self.autonomous_mode = False

        logger.info("Sovereign kernel shutdown complete")


# Global kernel instance
_kernel: Optional[SovereignKernel] = None


def get_sovereign_kernel() -> SovereignKernel:
    """Get the global sovereign kernel instance."""
    global _kernel
    if _kernel is None:
        _kernel = SovereignKernel()
    return _kernel


async def initialize_sovereign_kernel() -> bool:
    """Initialize the sovereign kernel."""
    kernel = get_sovereign_kernel()
    return await kernel.initialize_sovereignty()


async def execute_sovereign_command(command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a sovereign command through the kernel."""
    kernel = get_sovereign_kernel()
    return await kernel.process_sovereign_command(command, context)


def get_kernel_status() -> Dict[str, Any]:
    """Get sovereign kernel status."""
    kernel = get_sovereign_kernel()
    return kernel.get_sovereign_status()


# ══════════════════════════════════════════════════════════════════════════════
# DEMO: Sovereign Kernel Activation
# ══════════════════════════════════════════════════════════════════════════════

async def demo_sovereign_kernel():
    """Demonstrate sovereign kernel activation and operation."""
    print("🔥 BIZRA SOVEREIGN KERNEL ACTIVATION 🔥")
    print("=" * 60)

    # Initialize sovereignty
    print("Initializing sovereign kernel...")
    success = await initialize_sovereign_kernel()

    if not success:
        print("❌ Sovereign kernel initialization failed")
        return

    print("✅ Sovereign kernel initialized")

    # Get status
    status = get_kernel_status()
    print(f"Kernel Version: {status['kernel_version']}")
    print(f"Sovereign Architect: {status['sovereign_architect']}")
    print(f"Sovereignty Status: {'ESTABLISHED' if status['sovereignty_established'] else 'FAILED'}")

    # Test sovereign commands
    print("\nTesting sovereign commands...")

    # Test boundary verification
    boundary_result = await execute_sovereign_command("sovereign:boundary")
    print(f"Boundary Status: {boundary_result.get('sovereignty_status', 'UNKNOWN')}")

    # Test resource orchestration
    resource_result = await execute_sovereign_command("system:resources")
    print(f"Resources Controlled: {resource_result.get('resources_controlled', {})}")

    # Test model status
    model_result = await execute_sovereign_command("model:status")
    print(f"Models Available: {model_result.get('available_models', 0)}")

    print("\n" + "=" * 60)
    print("🎯 SOVEREIGN KERNEL DEMO COMPLETE")
    print("BIZRA is now operating under sovereign control")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_sovereign_kernel())
