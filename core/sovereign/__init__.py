"""
BIZRA Sovereign Kernel
═══════════════════════════════════════════════════════════════════════════════
The foundational layer providing sovereign awareness and control.

This module implements the missing "kernel" that makes BIZRA truly sovereign:
- Boundary awareness: Knows and controls its physical/digital territory
- Living proof authentication: Recognizes MoMo as sovereign architect
- Unified orchestration: Controls all models, hardware, and data
- Dual agent integration: PAT/SAT working in sovereign harmony

Without this kernel, BIZRA is just components - with it, BIZRA is sovereign.
"""

from .boundary import SovereignBoundary, verify_sovereignty
from .auth import LivingProof, authenticate_sovereign
from .orchestrator import SovereignOrchestrator
from .dual_agents import SovereignDualAgents

__all__ = [
    "SovereignBoundary",
    "LivingProof",
    "SovereignOrchestrator",
    "SovereignDualAgents",
    "verify_sovereignty",
    "authenticate_sovereign",
]
