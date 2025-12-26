"""
BIZRA PCI Gate Integration
═══════════════════════════════════════════════════════════════════════════════
Wires PCI (Proof-Carrying Inference) verification into the APEX orchestrator.

This module provides:
1. PCIGate - Verification boundary for action execution
2. PCIEnabledOrchestrator - APEX orchestrator with PCI enforcement
3. Integration helpers for existing components

PROTOCOL.md Section 5.4: GoT→Action Boundary
- Every state-mutating action MUST pass through PCI gate
- SAT agent verifies envelope before execution
- CommitReceipt binds action to audit log

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric import ed25519

# PCI imports
from core.pci.envelope import PCIEnvelope, Sender, Payload, Metadata, Signature
from core.pci.reject_codes import RejectCode, RejectionResponse
from core.pci.replay_guard import ReplayGuard, get_replay_guard

# Agent imports
from core.agents.pat import PATAgent, create_pat_agent, IHSAN_THRESHOLD
from core.agents.sat import SATAgent, create_sat_agent, VerificationResult, CommitReceipt


logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of PCI gate verification."""
    
    allowed: bool
    envelope: Optional[PCIEnvelope] = None
    receipt: Optional[CommitReceipt] = None
    rejection: Optional[RejectionResponse] = None
    latency_ms: float = 0.0
    
    @property
    def reject_code(self) -> Optional[RejectCode]:
        """Get reject code if rejected."""
        return self.rejection.code if self.rejection else None


class PCIGate:
    """
    PCI verification gate for action execution boundary.
    
    Implements fail-closed semantics:
    - Every action MUST be wrapped in signed PCIEnvelope
    - SAT agent verifies before execution
    - Success → CommitReceipt issued
    - Failure → Action blocked with RejectCode
    
    Usage:
        gate = PCIGate.create(policy_hash_provider=get_current_policy)
        result = await gate.check(
            action="state.mutate",
            data={"key": "value"},
            ihsan_score=0.97,
            origin_layer=4,
        )
        if result.allowed:
            execute_action(...)
        else:
            log_rejection(result.rejection)
    """
    
    def __init__(
        self,
        pat_agent: PATAgent,
        sat_agent: SATAgent,
        fail_open: bool = False,  # CRITICAL: Default is fail-closed
    ):
        """
        Initialize PCI gate.
        
        Args:
            pat_agent: Agent for creating proposals
            sat_agent: Agent for verifying proposals
            fail_open: If True, allows actions on gate failure (DANGEROUS)
        """
        self._pat = pat_agent
        self._sat = sat_agent
        self._fail_open = fail_open
        
        # Statistics
        self._total_checks = 0
        self._allowed = 0
        self._rejected = 0
        self._errors = 0
        
        if fail_open:
            logger.warning(
                "PCI Gate initialized with fail_open=True. "
                "This is INSECURE and should only be used for testing!"
            )
        
        logger.info(
            f"PCI Gate initialized: pat={pat_agent.agent_id}, "
            f"sat={sat_agent.agent_id}, fail_open={fail_open}"
        )
    
    async def check(
        self,
        action: str,
        data: Dict[str, Any],
        policy_hash: str,
        ihsan_score: float,
        origin_layer: int,
        snr_score: Optional[float] = None,
        trace_id: Optional[str] = None,
        timeout_ms: float = 200.0,
    ) -> GateResult:
        """
        Check if action is allowed through PCI verification.
        
        This is the main entry point for action verification.
        
        Args:
            action: Action identifier (e.g., "state.mutate", "token.transfer")
            data: Action payload
            policy_hash: Current constitution policy hash
            ihsan_score: Ihsān ethical alignment score
            origin_layer: APEX layer originating the action (1-7)
            snr_score: Optional signal-to-noise score
            trace_id: Optional trace ID for correlation
            timeout_ms: Maximum time for verification
            
        Returns:
            GateResult with allowed=True and receipt, or allowed=False and rejection
        """
        start_time = time.perf_counter()
        deadline = None
        if timeout_ms and timeout_ms > 0:
            deadline = start_time + (timeout_ms / 1000.0)
        
        self._total_checks += 1
        
        try:
            def remaining_timeout() -> Optional[float]:
                if deadline is None:
                    return None
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                return remaining

            # 1. PAT creates proposal
            proposal = await asyncio.wait_for(
                asyncio.to_thread(
                    self._pat.create_proposal,
                    action=action,
                    data=data,
                    policy_hash=policy_hash,
                    ihsan_score=ihsan_score,
                    origin_layer=origin_layer,
                    snr_score=snr_score,
                    trace_id=trace_id,
                ),
                timeout=remaining_timeout(),
            )
            
            if not proposal.success:
                self._rejected += 1
                latency = (time.perf_counter() - start_time) * 1000
                logger.debug(
                    f"PCI Gate: PAT rejected action={action}, "
                    f"code={proposal.rejection.code.name}"
                )
                return GateResult(
                    allowed=False,
                    rejection=proposal.rejection,
                    latency_ms=latency,
                )
            
            # 2. SAT verifies envelope
            verification = await asyncio.wait_for(
                asyncio.to_thread(self._sat.verify, proposal.envelope),
                timeout=remaining_timeout(),
            )
            
            latency = (time.perf_counter() - start_time) * 1000
            
            if verification.success:
                self._allowed += 1
                logger.debug(
                    f"PCI Gate: Allowed action={action}, "
                    f"receipt={verification.receipt.receipt_id}"
                )
                return GateResult(
                    allowed=True,
                    envelope=proposal.envelope,
                    receipt=verification.receipt,
                    latency_ms=latency,
                )
            else:
                self._rejected += 1
                logger.debug(
                    f"PCI Gate: SAT rejected action={action}, "
                    f"code={verification.rejection.code.name}"
                )
                return GateResult(
                    allowed=False,
                    envelope=proposal.envelope,
                    rejection=verification.rejection,
                    latency_ms=latency,
                )
                
        except asyncio.TimeoutError:
            self._errors += 1
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(f"PCI Gate: Timeout verifying action={action}")
            rejection = RejectionResponse.create(
                code=RejectCode.REJECT_BUDGET_EXCEEDED,
                envelope_digest="timeout",
                gate="BUDGET",
                latency_ms=latency,
                details={"error": "Verification timeout"},
            )
            if self._fail_open:
                logger.warning("Allowing action due to fail_open=True (INSECURE)")
                return GateResult(
                    allowed=True,
                    rejection=rejection,
                    latency_ms=latency,
                )
            
            return GateResult(
                allowed=False,
                rejection=rejection,
                latency_ms=latency,
            )
            
        except Exception as e:
            self._errors += 1
            latency = (time.perf_counter() - start_time) * 1000
            logger.exception(f"PCI Gate: Error verifying action={action}: {e}")
            rejection = RejectionResponse.create(
                code=RejectCode.REJECT_INTERNAL_ERROR,
                envelope_digest="error",
                gate="INTERNAL",
                latency_ms=latency,
                details={"error": str(e)},
            )
            if self._fail_open:
                logger.warning("Allowing action due to fail_open=True (INSECURE)")
                return GateResult(
                    allowed=True,
                    rejection=rejection,
                    latency_ms=latency,
                )
            
            return GateResult(
                allowed=False,
                rejection=rejection,
                latency_ms=latency,
            )
    
    def stats(self) -> Dict[str, Any]:
        """Get gate statistics."""
        return {
            "total_checks": self._total_checks,
            "allowed": self._allowed,
            "rejected": self._rejected,
            "errors": self._errors,
            "allow_rate": self._allowed / self._total_checks if self._total_checks > 0 else 0.0,
            "pat_stats": self._pat.stats(),
            "sat_stats": self._sat.stats(),
        }
    
    @classmethod
    def create(
        cls,
        policy_hash_provider: Callable[[], str],
        pat_agent_id: Optional[str] = None,
        sat_agent_id: Optional[str] = None,
        ihsan_threshold: float = IHSAN_THRESHOLD,
        fail_open: bool = False,
    ) -> "PCIGate":
        """
        Factory method to create a PCI gate with default agents.
        
        Args:
            policy_hash_provider: Callback to get current policy hash
            pat_agent_id: Optional PAT agent ID
            sat_agent_id: Optional SAT agent ID
            ihsan_threshold: Minimum Ihsān score
            fail_open: If True, allows on gate failure (DANGEROUS)
            
        Returns:
            Configured PCIGate instance
        """
        pat = create_pat_agent(
            agent_id=pat_agent_id,
            ihsan_threshold=ihsan_threshold,
        )
        
        sat = create_sat_agent(
            agent_id=sat_agent_id,
            ihsan_threshold=ihsan_threshold,
            policy_hash_provider=policy_hash_provider,
        )
        
        return cls(pat_agent=pat, sat_agent=sat, fail_open=fail_open)


# Global gate instance (lazily initialized)
_global_pci_gate: Optional[PCIGate] = None


def get_pci_gate(
    policy_hash_provider: Optional[Callable[[], str]] = None,
) -> PCIGate:
    """
    Get or create the global PCI gate instance.
    
    Args:
        policy_hash_provider: Required on first call to initialize
        
    Returns:
        Global PCIGate instance
    """
    global _global_pci_gate
    
    if _global_pci_gate is None:
        if policy_hash_provider is None:
            raise RuntimeError(
                "policy_hash_provider required for first initialization"
            )
        _global_pci_gate = PCIGate.create(policy_hash_provider=policy_hash_provider)
    
    return _global_pci_gate


def reset_pci_gate() -> None:
    """Reset the global PCI gate (for testing)."""
    global _global_pci_gate
    _global_pci_gate = None


# ═══════════════════════════════════════════════════════════════════════════════
# Decorator for PCI-protected functions
# ═══════════════════════════════════════════════════════════════════════════════

def pci_protected(
    action: Optional[str] = None,
    require_ihsan: float = IHSAN_THRESHOLD,
):
    """
    Decorator to wrap functions with PCI verification.
    
    Usage:
        @pci_protected(action="state.update", require_ihsan=0.95)
        async def update_state(data: dict, ihsan_score: float, policy_hash: str):
            # Only executed if PCI gate allows
            ...
    
    Args:
        action: Action identifier (defaults to function name)
        require_ihsan: Minimum Ihsān score required
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        import functools

        async def _run_gate(kwargs: Dict[str, Any]) -> GateResult:
            ihsan_score = kwargs.pop("ihsan_score", require_ihsan)
            policy_hash = kwargs.pop("policy_hash", None)
            if not policy_hash:
                raise RuntimeError("policy_hash is required for PCI verification")
            origin_layer = kwargs.pop("origin_layer", 4)
            data = kwargs.pop("data", {})

            gate = get_pci_gate()
            return await gate.check(
                action=action or func.__name__,
                data=data,
                policy_hash=policy_hash,
                ihsan_score=ihsan_score,
                origin_layer=origin_layer,
            )

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                result = await _run_gate(kwargs)
                if not result.allowed:
                    rejection = result.rejection or RejectionResponse.create(
                        code=RejectCode.REJECT_INTERNAL_ERROR,
                        envelope_digest="unknown",
                        gate="PCI_GATE",
                        latency_ms=0.0,
                        details={"reason": "missing rejection"},
                    )
                    raise PCIRejectionError(rejection)
                return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                result = asyncio.run(_run_gate(kwargs))
            else:
                raise RuntimeError(
                    "pci_protected sync wrapper called inside running event loop; "
                    "use an async function instead"
                )
            if not result.allowed:
                rejection = result.rejection or RejectionResponse.create(
                    code=RejectCode.REJECT_INTERNAL_ERROR,
                    envelope_digest="unknown",
                    gate="PCI_GATE",
                    latency_ms=0.0,
                    details={"reason": "missing rejection"},
                )
                raise PCIRejectionError(rejection)
            return func(*args, **kwargs)

        return sync_wrapper
    return decorator


class PCIRejectionError(Exception):
    """Raised when PCI gate rejects an action."""
    
    def __init__(self, rejection: RejectionResponse):
        self.rejection = rejection
        super().__init__(
            f"PCI rejection: {rejection.code.name} at gate {rejection.gate}"
        )


__all__ = [
    "PCIGate",
    "GateResult",
    "get_pci_gate",
    "reset_pci_gate",
    "pci_protected",
    "PCIRejectionError",
]
