"""
BIZRA PCI Module
═══════════════════════════════════════════════════════════════════════════════
Proof-Carrying Inference (PCI) protocol implementation.

This module implements the dual-agent verification protocol per PROTOCOL.md:
- PCIEnvelope: Canonical JSON wire format with domain-separated digest
- ReplayGuard: Nonce + timestamp replay resistance
- VerifierStack: Tiered verification chain (cheap → medium → expensive)
- RejectCodes: Unified rejection codes with stable numeric IDs
- PCIGate: Action boundary verification

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from core.pci.envelope import (
    DOMAIN_PREFIX,
    Metadata,
    Payload,
    PCIEnvelope,
    Sender,
    Signature,
    canonical_json,
    compute_digest,
)
from core.pci.reject_codes import (
    LatencyBudget,
    RejectCode,
    RejectionResponse,
    VerificationGate,
)
from core.pci.replay_guard import ReplayGuard, get_replay_guard, reset_replay_guard

# NOTE: Gate imports are deferred to avoid circular imports with core.agents
# Import gate module only when needed, not at module load time


def __getattr__(name: str):
    """Lazy import of gate module to avoid circular import with core.agents."""
    gate_exports = {
        "GateResult",
        "PCIGate",
        "PCIRejectionError",
        "get_pci_gate",
        "pci_protected",
        "reset_pci_gate",
    }
    if name in gate_exports:
        from core.pci import gate

        return getattr(gate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Reject codes
    "RejectCode",
    "RejectionResponse",
    "VerificationGate",
    "LatencyBudget",
    # Envelope
    "PCIEnvelope",
    "Sender",
    "Payload",
    "Metadata",
    "Signature",
    "canonical_json",
    "compute_digest",
    "DOMAIN_PREFIX",
    # Replay guard
    "ReplayGuard",
    "get_replay_guard",
    "reset_replay_guard",
    # Gate
    "PCIGate",
    "GateResult",
    "get_pci_gate",
    "reset_pci_gate",
    "pci_protected",
    "PCIRejectionError",
]
