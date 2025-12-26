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

from core.pci.reject_codes import (
    RejectCode,
    RejectionResponse,
    VerificationGate,
    LatencyBudget,
)

from core.pci.envelope import (
    PCIEnvelope,
    Sender,
    Payload,
    Metadata,
    Signature,
    canonical_json,
    compute_digest,
    DOMAIN_PREFIX,
)

from core.pci.replay_guard import (
    ReplayGuard,
    get_replay_guard,
    reset_replay_guard,
)

from core.pci.gate import (
    PCIGate,
    GateResult,
    get_pci_gate,
    reset_pci_gate,
    pci_protected,
    PCIRejectionError,
)

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
