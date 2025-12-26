"""
BIZRA PCI Reject Codes
═══════════════════════════════════════════════════════════════════════════════
Unified rejection codes with stable numeric IDs for cross-language compatibility.

PROTOCOL.md Section 4: RejectCode Registry
Stable numeric IDs ensure consistent audit logging across Python, Rust, and APIs.

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, Optional


class RejectCode(IntEnum):
    """
    Stable numeric rejection codes per PROTOCOL.md Section 4.

    CRITICAL: These IDs are consensus-grade. Changing an existing ID
    requires a PROTOCOL.md version bump and migration plan.
    """

    # Success (not a rejection)
    SUCCESS = 0

    # Cheap tier rejections (<10ms)
    REJECT_SCHEMA = 1  # Envelope failed JSON schema validation
    REJECT_SIGNATURE = 2  # Cryptographic signature invalid
    REJECT_NONCE_REPLAY = 3  # Nonce already seen within TTL window
    REJECT_TIMESTAMP_STALE = 4  # Timestamp too far in past (>120s)
    REJECT_TIMESTAMP_FUTURE = 5  # Timestamp too far in future (>120s)
    REJECT_ROLE_VIOLATION = 11  # Agent attempted unauthorized action

    # Medium tier rejections (<150ms)
    REJECT_IHSAN_BELOW_MIN = 6  # Ihsān score < 0.95 threshold
    REJECT_SNR_BELOW_MIN = 7  # SNR score below tier threshold
    REJECT_POLICY_MISMATCH = 9  # policy_hash doesn't match constitution
    REJECT_STATE_MISMATCH = 10  # state_hash doesn't match expected

    # Expensive tier rejections (bounded)
    REJECT_FATE_VIOLATION = 13  # FATE invariant check failed
    REJECT_INVARIANT_FAILED = 14  # Formal invariant verification failed

    # Operational rejections
    REJECT_BUDGET_EXCEEDED = 8  # Verification latency exceeded budget
    REJECT_QUORUM_FAILED = 12  # Insufficient verifier signatures
    REJECT_RATE_LIMITED = 15  # Too many requests from sender

    # Catch-all (fail-closed)
    REJECT_INTERNAL_ERROR = 99  # Unexpected internal error

    @property
    def tier(self) -> str:
        """Return the verification tier where this rejection occurs."""
        if self in (
            RejectCode.REJECT_SCHEMA,
            RejectCode.REJECT_SIGNATURE,
            RejectCode.REJECT_NONCE_REPLAY,
            RejectCode.REJECT_TIMESTAMP_STALE,
            RejectCode.REJECT_TIMESTAMP_FUTURE,
            RejectCode.REJECT_ROLE_VIOLATION,
        ):
            return "CHEAP"
        elif self in (
            RejectCode.REJECT_IHSAN_BELOW_MIN,
            RejectCode.REJECT_SNR_BELOW_MIN,
            RejectCode.REJECT_POLICY_MISMATCH,
            RejectCode.REJECT_STATE_MISMATCH,
        ):
            return "MEDIUM"
        elif self in (
            RejectCode.REJECT_FATE_VIOLATION,
            RejectCode.REJECT_INVARIANT_FAILED,
        ):
            return "EXPENSIVE"
        else:
            return "OPERATIONAL"

    @property
    def is_retriable(self) -> bool:
        """Return whether this rejection can be retried after correction."""
        non_retriable = {
            RejectCode.REJECT_SIGNATURE,  # Key compromise, needs new key
            RejectCode.REJECT_NONCE_REPLAY,  # Must use new nonce
            RejectCode.REJECT_ROLE_VIOLATION,  # Architectural violation
        }
        return self not in non_retriable

    def message(self, details: Optional[str] = None) -> str:
        """Generate human-readable rejection message."""
        base_messages = {
            RejectCode.SUCCESS: "Operation completed successfully",
            RejectCode.REJECT_SCHEMA: "Envelope failed JSON schema validation",
            RejectCode.REJECT_SIGNATURE: "Cryptographic signature invalid",
            RejectCode.REJECT_NONCE_REPLAY: "Nonce already seen within TTL window",
            RejectCode.REJECT_TIMESTAMP_STALE: "Timestamp too far in past (>120s)",
            RejectCode.REJECT_TIMESTAMP_FUTURE: "Timestamp too far in future (>120s)",
            RejectCode.REJECT_IHSAN_BELOW_MIN: "Ihsān score below minimum threshold (0.95)",
            RejectCode.REJECT_SNR_BELOW_MIN: "SNR score below tier threshold",
            RejectCode.REJECT_BUDGET_EXCEEDED: "Verification latency exceeded tier budget",
            RejectCode.REJECT_POLICY_MISMATCH: "Policy hash doesn't match current constitution",
            RejectCode.REJECT_STATE_MISMATCH: "State hash doesn't match expected state",
            RejectCode.REJECT_ROLE_VIOLATION: "Agent attempted unauthorized action",
            RejectCode.REJECT_QUORUM_FAILED: "Insufficient verifier signatures for quorum",
            RejectCode.REJECT_FATE_VIOLATION: "FATE invariant verification failed",
            RejectCode.REJECT_INVARIANT_FAILED: "Formal invariant verification failed",
            RejectCode.REJECT_RATE_LIMITED: "Rate limit exceeded for sender",
            RejectCode.REJECT_INTERNAL_ERROR: "Unexpected internal error (fail-closed)",
        }
        base = base_messages.get(self, f"Unknown rejection code: {self.value}")
        return f"{base}: {details}" if details else base


@dataclass(frozen=True)
class RejectionResponse:
    """
    Structured rejection response per PROTOCOL.md Section 4.1.

    Immutable record of a verification rejection with full audit trail.
    """

    code: RejectCode
    envelope_digest: str
    timestamp: datetime
    gate: str
    tier: str
    latency_ms: float
    details: Dict[str, Any]

    @staticmethod
    def create(
        code: RejectCode,
        envelope_digest: str,
        gate: str,
        latency_ms: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> RejectionResponse:
        """Factory method for creating rejection responses."""
        return RejectionResponse(
            code=code,
            envelope_digest=envelope_digest,
            timestamp=datetime.now(timezone.utc),
            gate=gate,
            tier=code.tier,
            latency_ms=latency_ms,
            details=details or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON encoding."""
        return {
            "rejected": self.code != RejectCode.SUCCESS,
            "code": self.code.value,
            "name": self.code.name,
            "message": self.code.message(str(self.details) if self.details else None),
            "envelope_digest": self.envelope_digest,
            "timestamp": self.timestamp.isoformat(),
            "audit_trail": {
                "gate": self.gate,
                "tier": self.tier,
                "latency_ms": self.latency_ms,
                "details": self.details,
            },
        }


# Gate identifiers per PROTOCOL.md Section 5.1
class VerificationGate:
    """Verification gate identifiers for audit logging."""

    # Cheap tier gates (<10ms)
    SCHEMA = "SCHEMA"
    SIGNATURE = "SIGNATURE"
    TIMESTAMP = "TIMESTAMP"
    REPLAY = "REPLAY"
    ROLE = "ROLE"

    # Medium tier gates (<150ms)
    SNR = "SNR"
    IHSAN = "IHSAN"
    POLICY = "POLICY"

    # Expensive tier gates (bounded)
    FATE = "FATE"
    FORMAL = "FORMAL"

    # All gates in execution order
    CHEAP_GATES = [SCHEMA, SIGNATURE, TIMESTAMP, REPLAY, ROLE]
    MEDIUM_GATES = [SNR, IHSAN, POLICY]
    EXPENSIVE_GATES = [FATE, FORMAL]
    ALL_GATES = CHEAP_GATES + MEDIUM_GATES + EXPENSIVE_GATES


# Latency budgets per PROTOCOL.md Section 5.2
class LatencyBudget:
    """Latency budgets in milliseconds for each verification tier."""

    CHEAP_MS = 10.0
    MEDIUM_MS = 150.0
    EXPENSIVE_MS = 2000.0

    @classmethod
    def for_tier(cls, tier: str) -> float:
        """Get latency budget for a tier."""
        return {
            "CHEAP": cls.CHEAP_MS,
            "MEDIUM": cls.MEDIUM_MS,
            "EXPENSIVE": cls.EXPENSIVE_MS,
        }.get(tier.upper(), cls.EXPENSIVE_MS)


__all__ = [
    "RejectCode",
    "RejectionResponse",
    "VerificationGate",
    "LatencyBudget",
]
