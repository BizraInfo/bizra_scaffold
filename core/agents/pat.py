"""
BIZRA PAT Agent (Prover/Builder)
═══════════════════════════════════════════════════════════════════════════════
Constructs and signs PCI envelopes for proposals.

PROTOCOL.md Section 6.1: PAT Capabilities & Constraints
- CAN: Construct PCIEnvelope, sign with Ed25519, validate Ihsān before emission
- CANNOT: Commit to event log, issue CommitReceipt

Design Principles:
- Fail-closed: Reject proposals with Ihsān < 0.95 before signing
- Deterministic: Same inputs produce same envelope (except nonce/timestamp)
- Auditable: All operations logged with structured context

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric import ed25519

from core.pci.envelope import (
    Metadata,
    Payload,
    PCIEnvelope,
    Sender,
    compute_digest,
)
from core.pci.reject_codes import RejectCode, RejectionResponse

logger = logging.getLogger(__name__)


# Ihsān threshold per BIZRA_SOT.md Section 3.1
IHSAN_THRESHOLD = 0.95


@dataclass
class PATConfig:
    """Configuration for PAT agent."""

    agent_id: str
    ihsan_threshold: float = IHSAN_THRESHOLD
    auto_sign: bool = True  # Automatically sign on create
    validate_ihsan: bool = True  # Pre-validate Ihsān before signing


@dataclass
class ProposalResult:
    """Result of a proposal creation attempt."""

    success: bool
    envelope: Optional[PCIEnvelope] = None
    rejection: Optional[RejectionResponse] = None

    @property
    def digest(self) -> Optional[str]:
        """Get envelope digest if successful."""
        return self.envelope.digest() if self.envelope else None


class PATAgent:
    """
    PAT (Prover/Builder) Agent per PROTOCOL.md Section 6.1.

    Responsible for:
    - Constructing PCIEnvelope proposals
    - Signing envelopes with Ed25519 private key
    - Pre-validating Ihsān threshold before emission
    - Submitting proposals to SAT for verification

    Constraints:
    - CANNOT commit to event log
    - CANNOT issue CommitReceipt
    - MUST set sender.agent_type = "PAT"
    """

    def __init__(
        self,
        config: PATConfig,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
        ihsan_scorer: Optional[Callable[[Dict[str, Any]], float]] = None,
    ):
        """
        Initialize PAT agent.

        Args:
            config: Agent configuration
            private_key: Ed25519 private key for signing (generated if None)
            ihsan_scorer: Optional callback to compute Ihsān score from payload
        """
        self._config = config
        self._ihsan_scorer = ihsan_scorer

        # Generate or use provided key
        if private_key is None:
            self._private_key = ed25519.Ed25519PrivateKey.generate()
        else:
            self._private_key = private_key

        # Extract public key
        self._public_key = self._private_key.public_key()
        from cryptography.hazmat.primitives import serialization

        self._public_key_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        # Statistics
        self._proposals_created = 0
        self._proposals_rejected = 0

        logger.info(
            f"PAT agent initialized: agent_id={config.agent_id}, "
            f"public_key={self._public_key_bytes.hex()[:16]}..."
        )

    @property
    def agent_id(self) -> str:
        """Get agent identifier."""
        return self._config.agent_id

    @property
    def public_key_hex(self) -> str:
        """Get hex-encoded public key."""
        return self._public_key_bytes.hex()

    def _compute_ihsan_score(self, payload_data: Dict[str, Any]) -> float:
        """
        Compute Ihsān score for payload.

        Uses external scorer if provided, otherwise returns 0.0 (fail-closed).
        """
        if self._ihsan_scorer:
            try:
                return self._ihsan_scorer(payload_data)
            except Exception as e:
                logger.warning(f"Ihsān scorer failed: {e}, defaulting to 0.0")
                return 0.0
        return 0.0  # Fail-closed: unknown → lowest score

    def _validate_ihsan(
        self,
        ihsan_score: float,
        envelope_id: str,
    ) -> Optional[RejectionResponse]:
        """
        Validate Ihsān score against threshold.

        Returns RejectionResponse if below threshold, None if valid.
        """
        if ihsan_score < self._config.ihsan_threshold:
            return RejectionResponse.create(
                code=RejectCode.REJECT_IHSAN_BELOW_MIN,
                envelope_digest=f"pre-sign:{envelope_id}",
                gate="IHSAN",
                latency_ms=0.1,
                details={
                    "score": ihsan_score,
                    "threshold": self._config.ihsan_threshold,
                    "agent": "PAT",
                    "phase": "pre-sign",
                },
            )
        return None

    def create_proposal(
        self,
        action: str,
        data: Dict[str, Any],
        policy_hash: str,
        state_hash: Optional[str] = None,
        ihsan_score: Optional[float] = None,
        snr_score: Optional[float] = None,
        urgency: str = "BATCH",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ProposalResult:
        """
        Create and sign a proposal envelope.

        Args:
            action: Action type identifier
            data: Action-specific payload data
            policy_hash: BLAKE3 hash of current constitution
            state_hash: Optional BLAKE3 hash of current state
            ihsan_score: Pre-computed Ihsān score (computed if None)
            snr_score: Optional SNR score
            urgency: Processing urgency tier
            extra_metadata: Additional metadata fields

        Returns:
            ProposalResult with envelope if successful, rejection if not
        """
        import time

        start_time = time.perf_counter()

        # Compute Ihsān if not provided
        if ihsan_score is None:
            ihsan_score = self._compute_ihsan_score(data)

        # Pre-validate Ihsān (fail-closed)
        if self._config.validate_ihsan:
            temp_id = str(uuid.uuid4())
            rejection = self._validate_ihsan(ihsan_score, temp_id)
            if rejection:
                self._proposals_rejected += 1
                logger.warning(
                    f"PAT proposal rejected: ihsan={ihsan_score:.4f} < "
                    f"{self._config.ihsan_threshold:.4f}"
                )
                return ProposalResult(success=False, rejection=rejection)

        # Build sender
        sender = Sender(
            agent_type="PAT",
            agent_id=self._config.agent_id,
            public_key=self._public_key_bytes.hex(),
        )

        # Build payload
        payload = Payload(
            action=action,
            data=data,
            policy_hash=policy_hash,
            state_hash=state_hash,
        )

        # Build metadata
        metadata = Metadata(
            ihsan_score=ihsan_score,
            snr_score=snr_score,
            urgency=urgency,
            extra=extra_metadata or {},
        )

        # Create envelope
        envelope = PCIEnvelope.create(
            sender=sender,
            payload=payload,
            metadata=metadata,
        )

        # Sign if auto_sign enabled
        if self._config.auto_sign:
            envelope = envelope.sign(self._private_key)

        self._proposals_created += 1
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"PAT proposal created: envelope_id={envelope.envelope_id}, "
            f"action={action}, ihsan={ihsan_score:.4f}, "
            f"latency_ms={elapsed_ms:.2f}"
        )

        return ProposalResult(success=True, envelope=envelope)

    def sign_envelope(self, envelope: PCIEnvelope) -> PCIEnvelope:
        """
        Sign an unsigned envelope.

        Raises:
            ValueError: If envelope is already signed
        """
        if envelope.signature is not None:
            raise ValueError("Envelope is already signed")

        return envelope.sign(self._private_key)

    def stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self._config.agent_id,
            "agent_type": "PAT",
            "proposals_created": self._proposals_created,
            "proposals_rejected": self._proposals_rejected,
            "public_key": self._public_key_bytes.hex(),
        }


def create_pat_agent(
    agent_id: Optional[str] = None,
    ihsan_threshold: float = IHSAN_THRESHOLD,
    private_key: Optional[ed25519.Ed25519PrivateKey] = None,
    ihsan_scorer: Optional[Callable[[Dict[str, Any]], float]] = None,
) -> PATAgent:
    """
    Factory function to create a PAT agent.

    Args:
        agent_id: Unique agent identifier (generated if None)
        ihsan_threshold: Minimum Ihsān score for proposals
        private_key: Ed25519 private key (generated if None)
        ihsan_scorer: Optional callback to compute Ihsān score

    Returns:
        Configured PATAgent instance
    """
    if agent_id is None:
        agent_id = f"pat-{uuid.uuid4().hex[:8]}"

    config = PATConfig(
        agent_id=agent_id,
        ihsan_threshold=ihsan_threshold,
    )

    return PATAgent(
        config=config,
        private_key=private_key,
        ihsan_scorer=ihsan_scorer,
    )


__all__ = [
    "PATAgent",
    "PATConfig",
    "ProposalResult",
    "create_pat_agent",
    "IHSAN_THRESHOLD",
]
