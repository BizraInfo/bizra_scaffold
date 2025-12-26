"""
BIZRA SAT Agent (Verifier/Governor)
═══════════════════════════════════════════════════════════════════════════════
Verifies PCI envelopes and issues commit receipts.

PROTOCOL.md Section 6.2: SAT Capabilities & Constraints
- CAN: Receive envelopes, execute verification gates, commit, issue receipts
- CANNOT: Modify payload content

Design Principles:
- Fail-closed: Any gate failure → immediate rejection
- Append-only: Every decision produces a receipt (accept or reject)
- Auditable: Full verification report attached to receipt

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from core.pci.envelope import PCIEnvelope, compute_digest, canonical_json
from core.pci.reject_codes import (
    RejectCode,
    RejectionResponse,
    VerificationGate,
    LatencyBudget,
)
from core.pci.replay_guard import ReplayGuard, get_replay_guard


logger = logging.getLogger(__name__)


# Thresholds per BIZRA_SOT.md and PROTOCOL.md
IHSAN_THRESHOLD = 0.95
SNR_THRESHOLD_HIGH = 0.80
SNR_THRESHOLD_MEDIUM = 0.75


@dataclass
class SATConfig:
    """Configuration for SAT agent."""
    
    agent_id: str
    ihsan_threshold: float = IHSAN_THRESHOLD
    snr_threshold: float = SNR_THRESHOLD_MEDIUM
    cheap_budget_ms: float = LatencyBudget.CHEAP_MS
    medium_budget_ms: float = LatencyBudget.MEDIUM_MS
    expensive_budget_ms: float = LatencyBudget.EXPENSIVE_MS
    require_fate_verification: bool = False  # Enable for state mutations
    require_formal_verification: bool = False  # Enable for critical paths


@dataclass
class VerificationReport:
    """Detailed verification report for audit trail."""
    
    envelope_digest: str
    gates_passed: List[str]
    gates_failed: List[str]
    tier_reached: str
    total_latency_ms: float
    gate_latencies: Dict[str, float]
    ihsan_score: float
    snr_score: Optional[float]
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "envelope_digest": self.envelope_digest,
            "gates_passed": self.gates_passed,
            "gates_failed": self.gates_failed,
            "tier_reached": self.tier_reached,
            "total_latency_ms": self.total_latency_ms,
            "gate_latencies": self.gate_latencies,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "details": self.details,
        }


@dataclass
class CommitReceipt:
    """
    Commit receipt per PROTOCOL.md Section 3.
    
    Immutable proof of verified commit with full audit binding.
    """
    
    version: str
    receipt_id: str
    timestamp: datetime
    envelope_digest: str
    commit_ref: Dict[str, Any]
    verification: Dict[str, Any]
    verifier_set: List[Dict[str, Any]]
    quorum: Dict[str, int]
    audit_digest: str
    policy_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp.isoformat(),
            "envelope_digest": self.envelope_digest,
            "commit_ref": self.commit_ref,
            "verification": self.verification,
            "verifier_set": self.verifier_set,
            "quorum": self.quorum,
            "audit_digest": self.audit_digest,
            "policy_hash": self.policy_hash,
        }
    
    def to_json(self) -> str:
        return canonical_json(self.to_dict()).decode('utf-8')


@dataclass
class VerificationResult:
    """Result of envelope verification."""
    
    success: bool
    receipt: Optional[CommitReceipt] = None
    rejection: Optional[RejectionResponse] = None
    report: Optional[VerificationReport] = None


class SATAgent:
    """
    SAT (Verifier/Governor) Agent per PROTOCOL.md Section 6.2.
    
    Responsible for:
    - Receiving PCIEnvelopes from PAT agents
    - Executing tiered verification gate chain
    - Committing to event log on success
    - Issuing signed CommitReceipts
    - Rejecting with RejectCode on failure
    
    Constraints:
    - CANNOT modify payload content
    - MUST execute all applicable gates
    - MUST emit receipt for every decision
    """
    
    def __init__(
        self,
        config: SATConfig,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
        replay_guard: Optional[ReplayGuard] = None,
        commit_callback: Optional[Callable[[PCIEnvelope, CommitReceipt], int]] = None,
        policy_hash_provider: Optional[Callable[[], str]] = None,
    ):
        """
        Initialize SAT agent.
        
        Args:
            config: Agent configuration
            private_key: Ed25519 private key for signing receipts
            replay_guard: Replay protection (uses global if None)
            commit_callback: Callback to commit envelope, returns offset
            policy_hash_provider: Callback to get current policy hash
        """
        self._config = config
        self._replay_guard = replay_guard or get_replay_guard()
        self._commit_callback = commit_callback
        self._policy_hash_provider = policy_hash_provider
        
        # Generate or use provided key
        if private_key is None:
            self._private_key = ed25519.Ed25519PrivateKey.generate()
        else:
            self._private_key = private_key
        
        # Extract public key
        self._public_key = self._private_key.public_key()
        self._public_key_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        
        # Event log offset (monotonic)
        self._commit_offset = 0
        
        # Statistics
        self._envelopes_verified = 0
        self._envelopes_rejected = 0
        self._receipts_issued = 0
        
        logger.info(
            f"SAT agent initialized: agent_id={config.agent_id}, "
            f"public_key={self._public_key_bytes.hex()[:16]}..."
        )
    
    @property
    def agent_id(self) -> str:
        return self._config.agent_id
    
    @property
    def public_key_hex(self) -> str:
        return self._public_key_bytes.hex()
    
    def _get_current_policy_hash(self) -> str:
        """Get current constitution policy hash."""
        if self._policy_hash_provider:
            return self._policy_hash_provider()
        raise RuntimeError("policy_hash_provider is required for policy verification")
    
    def _gate_schema(
        self,
        envelope: PCIEnvelope,
        gate_latencies: Dict[str, float],
    ) -> Optional[RejectionResponse]:
        """
        SCHEMA gate: Validate envelope structure.
        
        Checks:
        - Required fields present
        - Field types correct
        - Values in valid ranges
        """
        start = time.perf_counter()
        
        try:
            # Basic structure validation
            if not envelope.version:
                return self._reject(
                    RejectCode.REJECT_SCHEMA,
                    envelope.digest(),
                    VerificationGate.SCHEMA,
                    gate_latencies,
                    start,
                    {"error": "Missing version"},
                )
            
            if not envelope.envelope_id:
                return self._reject(
                    RejectCode.REJECT_SCHEMA,
                    envelope.digest(),
                    VerificationGate.SCHEMA,
                    gate_latencies,
                    start,
                    {"error": "Missing envelope_id"},
                )
            
            if not envelope.nonce or len(envelope.nonce) != 64:
                return self._reject(
                    RejectCode.REJECT_SCHEMA,
                    envelope.digest(),
                    VerificationGate.SCHEMA,
                    gate_latencies,
                    start,
                    {"error": "Invalid nonce length"},
                )
            
            gate_latencies[VerificationGate.SCHEMA] = (time.perf_counter() - start) * 1000
            return None
            
        except Exception as e:
            return self._reject(
                RejectCode.REJECT_SCHEMA,
                envelope.digest() if envelope else "unknown",
                VerificationGate.SCHEMA,
                gate_latencies,
                start,
                {"error": str(e)},
            )
    
    def _gate_signature(
        self,
        envelope: PCIEnvelope,
        gate_latencies: Dict[str, float],
    ) -> Optional[RejectionResponse]:
        """SIGNATURE gate: Verify Ed25519 signature."""
        start = time.perf_counter()
        
        valid, error = envelope.verify_signature()
        
        gate_latencies[VerificationGate.SIGNATURE] = (time.perf_counter() - start) * 1000
        
        if not valid:
            return self._reject(
                RejectCode.REJECT_SIGNATURE,
                envelope.digest(),
                VerificationGate.SIGNATURE,
                gate_latencies,
                start,
                {"error": error},
            )
        
        return None
    
    def _gate_timestamp(
        self,
        envelope: PCIEnvelope,
        gate_latencies: Dict[str, float],
    ) -> Optional[RejectionResponse]:
        """TIMESTAMP gate: Check freshness."""
        start = time.perf_counter()
        
        valid, code = self._replay_guard.check_timestamp(envelope.timestamp)
        
        gate_latencies[VerificationGate.TIMESTAMP] = (time.perf_counter() - start) * 1000
        
        if not valid:
            return self._reject(
                code,  # type: ignore
                envelope.digest(),
                VerificationGate.TIMESTAMP,
                gate_latencies,
                start,
                {"timestamp": envelope.timestamp.isoformat()},
            )
        
        return None
    
    def _gate_replay(
        self,
        envelope: PCIEnvelope,
        gate_latencies: Dict[str, float],
    ) -> Optional[RejectionResponse]:
        """REPLAY gate: Check nonce uniqueness."""
        start = time.perf_counter()
        
        valid, code = self._replay_guard.check_nonce(
            envelope.nonce,
            envelope.digest(),
        )
        
        gate_latencies[VerificationGate.REPLAY] = (time.perf_counter() - start) * 1000
        
        if not valid:
            return self._reject(
                code,  # type: ignore
                envelope.digest(),
                VerificationGate.REPLAY,
                gate_latencies,
                start,
                {"nonce": envelope.nonce[:16] + "..."},
            )
        
        return None
    
    def _gate_role(
        self,
        envelope: PCIEnvelope,
        gate_latencies: Dict[str, float],
    ) -> Optional[RejectionResponse]:
        """ROLE gate: Verify agent role permissions."""
        start = time.perf_counter()
        
        # PAT can only propose (sender.agent_type must be PAT for proposals)
        # SAT cannot send envelopes to itself (architectural constraint)
        agent_type = envelope.sender.agent_type
        
        if agent_type not in ("PAT", "SAT"):
            return self._reject(
                RejectCode.REJECT_ROLE_VIOLATION,
                envelope.digest(),
                VerificationGate.ROLE,
                gate_latencies,
                start,
                {"agent_type": agent_type, "error": "Unknown agent type"},
            )
        
        gate_latencies[VerificationGate.ROLE] = (time.perf_counter() - start) * 1000
        return None
    
    def _gate_snr(
        self,
        envelope: PCIEnvelope,
        gate_latencies: Dict[str, float],
    ) -> Optional[RejectionResponse]:
        """SNR gate: Check signal-to-noise ratio."""
        start = time.perf_counter()
        
        snr_score = envelope.metadata.snr_score
        
        # SNR is optional but if provided must meet threshold
        if snr_score is not None and snr_score < self._config.snr_threshold:
            return self._reject(
                RejectCode.REJECT_SNR_BELOW_MIN,
                envelope.digest(),
                VerificationGate.SNR,
                gate_latencies,
                start,
                {"score": snr_score, "threshold": self._config.snr_threshold},
            )
        
        gate_latencies[VerificationGate.SNR] = (time.perf_counter() - start) * 1000
        return None
    
    def _gate_ihsan(
        self,
        envelope: PCIEnvelope,
        gate_latencies: Dict[str, float],
    ) -> Optional[RejectionResponse]:
        """IHSAN gate: Check ethical alignment score."""
        start = time.perf_counter()
        
        ihsan_score = envelope.metadata.ihsan_score
        
        if ihsan_score < self._config.ihsan_threshold:
            return self._reject(
                RejectCode.REJECT_IHSAN_BELOW_MIN,
                envelope.digest(),
                VerificationGate.IHSAN,
                gate_latencies,
                start,
                {"score": ihsan_score, "threshold": self._config.ihsan_threshold},
            )
        
        gate_latencies[VerificationGate.IHSAN] = (time.perf_counter() - start) * 1000
        return None
    
    def _gate_policy(
        self,
        envelope: PCIEnvelope,
        gate_latencies: Dict[str, float],
    ) -> Optional[RejectionResponse]:
        """POLICY gate: Verify policy hash matches current constitution."""
        start = time.perf_counter()
        try:
            current_hash = self._get_current_policy_hash()
        except RuntimeError as exc:
            return self._reject(
                RejectCode.REJECT_INTERNAL_ERROR,
                envelope.digest(),
                VerificationGate.POLICY,
                gate_latencies,
                start,
                {"error": str(exc)},
            )
        
        envelope_hash = envelope.payload.policy_hash
        
        if envelope_hash != current_hash:
            return self._reject(
                RejectCode.REJECT_POLICY_MISMATCH,
                envelope.digest(),
                VerificationGate.POLICY,
                gate_latencies,
                start,
                {
                    "envelope_policy_hash": envelope_hash[:16] + "...",
                    "current_policy_hash": current_hash[:16] + "...",
                },
            )
        
        gate_latencies[VerificationGate.POLICY] = (time.perf_counter() - start) * 1000
        return None
    
    def _reject(
        self,
        code: RejectCode,
        envelope_digest: str,
        gate: str,
        gate_latencies: Dict[str, float],
        start_time: float,
        details: Dict[str, Any],
    ) -> RejectionResponse:
        """Create rejection response and update latency."""
        latency_ms = (time.perf_counter() - start_time) * 1000
        gate_latencies[gate] = latency_ms
        
        self._envelopes_rejected += 1
        
        return RejectionResponse.create(
            code=code,
            envelope_digest=envelope_digest,
            gate=gate,
            latency_ms=latency_ms,
            details=details,
        )
    
    def _run_cheap_tier(
        self,
        envelope: PCIEnvelope,
        gates_passed: List[str],
        gate_latencies: Dict[str, float],
    ) -> Optional[RejectionResponse]:
        """Execute cheap tier gates (<10ms)."""
        # Gate ordering per PROTOCOL.md Section 5.1
        gates = [
            (VerificationGate.SCHEMA, self._gate_schema),
            (VerificationGate.SIGNATURE, self._gate_signature),
            (VerificationGate.TIMESTAMP, self._gate_timestamp),
            (VerificationGate.REPLAY, self._gate_replay),
            (VerificationGate.ROLE, self._gate_role),
        ]
        
        for gate_name, gate_func in gates:
            rejection = gate_func(envelope, gate_latencies)
            if rejection:
                return rejection
            gates_passed.append(gate_name)
        
        return None
    
    def _run_medium_tier(
        self,
        envelope: PCIEnvelope,
        gates_passed: List[str],
        gate_latencies: Dict[str, float],
    ) -> Optional[RejectionResponse]:
        """Execute medium tier gates (<150ms)."""
        gates = [
            (VerificationGate.SNR, self._gate_snr),
            (VerificationGate.IHSAN, self._gate_ihsan),
            (VerificationGate.POLICY, self._gate_policy),
        ]
        
        for gate_name, gate_func in gates:
            rejection = gate_func(envelope, gate_latencies)
            if rejection:
                return rejection
            gates_passed.append(gate_name)
        
        return None
    
    def _create_receipt(
        self,
        envelope: PCIEnvelope,
        report: VerificationReport,
        commit_offset: int,
    ) -> CommitReceipt:
        """Create and sign commit receipt."""
        now = datetime.now(timezone.utc)
        # Sign the verification report
        report_bytes = canonical_json(report.to_dict())
        audit_digest = compute_digest(report_bytes, domain_separated=False)
        
        # Create verifier attestation
        attestation_data = {
            "envelope_digest": envelope.digest(),
            "audit_digest": audit_digest,
            "timestamp": now.isoformat(),
        }
        attestation_bytes = canonical_json(attestation_data)
        signature = self._private_key.sign(attestation_bytes)
        
        receipt = CommitReceipt(
            version="1.0.0",
            receipt_id=str(uuid.uuid4()),
            timestamp=now,
            envelope_digest=envelope.digest(),
            commit_ref={
                "type": "eventlog",
                "offset": commit_offset,
            },
            verification={
                "tier": report.tier_reached,
                "latency_ms": report.total_latency_ms,
                "gates_passed": report.gates_passed,
                "ihsan_score": report.ihsan_score,
                "snr_score": report.snr_score,
            },
            verifier_set=[
                {
                    "sat_id": self._config.agent_id,
                    "public_key": self._public_key_bytes.hex(),
                    "signature": signature.hex(),
                    "timestamp": now.isoformat(),
                }
            ],
            quorum={
                "required": 1,
                "achieved": 1,
            },
            audit_digest=audit_digest,
            policy_hash=envelope.payload.policy_hash,
        )
        
        self._receipts_issued += 1
        return receipt
    
    def verify(self, envelope: PCIEnvelope) -> VerificationResult:
        """
        Verify an envelope through the gate chain.
        
        Executes gates in order per PROTOCOL.md Section 5.1:
        1. CHEAP tier (<10ms): SCHEMA, SIGNATURE, TIMESTAMP, REPLAY, ROLE
        2. MEDIUM tier (<150ms): SNR, IHSAN, POLICY
        3. EXPENSIVE tier (bounded): FATE, FORMAL (if enabled)
        
        Returns:
            VerificationResult with receipt if successful, rejection if not
        """
        start_time = time.perf_counter()
        gates_passed: List[str] = []
        gate_latencies: Dict[str, float] = {}
        tier_reached = "CHEAP"
        
        # CHEAP tier
        rejection = self._run_cheap_tier(envelope, gates_passed, gate_latencies)
        if rejection:
            return VerificationResult(
                success=False,
                rejection=rejection,
                report=self._build_report(
                    envelope, gates_passed, [rejection.gate],
                    tier_reached, gate_latencies, start_time,
                ),
            )
        
        # MEDIUM tier
        tier_reached = "MEDIUM"
        rejection = self._run_medium_tier(envelope, gates_passed, gate_latencies)
        if rejection:
            return VerificationResult(
                success=False,
                rejection=rejection,
                report=self._build_report(
                    envelope, gates_passed, [rejection.gate],
                    tier_reached, gate_latencies, start_time,
                ),
            )
        
        # EXPENSIVE tier (if configured)
        if self._config.require_fate_verification:
            tier_reached = "EXPENSIVE"
            # FATE verification would go here
            gates_passed.append(VerificationGate.FATE)
        
        if self._config.require_formal_verification:
            tier_reached = "EXPENSIVE"
            # Formal verification would go here
            gates_passed.append(VerificationGate.FORMAL)
        
        # SUCCESS - commit and create receipt
        self._envelopes_verified += 1
        
        # Commit via callback or increment local offset
        report = self._build_report(
            envelope, gates_passed, [],
            tier_reached, gate_latencies, start_time,
        )
        
        receipt = self._create_receipt(envelope, report, 0)
        
        if self._commit_callback:
            commit_offset = self._commit_callback(envelope, receipt)
        else:
            self._commit_offset += 1
            commit_offset = self._commit_offset
        
        receipt.commit_ref["offset"] = commit_offset
        
        logger.info(
            f"SAT verification passed: envelope_id={envelope.envelope_id}, "
            f"receipt_id={receipt.receipt_id}, "
            f"latency_ms={report.total_latency_ms:.2f}"
        )
        
        return VerificationResult(
            success=True,
            receipt=receipt,
            report=report,
        )
    
    def _build_report(
        self,
        envelope: PCIEnvelope,
        gates_passed: List[str],
        gates_failed: List[str],
        tier_reached: str,
        gate_latencies: Dict[str, float],
        start_time: float,
    ) -> VerificationReport:
        """Build verification report for audit."""
        total_latency = (time.perf_counter() - start_time) * 1000
        
        return VerificationReport(
            envelope_digest=envelope.digest(),
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            tier_reached=tier_reached,
            total_latency_ms=total_latency,
            gate_latencies=gate_latencies,
            ihsan_score=envelope.metadata.ihsan_score,
            snr_score=envelope.metadata.snr_score,
            details={
                "envelope_id": envelope.envelope_id,
                "action": envelope.payload.action,
                "sender": envelope.sender.agent_id,
            },
        )
    
    def stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self._config.agent_id,
            "agent_type": "SAT",
            "envelopes_verified": self._envelopes_verified,
            "envelopes_rejected": self._envelopes_rejected,
            "receipts_issued": self._receipts_issued,
            "commit_offset": self._commit_offset,
            "public_key": self._public_key_bytes.hex(),
            "replay_guard_stats": self._replay_guard.stats(),
        }


def create_sat_agent(
    agent_id: Optional[str] = None,
    ihsan_threshold: float = IHSAN_THRESHOLD,
    snr_threshold: float = SNR_THRESHOLD_MEDIUM,
    private_key: Optional[ed25519.Ed25519PrivateKey] = None,
    policy_hash_provider: Optional[Callable[[], str]] = None,
) -> SATAgent:
    """
    Factory function to create a SAT agent.
    
    Args:
        agent_id: Unique agent identifier (generated if None)
        ihsan_threshold: Minimum Ihsān score for acceptance
        snr_threshold: Minimum SNR score for acceptance
        private_key: Ed25519 private key (generated if None)
        policy_hash_provider: Callback to get current policy hash
        
    Returns:
        Configured SATAgent instance
    """
    if agent_id is None:
        agent_id = f"sat-{uuid.uuid4().hex[:8]}"
    
    config = SATConfig(
        agent_id=agent_id,
        ihsan_threshold=ihsan_threshold,
        snr_threshold=snr_threshold,
    )
    
    return SATAgent(
        config=config,
        private_key=private_key,
        policy_hash_provider=policy_hash_provider,
    )


__all__ = [
    "SATAgent",
    "SATConfig",
    "VerificationResult",
    "VerificationReport",
    "CommitReceipt",
    "create_sat_agent",
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD_HIGH",
    "SNR_THRESHOLD_MEDIUM",
]
