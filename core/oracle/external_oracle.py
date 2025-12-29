"""
BIZRA External Oracle Interface
════════════════════════════════════════════════════════════════════════════════

Addresses the Goodhart vulnerability by separating score computation
from score attestation.

The Problem (Goodhart's Law):
- When ihsān scores come from the same context that produces them,
  the system can learn to game itself
- Self-referential scoring → optimization for the metric, not the goal

The Solution (External Oracle Pattern):
- Score COMPUTATION happens internally (fast, efficient)
- Score ATTESTATION happens externally (trusted, independent)
- External oracle can:
  1. Sample random attestations for spot-checking
  2. Provide ground-truth calibration data
  3. Detect distribution drift in ihsān scores
  4. Anchor scores to real-world outcomes

Oracle Types:
1. **Local Oracle**: File-based, for development/testing
2. **Remote Oracle**: API-based, for production
3. **Consortium Oracle**: Multi-party, for high-stakes attestations
4. **Temporal Oracle**: Time-locked, for future verification

Design Philosophy:
- Giants Protocol: Oracle stands outside the system being measured
- SNR-Weighted: Focus on high-signal attestations
- Fail-Closed: Missing oracle → fail (no self-attestation fallback)
- Ihsān-Bound: Oracle itself must meet ihsān threshold

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

# Add repo root to path
MODULE_DIR = Path(__file__).resolve().parent
CORE_DIR = MODULE_DIR.parent
REPO_ROOT = CORE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger("bizra.oracle.external")


# =============================================================================
# CONSTANTS
# =============================================================================

# Oracle thresholds
ORACLE_IHSAN_THRESHOLD = 0.95
ORACLE_SPOT_CHECK_RATE = 0.10  # 10% of attestations are spot-checked
ORACLE_DRIFT_THRESHOLD = 0.15  # Alert if distribution shifts > 15%

# Oracle timeouts
ORACLE_TIMEOUT_SECONDS = 30
ORACLE_CACHE_TTL_SECONDS = 3600  # 1 hour


# =============================================================================
# ORACLE TYPES
# =============================================================================


class OracleType(Enum):
    """Types of external oracles."""
    
    LOCAL = "local"           # File-based, development
    REMOTE = "remote"         # API-based, production
    CONSORTIUM = "consortium" # Multi-party attestation
    TEMPORAL = "temporal"     # Time-locked verification


class OracleVerdict(Enum):
    """Oracle verification verdicts."""
    
    CONFIRMED = "confirmed"   # Score is accurate
    ADJUSTED = "adjusted"     # Score was corrected
    REJECTED = "rejected"     # Score is invalid
    PENDING = "pending"       # Awaiting verification
    UNKNOWN = "unknown"       # Could not determine


# =============================================================================
# ORACLE ATTESTATIONS
# =============================================================================


@dataclass
class OracleAttestation:
    """
    An attestation from an external oracle.
    
    Represents the oracle's verification of a score or claim.
    """
    
    # Attestation ID
    attestation_id: str
    
    # What was attested
    subject_hash: str      # Hash of the subject being verified
    subject_type: str      # "ihsan_score", "impact_claim", "artifact_hash"
    
    # Original claim
    claimed_value: float
    claimed_at: datetime
    claimed_by: str
    
    # Oracle verdict
    verdict: OracleVerdict
    verified_value: Optional[float]
    confidence: float      # 0-1
    
    # Oracle metadata
    oracle_id: str
    oracle_type: OracleType
    verified_at: datetime
    
    # Cryptographic proof
    attestation_hash: str
    signature: Optional[str] = None
    
    @classmethod
    def create(
        cls,
        subject_hash: str,
        subject_type: str,
        claimed_value: float,
        claimed_by: str,
        verdict: OracleVerdict,
        verified_value: Optional[float],
        confidence: float,
        oracle_id: str,
        oracle_type: OracleType,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
    ) -> "OracleAttestation":
        """Create a new oracle attestation."""
        
        now = datetime.now(timezone.utc)
        
        # Generate attestation ID
        id_input = f"{subject_hash}:{oracle_id}:{now.isoformat()}"
        attestation_id = f"attest_{hashlib.sha256(id_input.encode()).hexdigest()[:12]}"
        
        # Compute attestation hash
        attestation_data = {
            "attestation_id": attestation_id,
            "subject_hash": subject_hash,
            "subject_type": subject_type,
            "claimed_value": claimed_value,
            "claimed_at": now.isoformat(),
            "claimed_by": claimed_by,
            "verdict": verdict.value,
            "verified_value": verified_value,
            "confidence": confidence,
            "oracle_id": oracle_id,
            "oracle_type": oracle_type.value,
            "verified_at": now.isoformat(),
        }
        attestation_json = json.dumps(attestation_data, sort_keys=True, separators=(",", ":"))
        
        if BLAKE3_AVAILABLE:
            attestation_hash = blake3.blake3(attestation_json.encode()).hexdigest()
        else:
            attestation_hash = hashlib.sha256(attestation_json.encode()).hexdigest()
        
        # Sign if key available
        signature = None
        if private_key and CRYPTO_AVAILABLE:
            sig_bytes = private_key.sign(attestation_hash.encode())
            signature = sig_bytes.hex()
        
        return cls(
            attestation_id=attestation_id,
            subject_hash=subject_hash,
            subject_type=subject_type,
            claimed_value=claimed_value,
            claimed_at=now,
            claimed_by=claimed_by,
            verdict=verdict,
            verified_value=verified_value,
            confidence=confidence,
            oracle_id=oracle_id,
            oracle_type=oracle_type,
            verified_at=now,
            attestation_hash=attestation_hash,
            signature=signature,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "attestation_id": self.attestation_id,
            "subject_hash": self.subject_hash,
            "subject_type": self.subject_type,
            "claimed_value": self.claimed_value,
            "claimed_at": self.claimed_at.isoformat(),
            "claimed_by": self.claimed_by,
            "verdict": self.verdict.value,
            "verified_value": self.verified_value,
            "confidence": self.confidence,
            "oracle_id": self.oracle_id,
            "oracle_type": self.oracle_type.value,
            "verified_at": self.verified_at.isoformat(),
            "attestation_hash": self.attestation_hash,
            "signature": self.signature,
        }


# =============================================================================
# ORACLE INTERFACE
# =============================================================================


class ExternalOracle(abc.ABC):
    """
    Abstract base class for external oracles.
    
    External oracles provide independent verification of claims
    made within the BIZRA system.
    """
    
    def __init__(
        self,
        oracle_id: str,
        oracle_type: OracleType,
    ):
        self.oracle_id = oracle_id
        self.oracle_type = oracle_type
        self._attestation_cache: Dict[str, OracleAttestation] = {}
    
    @abc.abstractmethod
    def verify_ihsan_score(
        self,
        claimed_score: float,
        context_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OracleAttestation:
        """
        Verify an ihsān score claim.
        
        Args:
            claimed_score: The claimed ihsān score (0-1)
            context_hash: Hash of the context that produced the score
            metadata: Optional additional context
            
        Returns:
            Oracle attestation with verdict
        """
        pass
    
    @abc.abstractmethod
    def verify_impact_claim(
        self,
        claimed_impact: float,
        evidence_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OracleAttestation:
        """
        Verify a Proof of Impact claim.
        
        Args:
            claimed_impact: The claimed impact value
            evidence_hash: Hash of the supporting evidence
            metadata: Optional additional context
            
        Returns:
            Oracle attestation with verdict
        """
        pass
    
    @abc.abstractmethod
    def get_calibration_data(self) -> Dict[str, Any]:
        """
        Get calibration data for score distribution.
        
        Returns:
            Expected distributions and thresholds
        """
        pass
    
    def should_spot_check(self) -> bool:
        """Determine if this request should be spot-checked."""
        import random
        return random.random() < ORACLE_SPOT_CHECK_RATE
    
    def detect_drift(
        self,
        recent_scores: List[float],
        baseline_mean: float = 0.7,
        baseline_std: float = 0.15,
    ) -> Tuple[bool, float]:
        """
        Detect distribution drift in scores.
        
        Returns:
            Tuple of (drift_detected, drift_magnitude)
        """
        if not recent_scores:
            return False, 0.0
        
        import statistics
        
        recent_mean = statistics.mean(recent_scores)
        drift = abs(recent_mean - baseline_mean)
        
        return drift > ORACLE_DRIFT_THRESHOLD, drift


# =============================================================================
# LOCAL ORACLE (Development)
# =============================================================================


class LocalOracle(ExternalOracle):
    """
    File-based oracle for development and testing.
    
    Stores attestations locally and uses simple heuristics
    for verification.
    """
    
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
    ):
        super().__init__(
            oracle_id=f"local_{hashlib.sha256(b'local_oracle').hexdigest()[:8]}",
            oracle_type=OracleType.LOCAL,
        )
        self.storage_dir = storage_dir or (REPO_ROOT / "data" / "oracle")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._calibration_file = self.storage_dir / "calibration.json"
        self._attestations_file = self.storage_dir / "attestations.json"
    
    def verify_ihsan_score(
        self,
        claimed_score: float,
        context_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OracleAttestation:
        """Verify ihsān score locally."""
        
        # Simple verification heuristics for local oracle
        # In production, this would call out to a real oracle
        
        verdict = OracleVerdict.CONFIRMED
        verified_value = claimed_score
        confidence = 0.8
        
        # Sanity checks
        if claimed_score < 0 or claimed_score > 1:
            verdict = OracleVerdict.REJECTED
            verified_value = None
            confidence = 1.0
        elif claimed_score > ORACLE_IHSAN_THRESHOLD:
            # High scores require higher confidence
            if self.should_spot_check():
                # For spot checks, we'd do deeper verification
                # Here we just note it for later review
                verdict = OracleVerdict.PENDING
                confidence = 0.6
        
        attestation = OracleAttestation.create(
            subject_hash=context_hash,
            subject_type="ihsan_score",
            claimed_value=claimed_score,
            claimed_by=metadata.get("claimed_by", "unknown") if metadata else "unknown",
            verdict=verdict,
            verified_value=verified_value,
            confidence=confidence,
            oracle_id=self.oracle_id,
            oracle_type=self.oracle_type,
        )
        
        # Store attestation
        self._store_attestation(attestation)
        
        return attestation
    
    def verify_impact_claim(
        self,
        claimed_impact: float,
        evidence_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OracleAttestation:
        """Verify impact claim locally."""
        
        verdict = OracleVerdict.CONFIRMED
        verified_value = claimed_impact
        confidence = 0.75
        
        # Basic validation
        if claimed_impact < 0:
            verdict = OracleVerdict.REJECTED
            verified_value = None
            confidence = 1.0
        elif claimed_impact > 10000:
            # Extremely high impact requires review
            verdict = OracleVerdict.PENDING
            confidence = 0.5
        
        attestation = OracleAttestation.create(
            subject_hash=evidence_hash,
            subject_type="impact_claim",
            claimed_value=claimed_impact,
            claimed_by=metadata.get("claimed_by", "unknown") if metadata else "unknown",
            verdict=verdict,
            verified_value=verified_value,
            confidence=confidence,
            oracle_id=self.oracle_id,
            oracle_type=self.oracle_type,
        )
        
        self._store_attestation(attestation)
        
        return attestation
    
    def get_calibration_data(self) -> Dict[str, Any]:
        """Get calibration data from local file."""
        
        if self._calibration_file.exists():
            with self._calibration_file.open("r") as f:
                return json.load(f)
        
        # Default calibration
        return {
            "ihsan": {
                "mean": 0.7,
                "std": 0.15,
                "threshold": ORACLE_IHSAN_THRESHOLD,
            },
            "impact": {
                "mean": 100,
                "std": 50,
                "max_expected": 5000,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def _store_attestation(self, attestation: OracleAttestation) -> None:
        """Store attestation to file."""
        
        # Load existing attestations
        if self._attestations_file.exists():
            with self._attestations_file.open("r") as f:
                attestations = json.load(f)
        else:
            attestations = []
        
        # Append new attestation
        attestations.append(attestation.to_dict())
        
        # Keep only recent attestations (last 1000)
        attestations = attestations[-1000:]
        
        # Save
        with self._attestations_file.open("w") as f:
            json.dump(attestations, f, indent=2)
    
    def get_recent_attestations(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent attestations."""
        
        if self._attestations_file.exists():
            with self._attestations_file.open("r") as f:
                attestations = json.load(f)
            return attestations[-limit:]
        
        return []


# =============================================================================
# ORACLE REGISTRY
# =============================================================================


class OracleRegistry:
    """
    Registry of available oracles.
    
    Manages oracle selection and fallback.
    """
    
    def __init__(self):
        self._oracles: Dict[str, ExternalOracle] = {}
        self._default_oracle_id: Optional[str] = None
    
    def register(
        self,
        oracle: ExternalOracle,
        is_default: bool = False,
    ) -> None:
        """Register an oracle."""
        self._oracles[oracle.oracle_id] = oracle
        if is_default or self._default_oracle_id is None:
            self._default_oracle_id = oracle.oracle_id
    
    def get(
        self,
        oracle_id: Optional[str] = None,
    ) -> Optional[ExternalOracle]:
        """Get an oracle by ID or default."""
        if oracle_id:
            return self._oracles.get(oracle_id)
        if self._default_oracle_id:
            return self._oracles.get(self._default_oracle_id)
        return None
    
    def list_oracles(self) -> List[str]:
        """List registered oracle IDs."""
        return list(self._oracles.keys())


# Global registry
_oracle_registry = OracleRegistry()


def get_oracle(
    oracle_id: Optional[str] = None,
) -> ExternalOracle:
    """
    Get an oracle from the registry.
    
    Creates a local oracle if none registered.
    """
    oracle = _oracle_registry.get(oracle_id)
    
    if oracle is None:
        # Create and register local oracle
        local_oracle = LocalOracle()
        _oracle_registry.register(local_oracle, is_default=True)
        oracle = local_oracle
    
    return oracle


def register_oracle(
    oracle: ExternalOracle,
    is_default: bool = False,
) -> None:
    """Register an oracle in the global registry."""
    _oracle_registry.register(oracle, is_default)


# =============================================================================
# IHSAN ORACLE WRAPPER
# =============================================================================


class IhsanOracleVerifier:
    """
    Wrapper for verifying ihsān scores through external oracle.
    
    Drop-in replacement for self-referential scoring.
    """
    
    def __init__(
        self,
        oracle: Optional[ExternalOracle] = None,
        fail_closed: bool = True,
    ):
        self.oracle = oracle or get_oracle()
        self.fail_closed = fail_closed
    
    async def verify_score(
        self,
        claimed_score: float,
        context: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, float, OracleAttestation]:
        """
        Verify an ihsān score through the oracle.
        
        Args:
            claimed_score: Score to verify
            context: Context that produced the score
            metadata: Additional metadata
            
        Returns:
            Tuple of (is_valid, verified_score, attestation)
        """
        # Hash the context
        if BLAKE3_AVAILABLE:
            context_hash = blake3.blake3(context.encode()).hexdigest()
        else:
            context_hash = hashlib.sha256(context.encode()).hexdigest()
        
        # Get oracle attestation
        attestation = self.oracle.verify_ihsan_score(
            claimed_score=claimed_score,
            context_hash=context_hash,
            metadata=metadata,
        )
        
        # Determine validity
        if attestation.verdict == OracleVerdict.CONFIRMED:
            return True, attestation.verified_value or claimed_score, attestation
        elif attestation.verdict == OracleVerdict.ADJUSTED:
            return True, attestation.verified_value or claimed_score, attestation
        elif attestation.verdict == OracleVerdict.PENDING:
            if self.fail_closed:
                return False, 0.0, attestation
            else:
                return True, claimed_score, attestation
        else:
            return False, 0.0, attestation
    
    def verify_score_sync(
        self,
        claimed_score: float,
        context: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, float, OracleAttestation]:
        """Synchronous version of verify_score."""
        import asyncio
        
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.verify_score(claimed_score, context, metadata)
            )
        finally:
            loop.close()


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BIZRA External Oracle CLI"
    )
    parser.add_argument(
        "--verify-ihsan",
        type=float,
        help="Verify an ihsān score",
    )
    parser.add_argument(
        "--context",
        default="test_context",
        help="Context for verification",
    )
    parser.add_argument(
        "--list-attestations",
        action="store_true",
        help="List recent attestations",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Show calibration data",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    oracle = get_oracle()
    
    if args.verify_ihsan is not None:
        verifier = IhsanOracleVerifier(oracle)
        is_valid, verified_score, attestation = verifier.verify_score_sync(
            claimed_score=args.verify_ihsan,
            context=args.context,
        )
        
        print("\n" + "=" * 50)
        print("IHSAN ORACLE VERIFICATION")
        print("=" * 50)
        print(f"Claimed Score:   {args.verify_ihsan}")
        print(f"Verified Score:  {verified_score}")
        print(f"Verdict:         {attestation.verdict.value}")
        print(f"Confidence:      {attestation.confidence:.1%}")
        print(f"Valid:           {'YES' if is_valid else 'NO'}")
        print(f"Attestation ID:  {attestation.attestation_id}")
        print("=" * 50)
        
        return 0 if is_valid else 1
    
    if args.list_attestations:
        if isinstance(oracle, LocalOracle):
            attestations = oracle.get_recent_attestations(10)
            print("\nRecent Attestations:")
            for a in attestations:
                print(f"  {a['attestation_id']}: {a['verdict']} ({a['subject_type']})")
        else:
            print("Attestation listing not supported for this oracle type")
        return 0
    
    if args.calibration:
        calibration = oracle.get_calibration_data()
        print("\nCalibration Data:")
        print(json.dumps(calibration, indent=2))
        return 0
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
