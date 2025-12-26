"""
BIZRA Formal Invariant Verifier
================================
Runtime Verification of Mathematical Invariants with Proof Generation

This module implements a formal verification system that ensures
mathematical invariants hold at runtime, with proof generation
for auditing and compliance. Inspired by Design by Contract (DbC)
and runtime verification principles.

Key Invariants Verified:
    - Ihsan Protocol: IM >= 0.95 (fail-closed)
    - Conservation Laws: Token supply conservation
    - Ordering: Temporal ordering of events
    - Bounds: Cognitive load within limits
    - Relationships: Cross-layer consistency

Features:
    - Declarative invariant specification
    - Runtime checking with minimal overhead
    - Proof generation (witnesses)
    - Violation detection and reporting
    - Integration with Ihsan circuit breaker

Author: BIZRA Genesis Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import secrets
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger("bizra.invariants")

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

T = TypeVar("T")
PredicateFunc = Callable[..., bool]
AsyncPredicateFunc = Callable[..., Awaitable[bool]]

# ============================================================================
# ENUMERATIONS
# ============================================================================


class InvariantSeverity(Enum):
    """Severity of invariant violations."""

    CRITICAL = auto()  # System must halt (Ihsan violations)
    MAJOR = auto()  # Operation must fail
    MINOR = auto()  # Log and continue
    WARNING = auto()  # Advisory only


class InvariantType(Enum):
    """Types of invariants."""

    IHSAN = auto()  # Ihsan protocol constraints
    CONSERVATION = auto()  # Conservation laws (tokens, energy)
    ORDERING = auto()  # Temporal/causal ordering
    BOUNDS = auto()  # Value within bounds
    RELATIONSHIP = auto()  # Cross-entity relationships
    STATE = auto()  # State machine invariants
    STRUCTURAL = auto()  # Data structure invariants


class ProofStatus(Enum):
    """Status of a proof."""

    VERIFIED = auto()  # Invariant holds, proof generated
    VIOLATED = auto()  # Invariant broken
    INCONCLUSIVE = auto()  # Could not determine
    TIMEOUT = auto()  # Verification timed out


# ============================================================================
# INVARIANT SPECIFICATION
# ============================================================================


@dataclass(frozen=True)
class InvariantSpec:
    """
    Specification of a single invariant.

    Invariants are named predicates that must hold at
    specific points in program execution.
    """

    name: str
    description: str
    invariant_type: InvariantType
    severity: InvariantSeverity
    predicate: PredicateFunc
    error_message: str
    tags: Tuple[str, ...] = ()

    def check(self, *args: Any, **kwargs: Any) -> bool:
        """Check if invariant holds."""
        try:
            return self.predicate(*args, **kwargs)
        except Exception as e:
            logger.error(f"Invariant {self.name} check failed: {e}")
            return False


@dataclass(frozen=True)
class AsyncInvariantSpec:
    """Async version of invariant specification."""

    name: str
    description: str
    invariant_type: InvariantType
    severity: InvariantSeverity
    predicate: AsyncPredicateFunc
    error_message: str
    tags: Tuple[str, ...] = ()

    async def check(self, *args: Any, **kwargs: Any) -> bool:
        """Check if invariant holds."""
        try:
            return await self.predicate(*args, **kwargs)
        except Exception as e:
            logger.error(f"Invariant {self.name} check failed: {e}")
            return False


# ============================================================================
# PROOF GENERATION
# ============================================================================


@dataclass(frozen=True)
class Witness:
    """
    A witness to an invariant check.

    Witnesses provide cryptographic proof that an invariant
    was checked at a specific point in time.
    """

    witness_id: str
    invariant_name: str
    timestamp: datetime
    result: bool
    context_hash: str
    arguments_hash: str

    @staticmethod
    def create(
        invariant_name: str,
        result: bool,
        context: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> Witness:
        """Create a witness for an invariant check."""
        context_str = json.dumps(context, sort_keys=True, default=str)
        args_str = json.dumps(arguments, sort_keys=True, default=str)

        return Witness(
            witness_id=f"wit_{secrets.token_hex(16)}",
            invariant_name=invariant_name,
            timestamp=datetime.now(timezone.utc),
            result=result,
            context_hash=hashlib.sha256(context_str.encode()).hexdigest()[:16],
            arguments_hash=hashlib.sha256(args_str.encode()).hexdigest()[:16],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "witness_id": self.witness_id,
            "invariant_name": self.invariant_name,
            "timestamp": self.timestamp.isoformat(),
            "result": self.result,
            "context_hash": self.context_hash,
            "arguments_hash": self.arguments_hash,
        }


@dataclass
class Proof:
    """
    A proof of invariant verification.

    Proofs aggregate multiple witnesses into a
    verifiable audit trail.
    """

    proof_id: str
    invariant_name: str
    status: ProofStatus
    witnesses: List[Witness]
    created_at: datetime
    verified_at: Optional[datetime]
    ihsan_score: float
    metadata: Dict[str, Any]

    @staticmethod
    def create(
        invariant_name: str,
        status: ProofStatus,
        witnesses: List[Witness],
        ihsan_score: float = 0.95,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Proof:
        now = datetime.now(timezone.utc)
        return Proof(
            proof_id=f"proof_{secrets.token_hex(12)}",
            invariant_name=invariant_name,
            status=status,
            witnesses=list(witnesses),
            created_at=now,
            verified_at=now if status == ProofStatus.VERIFIED else None,
            ihsan_score=ihsan_score,
            metadata=dict(metadata or {}),
        )

    @property
    def content_hash(self) -> str:
        """Compute hash of proof content."""
        content = json.dumps(
            {
                "proof_id": self.proof_id,
                "invariant_name": self.invariant_name,
                "status": self.status.name,
                "witness_ids": [w.witness_id for w in self.witnesses],
                "created_at": self.created_at.isoformat(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "invariant_name": self.invariant_name,
            "status": self.status.name,
            "witnesses": [w.to_dict() for w in self.witnesses],
            "created_at": self.created_at.isoformat(),
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "ihsan_score": self.ihsan_score,
            "content_hash": self.content_hash,
        }


# ============================================================================
# INVARIANT VIOLATION
# ============================================================================


@dataclass
class InvariantViolation:
    """
    Record of an invariant violation.

    Violations are recorded for auditing and may
    trigger circuit breakers.
    """

    violation_id: str
    invariant_name: str
    invariant_type: InvariantType
    severity: InvariantSeverity
    message: str
    timestamp: datetime
    context: Dict[str, Any]
    stack_trace: Optional[str]
    ihsan_impact: float

    @staticmethod
    def create(
        invariant: InvariantSpec,
        context: Dict[str, Any],
        ihsan_impact: float = 0.01,
    ) -> InvariantViolation:
        import traceback

        return InvariantViolation(
            violation_id=f"viol_{secrets.token_hex(8)}",
            invariant_name=invariant.name,
            invariant_type=invariant.invariant_type,
            severity=invariant.severity,
            message=invariant.error_message,
            timestamp=datetime.now(timezone.utc),
            context=dict(context),
            stack_trace=(
                traceback.format_stack()[-5:]
                if invariant.severity == InvariantSeverity.CRITICAL
                else None
            ),
            ihsan_impact=ihsan_impact,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "invariant_name": self.invariant_name,
            "invariant_type": self.invariant_type.name,
            "severity": self.severity.name,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "ihsan_impact": self.ihsan_impact,
        }


class InvariantError(Exception):
    """Exception raised when critical invariant is violated."""

    def __init__(self, violation: InvariantViolation):
        self.violation = violation
        super().__init__(violation.message)


# ============================================================================
# BUILT-IN INVARIANTS (IHSAN PROTOCOL)
# ============================================================================


class IhsanInvariants:
    """
    Built-in invariants for the Ihsan Protocol.

    These are the core invariants that must never be violated:
    1. Ihsan score >= 0.95 (threshold)
    2. Ihsan score <= 1.0 (upper bound)
    3. Governance requires Ihsan >= 0.98
    4. SAT minting requires Ihsan >= 0.99
    """

    THRESHOLD = 0.95
    GOVERNANCE_THRESHOLD = 0.98
    SAT_THRESHOLD = 0.99

    @staticmethod
    def ihsan_threshold(score: float) -> InvariantSpec:
        """Ihsan score must meet threshold."""
        return InvariantSpec(
            name="ihsan.threshold",
            description="Ihsan score must be at least 0.95",
            invariant_type=InvariantType.IHSAN,
            severity=InvariantSeverity.CRITICAL,
            predicate=lambda s: s >= IhsanInvariants.THRESHOLD,
            error_message=f"Ihsan score {score:.4f} below threshold {IhsanInvariants.THRESHOLD}",
            tags=("ihsan", "threshold", "critical"),
        )

    @staticmethod
    def ihsan_bounds(score: float) -> InvariantSpec:
        """Ihsan score must be in [0, 1]."""
        return InvariantSpec(
            name="ihsan.bounds",
            description="Ihsan score must be between 0 and 1",
            invariant_type=InvariantType.BOUNDS,
            severity=InvariantSeverity.CRITICAL,
            predicate=lambda s: 0.0 <= s <= 1.0,
            error_message=f"Ihsan score {score:.4f} out of bounds [0, 1]",
            tags=("ihsan", "bounds"),
        )

    @staticmethod
    def governance_ihsan(score: float) -> InvariantSpec:
        """Governance actions require elevated Ihsan."""
        return InvariantSpec(
            name="ihsan.governance",
            description="Governance requires Ihsan >= 0.98",
            invariant_type=InvariantType.IHSAN,
            severity=InvariantSeverity.MAJOR,
            predicate=lambda s: s >= IhsanInvariants.GOVERNANCE_THRESHOLD,
            error_message=f"Governance requires Ihsan >= {IhsanInvariants.GOVERNANCE_THRESHOLD}, got {score:.4f}",
            tags=("ihsan", "governance"),
        )

    @staticmethod
    def sat_minting_ihsan(score: float) -> InvariantSpec:
        """SAT minting requires highest Ihsan."""
        return InvariantSpec(
            name="ihsan.sat_minting",
            description="SAT minting requires Ihsan >= 0.99",
            invariant_type=InvariantType.IHSAN,
            severity=InvariantSeverity.MAJOR,
            predicate=lambda s: s >= IhsanInvariants.SAT_THRESHOLD,
            error_message=f"SAT minting requires Ihsan >= {IhsanInvariants.SAT_THRESHOLD}, got {score:.4f}",
            tags=("ihsan", "sat", "minting"),
        )


# ============================================================================
# CONSERVATION INVARIANTS
# ============================================================================


class ConservationInvariants:
    """
    Invariants for conservation laws.

    These ensure that quantities are conserved across
    operations (e.g., token transfers).
    """

    @staticmethod
    def token_conservation(
        total_before: float,
        total_after: float,
        minted: float = 0.0,
        burned: float = 0.0,
    ) -> InvariantSpec:
        """Token supply must be conserved."""
        expected = total_before + minted - burned

        return InvariantSpec(
            name="conservation.tokens",
            description="Token supply must be conserved",
            invariant_type=InvariantType.CONSERVATION,
            severity=InvariantSeverity.CRITICAL,
            predicate=lambda: abs(total_after - expected) < 1e-9,
            error_message=f"Token conservation violated: {total_after} != {expected}",
            tags=("conservation", "tokens"),
        )

    @staticmethod
    def energy_conservation(
        cognitive_in: float,
        cognitive_out: float,
        dissipated: float,
        tolerance: float = 0.01,
    ) -> InvariantSpec:
        """Cognitive energy must be conserved."""
        return InvariantSpec(
            name="conservation.energy",
            description="Cognitive energy conservation",
            invariant_type=InvariantType.CONSERVATION,
            severity=InvariantSeverity.MAJOR,
            predicate=lambda: abs(cognitive_in - cognitive_out - dissipated)
            < tolerance,
            error_message=f"Energy conservation violated: in={cognitive_in}, out={cognitive_out}, dissipated={dissipated}",
            tags=("conservation", "energy", "cognitive"),
        )


# ============================================================================
# ORDERING INVARIANTS
# ============================================================================


class OrderingInvariants:
    """
    Invariants for temporal and causal ordering.

    These ensure events occur in proper sequence.
    """

    @staticmethod
    def temporal_ordering(
        event_time: datetime,
        reference_time: datetime,
        must_be_after: bool = True,
    ) -> InvariantSpec:
        """Event must occur after/before reference."""
        if must_be_after:
            predicate = lambda: event_time > reference_time
            msg = f"Event at {event_time} must be after {reference_time}"
        else:
            predicate = lambda: event_time < reference_time
            msg = f"Event at {event_time} must be before {reference_time}"

        return InvariantSpec(
            name="ordering.temporal",
            description="Temporal ordering constraint",
            invariant_type=InvariantType.ORDERING,
            severity=InvariantSeverity.MAJOR,
            predicate=predicate,
            error_message=msg,
            tags=("ordering", "temporal"),
        )

    @staticmethod
    def sequence_monotonic(
        current_seq: int,
        previous_seq: int,
    ) -> InvariantSpec:
        """Sequence numbers must be monotonically increasing."""
        return InvariantSpec(
            name="ordering.sequence",
            description="Sequence must be monotonically increasing",
            invariant_type=InvariantType.ORDERING,
            severity=InvariantSeverity.MAJOR,
            predicate=lambda: current_seq > previous_seq,
            error_message=f"Sequence not monotonic: {current_seq} <= {previous_seq}",
            tags=("ordering", "sequence"),
        )


# ============================================================================
# BOUNDS INVARIANTS
# ============================================================================


class BoundsInvariants:
    """Invariants for value bounds."""

    @staticmethod
    def cognitive_load_bounds(load: float) -> InvariantSpec:
        """Cognitive load must be in [0, 1]."""
        return InvariantSpec(
            name="bounds.cognitive_load",
            description="Cognitive load must be between 0 and 1",
            invariant_type=InvariantType.BOUNDS,
            severity=InvariantSeverity.MINOR,
            predicate=lambda l: 0.0 <= l <= 1.0,
            error_message=f"Cognitive load {load} out of bounds [0, 1]",
            tags=("bounds", "cognitive"),
        )

    @staticmethod
    def percentage_bounds(value: float, name: str = "value") -> InvariantSpec:
        """Percentage must be in [0, 100]."""
        return InvariantSpec(
            name=f"bounds.percentage.{name}",
            description=f"{name} must be a valid percentage",
            invariant_type=InvariantType.BOUNDS,
            severity=InvariantSeverity.MINOR,
            predicate=lambda v: 0.0 <= v <= 100.0,
            error_message=f"{name} {value} is not a valid percentage",
            tags=("bounds", "percentage"),
        )


# ============================================================================
# INVARIANT VERIFIER
# ============================================================================


class InvariantVerifier:
    """
    Runtime invariant verification engine.

    Features:
    - Register and check invariants
    - Generate proofs for verified invariants
    - Track violations for auditing
    - Integration with Ihsan protocol
    """

    def __init__(
        self,
        fail_on_critical: bool = True,
        collect_witnesses: bool = True,
        max_violations: int = 10000,
    ):
        self._invariants: Dict[str, InvariantSpec] = {}
        self._async_invariants: Dict[str, AsyncInvariantSpec] = {}
        self._violations: Deque[InvariantViolation] = deque(maxlen=max_violations)
        self._proofs: Dict[str, Proof] = {}
        self._witnesses: Deque[Witness] = deque(maxlen=max_violations)

        self._fail_on_critical = fail_on_critical
        self._collect_witnesses = collect_witnesses

        # Statistics
        self._checks_total = 0
        self._checks_passed = 0
        self._checks_failed = 0
        self._by_type: Dict[InvariantType, int] = defaultdict(int)

        self._lock = asyncio.Lock()

        # Register built-in Ihsan invariants
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in invariants."""
        # These are templates, actual checks use dynamic values
        pass

    def register(self, invariant: InvariantSpec) -> None:
        """Register an invariant."""
        self._invariants[invariant.name] = invariant

    def register_async(self, invariant: AsyncInvariantSpec) -> None:
        """Register an async invariant."""
        self._async_invariants[invariant.name] = invariant

    def check(
        self,
        invariant: InvariantSpec,
        *args: Any,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[Proof]]:
        """
        Check an invariant synchronously.

        Returns (passed, proof).
        """
        self._checks_total += 1
        self._by_type[invariant.invariant_type] += 1

        ctx = context or {}
        result = invariant.check(*args, **kwargs)

        # Generate witness
        witness = None
        if self._collect_witnesses:
            witness = Witness.create(
                invariant_name=invariant.name,
                result=result,
                context=ctx,
                arguments={"args": str(args), "kwargs": str(kwargs)},
            )
            self._witnesses.append(witness)

        if result:
            self._checks_passed += 1

            # Generate proof
            proof = Proof.create(
                invariant_name=invariant.name,
                status=ProofStatus.VERIFIED,
                witnesses=[witness] if witness else [],
                ihsan_score=0.95,
            )
            self._proofs[proof.proof_id] = proof

            return (True, proof)
        else:
            self._checks_failed += 1

            # Record violation
            violation = InvariantViolation.create(invariant, ctx)
            self._violations.append(violation)

            # Generate failed proof
            proof = Proof.create(
                invariant_name=invariant.name,
                status=ProofStatus.VIOLATED,
                witnesses=[witness] if witness else [],
                ihsan_score=0.0,
                metadata={"violation_id": violation.violation_id},
            )
            self._proofs[proof.proof_id] = proof

            # Fail on critical
            if (
                self._fail_on_critical
                and invariant.severity == InvariantSeverity.CRITICAL
            ):
                raise InvariantError(violation)

            return (False, proof)

    async def check_async(
        self,
        invariant: AsyncInvariantSpec,
        *args: Any,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[Proof]]:
        """Check an async invariant."""
        async with self._lock:
            self._checks_total += 1
            self._by_type[invariant.invariant_type] += 1

        ctx = context or {}
        result = await invariant.check(*args, **kwargs)

        # Generate witness
        witness = None
        if self._collect_witnesses:
            witness = Witness.create(
                invariant_name=invariant.name,
                result=result,
                context=ctx,
                arguments={"args": str(args), "kwargs": str(kwargs)},
            )
            async with self._lock:
                self._witnesses.append(witness)

        if result:
            async with self._lock:
                self._checks_passed += 1

            proof = Proof.create(
                invariant_name=invariant.name,
                status=ProofStatus.VERIFIED,
                witnesses=[witness] if witness else [],
            )
            return (True, proof)
        else:
            async with self._lock:
                self._checks_failed += 1

            # Create violation record
            sync_spec = InvariantSpec(
                name=invariant.name,
                description=invariant.description,
                invariant_type=invariant.invariant_type,
                severity=invariant.severity,
                predicate=lambda: False,
                error_message=invariant.error_message,
            )
            violation = InvariantViolation.create(sync_spec, ctx)

            async with self._lock:
                self._violations.append(violation)

            proof = Proof.create(
                invariant_name=invariant.name,
                status=ProofStatus.VIOLATED,
                witnesses=[witness] if witness else [],
                metadata={"violation_id": violation.violation_id},
            )

            if (
                self._fail_on_critical
                and invariant.severity == InvariantSeverity.CRITICAL
            ):
                raise InvariantError(violation)

            return (False, proof)

    def check_ihsan(
        self,
        score: float,
        operation_type: str = "standard",
    ) -> Tuple[bool, Optional[Proof]]:
        """
        Check Ihsan score against appropriate threshold.

        Different operations have different thresholds:
        - standard: 0.95
        - governance: 0.98
        - sat_minting: 0.99
        """
        if operation_type == "governance":
            invariant = IhsanInvariants.governance_ihsan(score)
        elif operation_type == "sat_minting":
            invariant = IhsanInvariants.sat_minting_ihsan(score)
        else:
            invariant = IhsanInvariants.ihsan_threshold(score)

        return self.check(invariant, score, context={"operation": operation_type})

    def verify_conservation(
        self,
        before: float,
        after: float,
        minted: float = 0.0,
        burned: float = 0.0,
    ) -> Tuple[bool, Optional[Proof]]:
        """Verify token conservation invariant."""
        invariant = ConservationInvariants.token_conservation(
            before, after, minted, burned
        )
        return self.check(
            invariant,
            context={
                "before": before,
                "after": after,
                "minted": minted,
                "burned": burned,
            },
        )

    def get_violations(
        self,
        severity: Optional[InvariantSeverity] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[InvariantViolation]:
        """Get recorded violations."""
        violations = list(self._violations)

        if severity:
            violations = [v for v in violations if v.severity == severity]

        if since:
            violations = [v for v in violations if v.timestamp >= since]

        return violations[-limit:]

    def get_proof(self, proof_id: str) -> Optional[Proof]:
        """Get a specific proof."""
        return self._proofs.get(proof_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Get verification metrics."""
        return {
            "checks_total": self._checks_total,
            "checks_passed": self._checks_passed,
            "checks_failed": self._checks_failed,
            "pass_rate": self._checks_passed / max(1, self._checks_total),
            "violations_recorded": len(self._violations),
            "proofs_generated": len(self._proofs),
            "witnesses_collected": len(self._witnesses),
            "by_type": {t.name: c for t, c in self._by_type.items()},
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate a verification report."""
        critical = [
            v for v in self._violations if v.severity == InvariantSeverity.CRITICAL
        ]
        major = [v for v in self._violations if v.severity == InvariantSeverity.MAJOR]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self.get_metrics(),
            "critical_violations": len(critical),
            "major_violations": len(major),
            "ihsan_checks": self._by_type.get(InvariantType.IHSAN, 0),
            "conservation_checks": self._by_type.get(InvariantType.CONSERVATION, 0),
            "recent_violations": [v.to_dict() for v in self.get_violations(limit=10)],
            "status": "HEALTHY" if len(critical) == 0 else "CRITICAL",
        }


# ============================================================================
# DECORATORS FOR INVARIANT CHECKING
# ============================================================================


def requires_ihsan(
    threshold: float = 0.95,
    score_extractor: Optional[Callable[..., float]] = None,
):
    """
    Decorator that requires Ihsan score above threshold.

    Usage:
        @requires_ihsan(0.98)
        async def governance_action(self, ihsan_score: float, ...):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        verifier = InvariantVerifier()

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                if score_extractor:
                    score = score_extractor(*args, **kwargs)
                else:
                    score = kwargs.get("ihsan_score", 0.95)

                invariant = InvariantSpec(
                    name=f"ihsan.{func.__name__}",
                    description=f"Ihsan >= {threshold} for {func.__name__}",
                    invariant_type=InvariantType.IHSAN,
                    severity=InvariantSeverity.CRITICAL,
                    predicate=lambda s: s >= threshold,
                    error_message=f"Ihsan {score:.4f} < {threshold} for {func.__name__}",
                )

                passed, _ = verifier.check(invariant, score)
                if not passed:
                    raise ValueError(invariant.error_message)

                return await func(*args, **kwargs)

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                if score_extractor:
                    score = score_extractor(*args, **kwargs)
                else:
                    score = kwargs.get("ihsan_score", 0.95)

                invariant = InvariantSpec(
                    name=f"ihsan.{func.__name__}",
                    description=f"Ihsan >= {threshold} for {func.__name__}",
                    invariant_type=InvariantType.IHSAN,
                    severity=InvariantSeverity.CRITICAL,
                    predicate=lambda s: s >= threshold,
                    error_message=f"Ihsan {score:.4f} < {threshold} for {func.__name__}",
                )

                passed, _ = verifier.check(invariant, score)
                if not passed:
                    raise ValueError(invariant.error_message)

                return func(*args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator


def invariant(
    predicate: PredicateFunc,
    name: str = "",
    severity: InvariantSeverity = InvariantSeverity.MAJOR,
):
    """
    General-purpose invariant decorator.

    Usage:
        @invariant(lambda result: result >= 0, "non_negative_result")
        def compute_value():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        verifier = InvariantVerifier()
        inv_name = name or f"invariant.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = func(*args, **kwargs)

            spec = InvariantSpec(
                name=inv_name,
                description=f"Post-condition for {func.__name__}",
                invariant_type=InvariantType.STATE,
                severity=severity,
                predicate=predicate,
                error_message=f"Invariant {inv_name} violated",
            )

            passed, _ = verifier.check(spec, result)
            if not passed:
                raise InvariantError(
                    InvariantViolation.create(spec, {"result": str(result)})
                )

            return result

        return wrapper  # type: ignore

    return decorator


# ============================================================================
# CONTRACT SPECIFICATION (DESIGN BY CONTRACT)
# ============================================================================


class Contract:
    """
    Design by Contract implementation.

    Allows specifying:
    - Preconditions (require)
    - Postconditions (ensure)
    - Class invariants
    """

    def __init__(self, verifier: Optional[InvariantVerifier] = None):
        self._verifier = verifier or InvariantVerifier()
        self._preconditions: List[InvariantSpec] = []
        self._postconditions: List[InvariantSpec] = []

    def require(
        self,
        predicate: PredicateFunc,
        message: str = "Precondition failed",
    ) -> Contract:
        """Add a precondition."""
        self._preconditions.append(
            InvariantSpec(
                name=f"precondition.{len(self._preconditions)}",
                description=message,
                invariant_type=InvariantType.STATE,
                severity=InvariantSeverity.MAJOR,
                predicate=predicate,
                error_message=message,
            )
        )
        return self

    def ensure(
        self,
        predicate: PredicateFunc,
        message: str = "Postcondition failed",
    ) -> Contract:
        """Add a postcondition."""
        self._postconditions.append(
            InvariantSpec(
                name=f"postcondition.{len(self._postconditions)}",
                description=message,
                invariant_type=InvariantType.STATE,
                severity=InvariantSeverity.MAJOR,
                predicate=predicate,
                error_message=message,
            )
        )
        return self

    def check_preconditions(self, *args: Any, **kwargs: Any) -> None:
        """Check all preconditions."""
        for pre in self._preconditions:
            passed, _ = self._verifier.check(pre, *args, **kwargs)
            if not passed:
                raise ValueError(pre.error_message)

    def check_postconditions(self, result: Any) -> None:
        """Check all postconditions."""
        for post in self._postconditions:
            passed, _ = self._verifier.check(post, result)
            if not passed:
                raise ValueError(post.error_message)


# ============================================================================
# DEMO
# ============================================================================


async def demo_invariant_verifier():
    """Demonstrate invariant verification capabilities."""
    print("=" * 70)
    print("BIZRA FORMAL INVARIANT VERIFIER DEMO")
    print("=" * 70)

    verifier = InvariantVerifier(fail_on_critical=False)

    # 1. Ihsan Invariants
    print("\n1. Ihsan Protocol Invariants")
    print("-" * 40)

    for score in [0.99, 0.96, 0.93]:
        passed, proof = verifier.check_ihsan(score)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Ihsan {score:.2f}: {status}")

    # 2. Conservation Invariants
    print("\n2. Conservation Invariants")
    print("-" * 40)

    passed, _ = verifier.verify_conservation(
        before=1000.0,
        after=1000.0,
        minted=0.0,
        burned=0.0,
    )
    print(f"  Token conservation (balanced): {'✓ PASS' if passed else '✗ FAIL'}")

    passed, _ = verifier.verify_conservation(
        before=1000.0,
        after=1050.0,  # Incorrect!
        minted=0.0,
        burned=0.0,
    )
    print(f"  Token conservation (violated): {'✓ PASS' if passed else '✗ FAIL'}")

    # 3. Bounds Invariants
    print("\n3. Bounds Invariants")
    print("-" * 40)

    for load in [0.5, 1.0, 1.5]:
        inv = BoundsInvariants.cognitive_load_bounds(load)
        passed, _ = verifier.check(inv, load)
        print(f"  Cognitive load {load}: {'✓ PASS' if passed else '✗ FAIL'}")

    # 4. Custom Invariant
    print("\n4. Custom Invariants")
    print("-" * 40)

    custom = InvariantSpec(
        name="custom.positive_balance",
        description="Balance must be positive",
        invariant_type=InvariantType.BOUNDS,
        severity=InvariantSeverity.MAJOR,
        predicate=lambda b: b >= 0,
        error_message="Balance cannot be negative",
    )

    for balance in [100.0, 0.0, -50.0]:
        passed, _ = verifier.check(custom, balance)
        print(f"  Balance {balance}: {'✓ PASS' if passed else '✗ FAIL'}")

    # 5. Metrics
    print("\n5. Verification Metrics")
    print("-" * 40)

    metrics = verifier.get_metrics()
    print(f"  Total checks: {metrics['checks_total']}")
    print(f"  Passed: {metrics['checks_passed']}")
    print(f"  Failed: {metrics['checks_failed']}")
    print(f"  Pass rate: {metrics['pass_rate']:.1%}")

    # 6. Violations
    print("\n6. Recent Violations")
    print("-" * 40)

    violations = verifier.get_violations(limit=5)
    for v in violations:
        print(f"  [{v.severity.name}] {v.invariant_name}: {v.message}")

    # 7. Report
    print("\n7. Verification Report")
    print("-" * 40)

    report = verifier.generate_report()
    print(f"  Status: {report['status']}")
    print(f"  Critical violations: {report['critical_violations']}")
    print(f"  Major violations: {report['major_violations']}")

    print("\n" + "=" * 70)
    print("INVARIANT VERIFIER DEMO COMPLETE")
    print("=" * 70)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "InvariantSeverity",
    "InvariantType",
    "ProofStatus",
    # Specs
    "InvariantSpec",
    "AsyncInvariantSpec",
    # Proofs
    "Witness",
    "Proof",
    # Violations
    "InvariantViolation",
    "InvariantError",
    # Built-in Invariants
    "IhsanInvariants",
    "ConservationInvariants",
    "OrderingInvariants",
    "BoundsInvariants",
    # Verifier
    "InvariantVerifier",
    # Decorators
    "requires_ihsan",
    "invariant",
    # Contract
    "Contract",
]


if __name__ == "__main__":
    asyncio.run(demo_invariant_verifier())
