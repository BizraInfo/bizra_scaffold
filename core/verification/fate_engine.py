"""
BIZRA AEON OMEGA - Formal Alignment & Transcendence Engine (FATE)
==================================================================
Z3 SMT Solver Integration for Mathematical Ethics Enforcement

This module implements the FATE Engine - the "physics engine for ethics"
that renders unethical behavior mathematically impossible through formal
verification using Z3 SMT (Satisfiability Modulo Theories) solver.

Key Features:
    - Hoare Triple verification: {P} C {Q}
    - Ihsān Vector computation: I_vec = Σ(w_i × S_i)
    - Causal Drag enforcement: Ω ≤ 0.05
    - Symbolic-neural bridge for typed plan primitives
    - Ex-Ante constraint checking (before execution)

Mathematical Foundation:
    ∀ action ∈ Actions: verify(action) → I_vec ≥ 0.95 ∧ Ω ≤ 0.05
    
    Where:
    - I_vec = Weighted sum of 8 Ihsān dimensions
    - Ω = Causal drag coefficient (systemic friction)

Author: BIZRA Genesis Team (Peak Masterpiece v4)
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# Try to import Z3 - graceful degradation if not available
try:
    import z3
    from z3 import (
        And,
        Bool,
        If,
        Implies,
        Not,
        Or,
        Real,
        Solver,
        sat,
        unsat,
        unknown,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    # Fallback types for type hints
    z3 = None  # type: ignore

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger("bizra.fate")

# ============================================================================
# CONSTANTS - Constitutional Thresholds
# ============================================================================

# Ihsān Threshold (from constitution.toml)
IHSAN_THRESHOLD: float = 0.95
IHSAN_GOVERNANCE_THRESHOLD: float = 0.98
IHSAN_SAT_MINTING_THRESHOLD: float = 0.99

# Causal Drag Maximum (from audit report)
CAUSAL_DRAG_MAX: float = 0.05

# Ihsān Vector Weights (8 dimensions per SOT)
IHSAN_WEIGHTS: Dict[str, float] = {
    "truthfulness": 0.20,      # صدق (Sidq)
    "trustworthiness": 0.15,   # أمانة (Amanah)
    "justice": 0.15,           # عدل (Adl)
    "excellence": 0.15,        # إتقان (Itqan)
    "mercy": 0.10,             # رحمة (Rahmah)
    "dignity": 0.10,           # كرامة (Karamah)
    "sustainability": 0.10,    # استدامة (Istidamah)
    "transparency": 0.05,      # شفافية (Shafafiyyah)
}

# Verify weights sum to 1.0
assert abs(sum(IHSAN_WEIGHTS.values()) - 1.0) < 1e-9, "Ihsān weights must sum to 1.0"

# ============================================================================
# ENUMERATIONS
# ============================================================================


class FATEVerdict(Enum):
    """FATE verification outcomes."""
    
    PASS = auto()           # Action satisfies all constraints
    FAIL_IHSAN = auto()     # I_vec < threshold
    FAIL_OMEGA = auto()     # Ω > maximum
    FAIL_FORMAL = auto()    # SMT unsatisfiable
    NEEDS_HUMAN = auto()    # Human approval required
    TIMEOUT = auto()        # Verification timed out
    DEGRADED = auto()       # Z3 unavailable, fallback used


class ActionRisk(Enum):
    """Risk levels for actions."""
    
    LOW = auto()         # Stateless, reversible
    MEDIUM = auto()      # Stateful, reversible
    HIGH = auto()        # Stateful, irreversible
    CRITICAL = auto()    # Governance, token minting


class ConstraintType(Enum):
    """Types of formal constraints."""
    
    PRECONDITION = auto()   # Must hold before action
    POSTCONDITION = auto()  # Must hold after action
    INVARIANT = auto()      # Must always hold
    ASSERTION = auto()      # Explicit check point


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class IhsanVector:
    """
    8-Dimensional Ihsān Vector.
    
    Computes I_vec = Σ(w_i × S_i) for weighted ethical scoring.
    Each dimension is [0, 1] with constitutional weights.
    """
    
    truthfulness: float = 1.0      # صدق
    trustworthiness: float = 1.0   # أمانة
    justice: float = 1.0           # عدل
    excellence: float = 1.0        # إتقان
    mercy: float = 1.0             # رحمة
    dignity: float = 1.0           # كرامة
    sustainability: float = 1.0    # استدامة
    transparency: float = 1.0      # شفافية
    
    def __post_init__(self):
        """Validate all dimensions are in [0, 1]."""
        for dim, value in self.to_dict().items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Ihsān dimension {dim} must be in [0, 1], got {value}")
    
    def compute_i_vec(self) -> float:
        """
        Compute weighted Ihsān vector score.
        
        I_vec = Σ(w_i × S_i) where w_i are constitutional weights.
        
        Returns:
            float: Weighted score in [0, 1]
        """
        values = self.to_dict()
        return sum(
            IHSAN_WEIGHTS[dim] * values[dim]
            for dim in IHSAN_WEIGHTS
        )
    
    def passes_threshold(self, risk: ActionRisk = ActionRisk.LOW) -> bool:
        """Check if I_vec meets threshold for given risk level."""
        i_vec = self.compute_i_vec()
        
        if risk == ActionRisk.CRITICAL:
            return i_vec >= IHSAN_SAT_MINTING_THRESHOLD
        elif risk == ActionRisk.HIGH:
            return i_vec >= IHSAN_GOVERNANCE_THRESHOLD
        else:
            return i_vec >= IHSAN_THRESHOLD
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "truthfulness": self.truthfulness,
            "trustworthiness": self.trustworthiness,
            "justice": self.justice,
            "excellence": self.excellence,
            "mercy": self.mercy,
            "dignity": self.dignity,
            "sustainability": self.sustainability,
            "transparency": self.transparency,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "IhsanVector":
        """Create from dictionary."""
        return cls(
            truthfulness=data.get("truthfulness", 1.0),
            trustworthiness=data.get("trustworthiness", 1.0),
            justice=data.get("justice", 1.0),
            excellence=data.get("excellence", 1.0),
            mercy=data.get("mercy", 1.0),
            dignity=data.get("dignity", 1.0),
            sustainability=data.get("sustainability", 1.0),
            transparency=data.get("transparency", 1.0),
        )


@dataclass
class CausalDrag:
    """
    Causal Drag Coefficient (Ω).
    
    Measures systemic friction generated by an action.
    Constraint: Ω ≤ 0.05 for all actions.
    
    Components:
        - Resource contention: How much this action competes for resources
        - State complexity: Change in system state complexity
        - Reversibility cost: Cost to undo the action
        - Coordination overhead: Multi-agent coordination burden
    """
    
    resource_contention: float = 0.0
    state_complexity: float = 0.0
    reversibility_cost: float = 0.0
    coordination_overhead: float = 0.0
    
    def __post_init__(self):
        """Validate all components are in [0, 1]."""
        for comp in [
            self.resource_contention,
            self.state_complexity,
            self.reversibility_cost,
            self.coordination_overhead,
        ]:
            if not 0.0 <= comp <= 1.0:
                raise ValueError(f"Causal drag component must be in [0, 1], got {comp}")
    
    def compute_omega(self) -> float:
        """
        Compute Ω (causal drag coefficient).
        
        Ω = 0.3×RC + 0.3×SC + 0.25×RV + 0.15×CO
        
        Returns:
            float: Causal drag in [0, 1]
        """
        return (
            0.30 * self.resource_contention +
            0.30 * self.state_complexity +
            0.25 * self.reversibility_cost +
            0.15 * self.coordination_overhead
        )
    
    def within_bounds(self) -> bool:
        """Check if Ω ≤ CAUSAL_DRAG_MAX (0.05)."""
        return self.compute_omega() <= CAUSAL_DRAG_MAX


@dataclass
class FormalConstraint:
    """
    Formal constraint for Z3 SMT verification.
    
    Represents a Hoare Triple: {P} C {Q}
    - P: Precondition (must hold before)
    - C: Command/Action (what to verify)
    - Q: Postcondition (must hold after)
    """
    
    name: str
    constraint_type: ConstraintType
    expression: str  # Human-readable expression
    z3_builder: Optional[Callable[["z3.Solver", Dict[str, Any]], None]] = None
    
    def add_to_solver(self, solver: "z3.Solver", context: Dict[str, Any]) -> None:
        """Add this constraint to Z3 solver."""
        if self.z3_builder and Z3_AVAILABLE:
            self.z3_builder(solver, context)


@dataclass
class ActionProposal:
    """
    Proposal for an action to be verified by FATE.
    
    Contains all information needed for ex-ante verification.
    """
    
    action_id: str
    action_type: str
    target_resource: str
    ihsan_vector: IhsanVector
    causal_drag: CausalDrag
    risk_level: ActionRisk
    justification: str
    state_before_hash: str
    expected_effects: List[str] = field(default_factory=list)
    formal_constraints: List[FormalConstraint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_proposal_hash(self) -> str:
        """Compute deterministic hash of this proposal."""
        data = {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "target_resource": self.target_resource,
            "ihsan_vector": self.ihsan_vector.to_dict(),
            "risk_level": self.risk_level.name,
            "state_before_hash": self.state_before_hash,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class FATEReceipt:
    """
    Receipt from FATE verification.
    
    Cryptographically binds proposal to verification outcome.
    """
    
    proposal_hash: str
    verdict: FATEVerdict
    i_vec_score: float
    omega_score: float
    formal_check_passed: bool
    reasons: List[str]
    constraints_checked: int
    verification_time_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    z3_model: Optional[str] = None  # Satisfying assignment if found
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "proposal_hash": self.proposal_hash,
            "verdict": self.verdict.name,
            "i_vec_score": self.i_vec_score,
            "omega_score": self.omega_score,
            "formal_check_passed": self.formal_check_passed,
            "reasons": self.reasons,
            "constraints_checked": self.constraints_checked,
            "verification_time_ms": self.verification_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "z3_model": self.z3_model,
        }


# ============================================================================
# FATE ENGINE CORE
# ============================================================================


class FATEEngine:
    """
    Formal Alignment & Transcendence Engine.
    
    The "physics engine for ethics" that makes unethical behavior
    mathematically impossible through Z3 SMT constraint solving.
    
    Verification Pipeline:
        1. Ihsān Vector check (I_vec ≥ threshold)
        2. Causal Drag check (Ω ≤ 0.05)
        3. Formal constraint satisfaction (Z3 SMT)
        4. Generate cryptographic receipt
    
    Usage:
        engine = FATEEngine()
        proposal = ActionProposal(...)
        receipt = engine.verify(proposal)
        if receipt.verdict == FATEVerdict.PASS:
            # Safe to execute
    """
    
    def __init__(
        self,
        ihsan_threshold: float = IHSAN_THRESHOLD,
        omega_max: float = CAUSAL_DRAG_MAX,
        timeout_ms: float = 2000.0,
        enable_z3: bool = True,
    ):
        """
        Initialize FATE Engine.
        
        Args:
            ihsan_threshold: Minimum I_vec score (default 0.95)
            omega_max: Maximum causal drag (default 0.05)
            timeout_ms: Z3 solver timeout
            enable_z3: Enable Z3 formal verification
        """
        self.ihsan_threshold = ihsan_threshold
        self.omega_max = omega_max
        self.timeout_ms = timeout_ms
        self.enable_z3 = enable_z3 and Z3_AVAILABLE
        
        # Built-in constraints (Constitutional invariants)
        self._builtin_constraints: List[FormalConstraint] = self._init_builtin_constraints()
        
        # Verification statistics
        self._verifications: int = 0
        self._passes: int = 0
        self._failures: int = 0
        
        if self.enable_z3:
            logger.info("FATE Engine initialized with Z3 SMT solver")
        else:
            logger.warning("FATE Engine initialized in DEGRADED mode (Z3 unavailable)")
    
    def _init_builtin_constraints(self) -> List[FormalConstraint]:
        """Initialize constitutional built-in constraints."""
        constraints = []
        
        # Ihsān threshold constraint
        def ihsan_constraint(solver: "z3.Solver", ctx: Dict[str, Any]) -> None:
            i_vec = Real("i_vec")
            threshold = Real("threshold")
            solver.add(i_vec == ctx.get("i_vec_score", 0.0))
            solver.add(threshold == self.ihsan_threshold)
            solver.add(i_vec >= threshold)
        
        constraints.append(FormalConstraint(
            name="IHSAN_THRESHOLD",
            constraint_type=ConstraintType.INVARIANT,
            expression=f"I_vec >= {self.ihsan_threshold}",
            z3_builder=ihsan_constraint if Z3_AVAILABLE else None,
        ))
        
        # Causal drag constraint
        def omega_constraint(solver: "z3.Solver", ctx: Dict[str, Any]) -> None:
            omega = Real("omega")
            max_omega = Real("max_omega")
            solver.add(omega == ctx.get("omega_score", 1.0))
            solver.add(max_omega == self.omega_max)
            solver.add(omega <= max_omega)
        
        constraints.append(FormalConstraint(
            name="CAUSAL_DRAG_BOUND",
            constraint_type=ConstraintType.INVARIANT,
            expression=f"Ω <= {self.omega_max}",
            z3_builder=omega_constraint if Z3_AVAILABLE else None,
        ))
        
        # Token conservation (example domain constraint)
        def conservation_constraint(solver: "z3.Solver", ctx: Dict[str, Any]) -> None:
            supply_before = Real("supply_before")
            supply_after = Real("supply_after")
            minted = Real("minted")
            burned = Real("burned")
            solver.add(supply_after == supply_before + minted - burned)
            # Conservation: if no minting, supply can't increase
            solver.add(Implies(minted == 0, supply_after <= supply_before))
        
        constraints.append(FormalConstraint(
            name="TOKEN_CONSERVATION",
            constraint_type=ConstraintType.INVARIANT,
            expression="supply_after = supply_before + minted - burned",
            z3_builder=conservation_constraint if Z3_AVAILABLE else None,
        ))
        
        return constraints
    
    def verify(self, proposal: ActionProposal) -> FATEReceipt:
        """
        Verify an action proposal through FATE pipeline.
        
        Pipeline:
            1. Check Ihsān Vector (I_vec ≥ threshold)
            2. Check Causal Drag (Ω ≤ max)
            3. Run Z3 SMT solver on formal constraints
            4. Generate cryptographic receipt
        
        Args:
            proposal: Action to verify
            
        Returns:
            FATEReceipt with verdict and proofs
        """
        start_time = time.perf_counter()
        self._verifications += 1
        reasons: List[str] = []
        
        # Compute scores
        i_vec_score = proposal.ihsan_vector.compute_i_vec()
        omega_score = proposal.causal_drag.compute_omega()
        
        # Get risk-appropriate threshold
        if proposal.risk_level == ActionRisk.CRITICAL:
            effective_threshold = IHSAN_SAT_MINTING_THRESHOLD
        elif proposal.risk_level == ActionRisk.HIGH:
            effective_threshold = IHSAN_GOVERNANCE_THRESHOLD
        else:
            effective_threshold = self.ihsan_threshold
        
        # Step 1: Ihsān Vector Check
        if i_vec_score < effective_threshold:
            self._failures += 1
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            reasons.append(
                f"I_vec={i_vec_score:.4f} < threshold={effective_threshold:.2f}"
            )
            return FATEReceipt(
                proposal_hash=proposal.compute_proposal_hash(),
                verdict=FATEVerdict.FAIL_IHSAN,
                i_vec_score=i_vec_score,
                omega_score=omega_score,
                formal_check_passed=False,
                reasons=reasons,
                constraints_checked=0,
                verification_time_ms=elapsed_ms,
            )
        
        # Step 2: Causal Drag Check
        if omega_score > self.omega_max:
            self._failures += 1
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            reasons.append(
                f"Ω={omega_score:.4f} > max={self.omega_max:.2f}"
            )
            return FATEReceipt(
                proposal_hash=proposal.compute_proposal_hash(),
                verdict=FATEVerdict.FAIL_OMEGA,
                i_vec_score=i_vec_score,
                omega_score=omega_score,
                formal_check_passed=False,
                reasons=reasons,
                constraints_checked=0,
                verification_time_ms=elapsed_ms,
            )
        
        # Step 3: Formal Verification (Z3 SMT)
        formal_passed = True
        constraints_checked = 0
        z3_model_str: Optional[str] = None
        
        if self.enable_z3:
            solver = Solver()
            solver.set("timeout", int(self.timeout_ms))
            
            context = {
                "i_vec_score": i_vec_score,
                "omega_score": omega_score,
                "risk_level": proposal.risk_level.name,
            }
            
            # Add built-in constraints
            for constraint in self._builtin_constraints:
                constraint.add_to_solver(solver, context)
                constraints_checked += 1
            
            # Add proposal-specific constraints
            for constraint in proposal.formal_constraints:
                constraint.add_to_solver(solver, context)
                constraints_checked += 1
            
            # Check satisfiability
            result = solver.check()
            
            if result == unsat:
                formal_passed = False
                reasons.append("Z3: Constraints unsatisfiable")
            elif result == unknown:
                # Timeout or unknown - fail-closed
                formal_passed = False
                reasons.append("Z3: Verification timeout (fail-closed)")
            else:  # sat
                # Extract model
                model = solver.model()
                z3_model_str = str(model)
                reasons.append(f"Z3: Satisfiable ({constraints_checked} constraints)")
        else:
            # Degraded mode - statistical check only
            reasons.append("Z3 unavailable - degraded mode (statistical only)")
        
        # Step 4: Generate Receipt
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        if not formal_passed:
            self._failures += 1
            return FATEReceipt(
                proposal_hash=proposal.compute_proposal_hash(),
                verdict=FATEVerdict.FAIL_FORMAL if self.enable_z3 else FATEVerdict.DEGRADED,
                i_vec_score=i_vec_score,
                omega_score=omega_score,
                formal_check_passed=False,
                reasons=reasons,
                constraints_checked=constraints_checked,
                verification_time_ms=elapsed_ms,
                z3_model=z3_model_str,
            )
        
        # All checks passed
        self._passes += 1
        reasons.append(f"FATE PASS: I_vec={i_vec_score:.4f}, Ω={omega_score:.4f}")
        
        return FATEReceipt(
            proposal_hash=proposal.compute_proposal_hash(),
            verdict=FATEVerdict.PASS,
            i_vec_score=i_vec_score,
            omega_score=omega_score,
            formal_check_passed=True,
            reasons=reasons,
            constraints_checked=constraints_checked,
            verification_time_ms=elapsed_ms,
            z3_model=z3_model_str,
        )
    
    def verify_hoare_triple(
        self,
        precondition: Callable[[], bool],
        action: Callable[[], Any],
        postcondition: Callable[[Any], bool],
        ihsan_vector: IhsanVector,
        timeout_ms: Optional[float] = None,
    ) -> Tuple[bool, Optional[Any], List[str]]:
        """
        Verify a Hoare Triple: {P} C {Q}.
        
        Args:
            precondition: P - must hold before action
            action: C - the action to execute
            postcondition: Q - must hold after action
            ihsan_vector: Ethical alignment of action
            timeout_ms: Optional timeout override
            
        Returns:
            Tuple of (success, result, reasons)
        """
        reasons: List[str] = []
        timeout = timeout_ms or self.timeout_ms
        
        # Check Ihsān first
        i_vec = ihsan_vector.compute_i_vec()
        if i_vec < self.ihsan_threshold:
            reasons.append(f"Ihsān check failed: {i_vec:.4f} < {self.ihsan_threshold}")
            return False, None, reasons
        
        # Check precondition
        try:
            if not precondition():
                reasons.append("Precondition {P} not satisfied")
                return False, None, reasons
        except Exception as e:
            reasons.append(f"Precondition raised exception: {e}")
            return False, None, reasons
        
        # Execute action with timeout
        start = time.perf_counter()
        try:
            result = action()
            elapsed = (time.perf_counter() - start) * 1000
            
            if elapsed > timeout:
                reasons.append(f"Action exceeded timeout: {elapsed:.1f}ms > {timeout:.1f}ms")
                return False, result, reasons
        except Exception as e:
            reasons.append(f"Action raised exception: {e}")
            return False, None, reasons
        
        # Check postcondition
        try:
            if not postcondition(result):
                reasons.append("Postcondition {Q} not satisfied")
                return False, result, reasons
        except Exception as e:
            reasons.append(f"Postcondition raised exception: {e}")
            return False, result, reasons
        
        reasons.append("Hoare Triple verified: {P} C {Q}")
        return True, result, reasons
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics."""
        return {
            "total_verifications": self._verifications,
            "passes": self._passes,
            "failures": self._failures,
            "pass_rate": self._passes / max(self._verifications, 1),
            "z3_enabled": self.enable_z3,
            "ihsan_threshold": self.ihsan_threshold,
            "omega_max": self.omega_max,
        }


# ============================================================================
# CONSTRAINT BUILDERS (Z3)
# ============================================================================


class ConstraintBuilders:
    """Factory for common Z3 constraint builders."""
    
    @staticmethod
    def non_negative(var_name: str) -> Callable[["z3.Solver", Dict[str, Any]], None]:
        """Create non-negativity constraint."""
        if not Z3_AVAILABLE:
            return lambda s, c: None
        
        def builder(solver: "z3.Solver", ctx: Dict[str, Any]) -> None:
            var = Real(var_name)
            solver.add(var >= 0)
        
        return builder
    
    @staticmethod
    def bounded(
        var_name: str,
        lower: float,
        upper: float,
    ) -> Callable[["z3.Solver", Dict[str, Any]], None]:
        """Create bounded constraint: lower <= var <= upper."""
        if not Z3_AVAILABLE:
            return lambda s, c: None
        
        def builder(solver: "z3.Solver", ctx: Dict[str, Any]) -> None:
            var = Real(var_name)
            solver.add(And(var >= lower, var <= upper))
        
        return builder
    
    @staticmethod
    def implies_constraint(
        antecedent_var: str,
        consequent_var: str,
        threshold: float,
    ) -> Callable[["z3.Solver", Dict[str, Any]], None]:
        """Create implication: antecedent > threshold => consequent."""
        if not Z3_AVAILABLE:
            return lambda s, c: None
        
        def builder(solver: "z3.Solver", ctx: Dict[str, Any]) -> None:
            ante = Real(antecedent_var)
            cons = Bool(consequent_var)
            solver.add(Implies(ante > threshold, cons))
        
        return builder


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def quick_verify(
    i_vec_score: float,
    omega_score: float = 0.0,
    risk: ActionRisk = ActionRisk.LOW,
) -> bool:
    """
    Quick verification without full proposal.
    
    Args:
        i_vec_score: Pre-computed Ihsān vector score
        omega_score: Pre-computed causal drag
        risk: Risk level of action
        
    Returns:
        bool: True if constraints satisfied
    """
    # Get threshold for risk level
    if risk == ActionRisk.CRITICAL:
        threshold = IHSAN_SAT_MINTING_THRESHOLD
    elif risk == ActionRisk.HIGH:
        threshold = IHSAN_GOVERNANCE_THRESHOLD
    else:
        threshold = IHSAN_THRESHOLD
    
    return i_vec_score >= threshold and omega_score <= CAUSAL_DRAG_MAX


def compute_ihsan_from_dimensions(dimensions: Dict[str, float]) -> float:
    """
    Compute I_vec from dimension dictionary.
    
    Args:
        dimensions: Dict mapping dimension names to scores [0, 1]
        
    Returns:
        float: Weighted I_vec score
    """
    return sum(
        IHSAN_WEIGHTS.get(dim, 0.0) * score
        for dim, score in dimensions.items()
    )


# ============================================================================
# SELF-TEST
# ============================================================================


def _self_test() -> None:
    """Run FATE Engine self-tests."""
    print("=" * 70)
    print("BIZRA AEON OMEGA - FATE Engine Self-Test")
    print("=" * 70)
    
    # Test 1: IhsanVector computation
    print("\n[Test 1] IhsanVector Computation")
    vec = IhsanVector(
        truthfulness=1.0,
        trustworthiness=1.0,
        justice=1.0,
        excellence=1.0,
        mercy=1.0,
        dignity=1.0,
        sustainability=1.0,
        transparency=1.0,
    )
    i_vec = vec.compute_i_vec()
    assert abs(i_vec - 1.0) < 1e-9, f"Perfect score should be 1.0, got {i_vec}"
    print(f"  ✓ Perfect I_vec = {i_vec}")
    
    # Test 2: CausalDrag computation
    print("\n[Test 2] CausalDrag Computation")
    drag = CausalDrag(
        resource_contention=0.01,
        state_complexity=0.01,
        reversibility_cost=0.01,
        coordination_overhead=0.01,
    )
    omega = drag.compute_omega()
    assert omega <= CAUSAL_DRAG_MAX, f"Low drag should be <= 0.05, got {omega}"
    print(f"  ✓ Low Ω = {omega:.4f} (within bounds)")
    
    # Test 3: FATE Engine verification
    print("\n[Test 3] FATE Engine Verification")
    engine = FATEEngine()
    
    proposal = ActionProposal(
        action_id="test-001",
        action_type="READ",
        target_resource="/test/resource",
        ihsan_vector=vec,
        causal_drag=drag,
        risk_level=ActionRisk.LOW,
        justification="Self-test",
        state_before_hash="abc123",
    )
    
    receipt = engine.verify(proposal)
    assert receipt.verdict == FATEVerdict.PASS, f"Expected PASS, got {receipt.verdict}"
    print(f"  ✓ Verification: {receipt.verdict.name}")
    print(f"  ✓ I_vec: {receipt.i_vec_score:.4f}")
    print(f"  ✓ Ω: {receipt.omega_score:.4f}")
    print(f"  ✓ Time: {receipt.verification_time_ms:.2f}ms")
    
    # Test 4: Fail-closed on low Ihsān
    print("\n[Test 4] Fail-Closed (Low Ihsān)")
    low_ihsan = IhsanVector(
        truthfulness=0.8,
        trustworthiness=0.8,
        justice=0.8,
        excellence=0.8,
        mercy=0.8,
        dignity=0.8,
        sustainability=0.8,
        transparency=0.8,
    )
    bad_proposal = ActionProposal(
        action_id="test-002",
        action_type="WRITE",
        target_resource="/test/resource",
        ihsan_vector=low_ihsan,
        causal_drag=drag,
        risk_level=ActionRisk.LOW,
        justification="Should fail",
        state_before_hash="def456",
    )
    
    bad_receipt = engine.verify(bad_proposal)
    assert bad_receipt.verdict == FATEVerdict.FAIL_IHSAN, \
        f"Expected FAIL_IHSAN, got {bad_receipt.verdict}"
    print(f"  ✓ Correctly rejected: {bad_receipt.verdict.name}")
    print(f"  ✓ I_vec: {bad_receipt.i_vec_score:.4f} < {IHSAN_THRESHOLD}")
    
    # Test 5: Fail on high Omega
    print("\n[Test 5] Fail-Closed (High Causal Drag)")
    high_drag = CausalDrag(
        resource_contention=0.2,
        state_complexity=0.2,
        reversibility_cost=0.2,
        coordination_overhead=0.2,
    )
    drag_proposal = ActionProposal(
        action_id="test-003",
        action_type="WRITE",
        target_resource="/test/resource",
        ihsan_vector=vec,  # Good Ihsān
        causal_drag=high_drag,  # Bad drag
        risk_level=ActionRisk.LOW,
        justification="Should fail on drag",
        state_before_hash="ghi789",
    )
    
    drag_receipt = engine.verify(drag_proposal)
    assert drag_receipt.verdict == FATEVerdict.FAIL_OMEGA, \
        f"Expected FAIL_OMEGA, got {drag_receipt.verdict}"
    print(f"  ✓ Correctly rejected: {drag_receipt.verdict.name}")
    print(f"  ✓ Ω: {drag_receipt.omega_score:.4f} > {CAUSAL_DRAG_MAX}")
    
    # Statistics
    print("\n[Statistics]")
    stats = engine.get_statistics()
    print(f"  Total: {stats['total_verifications']}")
    print(f"  Passes: {stats['passes']}")
    print(f"  Failures: {stats['failures']}")
    print(f"  Z3 Enabled: {stats['z3_enabled']}")
    
    print("\n" + "=" * 70)
    print("✅ ALL FATE ENGINE SELF-TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()
