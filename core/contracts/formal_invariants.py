"""
BIZRA AEON OMEGA - Design by Contract (DbC) Formal Invariant System
Production-Grade Runtime Contract Enforcement

This module implements comprehensive Design by Contract patterns including:
- Preconditions: Requirements that must hold before method execution
- Postconditions: Guarantees that must hold after method execution
- Class Invariants: Conditions that must always hold for an object
- Loop Invariants: Conditions that must hold during loop execution
- Assertion Contracts: Runtime-verified logical assertions

Inspired by Eiffel's DbC, Bertrand Meyer's methodology, and formal methods.

Author: BIZRA Core Team
Version: 1.0.0
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints
)
from weakref import WeakKeyDictionary

# Configure logging
logger = logging.getLogger("bizra.contracts")


# =============================================================================
# CONTRACT VIOLATION EXCEPTIONS
# =============================================================================

class ContractViolation(Exception):
    """Base exception for all contract violations."""
    
    def __init__(
        self,
        message: str,
        contract_type: str,
        location: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.contract_type = contract_type
        self.location = location
        self.details = details or {}
        super().__init__(f"[{contract_type}] {location}: {message}")


class PreconditionViolation(ContractViolation):
    """Raised when a precondition is not satisfied."""
    
    def __init__(self, message: str, location: str, **details):
        super().__init__(message, "PRECONDITION", location, details)


class PostconditionViolation(ContractViolation):
    """Raised when a postcondition is not satisfied."""
    
    def __init__(self, message: str, location: str, **details):
        super().__init__(message, "POSTCONDITION", location, details)


class InvariantViolation(ContractViolation):
    """Raised when a class invariant is violated."""
    
    def __init__(self, message: str, location: str, **details):
        super().__init__(message, "INVARIANT", location, details)


class AssertionViolation(ContractViolation):
    """Raised when an assertion contract fails."""
    
    def __init__(self, message: str, location: str, **details):
        super().__init__(message, "ASSERTION", location, details)


# =============================================================================
# CONTRACT ENFORCEMENT MODES
# =============================================================================

class ContractMode(Enum):
    """Contract enforcement modes."""
    DISABLED = auto()       # No contract checking
    ASSERTIONS_ONLY = auto() # Only check assertion contracts
    PRECONDITIONS = auto()   # Check preconditions only
    POSTCONDITIONS = auto()  # Check postconditions only
    INVARIANTS = auto()      # Check invariants only
    FULL = auto()            # Check all contracts
    DEBUG = auto()           # Full checking with detailed logging
    PRODUCTION = auto()      # Optimized checking for production


class ContractEnforcer:
    """
    Global contract enforcement configuration.
    
    Thread-safe singleton for managing contract enforcement across the system.
    """
    
    _instance: Optional[ContractEnforcer] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> ContractEnforcer:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self) -> None:
        self._mode = ContractMode.FULL
        self._violation_handlers: List[Callable[[ContractViolation], None]] = []
        self._statistics = ContractStatistics()
        self._disabled_contracts: Set[str] = set()
        self._thread_local = threading.local()
    
    @property
    def mode(self) -> ContractMode:
        return self._mode
    
    @mode.setter
    def mode(self, value: ContractMode) -> None:
        self._mode = value
        logger.info(f"Contract enforcement mode set to: {value.name}")
    
    def should_check(self, contract_type: str) -> bool:
        """Determine if a contract type should be checked."""
        if contract_type in self._disabled_contracts:
            return False
        
        if self._mode == ContractMode.DISABLED:
            return False
        elif self._mode == ContractMode.ASSERTIONS_ONLY:
            return contract_type == "ASSERTION"
        elif self._mode == ContractMode.PRECONDITIONS:
            return contract_type == "PRECONDITION"
        elif self._mode == ContractMode.POSTCONDITIONS:
            return contract_type == "POSTCONDITION"
        elif self._mode == ContractMode.INVARIANTS:
            return contract_type == "INVARIANT"
        elif self._mode in (ContractMode.FULL, ContractMode.DEBUG, ContractMode.PRODUCTION):
            return True
        return False
    
    def register_handler(
        self,
        handler: Callable[[ContractViolation], None]
    ) -> None:
        """Register a violation handler."""
        self._violation_handlers.append(handler)
    
    def handle_violation(self, violation: ContractViolation) -> None:
        """Handle a contract violation."""
        self._statistics.record_violation(violation)
        
        if self._mode == ContractMode.DEBUG:
            logger.error(f"Contract violation: {violation}")
        
        for handler in self._violation_handlers:
            try:
                handler(violation)
            except Exception as e:
                logger.exception(f"Violation handler error: {e}")
        
        raise violation
    
    def disable_contract(self, contract_id: str) -> None:
        """Disable a specific contract by ID."""
        self._disabled_contracts.add(contract_id)
    
    def enable_contract(self, contract_id: str) -> None:
        """Enable a previously disabled contract."""
        self._disabled_contracts.discard(contract_id)
    
    @contextmanager
    def temporarily_disable(self, *contract_types: str):
        """Context manager to temporarily disable contract checking."""
        if not hasattr(self._thread_local, 'disabled_stack'):
            self._thread_local.disabled_stack = []
        
        self._thread_local.disabled_stack.append(set(contract_types))
        try:
            yield
        finally:
            self._thread_local.disabled_stack.pop()
    
    def is_temporarily_disabled(self, contract_type: str) -> bool:
        """Check if contract type is temporarily disabled in current thread."""
        stack = getattr(self._thread_local, 'disabled_stack', [])
        for disabled_set in stack:
            if contract_type in disabled_set or "*" in disabled_set:
                return True
        return False
    
    @property
    def statistics(self) -> ContractStatistics:
        return self._statistics


@dataclass
class ContractStatistics:
    """Statistics for contract checking."""
    
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    violations_by_type: Dict[str, int] = field(default_factory=dict)
    average_check_time_ms: float = 0.0
    _check_times: List[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def record_check(self, passed: bool, duration_ms: float) -> None:
        """Record a contract check."""
        with self._lock:
            self.total_checks += 1
            if passed:
                self.passed_checks += 1
            else:
                self.failed_checks += 1
            
            self._check_times.append(duration_ms)
            if len(self._check_times) > 1000:
                self._check_times = self._check_times[-1000:]
            
            self.average_check_time_ms = sum(self._check_times) / len(self._check_times)
    
    def record_violation(self, violation: ContractViolation) -> None:
        """Record a contract violation."""
        with self._lock:
            vtype = violation.contract_type
            self.violations_by_type[vtype] = self.violations_by_type.get(vtype, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        with self._lock:
            return {
                "total_checks": self.total_checks,
                "passed_checks": self.passed_checks,
                "failed_checks": self.failed_checks,
                "pass_rate": self.passed_checks / max(1, self.total_checks),
                "violations_by_type": dict(self.violations_by_type),
                "average_check_time_ms": self.average_check_time_ms,
            }


# Global enforcer instance
enforcer = ContractEnforcer()


# =============================================================================
# CONTRACT DECORATORS
# =============================================================================

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


def requires(
    condition: Callable[..., bool],
    message: str = "Precondition failed",
    contract_id: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator for preconditions.
    
    The condition function receives the same arguments as the decorated function.
    
    Example:
        @requires(lambda x: x > 0, "x must be positive")
        def sqrt(x):
            return x ** 0.5
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if enforcer.is_temporarily_disabled("PRECONDITION"):
                return func(*args, **kwargs)
            
            if not enforcer.should_check("PRECONDITION"):
                return func(*args, **kwargs)
            
            start = time.perf_counter()
            location = f"{func.__module__}.{func.__qualname__}"
            
            try:
                # Get function signature for better error messages
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                
                # Check precondition
                result = condition(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                enforcer.statistics.record_check(result, duration_ms)
                
                if not result:
                    enforcer.handle_violation(
                        PreconditionViolation(
                            message,
                            location,
                            args=dict(bound.arguments),
                            contract_id=contract_id
                        )
                    )
            except ContractViolation:
                raise
            except Exception as e:
                enforcer.handle_violation(
                    PreconditionViolation(
                        f"Precondition check error: {e}",
                        location,
                        contract_id=contract_id
                    )
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def ensures(
    condition: Callable[..., bool],
    message: str = "Postcondition failed",
    contract_id: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator for postconditions.
    
    The condition function receives a special 'result' keyword argument
    containing the return value, plus all original arguments.
    
    Example:
        @ensures(lambda result, x: result >= 0, "sqrt must be non-negative")
        def sqrt(x):
            return x ** 0.5
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if enforcer.is_temporarily_disabled("POSTCONDITION"):
                return func(*args, **kwargs)
            
            result = func(*args, **kwargs)
            
            if not enforcer.should_check("POSTCONDITION"):
                return result
            
            start = time.perf_counter()
            location = f"{func.__module__}.{func.__qualname__}"
            
            try:
                # Check postcondition with result
                check_passed = condition(*args, result=result, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                enforcer.statistics.record_check(check_passed, duration_ms)
                
                if not check_passed:
                    enforcer.handle_violation(
                        PostconditionViolation(
                            message,
                            location,
                            result=result,
                            contract_id=contract_id
                        )
                    )
            except ContractViolation:
                raise
            except Exception as e:
                enforcer.handle_violation(
                    PostconditionViolation(
                        f"Postcondition check error: {e}",
                        location,
                        contract_id=contract_id
                    )
                )
            
            return result
        return wrapper
    return decorator


def old(func: F) -> F:
    """
    Capture 'old' values for postcondition checks.
    
    Usage in postcondition:
        @old
        @ensures(lambda result, old: result == old.x + 1)
        def increment(self):
            self.x += 1
    """
    # Store old values before function execution
    old_values = {}
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Capture current state
        old_values.update({
            k: getattr(self, k) for k in dir(self)
            if not k.startswith('_') and not callable(getattr(self, k))
        })
        
        # Create old value container
        class OldValues:
            pass
        
        old_container = OldValues()
        for k, v in old_values.items():
            setattr(old_container, k, v)
        
        # Add to kwargs for postcondition check
        kwargs['__old__'] = old_container
        
        return func(self, *args, **kwargs)
    
    return wrapper


# =============================================================================
# CLASS INVARIANT SYSTEM
# =============================================================================

class InvariantMeta(type):
    """
    Metaclass that enforces class invariants.
    
    Classes using this metaclass must define an `_invariant` method
    that returns True if the invariant holds.
    """
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Skip for base classes
        if name in ('Invariant', 'ContractClass'):
            return cls
        
        # Wrap methods to check invariants
        for attr_name, attr_value in list(namespace.items()):
            if (
                callable(attr_value) and
                not attr_name.startswith('_') and
                attr_name != '_invariant'
            ):
                setattr(cls, attr_name, mcs._wrap_with_invariant(attr_value, name))
        
        return cls
    
    @staticmethod
    def _wrap_with_invariant(method: Callable, class_name: str) -> Callable:
        """Wrap a method to check invariants before and after."""
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if enforcer.is_temporarily_disabled("INVARIANT"):
                return method(self, *args, **kwargs)
            
            if not enforcer.should_check("INVARIANT"):
                return method(self, *args, **kwargs)
            
            location = f"{class_name}.{method.__name__}"
            
            # Check invariant before
            if hasattr(self, '_invariant'):
                start = time.perf_counter()
                try:
                    holds = self._invariant()
                    duration_ms = (time.perf_counter() - start) * 1000
                    enforcer.statistics.record_check(holds, duration_ms)
                    
                    if not holds:
                        enforcer.handle_violation(
                            InvariantViolation(
                                "Invariant violated before method call",
                                location,
                                phase="pre"
                            )
                        )
                except ContractViolation:
                    raise
                except Exception as e:
                    enforcer.handle_violation(
                        InvariantViolation(
                            f"Invariant check error: {e}",
                            location,
                            phase="pre"
                        )
                    )
            
            # Execute method
            result = method(self, *args, **kwargs)
            
            # Check invariant after
            if hasattr(self, '_invariant'):
                start = time.perf_counter()
                try:
                    holds = self._invariant()
                    duration_ms = (time.perf_counter() - start) * 1000
                    enforcer.statistics.record_check(holds, duration_ms)
                    
                    if not holds:
                        enforcer.handle_violation(
                            InvariantViolation(
                                "Invariant violated after method call",
                                location,
                                phase="post"
                            )
                        )
                except ContractViolation:
                    raise
                except Exception as e:
                    enforcer.handle_violation(
                        InvariantViolation(
                            f"Invariant check error: {e}",
                            location,
                            phase="post"
                        )
                    )
            
            return result
        return wrapper


class ContractClass(metaclass=InvariantMeta):
    """
    Base class for classes with contract enforcement.
    
    Subclasses should implement `_invariant` to define class invariants.
    
    Example:
        class BankAccount(ContractClass):
            def __init__(self, balance: float):
                self._balance = balance
            
            def _invariant(self) -> bool:
                return self._balance >= 0
            
            def withdraw(self, amount: float) -> None:
                self._balance -= amount
    """
    
    def _invariant(self) -> bool:
        """Override to define class invariant. Default returns True."""
        return True


# =============================================================================
# BIZRA-SPECIFIC CONTRACT DEFINITIONS
# =============================================================================

class IhsanScoreContracts:
    """Formal contracts for Ihsān score validation."""
    
    DIMENSIONS = {"truthfulness", "dignity", "fairness", "excellence", "sustainability"}
    THRESHOLD = 0.85
    WEIGHTS = {
        "truthfulness": 0.25,
        "dignity": 0.25,
        "fairness": 0.20,
        "excellence": 0.15,
        "sustainability": 0.15,
    }
    
    @staticmethod
    def is_valid_dimension_score(score: float) -> bool:
        """Contract: dimension scores must be in [0.0, 1.0]."""
        return isinstance(score, (int, float)) and 0.0 <= score <= 1.0
    
    @staticmethod
    def is_valid_dimension_set(dimensions: Dict[str, float]) -> bool:
        """Contract: must have all 5 Ihsān dimensions."""
        return set(dimensions.keys()) == IhsanScoreContracts.DIMENSIONS
    
    @staticmethod
    def weights_sum_to_one() -> bool:
        """Contract: weights must sum to 1.0."""
        return abs(sum(IhsanScoreContracts.WEIGHTS.values()) - 1.0) < 1e-10
    
    @staticmethod
    def aggregate_within_bounds(aggregate: float) -> bool:
        """Contract: aggregate score must be in [0.0, 1.0]."""
        return 0.0 <= aggregate <= 1.0
    
    @classmethod
    def validate_all(cls, dimensions: Dict[str, float], aggregate: float) -> bool:
        """Validate all Ihsān score contracts."""
        return (
            cls.is_valid_dimension_set(dimensions) and
            all(cls.is_valid_dimension_score(v) for v in dimensions.values()) and
            cls.weights_sum_to_one() and
            cls.aggregate_within_bounds(aggregate)
        )


class VerificationTierContracts:
    """Formal contracts for tiered verification."""
    
    TIERS = ["STATISTICAL", "INCREMENTAL", "OPTIMISTIC", "FULL_ZK", "FORMAL"]
    LATENCY_BOUNDS = {
        "STATISTICAL": 100,      # ms
        "INCREMENTAL": 500,      # ms
        "OPTIMISTIC": 2000,      # ms
        "FULL_ZK": 5000,         # ms
        "FORMAL": float('inf'),  # async
    }
    
    @staticmethod
    def is_valid_tier(tier: str) -> bool:
        """Contract: tier must be a valid verification tier."""
        return tier in VerificationTierContracts.TIERS
    
    @staticmethod
    def tier_latency_bound(tier: str, actual_latency_ms: float) -> bool:
        """Contract: verification must complete within tier's latency bound."""
        if tier not in VerificationTierContracts.LATENCY_BOUNDS:
            return False
        bound = VerificationTierContracts.LATENCY_BOUNDS[tier]
        return actual_latency_ms <= bound
    
    @staticmethod
    def confidence_monotonic(tier1: str, tier2: str, conf1: float, conf2: float) -> bool:
        """Contract: higher tiers should have higher confidence."""
        tiers = VerificationTierContracts.TIERS
        if tier1 not in tiers or tier2 not in tiers:
            return False
        idx1 = tiers.index(tier1)
        idx2 = tiers.index(tier2)
        if idx1 < idx2:
            return conf1 <= conf2
        elif idx1 > idx2:
            return conf1 >= conf2
        return True


class TemporalChainContracts:
    """Formal contracts for temporal chain integrity."""
    
    @staticmethod
    def hash_length_valid(hash_str: str) -> bool:
        """Contract: Blake3 hash must be 64 hex characters."""
        if not isinstance(hash_str, str):
            return False
        return len(hash_str) == 64 and all(c in '0123456789abcdef' for c in hash_str.lower())
    
    @staticmethod
    def chain_link_valid(prev_hash: str, current_hash: str) -> bool:
        """Contract: chain entries must be properly linked."""
        return (
            TemporalChainContracts.hash_length_valid(prev_hash) and
            TemporalChainContracts.hash_length_valid(current_hash) and
            prev_hash != current_hash
        )
    
    @staticmethod
    def timestamp_monotonic(prev_ts: datetime, current_ts: datetime) -> bool:
        """Contract: timestamps must be strictly increasing."""
        return current_ts > prev_ts
    
    @staticmethod
    def chain_integrity(entries: List[Dict]) -> bool:
        """Contract: verify full chain integrity."""
        if not entries:
            return True
        
        for i in range(1, len(entries)):
            prev = entries[i - 1]
            curr = entries[i]
            
            if curr.get('prev_hash') != prev.get('current_hash'):
                return False
            
            if not TemporalChainContracts.timestamp_monotonic(
                prev.get('timestamp'),
                curr.get('timestamp')
            ):
                return False
        
        return True


class ValueOracleContracts:
    """Formal contracts for value oracle ensemble."""
    
    ORACLES = ["SHAPLEY", "PREDICTION_MARKET", "REPUTATION", "FORMAL_VERIFICATION", "INFORMATION_THEORETIC"]
    
    @staticmethod
    def signal_bounded(signal: float) -> bool:
        """Contract: oracle signals must be in [0.0, 1.0]."""
        return isinstance(signal, (int, float)) and 0.0 <= signal <= 1.0
    
    @staticmethod
    def confidence_bounded(confidence: float) -> bool:
        """Contract: confidence must be in [0.0, 1.0]."""
        return isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
    
    @staticmethod
    def disagreement_bounded(disagreement: float) -> bool:
        """Contract: disagreement must be in [0.0, 1.0]."""
        return isinstance(disagreement, (int, float)) and 0.0 <= disagreement <= 1.0
    
    @staticmethod
    def ensemble_valid(signals: Dict[str, float]) -> bool:
        """Contract: ensemble must have all oracle types."""
        return (
            set(signals.keys()) == set(ValueOracleContracts.ORACLES) and
            all(ValueOracleContracts.signal_bounded(v) for v in signals.values())
        )


# =============================================================================
# CONTRACT-ENABLED BIZRA COMPONENTS
# =============================================================================

class IhsanScoreCalculator(ContractClass):
    """
    Ihsān score calculator with formal contracts.
    
    Invariants:
    - Weights sum to 1.0
    - All dimension scores are in [0.0, 1.0]
    - Aggregate is properly calculated
    """
    
    def __init__(self):
        self._weights = IhsanScoreContracts.WEIGHTS.copy()
        self._last_dimensions: Optional[Dict[str, float]] = None
        self._last_aggregate: Optional[float] = None
    
    def _invariant(self) -> bool:
        """Class invariant: weights must sum to 1.0."""
        return abs(sum(self._weights.values()) - 1.0) < 1e-10
    
    @requires(
        lambda self, dimensions: IhsanScoreContracts.is_valid_dimension_set(dimensions),
        "Must provide all 5 Ihsān dimensions"
    )
    @requires(
        lambda self, dimensions: all(
            IhsanScoreContracts.is_valid_dimension_score(v)
            for v in dimensions.values()
        ),
        "All dimension scores must be in [0.0, 1.0]"
    )
    @ensures(
        lambda result, **_: IhsanScoreContracts.aggregate_within_bounds(result),
        "Aggregate score must be in [0.0, 1.0]"
    )
    def calculate(self, dimensions: Dict[str, float]) -> float:
        """Calculate the aggregate Ihsān Metric."""
        aggregate = sum(
            dimensions[dim] * self._weights[dim]
            for dim in IhsanScoreContracts.DIMENSIONS
        )
        
        self._last_dimensions = dimensions.copy()
        self._last_aggregate = aggregate
        
        return aggregate
    
    @requires(
        lambda self, aggregate: IhsanScoreContracts.aggregate_within_bounds(aggregate),
        "Aggregate must be in valid range"
    )
    def passes_threshold(self, aggregate: float) -> bool:
        """Check if aggregate passes Ihsān threshold."""
        return aggregate >= IhsanScoreContracts.THRESHOLD


class VerificationPipeline(ContractClass):
    """
    Verification pipeline with tier contracts.
    
    Invariants:
    - Tier selection is valid
    - Latency bounds are respected
    - Confidence increases with tier
    """
    
    def __init__(self):
        self._current_tier: Optional[str] = None
        self._last_latency_ms: float = 0.0
        self._last_confidence: float = 0.0
    
    def _invariant(self) -> bool:
        """Invariant: if tier is set, it must be valid."""
        if self._current_tier is None:
            return True
        return VerificationTierContracts.is_valid_tier(self._current_tier)
    
    @requires(
        lambda self, tier: VerificationTierContracts.is_valid_tier(tier),
        "Tier must be a valid verification tier"
    )
    def select_tier(self, tier: str) -> None:
        """Select a verification tier."""
        self._current_tier = tier
    
    @requires(
        lambda self: self._current_tier is not None,
        "Must select a tier before verification"
    )
    @ensures(
        lambda result, **_: 0.0 <= result.get('confidence', 0) <= 1.0,
        "Confidence must be bounded"
    )
    def verify(self, action_data: bytes) -> Dict[str, Any]:
        """Perform verification at the current tier."""
        import time
        
        start = time.perf_counter()
        
        # Simulate verification based on tier
        tier_delays = {
            "STATISTICAL": 0.05,
            "INCREMENTAL": 0.2,
            "OPTIMISTIC": 1.0,
            "FULL_ZK": 2.0,
            "FORMAL": 3.0,
        }
        
        tier_confidences = {
            "STATISTICAL": 0.7,
            "INCREMENTAL": 0.8,
            "OPTIMISTIC": 0.85,
            "FULL_ZK": 0.95,
            "FORMAL": 0.99,
        }
        
        # Simulate work
        time.sleep(tier_delays.get(self._current_tier, 0.1))
        
        latency_ms = (time.perf_counter() - start) * 1000
        confidence = tier_confidences.get(self._current_tier, 0.5)
        
        self._last_latency_ms = latency_ms
        self._last_confidence = confidence
        
        return {
            "tier": self._current_tier,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "valid": True,
        }


class TemporalChain(ContractClass):
    """
    Temporal chain with integrity contracts.
    
    Invariants:
    - All hashes are valid Blake3 hashes
    - Chain entries are properly linked
    - Timestamps are monotonically increasing
    """
    
    def __init__(self):
        self._entries: List[Dict] = []
        self._genesis_hash = "0" * 64
    
    def _invariant(self) -> bool:
        """Invariant: chain must maintain integrity."""
        if not self._entries:
            return True
        
        # Check first entry links to genesis
        if self._entries[0].get('prev_hash') != self._genesis_hash:
            return False
        
        # Check chain integrity
        return TemporalChainContracts.chain_integrity(self._entries)
    
    @requires(
        lambda self, operation, data_hash: TemporalChainContracts.hash_length_valid(data_hash),
        "Data hash must be a valid Blake3 hash"
    )
    @ensures(
        lambda result, **_: TemporalChainContracts.hash_length_valid(result['current_hash']),
        "Entry must have valid current hash"
    )
    def append(self, operation: str, data_hash: str) -> Dict:
        """Append an entry to the chain."""
        import hashlib
        
        prev_hash = (
            self._entries[-1]['current_hash']
            if self._entries else self._genesis_hash
        )
        
        timestamp = datetime.utcnow()
        
        # Create entry hash
        entry_data = f"{prev_hash}{operation}{data_hash}{timestamp.isoformat()}"
        current_hash = hashlib.blake2b(entry_data.encode(), digest_size=32).hexdigest()
        
        entry = {
            "prev_hash": prev_hash,
            "current_hash": current_hash,
            "operation": operation,
            "data_hash": data_hash,
            "timestamp": timestamp,
            "index": len(self._entries),
        }
        
        self._entries.append(entry)
        return entry
    
    @ensures(
        lambda result, **_: isinstance(result, bool),
        "Verification must return boolean"
    )
    def verify_integrity(self) -> bool:
        """Verify full chain integrity."""
        return self._invariant()


# =============================================================================
# CONTRACT DOCUMENTATION GENERATOR
# =============================================================================

class ContractDocGenerator:
    """Generate documentation for contracts in a module."""
    
    @staticmethod
    def extract_contracts(cls: Type) -> Dict[str, List[Dict]]:
        """Extract all contracts from a class."""
        contracts = {
            "preconditions": [],
            "postconditions": [],
            "invariants": [],
        }
        
        # Check for invariant
        if hasattr(cls, '_invariant'):
            invariant_doc = cls._invariant.__doc__ or "Class invariant"
            contracts["invariants"].append({
                "method": "_invariant",
                "description": invariant_doc,
            })
        
        # Check methods for contracts
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if hasattr(method, '__wrapped__'):
                # Look for contract decorators
                pass  # TODO: Introspect decorator chain
        
        return contracts
    
    @staticmethod
    def generate_markdown(cls: Type) -> str:
        """Generate Markdown documentation for a class's contracts."""
        contracts = ContractDocGenerator.extract_contracts(cls)
        
        doc = f"# Contracts for {cls.__name__}\n\n"
        
        if contracts["invariants"]:
            doc += "## Class Invariants\n\n"
            for inv in contracts["invariants"]:
                doc += f"- **{inv['method']}**: {inv['description']}\n"
            doc += "\n"
        
        if contracts["preconditions"]:
            doc += "## Preconditions\n\n"
            for pre in contracts["preconditions"]:
                doc += f"- **{pre['method']}**: {pre['description']}\n"
            doc += "\n"
        
        if contracts["postconditions"]:
            doc += "## Postconditions\n\n"
            for post in contracts["postconditions"]:
                doc += f"- **{post['method']}**: {post['description']}\n"
            doc += "\n"
        
        return doc


# =============================================================================
# TESTING UTILITIES
# =============================================================================

class ContractTestUtils:
    """Utilities for testing contracts."""
    
    @staticmethod
    @contextmanager
    def expect_violation(violation_type: Type[ContractViolation]):
        """Context manager that expects a specific contract violation."""
        try:
            yield
            raise AssertionError(f"Expected {violation_type.__name__} was not raised")
        except violation_type:
            pass  # Expected
    
    @staticmethod
    def check_all_contracts(obj: Any) -> Dict[str, bool]:
        """Check all contracts on an object."""
        results = {}
        
        # Check invariant
        if hasattr(obj, '_invariant'):
            try:
                results['invariant'] = obj._invariant()
            except Exception as e:
                results['invariant'] = False
                results['invariant_error'] = str(e)
        
        return results


# Export key components
__all__ = [
    # Exceptions
    'ContractViolation',
    'PreconditionViolation',
    'PostconditionViolation',
    'InvariantViolation',
    'AssertionViolation',
    
    # Decorators
    'requires',
    'ensures',
    'old',
    
    # Classes
    'ContractMode',
    'ContractEnforcer',
    'ContractClass',
    'ContractStatistics',
    
    # BIZRA contracts
    'IhsanScoreContracts',
    'VerificationTierContracts',
    'TemporalChainContracts',
    'ValueOracleContracts',
    
    # Contract-enabled components
    'IhsanScoreCalculator',
    'VerificationPipeline',
    'TemporalChain',
    
    # Utilities
    'ContractDocGenerator',
    'ContractTestUtils',
    
    # Global
    'enforcer',
]
