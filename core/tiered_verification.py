"""
BIZRA AEON OMEGA - Tiered Verification System
═══════════════════════════════════════════════════════════════════════════════
Addresses the 247ms zk-proof latency wall through adaptive verification tiers.
Elite Practitioner Pattern: Optimistic Execution with Async Verification

Gap Addressed: Real-Time Requirements vs. Proof Generation Latency (2%)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
import secrets
import logging

from core.architecture.modular_components import ConvergenceResult, ConvergenceQuality

logger = logging.getLogger(__name__)

# Timeout constants for async operations
VERIFICATION_TIMEOUT_SECONDS = 30.0  # Max time for single verification
BATCH_GATHER_TIMEOUT_SECONDS = 60.0  # Max time for awaiting all pending verifications


class UrgencyLevel(Enum):
    """Verification urgency classification."""
    REAL_TIME = auto()      # <100ms required - statistical verification
    NEAR_REAL_TIME = auto() # 100ms-1s - incremental proof
    BATCH = auto()          # >1s acceptable - full Groth16 proof
    DEFERRED = auto()       # No time constraint - comprehensive audit


class VerificationTier(Enum):
    """Verification strength tiers."""
    STATISTICAL = auto()    # 95% confidence, ~10ms
    INCREMENTAL = auto()    # Partial proof, ~50ms
    OPTIMISTIC = auto()     # Immediate execution, async verify
    FULL_ZK = auto()        # Complete Groth16, ~200ms
    FORMAL = auto()         # Mathematical proof, unbounded


@dataclass
class VerificationResult:
    """Result of verification operation."""
    tier: VerificationTier
    confidence: float           # 0.0 to 1.0
    latency_ms: float          # Actual verification time
    proof_hash: Optional[str]  # Cryptographic proof identifier
    valid: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rollback_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Represents an action to be verified."""
    id: str
    payload: bytes
    urgency: UrgencyLevel
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RollbackEvent:
    """Event triggered when async verification fails."""
    action_id: str
    original_action: Action
    verification_result: VerificationResult
    rollback_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class VerificationStrategy(ABC):
    """Abstract base for verification strategies."""
    
    @abstractmethod
    async def verify(self, action: Action) -> VerificationResult:
        """Execute verification strategy."""
        pass
    
    @property
    @abstractmethod
    def tier(self) -> VerificationTier:
        """Return the verification tier."""
        pass
    
    @property
    @abstractmethod
    def expected_latency_ms(self) -> float:
        """Expected latency in milliseconds."""
        pass


class QuantizedConvergence:
    """
    Compatibility wrapper for QuantizedConvergence.

    Avoids eager imports to prevent circular dependencies with
    core.ultimate_integration while keeping the public API stable.
    """

    def __new__(cls, *args, **kwargs):
        from core.ultimate_integration import QuantizedConvergence as _QuantizedConvergence
        return _QuantizedConvergence(*args, **kwargs)


class StatisticalVerification(VerificationStrategy):
    """
    Statistical verification with 95% confidence.
    Uses sampling and probabilistic bounds for rapid verification.
    """
    
    def __init__(self, sample_size: int = 100, confidence_threshold: float = 0.95):
        self.sample_size = sample_size
        self.confidence_threshold = confidence_threshold
        self._verification_history: deque = deque(maxlen=10000)
    
    @property
    def tier(self) -> VerificationTier:
        return VerificationTier.STATISTICAL
    
    @property
    def expected_latency_ms(self) -> float:
        return 10.0
    
    async def verify(self, action: Action) -> VerificationResult:
        start_time = time.perf_counter()
        
        # Compute payload hash
        payload_hash = hashlib.sha256(action.payload).hexdigest()
        
        # Statistical sampling of verification criteria
        checks_passed = 0
        for _ in range(self.sample_size):
            # Simulate sampling check (in production: actual constraint checks)
            sample_valid = self._sample_constraint_check(action.payload)
            if sample_valid:
                checks_passed += 1
        
        # Compute confidence using binomial proportion
        confidence = checks_passed / self.sample_size
        valid = confidence >= self.confidence_threshold
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        result = VerificationResult(
            tier=self.tier,
            confidence=confidence,
            latency_ms=latency_ms,
            proof_hash=f"stat_{payload_hash[:16]}",
            valid=valid,
            metadata={
                "sample_size": self.sample_size,
                "checks_passed": checks_passed,
                "threshold": self.confidence_threshold
            }
        )
        
        self._verification_history.append(result)
        return result
    
    def _sample_constraint_check(self, payload: bytes) -> bool:
        """Sample a random constraint and verify."""
        # In production: check actual neural→symbolic constraints
        # Here: simulate with deterministic pseudo-randomness
        h = hashlib.sha256(payload + secrets.token_bytes(8)).digest()
        return h[0] > 12  # ~95% pass rate for valid payloads


class IncrementalProofVerification(VerificationStrategy):
    """
    Incremental zk-proof verification.
    Builds partial proofs that can be completed asynchronously.
    """
    
    def __init__(self, proof_segments: int = 4):
        self.proof_segments = proof_segments
        self._partial_proofs: Dict[str, List[bytes]] = {}
    
    @property
    def tier(self) -> VerificationTier:
        return VerificationTier.INCREMENTAL
    
    @property
    def expected_latency_ms(self) -> float:
        return 50.0
    
    async def verify(self, action: Action) -> VerificationResult:
        start_time = time.perf_counter()
        
        # Generate incremental proof segments
        proof_segments = []
        for i in range(self.proof_segments):
            segment = await self._generate_proof_segment(action.payload, i)
            proof_segments.append(segment)
        
        # Verify partial proof
        partial_valid = all(self._verify_segment(s) for s in proof_segments)
        
        # Compute combined proof hash
        combined = b"".join(proof_segments)
        proof_hash = hashlib.blake2b(combined, digest_size=32).hexdigest()
        
        # Store for potential completion
        self._partial_proofs[action.id] = proof_segments
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return VerificationResult(
            tier=self.tier,
            confidence=0.85 if partial_valid else 0.0,  # Partial proof = 85% confidence
            latency_ms=latency_ms,
            proof_hash=f"incr_{proof_hash[:16]}",
            valid=partial_valid,
            metadata={
                "segments_verified": self.proof_segments,
                "completion_pending": True
            }
        )
    
    async def _generate_proof_segment(self, payload: bytes, index: int) -> bytes:
        """Generate a single proof segment."""
        # Simulate incremental proof generation
        await asyncio.sleep(0.01)  # ~10ms per segment
        h = hashlib.sha256(payload + index.to_bytes(4, 'big')).digest()
        return h
    
    def _verify_segment(self, segment: bytes) -> bool:
        """Verify a proof segment."""
        return len(segment) == 32 and segment[0] != 0


class FullZKProofVerification(VerificationStrategy):
    """
    Full Groth16 zk-SNARK verification.
    Provides cryptographic certainty but with ~200ms latency.
    """
    
    def __init__(self, circuit_complexity: int = 2**16):
        self.circuit_complexity = circuit_complexity
        self._proof_cache: Dict[str, bytes] = {}
    
    @property
    def tier(self) -> VerificationTier:
        return VerificationTier.FULL_ZK
    
    @property
    def expected_latency_ms(self) -> float:
        return 200.0
    
    async def verify(self, action: Action) -> VerificationResult:
        start_time = time.perf_counter()
        
        # Check cache first
        cache_key = hashlib.sha256(action.payload).hexdigest()
        if cache_key in self._proof_cache:
            cached_proof = self._proof_cache[cache_key]
            latency_ms = (time.perf_counter() - start_time) * 1000
            return VerificationResult(
                tier=self.tier,
                confidence=1.0,
                latency_ms=latency_ms,
                proof_hash=cache_key,
                valid=True,
                metadata={"cached": True}
            )
        
        # Simulate Groth16 proof generation (200ms)
        proof = await self._generate_groth16_proof(action.payload)
        valid = self._verify_groth16_proof(proof)
        
        if valid:
            self._proof_cache[cache_key] = proof
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return VerificationResult(
            tier=self.tier,
            confidence=1.0 if valid else 0.0,
            latency_ms=latency_ms,
            proof_hash=cache_key if valid else None,
            valid=valid,
            metadata={
                "circuit_complexity": self.circuit_complexity,
                "proof_size_bytes": len(proof)
            }
        )
    
    async def _generate_groth16_proof(self, payload: bytes) -> bytes:
        """Simulate Groth16 proof generation."""
        # In production: actual zk-SNARK circuit execution
        await asyncio.sleep(0.2)  # ~200ms simulation
        
        # Generate deterministic "proof" for simulation
        proof_data = hashlib.sha3_512(payload).digest()
        return proof_data
    
    def _verify_groth16_proof(self, proof: bytes) -> bool:
        """Verify Groth16 proof."""
        return len(proof) == 64 and proof[0] != 0xFF


class TieredVerificationEngine:
    """
    Elite Practitioner Pattern: Tiered Verification System
    
    Selects optimal verification strategy based on urgency level,
    implements optimistic execution with async verification,
    and handles rollback on verification failure.
    """
    
    def __init__(self):
        self.strategies: Dict[UrgencyLevel, VerificationStrategy] = {
            UrgencyLevel.REAL_TIME: StatisticalVerification(),
            UrgencyLevel.NEAR_REAL_TIME: IncrementalProofVerification(),
            UrgencyLevel.BATCH: FullZKProofVerification(),
            UrgencyLevel.DEFERRED: FullZKProofVerification(),
        }
        
        # Async verification queue
        self._pending_verifications: Dict[str, asyncio.Task] = {}
        self._verification_results: Dict[str, VerificationResult] = {}
        
        # Rollback handlers
        self._rollback_handlers: List[Callable[[RollbackEvent], None]] = []
        
        # Metrics
        self._metrics = {
            "total_verifications": 0,
            "by_tier": {tier.name: 0 for tier in VerificationTier},
            "rollbacks": 0,
            "cache_hits": 0,
            "avg_latency_ms": 0.0
        }
    
    def register_rollback_handler(self, handler: Callable[[RollbackEvent], None]) -> None:
        """Register a handler for rollback events."""
        self._rollback_handlers.append(handler)
    
    async def verify(self, action: Action) -> VerificationResult:
        """
        Execute tiered verification based on urgency.
        Returns immediately for REAL_TIME, may block for BATCH.
        """
        strategy = self.strategies.get(action.urgency)
        if not strategy:
            raise ValueError(f"No strategy for urgency: {action.urgency}")
        
        result = await strategy.verify(action)
        
        # Update metrics
        self._metrics["total_verifications"] += 1
        self._metrics["by_tier"][result.tier.name] += 1
        self._update_avg_latency(result.latency_ms)
        
        self._verification_results[action.id] = result
        return result
    
    async def verify_optimistic(
        self, 
        action: Action,
        execute_callback: Callable[[Action], Any]
    ) -> Tuple[Any, asyncio.Task]:
        """
        Optimistic execution with async verification.
        
        1. Execute action immediately (optimistic)
        2. Fire-and-forget verification in background
        3. Return execution result + verification task
        4. If verification fails later, trigger rollback
        """
        # 1. Immediate optimistic execution (handle both sync and async callbacks)
        if asyncio.iscoroutinefunction(execute_callback):
            execution_result = await execute_callback(action)
        else:
            execution_result = execute_callback(action)
        
        # 2. Fire-and-forget full verification with timeout
        async def _verify_with_timeout() -> VerificationResult:
            """Wrap verification with timeout to prevent hung tasks."""
            try:
                return await asyncio.wait_for(
                    self._async_verify_with_rollback(action),
                    timeout=VERIFICATION_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Verification timeout for action {action.id} "
                    f"after {VERIFICATION_TIMEOUT_SECONDS}s"
                )
                # Create timeout result - treat as verification failure requiring rollback
                timeout_result = VerificationResult(
                    tier=VerificationTier.FULL_ZK,
                    confidence=0.0,
                    latency_ms=VERIFICATION_TIMEOUT_SECONDS * 1000,
                    proof_hash=None,
                    valid=False,
                    rollback_required=True,
                    metadata={"error": "verification_timeout"}
                )
                await self._trigger_rollback(action, timeout_result)
                return timeout_result
        
        verification_task = asyncio.create_task(_verify_with_timeout())
        self._pending_verifications[action.id] = verification_task
        
        return execution_result, verification_task
    
    async def _async_verify_with_rollback(self, action: Action) -> VerificationResult:
        """Async verification that triggers rollback on failure."""
        # Always use full ZK for optimistic verification
        strategy = self.strategies[UrgencyLevel.BATCH]
        result = await strategy.verify(action)
        
        if not result.valid:
            # Trigger rollback
            result.rollback_required = True
            await self._trigger_rollback(action, result)
        
        # Cleanup pending
        self._pending_verifications.pop(action.id, None)
        self._verification_results[action.id] = result
        
        return result
    
    async def _trigger_rollback(self, action: Action, result: VerificationResult) -> None:
        """Trigger rollback handlers for failed verification."""
        self._metrics["rollbacks"] += 1
        
        event = RollbackEvent(
            action_id=action.id,
            original_action=action,
            verification_result=result
        )
        
        for handler in self._rollback_handlers:
            try:
                # Handle both sync and async rollback handlers
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                # Log but don't fail on handler errors
                print(f"Rollback handler error: {e}")
    
    def _update_avg_latency(self, latency_ms: float) -> None:
        """Update running average latency."""
        n = self._metrics["total_verifications"]
        current_avg = self._metrics["avg_latency_ms"]
        self._metrics["avg_latency_ms"] = (current_avg * (n - 1) + latency_ms) / n
    
    async def await_pending_verifications(
        self,
        timeout: Optional[float] = None
    ) -> Dict[str, VerificationResult]:
        """Wait for all pending verifications to complete with timeout.
        
        Args:
            timeout: Max seconds to wait. Defaults to BATCH_GATHER_TIMEOUT_SECONDS.
        """
        if not self._pending_verifications:
            return {}
        
        effective_timeout = timeout or BATCH_GATHER_TIMEOUT_SECONDS
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *self._pending_verifications.values(),
                    return_exceptions=True
                ),
                timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"await_pending_verifications timeout after {effective_timeout}s, "
                f"{len(self._pending_verifications)} verifications still pending"
            )
            # Cancel remaining tasks
            for task in self._pending_verifications.values():
                if not task.done():
                    task.cancel()
            # Return what we have
            results = []
            for task in self._pending_verifications.values():
                try:
                    if task.done():
                        results.append(task.result())
                    else:
                        results.append(asyncio.CancelledError())
                except Exception as e:
                    results.append(e)
        
        return {
            action_id: r for action_id, r in zip(
                self._pending_verifications.keys(), 
                results
            ) if isinstance(r, VerificationResult)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return verification metrics."""
        return {
            **self._metrics,
            "pending_count": len(self._pending_verifications),
            "cache_size": sum(
                len(getattr(s, '_proof_cache', {})) 
                for s in self.strategies.values()
            )
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Tests
# ═══════════════════════════════════════════════════════════════════════════════

async def self_test():
    """Self-test for tiered verification system."""
    print("Tiered Verification Self-Test")
    print("=" * 50)
    
    engine = TieredVerificationEngine()
    
    # Track rollbacks
    rollbacks = []
    engine.register_rollback_handler(lambda e: rollbacks.append(e))
    
    # Test 1: Real-time verification
    action_rt = Action(
        id="test-rt-001",
        payload=b"real-time action payload",
        urgency=UrgencyLevel.REAL_TIME
    )
    result_rt = await engine.verify(action_rt)
    assert result_rt.tier == VerificationTier.STATISTICAL
    assert result_rt.latency_ms < 100, f"Real-time too slow: {result_rt.latency_ms}ms"
    print(f"✓ Real-time verification: {result_rt.latency_ms:.2f}ms, confidence={result_rt.confidence:.2f}")
    
    # Test 2: Near real-time verification
    action_nrt = Action(
        id="test-nrt-001",
        payload=b"near-real-time action payload",
        urgency=UrgencyLevel.NEAR_REAL_TIME
    )
    result_nrt = await engine.verify(action_nrt)
    assert result_nrt.tier == VerificationTier.INCREMENTAL
    print(f"✓ Near real-time verification: {result_nrt.latency_ms:.2f}ms, confidence={result_nrt.confidence:.2f}")
    
    # Test 3: Batch verification
    action_batch = Action(
        id="test-batch-001",
        payload=b"batch action payload",
        urgency=UrgencyLevel.BATCH
    )
    result_batch = await engine.verify(action_batch)
    assert result_batch.tier == VerificationTier.FULL_ZK
    assert result_batch.confidence == 1.0
    print(f"✓ Batch verification: {result_batch.latency_ms:.2f}ms, confidence={result_batch.confidence:.2f}")
    
    # Test 4: Optimistic execution
    executed = []
    action_opt = Action(
        id="test-opt-001",
        payload=b"optimistic action",
        urgency=UrgencyLevel.REAL_TIME
    )
    exec_result, verify_task = await engine.verify_optimistic(
        action_opt,
        lambda a: executed.append(a.id)
    )
    assert "test-opt-001" in executed, "Optimistic execution failed"
    print(f"✓ Optimistic execution completed immediately")
    
    # Wait for async verification
    await verify_task
    print(f"✓ Async verification completed")
    
    # Test 5: Metrics
    metrics = engine.get_metrics()
    assert metrics["total_verifications"] >= 3
    print(f"✓ Metrics: {metrics['total_verifications']} verifications, "
          f"avg latency={metrics['avg_latency_ms']:.2f}ms")
    
    print("=" * 50)
    print("All tiered verification tests passed ✓")


if __name__ == "__main__":
    asyncio.run(self_test())
