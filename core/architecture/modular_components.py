"""
BIZRA AEON OMEGA - Modular Architecture Components
Refactored from BIZRAVCCNode0Ultimate (God Class Decomposition)

This module provides Single Responsibility components that replace
the monolithic BIZRAVCCNode0Ultimate class with focused, testable units.

Addresses: Codebase Architecture recommendation to break down god classes

Components:
- CognitiveProcessor: Handles quantized convergence computation
- VerificationCoordinator: Manages tiered verification
- ValueAssessor: Handles pluralistic value assessment
- EthicsEvaluator: Manages consequential ethics
- HealthMonitorComponent: Monitors system health
- NarrativeGenerator: Generates human-readable explanations
- UltimateOrchestrator: Lightweight coordinator for all components

Author: BIZRA Architecture Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar

# Local imports for memory management
from core.memory.memory_management import BoundedList, SlidingWindowStats

logger = logging.getLogger("bizra.architecture")


# =============================================================================
# PROTOCOLS (Interface Contracts)
# =============================================================================

class Processor(Protocol):
    """Protocol for processing components."""
    
    async def process(self, data: Any) -> Any:
        """Process input data and return result."""
        ...


class Observer(Protocol):
    """Protocol for observer pattern."""
    
    def on_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle an event."""
        ...


# =============================================================================
# SHARED DATA STRUCTURES
# =============================================================================

class UrgencyLevel(Enum):
    """Processing urgency levels."""
    IMMEDIATE = auto()
    NEAR_REAL_TIME = auto()
    DEFERRED = auto()
    BACKGROUND = auto()


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = auto()
    DEGRADED = auto()
    CRITICAL = auto()
    RECOVERING = auto()


class ConvergenceQuality(Enum):
    """Convergence quality classification."""
    OPTIMAL = auto()
    EXCELLENT = auto()
    GOOD = auto()
    ACCEPTABLE = auto()
    POOR = auto()


@dataclass
class Observation:
    """Input observation to the cognitive system."""
    id: str
    data: bytes
    urgency: UrgencyLevel = UrgencyLevel.NEAR_REAL_TIME
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConvergenceResult:
    """Result of quantized convergence computation."""
    clarity: float
    mutual_information: float
    entropy: float
    synergy: float
    quantization_error: float
    quality: ConvergenceQuality
    action: Dict[str, Any]
    computation_time_ms: float = 0.0


@dataclass
class VerificationResult:
    """Result of tiered verification."""
    verified: bool
    tier: str
    proof_hash: str
    confidence: float
    latency_ms: float


@dataclass
class ValueResult:
    """Result of value assessment."""
    composite_score: float
    oracle_scores: Dict[str, float]
    convergence: float
    recommendations: List[str]


@dataclass
class EthicsResult:
    """Result of ethics evaluation."""
    permissible: bool
    severity: str
    considerations: List[str]
    confidence: float


@dataclass
class ProcessingResult:
    """Complete processing result."""
    action: Dict[str, Any]
    confidence: float
    convergence: ConvergenceResult
    verification: VerificationResult
    value: ValueResult
    ethics: EthicsResult
    health: HealthStatus
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# COMPONENT: COGNITIVE PROCESSOR (Single Responsibility: Convergence)
# =============================================================================

class CognitiveProcessor:
    """
    Handles quantized convergence computation.
    
    Single Responsibility: Mathematical processing of neural-symbolic convergence.
    
    Theory: dC/dt = α·I - β·H + γ·Synergy - δ·Qerr
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.5,
        delta: float = 10.0
    ):
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._fp_precision = 2**-23
        self._q_step = 1/1024
        
        # Bounded history for metrics
        self._computation_history = BoundedList[float](max_size=1000)
        self._quality_history = BoundedList[ConvergenceQuality](max_size=100)
    
    async def compute_convergence(self, observation: Observation) -> ConvergenceResult:
        """Compute quantized convergence for observation."""
        start = time.perf_counter()
        
        # Extract features
        neural = self._extract_neural_features(observation.data)
        symbolic = self._project_to_symbolic(neural)
        
        # Compute metrics
        mi = self._mutual_information(neural, symbolic)
        entropy = self._entropy(symbolic)
        synergy = self._synergy(neural, symbolic)
        q_error = self._quantization_error(neural, symbolic)
        
        # Compute clarity
        clarity = (
            self._alpha * mi -
            self._beta * entropy +
            self._gamma * synergy -
            self._delta * q_error
        )
        clarity = max(0.0, min(1.0, (clarity + 0.5) / 1.5))
        
        quality = self._classify_quality(synergy)
        action = self._generate_action(observation, clarity, synergy)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Record in bounded history
        self._computation_history.append(elapsed_ms)
        self._quality_history.append(quality)
        
        return ConvergenceResult(
            clarity=clarity,
            mutual_information=mi,
            entropy=entropy,
            synergy=synergy,
            quantization_error=q_error,
            quality=quality,
            action=action,
            computation_time_ms=elapsed_ms
        )
    
    def _extract_neural_features(self, data: bytes) -> List[float]:
        h = hashlib.sha256(data).digest()
        return [b / 255.0 for b in h[:32]]
    
    def _project_to_symbolic(self, features: List[float]) -> List[int]:
        return [int(f * 255) for f in features]
    
    def _mutual_information(self, neural: List[float], symbolic: List[int]) -> float:
        if not neural:
            return 0.0
        correlation = sum(n * (s / 255.0) for n, s in zip(neural, symbolic))
        return correlation / len(neural)
    
    def _entropy(self, symbolic: List[int]) -> float:
        if not symbolic:
            return 0.0
        counts: Dict[int, int] = {}
        for s in symbolic:
            counts[s] = counts.get(s, 0) + 1
        total = len(symbolic)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy / math.log2(256)
    
    def _synergy(self, neural: List[float], symbolic: List[int]) -> float:
        if not neural or not symbolic:
            return 0.0
        neural_norm = sum(n**2 for n in neural) ** 0.5
        symbolic_norm = sum((s/255)**2 for s in symbolic) ** 0.5
        if neural_norm == 0 or symbolic_norm == 0:
            return 0.0
        dot_product = sum(n * (s/255) for n, s in zip(neural, symbolic))
        cosine_sim = dot_product / (neural_norm * symbolic_norm)
        return (cosine_sim + 1) / 2
    
    def _quantization_error(self, neural: List[float], symbolic: List[int]) -> float:
        if not neural or not symbolic:
            return 0.0
        total_error = 0.0
        for n, s in zip(neural, symbolic):
            reconstructed = s / 255.0
            total_error += abs(n - reconstructed)
        avg_error = total_error / len(neural)
        precision_factor = abs(self._fp_precision - self._q_step)
        return avg_error * precision_factor
    
    def _classify_quality(self, synergy: float) -> ConvergenceQuality:
        if synergy > 0.95:
            return ConvergenceQuality.OPTIMAL
        elif synergy > 0.85:
            return ConvergenceQuality.EXCELLENT
        elif synergy > 0.70:
            return ConvergenceQuality.GOOD
        elif synergy > 0.50:
            return ConvergenceQuality.ACCEPTABLE
        return ConvergenceQuality.POOR
    
    def _generate_action(self, obs: Observation, clarity: float, synergy: float) -> Dict[str, Any]:
        return {
            "type": "cognitive_response",
            "observation_id": obs.id,
            "clarity": clarity,
            "synergy": synergy,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        history = self._computation_history.to_list()
        quality_list = self._quality_history.to_list()
        
        optimal_count = sum(1 for q in quality_list if q == ConvergenceQuality.OPTIMAL)
        
        return {
            "computation_count": len(history),
            "avg_computation_ms": sum(history) / max(1, len(history)),
            "optimal_rate": optimal_count / max(1, len(quality_list)),
            "history_stats": self._computation_history.stats
        }


# =============================================================================
# COMPONENT: VERIFICATION COORDINATOR (Single Responsibility: Verification)
# =============================================================================

class VerificationCoordinator:
    """
    Manages tiered verification with urgency-aware tier selection.
    
    Single Responsibility: Cryptographic verification coordination.
    """
    
    TIER_THRESHOLDS = {
        UrgencyLevel.IMMEDIATE: ("L1_HASH", 5),
        UrgencyLevel.NEAR_REAL_TIME: ("L2_MERKLE", 50),
        UrgencyLevel.DEFERRED: ("L3_ZK", 500),
        UrgencyLevel.BACKGROUND: ("L4_FORMAL", 5000)
    }
    
    def __init__(self):
        self._verification_stats = SlidingWindowStats(window_seconds=300.0)
        self._tier_usage: Dict[str, int] = {}
    
    async def verify(
        self,
        observation: Observation,
        convergence: ConvergenceResult
    ) -> VerificationResult:
        """Verify with urgency-appropriate tier."""
        tier, max_latency = self.TIER_THRESHOLDS.get(
            observation.urgency,
            ("L2_MERKLE", 50)
        )
        
        start = time.perf_counter()
        
        # Compute verification based on tier
        if tier == "L1_HASH":
            proof_hash, verified = await self._l1_hash_verify(observation, convergence)
        elif tier == "L2_MERKLE":
            proof_hash, verified = await self._l2_merkle_verify(observation, convergence)
        elif tier == "L3_ZK":
            proof_hash, verified = await self._l3_zk_verify(observation, convergence)
        else:
            proof_hash, verified = await self._l4_formal_verify(observation, convergence)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Record stats
        self._verification_stats.record(elapsed_ms)
        self._tier_usage[tier] = self._tier_usage.get(tier, 0) + 1
        
        return VerificationResult(
            verified=verified,
            tier=tier,
            proof_hash=proof_hash,
            confidence=convergence.clarity if verified else 0.0,
            latency_ms=elapsed_ms
        )
    
    async def _l1_hash_verify(
        self, obs: Observation, conv: ConvergenceResult
    ) -> Tuple[str, bool]:
        """Fast hash-based verification."""
        data = obs.data + str(conv.clarity).encode()
        proof = hashlib.sha256(data).hexdigest()
        return proof, True
    
    async def _l2_merkle_verify(
        self, obs: Observation, conv: ConvergenceResult
    ) -> Tuple[str, bool]:
        """Merkle tree verification."""
        await asyncio.sleep(0.001)  # Simulate Merkle computation
        leaves = [obs.data, str(conv).encode()]
        leaf_hashes = [hashlib.sha256(l).hexdigest() for l in leaves]
        root = hashlib.sha256("".join(leaf_hashes).encode()).hexdigest()
        return root, True
    
    async def _l3_zk_verify(
        self, obs: Observation, conv: ConvergenceResult
    ) -> Tuple[str, bool]:
        """Zero-knowledge proof verification."""
        await asyncio.sleep(0.05)  # Simulate ZK proof
        proof = hashlib.sha512(obs.data + str(conv.synergy).encode()).hexdigest()
        return proof[:64], conv.synergy > 0.5
    
    async def _l4_formal_verify(
        self, obs: Observation, conv: ConvergenceResult
    ) -> Tuple[str, bool]:
        """Formal verification (most thorough)."""
        await asyncio.sleep(0.1)  # Simulate formal verification
        proof = hashlib.blake2b(obs.data).hexdigest()
        return proof, conv.quality in (ConvergenceQuality.OPTIMAL, ConvergenceQuality.EXCELLENT)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "tier_usage": self._tier_usage,
            "latency_stats": self._verification_stats.stats()
        }


# =============================================================================
# COMPONENT: VALUE ASSESSOR (Single Responsibility: Value Assessment)
# =============================================================================

class ValueAssessor:
    """
    Handles pluralistic value assessment.
    
    Single Responsibility: Multi-oracle value estimation.
    """
    
    # Default Ihsān weights
    DEFAULT_WEIGHTS = {
        "adl": 0.30,       # Justice
        "ihsan": 0.25,     # Excellence
        "hikmah": 0.20,    # Wisdom
        "amanah": 0.15,    # Trust
        "rahmah": 0.10     # Mercy
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self._weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._assessment_history = BoundedList[float](max_size=500)
    
    async def assess(self, convergence: ConvergenceResult) -> ValueResult:
        """Assess value based on convergence result."""
        # Compute oracle scores
        oracle_scores = {
            "adl": self._assess_justice(convergence),
            "ihsan": self._assess_excellence(convergence),
            "hikmah": self._assess_wisdom(convergence),
            "amanah": self._assess_trust(convergence),
            "rahmah": self._assess_mercy(convergence)
        }
        
        # Compute weighted composite
        composite = sum(
            self._weights.get(k, 0.0) * v
            for k, v in oracle_scores.items()
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(oracle_scores, composite)
        
        # Record
        self._assessment_history.append(composite)
        
        return ValueResult(
            composite_score=composite,
            oracle_scores=oracle_scores,
            convergence=convergence.synergy,
            recommendations=recommendations
        )
    
    def _assess_justice(self, conv: ConvergenceResult) -> float:
        # Justice correlates with balanced processing
        return 0.8 + 0.2 * (1 - conv.quantization_error * 100)
    
    def _assess_excellence(self, conv: ConvergenceResult) -> float:
        # Excellence correlates with synergy
        return min(1.0, conv.synergy + 0.1)
    
    def _assess_wisdom(self, conv: ConvergenceResult) -> float:
        # Wisdom correlates with clarity
        return conv.clarity
    
    def _assess_trust(self, conv: ConvergenceResult) -> float:
        # Trust correlates with low entropy
        return 1.0 - conv.entropy * 0.5
    
    def _assess_mercy(self, conv: ConvergenceResult) -> float:
        # Mercy is base level of consideration
        return 0.85
    
    def _generate_recommendations(
        self, scores: Dict[str, float], composite: float
    ) -> List[str]:
        recommendations = []
        if composite < 0.7:
            recommendations.append("Consider increasing process transparency")
        if scores.get("adl", 0) < 0.8:
            recommendations.append("Review fairness considerations")
        if scores.get("hikmah", 0) < 0.7:
            recommendations.append("Seek additional context for decision")
        return recommendations


# =============================================================================
# COMPONENT: ETHICS EVALUATOR (Single Responsibility: Ethics)
# =============================================================================

class EthicsEvaluator:
    """
    Manages consequential ethics evaluation.
    
    Single Responsibility: Ethical permissibility assessment.
    """
    
    def __init__(self):
        self._evaluation_history = BoundedList[bool](max_size=500)
    
    async def evaluate(
        self,
        convergence: ConvergenceResult,
        observation: Observation
    ) -> EthicsResult:
        """Evaluate ethical permissibility."""
        considerations = []
        severity_score = 0.0
        
        # Check urgency ethics
        if observation.urgency == UrgencyLevel.IMMEDIATE:
            considerations.append("High urgency may limit deliberation time")
            severity_score += 0.1
        
        # Check convergence quality ethics
        if convergence.quality == ConvergenceQuality.POOR:
            considerations.append("Low convergence quality raises concerns")
            severity_score += 0.3
        
        # Check entropy ethics
        if convergence.entropy > 0.8:
            considerations.append("High uncertainty in decision basis")
            severity_score += 0.2
        
        # Determine severity level
        if severity_score < 0.2:
            severity = "NONE"
        elif severity_score < 0.4:
            severity = "LOW"
        elif severity_score < 0.6:
            severity = "MEDIUM"
        else:
            severity = "HIGH"
        
        # Determine permissibility
        permissible = (
            convergence.clarity > 0.3 and
            convergence.synergy > 0.4 and
            severity_score < 0.6
        )
        
        confidence = 1.0 - severity_score
        
        self._evaluation_history.append(permissible)
        
        return EthicsResult(
            permissible=permissible,
            severity=severity,
            considerations=considerations,
            confidence=confidence
        )
    
    def get_permissibility_rate(self) -> float:
        history = self._evaluation_history.to_list()
        if not history:
            return 1.0
        return sum(1 for p in history if p) / len(history)


# =============================================================================
# COMPONENT: HEALTH MONITOR (Single Responsibility: Health)
# =============================================================================

class HealthMonitorComponent:
    """
    Monitors system health.
    
    Single Responsibility: Health status tracking and alerting.
    """
    
    def __init__(
        self,
        latency_threshold_ms: float = 500.0,
        error_rate_threshold: float = 0.1,
        success_threshold: float = 0.9
    ):
        self._latency_threshold = latency_threshold_ms
        self._error_threshold = error_rate_threshold
        self._success_threshold = success_threshold
        
        # Use bounded metrics
        self._latency_window = SlidingWindowStats(window_seconds=60.0)
        self._error_window = SlidingWindowStats(window_seconds=60.0)
        self._success_window = SlidingWindowStats(window_seconds=60.0)
        
        self._current_status = HealthStatus.HEALTHY
        self._observers: List[Observer] = []
    
    def record_latency(self, latency_ms: float) -> None:
        self._latency_window.record(latency_ms)
    
    def record_error(self) -> None:
        self._error_window.record(1.0)
    
    def record_success(self, verified: bool) -> None:
        self._success_window.record(1.0 if verified else 0.0)
    
    def add_observer(self, observer: Observer) -> None:
        self._observers.append(observer)
    
    def evaluate_health(self) -> HealthStatus:
        """Evaluate and return current health status."""
        issues = 0
        
        # Check latency
        avg_latency = self._latency_window.mean() or 0.0
        if avg_latency > self._latency_threshold:
            issues += 1
        
        # Check errors
        error_count = self._error_window.count()
        total_ops = self._success_window.count() + error_count
        if total_ops > 0 and error_count / total_ops > self._error_threshold:
            issues += 2
        
        # Check success rate
        success_mean = self._success_window.mean() or 1.0
        if success_mean < self._success_threshold:
            issues += 1
        
        # Determine status
        previous_status = self._current_status
        
        if issues >= 3:
            self._current_status = HealthStatus.CRITICAL
        elif issues >= 1:
            self._current_status = HealthStatus.DEGRADED
        else:
            self._current_status = HealthStatus.HEALTHY
        
        # Notify observers on status change
        if self._current_status != previous_status:
            self._notify_observers("health_change", {
                "previous": previous_status.name,
                "current": self._current_status.name,
                "issues": issues
            })
        
        return self._current_status
    
    def _notify_observers(self, event_type: str, data: Dict[str, Any]) -> None:
        for observer in self._observers:
            try:
                observer.on_event(event_type, data)
            except Exception as e:
                logger.warning(f"Observer notification failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "status": self._current_status.name,
            "latency": self._latency_window.stats(),
            "error_count": self._error_window.count(),
            "success_rate": self._success_window.mean() or 1.0
        }


# =============================================================================
# COMPONENT: NARRATIVE GENERATOR (Single Responsibility: Explanation)
# =============================================================================

class NarrativeGenerator:
    """
    Generates human-readable explanations.
    
    Single Responsibility: Transform technical results into narratives.
    """
    
    def generate(
        self,
        convergence: ConvergenceResult,
        verification: VerificationResult,
        value: ValueResult,
        ethics: EthicsResult,
        health: HealthStatus
    ) -> str:
        """Generate comprehensive explanation."""
        sections = []
        
        # Convergence summary
        sections.append(
            f"Cognitive Analysis: {convergence.quality.name} quality achieved "
            f"with {convergence.clarity:.1%} clarity and {convergence.synergy:.1%} synergy."
        )
        
        # Verification summary
        status = "verified" if verification.verified else "unverified"
        sections.append(
            f"Verification: {status} via {verification.tier} "
            f"({verification.latency_ms:.1f}ms)."
        )
        
        # Value summary
        sections.append(
            f"Value Assessment: {value.composite_score:.1%} composite score. "
            f"Key: {', '.join(f'{k}={v:.0%}' for k, v in value.oracle_scores.items())}."
        )
        
        # Ethics summary
        permission = "PERMISSIBLE" if ethics.permissible else "REQUIRES REVIEW"
        sections.append(
            f"Ethics: {permission} (severity: {ethics.severity}). "
            f"{' '.join(ethics.considerations) if ethics.considerations else 'No concerns.'}"
        )
        
        # Health summary
        sections.append(f"System Health: {health.name}")
        
        # Recommendations
        if value.recommendations:
            sections.append(f"Recommendations: {'; '.join(value.recommendations)}")
        
        return " | ".join(sections)


# =============================================================================
# ORCHESTRATOR: ULTIMATE ORCHESTRATOR (Composition Root)
# =============================================================================

class UltimateOrchestrator:
    """
    Lightweight orchestrator that coordinates all components.
    
    This replaces the monolithic BIZRAVCCNode0Ultimate with proper
    composition and dependency injection.
    
    Responsibilities:
    - Component lifecycle management
    - Request routing and coordination
    - Result aggregation
    """
    
    def __init__(
        self,
        cognitive_processor: Optional[CognitiveProcessor] = None,
        verification_coordinator: Optional[VerificationCoordinator] = None,
        value_assessor: Optional[ValueAssessor] = None,
        ethics_evaluator: Optional[EthicsEvaluator] = None,
        health_monitor: Optional[HealthMonitorComponent] = None,
        narrative_generator: Optional[NarrativeGenerator] = None
    ):
        """
        Initialize with dependency injection.
        
        All parameters are optional for easy testing and customization.
        """
        self._cognitive = cognitive_processor or CognitiveProcessor()
        self._verification = verification_coordinator or VerificationCoordinator()
        self._value = value_assessor or ValueAssessor()
        self._ethics = ethics_evaluator or EthicsEvaluator()
        self._health = health_monitor or HealthMonitorComponent()
        self._narrative = narrative_generator or NarrativeGenerator()
        
        # Bounded processing history
        self._processing_count = 0
        self._singularity_events = BoundedList[Dict[str, Any]](max_size=100)
    
    async def process(self, observation: Observation) -> ProcessingResult:
        """
        Process observation through the complete cognitive pipeline.
        
        This is the entry point that coordinates all components.
        """
        start_time = time.perf_counter()
        
        try:
            # 1. Cognitive processing
            convergence = await self._cognitive.compute_convergence(observation)
            
            # 2. Verification
            verification = await self._verification.verify(observation, convergence)
            
            # 3. Value assessment
            value = await self._value.assess(convergence)
            
            # 4. Ethics evaluation
            ethics = await self._ethics.evaluate(convergence, observation)
            
            # 5. Health monitoring
            self._health.record_latency(convergence.computation_time_ms)
            self._health.record_success(verification.verified)
            health = self._health.evaluate_health()
            
            # 6. Narrative generation
            explanation = self._narrative.generate(
                convergence, verification, value, ethics, health
            )
            
            # 7. Singularity detection
            if convergence.synergy > 0.999:
                self._record_singularity(observation, convergence)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            self._processing_count += 1
            
            return ProcessingResult(
                action=convergence.action,
                confidence=convergence.clarity,
                convergence=convergence,
                verification=verification,
                value=value,
                ethics=ethics,
                health=health,
                explanation=explanation,
                metadata={
                    "processing_count": self._processing_count,
                    "singularity_count": len(self._singularity_events)
                },
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self._health.record_error()
            logger.error(f"Processing error: {e}")
            raise
    
    def _record_singularity(
        self, observation: Observation, convergence: ConvergenceResult
    ) -> None:
        """Record singularity event."""
        event = {
            "observation_id": observation.id,
            "synergy": convergence.synergy,
            "clarity": convergence.clarity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self._singularity_events.append(event)
        logger.info(f"Singularity event recorded: {convergence.synergy:.4f}")
    
    def get_component_stats(self) -> Dict[str, Any]:
        """Get statistics from all components."""
        return {
            "cognitive": self._cognitive.get_stats(),
            "verification": self._verification.get_stats(),
            "health": self._health.get_metrics(),
            "ethics_permissibility_rate": self._ethics.get_permissibility_rate(),
            "processing_count": self._processing_count,
            "singularity_events": len(self._singularity_events)
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_default_orchestrator() -> UltimateOrchestrator:
    """Create orchestrator with default components."""
    return UltimateOrchestrator()


def create_custom_orchestrator(
    ihsan_weights: Optional[Dict[str, float]] = None,
    convergence_config: Optional[Dict[str, float]] = None
) -> UltimateOrchestrator:
    """Create orchestrator with custom configuration."""
    cognitive = CognitiveProcessor(
        **convergence_config
    ) if convergence_config else CognitiveProcessor()
    
    value = ValueAssessor(weights=ihsan_weights)
    
    return UltimateOrchestrator(
        cognitive_processor=cognitive,
        value_assessor=value
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    "UrgencyLevel",
    "HealthStatus",
    "ConvergenceQuality",
    "Observation",
    "ConvergenceResult",
    "VerificationResult",
    "ValueResult",
    "EthicsResult",
    "ProcessingResult",
    
    # Components
    "CognitiveProcessor",
    "VerificationCoordinator",
    "ValueAssessor",
    "EthicsEvaluator",
    "HealthMonitorComponent",
    "NarrativeGenerator",
    
    # Orchestrator
    "UltimateOrchestrator",
    
    # Factory
    "create_default_orchestrator",
    "create_custom_orchestrator",
]
