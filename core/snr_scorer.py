"""
BIZRA AEON OMEGA - Signal-to-Noise Ratio (SNR) Scorer
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Pattern: Information-Theoretic Quality Scoring

Computes SNR to rank insights, thoughts, and knowledge graph elements:
  SNR = Signal_Strength / Noise_Floor
  Signal = Clarity × Synergy × Interdisciplinary_Consistency  
  Noise = Entropy + Quantization_Error + Oracle_Disagreement

Thresholds (calibrated to existing convergence distribution):
  - HIGH: SNR > 0.80 (Top 10% - Breakthrough insights)
  - MEDIUM: 0.50 ≤ SNR ≤ 0.80 (60% - Valuable knowledge)
  - LOW: SNR < 0.50 (Bottom 30% - Requires refinement)

Ethical Constraint: All HIGH-SNR insights must maintain Ihsān metric (IM) ≥ 0.95
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import logging

# BIZRA imports
from core.tiered_verification import QuantizedConvergence, ConvergenceResult


logger = logging.getLogger(__name__)


class SNRLevel(Enum):
    """SNR quality classification."""
    HIGH = auto()       # SNR > 0.80 - Breakthrough insights
    MEDIUM = auto()     # 0.50 ≤ SNR ≤ 0.80 - Valuable knowledge
    LOW = auto()        # SNR < 0.50 - Requires refinement
    UNKNOWN = auto()    # Insufficient data for classification


@dataclass
class SNRMetrics:
    """Comprehensive SNR scoring metrics."""
    
    # Core SNR components
    snr_score: float                    # Overall SNR: signal/noise
    signal_strength: float              # Clarity × Synergy × Consistency
    noise_floor: float                  # Entropy + Quantization + Disagreement
    
    # Signal decomposition
    clarity_component: float            # From QuantizedConvergence
    synergy_component: float            # Neural-symbolic alignment
    consistency_component: float        # Interdisciplinary agreement
    
    # Noise decomposition
    entropy_component: float            # Uncertainty/randomness
    quantization_error: float           # Discretization artifacts
    disagreement_component: float       # Oracle/cross-domain variance
    
    # Classification
    level: SNRLevel                     # HIGH/MEDIUM/LOW/UNKNOWN
    confidence: float                   # Statistical confidence in SNR estimate
    
    # Ethical validation
    ihsan_metric: float                 # IM score (must be ≥ 0.95 for HIGH)
    ethical_override: bool = False      # True if SNR downgraded due to ethics
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    computation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/transmission."""
        return {
            "snr_score": self.snr_score,
            "signal_strength": self.signal_strength,
            "noise_floor": self.noise_floor,
            "clarity": self.clarity_component,
            "synergy": self.synergy_component,
            "consistency": self.consistency_component,
            "entropy": self.entropy_component,
            "quantization_error": self.quantization_error,
            "disagreement": self.disagreement_component,
            "level": self.level.name,
            "confidence": self.confidence,
            "ihsan_metric": self.ihsan_metric,
            "ethical_override": self.ethical_override,
            "timestamp": self.timestamp.isoformat(),
            "computation_time_ms": self.computation_time_ms,
            "metadata": self.metadata
        }


@dataclass
class SNRThresholds:
    """Configurable SNR thresholds with Ihsān constraints."""
    
    high_threshold: float = 0.80        # SNR > 0.80 = HIGH
    medium_threshold: float = 0.50      # 0.50 ≤ SNR ≤ 0.80 = MEDIUM
    min_ihsan_for_high: float = 0.95    # IM ≥ 0.95 required for HIGH
    min_confidence: float = 0.70        # Minimum statistical confidence
    
    def classify(self, snr: float, ihsan: float, confidence: float) -> SNRLevel:
        """
        Classify SNR with ethical constraints.
        
        HIGH requires both SNR > threshold AND Ihsān ≥ 0.95
        This ensures breakthrough insights align with ethical excellence.
        """
        if confidence < self.min_confidence:
            return SNRLevel.UNKNOWN
        
        if snr > self.high_threshold:
            # Ethical constraint: HIGH SNR requires HIGH ethics
            if ihsan >= self.min_ihsan_for_high:
                return SNRLevel.HIGH
            else:
                # Downgrade to MEDIUM if ethics insufficient
                logger.warning(
                    f"SNR {snr:.3f} exceeds HIGH threshold but Ihsān {ihsan:.3f} < {self.min_ihsan_for_high:.3f}. "
                    f"Downgrading to MEDIUM for ethical compliance."
                )
                return SNRLevel.MEDIUM
        elif snr >= self.medium_threshold:
            return SNRLevel.MEDIUM
        else:
            return SNRLevel.LOW


class SNRScorer:
    """
    Signal-to-Noise Ratio scorer for BIZRA insights.
    
    Integrates with:
    - QuantizedConvergence: Provides clarity/synergy/entropy
    - Value Oracle: Provides oracle disagreement metric
    - Consequential Ethics: Provides Ihsān metric (IM)
    - Ultimate Integration: Provides interdisciplinary consistency
    
    Ranking Strategy:
    1. Compute signal = clarity × synergy × consistency
    2. Compute noise = entropy + quantization_error + disagreement  
    3. SNR = signal / (noise + ε) where ε=1e-6 prevents division by zero
    4. Classify with ethical constraints
    """
    
    def __init__(
        self,
        thresholds: Optional[SNRThresholds] = None,
        epsilon: float = 1e-6,
        enable_ethical_constraints: bool = True
    ):
        """
        Initialize SNR scorer.
        
        Args:
            thresholds: SNR classification thresholds
            epsilon: Small constant to prevent division by zero
            enable_ethical_constraints: Apply Ihsān constraints to HIGH classification
        """
        self.thresholds = thresholds or SNRThresholds()
        self.epsilon = epsilon
        self.enable_ethical_constraints = enable_ethical_constraints
        
        logger.info(
            f"SNRScorer initialized: HIGH={self.thresholds.high_threshold}, "
            f"MEDIUM={self.thresholds.medium_threshold}, "
            f"Ihsān_constraint={'enabled' if enable_ethical_constraints else 'disabled'}"
        )
    
    @staticmethod
    def _quality_name(quality: Any) -> str:
        """Normalize convergence quality to a string label."""
        if quality is None:
            return "UNKNOWN"
        if hasattr(quality, "name"):
            return quality.name
        if isinstance(quality, str):
            return quality
        return str(quality)

    def compute_from_convergence(
        self,
        convergence: ConvergenceResult,
        consistency: float,
        disagreement: float,
        ihsan_metric: float
    ) -> SNRMetrics:
        """
        Compute SNR from QuantizedConvergence result.
        
        Args:
            convergence: Result from QuantizedConvergence.compute()
            consistency: Interdisciplinary consistency [0,1]
            disagreement: Oracle disagreement metric [0,1]
            ihsan_metric: Ethical Ihsān metric (IM) [0,1]
        
        Returns:
            SNRMetrics with full decomposition and classification
        """
        start_time = time.perf_counter()
        
        # Extract components from convergence
        clarity = convergence.clarity
        synergy = convergence.synergy
        entropy = convergence.entropy
        quantization = convergence.quantization_error
        
        # Compute signal strength (product of positive indicators)
        signal = clarity * synergy * consistency
        
        # Compute noise floor (sum of negative indicators)
        noise = entropy + quantization + disagreement + self.epsilon
        
        # Core SNR computation
        snr = signal / noise
        
        # Statistical confidence from convergence quality
        confidence = self._estimate_confidence(convergence, consistency)
        quality_name = self._quality_name(convergence.quality)
        
        # Classification with ethical constraints
        level = self.thresholds.classify(snr, ihsan_metric, confidence)
        
        # Check if ethical override occurred
        ethical_override = False
        if self.enable_ethical_constraints:
            if snr > self.thresholds.high_threshold and ihsan_metric < self.thresholds.min_ihsan_for_high:
                ethical_override = True
        
        computation_time = (time.perf_counter() - start_time) * 1000
        
        metrics = SNRMetrics(
            snr_score=snr,
            signal_strength=signal,
            noise_floor=noise,
            clarity_component=clarity,
            synergy_component=synergy,
            consistency_component=consistency,
            entropy_component=entropy,
            quantization_error=quantization,
            disagreement_component=disagreement,
            level=level,
            confidence=confidence,
            ihsan_metric=ihsan_metric,
            ethical_override=ethical_override,
            computation_time_ms=computation_time,
            metadata={
                "convergence_quality": quality_name,
                "original_snr_before_ethics": snr,
                "epsilon": self.epsilon
            }
        )
        
        logger.debug(
            f"SNR computed: {snr:.4f} ({level.name}) | "
            f"Signal={signal:.4f}, Noise={noise:.4f}, IM={ihsan_metric:.4f}"
        )
        
        return metrics
    
    def compute_batch(
        self,
        convergence_results: List[ConvergenceResult],
        consistency_scores: List[float],
        disagreement_scores: List[float],
        ihsan_metrics: List[float]
    ) -> List[SNRMetrics]:
        """
        Batch SNR computation for multiple results.
        
        Efficient for ranking large sets of insights from knowledge graph.
        """
        if not (len(convergence_results) == len(consistency_scores) == 
                len(disagreement_scores) == len(ihsan_metrics)):
            raise ValueError("All input lists must have same length")
        
        return [
            self.compute_from_convergence(conv, cons, dis, im)
            for conv, cons, dis, im in zip(
                convergence_results, 
                consistency_scores, 
                disagreement_scores,
                ihsan_metrics
            )
        ]
    
    def rank_by_snr(
        self,
        metrics_list: List[SNRMetrics],
        top_k: Optional[int] = None,
        min_level: Optional[SNRLevel] = None
    ) -> List[Tuple[int, SNRMetrics]]:
        """
        Rank metrics by SNR score with optional filtering.
        
        Args:
            metrics_list: List of SNR metrics to rank
            top_k: Return only top K results (None = all)
            min_level: Filter to minimum SNR level (None = no filter)
        
        Returns:
            List of (original_index, metrics) tuples sorted by SNR (descending)
        """
        # Enumerate to preserve original indices
        indexed = list(enumerate(metrics_list))
        
        # Filter by minimum level if specified
        if min_level is not None:
            level_order = {
                SNRLevel.HIGH: 3,
                SNRLevel.MEDIUM: 2,
                SNRLevel.LOW: 1,
                SNRLevel.UNKNOWN: 0
            }
            min_order = level_order[min_level]
            indexed = [
                (idx, m) for idx, m in indexed 
                if level_order[m.level] >= min_order
            ]
        
        # Sort by SNR descending
        ranked = sorted(indexed, key=lambda x: x[1].snr_score, reverse=True)
        
        # Limit to top K if specified
        if top_k is not None:
            ranked = ranked[:top_k]
        
        return ranked
    
    def _estimate_confidence(
        self,
        convergence: ConvergenceResult,
        consistency: float
    ) -> float:
        """
        Estimate statistical confidence in SNR measurement.
        
        High confidence requires:
        - Good convergence quality
        - High interdisciplinary consistency
        - Low quantization error
        """
        # Map convergence quality to base confidence
        quality_confidence = {
            "OPTIMAL": 0.95,
            "EXCELLENT": 0.90,
            "GOOD": 0.80,
            "ACCEPTABLE": 0.70,
            "POOR": 0.50
        }
        
        quality_name = self._quality_name(convergence.quality)
        base = quality_confidence.get(quality_name, 0.50)
        
        # Adjust for consistency (consistent cross-domain = higher confidence)
        consistency_factor = 0.5 + (0.5 * consistency)
        
        # Adjust for quantization error (low error = higher confidence)
        quantization_factor = 1.0 - (0.3 * convergence.quantization_error)
        
        confidence = base * consistency_factor * quantization_factor
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def get_high_snr_insights(
        self,
        metrics_list: List[SNRMetrics],
        top_k: int = 10
    ) -> List[Tuple[int, SNRMetrics]]:
        """
        Convenience method to get top HIGH-SNR insights.
        
        Returns top K insights with SNRLevel.HIGH, ranked by score.
        Useful for surfacing breakthrough discoveries.
        """
        return self.rank_by_snr(
            metrics_list,
            top_k=top_k,
            min_level=SNRLevel.HIGH
        )


# Convenience function for quick SNR computation
def compute_snr(
    clarity: float,
    synergy: float,
    consistency: float,
    entropy: float,
    quantization_error: float,
    disagreement: float,
    ihsan_metric: float = 1.0,
    epsilon: float = 1e-6
) -> float:
    """
    Quick SNR computation without full metrics.
    
    Useful for inline calculations in performance-critical paths.
    """
    signal = clarity * synergy * consistency
    noise = entropy + quantization_error + disagreement + epsilon
    return signal / noise


# Export public API
__all__ = [
    "SNRLevel",
    "SNRMetrics",
    "SNRThresholds",
    "SNRScorer",
    "compute_snr"
]
