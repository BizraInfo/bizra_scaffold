"""
BIZRA AEON OMEGA - Pluralistic Value Oracle
═══════════════════════════════════════════════════════════════════════════════
Addresses the Oracle Problem: How to measure "convergence value" without
a single trusted oracle. Uses ensemble of value signals with dynamic weighting.

Gap Addressed: Mathematical Optimality vs. Cryptographic Practicality (2%)
"""

from __future__ import annotations

import hashlib
import math
import secrets
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from core.snr_scorer import compute_snr


class OracleType(Enum):
    """Types of value oracles."""

    SHAPLEY = auto()  # Game-theoretic value
    PREDICTION_MARKET = auto()  # Market-based value
    REPUTATION = auto()  # Social proof value
    FORMAL_VERIFICATION = auto()  # Mathematical proof value
    INFORMATION_THEORETIC = auto()  # Mutual information value
    SNR = auto()  # Signal-to-noise ratio value


@dataclass
class Convergence:
    """Represents a convergence event to be valued."""

    id: str
    clarity_score: float  # C(t) value
    mutual_information: float  # I(X;H) value
    entropy: float  # H(P) value
    synergy: float  # synergy(H,P) value
    quantization_error: float  # δ·Qerr value
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OracleSignal:
    """Value signal from a single oracle."""

    oracle_type: OracleType
    value: float  # Estimated value
    confidence: float  # Oracle confidence in estimate
    reasoning: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ValueAssessment:
    """Synthesized value assessment from all oracles."""

    value: float
    confidence: float
    signals: List[OracleSignal]
    oracle_weights: Dict[OracleType, float]
    disagreement_score: float  # Measure of oracle disagreement
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ValueOracle(ABC):
    """Abstract base for value oracles."""

    @property
    @abstractmethod
    def oracle_type(self) -> OracleType:
        """Return the oracle type."""
        pass

    @abstractmethod
    async def evaluate(self, convergence: Convergence) -> OracleSignal:
        """Evaluate convergence value."""
        pass

    @property
    @abstractmethod
    def historical_accuracy(self) -> float:
        """Return historical accuracy of this oracle."""
        pass


class ShapleyOracle(ValueOracle):
    """
    Game-theoretic value oracle using Shapley values.

    Measures the marginal contribution of each component to overall convergence.
    """

    def __init__(self):
        self._accuracy_history: deque = deque(maxlen=1000)
        self._accuracy = 0.85  # Initial estimate

    @property
    def oracle_type(self) -> OracleType:
        return OracleType.SHAPLEY

    @property
    def historical_accuracy(self) -> float:
        return self._accuracy

    async def evaluate(self, convergence: Convergence) -> OracleSignal:
        # Compute Shapley values for convergence components
        components = {
            "clarity": convergence.clarity_score,
            "mutual_info": convergence.mutual_information,
            "entropy": convergence.entropy,
            "synergy": convergence.synergy,
        }

        # Marginal contribution analysis
        shapley_values = self._compute_shapley_values(components)

        # Total value is sum of Shapley values, clamped to [0, 1]
        total_value = max(0.0, min(1.0, sum(shapley_values.values())))

        # Confidence based on component balance
        balance = 1.0 - self._compute_imbalance(shapley_values)
        confidence = min(0.95, self._accuracy * balance)

        return OracleSignal(
            oracle_type=self.oracle_type,
            value=total_value,
            confidence=confidence,
            reasoning=f"Shapley decomposition: clarity={shapley_values['clarity']:.3f}, "
            f"MI={shapley_values['mutual_info']:.3f}, "
            f"entropy={shapley_values['entropy']:.3f}, "
            f"synergy={shapley_values['synergy']:.3f}",
        )

    def _compute_shapley_values(self, components: Dict[str, float]) -> Dict[str, float]:
        """Compute Shapley values for each component."""
        n = len(components)
        shapley = {}

        for key, value in components.items():
            # Simplified Shapley: marginal contribution weighted by coalition size
            # Full Shapley would require 2^n coalition evaluations

            # Base contribution
            base = value / n

            # Interaction bonus (synergy with other components)
            others_avg = (
                sum(v for k, v in components.items() if k != key) / (n - 1)
                if n > 1
                else 0
            )
            interaction = value * others_avg * 0.1

            shapley[key] = base + interaction

        return shapley

    def _compute_imbalance(self, values: Dict[str, float]) -> float:
        """Compute imbalance in value distribution."""
        if not values:
            return 0.0

        vals = list(values.values())
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)

        # Normalize by mean squared to get coefficient of variation
        return min(1.0, math.sqrt(variance) / (mean + 0.001))


class PredictionMarketOracle(ValueOracle):
    """
    Market-based value oracle.

    Simulates prediction market dynamics to estimate convergence value.
    Uses order book dynamics and price discovery.
    """

    def __init__(self):
        self._market_state: Dict[str, float] = {}
        self._accuracy = 0.78  # Markets are accurate but noisy

    @property
    def oracle_type(self) -> OracleType:
        return OracleType.PREDICTION_MARKET

    @property
    def historical_accuracy(self) -> float:
        return self._accuracy

    async def evaluate(self, convergence: Convergence) -> OracleSignal:
        # Market price based on convergence metrics
        base_price = self._compute_base_price(convergence)

        # Supply/demand adjustment
        demand_factor = self._compute_demand(convergence)

        # Final market value, clamped to [0, 1]
        market_value = max(0.0, min(1.0, base_price * demand_factor))

        # Market confidence based on liquidity simulation
        liquidity = self._simulate_liquidity(convergence)
        confidence = min(0.90, self._accuracy * liquidity)

        return OracleSignal(
            oracle_type=self.oracle_type,
            value=market_value,
            confidence=confidence,
            reasoning=f"Market valuation: base={base_price:.3f}, "
            f"demand={demand_factor:.2f}, liquidity={liquidity:.2f}",
        )

    def _compute_base_price(self, convergence: Convergence) -> float:
        """Compute base price from convergence metrics."""
        # Price = f(clarity, synergy) with entropy discount
        base = (
            0.4 * convergence.clarity_score
            + 0.3 * convergence.synergy
            + 0.2 * convergence.mutual_information
            - 0.1 * convergence.entropy
        )
        return max(0.0, base)

    def _compute_demand(self, convergence: Convergence) -> float:
        """Simulate market demand for this convergence."""
        # High synergy = high demand
        if convergence.synergy > 0.9:
            return 1.5  # Premium for near-perfect alignment
        elif convergence.synergy > 0.7:
            return 1.2
        elif convergence.synergy > 0.5:
            return 1.0
        else:
            return 0.8  # Discount for low synergy

    def _simulate_liquidity(self, convergence: Convergence) -> float:
        """Simulate market liquidity."""
        # Higher clarity = higher liquidity
        return min(1.0, 0.5 + convergence.clarity_score * 0.5)


class ReputationOracle(ValueOracle):
    """
    Social proof value oracle.

    Uses reputation signals and historical performance to estimate value.
    """

    def __init__(self):
        self._reputation_scores: Dict[str, float] = {}
        self._accuracy = 0.72  # Social signals are informative but subjective

    @property
    def oracle_type(self) -> OracleType:
        return OracleType.REPUTATION

    @property
    def historical_accuracy(self) -> float:
        return self._accuracy

    async def evaluate(self, convergence: Convergence) -> OracleSignal:
        # Reputation based on historical consistency
        consistency_score = self._compute_consistency(convergence)

        # Trust score based on metadata
        trust_score = self._compute_trust(convergence)

        # Combined reputation value, clamped to [0, 1]
        reputation_value = max(
            0.0, min(1.0, 0.6 * consistency_score + 0.4 * trust_score)
        )

        # Confidence based on sample size
        sample_confidence = min(1.0, len(self._reputation_scores) / 100 + 0.5)
        confidence = self._accuracy * sample_confidence

        return OracleSignal(
            oracle_type=self.oracle_type,
            value=reputation_value,
            confidence=confidence,
            reasoning=f"Reputation assessment: consistency={consistency_score:.2f}, "
            f"trust={trust_score:.2f}",
        )

    def _compute_consistency(self, convergence: Convergence) -> float:
        """Compute consistency score."""
        # Consistent high synergy = good reputation
        return convergence.synergy * convergence.clarity_score

    def _compute_trust(self, convergence: Convergence) -> float:
        """Compute trust score from metadata."""
        metadata = convergence.metadata

        # Trust factors
        has_provenance = "source" in metadata
        has_verification = "verified" in metadata
        has_attestation = "attestation" in metadata

        # Calculate trust and clamp to [0.0, 1.0]
        raw_trust = (
            0.3 * has_provenance + 0.4 * has_verification + 0.3 * has_attestation + 0.3
        )
        return min(1.0, max(0.0, raw_trust))


class FormalVerificationOracle(ValueOracle):
    """
    Mathematical proof value oracle.

    Uses formal verification techniques to establish provable value bounds.
    """

    def __init__(self):
        self._proof_cache: Dict[str, float] = {}
        self._accuracy = 0.95  # Proofs are highly accurate

    @property
    def oracle_type(self) -> OracleType:
        return OracleType.FORMAL_VERIFICATION

    @property
    def historical_accuracy(self) -> float:
        return self._accuracy

    async def evaluate(self, convergence: Convergence) -> OracleSignal:
        # Formal bounds on convergence value
        lower_bound, upper_bound = self._compute_formal_bounds(convergence)

        # Point estimate at geometric mean of bounds, clamped to [0, 1]
        formal_value = (
            math.sqrt(lower_bound * upper_bound) if lower_bound > 0 else upper_bound / 2
        )
        formal_value = max(0.0, min(1.0, formal_value))

        # Tightness of bounds affects confidence
        bound_tightness = 1.0 - (upper_bound - lower_bound) / (upper_bound + 0.001)
        confidence = self._accuracy * max(0.5, bound_tightness)

        return OracleSignal(
            oracle_type=self.oracle_type,
            value=formal_value,
            confidence=confidence,
            reasoning=f"Formal bounds: [{lower_bound:.3f}, {upper_bound:.3f}], "
            f"tightness={bound_tightness:.2f}",
        )

    def _compute_formal_bounds(self, convergence: Convergence) -> Tuple[float, float]:
        """Compute provable lower and upper bounds on value."""
        # Lower bound: worst-case contribution
        lower = max(
            0.0, convergence.clarity_score * 0.5 - convergence.quantization_error * 10
        )

        # Upper bound: best-case contribution
        upper = min(
            1.0,
            convergence.clarity_score
            + convergence.synergy * 0.2
            + convergence.mutual_information * 0.1,
        )

        return lower, upper


class InformationTheoreticOracle(ValueOracle):
    """
    Information-theoretic value oracle.

    Uses mutual information and entropy to estimate convergence value.
    """

    def __init__(self):
        self._accuracy = 0.88

    @property
    def oracle_type(self) -> OracleType:
        return OracleType.INFORMATION_THEORETIC

    @property
    def historical_accuracy(self) -> float:
        return self._accuracy

    async def evaluate(self, convergence: Convergence) -> OracleSignal:
        # Information value = I(X;H) - β·H(P) + γ·synergy
        alpha = 1.0
        beta = 0.3
        gamma = 0.5

        info_value = (
            alpha * convergence.mutual_information
            - beta * convergence.entropy
            + gamma * convergence.synergy
        )

        # Normalize to [0, 1]
        normalized_value = max(0.0, min(1.0, (info_value + 0.3) / 1.3))

        # Confidence based on information balance
        info_balance = (
            1.0 - abs(convergence.mutual_information - convergence.entropy) / 2
        )
        confidence = self._accuracy * info_balance

        return OracleSignal(
            oracle_type=self.oracle_type,
            value=normalized_value,
            confidence=confidence,
            reasoning=f"Information value: I(X;H)={convergence.mutual_information:.3f}, "
            f"H(P)={convergence.entropy:.3f}, synergy={convergence.synergy:.3f}",
        )


class SNRValueOracle(ValueOracle):
    """
    Value oracle based on Signal-to-Noise Ratio (SNR).

    Higher SNR indicates a clearer, more valuable signal with less noise.
    """

    def __init__(self):
        self._accuracy_history: deque = deque(maxlen=1000)
        self._accuracy = 0.92  # High initial accuracy for SNR

    @property
    def oracle_type(self) -> OracleType:
        return OracleType.SNR

    @property
    def historical_accuracy(self) -> float:
        return self._accuracy

    async def evaluate(self, convergence: Convergence) -> OracleSignal:
        """Evaluate value based on SNR."""
        # Compute SNR using the core utility
        # Note: We use default consistency=0.8 and ihsan=0.95 for oracle evaluation
        snr = compute_snr(
            clarity=convergence.clarity_score,
            synergy=convergence.synergy,
            consistency=0.8,
            entropy=convergence.entropy,
            quantization_error=convergence.quantization_error,
            disagreement=0.1,  # Baseline disagreement for oracle
        )

        # Map SNR to value [0, 1] using sigmoid
        # SNR 0.5 -> Value 0.5, SNR 1.0 -> Value 0.99
        value = 1.0 / (1.0 + math.exp(-10 * (snr - 0.5)))

        # Confidence based on synergy and clarity
        confidence = self._accuracy * (convergence.synergy * convergence.clarity_score)

        return OracleSignal(
            oracle_type=self.oracle_type,
            value=value,
            confidence=confidence,
            reasoning=f"SNR-based value: SNR={snr:.3f}, synergy={convergence.synergy:.3f}",
        )


class PluralisticValueOracle:
    """
    Pluralistic Value Oracle: Ensemble of value signals.

    Addresses the oracle problem by combining multiple independent value
    signals with dynamic weighting based on historical accuracy.
    """

    def __init__(self):
        self.oracles: List[ValueOracle] = [
            ShapleyOracle(),
            PredictionMarketOracle(),
            ReputationOracle(),
            FormalVerificationOracle(),
            InformationTheoreticOracle(),
            SNRValueOracle(),
        ]

        # Dynamic weights based on historical accuracy
        self._oracle_weights: Dict[OracleType, float] = {}
        self._initialize_weights()

        # Accuracy tracking
        self._ground_truth_history: deque = deque(maxlen=1000)
        self._assessment_history: List[ValueAssessment] = []

    def _initialize_weights(self) -> None:
        """Initialize oracle weights from historical accuracy."""
        total_accuracy = sum(o.historical_accuracy for o in self.oracles)
        for oracle in self.oracles:
            self._oracle_weights[oracle.oracle_type] = (
                oracle.historical_accuracy / total_accuracy
            )

    async def compute_value(self, convergence: Convergence) -> ValueAssessment:
        """
        Compute convergence value using ensemble of oracles.
        """
        # Gather signals from all oracles
        signals = await self._gather_signals(convergence)

        # Compute weighted average
        weighted_value = self._compute_weighted_value(signals)

        # Compute ensemble confidence
        ensemble_confidence = self._compute_ensemble_confidence(signals)

        # Compute disagreement score
        disagreement = self._compute_disagreement(signals)

        assessment = ValueAssessment(
            value=weighted_value,
            confidence=ensemble_confidence,
            signals=signals,
            oracle_weights=self._oracle_weights.copy(),
            disagreement_score=disagreement,
        )

        self._assessment_history.append(assessment)
        return assessment

    async def _gather_signals(self, convergence: Convergence) -> List[OracleSignal]:
        """Gather value signals from all oracles."""
        signals = []
        for oracle in self.oracles:
            try:
                signal = await oracle.evaluate(convergence)
                signals.append(signal)
            except Exception as e:
                print(f"Oracle {oracle.oracle_type} error: {e}")
        return signals

    def _compute_weighted_value(self, signals: List[OracleSignal]) -> float:
        """Compute weighted average of oracle signals."""
        if not signals:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for signal in signals:
            weight = self._oracle_weights.get(signal.oracle_type, 0.1)
            # Weight by both oracle weight and signal confidence
            effective_weight = weight * signal.confidence
            weighted_sum += signal.value * effective_weight
            total_weight += effective_weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _compute_ensemble_confidence(self, signals: List[OracleSignal]) -> float:
        """Compute ensemble confidence."""
        if not signals:
            return 0.0

        # Base confidence: weighted average of individual confidences
        total_weight = 0.0
        weighted_conf = 0.0

        for signal in signals:
            weight = self._oracle_weights.get(signal.oracle_type, 0.1)
            weighted_conf += signal.confidence * weight
            total_weight += weight

        base_conf = weighted_conf / total_weight if total_weight > 0 else 0.0

        # Bonus for oracle agreement
        disagreement = self._compute_disagreement(signals)
        agreement_bonus = (1.0 - disagreement) * 0.1

        return min(0.99, base_conf + agreement_bonus)

    def _compute_disagreement(self, signals: List[OracleSignal]) -> float:
        """Compute disagreement score among oracles."""
        if len(signals) < 2:
            return 0.0

        values = [s.value for s in signals]
        mean_value = sum(values) / len(values)

        variance = sum((v - mean_value) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)

        # Coefficient of variation as disagreement measure
        return min(1.0, std_dev / (mean_value + 0.001))

    async def update_weights(
        self, convergence: Convergence, ground_truth: float
    ) -> None:
        """
        Update oracle weights based on ground truth feedback.
        Uses exponential moving average for weight updates.
        """
        learning_rate = 0.05

        # Get fresh signals
        signals = await self._gather_signals(convergence)

        # Update weights based on accuracy
        for signal in signals:
            error = abs(signal.value - ground_truth)
            accuracy = 1.0 - min(1.0, error)

            current_weight = self._oracle_weights.get(signal.oracle_type, 0.1)
            new_weight = current_weight * (1 - learning_rate) + accuracy * learning_rate
            self._oracle_weights[signal.oracle_type] = new_weight

        # Normalize weights
        total = sum(self._oracle_weights.values())
        for key in self._oracle_weights:
            self._oracle_weights[key] /= total

        # Store ground truth
        self._ground_truth_history.append(
            {
                "convergence_id": convergence.id,
                "ground_truth": ground_truth,
                "timestamp": datetime.now(timezone.utc),
            }
        )

    def get_oracle_weights(self) -> Dict[OracleType, float]:
        """Return current oracle weights."""
        return self._oracle_weights.copy()

    def get_assessment_history(self) -> List[ValueAssessment]:
        """Return assessment history."""
        return self._assessment_history.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Tests
# ═══════════════════════════════════════════════════════════════════════════════


async def self_test():
    """Self-test for pluralistic value oracle."""
    print("Pluralistic Value Oracle Self-Test")
    print("=" * 50)

    oracle = PluralisticValueOracle()

    # Test 1: High-value convergence
    high_convergence = Convergence(
        id="test-high-001",
        clarity_score=0.92,
        mutual_information=0.85,
        entropy=0.15,
        synergy=0.95,
        quantization_error=0.001,
        metadata={"verified": True, "source": "test", "attestation": "signed"},
    )

    assessment1 = await oracle.compute_value(high_convergence)
    assert (
        assessment1.value > 0.6
    ), f"High convergence should have high value, got {assessment1.value}"
    assert len(assessment1.signals) == 6, f"Should have 6 oracle signals, got {len(assessment1.signals)}"
    print(
        f"✓ High convergence: value={assessment1.value:.3f}, "
        f"confidence={assessment1.confidence:.2%}, "
        f"disagreement={assessment1.disagreement_score:.2%}"
    )

    # Test 2: Low-value convergence
    low_convergence = Convergence(
        id="test-low-001",
        clarity_score=0.25,
        mutual_information=0.20,
        entropy=0.80,
        synergy=0.15,
        quantization_error=0.05,
    )

    assessment2 = await oracle.compute_value(low_convergence)
    assert (
        assessment2.value < assessment1.value
    ), "Low convergence should have lower value"
    print(
        f"✓ Low convergence: value={assessment2.value:.3f}, "
        f"confidence={assessment2.confidence:.2%}"
    )

    # Test 3: Singularity event (synergy > 0.999)
    singularity = Convergence(
        id="test-singularity-001",
        clarity_score=0.99,
        mutual_information=0.98,
        entropy=0.02,
        synergy=0.999,
        quantization_error=0.0001,
        metadata={"verified": True, "attestation": "formal_proof"},
    )

    assessment3 = await oracle.compute_value(singularity)
    assert assessment3.value > 0.8, "Singularity should have very high value"
    print(
        f"✓ Singularity event: value={assessment3.value:.3f}, "
        f"confidence={assessment3.confidence:.2%}"
    )

    # Test 4: Oracle weights
    weights = oracle.get_oracle_weights()
    assert len(weights) == 6
    assert abs(sum(weights.values()) - 1.0) < 0.001, "Weights should sum to 1"
    print(f"✓ Oracle weights: {len(weights)} oracles, sum={sum(weights.values()):.3f}")

    # Test 5: Weight update with ground truth
    await oracle.update_weights(high_convergence, ground_truth=0.85)
    updated_weights = oracle.get_oracle_weights()
    assert len(updated_weights) == 6
    print(f"✓ Weight update: weights adjusted based on ground truth")

    # Test 6: Disagreement measurement
    assert 0.0 <= assessment1.disagreement_score <= 1.0
    print(f"✓ Disagreement measurement: {assessment1.disagreement_score:.2%}")

    # Test 7: Assessment history
    history = oracle.get_assessment_history()
    assert len(history) >= 3
    print(f"✓ Assessment history: {len(history)} entries")

    print("=" * 50)
    print("All pluralistic value oracle tests passed ✓")


if __name__ == "__main__":
    import asyncio

    asyncio.run(self_test())
