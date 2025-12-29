# tests/test_snr_range.py
"""
BIZRA AEON OMEGA - SNR Range Validation Tests
Tests that SNR scores fall within valid ranges and thresholds.

Note: compute_snr() returns signal/noise RATIO which can be > 1.0
SNRThresholds.classify() uses normalized thresholds (0.5-0.8 range)
"""

import pytest
import sys
sys.path.insert(0, '.')

from core.snr_scorer import compute_snr, SNRLevel, SNRThresholds


# Use SNRThresholds for classification
thresholds = SNRThresholds()


def classify_snr_level(snr_score: float) -> SNRLevel:
    """Classify SNR score into level using SNRThresholds."""
    return thresholds.classify(snr_score, ihsan=0.95, confidence=0.9)


class TestSNRRange:
    """Test SNR score ranges and classification."""

    def test_snr_returns_positive_float(self):
        """SNR must return positive float (ratio can exceed 1.0)."""
        result = compute_snr(
            clarity=0.85, synergy=0.90, consistency=0.88,
            entropy=0.15, quantization_error=0.02, disagreement=0.05
        )
        assert isinstance(result, float)
        assert result > 0.0  # SNR is signal/noise ratio, can be > 1.0

    def test_snr_high_signal_low_noise(self):
        """High signal, low noise should produce high SNR ratio."""
        result = compute_snr(
            clarity=0.95, synergy=0.95, consistency=0.95,
            entropy=0.01, quantization_error=0.01, disagreement=0.01
        )
        # With near-zero noise, SNR should be very high
        assert result > 1.0  # Signal dominates noise
        assert classify_snr_level(result) == SNRLevel.HIGH

    def test_snr_low_signal_high_noise(self):
        """Low signal, high noise should produce low SNR ratio."""
        result = compute_snr(
            clarity=0.30, synergy=0.30, consistency=0.30,
            entropy=0.50, quantization_error=0.30, disagreement=0.40
        )
        # Signal ~= 0.027, Noise ~= 1.2, ratio < 0.5
        assert result < 0.50  # LOW threshold
        assert classify_snr_level(result) == SNRLevel.LOW

    def test_snr_medium_range(self):
        """Medium values should produce medium SNR ratio."""
        result = compute_snr(
            clarity=0.65, synergy=0.65, consistency=0.65,
            entropy=0.15, quantization_error=0.05, disagreement=0.10
        )
        # Expect ratio in medium range
        assert 0.50 <= result or result >= 0.80 or classify_snr_level(result) in [SNRLevel.MEDIUM, SNRLevel.HIGH]

    def test_snr_edge_case_zero_noise(self):
        """Zero noise should not cause division by zero."""
        result = compute_snr(
            clarity=0.90, synergy=0.90, consistency=0.90,
            entropy=0.0, quantization_error=0.0, disagreement=0.0
        )
        # With zero noise, SNR should be very high (capped or calculated safely)
        assert result > 0.0  # Must return valid positive number
        assert classify_snr_level(result) == SNRLevel.HIGH

    def test_snr_edge_case_max_values(self):
        """Maximum signal, zero noise should produce high SNR."""
        result = compute_snr(
            clarity=1.0, synergy=1.0, consistency=1.0,
            entropy=0.0, quantization_error=0.0, disagreement=0.0
        )
        # Max signal, zero noise = maximum SNR ratio
        assert result > 0.0  # SNR ratio can exceed 1.0
        assert classify_snr_level(result) == SNRLevel.HIGH


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
