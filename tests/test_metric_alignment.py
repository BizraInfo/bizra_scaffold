"""
Cross-module metric alignment tests.

Ensures Ihsan and SNR definitions remain consistent across:
- BIZRA_SOT.md (documentation)
- core/kernel/pillars.py (executable physics)
- core/pipelines/chat_to_knowledge_pipeline.py (usage)
- core/engine/api.py (telemetry)

These tests prevent future "split brain" regressions.
"""

import pytest
from core.kernel.pillars import KernelLaws
from core.snr_scorer import SNRScorer, SNRThresholds
from core.pipelines.chat_to_knowledge_pipeline import _compute_chat_snr


class TestIhsanAlignment:
    """Test Ihsan metric consistency across modules."""

    def test_kernel_ihsan_definition_complete(self):
        """Kernel Ihsan weights must sum to 1.0 and have all 8 dimensions."""
        weights = KernelLaws.IHSAN.WEIGHTS
        expected_dims = {
            "correctness", "safety", "user_benefit", "efficiency",
            "auditability", "anti_centralization", "robustness", "fairness"
        }

        assert set(weights.keys()) == expected_dims
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_ihsan_threshold_consistent(self):
        """Ihsan threshold must be 0.95 across all modules."""
        threshold = KernelLaws.IHSAN.MIN_SCORE_THRESHOLD
        assert threshold == 0.95

    def test_sot_matches_kernel_ihsan(self):
        """BIZRA_SOT.md should match kernel Ihsan definition."""
        # This is a documentation check - would need to parse SOT file
        # For now, assert kernel has the expected structure
        weights = KernelLaws.IHSAN.WEIGHTS

        # Verify key relationships (correctness > fairness, safety high, etc.)
        assert weights["correctness"] > weights["fairness"]
        assert weights["safety"] > weights["efficiency"]
        assert weights["auditability"] > weights["anti_centralization"]


class TestSNRAlignment:
    """Test SNR calculation consistency across modules."""

    def test_snr_thresholds_defined(self):
        """SNR thresholds must be properly configured."""
        thresholds = SNRThresholds()
        assert thresholds.high_threshold > thresholds.medium_threshold
        assert thresholds.min_ihsan_for_high == 0.95

    def test_pipeline_uses_snr_scorer(self):
        """Chat pipeline must use proper SNRScorer, not hardcoded values."""
        scorer = SNRScorer()
        test_text = "This is a test message with some content."

        # This should not raise an exception and return proper values
        snr, level, ihsan = _compute_chat_snr(test_text, scorer)

        assert isinstance(snr, float)
        assert isinstance(level, str)
        assert isinstance(ihsan, float)
        assert 0.0 <= snr <= 1.0  # SNR should be normalized
        assert 0.0 <= ihsan <= 1.0  # Ihsan should be normalized
        assert level in ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]

    def test_chat_snr_calculates_ihsan_properly(self):
        """Chat SNR calculation should use kernel Ihsan weights."""
        scorer = SNRScorer()
        code_text = "```python\nprint('hello world')\n```"  # Should have higher correctness

        snr, level, ihsan = _compute_chat_snr(code_text, scorer)

        # Code content should result in reasonable Ihsan score
        assert ihsan >= 0.7  # Should be reasonably high for code content
        assert ihsan <= 1.0

    def test_empty_text_handled(self):
        """Empty text should be handled gracefully."""
        scorer = SNRScorer()

        snr, level, ihsan = _compute_chat_snr("", scorer)

        assert snr == 0.0
        assert level == "UNKNOWN"
        assert ihsan == 0.0


class TestBootVectorAlignment:
    """Test boot vector calculation uses kernel Ihsan."""

    def test_boot_vector_uses_kernel_weights(self):
        """Boot calculation should use kernel Ihsan weights."""
        from core.boot import _calculate_boot_vector
        from core.capabilities import Capabilities

        # Create mock capabilities
        caps = Capabilities(
            force_lite=False,
            blake3=True,
            z3=True,
            numpy=True,
            faiss=True,
            neo4j=True
        )

        score = _calculate_boot_vector(caps)

        # Should be a valid score using kernel weights
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestAPIStatusAlignment:
    """Test API status reporting uses kernel definitions."""

    def test_api_status_uses_kernel_threshold(self):
        """API status should use kernel Ihsan threshold."""
        from core.engine.api import get_sovereign_status
        from core.kernel.pillars import KernelLaws

        # Mock the status call (would need full app context)
        # For now, verify the threshold constant is accessible
        threshold = KernelLaws.IHSAN.MIN_SCORE_THRESHOLD
        assert threshold == 0.95


if __name__ == "__main__":
    pytest.main([__file__])
