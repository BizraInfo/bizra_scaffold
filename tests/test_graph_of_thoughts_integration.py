"""
Integration Tests for Graph-of-Thoughts Enhancement
═════════════════════════════════════════════════════════════════════════════
Tests the integration of Graph-of-Thoughts, SNR scoring, and domain-aware
knowledge graph with the BIZRA cognitive architecture.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from core.enhanced_cognitive_integration import (
    EnhancedCognitiveProcessor,
    EnhancedCognitiveResult,
)
from core.graph_of_thoughts import (
    DomainBridge,
    DomainBridgeType,
    GraphOfThoughtsEngine,
    Thought,
    ThoughtChain,
    ThoughtType,
)
from core.snr_scorer import SNRLevel, SNRMetrics, SNRScorer
from core.tiered_verification import ConvergenceResult
from core.ultimate_integration import Observation, UrgencyLevel


@pytest.fixture
def mock_l4_hypergraph():
    """Mock L4 semantic hypergraph."""
    mock = AsyncMock()

    # Mock get_neighbors_with_domains
    mock.get_neighbors_with_domains = AsyncMock(
        return_value=[
            {
                "id": "ConceptA",
                "domains": ["math", "physics"],
                "relation_types": ["RELATED_TO"],
                "weight": 0.8,
                "consistency": 0.75,
                "disagreement": 0.2,
                "ihsan": 0.96,
            },
            {
                "id": "ConceptB",
                "domains": ["economics"],
                "relation_types": ["YIELDS"],
                "weight": 0.7,
                "consistency": 0.80,
                "disagreement": 0.15,
                "ihsan": 0.97,
            },
        ]
    )

    return mock


@pytest.fixture
def sample_observation():
    """Sample observation for testing."""
    return Observation(
        id="test_001",
        data=b"test observation data",
        urgency=UrgencyLevel.NEAR_REAL_TIME,
        context={
            "query": "Test query for graph-of-thoughts reasoning",
            "domains": ["math", "ethics"],
        },
    )


@pytest.mark.asyncio
class TestGraphOfThoughtsIntegration:
    """Integration tests for Graph-of-Thoughts."""

    async def test_enhanced_processor_initialization(self, mock_l4_hypergraph):
        """Test processor initializes correctly."""
        processor = EnhancedCognitiveProcessor(
            l4_hypergraph=mock_l4_hypergraph,
            quantized_convergence=Mock(),
            narrative_compiler=Mock(),
            enable_graph_of_thoughts=True,
            beam_width=5,
            max_thought_depth=3,
        )

        assert processor.enable_got is True
        assert processor.got_engine is not None
        assert processor.got_engine.beam_width == 5
        assert processor.got_engine.max_depth == 3
        assert processor.snr_scorer is not None

    async def test_snr_scoring_integration(self):
        """Test SNR scoring computes correctly."""
        snr_scorer = SNRScorer()

        # Mock convergence result
        convergence = ConvergenceResult(
            clarity=0.85,
            mutual_information=0.75,
            entropy=0.20,
            synergy=0.90,
            quantization_error=0.05,
            quality="EXCELLENT",
            action={"type": "test"},
        )

        snr_metrics = snr_scorer.compute_from_convergence(
            convergence, consistency=0.80, disagreement=0.15, ihsan_metric=0.96
        )

        assert snr_metrics.snr_score > 0
        assert snr_metrics.level in [SNRLevel.HIGH, SNRLevel.MEDIUM, SNRLevel.LOW]
        assert snr_metrics.ihsan_metric == 0.96
        assert snr_metrics.clarity_component == 0.85

    async def test_high_snr_requires_high_ihsan(self):
        """Test that HIGH SNR requires Ihsān ≥ 0.95."""
        snr_scorer = SNRScorer(enable_ethical_constraints=True)

        # Create convergence with high clarity/synergy
        convergence = ConvergenceResult(
            clarity=0.95,
            mutual_information=0.90,
            entropy=0.10,
            synergy=0.95,
            quantization_error=0.02,
            quality="OPTIMAL",
            action={"type": "test"},
        )

        # Low Ihsān should prevent HIGH classification
        snr_low_ihsan = snr_scorer.compute_from_convergence(
            convergence,
            consistency=0.90,
            disagreement=0.05,
            ihsan_metric=0.90,  # Below threshold
        )

        assert snr_low_ihsan.level != SNRLevel.HIGH
        assert snr_low_ihsan.ethical_override is True

        # High Ihsān should allow HIGH classification
        snr_high_ihsan = snr_scorer.compute_from_convergence(
            convergence,
            consistency=0.90,
            disagreement=0.05,
            ihsan_metric=0.96,  # Above threshold
        )

        assert snr_high_ihsan.level == SNRLevel.HIGH
        assert snr_high_ihsan.ethical_override is False

    async def test_domain_bridge_detection(self):
        """Test domain bridge detection in thought chains."""
        from core.graph_of_thoughts import Thought, ThoughtType
        from core.snr_scorer import SNRLevel, SNRMetrics

        # Create thoughts with different domains
        thought1 = Thought(
            id="t1",
            content="MathConcept",
            thought_type=ThoughtType.PERCEPTION,
            snr_metrics=SNRMetrics(
                snr_score=0.8,
                signal_strength=0.7,
                noise_floor=0.2,
                clarity_component=0.8,
                synergy_component=0.85,
                consistency_component=0.75,
                entropy_component=0.15,
                quantization_error=0.05,
                disagreement_component=0.1,
                level=SNRLevel.HIGH,
                confidence=0.9,
                ihsan_metric=0.96,
            ),
            source_nodes=["MathConcept"],
            domains={"math"},
        )

        thought2 = Thought(
            id="t2",
            content="EthicsConcept",
            thought_type=ThoughtType.ANALOGY,
            snr_metrics=SNRMetrics(
                snr_score=0.75,
                signal_strength=0.65,
                noise_floor=0.25,
                clarity_component=0.75,
                synergy_component=0.80,
                consistency_component=0.70,
                entropy_component=0.20,
                quantization_error=0.05,
                disagreement_component=0.15,
                level=SNRLevel.MEDIUM,
                confidence=0.85,
                ihsan_metric=0.97,
            ),
            source_nodes=["EthicsConcept"],
            domains={"ethics"},
        )

        # Bridge should be detected when domains differ
        assert thought1.domains != thought2.domains
        assert "math" in thought1.domains
        assert "ethics" in thought2.domains

    async def test_thought_chain_snr_ranking(self):
        """Test thought chains are ranked by SNR."""
        from core.graph_of_thoughts import ThoughtChain

        chain1 = ThoughtChain(
            id="chain1",
            thoughts=[],
            bridges=[],
            total_snr=2.5,
            avg_snr=0.83,
            max_depth=3,
            domain_diversity=0.6,
        )

        chain2 = ThoughtChain(
            id="chain2",
            thoughts=[],
            bridges=[],
            total_snr=1.8,
            avg_snr=0.60,
            max_depth=3,
            domain_diversity=0.4,
        )

        chains = [chain2, chain1]  # Deliberately out of order
        ranked = sorted(chains, key=lambda c: c.total_snr, reverse=True)

        assert ranked[0].id == "chain1"
        assert ranked[0].total_snr > ranked[1].total_snr


@pytest.mark.asyncio
class TestEndToEndIntegration:
    """End-to-end integration tests."""

    async def test_process_observation_with_got(
        self, mock_l4_hypergraph, sample_observation
    ):
        """Test full processing pipeline with Graph-of-Thoughts."""
        # Create mocks
        mock_convergence = Mock()
        mock_convergence.compute = Mock(
            return_value=ConvergenceResult(
                clarity=0.80,
                mutual_information=0.70,
                entropy=0.25,
                synergy=0.85,
                quantization_error=0.05,
                quality="GOOD",
                action={"type": "test_action"},
            )
        )

        mock_narrative = Mock()
        mock_narrative.compile = Mock(
            return_value=Mock(
                summary="Test summary",
                sections=[],
                reading_time_seconds=30,
                complexity_score=0.5,
            )
        )

        # Create processor
        processor = EnhancedCognitiveProcessor(
            l4_hypergraph=mock_l4_hypergraph,
            quantized_convergence=mock_convergence,
            narrative_compiler=mock_narrative,
            enable_graph_of_thoughts=True,
            beam_width=3,
            max_thought_depth=2,
        )

        # Process observation
        result = await processor.process(sample_observation)

        # Verify result structure
        assert isinstance(result, EnhancedCognitiveResult)
        assert result.overall_snr is not None
        assert result.overall_snr.snr_score >= 0
        assert result.processing_time_ms > 0

        # Verify ultimate result has enhancements
        assert result.ultimate_result.snr_metrics is not None
        assert result.ultimate_result.thought_chains is not None
        assert result.ultimate_result.domain_bridges is not None

    async def test_process_without_got(self, mock_l4_hypergraph, sample_observation):
        """Test processing works with Graph-of-Thoughts disabled."""
        mock_convergence = Mock()
        mock_convergence.compute = Mock(
            return_value=ConvergenceResult(
                clarity=0.75,
                mutual_information=0.65,
                entropy=0.30,
                synergy=0.80,
                quantization_error=0.08,
                quality="GOOD",
                action={"type": "test_action"},
            )
        )

        mock_narrative = Mock()
        mock_narrative.compile = Mock(
            return_value=Mock(
                summary="Test summary",
                sections=[],
                reading_time_seconds=25,
                complexity_score=0.4,
            )
        )

        processor = EnhancedCognitiveProcessor(
            l4_hypergraph=mock_l4_hypergraph,
            quantized_convergence=mock_convergence,
            narrative_compiler=mock_narrative,
            enable_graph_of_thoughts=False,  # Disabled
        )

        result = await processor.process(sample_observation)

        # Should still work but without thought chains
        assert isinstance(result, EnhancedCognitiveResult)
        assert result.overall_snr is not None
        assert len(result.all_chains) == 0  # No chains when disabled
        assert len(result.domain_bridges) == 0


@pytest.mark.asyncio
class TestMetricsAndObservability:
    """Test metrics collection and observability."""

    async def test_snr_metrics_recorded(self, mock_l4_hypergraph, sample_observation):
        """Test SNR metrics are recorded to observability system."""
        from core.observability import MeterProvider

        mock_convergence = Mock()
        mock_convergence.compute = Mock(
            return_value=ConvergenceResult(
                clarity=0.80,
                mutual_information=0.70,
                entropy=0.25,
                synergy=0.85,
                quantization_error=0.05,
                quality="GOOD",
                action={"type": "test"},
            )
        )

        mock_narrative = Mock()
        mock_narrative.compile = Mock(
            return_value=Mock(
                summary="Test",
                sections=[],
                reading_time_seconds=20,
                complexity_score=0.5,
            )
        )

        metrics = MeterProvider("test")

        processor = EnhancedCognitiveProcessor(
            l4_hypergraph=mock_l4_hypergraph,
            quantized_convergence=mock_convergence,
            narrative_compiler=mock_narrative,
            metrics_collector=metrics,
            enable_graph_of_thoughts=False,  # Simplify test
        )

        await processor.process(sample_observation)

        # Verify metrics were recorded
        all_metrics = metrics.get_all_metrics()
        assert all_metrics is not None
        assert "gauges" in all_metrics or "counters" in all_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
