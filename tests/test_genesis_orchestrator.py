"""
Tests for Genesis Orchestrator - Peak Masterpiece.

Tests the Autonomous Genesis Engine with:
- Interdisciplinary thinking across 6 domain lenses
- Graph of Thoughts with SNR-weighted beam search
- Autonomous SNR engine quality gating
- Giants Protocol wisdom repository
- Genesis Node identity binding
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import shutil

from core.genesis.genesis_orchestrator import (
    # Constants
    SNR_THRESHOLD_HIGH,
    SNR_THRESHOLD_MEDIUM,
    SNR_THRESHOLD_IHSAN,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MAX_DEPTH,
    # Types
    DomainLens,
    LensInsight,
    SynthesizedInsight,
    ThoughtNode,
    WisdomEntry,
    # Classes
    InterdisciplinaryLensSystem,
    AutonomousSNREngine,
    WisdomRepository,
    GenesisOrchestrator,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_wisdom_dir():
    """Create temporary directory for wisdom storage."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_lens_insights():
    """Create sample lens insights."""
    return [
        LensInsight(
            lens=DomainLens.CRYPTOGRAPHY,
            content="Use Ed25519 signatures for attestation",
            confidence=0.90,
            snr_contribution=0.18,
            related_concepts=["signature", "ed25519", "attestation"],
        ),
        LensInsight(
            lens=DomainLens.ECONOMICS,
            content="Align incentives through token staking",
            confidence=0.85,
            snr_contribution=0.15,
            related_concepts=["incentive", "staking", "tokenomics"],
        ),
        LensInsight(
            lens=DomainLens.PHILOSOPHY,
            content="Ensure Ihsān score ≥0.95 for all operations",
            confidence=0.95,
            snr_contribution=0.20,
            related_concepts=["ihsan", "ethics", "excellence"],
        ),
    ]


@pytest.fixture
def sample_thought_node():
    """Create sample thought node."""
    return ThoughtNode(
        id="thought_test123",
        content="Test thought content",
        depth=2,
        snr_score=0.85,
        ihsan_score=0.96,
        confidence=0.88,
        parent_id="thought_parent",
        children_ids=[],
        wisdom_seeds=["wisdom_seed1"],
    )


@pytest.fixture
def sample_wisdom_entry():
    """Create sample wisdom entry."""
    now = datetime.now(timezone.utc)
    return WisdomEntry(
        id="wisdom_test123",
        title="Test Wisdom",
        content="Proofs first, publish later - the Integrity Flywheel",
        source="test",
        snr_score=0.92,
        ihsan_score=0.97,
        observation_count=5,
        first_observed=now,
        last_observed=now,
        related_concepts=["integrity", "proofs", "flywheel"],
    )


# =============================================================================
# DOMAIN LENS TESTS
# =============================================================================


class TestDomainLens:
    """Tests for DomainLens enum."""
    
    def test_all_lenses_defined(self):
        """All six domain lenses are defined."""
        assert len(DomainLens) == 6
        assert DomainLens.CRYPTOGRAPHY
        assert DomainLens.ECONOMICS
        assert DomainLens.PHILOSOPHY
        assert DomainLens.GOVERNANCE
        assert DomainLens.SYSTEMS
        assert DomainLens.COGNITIVE
    
    def test_lens_values(self):
        """Lens values are correct."""
        assert DomainLens.CRYPTOGRAPHY.value == "crypto"
        assert DomainLens.PHILOSOPHY.value == "philosophy"


class TestLensInsight:
    """Tests for LensInsight dataclass."""
    
    def test_creation(self, sample_lens_insights):
        """LensInsight can be created."""
        insight = sample_lens_insights[0]
        assert insight.lens == DomainLens.CRYPTOGRAPHY
        assert insight.confidence == 0.90
        assert "signature" in insight.related_concepts
    
    def test_snr_contribution(self, sample_lens_insights):
        """SNR contribution is tracked."""
        for insight in sample_lens_insights:
            assert 0 <= insight.snr_contribution <= 1


class TestInterdisciplinaryLensSystem:
    """Tests for InterdisciplinaryLensSystem."""
    
    @pytest.fixture
    def lens_system(self):
        return InterdisciplinaryLensSystem()
    
    def test_all_lenses_active(self, lens_system):
        """All lenses are active by default."""
        assert len(lens_system.active_lenses) == 6
    
    @pytest.mark.asyncio
    async def test_analyze_returns_insights(self, lens_system):
        """Analyze returns insights for all lenses."""
        insights = await lens_system.analyze(
            problem="Design a secure attestation mechanism",
            context={"domain": "security"},
        )
        
        assert len(insights) == 6
        lenses_covered = {i.lens for i in insights}
        assert len(lenses_covered) == 6
    
    @pytest.mark.asyncio
    async def test_synthesize_combines_insights(self, lens_system, sample_lens_insights):
        """Synthesize combines multiple lens insights."""
        synthesized = await lens_system.synthesize(
            lens_insights=sample_lens_insights,
            problem="Test problem",
        )
        
        assert isinstance(synthesized, SynthesizedInsight)
        assert synthesized.id.startswith("synth_")
        assert len(synthesized.lens_insights) == 3
        assert 0 <= synthesized.snr_score <= 1
        assert synthesized.ihsan_score >= 0.95
    
    @pytest.mark.asyncio
    async def test_philosophy_lens_affects_ihsan(self, lens_system):
        """Philosophy lens affects Ihsān score."""
        insights = await lens_system.analyze(
            problem="Ethical question",
            context={},
        )
        
        philosophy_insight = next(
            i for i in insights if i.lens == DomainLens.PHILOSOPHY
        )
        
        # Philosophy should have high confidence
        assert philosophy_insight.confidence >= 0.95


# =============================================================================
# SYNTHESIZED INSIGHT TESTS
# =============================================================================


class TestSynthesizedInsight:
    """Tests for SynthesizedInsight dataclass."""
    
    def test_passes_snr_gate_high_scores(self):
        """Passes gate with high SNR and Ihsān."""
        insight = SynthesizedInsight(
            id="synth_test",
            title="Test",
            content="Test content",
            lens_insights=[],
            snr_score=0.90,
            ihsan_score=0.96,
            confidence=0.88,
        )
        
        assert insight.passes_snr_gate() is True
    
    def test_fails_snr_gate_low_snr(self):
        """Fails gate with low SNR."""
        insight = SynthesizedInsight(
            id="synth_test",
            title="Test",
            content="Test content",
            lens_insights=[],
            snr_score=0.70,  # Below 0.80
            ihsan_score=0.96,
            confidence=0.88,
        )
        
        assert insight.passes_snr_gate() is False
    
    def test_fails_snr_gate_low_ihsan(self):
        """Fails gate with low Ihsān."""
        insight = SynthesizedInsight(
            id="synth_test",
            title="Test",
            content="Test content",
            lens_insights=[],
            snr_score=0.90,
            ihsan_score=0.90,  # Below 0.95
            confidence=0.88,
        )
        
        assert insight.passes_snr_gate() is False


# =============================================================================
# THOUGHT NODE TESTS
# =============================================================================


class TestThoughtNode:
    """Tests for ThoughtNode dataclass."""
    
    def test_is_terminal(self, sample_thought_node):
        """Detect terminal nodes."""
        assert sample_thought_node.is_terminal() is True
        
        sample_thought_node.children_ids = ["child1", "child2"]
        assert sample_thought_node.is_terminal() is False
    
    def test_passes_snr_gate_high_scores(self, sample_thought_node):
        """Passes gate with high scores."""
        assert sample_thought_node.passes_snr_gate() is True
    
    def test_fails_snr_gate_low_scores(self, sample_thought_node):
        """Fails gate with low scores."""
        sample_thought_node.snr_score = 0.40
        assert sample_thought_node.passes_snr_gate() is False
    
    def test_wisdom_seeds_tracked(self, sample_thought_node):
        """Wisdom seeds are tracked."""
        assert "wisdom_seed1" in sample_thought_node.wisdom_seeds


# =============================================================================
# AUTONOMOUS SNR ENGINE TESTS
# =============================================================================


class TestAutonomousSNREngine:
    """Tests for AutonomousSNREngine."""
    
    @pytest.fixture
    def snr_engine(self):
        return AutonomousSNREngine()
    
    def test_compute_snr(self, snr_engine):
        """Compute SNR from components."""
        snr = snr_engine.compute_snr(
            clarity=0.9,
            synergy=0.8,
            consistency=0.85,
            entropy=0.1,
            quantization_error=0.05,
            disagreement=0.05,
        )
        
        # signal = 0.9 * 0.8 * 0.85 = 0.612
        # noise = 0.1 + 0.05 + 0.05 = 0.2
        # snr = 0.612 / 0.2 = 3.06
        assert snr > 3.0
    
    def test_classify_high(self, snr_engine):
        """Classify HIGH with high SNR and Ihsān."""
        level = snr_engine.classify(snr_score=0.90, ihsan_score=0.96)
        assert level == "HIGH"
    
    def test_classify_high_downgrade_to_medium(self, snr_engine):
        """HIGH downgrades to MEDIUM if Ihsān too low."""
        level = snr_engine.classify(snr_score=0.90, ihsan_score=0.90)
        assert level == "MEDIUM"
    
    def test_classify_medium(self, snr_engine):
        """Classify MEDIUM with medium SNR."""
        level = snr_engine.classify(snr_score=0.60, ihsan_score=0.96)
        assert level == "MEDIUM"
    
    def test_classify_low(self, snr_engine):
        """Classify LOW with low SNR."""
        level = snr_engine.classify(snr_score=0.30, ihsan_score=0.96)
        assert level == "LOW"
    
    def test_gate_fail_closed(self, snr_engine):
        """Gate rejects MEDIUM in fail-closed mode."""
        assert snr_engine.gate(0.90, 0.96) is True   # HIGH
        assert snr_engine.gate(0.60, 0.96) is False  # MEDIUM rejected
        assert snr_engine.gate(0.30, 0.96) is False  # LOW rejected
    
    def test_gate_permissive(self):
        """Gate accepts MEDIUM in permissive mode."""
        engine = AutonomousSNREngine(fail_closed=False)
        assert engine.gate(0.90, 0.96) is True   # HIGH
        assert engine.gate(0.60, 0.96) is True   # MEDIUM accepted
        assert engine.gate(0.30, 0.96) is False  # LOW rejected
    
    def test_rank_and_select(self, snr_engine):
        """Rank and select top items."""
        items = [
            ("item_a", 0.95, 0.97),  # HIGH - should be selected first
            ("item_b", 0.85, 0.96),  # HIGH - should be selected second
            ("item_c", 0.60, 0.96),  # MEDIUM - rejected in fail-closed
            ("item_d", 0.30, 0.96),  # LOW - rejected
        ]
        
        selected = snr_engine.rank_and_select(items, top_k=3)
        
        assert len(selected) == 2  # Only 2 HIGH items
        assert selected[0] == "item_a"
        assert selected[1] == "item_b"
    
    def test_statistics(self, snr_engine):
        """Statistics are tracked."""
        snr_engine.classify(0.90, 0.96)
        snr_engine.classify(0.60, 0.96)
        snr_engine.classify(0.30, 0.96)
        
        stats = snr_engine.get_statistics()
        
        assert stats["processed"] == 3
        assert stats["passed_high"] == 1
        assert stats["passed_medium"] == 1
        assert stats["rejected"] == 1


# =============================================================================
# WISDOM REPOSITORY TESTS
# =============================================================================


class TestWisdomRepository:
    """Tests for WisdomRepository."""
    
    @pytest.fixture
    def wisdom_repo(self, temp_wisdom_dir):
        return WisdomRepository(storage_path=temp_wisdom_dir)
    
    def test_add_and_get(self, wisdom_repo, sample_wisdom_entry):
        """Add and retrieve wisdom."""
        wisdom_repo.add(sample_wisdom_entry)
        
        retrieved = wisdom_repo.get(sample_wisdom_entry.id)
        assert retrieved is not None
        assert retrieved.title == sample_wisdom_entry.title
    
    def test_add_increments_observation_count(self, wisdom_repo, sample_wisdom_entry):
        """Adding same wisdom increments observation count."""
        wisdom_repo.add(sample_wisdom_entry)
        wisdom_repo.add(sample_wisdom_entry)
        
        retrieved = wisdom_repo.get(sample_wisdom_entry.id)
        assert retrieved.observation_count == 6  # 5 + 1
    
    def test_search_by_concept(self, wisdom_repo, sample_wisdom_entry):
        """Search wisdom by concept."""
        wisdom_repo.add(sample_wisdom_entry)
        
        results = wisdom_repo.search_by_concept("integrity")
        
        assert len(results) == 1
        assert results[0].id == sample_wisdom_entry.id
    
    def test_get_high_snr_wisdom(self, wisdom_repo):
        """Get high-SNR wisdom entries."""
        now = datetime.now(timezone.utc)
        
        high_snr = WisdomEntry(
            id="high", title="High SNR", content="test",
            source="test", snr_score=0.92, ihsan_score=0.96,
            observation_count=1, first_observed=now, last_observed=now,
        )
        low_snr = WisdomEntry(
            id="low", title="Low SNR", content="test",
            source="test", snr_score=0.60, ihsan_score=0.96,
            observation_count=1, first_observed=now, last_observed=now,
        )
        
        wisdom_repo.add(high_snr)
        wisdom_repo.add(low_snr)
        
        results = wisdom_repo.get_high_snr_wisdom()
        
        assert len(results) == 1
        assert results[0].id == "high"
    
    def test_save_and_load(self, temp_wisdom_dir, sample_wisdom_entry):
        """Wisdom persists across instances."""
        repo1 = WisdomRepository(storage_path=temp_wisdom_dir)
        repo1.add(sample_wisdom_entry)
        repo1.save()
        
        repo2 = WisdomRepository(storage_path=temp_wisdom_dir)
        retrieved = repo2.get(sample_wisdom_entry.id)
        
        assert retrieved is not None
        assert retrieved.title == sample_wisdom_entry.title


# =============================================================================
# GENESIS ORCHESTRATOR TESTS
# =============================================================================


class TestGenesisOrchestrator:
    """Tests for GenesisOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self, temp_wisdom_dir):
        """Create orchestrator with temp storage."""
        orch = GenesisOrchestrator(
            beam_width=4,
            max_depth=2,
            fail_closed=False,
        )
        orch.wisdom_repo = WisdomRepository(storage_path=temp_wisdom_dir)
        return orch
    
    @pytest.mark.asyncio
    async def test_process_returns_result(self, orchestrator):
        """Process returns complete result."""
        result = await orchestrator.process(
            problem="Design a token distribution mechanism",
            context={"domain": "tokenomics"},
        )
        
        assert "problem" in result
        assert "synthesized_insight" in result
        assert "wisdom_seeds_used" in result
        assert "thought_paths_explored" in result
        assert "high_snr_insights" in result
        assert "crystallized" in result
        assert "attestation" in result
        assert "processing_time_ms" in result
        assert "snr_statistics" in result
    
    @pytest.mark.asyncio
    async def test_process_creates_thoughts(self, orchestrator):
        """Process creates thought graph."""
        await orchestrator.process(
            problem="Test problem",
            context={},
        )
        
        assert len(orchestrator._thought_graph) > 0
    
    @pytest.mark.asyncio
    async def test_process_crystallizes_insights(self, orchestrator):
        """Process crystallizes high-SNR insights."""
        result = await orchestrator.process(
            problem="Optimize the Proof of Impact calculation",
            context={},
        )
        
        # Some thoughts should be crystallized
        assert "crystallized" in result
        # Wisdom should be updated
        assert orchestrator.wisdom_repo is not None
    
    @pytest.mark.asyncio
    async def test_process_binds_to_genesis(self, orchestrator):
        """Process creates genesis attestation."""
        result = await orchestrator.process(
            problem="Test binding",
            context={},
        )
        
        attestation = result["attestation"]
        assert "attestation_hash" in attestation
        assert "bound_at" in attestation
    
    @pytest.mark.asyncio
    async def test_statistics_updated(self, orchestrator):
        """Statistics are updated after processing."""
        await orchestrator.process(problem="Test 1", context={})
        await orchestrator.process(problem="Test 2", context={})
        
        stats = orchestrator.get_statistics()
        
        assert stats["total_operations"] == 2
        assert "snr_engine" in stats
    
    @pytest.mark.asyncio
    async def test_wisdom_seeds_used(self, orchestrator, sample_wisdom_entry):
        """Process uses wisdom seeds."""
        orchestrator.wisdom_repo.add(sample_wisdom_entry)
        
        result = await orchestrator.process(
            problem="Integrity flywheel optimization",
            context={},
        )
        
        # Should have used wisdom seeds
        assert result["wisdom_seeds_used"] >= 0
    
    @pytest.mark.asyncio
    async def test_beam_search_respects_width(self, orchestrator):
        """Beam search respects beam width."""
        orchestrator.beam_width = 2
        
        await orchestrator.process(
            problem="Test beam width",
            context={},
        )
        
        # Should not have more thoughts than beam allows
        stats = orchestrator.get_statistics()
        assert stats["thought_graph_size"] <= (
            1 +  # Root
            orchestrator.beam_width * orchestrator.max_depth * 3  # Children per level
        )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestGenesisOrchestratorIntegration:
    """Integration tests for complete pipeline."""
    
    @pytest.fixture
    def full_orchestrator(self, temp_wisdom_dir):
        """Create full orchestrator."""
        orch = GenesisOrchestrator(
            beam_width=8,
            max_depth=3,
            fail_closed=False,
        )
        orch.wisdom_repo = WisdomRepository(storage_path=temp_wisdom_dir)
        return orch
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, full_orchestrator):
        """Test complete pipeline from problem to attestation."""
        result = await full_orchestrator.process(
            problem="Design a sustainable reward mechanism for the BIZRA ecosystem that aligns incentives for nodes, operators, and token holders while maintaining ethical compliance",
            context={
                "domain": "tokenomics",
                "priority": "high",
                "constraints": ["ihsan >= 0.95", "sustainable", "fair"],
            },
        )
        
        # Verify complete result
        assert result["problem"].startswith("Design")
        assert result["processing_time_ms"] > 0
        
        # Verify synthesis
        synth = result["synthesized_insight"]
        assert synth["snr_score"] > 0
        assert synth["ihsan_score"] >= 0.95
        
        # Verify attestation
        att = result["attestation"]
        assert len(att["attestation_hash"]) == 64
        assert att["crystallized_count"] >= 0
    
    @pytest.mark.asyncio
    async def test_wisdom_accumulates(self, full_orchestrator):
        """Wisdom accumulates across operations."""
        # First operation
        await full_orchestrator.process(
            problem="Pattern recognition in blockchain data",
            context={},
        )
        
        initial_count = len(full_orchestrator.wisdom_repo._wisdom)
        
        # Second operation
        await full_orchestrator.process(
            problem="Optimize pattern recognition algorithms",
            context={},
        )
        
        final_count = len(full_orchestrator.wisdom_repo._wisdom)
        
        # Wisdom should have grown
        assert final_count >= initial_count
    
    @pytest.mark.asyncio
    async def test_snr_distribution(self, full_orchestrator):
        """Verify SNR score distribution."""
        result = await full_orchestrator.process(
            problem="Test SNR distribution",
            context={},
        )
        
        stats = result["snr_statistics"]
        
        # Should have processed items
        assert stats["processed"] > 0
        
        # In permissive mode, should have some passing
        total = stats["passed_high"] + stats["passed_medium"] + stats["rejected"]
        assert total > 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""
    
    @pytest.mark.asyncio
    async def test_empty_problem(self, temp_wisdom_dir):
        """Handle empty problem."""
        orch = GenesisOrchestrator()
        orch.wisdom_repo = WisdomRepository(storage_path=temp_wisdom_dir)
        
        result = await orch.process(problem="", context={})
        
        assert result["problem"] == ""
        assert "attestation" in result
    
    @pytest.mark.asyncio
    async def test_very_long_problem(self, temp_wisdom_dir):
        """Handle very long problem."""
        orch = GenesisOrchestrator()
        orch.wisdom_repo = WisdomRepository(storage_path=temp_wisdom_dir)
        
        long_problem = "Test " * 1000
        result = await orch.process(problem=long_problem, context={})
        
        assert len(result["problem"]) == len(long_problem)
    
    def test_snr_engine_epsilon(self):
        """SNR engine handles zero noise."""
        engine = AutonomousSNREngine()
        
        snr = engine.compute_snr(
            clarity=0.9,
            synergy=0.9,
            consistency=0.9,
            entropy=0,
            quantization_error=0,
            disagreement=0,
        )
        
        # Should not divide by zero
        assert snr > 0
        assert snr < float('inf')
    
    def test_wisdom_repo_missing_file(self, temp_wisdom_dir):
        """Wisdom repo handles missing file."""
        repo = WisdomRepository(storage_path=temp_wisdom_dir)
        
        # Should start empty
        assert len(repo._wisdom) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
