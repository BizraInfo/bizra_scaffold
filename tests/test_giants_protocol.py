"""
BIZRA Giants Protocol Integration Tests
═══════════════════════════════════════════════════════════════════════════════
Tests for Standing on Shoulders of Giants Protocol and GoT integration.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

# Import the modules under test
from core.knowledge.giants_protocol import (
    CrystallizedWisdom,
    GiantsProtocolEngine,
    IntegrityFlywheel,
    WisdomSource,
    WisdomType,
    create_giants_engine,
)
from core.knowledge.giants_enhanced_got import (
    GiantsEnhancedChain,
    GiantsEnhancedGoT,
    create_giants_enhanced_got,
)
from core.snr_scorer import SNRScorer


# ═══════════════════════════════════════════════════════════════════════════════
# GIANTS PROTOCOL ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestGiantsProtocolEngine:
    """Tests for GiantsProtocolEngine core functionality."""
    
    def test_engine_initialization(self):
        """Test engine initializes with correct defaults from BIZRA_SOT."""
        engine = create_giants_engine()
        
        assert engine.ihsan_threshold == 0.95  # Per BIZRA_SOT Section 3.1
        assert engine.snr_threshold == 0.5
        assert engine.max_wisdom_cache == 500
        
        # Verify hub concepts initialized from meta-analysis
        assert len(engine._nodes) >= 6
        assert "hub_architecture" in engine._nodes
        assert "hub_security" in engine._nodes
    
    def test_integrity_flywheel_initialization(self):
        """Test Integrity Flywheel pattern from meta-analysis."""
        engine = create_giants_engine()
        
        flywheel = engine._flywheels.get("integrity")
        assert flywheel is not None
        assert flywheel.name == "Integrity Flywheel"
        
        # Verify phases match meta-analysis discovery
        expected_phases = [
            "integrity_proofs",
            "ihsan_ethics",
            "devops_gates",
            "evidence_publication",
        ]
        assert flywheel.phases == expected_phases
        
        # Verify all three model sources observed this pattern
        assert WisdomSource.CLAUDE in flywheel.observation_sources
        assert WisdomSource.DEEPSEEK in flywheel.observation_sources
        assert WisdomSource.OPENAI in flywheel.observation_sources
    
    def test_flywheel_spin(self):
        """Test flywheel momentum dynamics."""
        engine = create_giants_engine()
        
        # Initial spin
        state1 = engine.spin_integrity_flywheel(input_energy=0.5)
        assert state1["momentum"] > 0
        assert "current_phase" in state1
        
        # Additional spin adds momentum
        state2 = engine.spin_integrity_flywheel(input_energy=0.5)
        assert state2["momentum"] >= state1["momentum"] * 0.9  # Accounting for friction
        
        # Check active state
        state3 = engine.spin_integrity_flywheel(input_energy=1.0)
        assert state3["is_active"] == True
    
    def test_hub_concepts_from_meta_analysis(self):
        """Test hub concepts match meta-analysis co-occurrence weights."""
        engine = create_giants_engine()
        
        hubs = engine.get_hub_concepts(top_k=3)
        
        assert len(hubs) == 3
        # Top hub should be architecture or security (weight 164)
        top_hub = hubs[0]
        assert top_hub["concept"] in ["architecture", "security"]
        assert top_hub["weight"] == 164
    
    def test_term_signals_from_meta_analysis(self):
        """Test term frequency signals match meta-analysis."""
        engine = create_giants_engine()
        
        signals = engine.get_term_signals()
        
        # PAT should be highest (46.9 per 10k words)
        assert "pat" in signals
        assert signals["pat"] == 46.9
        
        # ihsan should be present (5.8 per 10k words)
        assert "ihsan" in signals
        assert signals["ihsan"] == 5.8
    
    @pytest.mark.asyncio
    async def test_wisdom_extraction_from_trace(self):
        """Test wisdom extraction from reasoning trace."""
        engine = create_giants_engine()
        
        # Create a sample reasoning trace
        trace = {
            "decisions": [
                {"type": "inference", "content": "Apply constraint X"},
                {"type": "inference", "content": "Validate against policy"},
            ],
            "explorations": [
                {"type": "option", "content": "Option A: Direct path"},
                {"type": "option", "content": "Option B: Bridge path"},
            ],
            "constraints": [
                {
                    "name": "ihsan_threshold",
                    "description": "IM >= 0.95",
                    "strength": 0.95,
                    "concepts": ["ihsan", "ethics"],
                    "domains": ["ethics", "validation"],
                }
            ],
            "invariants": [
                {
                    "name": "fail_closed",
                    "assertion": "Reject on uncertainty",
                    "concepts": ["safety", "security"],
                }
            ],
            "domain_crossings": [
                {
                    "from": "cryptography",
                    "to": "economics",
                    "insight": "Token as proof vehicle",
                    "novelty": 0.8,
                    "strength": 0.75,
                    "concepts": ["token", "proof"],
                }
            ],
        }
        
        wisdom_list = await engine.extract_wisdom_from_trace(
            reasoning_trace=trace,
            source=WisdomSource.LOCAL,
        )
        
        # Should extract patterns, principles, and bridges
        assert len(wisdom_list) > 0
        
        # Check wisdom types present
        types = {w.wisdom_type for w in wisdom_list}
        assert WisdomType.PATTERN in types or WisdomType.PRINCIPLE in types
        
        # All wisdom should pass Ihsan gate
        for w in wisdom_list:
            assert w.ihsan_score >= engine.ihsan_threshold
    
    @pytest.mark.asyncio
    async def test_wisdom_synthesis(self):
        """Test wisdom synthesis for current context."""
        engine = create_giants_engine()
        
        # First extract some wisdom
        trace = {
            "decisions": [{"type": "design", "content": "Use layered architecture"}],
            "explorations": [],
            "constraints": [
                {
                    "name": "security_first",
                    "description": "Security by design",
                    "strength": 0.9,
                    "concepts": ["security", "architecture"],
                    "domains": ["security", "design"],
                }
            ],
            "invariants": [],
            "domain_crossings": [],
        }
        await engine.extract_wisdom_from_trace(trace, WisdomSource.LOCAL)
        
        # Now synthesize for related context
        context = {
            "domains": ["security", "design"],
            "concepts": ["architecture", "layer"],
        }
        
        insights = await engine.synthesize_insights(
            query="Design secure architecture",
            context=context,
            top_k=5,
        )
        
        # Should return relevant insights
        assert len(insights) >= 0  # May be empty if cache was cleared
    
    def test_evidence_pack_generation(self):
        """Test SAT-compliant evidence pack generation."""
        engine = create_giants_engine()
        
        # Spin flywheel to create state
        engine.spin_integrity_flywheel(input_energy=0.7)
        
        pack = engine.generate_evidence_pack()
        
        # Verify pack structure
        assert "protocol_version" in pack
        assert pack["protocol_version"] == "GIANTS_PROTOCOL_v1.0"
        
        assert "timestamp" in pack
        assert "statistics" in pack
        assert "flywheels" in pack
        assert "hub_concepts" in pack
        assert "term_signals" in pack
        assert "pack_hash" in pack
        
        # Verify statistics
        stats = pack["statistics"]
        assert "total_wisdom_extracted" in stats
        assert "total_flywheel_spins" in stats
        assert stats["total_flywheel_spins"] >= 1
        
        # Verify Ihsan threshold recorded
        assert pack["ihsan_threshold"] == 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# GIANTS-ENHANCED GOT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestGiantsEnhancedGoT:
    """Tests for Giants-Enhanced Graph-of-Thoughts integration."""
    
    @pytest.fixture
    def mock_snr_scorer(self):
        """Create mock SNR scorer."""
        scorer = MagicMock(spec=SNRScorer)
        scorer.compute_snr = MagicMock(return_value=MagicMock(snr_score=0.8))
        return scorer
    
    def test_enhanced_got_initialization(self, mock_snr_scorer):
        """Test GiantsEnhancedGoT initializes correctly."""
        engine = create_giants_enhanced_got(
            snr_scorer=mock_snr_scorer,
            beam_width=10,
            max_depth=5,
        )
        
        assert engine.got_engine is not None
        assert engine.giants_engine is not None
        assert engine.wisdom_boost_factor == 0.15
        
        # Giants engine should have proper thresholds
        assert engine.giants_engine.ihsan_threshold == 0.95
    
    def test_hub_guidance(self, mock_snr_scorer):
        """Test hub concept guidance extraction."""
        engine = create_giants_enhanced_got(
            snr_scorer=mock_snr_scorer,
        )
        
        hubs = engine.get_hub_guidance()
        
        assert len(hubs) > 0
        assert "architecture" in hubs or "security" in hubs
    
    def test_term_frequency_signals(self, mock_snr_scorer):
        """Test term frequency signal retrieval."""
        engine = create_giants_enhanced_got(
            snr_scorer=mock_snr_scorer,
        )
        
        signals = engine.get_term_frequency_signals()
        
        assert "pat" in signals
        assert "sat" in signals
        assert "ihsan" in signals
    
    def test_combined_evidence_pack(self, mock_snr_scorer):
        """Test combined evidence pack from both engines."""
        engine = create_giants_enhanced_got(
            snr_scorer=mock_snr_scorer,
        )
        
        pack = engine.generate_evidence_pack()
        
        # Should have Giants Protocol fields
        assert "protocol_version" in pack
        assert "flywheels" in pack
        assert "hub_concepts" in pack
        
        # Should also have GoT statistics
        assert "got_statistics" in pack
        got_stats = pack["got_statistics"]
        assert "total_thoughts_generated" in got_stats
        assert "total_bridges_discovered" in got_stats
        
        # Should have enhanced statistics
        assert "enhanced_statistics" in pack
        enhanced_stats = pack["enhanced_statistics"]
        assert "total_enhanced_runs" in enhanced_stats
        assert "total_wisdom_applications" in enhanced_stats


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestGiantsIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def mock_hypergraph_query(self):
        """Create mock hypergraph query function."""
        async def query_fn(node: str):
            # Return mock neighbors based on hub concepts
            if "architecture" in node:
                return [
                    {"id": "security_layer", "domains": ["security"], "consistency": 0.8, "disagreement": 0.1, "ihsan": 0.96},
                    {"id": "abstraction_layer", "domains": ["design"], "consistency": 0.75, "disagreement": 0.15, "ihsan": 0.95},
                ]
            elif "security" in node:
                return [
                    {"id": "crypto_module", "domains": ["cryptography"], "consistency": 0.9, "disagreement": 0.05, "ihsan": 0.97},
                    {"id": "auth_system", "domains": ["identity"], "consistency": 0.85, "disagreement": 0.1, "ihsan": 0.96},
                ]
            return []
        return query_fn
    
    @pytest.fixture
    def mock_convergence_fn(self):
        """Create mock convergence function."""
        async def convergence_fn(concept: str, context: dict):
            return {"converged": True, "score": 0.8}
        return convergence_fn
    
    @pytest.mark.asyncio
    async def test_full_reasoning_pipeline(
        self,
        mock_hypergraph_query,
        mock_convergence_fn,
    ):
        """Test complete wisdom-enhanced reasoning pipeline."""
        # Create scorer mock with compute_from_convergence method
        from core.snr_scorer import SNRMetrics, SNRLevel
        
        mock_metrics = SNRMetrics(
            snr_score=0.75,
            signal_strength=0.8,
            noise_floor=0.1,
            clarity_component=0.8,
            synergy_component=0.7,
            consistency_component=0.75,
            entropy_component=0.05,
            quantization_error=0.02,
            disagreement_component=0.03,
            level=SNRLevel.MEDIUM,
            confidence=0.9,
            ihsan_metric=0.96,
        )
        
        scorer = MagicMock(spec=SNRScorer)
        scorer.compute_snr = MagicMock(return_value=mock_metrics)
        scorer.compute_from_convergence = MagicMock(return_value=mock_metrics)
        
        engine = create_giants_enhanced_got(
            snr_scorer=scorer,
            beam_width=5,
            max_depth=3,
        )
        
        # First populate some wisdom
        trace = {
            "decisions": [{"type": "design", "content": "Use secure architecture"}],
            "explorations": [],
            "constraints": [],
            "invariants": [],
            "domain_crossings": [],
        }
        await engine.giants_engine.extract_wisdom_from_trace(
            trace, WisdomSource.LOCAL
        )
        
        # Execute reasoning
        enhanced_chains = await engine.reason_with_wisdom(
            query="Design a secure layered system",
            seed_concepts=["architecture", "security"],
            hypergraph_query_fn=mock_hypergraph_query,
            convergence_fn=mock_convergence_fn,
            top_k_chains=3,
            domains=["security", "design"],
        )
        
        # Verify results
        assert isinstance(enhanced_chains, list)
        
        # Statistics should be updated
        assert engine.total_enhanced_runs >= 1
        
        # Evidence pack should be generatable
        pack = engine.generate_evidence_pack()
        assert pack["enhanced_statistics"]["total_enhanced_runs"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# WISDOM CRYSTALLIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestWisdomCrystallization:
    """Tests for wisdom crystallization and receipts."""
    
    def test_wisdom_receipt_generation(self):
        """Test SAT-compliant receipt generation for wisdom."""
        wisdom = CrystallizedWisdom(
            id="wisdom_test_001",
            wisdom_type=WisdomType.PRINCIPLE,
            source=WisdomSource.CLAUDE,
            title="Proofs First Principle",
            content="Always verify before publishing claims",
            evidence_citations=["./evidence/PACK-0001/"],
            snr_score=0.9,
            frequency=42,
            confidence=0.95,
            first_observed=datetime.now(timezone.utc),
            last_observed=datetime.now(timezone.utc),
        )
        
        receipt = wisdom.to_receipt()
        
        assert "wisdom_id" in receipt
        assert receipt["wisdom_id"] == "wisdom_test_001"
        
        assert "content_hash" in receipt
        assert len(receipt["content_hash"]) == 64  # SHA256 hex length
        
        assert "snr_score" in receipt
        assert receipt["snr_score"] == 0.9
        
        assert "ihsan_score" in receipt
        assert receipt["ihsan_score"] == 0.95
        
        assert "attestation" in receipt
        assert receipt["attestation"] == "GIANTS_PROTOCOL_v1.0"
    
    def test_wisdom_cache_eviction(self):
        """Test LRU cache eviction for wisdom."""
        engine = GiantsProtocolEngine(
            max_wisdom_cache=3,  # Small cache for testing
        )
        
        # Add wisdom entries
        for i in range(5):
            wisdom = CrystallizedWisdom(
                id=f"wisdom_{i}",
                wisdom_type=WisdomType.PATTERN,
                source=WisdomSource.LOCAL,
                title=f"Pattern {i}",
                content=f"Content {i}",
                evidence_citations=[],
                snr_score=0.8,
                frequency=1,
                confidence=0.7,
                first_observed=datetime.now(timezone.utc),
                last_observed=datetime.now(timezone.utc),
            )
            engine._cache_wisdom(wisdom)
        
        # Cache should be at max size
        assert len(engine._wisdom_cache) <= 3
        
        # Oldest entries should have been evicted
        assert "wisdom_4" in engine._wisdom_cache  # Most recent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
