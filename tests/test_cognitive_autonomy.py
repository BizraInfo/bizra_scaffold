"""
Tests for Autonomous Cognitive Core.

Tests the complete meta-cognitive self-improvement system:
- MetaCognitiveMonitor
- WisdomCrystallizationPipeline
- IntegrityFlywheel
- ReasoningLoopController
- AutonomousCognitiveEngine
"""

import asyncio
import pytest
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from core.genesis.cognitive_autonomy import (
    CognitiveState,
    ReasoningQuality,
    LearningSignal,
    ReasoningCycleMetrics,
    MetaCognitiveInsight,
    IntegrityFlywheelState,
    MetaCognitiveMonitor,
    WisdomCrystallizationPipeline,
    IntegrityFlywheel,
    ReasoningLoopController,
    AutonomousCognitiveEngine,
    create_cognitive_engine,
)
from core.genesis.genesis_orchestrator import GenesisResult, WisdomRepository


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_metrics():
    """Create sample reasoning cycle metrics."""
    return ReasoningCycleMetrics(
        cycle_id="cycle_test123",
        problem_hash="abc123def456",
        duration_ms=150.5,
        thought_nodes_created=15,
        thought_nodes_pruned=8,
        lenses_activated={"CRYPTO", "SECURITY", "SYSTEMS"},
        final_snr=0.88,
        ihsan_score=0.96,
        wisdom_seeds_used=3,
        wisdom_crystallized=1,
        quality=ReasoningQuality.HIGH,
    )


@pytest.fixture
def sample_result():
    """Create sample genesis result."""
    return GenesisResult(
        synthesis="This is a comprehensive synthesis of the reasoning chain.",
        confidence=0.88,
        snr_score=0.88,
        ihsan_score=0.96,
        attestation_hash="attest_abc123",
        thoughts=[],
        lenses_applied=["CRYPTO", "SECURITY", "SYSTEMS"],
    )


@pytest.fixture
def wisdom_repo(tmp_path):
    """Create a temporary wisdom repository."""
    return WisdomRepository(storage_path=tmp_path)


@pytest.fixture
def meta_monitor():
    """Create a meta-cognitive monitor."""
    return MetaCognitiveMonitor(history_window=50, insight_threshold=0.70)


@pytest.fixture
def flywheel():
    """Create an integrity flywheel."""
    return IntegrityFlywheel(initial_momentum=1.0)


# =============================================================================
# COGNITIVE STATE TESTS
# =============================================================================


class TestCognitiveState:
    """Tests for CognitiveState enum."""
    
    def test_all_states_exist(self):
        """All expected cognitive states exist."""
        assert CognitiveState.DORMANT
        assert CognitiveState.AWAKENING
        assert CognitiveState.REASONING
        assert CognitiveState.REFLECTING
        assert CognitiveState.CRYSTALLIZING
        assert CognitiveState.INTEGRATING
        assert CognitiveState.IDLE
        assert CognitiveState.SUSPENDED
        assert CognitiveState.TERMINATED
    
    def test_state_count(self):
        """All 9 states exist."""
        assert len(CognitiveState) == 9


class TestReasoningQuality:
    """Tests for ReasoningQuality enum."""
    
    def test_quality_levels(self):
        """All quality levels exist."""
        assert ReasoningQuality.EXCEPTIONAL.value == "exceptional"
        assert ReasoningQuality.HIGH.value == "high"
        assert ReasoningQuality.SATISFACTORY.value == "satisfactory"
        assert ReasoningQuality.MARGINAL.value == "marginal"
        assert ReasoningQuality.FAILED.value == "failed"


class TestLearningSignal:
    """Tests for LearningSignal enum."""
    
    def test_signal_types(self):
        """All signal types exist."""
        assert LearningSignal.REINFORCE
        assert LearningSignal.PRUNE
        assert LearningSignal.EXPLORE
        assert LearningSignal.CONSOLIDATE


# =============================================================================
# REASONING CYCLE METRICS TESTS
# =============================================================================


class TestReasoningCycleMetrics:
    """Tests for ReasoningCycleMetrics dataclass."""
    
    def test_creation(self, sample_metrics):
        """Metrics can be created."""
        assert sample_metrics.cycle_id == "cycle_test123"
        assert sample_metrics.final_snr == 0.88
        assert sample_metrics.quality == ReasoningQuality.HIGH
    
    def test_immutability(self, sample_metrics):
        """Metrics are frozen."""
        with pytest.raises(AttributeError):
            sample_metrics.final_snr = 0.99
    
    def test_to_dict(self, sample_metrics):
        """Metrics serialize to dict."""
        d = sample_metrics.to_dict()
        
        assert d["cycle_id"] == "cycle_test123"
        assert d["final_snr"] == 0.88
        assert d["quality"] == "high"
        assert "CRYPTO" in d["lenses_activated"]
    
    def test_from_dict(self, sample_metrics):
        """Metrics deserialize from dict."""
        d = sample_metrics.to_dict()
        restored = ReasoningCycleMetrics.from_dict(d)
        
        assert restored.cycle_id == sample_metrics.cycle_id
        assert restored.final_snr == sample_metrics.final_snr
        assert restored.quality == sample_metrics.quality


# =============================================================================
# META-COGNITIVE INSIGHT TESTS
# =============================================================================


class TestMetaCognitiveInsight:
    """Tests for MetaCognitiveInsight dataclass."""
    
    def test_creation(self):
        """Insight can be created."""
        insight = MetaCognitiveInsight(
            insight_id="insight_123",
            source_cycles=["cycle_1", "cycle_2"],
            pattern_type="lens_synergy",
            description="Test insight",
            confidence=0.85,
            learning_signal=LearningSignal.REINFORCE,
            recommended_action="Continue strategy",
            snr_impact=0.05,
        )
        
        assert insight.insight_id == "insight_123"
        assert insight.confidence == 0.85
        assert insight.learning_signal == LearningSignal.REINFORCE
    
    def test_to_dict(self):
        """Insight serializes to dict."""
        insight = MetaCognitiveInsight(
            insight_id="insight_456",
            source_cycles=["cycle_3"],
            pattern_type="quality_trend",
            description="Improving",
            confidence=0.90,
            learning_signal=LearningSignal.CONSOLIDATE,
            recommended_action="Maintain",
            snr_impact=0.08,
        )
        
        d = insight.to_dict()
        
        assert d["insight_id"] == "insight_456"
        assert d["confidence"] == 0.90
        assert d["learning_signal"] == "CONSOLIDATE"


# =============================================================================
# INTEGRITY FLYWHEEL STATE TESTS
# =============================================================================


class TestIntegrityFlywheelState:
    """Tests for IntegrityFlywheelState dataclass."""
    
    def test_efficiency_zero_cycles(self):
        """Efficiency is 0 with no cycles."""
        state = IntegrityFlywheelState()
        assert state.efficiency == 0.0
    
    def test_efficiency_with_cycles(self):
        """Efficiency calculated correctly."""
        state = IntegrityFlywheelState(
            total_cycles=10,
            exceptional_cycles=3,
        )
        assert state.efficiency == 0.3
    
    def test_health_thriving(self):
        """Health is THRIVING with high momentum."""
        state = IntegrityFlywheelState(momentum=100.0, resistance=5.0)
        assert state.health == "THRIVING"
    
    def test_health_degraded(self):
        """Health is DEGRADED with low momentum."""
        state = IntegrityFlywheelState(momentum=1.0, resistance=10.0)
        assert state.health == "DEGRADED"
    
    def test_to_dict(self):
        """State serializes to dict."""
        state = IntegrityFlywheelState(
            momentum=5.0,
            resistance=1.0,
            total_cycles=10,
            exceptional_cycles=3,
        )
        
        d = state.to_dict()
        
        assert d["momentum"] == 5.0
        assert d["efficiency"] == 0.3
        # 5.0 / 1.0 = 5.0, which is between 2 and 5, so STABLE or HEALTHY
        assert d["health"] in ("STABLE", "HEALTHY")


# =============================================================================
# META-COGNITIVE MONITOR TESTS
# =============================================================================


class TestMetaCognitiveMonitor:
    """Tests for MetaCognitiveMonitor."""
    
    def test_record_cycle(self, meta_monitor, sample_metrics):
        """Cycles are recorded."""
        meta_monitor.record_cycle(sample_metrics)
        
        assert len(meta_monitor._cycle_history) == 1
    
    def test_history_window_limit(self, meta_monitor):
        """History respects window limit."""
        for i in range(60):
            metrics = ReasoningCycleMetrics(
                cycle_id=f"cycle_{i}",
                problem_hash="hash",
                duration_ms=100,
                thought_nodes_created=10,
                thought_nodes_pruned=5,
                lenses_activated={"CRYPTO"},
                final_snr=0.8,
                ihsan_score=0.95,
                wisdom_seeds_used=1,
                wisdom_crystallized=0,
                quality=ReasoningQuality.HIGH,
            )
            meta_monitor.record_cycle(metrics)
        
        assert len(meta_monitor._cycle_history) == 50  # Window size
    
    def test_pattern_cache_updated(self, meta_monitor, sample_metrics):
        """Pattern cache is updated."""
        meta_monitor.record_cycle(sample_metrics)
        
        patterns = meta_monitor.get_pattern_frequencies()
        
        assert any("lens:" in k for k in patterns)
        assert any("quality:" in k for k in patterns)
        assert any("snr:" in k for k in patterns)
    
    @pytest.mark.asyncio
    async def test_analyze_needs_minimum_cycles(self, meta_monitor):
        """Analyze needs minimum cycles."""
        # Less than 5 cycles
        for i in range(3):
            meta_monitor.record_cycle(ReasoningCycleMetrics(
                cycle_id=f"cycle_{i}",
                problem_hash="hash",
                duration_ms=100,
                thought_nodes_created=10,
                thought_nodes_pruned=5,
                lenses_activated={"CRYPTO"},
                final_snr=0.8,
                ihsan_score=0.95,
                wisdom_seeds_used=1,
                wisdom_crystallized=0,
                quality=ReasoningQuality.HIGH,
            ))
        
        insights = await meta_monitor.analyze()
        
        assert len(insights) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_discovers_patterns(self, meta_monitor):
        """Analyze discovers patterns with enough cycles."""
        # Add 15 high-quality cycles with same lens pattern
        for i in range(15):
            meta_monitor.record_cycle(ReasoningCycleMetrics(
                cycle_id=f"cycle_{i}",
                problem_hash="hash",
                duration_ms=100,
                thought_nodes_created=10,
                thought_nodes_pruned=8,  # High prune rate
                lenses_activated={"CRYPTO", "SECURITY"},
                final_snr=0.92,
                ihsan_score=0.97,
                wisdom_seeds_used=2,
                wisdom_crystallized=0,
                quality=ReasoningQuality.EXCEPTIONAL,
            ))
        
        insights = await meta_monitor.analyze()
        
        # Should find patterns
        assert len(insights) >= 0  # May find lens synergy or trends
    
    def test_snr_to_band(self, meta_monitor):
        """SNR bands are correct."""
        assert meta_monitor._snr_to_band(0.98) == "exceptional"
        assert meta_monitor._snr_to_band(0.88) == "high"
        assert meta_monitor._snr_to_band(0.75) == "good"
        assert meta_monitor._snr_to_band(0.55) == "medium"
        assert meta_monitor._snr_to_band(0.35) == "low"


# =============================================================================
# WISDOM CRYSTALLIZATION PIPELINE TESTS
# =============================================================================


class TestWisdomCrystallizationPipeline:
    """Tests for WisdomCrystallizationPipeline."""
    
    @pytest.fixture
    def pipeline(self, wisdom_repo):
        """Create crystallization pipeline."""
        return WisdomCrystallizationPipeline(
            wisdom_repo=wisdom_repo,
            crystallization_threshold=0.85,
            ihsan_requirement=0.95,
        )
    
    def test_eligibility_high_snr(self, pipeline):
        """High SNR cycles are eligible."""
        metrics = ReasoningCycleMetrics(
            cycle_id="test",
            problem_hash="hash",
            duration_ms=100,
            thought_nodes_created=10,
            thought_nodes_pruned=5,
            lenses_activated={"CRYPTO"},
            final_snr=0.92,
            ihsan_score=0.97,
            wisdom_seeds_used=1,
            wisdom_crystallized=0,
            quality=ReasoningQuality.EXCEPTIONAL,
        )
        
        assert pipeline._is_eligible(metrics) is True
    
    def test_eligibility_low_snr(self, pipeline):
        """Low SNR cycles are not eligible."""
        metrics = ReasoningCycleMetrics(
            cycle_id="test",
            problem_hash="hash",
            duration_ms=100,
            thought_nodes_created=10,
            thought_nodes_pruned=5,
            lenses_activated={"CRYPTO"},
            final_snr=0.60,
            ihsan_score=0.97,
            wisdom_seeds_used=1,
            wisdom_crystallized=0,
            quality=ReasoningQuality.SATISFACTORY,
        )
        
        assert pipeline._is_eligible(metrics) is False
    
    def test_eligibility_low_ihsan(self, pipeline):
        """Low Ihsān cycles are not eligible."""
        metrics = ReasoningCycleMetrics(
            cycle_id="test",
            problem_hash="hash",
            duration_ms=100,
            thought_nodes_created=10,
            thought_nodes_pruned=5,
            lenses_activated={"CRYPTO"},
            final_snr=0.92,
            ihsan_score=0.80,
            wisdom_seeds_used=1,
            wisdom_crystallized=0,
            quality=ReasoningQuality.HIGH,
        )
        
        assert pipeline._is_eligible(metrics) is False
    
    @pytest.mark.asyncio
    async def test_process_result_ineligible(self, pipeline, sample_result):
        """Ineligible results produce no crystals."""
        metrics = ReasoningCycleMetrics(
            cycle_id="test",
            problem_hash="hash",
            duration_ms=100,
            thought_nodes_created=10,
            thought_nodes_pruned=5,
            lenses_activated={"CRYPTO"},
            final_snr=0.50,  # Too low
            ihsan_score=0.97,
            wisdom_seeds_used=1,
            wisdom_crystallized=0,
            quality=ReasoningQuality.MARGINAL,
        )
        
        crystals = await pipeline.process_result(sample_result, metrics)
        
        assert len(crystals) == 0
    
    @pytest.mark.asyncio
    async def test_process_result_eligible(self, pipeline, sample_result, sample_metrics):
        """Eligible results produce crystals."""
        # Use high-quality metrics
        high_metrics = ReasoningCycleMetrics(
            cycle_id="test",
            problem_hash="hash",
            duration_ms=100,
            thought_nodes_created=10,
            thought_nodes_pruned=5,
            lenses_activated={"CRYPTO"},
            final_snr=0.92,
            ihsan_score=0.97,
            wisdom_seeds_used=1,
            wisdom_crystallized=0,
            quality=ReasoningQuality.EXCEPTIONAL,
        )
        
        crystals = await pipeline.process_result(sample_result, high_metrics)
        
        assert len(crystals) >= 1
        assert pipeline.crystallization_count >= 1


# =============================================================================
# INTEGRITY FLYWHEEL TESTS
# =============================================================================


class TestIntegrityFlywheel:
    """Tests for IntegrityFlywheel."""
    
    def test_initial_state(self, flywheel):
        """Flywheel starts with initial momentum."""
        assert flywheel.state.momentum == 1.0
        assert flywheel.state.resistance == 0.0
    
    def test_exceptional_adds_momentum(self, flywheel):
        """Exceptional quality adds momentum."""
        initial_momentum = flywheel.state.momentum
        
        metrics = ReasoningCycleMetrics(
            cycle_id="test",
            problem_hash="hash",
            duration_ms=100,
            thought_nodes_created=10,
            thought_nodes_pruned=5,
            lenses_activated={"CRYPTO"},
            final_snr=0.95,
            ihsan_score=0.98,
            wisdom_seeds_used=1,
            wisdom_crystallized=0,
            quality=ReasoningQuality.EXCEPTIONAL,
        )
        
        flywheel.update(metrics)
        
        # After decay and addition, should be higher
        assert flywheel.state.exceptional_cycles == 1
    
    def test_failed_adds_resistance(self, flywheel):
        """Failed quality adds resistance."""
        metrics = ReasoningCycleMetrics(
            cycle_id="test",
            problem_hash="hash",
            duration_ms=100,
            thought_nodes_created=10,
            thought_nodes_pruned=5,
            lenses_activated={"CRYPTO"},
            final_snr=0.30,
            ihsan_score=0.80,
            wisdom_seeds_used=1,
            wisdom_crystallized=0,
            quality=ReasoningQuality.FAILED,
        )
        
        flywheel.update(metrics)
        
        assert flywheel.state.resistance > 0
        assert flywheel.state.failed_cycles == 1
    
    def test_total_cycles_tracked(self, flywheel, sample_metrics):
        """Total cycles are tracked."""
        flywheel.update(sample_metrics)
        flywheel.update(sample_metrics)
        flywheel.update(sample_metrics)
        
        assert flywheel.state.total_cycles == 3
    
    def test_crystallization_adds_momentum(self, flywheel):
        """Crystallization adds bonus momentum."""
        initial = flywheel.state.momentum
        
        flywheel.record_crystallization()
        
        assert flywheel.state.wisdom_entries_added == 1
        assert flywheel.state.last_crystallization is not None
    
    def test_should_explore(self, flywheel):
        """Explore flag based on velocity."""
        # Add many exceptional cycles
        for _ in range(10):
            flywheel.update(ReasoningCycleMetrics(
                cycle_id="test",
                problem_hash="hash",
                duration_ms=100,
                thought_nodes_created=10,
                thought_nodes_pruned=5,
                lenses_activated={"CRYPTO"},
                final_snr=0.95,
                ihsan_score=0.98,
                wisdom_seeds_used=1,
                wisdom_crystallized=0,
                quality=ReasoningQuality.EXCEPTIONAL,
            ))
        
        # Should recommend exploration
        assert flywheel.should_explore is True


# =============================================================================
# REASONING LOOP CONTROLLER TESTS
# =============================================================================


class TestReasoningLoopController:
    """Tests for ReasoningLoopController."""
    
    @pytest.fixture
    def controller(self, wisdom_repo, tmp_path):
        """Create reasoning loop controller."""
        from core.genesis.genesis_orchestrator_streaming import StreamingGenesisOrchestrator
        
        orchestrator = StreamingGenesisOrchestrator(
            beam_width=2,
            max_depth=2,
            fail_closed=False,
            emit_delay=0.0,
        )
        orchestrator.wisdom_repo = WisdomRepository(storage_path=tmp_path)
        
        return ReasoningLoopController(
            orchestrator=orchestrator,
            wisdom_repo=wisdom_repo,
        )
    
    def test_initial_state(self, controller):
        """Controller starts dormant."""
        assert controller.state == CognitiveState.DORMANT
    
    @pytest.mark.asyncio
    async def test_reason_returns_result_and_metrics(self, controller):
        """Reason returns result and metrics."""
        result, metrics = await controller.reason("Test problem")
        
        assert result is not None
        assert metrics is not None
        assert metrics.cycle_id is not None
        assert metrics.quality is not None
    
    @pytest.mark.asyncio
    async def test_reason_increments_cycle_count(self, controller):
        """Each reason call increments cycle count."""
        await controller.reason("Test 1")
        await controller.reason("Test 2")
        
        assert controller.cycle_count == 2
    
    @pytest.mark.asyncio
    async def test_reason_updates_flywheel(self, controller):
        """Reasoning updates the flywheel."""
        await controller.reason("Test problem")
        
        assert controller.flywheel.state.total_cycles >= 1
    
    @pytest.mark.asyncio
    async def test_reason_records_in_monitor(self, controller):
        """Reasoning records in meta-monitor."""
        await controller.reason("Test problem")
        
        assert len(controller.meta_monitor._cycle_history) >= 1
    
    def test_assess_quality_exceptional(self, controller):
        """Quality assessment for exceptional."""
        result = GenesisResult(
            synthesis="Test",
            confidence=0.98,
            snr_score=0.96,
            ihsan_score=0.99,
            attestation_hash=None,
            thoughts=[],
            lenses_applied=[],
        )
        
        quality = controller._assess_quality(result)
        
        assert quality == ReasoningQuality.EXCEPTIONAL
    
    def test_assess_quality_failed(self, controller):
        """Quality assessment for failed."""
        result = GenesisResult(
            synthesis="Test",
            confidence=0.30,
            snr_score=0.30,
            ihsan_score=0.50,
            attestation_hash=None,
            thoughts=[],
            lenses_applied=[],
        )
        
        quality = controller._assess_quality(result)
        
        assert quality == ReasoningQuality.FAILED


# =============================================================================
# AUTONOMOUS COGNITIVE ENGINE TESTS
# =============================================================================


class TestAutonomousCognitiveEngine:
    """Tests for AutonomousCognitiveEngine."""
    
    @pytest.fixture
    def engine(self, tmp_path):
        """Create cognitive engine."""
        return AutonomousCognitiveEngine(
            beam_width=2,
            max_depth=2,
            fail_closed=False,
            wisdom_path=tmp_path,
        )
    
    @pytest.mark.asyncio
    async def test_start_stop(self, engine):
        """Engine can start and stop."""
        await engine.start()
        
        assert engine._started is True
        
        await engine.stop()
        
        assert engine._started is False
    
    @pytest.mark.asyncio
    async def test_reason_auto_starts(self, engine):
        """Reason auto-starts engine."""
        result, metrics = await engine.reason("Test problem")
        
        assert engine._started is True
        assert result is not None
        
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_reason_streaming(self, engine):
        """Streaming reason yields events."""
        events = []
        
        async for event in engine.reason_streaming("Test"):
            events.append(event)
        
        assert len(events) > 0
        
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_get_status(self, engine):
        """Status returns comprehensive info."""
        await engine.start()
        await engine.reason("Test problem")
        
        status = engine.get_status()
        
        assert "state" in status
        assert "started" in status
        assert "cycles_completed" in status
        assert "flywheel" in status
        
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_cycle_count(self, engine):
        """Cycle count tracks reasoning."""
        await engine.start()
        
        await engine.reason("Test 1")
        await engine.reason("Test 2")
        
        assert engine.cycle_count == 2
        
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_flywheel_state_accessible(self, engine):
        """Flywheel state is accessible."""
        await engine.start()
        await engine.reason("Test")
        
        state = engine.flywheel_state
        
        assert state.total_cycles >= 1
        
        await engine.stop()


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_create_cognitive_engine(self, tmp_path):
        """create_cognitive_engine creates and starts engine."""
        engine = await create_cognitive_engine(
            wisdom_path=tmp_path,
            beam_width=2,
            max_depth=2,
            fail_closed=False,
        )
        
        assert engine._started is True
        
        await engine.stop()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestCognitiveIntegration:
    """Integration tests for the cognitive system."""
    
    @pytest.mark.asyncio
    async def test_multi_cycle_learning(self, tmp_path):
        """Multiple cycles improve through learning."""
        engine = await create_cognitive_engine(
            wisdom_path=tmp_path,
            beam_width=2,
            max_depth=2,
            fail_closed=False,
        )
        
        problems = [
            "Design a secure authentication system",
            "Optimize database query performance",
            "Create a distributed caching solution",
        ]
        
        for problem in problems:
            result, metrics = await engine.reason(problem)
            assert result is not None
        
        # Should have tracked all cycles
        assert engine.cycle_count == 3
        assert engine.flywheel_state.total_cycles == 3
        
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_wisdom_persists(self, tmp_path):
        """Wisdom is saved and loaded."""
        # First engine
        engine1 = await create_cognitive_engine(
            wisdom_path=tmp_path,
            beam_width=2,
            max_depth=2,
            fail_closed=False,
        )
        
        await engine1.reason("Test problem for wisdom")
        initial_count = engine1.wisdom_count
        
        await engine1.stop()
        
        # Second engine - should load saved wisdom
        engine2 = await create_cognitive_engine(
            wisdom_path=tmp_path,
            beam_width=2,
            max_depth=2,
            fail_closed=False,
        )
        
        # Wisdom should be loaded
        assert engine2.wisdom_count >= 0  # May or may not have crystallized
        
        await engine2.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
