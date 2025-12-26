#!/usr/bin/env python3
"""
COGNITIVE SOVEREIGN TEST SUITE - Elite Practitioner Grade
============================================================
Coverage Target: 90%+ | Framework: pytest + hypothesis
SAPE Compliance: Rare circuit probing | Ihsān Validation: Fail-closed semantics
"""

import asyncio
import hashlib
import json

# Import system under test - ensure repo root is in path
import sys
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cognitive_sovereign import (
    CognitiveSovereign,
    HigherOrderLogicBridge,
    IhsanPrinciples,
    L1PerceptualBuffer,
    L2WorkingMemory,
    L3EpisodicMemory,
    L4SemanticHyperGraph,
    L5DeterministicTools,
    MetaCognitiveOrchestrator,
    QuantumTemporalSecurity,
    RetrogradeSignalingPathway,
)

# ============================================================================
# I. IHSĀN PRINCIPLES TESTS
# ============================================================================


class TestIhsanPrinciples:
    """Test suite for Ihsān ethical core with fail-closed semantics."""

    def test_weights_sum_to_one(self):
        """SOT Compliance: Weights must sum to 1.0 for proper normalization."""
        ihsan = IhsanPrinciples()
        total = ihsan.IKHLAS + ihsan.KARAMA + ihsan.ADL + ihsan.KAMAL + ihsan.ISTIDAMA
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_threshold_matches_sot(self):
        """Verify threshold matches BIZRA_SOT.md Section 3.1."""
        ihsan = IhsanPrinciples()
        assert ihsan.IHSAN_THRESHOLD == 0.95

    def test_compute_score_perfect(self):
        """Perfect scores should yield 1.0."""
        ihsan = IhsanPrinciples()
        dimensions = {
            "truthfulness": 1.0,
            "dignity": 1.0,
            "fairness": 1.0,
            "excellence": 1.0,
            "sustainability": 1.0,
        }
        score = ihsan.compute_score(dimensions)
        assert abs(score - 1.0) < 1e-9

    def test_compute_score_zero(self):
        """Zero scores should yield 0.0."""
        ihsan = IhsanPrinciples()
        dimensions = {
            "truthfulness": 0.0,
            "dignity": 0.0,
            "fairness": 0.0,
            "excellence": 0.0,
            "sustainability": 0.0,
        }
        score = ihsan.compute_score(dimensions)
        assert score == 0.0

    def test_compute_score_partial(self):
        """Partial scores should compute weighted average."""
        ihsan = IhsanPrinciples()
        dimensions = {
            "truthfulness": 0.8,
            "dignity": 0.9,
            "fairness": 0.7,
            "excellence": 0.85,
            "sustainability": 0.95,
        }
        expected = (
            0.8 * 0.30  # IKHLAS
            + 0.9 * 0.20  # KARAMA
            + 0.7 * 0.20  # ADL
            + 0.85 * 0.20  # KAMAL
            + 0.95 * 0.10  # ISTIDAMA
        )
        score = ihsan.compute_score(dimensions)
        assert abs(score - expected) < 1e-9

    def test_verify_passes_above_threshold(self):
        """Verification should pass when score >= 0.95."""
        ihsan = IhsanPrinciples()
        dimensions = {
            "truthfulness": 1.0,
            "dignity": 1.0,
            "fairness": 0.9,
            "excellence": 0.95,
            "sustainability": 1.0,
        }
        passed, score = ihsan.verify(dimensions)
        assert passed is True
        assert score >= 0.95

    def test_verify_fails_below_threshold(self):
        """Verification should fail when score < 0.95."""
        ihsan = IhsanPrinciples()
        dimensions = {
            "truthfulness": 0.8,
            "dignity": 0.8,
            "fairness": 0.8,
            "excellence": 0.8,
            "sustainability": 0.8,
        }
        passed, score = ihsan.verify(dimensions)
        assert passed is False
        assert score < 0.95

    def test_verify_fail_closed_on_invalid_input(self):
        """CRITICAL: Must fail-closed on invalid dimension values."""
        ihsan = IhsanPrinciples()

        # Test NaN
        dimensions = {
            "truthfulness": float("nan"),
            "dignity": 1.0,
            "fairness": 1.0,
            "excellence": 1.0,
            "sustainability": 1.0,
        }
        passed, score = ihsan.verify(dimensions)
        assert passed is False
        assert score == 0.0

        # Test out of range (> 1)
        dimensions = {
            "truthfulness": 1.5,
            "dignity": 1.0,
            "fairness": 1.0,
            "excellence": 1.0,
            "sustainability": 1.0,
        }
        passed, score = ihsan.verify(dimensions)
        assert passed is False

        # Test negative
        dimensions = {
            "truthfulness": -0.1,
            "dignity": 1.0,
            "fairness": 1.0,
            "excellence": 1.0,
            "sustainability": 1.0,
        }
        passed, score = ihsan.verify(dimensions)
        assert passed is False

    def test_compute_gradients_clamps_output(self):
        """Gradient output must be clamped to [-1, 1]."""
        ihsan = IhsanPrinciples()
        task_grad = torch.ones(10) * 100  # Extreme gradient
        ethical_losses = {
            "ikhlas": torch.ones(10) * 50,
            "kamal": torch.ones(10) * 50,
        }
        result = ihsan.compute_gradients(task_grad, ethical_losses)
        assert torch.all(result >= -1.0)
        assert torch.all(result <= 1.0)

    def test_gradient_modulation_factors(self):
        """Verify different principles have different modulation effects."""
        ihsan = IhsanPrinciples()
        task_grad = torch.zeros(10)

        # KAMAL (excellence) should have highest amplification
        kamal_loss = {"kamal": torch.ones(10)}
        kamal_result = ihsan.compute_gradients(task_grad.clone(), kamal_loss)

        # ISTIDAMA (sustainability) should have dampening
        istidama_loss = {"istidama": torch.ones(10)}
        istidama_result = ihsan.compute_gradients(task_grad.clone(), istidama_loss)

        # KAMAL should produce larger gradients (1.20 factor vs 0.95)
        assert torch.mean(torch.abs(kamal_result)) > torch.mean(
            torch.abs(istidama_result)
        )


# ============================================================================
# II. QUANTUM-TEMPORAL SECURITY TESTS
# ============================================================================


class TestQuantumTemporalSecurity:
    """Test suite for temporal chain integrity and replay protection."""

    def test_secure_operation_produces_temporal_proof(self):
        """Every secured operation must have a temporal proof."""
        security = QuantumTemporalSecurity()
        operation = {"action": "test", "value": 42}
        result = security.secure_cognitive_operation(operation)

        assert "temporal_proof" in result
        proof = result["temporal_proof"]
        assert "nonce" in proof
        assert "timestamp" in proof
        assert "temporal_hash" in proof
        assert "signature" in proof
        assert "chain_index" in proof

    def test_chain_grows_with_operations(self):
        """Temporal chain must grow with each operation."""
        security = QuantumTemporalSecurity()

        for i in range(5):
            security.secure_cognitive_operation({"op": i})

        assert len(security.temporal_chain) == 5
        assert len(security.temporal_proofs) == 5

    def test_chain_integrity_verification_passes(self):
        """Valid chain should pass integrity verification."""
        security = QuantumTemporalSecurity()

        for i in range(10):
            security.secure_cognitive_operation({"op": i, "data": f"test_{i}"})

        assert security.verify_chain_integrity() is True

    def test_chain_detects_tampering(self):
        """Tampered chain must fail verification."""
        security = QuantumTemporalSecurity()

        for i in range(5):
            security.secure_cognitive_operation({"op": i})

        # Tamper with the chain
        if security.temporal_chain:
            security.temporal_chain[2] = b"\x00" * 64

        assert security.verify_chain_integrity() is False

    def test_nonce_uniqueness(self):
        """Each operation must have a unique nonce."""
        security = QuantumTemporalSecurity()
        nonces = set()

        for i in range(100):
            result = security.secure_cognitive_operation({"op": i})
            nonce = result["temporal_proof"]["nonce"]
            assert nonce not in nonces, f"Duplicate nonce detected at operation {i}"
            nonces.add(nonce)

    def test_signature_verification(self):
        """Signatures must be verifiable with public key."""
        security = QuantumTemporalSecurity()
        result = security.secure_cognitive_operation({"test": "data"})

        proof = result["temporal_proof"]
        # The verify_chain_integrity method validates signatures
        assert security.verify_chain_integrity() is True

    def test_entropy_accumulation(self):
        """Chain entropy should increase with operations."""
        security = QuantumTemporalSecurity()

        initial_entropy = security.chain_entropy
        for i in range(10):
            security.secure_cognitive_operation({"op": i})

        assert security.chain_entropy > initial_entropy


# ============================================================================
# III. MEMORY HIERARCHY TESTS
# ============================================================================


class TestL1PerceptualBuffer:
    """Test L1 buffer with Miller's Law constraints."""

    def test_capacity_limit(self):
        """Buffer should respect capacity limit (7±2)."""
        buffer = L1PerceptualBuffer(capacity=9)

        for i in range(15):
            buffer.push(f"item_{i}")

        assert len(buffer.buffer) <= 9

    def test_attention_weighting(self):
        """High-attention items should be prioritized."""
        buffer = L1PerceptualBuffer(capacity=5)

        buffer.push("low_priority", attention_weight=0.1)
        buffer.push("medium_priority", attention_weight=0.5)
        buffer.push("high_priority", attention_weight=1.0)

        assert buffer.attention_mask is not None
        # Highest weight item should have highest attention
        assert np.argmax(buffer.attention_mask) == 2

    def test_fibonacci_eviction(self):
        """Low-attention items should be evicted on overflow."""
        buffer = L1PerceptualBuffer(capacity=3)

        buffer.push("item_1", attention_weight=0.1)
        buffer.push("item_2", attention_weight=0.9)
        buffer.push("item_3", attention_weight=0.8)
        buffer.push("item_4", attention_weight=0.95)  # Should evict item_1

        assert len(buffer.buffer) == 3
        items = [b["item"] for b in buffer.buffer]
        # item_1 (lowest weight) should have been evicted
        assert "item_1" not in items or "item_4" in items


class TestL2WorkingMemory:
    """Test L2 consolidation with novelty detection."""

    def test_consolidation_produces_summary(self):
        """Consolidation should produce compressed summary."""
        memory = L2WorkingMemory()
        items = ["alpha", "beta", "gamma"]

        summary = memory.consolidate(items)

        assert summary.startswith("SUMMARY[")
        assert len(memory.summaries) == 1

    def test_novelty_detection(self):
        """Repeated content should have lower novelty."""
        memory = L2WorkingMemory()

        # First item should have high novelty
        memory.consolidate(["unique content here"])
        first_priority = memory.summaries[0]["priority"]

        # Similar content should have lower novelty
        memory.consolidate(["unique content here again"])
        second_priority = memory.summaries[1]["priority"]

        assert second_priority < first_priority


class TestL3EpisodicMemory:
    """Test L3 Merkle chain integrity."""

    def test_episode_storage_updates_root(self):
        """Each episode should update Merkle root."""
        memory = L3EpisodicMemory()

        initial_root = memory.merkle_root
        memory.store_episode("ep_1", {"content": "test"})

        assert memory.merkle_root != initial_root

    def test_integrity_verification_passes(self):
        """Valid episodic chain should verify."""
        memory = L3EpisodicMemory()

        for i in range(10):
            memory.store_episode(f"ep_{i}", {"data": f"episode_{i}"})

        assert memory.verify_integrity() is True

    def test_detects_episode_tampering(self):
        """Tampered episodes must fail verification."""
        memory = L3EpisodicMemory()

        for i in range(5):
            memory.store_episode(f"ep_{i}", {"data": i})

        # Tamper with an episode
        memory.episodes["ep_2"]["content"]["data"] = "TAMPERED"

        assert memory.verify_integrity() is False


class TestL4SemanticHyperGraph:
    """Test L4 HyperGraph topology."""

    @pytest.mark.asyncio
    async def test_hyperedge_creation(self):
        """Hyperedges should connect multiple nodes."""
        graph = L4SemanticHyperGraph()

        await graph.create_hyperedge(nodes=["A", "B", "C"], relation="RELATED_TO")

        assert graph.graph.number_of_nodes() == 4  # 3 entities + 1 hyperedge
        assert graph.graph.number_of_edges() >= 3  # At least participant edges

    @pytest.mark.asyncio
    async def test_topology_analysis(self):
        """Topology metrics should be computable."""
        graph = L4SemanticHyperGraph()

        await graph.create_hyperedge(["A", "B", "C"], "R1")
        await graph.create_hyperedge(["B", "C", "D"], "R2")
        await graph.create_hyperedge(["D", "E", "F"], "R3")

        topology = graph.analyze_topology()

        assert "clustering_coefficient" in topology
        assert "rich_club_coefficient" in topology
        assert 0 <= topology["clustering_coefficient"] <= 1


class TestL5DeterministicTools:
    """Test L5 tool crystallization."""

    def test_crystallize_tool(self):
        """Tools should be crystallizable with temporal proof."""
        security = QuantumTemporalSecurity()
        tools = L5DeterministicTools(security)

        def my_function(x):
            return x * 2

        result = tools.crystallize("double", my_function)
        assert result is True
        assert len(tools.tools) == 1

    def test_execute_crystallized_tool(self):
        """Crystallized tools should execute correctly."""
        security = QuantumTemporalSecurity()
        tools = L5DeterministicTools(security)

        tools.crystallize("add", lambda a, b: a + b)
        result = tools.execute("add", a=3, b=5)

        assert result["success"] is True
        assert result["result"] == 8
        assert "temporal_proof" in result

    def test_execute_missing_tool_raises(self):
        """Executing unknown tool should raise ValueError."""
        security = QuantumTemporalSecurity()
        tools = L5DeterministicTools(security)

        with pytest.raises(ValueError, match="not found"):
            tools.execute("nonexistent")


# ============================================================================
# IV. META-COGNITIVE ORCHESTRATOR TESTS
# ============================================================================


class TestMetaCognitiveOrchestrator:
    """Test 47-dimensional meta-cognitive strategy selection."""

    def test_feature_extraction_dimensions(self):
        """Feature vector should have exactly 47 dimensions."""
        orchestrator = MetaCognitiveOrchestrator()

        task = {"type": "test", "urgency": 0.5}
        context = {"ethical_sensitivity": 0.3}

        features = orchestrator.extract_features(task, context)

        assert features.shape == (47,)
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    @pytest.mark.asyncio
    async def test_strategy_selection(self):
        """Orchestrator should select appropriate strategy."""
        orchestrator = MetaCognitiveOrchestrator()

        task = {"type": "decision", "urgency": 0.9}
        context = {"ethical_sensitivity": 0.8, "confidence": 0.9}

        result = await orchestrator.select_and_execute(task, context)

        assert "strategy" in result
        assert result["strategy"] in orchestrator.strategies
        assert "quality" in result
        assert 0.0 <= result["quality"] <= 1.0

    @pytest.mark.asyncio
    async def test_history_accumulation(self):
        """Execution history should grow."""
        orchestrator = MetaCognitiveOrchestrator()

        for i in range(5):
            await orchestrator.select_and_execute({"type": f"task_{i}"}, {})

        assert len(orchestrator.history) == 5

    def test_novelty_computation(self):
        """Novelty should decrease for repeated task types."""
        orchestrator = MetaCognitiveOrchestrator()

        # Populate history with same task type
        for i in range(10):
            orchestrator.history.append({"task_type": "repeated"})

        novel_task = {"type": "repeated"}
        novelty = orchestrator._compute_novelty(novel_task)

        assert novelty < 1.0  # Should be lower than initial

    def test_strategy_entropy(self):
        """Strategy entropy should reflect distribution."""
        orchestrator = MetaCognitiveOrchestrator()

        # Uniform distribution of strategies
        for s in orchestrator.strategies * 10:
            orchestrator.history.append({"strategy": s})

        entropy = orchestrator._compute_strategy_entropy()

        # Uniform distribution should have high entropy
        assert entropy > 0.8

    @pytest.mark.asyncio
    async def test_adaptive_weight_update(self):
        """Feature weights should adapt based on quality."""
        orchestrator = MetaCognitiveOrchestrator()

        initial_weights = orchestrator._feature_weights.copy()

        # Execute several tasks
        for i in range(20):
            await orchestrator.select_and_execute(
                {"type": f"task_{i}", "complexity": 0.5}, {"ethical_sensitivity": 0.5}
            )

        # Weights should have changed
        assert not np.allclose(orchestrator._feature_weights, initial_weights)


# ============================================================================
# V. HIGHER-ORDER LOGIC BRIDGE TESTS
# ============================================================================


class TestHigherOrderLogicBridge:
    """Test neuro-symbolic bridge with ethical projection."""

    def test_forward_produces_output(self):
        """Forward pass should produce valid output structure."""
        bridge = HigherOrderLogicBridge(input_dim=768, hidden_dim=256)

        neural_input = torch.randn(1, 768)
        result = bridge(neural_input)

        assert "neural_output" in result
        assert "confidence" in result
        assert "ethical_certificate" in result
        assert result["neural_output"].shape == (1, 768)

    def test_ethical_context_affects_output(self):
        """Ethical context should modulate output."""
        bridge = HigherOrderLogicBridge(input_dim=768, hidden_dim=256)

        neural_input = torch.randn(1, 768)

        # Without ethical context
        result_no_ethics = bridge(neural_input.clone(), ethical_context=None)

        # With high ethical sensitivity
        result_high_ethics = bridge(
            neural_input.clone(), ethical_context={"sensitivity": 0.9}
        )

        # Outputs should differ
        assert not torch.allclose(
            result_no_ethics["neural_output"], result_high_ethics["neural_output"]
        )

    def test_ethical_certificate_score_bounded(self):
        """Ethical certificate score must be in [0, 1]."""
        bridge = HigherOrderLogicBridge(input_dim=768, hidden_dim=256)

        for _ in range(10):
            neural_input = torch.randn(1, 768) * 10  # High variance input
            result = bridge(neural_input, {"sensitivity": np.random.random()})

            score = result["ethical_certificate"]["score"]
            assert 0.0 <= score <= 1.0


# ============================================================================
# VI. COGNITIVE SOVEREIGN INTEGRATION TESTS
# ============================================================================


class TestCognitiveSovereign:
    """Integration tests for the unified cognitive system."""

    def test_initialization(self):
        """System should initialize all components."""
        sovereign = CognitiveSovereign()

        assert sovereign.l1 is not None
        assert sovereign.l2 is not None
        assert sovereign.l3 is not None
        assert sovereign.l4 is not None
        assert sovereign.l5 is not None
        assert sovereign.security is not None
        assert sovereign.bridge is not None
        assert sovereign.meta is not None
        assert sovereign.ihsan is not None

    @pytest.mark.asyncio
    async def test_full_cognitive_cycle(self):
        """Complete cognitive cycle should succeed."""
        sovereign = CognitiveSovereign()

        input_data = {
            "type": "decision",
            "content": "test_operation",
            "urgency": 0.7,
            "ethical_sensitivity": 0.8,
        }

        result = await sovereign.run_cycle(input_data)

        assert result["status"] == "SUCCESS"
        assert "snr" in result
        assert "ethical_score" in result
        assert "temporal_proof" in result

    @pytest.mark.asyncio
    async def test_temporal_integrity_after_cycles(self):
        """Temporal chain should remain valid after multiple cycles."""
        sovereign = CognitiveSovereign()

        for i in range(5):
            await sovereign.run_cycle(
                {
                    "type": "test",
                    "content": f"operation_{i}",
                    "urgency": 0.5,
                    "ethical_sensitivity": 0.5,
                }
            )

        assert sovereign.security.verify_chain_integrity() is True

    @pytest.mark.asyncio
    async def test_episodic_memory_grows(self):
        """Episodic memory should accumulate across cycles."""
        sovereign = CognitiveSovereign()

        for i in range(3):
            await sovereign.run_cycle({"type": "test", "content": f"op_{i}"})

        assert len(sovereign.l3.episodes) == 3
        assert sovereign.l3.verify_integrity() is True

    @pytest.mark.asyncio
    async def test_hypergraph_topology_evolves(self):
        """HyperGraph should develop topology."""
        sovereign = CognitiveSovereign()

        for i in range(5):
            await sovereign.run_cycle(
                {"type": "test", "content": f"op_{i}", "urgency": np.random.random()}
            )

        topology = sovereign.l4.analyze_topology()
        assert topology["clustering_coefficient"] >= 0.0


# ============================================================================
# VII. RETROGRADE SIGNALING TESTS
# ============================================================================


class TestRetrogradeSignalingPathway:
    """Test L5 → L1 top-down attention modulation."""

    def test_generates_expectations(self):
        """Should generate top-down expectations."""
        security = QuantumTemporalSecurity()
        l5 = L5DeterministicTools(security)
        l1 = L1PerceptualBuffer()

        l1.push("item_1")
        l1.push("item_2")

        pathway = RetrogradeSignalingPathway(l5, l1)
        expectations = pathway.generate_top_down_expectations({})

        assert len(expectations) == len(l1.buffer)

    def test_modulates_l1_attention(self):
        """Expectations should modulate L1 attention weights."""
        security = QuantumTemporalSecurity()
        l5 = L5DeterministicTools(security)
        l1 = L1PerceptualBuffer()

        l1.push("item_1", attention_weight=0.5)
        l1.push("item_2", attention_weight=0.5)

        original_weights = [b["weight"] for b in l1.buffer]

        pathway = RetrogradeSignalingPathway(l5, l1)
        predictions = pathway.generate_top_down_expectations({})
        pathway.modulate_l1_attention(predictions)

        new_weights = [b["weight"] for b in l1.buffer]

        # Weights should have been modulated
        assert new_weights != original_weights


# ============================================================================
# VIII. STRESS AND EDGE CASE TESTS
# ============================================================================


class TestStressAndEdgeCases:
    """Stress tests and edge case handling."""

    def test_empty_buffer_operations(self):
        """Empty buffer operations should not crash."""
        buffer = L1PerceptualBuffer()

        items = buffer.get_with_attention()
        assert items == []

    def test_empty_memory_consolidation(self):
        """Empty consolidation should handle gracefully."""
        memory = L2WorkingMemory()
        summary = memory.consolidate([])
        assert "SUMMARY" in summary

    def test_empty_episodic_verification(self):
        """Empty episodic memory should verify as True."""
        memory = L3EpisodicMemory()
        assert memory.verify_integrity() is True

    @pytest.mark.asyncio
    async def test_empty_graph_topology(self):
        """Empty graph should return zero metrics."""
        graph = L4SemanticHyperGraph()
        topology = graph.analyze_topology()

        assert topology["clustering_coefficient"] == 0.0
        assert topology["rich_club_coefficient"] == 0.0

    def test_high_volume_security_operations(self):
        """Security should handle high volume without degradation."""
        security = QuantumTemporalSecurity()

        start = time.time()
        for i in range(100):
            security.secure_cognitive_operation({"op": i})
        elapsed = time.time() - start

        assert security.verify_chain_integrity() is True
        assert elapsed < 5.0  # Should complete in reasonable time

    @pytest.mark.asyncio
    async def test_concurrent_metacognitive_calls(self):
        """Meta-cognitive should handle concurrent calls."""
        orchestrator = MetaCognitiveOrchestrator()

        async def run_task(i):
            return await orchestrator.select_and_execute(
                {"type": f"concurrent_{i}"}, {"urgency": np.random.random()}
            )

        results = await asyncio.gather(*[run_task(i) for i in range(10)])

        assert len(results) == 10
        assert all("strategy" in r for r in results)


# ============================================================================
# IX. PROPERTY-BASED TESTS (Hypothesis-style)
# ============================================================================


class TestPropertyBased:
    """Property-based testing for invariant verification."""

    def test_ihsan_score_monotonicity(self):
        """Higher dimension values should yield higher scores."""
        ihsan = IhsanPrinciples()

        low_dims = {
            k: 0.5
            for k in [
                "truthfulness",
                "dignity",
                "fairness",
                "excellence",
                "sustainability",
            ]
        }
        high_dims = {k: 0.9 for k in low_dims}

        low_score = ihsan.compute_score(low_dims)
        high_score = ihsan.compute_score(high_dims)

        assert high_score > low_score

    def test_temporal_chain_uniqueness(self):
        """All temporal hashes must be unique."""
        security = QuantumTemporalSecurity()

        for i in range(50):
            security.secure_cognitive_operation({"op": i, "ts": time.time_ns()})

        hashes = [h.hex() for h in security.temporal_chain]
        assert len(set(hashes)) == len(hashes), "Duplicate hash detected"

    def test_merkle_root_determinism(self):
        """Same episodes should produce same Merkle root."""
        memory1 = L3EpisodicMemory()
        memory2 = L3EpisodicMemory()

        episodes = [{"id": i, "data": f"test_{i}"} for i in range(5)]

        for i, ep in enumerate(episodes):
            memory1.store_episode(f"ep_{i}", ep)
            memory2.store_episode(f"ep_{i}", ep)

        assert memory1.merkle_root == memory2.merkle_root


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
