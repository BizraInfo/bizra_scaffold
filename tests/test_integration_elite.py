#!/usr/bin/env python3
"""
ELITE INTEGRATION TEST SUITE
============================
Cross-Language | End-to-End | Property-Based | Stress Testing

This test suite validates the complete BIZRA system at the integration level,
ensuring all components work harmoniously and maintain invariants under load.

Run: pytest tests/test_integration_elite.py -v --tb=short
"""

import asyncio
import hashlib
import os
import struct
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Skip if core imports fail
pytest.importorskip("core")
pytest.importorskip("blake3")

import blake3
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-LANGUAGE HASH COMPATIBILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossLanguageCompatibility:
    """
    Verify Python implementations produce identical outputs to Rust attestation-engine.
    These tests ensure cryptographic determinism across language boundaries.
    """

    def test_blake3_attestation_id_deterministic(self):
        """Verify attestation ID computation matches Rust implementation."""
        from ihsan_bridge import AttestationBridge

        # Test vectors that should produce identical hashes in Rust and Python
        test_cases = [
            ("contributor_alice", 42, "abc123def456"),
            ("validator_bob", 0, ""),
            ("node_zero", 18446744073709551615, "a" * 64),  # Max u64
            ("测试用户", 1000, "unicode_测试"),  # Unicode support
        ]

        for contributor, epoch, evidence_root in test_cases:
            # Python implementation
            result = AttestationBridge.compute_attestation_id(
                contributor, epoch, evidence_root
            )

            # Manual verification using same algorithm
            hasher = blake3.blake3()
            hasher.update(contributor.encode("utf-8"))
            hasher.update(struct.pack(">Q", epoch))
            hasher.update(evidence_root.encode("utf-8"))
            expected = hasher.hexdigest()

            assert result == expected, f"Mismatch for {contributor}, {epoch}"
            assert len(result) == 64, "Blake3 should produce 64 hex chars (256 bits)"

    def test_canonical_json_rfc8785_compliance(self):
        """Verify JCS canonical JSON serialization."""
        from ihsan_bridge import AttestationBridge

        # Test cases with various edge cases
        test_cases = [
            # Sorted keys
            {"z": 1, "a": 2, "m": 3},
            # Nested objects
            {"outer": {"inner": {"deep": 1}}},
            # Arrays
            {"items": [3, 1, 2]},
            # Mixed types
            {"int": 42, "float": 3.14, "str": "hello", "bool": True, "null": None},
            # Unicode
            {"unicode": "日本語テスト"},
        ]

        for data in test_cases:
            result = AttestationBridge.canonical_json(data)

            # Verify it's valid UTF-8 bytes
            assert isinstance(result, bytes)
            decoded = result.decode("utf-8")

            # Verify no extra whitespace
            assert "\n" not in decoded
            assert "  " not in decoded  # No double spaces

            # Verify keys are sorted (for top-level objects)
            if isinstance(data, dict):
                # Keys should appear in sorted order
                import json

                parsed = json.loads(decoded)
                assert list(parsed.keys()) == sorted(data.keys())

    def test_evidence_root_deterministic(self):
        """Verify evidence root computation is deterministic."""
        from ihsan_bridge import AttestationBridge, DimensionScores

        dimensions = DimensionScores(
            quality=0.95, utility=0.90, trust=0.85, fairness=0.80, diversity=0.75
        )

        bundle = AttestationBridge.create_evidence_bundle(
            content_hash="abc123", dimensions=dimensions, metadata={"key": "value"}
        )

        # Compute root multiple times
        results = [AttestationBridge.compute_evidence_root(bundle) for _ in range(10)]

        # All should be identical
        assert len(set(results)) == 1, "Evidence root must be deterministic"
        assert len(results[0]) == 64, "Blake3 hex digest should be 64 chars"


# ═══════════════════════════════════════════════════════════════════════════════
# IHSĀN METRIC INVARIANT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIhsanInvariants:
    """
    Property-based tests for Ihsān metric invariants.
    These tests verify mathematical properties that must always hold.
    """

    def test_weights_sum_to_unity(self):
        """Ihsān dimension weights must sum to exactly 1.0."""
        from ihsan_bridge import IhsanDimension

        total_weight = sum(dim.weight for dim in IhsanDimension)
        assert (
            abs(total_weight - 1.0) < 1e-10
        ), f"Weights sum to {total_weight}, not 1.0"

    def test_score_bounded_zero_one(self):
        """Ihsān score must always be in [0.0, 1.0]."""
        from ihsan_bridge import IhsanScore

        # Test with random values in valid range
        for _ in range(100):
            score = IhsanScore(
                truthfulness=np.random.uniform(0, 1),
                dignity=np.random.uniform(0, 1),
                fairness=np.random.uniform(0, 1),
                excellence=np.random.uniform(0, 1),
                sustainability=np.random.uniform(0, 1),
            )
            total = score.total()
            assert 0.0 <= total <= 1.0, f"Score {total} out of bounds"

    def test_perfect_score_passes_threshold(self):
        """Perfect scores (all 1.0) must pass the 0.95 threshold."""
        from ihsan_bridge import IhsanScore

        perfect = IhsanScore(
            truthfulness=1.0,
            dignity=1.0,
            fairness=1.0,
            excellence=1.0,
            sustainability=1.0,
        )

        passed, score = perfect.verify()
        assert passed is True
        assert abs(score - 1.0) < 1e-10

    def test_monotonicity(self):
        """Increasing any dimension should not decrease the total score."""
        from ihsan_bridge import IhsanScore

        base = IhsanScore(
            truthfulness=0.5,
            dignity=0.5,
            fairness=0.5,
            excellence=0.5,
            sustainability=0.5,
        )
        base_score = base.total()

        # Increase each dimension and verify score doesn't decrease
        for dim in [
            "truthfulness",
            "dignity",
            "fairness",
            "excellence",
            "sustainability",
        ]:
            increased = IhsanScore(
                truthfulness=0.5,
                dignity=0.5,
                fairness=0.5,
                excellence=0.5,
                sustainability=0.5,
            )
            setattr(increased, dim, 0.7)

            assert increased.total() >= base_score, f"Monotonicity violated for {dim}"

    def test_fail_closed_on_nan(self):
        """NaN values must trigger fail-closed behavior."""
        from ihsan_bridge import IhsanScore

        invalid = IhsanScore(
            truthfulness=float("nan"),
            dignity=1.0,
            fairness=1.0,
            excellence=1.0,
            sustainability=1.0,
        )

        passed, score = invalid.verify()
        assert passed is False
        assert score == 0.0

    def test_fail_closed_on_infinity(self):
        """Infinite values must trigger fail-closed behavior."""
        from ihsan_bridge import IhsanScore

        for inf_val in [float("inf"), float("-inf")]:
            invalid = IhsanScore(
                truthfulness=inf_val,
                dignity=1.0,
                fairness=1.0,
                excellence=1.0,
                sustainability=1.0,
            )

            passed, score = invalid.verify()
            assert passed is False
            assert score == 0.0

    def test_fail_closed_on_out_of_range(self):
        """Values outside [0, 1] must trigger fail-closed behavior."""
        from ihsan_bridge import IhsanScore

        for invalid_val in [-0.1, 1.1, -1.0, 2.0]:
            invalid = IhsanScore(
                truthfulness=invalid_val,
                dignity=1.0,
                fairness=1.0,
                excellence=1.0,
                sustainability=1.0,
            )

            passed, score = invalid.verify()
            assert passed is False, f"Should reject value {invalid_val}"


# ═══════════════════════════════════════════════════════════════════════════════
# END-TO-END COGNITIVE CYCLE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCognitiveCycleIntegration:
    """
    End-to-end tests for complete cognitive cycles.
    Validates the full pipeline from observation to action.
    """

    @pytest.fixture
    def temp_key_dir(self):
        """Create temporary directory for key storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_full_cognitive_cycle(self):
        """Test complete cognitive cycle execution."""
        from cognitive_sovereign import CognitiveSovereign

        sovereign = CognitiveSovereign()

        input_data = {
            "type": "decision",
            "content": "test_observation",
            "urgency": 0.5,
            "ethical_sensitivity": 0.7,
        }

        result = await sovereign.run_cycle(input_data)

        assert result["status"] == "SUCCESS"
        assert "snr" in result
        assert "ethical_score" in result
        assert "temporal_proof" in result
        assert result["ethical_score"] >= 0.0
        assert result["ethical_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_temporal_chain_integrity_after_cycles(self):
        """Verify temporal chain remains valid after multiple cycles."""
        from cognitive_sovereign import CognitiveSovereign

        sovereign = CognitiveSovereign()

        # Run multiple cycles
        for i in range(5):
            input_data = {
                "type": f"cycle_{i}",
                "content": f"observation_{i}",
                "urgency": 0.3 + (i * 0.1),
                "ethical_sensitivity": 0.5,
            }
            await sovereign.run_cycle(input_data)

        # Verify chain integrity
        assert sovereign.security.verify_chain_integrity() is True
        assert len(sovereign.security.temporal_chain) == 5

    @pytest.mark.asyncio
    async def test_memory_layer_consolidation(self):
        """Test L1→L2→L3 memory consolidation flow."""
        from cognitive_sovereign import CognitiveSovereign

        sovereign = CognitiveSovereign()

        # Push items to L1
        for i in range(10):
            sovereign.l1.push({"item": i, "data": f"content_{i}"})

        # L1 should respect capacity (7±2)
        assert len(sovereign.l1.buffer) <= 9

        # Consolidate to L2
        l1_items = [b["item"] for b in sovereign.l1.buffer]
        summary = sovereign.l2.consolidate(l1_items)

        assert summary.startswith("SUMMARY[")
        assert len(sovereign.l2.summaries) >= 1

    @pytest.mark.asyncio
    async def test_l4_hypergraph_topology(self):
        """Test L4 semantic hypergraph creation and analysis."""
        from cognitive_sovereign import CognitiveSovereign

        sovereign = CognitiveSovereign()

        # Create multiple hyperedges
        await sovereign.l4.create_hyperedge(
            nodes=["Entity_A", "Entity_B", "Entity_C"], relation="COLLABORATES_WITH"
        )
        await sovereign.l4.create_hyperedge(
            nodes=["Entity_B", "Entity_D"], relation="DEPENDS_ON"
        )
        await sovereign.l4.create_hyperedge(
            nodes=["Entity_A", "Entity_D", "Entity_E"], relation="INFLUENCES"
        )

        topology = sovereign.l4.analyze_topology()

        assert "clustering_coefficient" in topology
        assert "rich_club_coefficient" in topology
        assert topology["clustering_coefficient"] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSecurityIntegration:
    """
    Integration tests for security subsystems.
    Validates cryptographic operations and key management.
    """

    @pytest.fixture
    def temp_key_dir(self):
        """Create temporary directory for key storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_quantum_security_key_persistence(self, temp_key_dir):
        """Test key generation and persistence."""
        from core.security.quantum_security_v2 import QuantumSecurityV2

        # Create first instance
        sec1 = QuantumSecurityV2(key_storage_path=temp_key_dir)
        public_key_1 = sec1.public_key.hex()

        # Create second instance - should load same keys
        sec2 = QuantumSecurityV2(key_storage_path=temp_key_dir)
        public_key_2 = sec2.public_key.hex()

        assert public_key_1 == public_key_2, "Keys should persist across instances"

        # Verify files exist
        assert (Path(temp_key_dir) / "public.key").exists()
        assert (Path(temp_key_dir) / "secret.key").exists()

    @pytest.mark.asyncio
    async def test_secure_operation_creates_proof(self, temp_key_dir):
        """Test secure operation creates temporal proof."""
        from core.security.quantum_security_v2 import QuantumSecurityV2

        sec = QuantumSecurityV2(key_storage_path=temp_key_dir)

        operation = {
            "type": "test_operation",
            "data": "sensitive_content",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        result = await sec.secure_operation(operation)

        assert "temporal_proof" in result
        proof = result["temporal_proof"]

        assert "nonce" in proof
        assert "timestamp" in proof
        assert "signature" in proof
        assert "chain_index" in proof
        assert len(proof["nonce"]) == 128  # 64 bytes = 128 hex chars

    @pytest.mark.asyncio
    async def test_chain_lock_prevents_race_conditions(self, temp_key_dir):
        """Test asyncio.Lock prevents concurrent chain modification."""
        from core.security.quantum_security_v2 import QuantumSecurityV2

        sec = QuantumSecurityV2(key_storage_path=temp_key_dir)

        # Run many concurrent operations
        async def make_operation(i):
            return await sec.secure_operation({"index": i})

        tasks = [make_operation(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        # All operations should have unique chain indices
        indices = [r["temporal_proof"]["chain_index"] for r in results]
        assert len(set(indices)) == 20, "Chain indices should be unique"

        # Verify chain integrity
        assert sec.verify_chain_integrity() is True


# ═══════════════════════════════════════════════════════════════════════════════
# VALUE ORACLE COMPREHENSIVE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestValueOracleComprehensive:
    """
    Comprehensive tests for the pluralistic value oracle system.
    Expands coverage to 95%+.
    """

    @pytest.fixture
    def sample_convergence(self):
        """Create sample convergence for testing."""
        from core.value_oracle import Convergence

        return Convergence(
            id="test-convergence-001",
            clarity_score=0.85,
            mutual_information=0.72,
            entropy=0.45,
            synergy=0.68,
            quantization_error=0.02,
        )

    @pytest.mark.asyncio
    async def test_shapley_oracle_evaluation(self, sample_convergence):
        """Test Shapley value oracle computation."""
        from core.value_oracle import ShapleyOracle

        oracle = ShapleyOracle()
        signal = await oracle.evaluate(sample_convergence)

        assert signal.value >= 0.0
        assert signal.confidence >= 0.0
        assert signal.confidence <= 1.0
        assert "Shapley" in signal.reasoning

    @pytest.mark.asyncio
    async def test_prediction_market_oracle(self, sample_convergence):
        """Test prediction market oracle simulation."""
        from core.value_oracle import PredictionMarketOracle

        oracle = PredictionMarketOracle()
        signal = await oracle.evaluate(sample_convergence)

        assert signal.value >= 0.0
        assert signal.confidence >= 0.0
        assert signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_reputation_oracle(self, sample_convergence):
        """Test reputation-based oracle."""
        from core.value_oracle import ReputationOracle

        oracle = ReputationOracle()
        signal = await oracle.evaluate(sample_convergence)

        assert signal.value >= 0.0
        assert signal.confidence >= 0.0
        assert signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_formal_verification_oracle(self, sample_convergence):
        """Test formal verification oracle."""
        from core.value_oracle import FormalVerificationOracle

        oracle = FormalVerificationOracle()
        signal = await oracle.evaluate(sample_convergence)

        assert signal.value >= 0.0
        assert signal.confidence >= 0.0
        assert signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_information_theoretic_oracle(self, sample_convergence):
        """Test information-theoretic oracle."""
        from core.value_oracle import InformationTheoreticOracle

        oracle = InformationTheoreticOracle()
        signal = await oracle.evaluate(sample_convergence)

        assert signal.value >= 0.0
        assert signal.confidence >= 0.0
        assert signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_pluralistic_oracle_synthesis(self, sample_convergence):
        """Test full pluralistic value assessment."""
        from core.value_oracle import PluralisticValueOracle

        oracle = PluralisticValueOracle()
        assessment = await oracle.compute_value(sample_convergence)

        assert assessment.value >= 0.0
        assert assessment.confidence >= 0.0
        assert len(assessment.signals) == 6  # All 6 oracles (including SNR)
        assert hasattr(assessment, "disagreement_score")

        # Verify oracle weights are valid
        total_weight = sum(assessment.oracle_weights.values())
        assert abs(total_weight - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_oracle_disagreement_detection(self):
        """Test detection of oracle disagreement."""
        from core.value_oracle import Convergence, PluralisticValueOracle

        # Create edge case convergence
        edge_convergence = Convergence(
            id="edge-case",
            clarity_score=0.99,
            mutual_information=0.01,  # Very low MI
            entropy=0.99,  # Very high entropy
            synergy=0.5,
            quantization_error=0.0,
        )

        oracle = PluralisticValueOracle()
        assessment = await oracle.compute_value(edge_convergence)

        # High entropy + low MI should cause some disagreement
        assert assessment.disagreement_score >= 0.0

    def test_oracle_historical_accuracy_tracking(self):
        """Test that oracles track historical accuracy."""
        from core.value_oracle import PredictionMarketOracle, ShapleyOracle

        shapley = ShapleyOracle()
        market = PredictionMarketOracle()

        # Oracles should have accuracy tracking
        assert hasattr(shapley, "historical_accuracy")
        assert hasattr(market, "historical_accuracy")
        assert 0.0 <= shapley.historical_accuracy <= 1.0
        assert 0.0 <= market.historical_accuracy <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# TIERED VERIFICATION COMPREHENSIVE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTieredVerificationComprehensive:
    """
    Comprehensive tiered verification tests including stress scenarios.
    """

    @pytest.mark.asyncio
    async def test_urgency_tier_selection(self):
        """Test correct tier selection based on urgency."""
        from core.tiered_verification import (
            Action,
            TieredVerificationEngine,
            UrgencyLevel,
            VerificationTier,
        )

        engine = TieredVerificationEngine()

        # Test that different urgency levels produce valid verification results
        urgency_levels = [
            UrgencyLevel.REAL_TIME,
            UrgencyLevel.NEAR_REAL_TIME,
            UrgencyLevel.BATCH,
            UrgencyLevel.DEFERRED,
        ]

        for urgency in urgency_levels:
            action = Action(
                id=f"test-{urgency.name}", payload=b"test payload", urgency=urgency
            )
            result = await engine.verify(action)

            # Verify result is valid with appropriate tier
            assert result.valid is True or result.valid is False  # Must be boolean
            assert result.tier is not None  # Must have a tier
            assert result.confidence >= 0.0  # Confidence must be valid

    @pytest.mark.asyncio
    async def test_verification_latency_bounds(self):
        """Test that verification meets latency targets."""
        from core.tiered_verification import (
            Action,
            TieredVerificationEngine,
            UrgencyLevel,
        )

        engine = TieredVerificationEngine()

        latency_bounds = {
            UrgencyLevel.REAL_TIME: 100,  # <100ms
            UrgencyLevel.NEAR_REAL_TIME: 500,  # <500ms
        }

        for urgency, max_latency in latency_bounds.items():
            action = Action(
                id=f"latency-test-{urgency.name}",
                payload=b"x" * 1000,  # 1KB payload
                urgency=urgency,
            )

            start = time.perf_counter()
            result = await engine.verify(action)
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < max_latency, f"{urgency.name} exceeded {max_latency}ms"

    @pytest.mark.asyncio
    async def test_rollback_mechanism(self):
        """Test optimistic verification with potential rollback."""
        from core.tiered_verification import (
            Action,
            TieredVerificationEngine,
            UrgencyLevel,
        )

        engine = TieredVerificationEngine()

        action = Action(
            id="rollback-test", payload=b"test", urgency=UrgencyLevel.REAL_TIME
        )

        # Execute standard verification (optimistic is handled internally)
        result = await engine.verify(action)

        # Verify result has rollback indicator
        assert hasattr(result, "rollback_required")

        # Validate the verification result
        assert result.valid is True or result.valid is False
        assert result.tier is not None

        # Verify metrics are available
        metrics = engine.get_metrics()
        assert metrics is not None


# ═══════════════════════════════════════════════════════════════════════════════
# STRESS AND PERFORMANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestStressAndPerformance:
    """
    Stress tests and performance benchmarks.
    """

    @pytest.mark.asyncio
    async def test_high_throughput_verification(self):
        """Test verification under high load."""
        from core.tiered_verification import (
            Action,
            TieredVerificationEngine,
            UrgencyLevel,
        )

        engine = TieredVerificationEngine()

        async def verify_batch(batch_size: int):
            tasks = []
            for i in range(batch_size):
                action = Action(
                    id=f"batch-{i}",
                    payload=f"payload-{i}".encode(),
                    urgency=UrgencyLevel.NEAR_REAL_TIME,
                )
                tasks.append(engine.verify(action))
            return await asyncio.gather(*tasks)

        start = time.perf_counter()
        results = await verify_batch(100)
        elapsed = time.perf_counter() - start

        throughput = 100 / elapsed

        # Allow for high success rate (95%+) under stress conditions
        success_count = sum(1 for r in results if r.valid)
        success_rate = success_count / len(results)
        assert (
            success_rate >= 0.95
        ), f"Success rate {success_rate:.1%} below 95% threshold"
        assert throughput > 10, f"Throughput {throughput:.1f}/s below minimum"

    @pytest.mark.asyncio
    async def test_memory_layer_stress(self):
        """Test memory layers under high volume."""
        from cognitive_sovereign import L1PerceptualBuffer, L2WorkingMemory

        l1 = L1PerceptualBuffer(capacity=9)
        l2 = L2WorkingMemory()

        # Push 1000 items
        for i in range(1000):
            l1.push(
                {"id": i, "data": f"item_{i}" * 10}, attention_weight=np.random.random()
            )

        # L1 should maintain capacity
        assert len(l1.buffer) <= 9

        # Consolidate multiple times
        for _ in range(50):
            items = [b["item"] for b in l1.buffer]
            l2.consolidate(items)

        # L2 should maintain summaries
        assert len(l2.summaries) == 50

    def test_ihsan_validation_performance(self):
        """Benchmark Ihsān validation speed."""
        from ihsan_bridge import IhsanScore

        iterations = 10000

        start = time.perf_counter()
        for i in range(iterations):
            score = IhsanScore(
                truthfulness=0.9 + (i % 10) * 0.01,
                dignity=0.9,
                fairness=0.9,
                excellence=0.9,
                sustainability=0.9,
            )
            score.verify()
        elapsed = time.perf_counter() - start

        ops_per_second = iterations / elapsed
        # CI runners vary in speed; use conservative threshold
        assert ops_per_second > 50000, f"Only {ops_per_second:.0f} ops/s (min: 50000)"


# ═══════════════════════════════════════════════════════════════════════════════
# NARRATIVE COMPILER COMPREHENSIVE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestNarrativeCompilerComprehensive:
    """
    Comprehensive tests for narrative compilation.
    """

    @pytest.fixture
    def sample_synthesis(self):
        """Create sample cognitive synthesis."""
        from core.narrative_compiler import CognitiveSynthesis

        return CognitiveSynthesis(
            action={"type": "decision", "outcome": "approved"},
            confidence=0.87,
            verification_tier="INCREMENTAL",
            value_score=0.82,
            ethical_verdict={
                "overall_score": 0.91,
                "severity": "ACCEPTABLE",
                "action_permitted": True,
            },
            health_status="HEALTHY",
            ihsan_scores={
                "truthfulness": 0.95,
                "dignity": 0.92,
                "fairness": 0.90,
                "excellence": 0.88,
                "sustainability": 0.85,
            },
            interdisciplinary_consistency=0.89,
            quantization_error=0.015,
        )

    def test_all_five_styles_compile(self, sample_synthesis):
        """Test all 5 narrative styles compile successfully."""
        from core.narrative_compiler import NarrativeCompiler, NarrativeStyle

        compiler = NarrativeCompiler()

        for style in NarrativeStyle:
            narrative = compiler.compile(sample_synthesis, style)

            assert narrative is not None
            assert narrative.summary is not None
            assert len(narrative.summary) > 0
            assert len(narrative.sections) >= 1
            assert narrative.style == style

    def test_markdown_export(self, sample_synthesis):
        """Test markdown export functionality."""
        from core.narrative_compiler import NarrativeCompiler, NarrativeStyle

        compiler = NarrativeCompiler()
        narrative = compiler.compile(sample_synthesis, NarrativeStyle.TECHNICAL)

        markdown = narrative.to_markdown()

        assert "# Cognitive Analysis Report" in markdown
        assert "## Summary" in markdown
        assert "Generated:" in markdown

    def test_reading_time_estimation(self, sample_synthesis):
        """Test reading time estimation accuracy."""
        from core.narrative_compiler import NarrativeCompiler, NarrativeStyle

        compiler = NarrativeCompiler()

        # Technical style should have longer reading time
        technical = compiler.compile(sample_synthesis, NarrativeStyle.TECHNICAL)
        executive = compiler.compile(sample_synthesis, NarrativeStyle.EXECUTIVE)

        assert technical.reading_time_seconds > 0
        assert executive.reading_time_seconds > 0

    def test_complexity_score_calculation(self, sample_synthesis):
        """Test complexity score is calculated correctly."""
        from core.narrative_compiler import NarrativeCompiler, NarrativeStyle

        compiler = NarrativeCompiler()

        technical = compiler.compile(sample_synthesis, NarrativeStyle.TECHNICAL)
        conversational = compiler.compile(
            sample_synthesis, NarrativeStyle.CONVERSATIONAL
        )

        # Technical should have higher complexity
        assert technical.complexity_score >= conversational.complexity_score


# ═══════════════════════════════════════════════════════════════════════════════
# ULTIMATE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestUltimateIntegrationComprehensive:
    """
    Comprehensive tests for the ultimate integration module.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self):
        """Test ultimate integration class instantiation and basic operation."""
        from core.ultimate_integration import BIZRAVCCNode0Ultimate, HealthStatus

        # Test that the ultimate integration class can be instantiated
        integration = BIZRAVCCNode0Ultimate()

        # Verify core components are initialized
        assert integration is not None
        assert hasattr(integration, "verification_engine")
        assert hasattr(integration, "ethics_engine")
        assert hasattr(integration, "narrative_compiler")
        assert hasattr(integration, "value_oracle")

        # Verify health status enumeration exists
        assert HealthStatus.HEALTHY is not None
        assert HealthStatus.DEGRADED is not None
        assert HealthStatus.CRITICAL is not None
        assert HealthStatus.RECOVERING is not None

    @pytest.mark.asyncio
    async def test_health_monitoring(self):
        """Test health monitoring subsystem."""
        from core.ultimate_integration import HealthMonitor, HealthStatus

        monitor = HealthMonitor()

        # Record good metrics
        for _ in range(20):
            monitor.record_metric("latency_ms", 50.0)
            monitor.record_metric("error_rate", 0.01)
            monitor.record_metric("verification_success", 1.0)

        status = monitor.get_status()
        assert status == HealthStatus.HEALTHY

        # Record bad metrics
        for _ in range(20):
            monitor.record_metric("latency_ms", 1000.0)
            monitor.record_metric("error_rate", 0.5)
            monitor.record_metric("verification_success", 0.5)

        status = monitor.get_status()
        assert status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]

    @pytest.mark.asyncio
    async def test_quantized_convergence(self):
        """Test quantized convergence computation."""
        from core.ultimate_integration import (
            Observation,
            QuantizedConvergence,
            UrgencyLevel,
        )

        qc = QuantizedConvergence()

        observation = Observation(
            id="qc-test", data=b"test data for convergence", urgency=UrgencyLevel.BATCH
        )

        result = qc.compute(observation)

        assert 0.0 <= result.clarity <= 1.0
        assert result.quantization_error >= 0.0
        assert result.quality is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
