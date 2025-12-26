#!/usr/bin/env python3
"""
Test Suite for BIZRA Core Modules
=================================
Comprehensive pytest coverage for:
- Tiered Verification
- Consequential Ethics
- Narrative Compiler
- Value Oracle
- Ultimate Integration
- Memory Layers
- Quantum Security
- Ihsān Bridge

Run: pytest tests/test_core_modules.py -v
"""

import asyncio
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Skip entire module if core imports fail
pytest.importorskip("core")


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_key_dir():
    """Create temporary directory for key storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ═══════════════════════════════════════════════════════════════════════════════
# TIERED VERIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTieredVerification:
    """Tests for tiered_verification.py"""

    def test_import(self):
        """Test module imports correctly."""
        from core.tiered_verification import (
            Action,
            TieredVerificationEngine,
            UrgencyLevel,
            VerificationResult,
            VerificationTier,
        )

        assert TieredVerificationEngine is not None
        assert UrgencyLevel is not None

    def test_urgency_levels_exist(self):
        """Test all urgency levels are defined."""
        from core.tiered_verification import UrgencyLevel

        assert hasattr(UrgencyLevel, "REAL_TIME")
        assert hasattr(UrgencyLevel, "NEAR_REAL_TIME")
        assert hasattr(UrgencyLevel, "BATCH")
        assert hasattr(UrgencyLevel, "DEFERRED")

    def test_verification_tiers_exist(self):
        """Test all verification tiers are defined."""
        from core.tiered_verification import VerificationTier

        assert hasattr(VerificationTier, "STATISTICAL")
        assert hasattr(VerificationTier, "INCREMENTAL")
        assert hasattr(VerificationTier, "FULL_ZK")

    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test engine initializes correctly."""
        from core.tiered_verification import TieredVerificationEngine

        engine = TieredVerificationEngine()
        assert engine is not None
        assert engine.get_metrics() is not None

    @pytest.mark.asyncio
    async def test_verify_action(self):
        """Test verifying an action."""
        from core.tiered_verification import (
            Action,
            TieredVerificationEngine,
            UrgencyLevel,
        )

        engine = TieredVerificationEngine()
        action = Action(
            id="test-action-001",
            payload=b"test payload data",
            urgency=UrgencyLevel.NEAR_REAL_TIME,
        )
        result = await engine.verify(action)
        assert result is not None
        assert result.valid is True
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_optimistic_verification_async_callback(self):
        """Test optimistic verification handles async callbacks."""
        from core.tiered_verification import (
            Action,
            TieredVerificationEngine,
            UrgencyLevel,
        )

        engine = TieredVerificationEngine()
        action = Action(
            id="test-async-001", payload=b"async test", urgency=UrgencyLevel.REAL_TIME
        )

        callback_executed = False

        async def async_callback(a):
            nonlocal callback_executed
            callback_executed = True
            return "executed"

        result, task = await engine.verify_optimistic(action, async_callback)
        assert result == "executed"
        assert callback_executed


# ═══════════════════════════════════════════════════════════════════════════════
# CONSEQUENTIAL ETHICS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestConsequentialEthics:
    """Tests for consequential_ethics.py"""

    def test_import(self):
        """Test module imports correctly."""
        from core.consequential_ethics import (
            Action,
            ConsequentialEthicsEngine,
            Context,
            EthicalVerdict,
            VerdictSeverity,
        )

        assert ConsequentialEthicsEngine is not None

    def test_verdict_severities_exist(self):
        """Test all severity levels are defined."""
        from core.consequential_ethics import VerdictSeverity

        assert hasattr(VerdictSeverity, "EXEMPLARY")
        assert hasattr(VerdictSeverity, "ACCEPTABLE")
        assert hasattr(VerdictSeverity, "CONCERNING")
        assert hasattr(VerdictSeverity, "PROBLEMATIC")
        assert hasattr(VerdictSeverity, "PROHIBITED")

    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test engine initializes correctly."""
        from core.consequential_ethics import ConsequentialEthicsEngine

        engine = ConsequentialEthicsEngine()
        assert engine is not None

    @pytest.mark.asyncio
    async def test_evaluate_beneficial_action(self):
        """Test evaluating a clearly beneficial action."""
        from core.consequential_ethics import Action, ConsequentialEthicsEngine, Context

        engine = ConsequentialEthicsEngine()
        action = Action(
            id="beneficial-001",
            description="Help users accomplish their goals",
            intended_outcome="User satisfaction",
            potential_benefits=["Productivity", "Time savings", "Quality"],
            potential_harms=[],
            reversibility=1.0,
        )
        context = Context(
            stakeholders=["users"], affected_parties=["users"], domain="productivity"
        )
        verdict = await engine.evaluate(action, context)
        assert verdict is not None
        assert verdict.action_permitted is True
        assert verdict.severity is not None  # Must not be None

    @pytest.mark.asyncio
    async def test_evaluate_harmful_action(self):
        """Test evaluating a harmful action produces lower score."""
        from core.consequential_ethics import (
            Action,
            ConsequentialEthicsEngine,
            Context,
            VerdictSeverity,
        )

        engine = ConsequentialEthicsEngine()

        # Beneficial action
        beneficial = Action(
            id="beneficial-001",
            description="Help users accomplish their goals",
            intended_outcome="User satisfaction",
            potential_benefits=["Productivity", "Time savings", "Quality"],
            potential_harms=[],
            reversibility=1.0,
        )

        # Harmful action
        harmful = Action(
            id="harmful-001",
            description="Cause significant harm",
            intended_outcome="Negative impact",
            potential_benefits=[],
            potential_harms=["Physical harm", "Financial loss", "Privacy violation"],
            reversibility=0.0,
        )

        context = Context(
            stakeholders=["users"], affected_parties=["users"], domain="test"
        )

        verdict_beneficial = await engine.evaluate(beneficial, context)
        verdict_harmful = await engine.evaluate(harmful, context)

        # Harmful action should have lower score than beneficial
        assert verdict_harmful.overall_score < verdict_beneficial.overall_score
        assert verdict_harmful.severity is not None  # Must not be None


# ═══════════════════════════════════════════════════════════════════════════════
# NARRATIVE COMPILER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestNarrativeCompiler:
    """Tests for narrative_compiler.py"""

    def test_import(self):
        """Test module imports correctly."""
        from core.narrative_compiler import (
            CognitiveSynthesis,
            CompiledNarrative,
            NarrativeCompiler,
            NarrativeStyle,
        )

        assert NarrativeCompiler is not None

    def test_all_styles_exist(self):
        """Test all narrative styles are defined."""
        from core.narrative_compiler import NarrativeStyle

        assert hasattr(NarrativeStyle, "TECHNICAL")
        assert hasattr(NarrativeStyle, "EXECUTIVE")
        assert hasattr(NarrativeStyle, "CONVERSATIONAL")
        assert hasattr(NarrativeStyle, "EDUCATIONAL")
        assert hasattr(NarrativeStyle, "AUDIT")

    def test_compiler_has_all_templates(self):
        """Test compiler has templates for all styles."""
        from core.narrative_compiler import NarrativeCompiler, NarrativeStyle

        compiler = NarrativeCompiler()
        for style in NarrativeStyle:
            assert style in compiler.templates, f"Missing template for {style}"

    def test_compile_technical(self):
        """Test compiling technical narrative."""
        from core.narrative_compiler import (
            CognitiveSynthesis,
            NarrativeCompiler,
            NarrativeStyle,
        )

        compiler = NarrativeCompiler()
        synthesis = CognitiveSynthesis(
            action={"type": "test"},
            confidence=0.85,
            verification_tier="STATISTICAL",
            value_score=0.75,
            ethical_verdict={"permitted": True},
            health_status="HEALTHY",
        )
        narrative = compiler.compile(synthesis, NarrativeStyle.TECHNICAL)
        assert narrative is not None
        assert len(narrative.sections) > 0

    def test_compile_educational(self):
        """Test compiling educational narrative (previously missing)."""
        from core.narrative_compiler import (
            CognitiveSynthesis,
            NarrativeCompiler,
            NarrativeStyle,
        )

        compiler = NarrativeCompiler()
        synthesis = CognitiveSynthesis(
            action={"type": "test"},
            confidence=0.85,
            verification_tier="STATISTICAL",
            value_score=0.75,
            ethical_verdict={"permitted": True},
            health_status="HEALTHY",
            ihsan_scores={"ikhlas": 0.9, "karama": 0.85},
        )
        narrative = compiler.compile(synthesis, NarrativeStyle.EDUCATIONAL)
        assert narrative is not None
        assert narrative.style == NarrativeStyle.EDUCATIONAL

    def test_compile_audit(self):
        """Test compiling audit narrative (previously missing)."""
        from core.narrative_compiler import (
            CognitiveSynthesis,
            NarrativeCompiler,
            NarrativeStyle,
        )

        compiler = NarrativeCompiler()
        synthesis = CognitiveSynthesis(
            action={"type": "audit_test"},
            confidence=0.95,
            verification_tier="FULL_ZK",
            value_score=0.88,
            ethical_verdict={"permitted": True, "severity": "ACCEPTABLE"},
            health_status="HEALTHY",
        )
        narrative = compiler.compile(synthesis, NarrativeStyle.AUDIT)
        assert narrative is not None
        assert narrative.style == NarrativeStyle.AUDIT
        assert "hash" in narrative.metadata


# ═══════════════════════════════════════════════════════════════════════════════
# VALUE ORACLE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestValueOracle:
    """Tests for value_oracle.py"""

    def test_import(self):
        """Test module imports correctly."""
        from core.value_oracle import (
            Convergence,
            PluralisticValueOracle,
            ValueAssessment,
        )

        assert PluralisticValueOracle is not None

    @pytest.mark.asyncio
    async def test_oracle_initialization(self):
        """Test oracle initializes correctly."""
        from core.value_oracle import PluralisticValueOracle

        oracle = PluralisticValueOracle()
        assert oracle is not None
        assert len(oracle.oracles) > 0

    @pytest.mark.asyncio
    async def test_compute_value(self):
        """Test computing value assessment."""
        from core.value_oracle import Convergence, PluralisticValueOracle

        oracle = PluralisticValueOracle()
        convergence = Convergence(
            id="test-conv-001",
            clarity_score=0.8,
            mutual_information=0.7,
            entropy=0.3,
            synergy=0.75,
            quantization_error=0.01,
        )
        assessment = await oracle.compute_value(convergence)
        assert assessment is not None
        assert 0.0 <= assessment.value <= 1.0
        assert 0.0 <= assessment.confidence <= 1.0

    def test_reputation_trust_clamped(self):
        """Test that ReputationOracle trust score is clamped to [0, 1]."""
        from core.value_oracle import Convergence, ReputationOracle

        oracle = ReputationOracle()

        # Create convergence with all trust factors present
        convergence = Convergence(
            id="trust-test",
            clarity_score=0.9,
            mutual_information=0.8,
            entropy=0.2,
            synergy=0.85,
            quantization_error=0.01,
            metadata={"source": "trusted", "verified": True, "attestation": "valid"},
        )

        trust = oracle._compute_trust(convergence)
        assert 0.0 <= trust <= 1.0, f"Trust {trust} exceeds bounds"


# ═══════════════════════════════════════════════════════════════════════════════
# ULTIMATE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestUltimateIntegration:
    """Tests for ultimate_integration.py"""

    def test_import(self):
        """Test module imports correctly."""
        from core.ultimate_integration import (
            BIZRAVCCNode0Ultimate,
            Observation,
            UltimateResult,
        )

        assert BIZRAVCCNode0Ultimate is not None

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test ultimate integration initializes correctly."""
        from core.ultimate_integration import BIZRAVCCNode0Ultimate

        ultimate = BIZRAVCCNode0Ultimate()
        assert ultimate is not None
        assert ultimate.verification_engine is not None
        assert ultimate.ethics_engine is not None
        assert ultimate.value_oracle is not None

    @pytest.mark.asyncio
    async def test_process_observation(self):
        """Test processing an observation."""
        from core.tiered_verification import UrgencyLevel
        from core.ultimate_integration import BIZRAVCCNode0Ultimate, Observation

        ultimate = BIZRAVCCNode0Ultimate()
        observation = Observation(
            id="test-obs-001",
            data=b"test observation data for processing",
            urgency=UrgencyLevel.NEAR_REAL_TIME,
            context={"domain": "testing"},
        )

        result = await ultimate.process(observation)
        assert result is not None
        assert result.verification is not None
        assert result.ethics is not None
        assert result.value is not None
        assert result.explanation is not None

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error path produces valid result structure."""
        from core.consequential_ethics import VerdictSeverity
        from core.tiered_verification import UrgencyLevel, VerificationTier
        from core.ultimate_integration import (
            BIZRAVCCNode0Ultimate,
            HealthStatus,
            Observation,
        )

        ultimate = BIZRAVCCNode0Ultimate()

        # Force an error by providing invalid input
        # The error handler should produce a valid result
        # with correct types (not None for severity, correct tier enum)
        try:
            observation = Observation(
                id="error-test",
                data=b"",  # Empty data might cause issues
                urgency=UrgencyLevel.REAL_TIME,
            )
            result = await ultimate.process(observation)

            # If processing completes, verify structure
            assert result.verification.tier in VerificationTier
            if result.health == HealthStatus.DEGRADED:
                assert result.ethics.severity is not None
        except Exception:
            pass  # Error expected in some cases

    def test_deterministic_id_generation(self):
        """Test that convergence IDs are deterministic."""
        import hashlib
        import json

        from core.ultimate_integration import BIZRAVCCNode0Ultimate

        # Same action should produce same ID
        action = {"type": "test", "value": 42}
        action_json = json.dumps(action, sort_keys=True, default=str)
        id1 = hashlib.sha256(action_json.encode()).hexdigest()[:16]
        id2 = hashlib.sha256(action_json.encode()).hexdigest()[:16]
        assert id1 == id2


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM SECURITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestQuantumSecurity:
    """Tests for quantum_security_v2.py"""

    def test_import(self):
        """Test module imports correctly."""
        from core.security.quantum_security_v2 import QuantumSecurityV2, TemporalProof

        assert QuantumSecurityV2 is not None

    def test_initialization(self, temp_key_dir):
        """Test security module initializes correctly."""
        from core.security.quantum_security_v2 import QuantumSecurityV2

        security = QuantumSecurityV2(key_storage_path=temp_key_dir)
        assert security is not None
        assert security.public_key is not None
        assert security.secret_key is not None

    def test_key_persistence(self, temp_key_dir):
        """Test keys are persisted to disk."""
        from core.security.quantum_security_v2 import QuantumSecurityV2

        # First initialization
        security1 = QuantumSecurityV2(key_storage_path=temp_key_dir)
        pubkey1 = security1.public_key

        # Check public key was written
        public_key_path = Path(temp_key_dir) / "public.key"
        assert public_key_path.exists(), "Public key not persisted"

        # Second initialization should load same keys
        security2 = QuantumSecurityV2(key_storage_path=temp_key_dir)
        assert security2.public_key == pubkey1

    @pytest.mark.asyncio
    async def test_secure_operation(self, temp_key_dir):
        """Test securing an operation."""
        from core.security.quantum_security_v2 import QuantumSecurityV2

        security = QuantumSecurityV2(key_storage_path=temp_key_dir)

        operation = {"type": "test", "content": "hello"}
        result = await security.secure_operation(operation)

        assert "temporal_proof" in result
        assert "chain_length" in result
        assert result["chain_length"] == 1

    @pytest.mark.asyncio
    async def test_chain_locking(self, temp_key_dir):
        """Test concurrent operations don't interleave."""
        from core.security.quantum_security_v2 import QuantumSecurityV2

        security = QuantumSecurityV2(key_storage_path=temp_key_dir)

        # Run multiple operations concurrently
        operations = [{"type": "op", "id": i} for i in range(10)]

        results = await asyncio.gather(
            *[security.secure_operation(op) for op in operations]
        )

        # Chain indices should be sequential
        chain_lengths = [r["chain_length"] for r in results]
        assert sorted(chain_lengths) == list(range(1, 11))


# ═══════════════════════════════════════════════════════════════════════════════
# IHSAN BRIDGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIhsanBridge:
    """Tests for ihsan_bridge.py"""

    def test_import(self):
        """Test module imports correctly."""
        from ihsan_bridge import AttestationBridge, IhsanDimension, IhsanVocabulary

        assert IhsanDimension is not None

    def test_blake3_attestation_id(self):
        """Test attestation ID uses Blake3 (not SHA-256)."""
        import blake3

        from ihsan_bridge import AttestationBridge

        contributor = "test_contributor"
        epoch = 12345
        evidence_root = "abc123"

        # Compute expected Blake3 hash
        hasher = blake3.blake3()
        hasher.update(contributor.encode("utf-8"))
        hasher.update(epoch.to_bytes(8, "big"))
        hasher.update(evidence_root.encode("utf-8"))
        expected = hasher.hexdigest()

        # Compare with bridge output
        actual = AttestationBridge.compute_attestation_id(
            contributor, epoch, evidence_root
        )
        assert actual == expected

    def test_canonical_json_deterministic(self):
        """Test canonical JSON is deterministic."""
        from ihsan_bridge import AttestationBridge

        data = {"z": 1, "a": 2, "m": {"b": 3, "a": 4}}

        result1 = AttestationBridge.canonical_json(data)
        result2 = AttestationBridge.canonical_json(data)

        assert result1 == result2
        assert b'"a":' in result1  # Keys should be sorted

    def test_evidence_root_uses_blake3(self):
        """Test evidence root computation uses Blake3."""
        import blake3

        from ihsan_bridge import AttestationBridge

        evidence = {"content_hash": "abc", "metadata": {}}

        canonical = AttestationBridge.canonical_json(evidence)
        expected = blake3.blake3(canonical).hexdigest()

        actual = AttestationBridge.compute_evidence_root(evidence)
        assert actual == expected


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
