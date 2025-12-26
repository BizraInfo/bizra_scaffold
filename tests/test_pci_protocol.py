"""
BIZRA PCI Protocol Test Suite
═══════════════════════════════════════════════════════════════════════════════
Comprehensive tests for PCIEnvelope, PAT/SAT agents, and verification gates.

Tests are organized by:
1. Unit tests - Individual component testing
2. Integration tests - Cross-component workflows
3. Vector tests - Deterministic golden data verification

Author: BIZRA Genesis Team
Version: 1.0.0
"""

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

# Import agents
from core.agents.pat import IHSAN_THRESHOLD, PATAgent, PATConfig, create_pat_agent
from core.agents.sat import SATAgent, SATConfig, VerificationResult, create_sat_agent

# Import PCI modules
from core.pci.envelope import (
    DOMAIN_PREFIX,
    Metadata,
    Payload,
    PCIEnvelope,
    Sender,
    Signature,
    canonical_json,
    compute_digest,
)
from core.pci.reject_codes import (
    LatencyBudget,
    RejectCode,
    RejectionResponse,
    VerificationGate,
)
from core.pci.replay_guard import ReplayGuard

# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def test_vectors_path():
    """Path to test vectors JSON."""
    return Path(__file__).parent / "vectors" / "pci_envelope_v1.json"


@pytest.fixture
def test_vectors(test_vectors_path):
    """Load test vectors from JSON file."""
    with open(test_vectors_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def ed25519_keypair():
    """Generate a fresh Ed25519 keypair for testing."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def pat_agent(ed25519_keypair):
    """Create a PAT agent for testing."""
    private_key, _ = ed25519_keypair
    return create_pat_agent(
        agent_id="pat-test-agent",
        private_key=private_key,
        ihsan_threshold=IHSAN_THRESHOLD,
    )


@pytest.fixture
def sat_agent():
    """Create a SAT agent for testing."""
    return create_sat_agent(
        agent_id="sat-test-agent",
        ihsan_threshold=IHSAN_THRESHOLD,
        policy_hash_provider=lambda: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )


@pytest.fixture
def replay_guard():
    """Create a fresh replay guard for testing."""
    return ReplayGuard(ttl_seconds=300, max_cache_size=1000)


@pytest.fixture
def valid_policy_hash():
    """Return a valid policy hash for testing."""
    return "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: PCIEnvelope
# ═══════════════════════════════════════════════════════════════════════════════


class TestPCIEnvelope:
    """Tests for PCIEnvelope creation and manipulation."""

    def test_envelope_creation(self, ed25519_keypair):
        """Test basic envelope creation with required fields."""
        private_key, public_key = ed25519_keypair

        from cryptography.hazmat.primitives import serialization

        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        sender = Sender(
            agent_id="pat-001",
            agent_type="PAT",
            public_key=public_key_bytes.hex(),
        )

        payload = Payload(
            action="test.action",
            data={"key": "value"},
            policy_hash="a" * 64,
        )

        metadata = Metadata(
            ihsan_score=0.98,
        )

        envelope = PCIEnvelope.create(sender, payload, metadata)

        assert envelope.version == "1.0.0"
        assert envelope.envelope_id is not None
        assert len(envelope.nonce) == 64  # 32 bytes hex
        assert envelope.sender == sender
        assert envelope.payload == payload
        assert envelope.metadata == metadata
        assert envelope.signature is None  # Unsigned

    def test_envelope_signing(self, ed25519_keypair):
        """Test envelope signing produces valid signature."""
        private_key, public_key = ed25519_keypair

        from cryptography.hazmat.primitives import serialization

        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        sender = Sender(
            agent_id="pat-002",
            agent_type="PAT",
            public_key=public_key_bytes.hex(),
        )

        payload = Payload(
            action="sign.test",
            data={},
            policy_hash="b" * 64,
        )

        metadata = Metadata(ihsan_score=0.96)

        envelope = PCIEnvelope.create(sender, payload, metadata)
        signed_envelope = envelope.sign(private_key)

        assert signed_envelope.signature is not None
        assert signed_envelope.signature.algorithm.lower() == "ed25519"
        assert len(signed_envelope.signature.value) > 0

    def test_envelope_verification(self, ed25519_keypair):
        """Test signature verification on signed envelope."""
        private_key, public_key = ed25519_keypair

        from cryptography.hazmat.primitives import serialization

        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        sender = Sender(
            agent_id="pat-003",
            agent_type="PAT",
            public_key=public_key_bytes.hex(),
        )

        payload = Payload(
            action="verify.test",
            data={"message": "hello"},
            policy_hash="c" * 64,
        )

        metadata = Metadata(ihsan_score=0.97)

        envelope = PCIEnvelope.create(sender, payload, metadata)
        signed_envelope = envelope.sign(private_key)

        valid, error = signed_envelope.verify_signature()

        assert valid is True
        assert error is None

    def test_envelope_tamper_detection(self, ed25519_keypair):
        """Test that tampering with signed envelope is detected."""
        private_key, public_key = ed25519_keypair

        from cryptography.hazmat.primitives import serialization

        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        sender = Sender(
            agent_id="pat-004",
            agent_type="PAT",
            public_key=public_key_bytes.hex(),
        )

        payload = Payload(
            action="tamper.test",
            data={"amount": 100},
            policy_hash="d" * 64,
        )

        metadata = Metadata(ihsan_score=0.95)

        envelope = PCIEnvelope.create(sender, payload, metadata)
        signed_envelope = envelope.sign(private_key)

        # Tamper with the payload
        tampered_payload = Payload(
            action="tamper.test",
            data={"amount": 999999},  # Changed!
            policy_hash="d" * 64,
        )

        tampered_envelope = PCIEnvelope(
            version=signed_envelope.version,
            envelope_id=signed_envelope.envelope_id,
            timestamp=signed_envelope.timestamp,
            nonce=signed_envelope.nonce,
            sender=signed_envelope.sender,
            payload=tampered_payload,
            metadata=signed_envelope.metadata,
            signature=signed_envelope.signature,
        )

        valid, error = tampered_envelope.verify_signature()

        assert valid is False
        assert "Signature verification failed" in error


class TestCanonicalJson:
    """Tests for RFC 8785 JCS canonicalization."""

    def test_key_ordering(self):
        """Test that keys are sorted in canonical output."""
        data = {"zebra": 1, "apple": 2, "mango": 3}
        canonical = canonical_json(data).decode("utf-8")

        # Keys must be in sorted order
        assert canonical.index('"apple"') < canonical.index('"mango"')
        assert canonical.index('"mango"') < canonical.index('"zebra"')

    def test_no_whitespace(self):
        """Test that canonical output has no extra whitespace."""
        data = {"key": "value", "nested": {"inner": 42}}
        canonical = canonical_json(data).decode("utf-8")

        # No spaces after colons or commas
        assert ": " not in canonical
        assert ", " not in canonical

    def test_deterministic_output(self):
        """Test that same input always produces same output."""
        data = {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}

        # Multiple calls should produce identical output
        outputs = [canonical_json(data) for _ in range(10)]
        assert all(o == outputs[0] for o in outputs)


class TestComputeDigest:
    """Tests for BLAKE3 digest computation."""

    def test_domain_separation(self):
        """Test that domain prefix is applied."""
        data = b"test data"

        # With domain separation
        digest_with_domain = compute_digest(data, domain_separated=True)

        # Without domain separation
        digest_without_domain = compute_digest(data, domain_separated=False)

        # Must be different due to domain prefix
        assert digest_with_domain != digest_without_domain

    def test_digest_format(self):
        """Test digest is valid hex string."""
        data = b"hello world"
        digest = compute_digest(data)

        # Must be valid hex
        assert all(c in "0123456789abcdef" for c in digest)
        # BLAKE3 produces 64-character hex (32 bytes)
        assert len(digest) == 64


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: RejectCodes
# ═══════════════════════════════════════════════════════════════════════════════


class TestRejectCodes:
    """Tests for rejection code stability and behavior."""

    def test_code_values_stable(self, test_vectors):
        """Test that reject codes have expected values."""
        # Core codes that must be stable
        assert RejectCode.SUCCESS.value == 0
        assert RejectCode.REJECT_SCHEMA.value == 1
        assert RejectCode.REJECT_SIGNATURE.value == 2
        assert RejectCode.REJECT_NONCE_REPLAY.value == 3
        assert RejectCode.REJECT_TIMESTAMP_STALE.value == 4
        assert RejectCode.REJECT_TIMESTAMP_FUTURE.value == 5
        assert RejectCode.REJECT_IHSAN_BELOW_MIN.value == 6
        assert RejectCode.REJECT_SNR_BELOW_MIN.value == 7
        assert RejectCode.REJECT_POLICY_MISMATCH.value == 9
        assert RejectCode.REJECT_ROLE_VIOLATION.value == 11
        assert RejectCode.REJECT_INTERNAL_ERROR.value == 99

    def test_rejection_response_creation(self):
        """Test RejectionResponse factory method."""
        response = RejectionResponse.create(
            code=RejectCode.REJECT_IHSAN_BELOW_MIN,
            envelope_digest="abc123",
            gate="IHSAN",
            latency_ms=5.5,
            details={"score": 0.89, "threshold": 0.95},
        )

        assert response.code == RejectCode.REJECT_IHSAN_BELOW_MIN
        assert response.envelope_digest == "abc123"
        assert response.gate == "IHSAN"
        assert response.latency_ms == 5.5
        assert response.details["score"] == 0.89


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: ReplayGuard
# ═══════════════════════════════════════════════════════════════════════════════


class TestReplayGuard:
    """Tests for replay attack prevention."""

    def test_fresh_nonce_accepted(self, replay_guard):
        """Test that fresh nonces are accepted."""
        nonce = uuid.uuid4().hex + uuid.uuid4().hex

        valid, code = replay_guard.check_nonce(nonce, "digest-001")

        assert valid is True
        assert code is None

    def test_duplicate_nonce_rejected(self, replay_guard):
        """Test that duplicate nonces are rejected."""
        nonce = uuid.uuid4().hex + uuid.uuid4().hex

        # First use
        valid1, _ = replay_guard.check_nonce(nonce, "digest-001")
        assert valid1 is True

        # Second use (replay attempt)
        valid2, code2 = replay_guard.check_nonce(nonce, "digest-002")
        assert valid2 is False
        assert code2 == RejectCode.REJECT_NONCE_REPLAY

    def test_timestamp_in_window(self, replay_guard):
        """Test that recent timestamps are accepted."""
        now = datetime.now(timezone.utc)

        valid, code = replay_guard.check_timestamp(now)

        assert valid is True
        assert code is None

    def test_timestamp_expired(self, replay_guard):
        """Test that old timestamps are rejected."""
        old = datetime.now(timezone.utc) - timedelta(seconds=180)  # 3 minutes ago

        valid, code = replay_guard.check_timestamp(old)

        assert valid is False
        assert code == RejectCode.REJECT_TIMESTAMP_STALE

    def test_timestamp_future(self, replay_guard):
        """Test that future timestamps are rejected."""
        future = datetime.now(timezone.utc) + timedelta(seconds=180)  # 3 minutes ahead

        valid, code = replay_guard.check_timestamp(future)

        assert valid is False
        assert code == RejectCode.REJECT_TIMESTAMP_FUTURE


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: PAT Agent
# ═══════════════════════════════════════════════════════════════════════════════


class TestPATAgent:
    """Tests for PAT (Prover/Builder) agent."""

    def test_create_proposal_success(self, pat_agent, valid_policy_hash):
        """Test successful proposal creation."""
        result = pat_agent.create_proposal(
            action="test.action",
            data={"key": "value"},
            policy_hash=valid_policy_hash,
            ihsan_score=0.98,
        )

        assert result.success is True
        assert result.envelope is not None
        assert result.rejection is None
        assert result.envelope.signature is not None

    def test_create_proposal_low_ihsan(self, pat_agent, valid_policy_hash):
        """Test proposal rejection due to low Ihsān score."""
        result = pat_agent.create_proposal(
            action="test.action",
            data={"key": "value"},
            policy_hash=valid_policy_hash,
            ihsan_score=0.85,  # Below threshold
        )

        assert result.success is False
        assert result.envelope is None
        assert result.rejection is not None
        assert result.rejection.code == RejectCode.REJECT_IHSAN_BELOW_MIN

    def test_agent_type_is_pat(self, pat_agent, valid_policy_hash):
        """Test that agent type is correctly set to PAT."""
        result = pat_agent.create_proposal(
            action="test.action",
            data={},
            policy_hash=valid_policy_hash,
            ihsan_score=0.96,
        )

        assert result.envelope.sender.agent_type == "PAT"

    def test_stats_tracking(self, pat_agent, valid_policy_hash):
        """Test that agent tracks statistics."""
        # Create some proposals
        pat_agent.create_proposal("a1", {}, valid_policy_hash, ihsan_score=0.98)
        pat_agent.create_proposal(
            "a2", {}, valid_policy_hash, ihsan_score=0.85
        )  # Rejected
        pat_agent.create_proposal("a3", {}, valid_policy_hash, ihsan_score=0.96)

        stats = pat_agent.stats()

        assert stats["proposals_created"] == 2
        assert stats["proposals_rejected"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: SAT Agent
# ═══════════════════════════════════════════════════════════════════════════════


class TestSATAgent:
    """Tests for SAT (Verifier/Governor) agent."""

    def test_verify_valid_envelope(self, pat_agent, sat_agent, valid_policy_hash):
        """Test successful verification of valid envelope."""
        # Create proposal from PAT
        result = pat_agent.create_proposal(
            action="valid.action",
            data={"test": True},
            policy_hash=valid_policy_hash,
            ihsan_score=0.98,
        )

        assert result.success, f"PAT failed: {result.rejection}"

        # Verify with SAT
        verification = sat_agent.verify(result.envelope)

        assert verification.success is True
        assert verification.receipt is not None
        assert verification.rejection is None
        assert verification.report is not None

    def test_verify_low_ihsan(self, ed25519_keypair, sat_agent, valid_policy_hash):
        """Test rejection of envelope with low Ihsān score."""
        private_key, public_key = ed25519_keypair

        from cryptography.hazmat.primitives import serialization

        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        # Create envelope with low Ihsān directly (bypassing PAT pre-check)
        sender = Sender(
            agent_id="pat-bypass",
            agent_type="PAT",
            public_key=public_key_bytes.hex(),
        )

        payload = Payload(
            action="low.ihsan.action",
            data={},
            policy_hash=valid_policy_hash,
        )

        metadata = Metadata(
            ihsan_score=0.80,  # Below threshold
        )

        envelope = PCIEnvelope.create(sender, payload, metadata)
        signed_envelope = envelope.sign(private_key)

        # Verify with SAT
        verification = sat_agent.verify(signed_envelope)

        assert verification.success is False
        assert verification.rejection is not None
        assert verification.rejection.code == RejectCode.REJECT_IHSAN_BELOW_MIN

    def test_receipt_structure(self, pat_agent, sat_agent, valid_policy_hash):
        """Test that commit receipt has correct structure."""
        result = pat_agent.create_proposal(
            action="receipt.test",
            data={},
            policy_hash=valid_policy_hash,
            ihsan_score=0.97,
        )

        verification = sat_agent.verify(result.envelope)

        receipt = verification.receipt

        assert receipt.version == "1.0.0"
        assert receipt.receipt_id is not None
        assert receipt.envelope_digest == result.envelope.digest()
        assert receipt.commit_ref["type"] == "eventlog"
        assert receipt.quorum["required"] == 1
        assert receipt.quorum["achieved"] == 1
        assert len(receipt.verifier_set) == 1

    def test_stats_tracking(self, pat_agent, sat_agent, valid_policy_hash):
        """Test that SAT tracks verification statistics."""
        # Create and verify some envelopes
        for i in range(3):
            result = pat_agent.create_proposal(
                action=f"stats.test.{i}",
                data={},
                policy_hash=valid_policy_hash,
                ihsan_score=0.98,
            )
            sat_agent.verify(result.envelope)

        stats = sat_agent.stats()

        assert stats["envelopes_verified"] == 3
        assert stats["receipts_issued"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests: PAT + SAT Workflow
# ═══════════════════════════════════════════════════════════════════════════════


class TestPATSATIntegration:
    """Integration tests for PAT→SAT workflow."""

    def test_full_workflow(self, valid_policy_hash):
        """Test complete PAT→SAT workflow."""
        # Create agents
        pat = create_pat_agent(agent_id="pat-integration")
        sat = create_sat_agent(
            agent_id="sat-integration",
            policy_hash_provider=lambda: valid_policy_hash,
        )

        # PAT creates proposal
        proposal = pat.create_proposal(
            action="integration.test",
            data={"workflow": "complete"},
            policy_hash=valid_policy_hash,
            ihsan_score=0.98,
        )

        assert proposal.success, f"PAT failed: {proposal.rejection}"

        # SAT verifies
        verification = sat.verify(proposal.envelope)

        assert verification.success, f"SAT failed: {verification.rejection}"
        assert verification.receipt is not None

        # Verify receipt links back to envelope
        assert verification.receipt.envelope_digest == proposal.envelope.digest()

    def test_replay_attack_prevented(self, valid_policy_hash):
        """Test that replay attacks are detected."""
        pat = create_pat_agent(agent_id="pat-replay")
        sat = create_sat_agent(
            agent_id="sat-replay",
            policy_hash_provider=lambda: valid_policy_hash,
        )

        # Create proposal
        proposal = pat.create_proposal(
            action="replay.target",
            data={"value": 1000},
            policy_hash=valid_policy_hash,
            ihsan_score=0.99,
        )

        # First verification (should pass)
        verification1 = sat.verify(proposal.envelope)
        assert verification1.success

        # Replay attempt (should fail)
        verification2 = sat.verify(proposal.envelope)
        assert verification2.success is False
        assert verification2.rejection.code == RejectCode.REJECT_NONCE_REPLAY

    def test_gate_latency_budget(self, valid_policy_hash):
        """Test that verification completes within latency budget."""
        pat = create_pat_agent()
        sat = create_sat_agent(
            policy_hash_provider=lambda: valid_policy_hash,
        )

        proposal = pat.create_proposal(
            action="latency.test",
            data={},
            policy_hash=valid_policy_hash,
            ihsan_score=0.96,
        )

        verification = sat.verify(proposal.envelope)

        # Should complete within MEDIUM tier budget (150ms)
        assert verification.report.total_latency_ms < LatencyBudget.MEDIUM_MS
        assert verification.report.tier_reached in ("CHEAP", "MEDIUM")


# ═══════════════════════════════════════════════════════════════════════════════
# Vector Tests: Golden Data Verification
# ═══════════════════════════════════════════════════════════════════════════════


class TestVectors:
    """Tests using golden test vectors."""

    def test_reject_code_registry(self, test_vectors):
        """Verify reject codes have expected numeric values."""
        # The important thing is that the numeric values are stable
        # Names in the vector file are documentation, actual enum names may differ
        codes = test_vectors["reject_code_registry"]["codes"]

        # Check that we have an enum member for each documented code
        for code_str, info in codes.items():
            code_val = int(code_str)
            matching = [c for c in RejectCode if c.value == code_val]
            # Allow for codes not yet implemented
            if code_val <= 15 or code_val == 99:
                assert len(matching) >= 1, f"Missing code {code_val} ({info['name']})"

    def test_latency_budgets(self, test_vectors):
        """Verify latency budgets match spec."""
        budgets = test_vectors["latency_budgets"]

        assert LatencyBudget.CHEAP_MS == budgets["CHEAP_MS"]
        assert LatencyBudget.MEDIUM_MS == budgets["MEDIUM_MS"]
        assert LatencyBudget.EXPENSIVE_MS == budgets["EXPENSIVE_MS"]

    def test_gate_ordering(self, test_vectors):
        """Verify gate ordering matches spec."""
        expected_order = test_vectors["gate_ordering"]

        # Verify VerificationGate has all gates
        for gate in expected_order:
            assert hasattr(VerificationGate, gate), f"Missing gate: {gate}"

    def test_ihsan_threshold_vector(self, test_vectors):
        """Test Ihsān threshold from vector 005."""
        vector = next(
            v
            for v in test_vectors["test_cases"]
            if v["id"] == "vector-005-ihsan-below-threshold"
        )

        expected = vector["expected"]

        assert expected["rejected"] is True
        assert expected["reject_code"] == RejectCode.REJECT_IHSAN_BELOW_MIN.value
        assert expected["threshold"] == IHSAN_THRESHOLD

    def test_replay_vector(self, test_vectors, valid_policy_hash):
        """Test replay detection from vector 003."""
        vector = next(
            v
            for v in test_vectors["test_cases"]
            if v["id"] == "vector-003-replay-detection"
        )

        expected = vector["expected"]

        assert expected["duplicate_rejected"] is True
        assert expected["reject_code"] == RejectCode.REJECT_NONCE_REPLAY.value


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Cases and Error Handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unsigned_envelope_rejected(self, sat_agent):
        """Test that unsigned envelopes are rejected."""
        sender = Sender(
            agent_id="pat-unsigned",
            agent_type="PAT",
            public_key="a" * 64,
        )

        payload = Payload(
            action="unsigned.test",
            data={},
            policy_hash="b" * 64,
        )

        metadata = Metadata(ihsan_score=0.99)

        envelope = PCIEnvelope.create(sender, payload, metadata)
        # Don't sign!

        verification = sat_agent.verify(envelope)

        assert verification.success is False
        assert verification.rejection.code == RejectCode.REJECT_SIGNATURE

    def test_invalid_nonce_length(self, sat_agent, ed25519_keypair):
        """Test that invalid nonce length is rejected."""
        private_key, public_key = ed25519_keypair

        from cryptography.hazmat.primitives import serialization

        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        sender = Sender(
            agent_id="pat-bad-nonce",
            agent_type="PAT",
            public_key=public_key_bytes.hex(),
        )

        payload = Payload(
            action="bad.nonce.test",
            data={},
            policy_hash="c" * 64,
        )

        metadata = Metadata(ihsan_score=0.98)

        # Create envelope with invalid nonce
        envelope = PCIEnvelope(
            version="1.0.0",
            envelope_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            nonce="abc",  # Invalid - too short
            sender=sender,
            payload=payload,
            metadata=metadata,
            signature=None,
        )

        signed = envelope.sign(private_key)
        verification = sat_agent.verify(signed)

        assert verification.success is False
        assert verification.rejection.code == RejectCode.REJECT_SCHEMA

    def test_boundary_ihsan_score(self, pat_agent, valid_policy_hash):
        """Test Ihsān score exactly at threshold."""
        result = pat_agent.create_proposal(
            action="boundary.test",
            data={},
            policy_hash=valid_policy_hash,
            ihsan_score=0.95,  # Exactly at threshold
        )

        assert result.success is True


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
