"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    BIZRA ZK Bridge Test Suite                                 ║
║              Validates PCIEnvelope → IhsanReceipt conversion                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Tests cover:                                                                 ║
║    - IhsanReceipt serialization/deserialization (104-byte format)            ║
║    - Fixed-point score conversion (0.95 → 950)                               ║
║    - ZKBridge.convert_envelope() with mock envelopes                         ║
║    - BatchConverter operations and Merkle root computation                   ║
║    - ProofInput preparation for RiscZero                                     ║
║    - Threshold validation                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import json
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

# Import with availability checking
try:
    from core.zk.bridge import (
        ZKBridge,
        ZKBridgeError,
        ConversionError,
        SerializationError,
        ValidationError,
        IhsanReceipt,
        ProofInput,
        BatchConverter,
        FIXED_POINT_SCALE,
    )
    BRIDGE_AVAILABLE = True
except ImportError as e:
    BRIDGE_AVAILABLE = False
    IMPORT_ERROR = str(e)

try:
    from core.zk.accumulator import (
        ZKAccumulator,
        MerkleProof,
        AccumulatorState,
        AccumulatorStatus,
        ProofDirection,
    )
    ACCUMULATOR_AVAILABLE = True
except ImportError:
    ACCUMULATOR_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# MOCK DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MockSender:
    """Mock sender for testing."""
    agent_id: str
    agent_type: str
    public_key: str = "mock_public_key"


@dataclass
class MockPayload:
    """Mock payload for testing."""
    action: str
    data: Dict[str, Any]


@dataclass
class MockMetadata:
    """Mock metadata for testing."""
    ihsan_score: float
    snr_score: Optional[float] = None
    urgency: str = "BATCH"
    extra: Optional[Dict[str, Any]] = None


@dataclass
class MockEnvelope:
    """Mock PCIEnvelope for testing without PCI dependency."""
    envelope_id: str
    timestamp: datetime
    nonce: str  # Hex string
    sender: MockSender
    payload: MockPayload
    metadata: MockMetadata
    version: str = "1.0.0"
    
    def digest(self) -> str:
        """Compute mock digest."""
        data = f"{self.envelope_id}:{self.timestamp.isoformat()}:{self.nonce}"
        return hashlib.sha256(data.encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_envelope() -> MockEnvelope:
    """Create a mock envelope for testing."""
    return MockEnvelope(
        envelope_id="env-12345678-1234-5678-9abc-def012345678",
        timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        nonce="a1b2c3d4e5f6" + "0" * 52,  # 64 hex chars = 32 bytes
        sender=MockSender(
            agent_id="pat-agent-001",
            agent_type="PAT",
        ),
        payload=MockPayload(
            action="PROPOSE_TRADE",
            data={"asset": "ETH", "amount": 1.5},
        ),
        metadata=MockMetadata(
            ihsan_score=0.97,
            snr_score=0.85,
            extra={"impact_score": 0.75},
        ),
    )


@pytest.fixture
def low_score_envelope() -> MockEnvelope:
    """Create envelope with low Ihsān score."""
    return MockEnvelope(
        envelope_id="env-low-score-test",
        timestamp=datetime.now(timezone.utc),
        nonce="00" * 32,
        sender=MockSender(agent_id="agent-low", agent_type="SAT"),
        payload=MockPayload(action="TEST", data={}),
        metadata=MockMetadata(ihsan_score=0.80, snr_score=0.60),
    )


@pytest.fixture
def sample_receipt() -> "IhsanReceipt":
    """Create a sample IhsanReceipt for testing."""
    if not BRIDGE_AVAILABLE:
        pytest.skip("Bridge module not available")
    
    return IhsanReceipt(
        agent_id=12345678,
        transaction_hash=bytes.fromhex("a" * 64),
        snr_score=850,
        ihsan_score=970,
        impact_score=750,
        timestamp=1705320000,  # 2024-01-15 12:00:00 UTC
        nonce=bytes.fromhex("b" * 64),
        metadata={"test": True},
    )


@pytest.fixture
def vectors_path() -> Path:
    """Path to test vectors."""
    return Path(__file__).parent / "vectors" / "zk_receipts_v1.json"


# ══════════════════════════════════════════════════════════════════════════════
# IHSAN RECEIPT TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not BRIDGE_AVAILABLE,
    reason=f"Bridge module not available: {IMPORT_ERROR if not BRIDGE_AVAILABLE else ''}"
)
class TestIhsanReceipt:
    """Tests for IhsanReceipt dataclass."""
    
    def test_receipt_creation(self, sample_receipt: IhsanReceipt):
        """Test basic receipt creation."""
        assert sample_receipt.agent_id == 12345678
        assert sample_receipt.ihsan_score == 970
        assert sample_receipt.snr_score == 850
    
    def test_receipt_validation_agent_id(self):
        """Test agent_id validation (u64 range)."""
        # Valid max u64
        receipt = IhsanReceipt(
            agent_id=2**64 - 1,
            transaction_hash=bytes(32),
            snr_score=0,
            ihsan_score=0,
            impact_score=0,
            timestamp=0,
            nonce=bytes(32),
        )
        assert receipt.agent_id == 2**64 - 1
        
        # Invalid: negative
        with pytest.raises(ValidationError):
            IhsanReceipt(
                agent_id=-1,
                transaction_hash=bytes(32),
                snr_score=0,
                ihsan_score=0,
                impact_score=0,
                timestamp=0,
                nonce=bytes(32),
            )
    
    def test_receipt_validation_hash_length(self):
        """Test transaction_hash length validation."""
        with pytest.raises(ValidationError, match="32 bytes"):
            IhsanReceipt(
                agent_id=0,
                transaction_hash=bytes(16),  # Wrong length!
                snr_score=0,
                ihsan_score=0,
                impact_score=0,
                timestamp=0,
                nonce=bytes(32),
            )
    
    def test_receipt_validation_nonce_length(self):
        """Test nonce length validation."""
        with pytest.raises(ValidationError, match="32 bytes"):
            IhsanReceipt(
                agent_id=0,
                transaction_hash=bytes(32),
                snr_score=0,
                ihsan_score=0,
                impact_score=0,
                timestamp=0,
                nonce=bytes(16),  # Wrong length!
            )
    
    def test_to_circuit_input_format(self, sample_receipt: IhsanReceipt):
        """Test serialization to 104-byte circuit input."""
        data = sample_receipt.to_circuit_input()
        
        assert isinstance(data, bytes)
        assert len(data) == 104
    
    def test_to_circuit_input_structure(self, sample_receipt: IhsanReceipt):
        """Test circuit input byte structure."""
        data = sample_receipt.to_circuit_input()
        
        # Unpack and verify structure
        (
            agent_id,
            tx_hash,
            snr,
            ihsan,
            impact,
            ts,
            nonce,
        ) = struct.unpack("<Q32sQQQQ32s", data)
        
        assert agent_id == sample_receipt.agent_id
        assert tx_hash == sample_receipt.transaction_hash
        assert snr == sample_receipt.snr_score
        assert ihsan == sample_receipt.ihsan_score
        assert impact == sample_receipt.impact_score
        assert ts == sample_receipt.timestamp
        assert nonce == sample_receipt.nonce
    
    def test_from_circuit_input_roundtrip(self, sample_receipt: IhsanReceipt):
        """Test serialization/deserialization roundtrip."""
        data = sample_receipt.to_circuit_input()
        restored = IhsanReceipt.from_circuit_input(data)
        
        assert restored.agent_id == sample_receipt.agent_id
        assert restored.transaction_hash == sample_receipt.transaction_hash
        assert restored.snr_score == sample_receipt.snr_score
        assert restored.ihsan_score == sample_receipt.ihsan_score
        assert restored.impact_score == sample_receipt.impact_score
        assert restored.timestamp == sample_receipt.timestamp
        assert restored.nonce == sample_receipt.nonce
    
    def test_from_circuit_input_wrong_length(self):
        """Test deserialization with wrong length fails."""
        with pytest.raises(SerializationError, match="104 bytes"):
            IhsanReceipt.from_circuit_input(bytes(50))
    
    def test_to_json_roundtrip(self, sample_receipt: IhsanReceipt):
        """Test JSON serialization roundtrip."""
        json_str = sample_receipt.to_json()
        restored = IhsanReceipt.from_json(json_str)
        
        assert restored.agent_id == sample_receipt.agent_id
        assert restored.transaction_hash == sample_receipt.transaction_hash
        assert restored.ihsan_score == sample_receipt.ihsan_score
    
    def test_meets_threshold_passing(self, sample_receipt: IhsanReceipt):
        """Test threshold check with passing score."""
        assert sample_receipt.ihsan_score == 970
        assert sample_receipt.meets_threshold(950)  # Default
        assert sample_receipt.meets_threshold(970)  # Exact
    
    def test_meets_threshold_failing(self):
        """Test threshold check with failing score."""
        receipt = IhsanReceipt(
            agent_id=0,
            transaction_hash=bytes(32),
            snr_score=700,
            ihsan_score=900,  # Below 950
            impact_score=0,
            timestamp=0,
            nonce=bytes(32),
        )
        
        assert not receipt.meets_threshold(950)


# ══════════════════════════════════════════════════════════════════════════════
# FIXED-POINT CONVERSION TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
class TestFixedPointConversion:
    """Tests for fixed-point score conversion."""
    
    def test_scale_constant(self):
        """Test FIXED_POINT_SCALE is 1000."""
        assert FIXED_POINT_SCALE == 1000
    
    def test_to_fixed_point_standard(self):
        """Test standard conversions."""
        bridge = ZKBridge()
        
        assert bridge._to_fixed_point(0.95) == 950
        assert bridge._to_fixed_point(0.75) == 750
        assert bridge._to_fixed_point(1.0) == 1000
        assert bridge._to_fixed_point(0.0) == 0
    
    def test_to_fixed_point_precision(self):
        """Test precision handling."""
        bridge = ZKBridge()
        
        # Should round down
        assert bridge._to_fixed_point(0.9999) == 999
        assert bridge._to_fixed_point(0.001) == 1
    
    def test_to_fixed_point_out_of_range(self):
        """Test out-of-range values raise error."""
        bridge = ZKBridge()
        
        with pytest.raises(ConversionError, match="out of range"):
            bridge._to_fixed_point(1.5)
        
        with pytest.raises(ConversionError, match="out of range"):
            bridge._to_fixed_point(-0.1)
    
    def test_from_fixed_point(self):
        """Test reverse conversion."""
        bridge = ZKBridge()
        
        assert bridge._from_fixed_point(950) == 0.95
        assert bridge._from_fixed_point(750) == 0.75
        assert bridge._from_fixed_point(1000) == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# ZK BRIDGE TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
class TestZKBridge:
    """Tests for ZKBridge conversion."""
    
    def test_bridge_creation(self):
        """Test ZKBridge initialization."""
        bridge = ZKBridge(constitution_hash="test_hash_123")
        
        assert bridge.constitution_hash == "test_hash_123"
        assert bridge.conversion_count == 0
    
    def test_bridge_default_constitution(self):
        """Test bridge with default constitution hash."""
        bridge = ZKBridge()
        
        # Should have some hash (either loaded or default)
        assert bridge.constitution_hash is not None
    
    def test_hash_agent_id(self):
        """Test agent ID hashing."""
        bridge = ZKBridge()
        
        hash1 = bridge._hash_agent_id("agent-001")
        hash2 = bridge._hash_agent_id("agent-002")
        
        # Different agents should have different hashes
        assert hash1 != hash2
        
        # Same agent should have same hash
        assert hash1 == bridge._hash_agent_id("agent-001")
        
        # Hash should be u64 range
        assert 0 <= hash1 < 2**64
    
    def test_compute_transaction_hash(self):
        """Test transaction hash computation."""
        bridge = ZKBridge()
        
        tx_hash = bridge._compute_transaction_hash("env-123", "digest-abc")
        
        assert len(tx_hash) == 32
        assert isinstance(tx_hash, bytes)
    
    def test_convert_from_dict(self):
        """Test conversion from dictionary."""
        bridge = ZKBridge()
        
        envelope_dict = {
            "envelope_id": "test-envelope-id",
            "timestamp": "2024-01-15T12:00:00+00:00",
            "nonce": "00" * 32,
            "sender": {"agent_id": "test-agent"},
            "metadata": {
                "ihsan_score": 0.96,
                "snr_score": 0.80,
            },
        }
        
        receipt = bridge.convert_from_dict(envelope_dict)
        
        assert receipt.ihsan_score == 960
        assert receipt.snr_score == 800
        assert receipt.metadata["envelope_id"] == "test-envelope-id"
    
    def test_prepare_proof_input(self, sample_receipt: IhsanReceipt):
        """Test proof input preparation."""
        bridge = ZKBridge(constitution_hash="test_constitution")
        
        proof_input = bridge.prepare_proof_input(
            receipt=sample_receipt,
            ihsan_threshold=950,
            snr_threshold=750,
        )
        
        assert proof_input.receipt is sample_receipt
        assert proof_input.constitution_hash == "test_constitution"
        assert proof_input.public_inputs["ihsan_threshold"] == 950
        assert proof_input.public_inputs["snr_threshold"] == 750
    
    def test_proof_input_risc0_format(self, sample_receipt: IhsanReceipt):
        """Test RiscZero input format."""
        bridge = ZKBridge(constitution_hash="test_hash")
        proof_input = bridge.prepare_proof_input(sample_receipt)
        
        risc0_input = proof_input.to_risc0_input()
        
        assert "input" in risc0_input
        assert "assumptions" in risc0_input
        assert risc0_input["assumptions"]["constitution_hash"] == "test_hash"
    
    def test_validate_receipt_passing(self, sample_receipt: IhsanReceipt):
        """Test receipt validation with passing scores."""
        bridge = ZKBridge()
        
        is_valid, reason = bridge.validate_receipt(sample_receipt)
        
        assert is_valid is True
        assert reason is None
    
    def test_validate_receipt_failing_ihsan(self):
        """Test receipt validation with failing Ihsān."""
        bridge = ZKBridge()
        
        receipt = IhsanReceipt(
            agent_id=0,
            transaction_hash=bytes(32),
            snr_score=800,
            ihsan_score=900,  # Below 950
            impact_score=0,
            timestamp=0,
            nonce=bytes(32),
        )
        
        is_valid, reason = bridge.validate_receipt(receipt)
        
        assert is_valid is False
        assert "Ihsān" in reason
    
    def test_validate_receipt_failing_snr(self):
        """Test receipt validation with failing SNR."""
        bridge = ZKBridge()
        
        receipt = IhsanReceipt(
            agent_id=0,
            transaction_hash=bytes(32),
            snr_score=700,  # Below 750
            ihsan_score=980,
            impact_score=0,
            timestamp=0,
            nonce=bytes(32),
        )
        
        is_valid, reason = bridge.validate_receipt(receipt)
        
        assert is_valid is False
        assert "SNR" in reason


# ══════════════════════════════════════════════════════════════════════════════
# BATCH CONVERTER TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
class TestBatchConverter:
    """Tests for BatchConverter operations."""
    
    def test_batch_creation(self):
        """Test batch converter creation."""
        batch = BatchConverter()
        
        assert batch.size == 0
    
    def test_add_receipt(self, sample_receipt: IhsanReceipt):
        """Test adding receipt to batch."""
        batch = BatchConverter()
        
        index = batch.add_receipt(sample_receipt)
        
        assert index == 0
        assert batch.size == 1
    
    def test_add_multiple_receipts(self, sample_receipt: IhsanReceipt):
        """Test adding multiple receipts."""
        batch = BatchConverter()
        
        for i in range(5):
            # Create unique receipts
            receipt = IhsanReceipt(
                agent_id=i,
                transaction_hash=hashlib.sha256(str(i).encode()).digest(),
                snr_score=800 + i,
                ihsan_score=960 + i,
                impact_score=700 + i,
                timestamp=1705320000 + i,
                nonce=bytes.fromhex(f"{i:064x}"),
            )
            index = batch.add_receipt(receipt)
            assert index == i
        
        assert batch.size == 5
    
    def test_add_dict(self):
        """Test adding envelope dict to batch."""
        batch = BatchConverter()
        
        envelope_dict = {
            "envelope_id": "test-id",
            "timestamp": "2024-01-15T12:00:00Z",
            "nonce": "00" * 32,
            "sender": {"agent_id": "agent-1"},
            "metadata": {"ihsan_score": 0.95, "snr_score": 0.80},
        }
        
        index = batch.add_dict(envelope_dict)
        
        assert index == 0
        assert batch.size == 1
    
    def test_get_batch(self, sample_receipt: IhsanReceipt):
        """Test retrieving batch contents."""
        batch = BatchConverter()
        batch.add_receipt(sample_receipt)
        
        receipts = batch.get_batch()
        
        assert len(receipts) == 1
        assert receipts[0].agent_id == sample_receipt.agent_id
    
    def test_clear_batch(self, sample_receipt: IhsanReceipt):
        """Test clearing batch."""
        batch = BatchConverter()
        batch.add_receipt(sample_receipt)
        assert batch.size == 1
        
        batch.clear()
        
        assert batch.size == 0
    
    def test_compute_batch_hash_empty(self):
        """Test batch hash of empty batch."""
        batch = BatchConverter()
        
        batch_hash = batch.compute_batch_hash()
        
        assert batch_hash == bytes(32)
    
    def test_compute_batch_hash_single(self, sample_receipt: IhsanReceipt):
        """Test batch hash with single receipt."""
        batch = BatchConverter()
        batch.add_receipt(sample_receipt)
        
        batch_hash = batch.compute_batch_hash()
        
        # Single item = hash of that item
        expected = hashlib.sha256(sample_receipt.to_circuit_input()).digest()
        assert batch_hash == expected
    
    def test_compute_batch_hash_multiple(self):
        """Test batch hash with multiple receipts."""
        batch = BatchConverter()
        
        # Add two receipts
        r1 = IhsanReceipt(
            agent_id=1,
            transaction_hash=bytes(32),
            snr_score=800,
            ihsan_score=950,
            impact_score=700,
            timestamp=100,
            nonce=bytes(32),
        )
        r2 = IhsanReceipt(
            agent_id=2,
            transaction_hash=bytes.fromhex("ff" * 32),
            snr_score=850,
            ihsan_score=960,
            impact_score=750,
            timestamp=200,
            nonce=bytes.fromhex("aa" * 32),
        )
        
        batch.add_receipt(r1)
        batch.add_receipt(r2)
        
        batch_hash = batch.compute_batch_hash()
        
        # Merkle root of two items
        h1 = hashlib.sha256(r1.to_circuit_input()).digest()
        h2 = hashlib.sha256(r2.to_circuit_input()).digest()
        expected = hashlib.sha256(h1 + h2).digest()
        
        assert batch_hash == expected
    
    def test_serialize_batch(self, sample_receipt: IhsanReceipt):
        """Test batch serialization."""
        batch = BatchConverter()
        batch.add_receipt(sample_receipt)
        
        data = batch.serialize_batch()
        
        # Header: 4 (count) + 32 (hash) = 36
        # Body: 104 * 1 = 104
        # Total: 140
        assert len(data) == 140
    
    def test_deserialize_batch(self, sample_receipt: IhsanReceipt):
        """Test batch deserialization."""
        batch = BatchConverter()
        batch.add_receipt(sample_receipt)
        
        data = batch.serialize_batch()
        restored = BatchConverter.deserialize_batch(data)
        
        assert len(restored) == 1
        assert restored[0].agent_id == sample_receipt.agent_id


# ══════════════════════════════════════════════════════════════════════════════
# ACCUMULATOR INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not (BRIDGE_AVAILABLE and ACCUMULATOR_AVAILABLE),
    reason="Bridge or Accumulator not available"
)
class TestAccumulatorIntegration:
    """Integration tests for ZKAccumulator with bridge."""
    
    def test_accumulator_creation(self):
        """Test accumulator creation."""
        acc = ZKAccumulator(constitution_hash="test_hash")
        
        assert acc.constitution_hash == "test_hash"
        assert acc.status == AccumulatorStatus.EMPTY
        assert acc.size == 0
    
    def test_accumulator_add_receipt(self, sample_receipt: IhsanReceipt):
        """Test adding receipt to accumulator."""
        acc = ZKAccumulator(constitution_hash="test")
        
        index = acc.add(sample_receipt)
        
        assert index == 0
        assert acc.size == 1
        assert acc.status == AccumulatorStatus.ACCUMULATING
    
    def test_accumulator_merkle_root(self, sample_receipt: IhsanReceipt):
        """Test Merkle root computation."""
        acc = ZKAccumulator(constitution_hash="test")
        acc.add(sample_receipt)
        
        root = acc.get_merkle_root()
        
        assert len(root) == 32
        assert root != bytes(32)
    
    def test_accumulator_generate_proof(self, sample_receipt: IhsanReceipt):
        """Test Merkle proof generation."""
        acc = ZKAccumulator(constitution_hash="test")
        acc.add(sample_receipt)
        
        proof = acc.generate_proof(0)
        
        assert isinstance(proof, MerkleProof)
        assert proof.leaf_index == 0
        assert len(proof.leaf_hash) == 32
        assert proof.root == acc.get_merkle_root()
    
    def test_accumulator_verify_membership(self, sample_receipt: IhsanReceipt):
        """Test membership verification."""
        acc = ZKAccumulator(constitution_hash="test")
        acc.add(sample_receipt)
        
        proof = acc.generate_proof(0)
        
        assert acc.verify_membership(sample_receipt, proof)
    
    def test_accumulator_seal(self, sample_receipt: IhsanReceipt):
        """Test sealing accumulator."""
        acc = ZKAccumulator(constitution_hash="test_constitution")
        acc.add(sample_receipt)
        
        state = acc.seal()
        
        assert acc.is_sealed
        assert acc.status == AccumulatorStatus.SEALED
        assert isinstance(state, AccumulatorState)
        assert state.leaf_count == 1
        assert state.constitution_hash == "test_constitution"
    
    def test_accumulator_cannot_add_after_seal(self, sample_receipt: IhsanReceipt):
        """Test that adding to sealed accumulator fails."""
        acc = ZKAccumulator(constitution_hash="test")
        acc.add(sample_receipt)
        acc.seal()
        
        with pytest.raises(Exception):  # StateError
            acc.add(sample_receipt)
    
    def test_accumulator_prepare_recursive_input(self, sample_receipt: IhsanReceipt):
        """Test recursive proof input preparation."""
        acc = ZKAccumulator(constitution_hash="test_constitution")
        acc.add(sample_receipt)
        acc.seal()
        
        recursive_input = acc.prepare_recursive_input()
        
        assert len(recursive_input.receipts) == 1
        assert recursive_input.constitution_hash == "test_constitution"
        assert recursive_input.batch_size == 1
        assert recursive_input.merkle_root == acc.get_merkle_root()


# ══════════════════════════════════════════════════════════════════════════════
# MERKLE PROOF TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not ACCUMULATOR_AVAILABLE, reason="Accumulator not available")
class TestMerkleProof:
    """Tests for MerkleProof operations."""
    
    def test_proof_creation(self):
        """Test MerkleProof creation."""
        proof = MerkleProof(
            leaf_hash=bytes(32),
            leaf_index=0,
            siblings=[bytes(32)],
            directions=[ProofDirection.RIGHT],
            root=bytes(32),
        )
        
        assert proof.leaf_index == 0
        assert len(proof.siblings) == 1
    
    def test_proof_validation_mismatch(self):
        """Test proof validation catches mismatched lengths."""
        with pytest.raises(Exception):  # MerkleProofError
            MerkleProof(
                leaf_hash=bytes(32),
                leaf_index=0,
                siblings=[bytes(32), bytes(32)],
                directions=[ProofDirection.RIGHT],  # Only one!
                root=bytes(32),
            )
    
    def test_proof_serialization_roundtrip(self):
        """Test proof serialization/deserialization."""
        original = MerkleProof(
            leaf_hash=bytes.fromhex("aa" * 32),
            leaf_index=42,
            siblings=[bytes.fromhex("bb" * 32), bytes.fromhex("cc" * 32)],
            directions=[ProofDirection.LEFT, ProofDirection.RIGHT],
            root=bytes.fromhex("dd" * 32),
        )
        
        data = original.to_bytes()
        restored = MerkleProof.from_bytes(data)
        
        assert restored.leaf_hash == original.leaf_hash
        assert restored.leaf_index == original.leaf_index
        assert restored.siblings == original.siblings
        assert restored.directions == original.directions
        assert restored.root == original.root
    
    def test_proof_json_roundtrip(self):
        """Test proof JSON serialization."""
        original = MerkleProof(
            leaf_hash=bytes.fromhex("11" * 32),
            leaf_index=7,
            siblings=[bytes.fromhex("22" * 32)],
            directions=[ProofDirection.LEFT],
            root=bytes.fromhex("33" * 32),
        )
        
        json_str = original.to_json()
        restored = MerkleProof.from_json(json_str)
        
        assert restored.leaf_index == original.leaf_index


# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN VECTOR TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
class TestGoldenVectors:
    """Tests using golden test vectors."""
    
    def test_vectors_file_exists(self, vectors_path: Path):
        """Test that vectors file exists (if created)."""
        if not vectors_path.exists():
            pytest.skip("Golden vectors not yet created")
        
        with open(vectors_path) as f:
            vectors = json.load(f)
        
        assert "receipts" in vectors
    
    def test_known_receipt_hash(self):
        """Test known receipt produces expected hash."""
        # Create deterministic receipt
        receipt = IhsanReceipt(
            agent_id=0,
            transaction_hash=bytes(32),
            snr_score=750,
            ihsan_score=950,
            impact_score=500,
            timestamp=0,
            nonce=bytes(32),
        )
        
        data = receipt.to_circuit_input()
        receipt_hash = hashlib.sha256(data).hexdigest()
        
        # This hash should be deterministic
        assert len(receipt_hash) == 64
        # Note: Actual expected value would be computed and stored in vectors


# ══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
class TestErrorHandling:
    """Tests for error handling."""
    
    def test_conversion_error_hierarchy(self):
        """Test exception hierarchy."""
        assert issubclass(ConversionError, ZKBridgeError)
        assert issubclass(SerializationError, ZKBridgeError)
        assert issubclass(ValidationError, ZKBridgeError)
    
    def test_conversion_error_message(self):
        """Test error messages are informative."""
        bridge = ZKBridge()
        
        with pytest.raises(ConversionError) as exc_info:
            bridge._to_fixed_point(2.0)
        
        assert "out of range" in str(exc_info.value)
