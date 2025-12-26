"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    BIZRA ZK Bridge Module v1.0.0                              ║
║              PCIEnvelope → IhsanReceipt Conversion Layer                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  This module bridges the PCI protocol layer to the ZK settlement layer:       ║
║    - Converts PCIEnvelope to IhsanReceipt format                             ║
║    - Prepares data for RiscZero STARK circuit                                ║
║    - Handles fixed-point conversion for arithmetic circuits                   ║
║    - Provides serialization for proof inputs                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

# Type checking imports (don't execute at runtime)
if TYPE_CHECKING:
    from core.pci.envelope import PCIEnvelope
    from core.constitution import Constitution

# Runtime imports with availability flags
PCI_AVAILABLE = False
CONSTITUTION_AVAILABLE = False
BLAKE3_AVAILABLE = False

try:
    from core.pci.envelope import PCIEnvelope as _PCIEnvelope
    PCI_AVAILABLE = True
except ImportError:
    pass

try:
    from core.constitution import Constitution as _Constitution
    CONSTITUTION_AVAILABLE = True
except ImportError:
    pass

try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Fixed-point precision (3 decimal places → multiply by 1000)
FIXED_POINT_SCALE = 1000

# Hash output sizes
SHA256_BYTES = 32
BLAKE3_BYTES = 32

# Field sizes for ZK circuit (in bits)
AGENT_ID_BITS = 64
TRANSACTION_HASH_BITS = 256
SCORE_BITS = 64


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class ZKBridgeError(Exception):
    """Base exception for ZK bridge errors."""
    pass


class ConversionError(ZKBridgeError):
    """Error during envelope to receipt conversion."""
    pass


class SerializationError(ZKBridgeError):
    """Error during serialization for ZK circuit."""
    pass


class ValidationError(ZKBridgeError):
    """Validation error for ZK inputs."""
    pass


class ConstitutionNotBoundError(ZKBridgeError):
    """Raised when ZKBridge is used without constitutional binding."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IhsanReceipt:
    """
    ZK-compatible representation of a verified action.
    
    This structure maps to the RiscZero guest circuit input format.
    All floating-point values are converted to fixed-point integers.
    
    Attributes:
        agent_id: 64-bit hash of agent identifier
        transaction_hash: 256-bit hash of the original envelope
        snr_score: Fixed-point SNR score (0.95 → 950)
        ihsan_score: Fixed-point Ihsān score (0.95 → 950)
        impact_score: Fixed-point impact score
        timestamp: Unix timestamp of the action
        nonce: Original nonce for replay verification
        metadata: Additional context for the proof
    """
    agent_id: int  # u64
    transaction_hash: bytes  # [u8; 32]
    snr_score: int  # u64, fixed-point
    ihsan_score: int  # u64, fixed-point
    impact_score: int  # u64, fixed-point
    timestamp: int  # u64, Unix epoch
    nonce: bytes  # [u8; 32]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate receipt fields."""
        if self.agent_id < 0 or self.agent_id >= 2**64:
            raise ValidationError(f"agent_id out of u64 range: {self.agent_id}")
        
        if len(self.transaction_hash) != 32:
            raise ValidationError(f"transaction_hash must be 32 bytes, got {len(self.transaction_hash)}")
        
        if len(self.nonce) != 32:
            raise ValidationError(f"nonce must be 32 bytes, got {len(self.nonce)}")
        
        # Validate scores are non-negative
        for name, value in [("snr_score", self.snr_score), 
                            ("ihsan_score", self.ihsan_score),
                            ("impact_score", self.impact_score)]:
            if value < 0 or value >= 2**64:
                raise ValidationError(f"{name} out of u64 range: {value}")
    
    def to_circuit_input(self) -> bytes:
        """
        Serialize to binary format for ZK circuit input.
        
        Format (little-endian):
            - agent_id: u64 (8 bytes)
            - transaction_hash: [u8; 32] (32 bytes)
            - snr_score: u64 (8 bytes)
            - ihsan_score: u64 (8 bytes)
            - impact_score: u64 (8 bytes)
            - timestamp: u64 (8 bytes)
            - nonce: [u8; 32] (32 bytes)
            
        Total: 104 bytes
        """
        return struct.pack(
            "<Q32sQQQQ32s",
            self.agent_id,
            self.transaction_hash,
            self.snr_score,
            self.ihsan_score,
            self.impact_score,
            self.timestamp,
            self.nonce,
        )
    
    @classmethod
    def from_circuit_input(cls, data: bytes) -> IhsanReceipt:
        """Deserialize from binary circuit input format."""
        if len(data) != 104:
            raise SerializationError(f"Expected 104 bytes, got {len(data)}")
        
        (
            agent_id,
            transaction_hash,
            snr_score,
            ihsan_score,
            impact_score,
            timestamp,
            nonce,
        ) = struct.unpack("<Q32sQQQQ32s", data)
        
        return cls(
            agent_id=agent_id,
            transaction_hash=transaction_hash,
            snr_score=snr_score,
            ihsan_score=ihsan_score,
            impact_score=impact_score,
            timestamp=timestamp,
            nonce=nonce,
        )
    
    def to_json(self) -> str:
        """Serialize to JSON with hex-encoded bytes."""
        return json.dumps({
            "agent_id": self.agent_id,
            "transaction_hash": self.transaction_hash.hex(),
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "impact_score": self.impact_score,
            "timestamp": self.timestamp,
            "nonce": self.nonce.hex(),
            "metadata": self.metadata,
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> IhsanReceipt:
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(
            agent_id=data["agent_id"],
            transaction_hash=bytes.fromhex(data["transaction_hash"]),
            snr_score=data["snr_score"],
            ihsan_score=data["ihsan_score"],
            impact_score=data["impact_score"],
            timestamp=data["timestamp"],
            nonce=bytes.fromhex(data["nonce"]),
            metadata=data.get("metadata", {}),
        )
    
    def meets_threshold(self, threshold_fixed: int = 950) -> bool:
        """Check if Ihsān score meets threshold (fixed-point comparison)."""
        return self.ihsan_score >= threshold_fixed


@dataclass
class ProofInput:
    """
    Complete input package for ZK proof generation.
    
    Contains the receipt plus public/private witness separation.
    """
    receipt: IhsanReceipt
    public_inputs: Dict[str, Any]  # Revealed to verifier
    private_inputs: Dict[str, Any]  # Hidden from verifier
    constitution_hash: str
    
    def to_risc0_input(self) -> Dict[str, Any]:
        """
        Format for RiscZero Bonsai API.
        
        Returns dict with 'input' (private) and 'assumptions' (public).
        """
        return {
            "input": self.receipt.to_circuit_input().hex(),
            "assumptions": {
                "constitution_hash": self.constitution_hash,
                "ihsan_threshold": self.public_inputs.get("ihsan_threshold", 950),
                "snr_threshold": self.public_inputs.get("snr_threshold", 750),
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# ZK BRIDGE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class ZKBridge:
    """
    Bridge between PCI protocol and ZK settlement layer.
    
    Converts PCIEnvelope instances to IhsanReceipt format suitable for
    zero-knowledge proof generation.
    
    Usage:
        bridge = ZKBridge()  # Loads constitution automatically
        receipt = bridge.convert_envelope(envelope)
        proof_input = bridge.prepare_proof_input(receipt)
    """
    
    def __init__(
        self,
        constitution_hash: Optional[str] = None,
        require_constitution: bool = True,
    ):
        """
        Initialize ZK bridge.
        
        Args:
            constitution_hash: Hash of constitution.toml for binding.
                              If None, attempts to load from Constitution singleton.
            require_constitution: If True (default), raises error if constitution
                                  is not available. If False, allows operation
                                  without binding (NOT RECOMMENDED for production).
                                  
        Raises:
            ConstitutionNotBoundError: If require_constitution=True and constitution
                                       is not loaded.
        """
        self._constitution_hash: str = constitution_hash or ""
        self._ihsan_threshold: int = 950  # Default, will be overridden by constitution
        self._snr_threshold: int = 750    # Default, will be overridden by constitution
        self._conversion_count = 0
        self._constitution_bound = False
        
        # Try to load constitution if not provided
        if constitution_hash is None and CONSTITUTION_AVAILABLE:
            try:
                from core.constitution import Constitution
                constitution = Constitution.get()
                self._constitution_hash = constitution.hash
                # Derive thresholds from constitution
                self._ihsan_threshold = constitution.zk.ihsan_minimum_fixed
                self._snr_threshold = constitution.zk.snr_minimum_fixed
                self._constitution_bound = True
            except Exception:
                if require_constitution:
                    raise ConstitutionNotBoundError(
                        "ZKBridge requires constitution binding. Call Constitution.load() first "
                        "or pass require_constitution=False to disable (NOT RECOMMENDED)."
                    )
                self._constitution_hash = "NO_CONSTITUTION"
        elif constitution_hash:
            self._constitution_bound = True
        elif require_constitution:
            raise ConstitutionNotBoundError(
                "ZKBridge requires constitution binding. Call Constitution.load() first "
                "or pass require_constitution=False to disable (NOT RECOMMENDED)."
            )
        else:
            self._constitution_hash = "NO_CONSTITUTION"
    
    @property
    def constitution_bound(self) -> bool:
        """Check if bridge is bound to a constitution."""
        return self._constitution_bound
    
    @property
    def ihsan_threshold(self) -> int:
        """Get the Ihsān threshold from constitution (fixed-point)."""
        return self._ihsan_threshold
    
    @property
    def snr_threshold(self) -> int:
        """Get the SNR threshold from constitution (fixed-point)."""
        return self._snr_threshold
    
    @property
    def constitution_hash(self) -> str:
        """Get the constitution hash used for binding."""
        return self._constitution_hash
    
    @property
    def conversion_count(self) -> int:
        """Number of envelopes converted."""
        return self._conversion_count
    
    def _hash_agent_id(self, agent_id: str) -> int:
        """
        Convert agent ID string to 64-bit integer.
        
        Uses first 8 bytes of SHA256 hash.
        """
        h = hashlib.sha256(agent_id.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little")
    
    def _compute_transaction_hash(self, envelope_id: str, digest: str) -> bytes:
        """
        Compute 256-bit transaction hash for ZK circuit.
        
        Combines envelope_id and digest using SHA256 (required by ZK circuit).
        """
        combined = f"{envelope_id}:{digest}".encode("utf-8")
        return hashlib.sha256(combined).digest()
    
    def _to_fixed_point(self, value: float, scale: int = FIXED_POINT_SCALE) -> int:
        """Convert float to fixed-point integer."""
        if value < 0 or value > 1.0:
            raise ConversionError(f"Score out of range [0, 1]: {value}")
        return int(value * scale)
    
    def _from_fixed_point(self, value: int, scale: int = FIXED_POINT_SCALE) -> float:
        """Convert fixed-point integer to float."""
        return value / scale
    
    def convert_envelope(self, envelope: Any) -> IhsanReceipt:
        """
        Convert a PCIEnvelope to IhsanReceipt format.
        
        Args:
            envelope: Verified PCI envelope (PCIEnvelope instance)
            
        Returns:
            IhsanReceipt ready for ZK proof generation
            
        Raises:
            ConversionError: If conversion fails
        """
        if not PCI_AVAILABLE:
            raise ConversionError("PCI module not available")
        
        try:
            # Extract and convert agent ID
            agent_id = self._hash_agent_id(envelope.sender.agent_id)
            
            # Compute transaction hash using envelope's digest() method
            envelope_digest = envelope.digest()
            transaction_hash = self._compute_transaction_hash(
                envelope.envelope_id,
                envelope_digest
            )
            
            # Convert scores to fixed-point
            ihsan_score = self._to_fixed_point(
                envelope.metadata.ihsan_score if envelope.metadata.ihsan_score else 0.0
            )
            snr_score = self._to_fixed_point(
                envelope.metadata.snr_score if envelope.metadata.snr_score else 0.0
            )
            
            # Impact score from extra dict if present
            impact_score = 0
            if hasattr(envelope.metadata, "extra") and envelope.metadata.extra:
                extra_impact = envelope.metadata.extra.get("impact_score", 0.0)
                if extra_impact:
                    impact_score = self._to_fixed_point(extra_impact)
            
            # Parse timestamp from envelope (datetime object)
            if isinstance(envelope.timestamp, datetime):
                timestamp = int(envelope.timestamp.timestamp())
            elif isinstance(envelope.timestamp, str):
                dt = datetime.fromisoformat(
                    envelope.timestamp.replace("Z", "+00:00")
                )
                timestamp = int(dt.timestamp())
            else:
                timestamp = int(datetime.now(timezone.utc).timestamp())
            
            # Get nonce from envelope (hex string) - pad or truncate to 32 bytes
            nonce_hex = envelope.nonce if envelope.nonce else ("00" * 32)
            nonce_bytes = bytes.fromhex(nonce_hex)
            if len(nonce_bytes) < 32:
                nonce_bytes = nonce_bytes + bytes(32 - len(nonce_bytes))
            elif len(nonce_bytes) > 32:
                nonce_bytes = nonce_bytes[:32]
            
            # Build receipt
            receipt = IhsanReceipt(
                agent_id=agent_id,
                transaction_hash=transaction_hash,
                snr_score=snr_score,
                ihsan_score=ihsan_score,
                impact_score=impact_score,
                timestamp=timestamp,
                nonce=nonce_bytes,
                metadata={
                    "envelope_id": envelope.envelope_id,
                    "action": envelope.payload.action,
                    "agent_type": envelope.sender.agent_type,
                    "converted_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            
            self._conversion_count += 1
            return receipt
            
        except Exception as e:
            raise ConversionError(f"Failed to convert envelope: {e}") from e
    
    def convert_from_dict(self, envelope_dict: Dict[str, Any]) -> IhsanReceipt:
        """
        Convert an envelope dictionary to IhsanReceipt.
        
        Useful when working with serialized envelopes.
        Uses the same hashing and field access patterns as convert_envelope
        for deterministic results.
        """
        try:
            # Extract fields
            agent_id = self._hash_agent_id(
                envelope_dict.get("sender", {}).get("agent_id", "unknown")
            )
            
            envelope_id = envelope_dict.get("envelope_id", "")
            metadata = envelope_dict.get("metadata", {})
            
            # Compute transaction hash using same method as convert_envelope:
            # Combine envelope_id and digest (or compute digest from payload)
            payload = envelope_dict.get("payload", {})
            if "digest" in envelope_dict:
                digest = envelope_dict["digest"]
            else:
                # Compute digest from payload using canonical JSON
                digest = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
                ).hexdigest()
            
            transaction_hash = self._compute_transaction_hash(envelope_id, digest)
            
            # Convert scores - check both metadata.X and metadata.extra.X for consistency
            ihsan_score = self._to_fixed_point(
                metadata.get("ihsan_score", 0.0)
            )
            snr_score = self._to_fixed_point(
                metadata.get("snr_score", 0.0)
            )
            
            # Impact score: check metadata.extra.impact_score first (matches convert_envelope),
            # then fall back to metadata.impact_score for dict-based input compatibility
            impact_score = 0
            extra = metadata.get("extra", {})
            if extra and "impact_score" in extra:
                impact_score = self._to_fixed_point(extra["impact_score"])
            elif "impact_score" in metadata:
                impact_score = self._to_fixed_point(metadata["impact_score"])
            
            # Parse timestamp from envelope level
            ts_str = envelope_dict.get("timestamp", "")
            if ts_str:
                if isinstance(ts_str, str):
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    timestamp = int(dt.timestamp())
                else:
                    timestamp = int(datetime.now(timezone.utc).timestamp())
            else:
                timestamp = int(datetime.now(timezone.utc).timestamp())
            
            # Get nonce from envelope level
            nonce_hex = envelope_dict.get("nonce", "00" * 32)
            nonce_bytes = bytes.fromhex(nonce_hex)
            if len(nonce_bytes) < 32:
                nonce_bytes = nonce_bytes + bytes(32 - len(nonce_bytes))
            elif len(nonce_bytes) > 32:
                nonce_bytes = nonce_bytes[:32]
            
            receipt = IhsanReceipt(
                agent_id=agent_id,
                transaction_hash=transaction_hash,
                snr_score=snr_score,
                ihsan_score=ihsan_score,
                impact_score=impact_score,
                timestamp=timestamp,
                nonce=nonce_bytes,
                metadata={
                    "envelope_id": envelope_id,
                    "source": "dict",
                    "converted_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            
            self._conversion_count += 1
            return receipt
            
        except Exception as e:
            raise ConversionError(f"Failed to convert dict: {e}") from e
    
    def prepare_proof_input(
        self,
        receipt: IhsanReceipt,
        ihsan_threshold: Optional[int] = None,
        snr_threshold: Optional[int] = None,
    ) -> ProofInput:
        """
        Prepare complete proof input package.
        
        Args:
            receipt: IhsanReceipt to prove
            ihsan_threshold: Fixed-point Ihsān threshold. If None, uses
                            constitution threshold (recommended).
            snr_threshold: Fixed-point SNR threshold. If None, uses
                          constitution threshold (recommended).
            
        Returns:
            ProofInput ready for ZK prover
        """
        # Use constitutional thresholds if not overridden
        if ihsan_threshold is None:
            ihsan_threshold = self._ihsan_threshold
        if snr_threshold is None:
            snr_threshold = self._snr_threshold
            
        return ProofInput(
            receipt=receipt,
            public_inputs={
                "ihsan_threshold": ihsan_threshold,
                "snr_threshold": snr_threshold,
                "agent_id": receipt.agent_id,
                "transaction_hash": receipt.transaction_hash.hex(),
            },
            private_inputs={
                "ihsan_score": receipt.ihsan_score,
                "snr_score": receipt.snr_score,
                "impact_score": receipt.impact_score,
            },
            constitution_hash=self._constitution_hash,
        )
    
    def validate_receipt(
        self,
        receipt: IhsanReceipt,
        ihsan_threshold: Optional[int] = None,
        snr_threshold: Optional[int] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate receipt against thresholds (pre-proof check).
        
        Args:
            receipt: Receipt to validate
            ihsan_threshold: Minimum Ihsān score (fixed-point). If None, uses
                            constitution threshold.
            snr_threshold: Minimum SNR score (fixed-point). If None, uses
                          constitution threshold.
            
        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        # Use constitutional thresholds if not overridden
        if ihsan_threshold is None:
            ihsan_threshold = self._ihsan_threshold
        if snr_threshold is None:
            snr_threshold = self._snr_threshold
            
        if receipt.ihsan_score < ihsan_threshold:
            return False, f"Ihsān {receipt.ihsan_score} < threshold {ihsan_threshold}"
        
        if receipt.snr_score < snr_threshold:
            return False, f"SNR {receipt.snr_score} < threshold {snr_threshold}"
        
        return True, None


# ══════════════════════════════════════════════════════════════════════════════
# BATCH OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

class BatchConverter:
    """
    Batch conversion of multiple envelopes for recursive proof composition.
    """
    
    def __init__(self, bridge: Optional[ZKBridge] = None):
        """Initialize with optional existing bridge."""
        self._bridge = bridge or ZKBridge()
        self._batch: List[IhsanReceipt] = []
    
    def add_envelope(self, envelope: Any) -> int:
        """Add envelope to batch, return batch index."""
        receipt = self._bridge.convert_envelope(envelope)
        self._batch.append(receipt)
        return len(self._batch) - 1
    
    def add_dict(self, envelope_dict: Dict[str, Any]) -> int:
        """Add envelope dict to batch, return batch index."""
        receipt = self._bridge.convert_from_dict(envelope_dict)
        self._batch.append(receipt)
        return len(self._batch) - 1
    
    def add_receipt(self, receipt: IhsanReceipt) -> int:
        """Add pre-converted receipt to batch."""
        self._batch.append(receipt)
        return len(self._batch) - 1
    
    def get_batch(self) -> List[IhsanReceipt]:
        """Get all receipts in batch."""
        return self._batch.copy()
    
    def clear(self):
        """Clear the batch."""
        self._batch.clear()
    
    @property
    def size(self) -> int:
        """Current batch size."""
        return len(self._batch)
    
    def compute_batch_hash(self) -> bytes:
        """
        Compute Merkle root of batch for aggregated proof.
        
        Returns 32-byte hash.
        """
        if not self._batch:
            return bytes(32)
        
        # Compute leaf hashes
        leaves = [
            hashlib.sha256(r.to_circuit_input()).digest()
            for r in self._batch
        ]
        
        # Build Merkle tree (simple implementation)
        while len(leaves) > 1:
            if len(leaves) % 2 == 1:
                leaves.append(leaves[-1])  # Duplicate last if odd
            
            new_leaves = []
            for i in range(0, len(leaves), 2):
                combined = leaves[i] + leaves[i + 1]
                new_leaves.append(hashlib.sha256(combined).digest())
            leaves = new_leaves
        
        return leaves[0]
    
    def serialize_batch(self) -> bytes:
        """Serialize entire batch for proof generation."""
        # Header: count (u32) + batch_hash (32 bytes)
        header = struct.pack("<I", len(self._batch)) + self.compute_batch_hash()
        
        # Body: concatenated receipts
        body = b"".join(r.to_circuit_input() for r in self._batch)
        
        return header + body
    
    @classmethod
    def deserialize_batch(cls, data: bytes) -> List[IhsanReceipt]:
        """Deserialize batch from bytes."""
        if len(data) < 36:  # 4 + 32 header
            raise SerializationError("Batch data too short")
        
        count = struct.unpack("<I", data[:4])[0]
        # batch_hash = data[4:36]  # Could verify
        
        receipts = []
        offset = 36
        for _ in range(count):
            if offset + 104 > len(data):
                raise SerializationError("Incomplete receipt in batch")
            receipt = IhsanReceipt.from_circuit_input(data[offset:offset + 104])
            receipts.append(receipt)
            offset += 104
        
        return receipts


# ══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "ZKBridge",
    "ZKBridgeError",
    "ConversionError",
    "SerializationError",
    "ValidationError",
    "ConstitutionNotBoundError",
    "IhsanReceipt",
    "ProofInput",
    "BatchConverter",
    "FIXED_POINT_SCALE",
]
