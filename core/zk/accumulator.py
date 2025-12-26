"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    BIZRA ZK Accumulator Module v1.0.0                         ║
║              Merkle Tree + Recursive Proof Composition Layer                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  This module provides:                                                        ║
║    - ZKAccumulator: Incremental receipt accumulation with Merkle proofs      ║
║    - MerkleProof: Membership proof with path and directions                  ║
║    - RecursiveProofInput: Batch aggregation for recursive ZK proofs          ║
║    - AccumulatorState: Commitment binding for on-chain verification          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.zk.bridge import IhsanReceipt

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Maximum batch size per constitution.toml [zk.proof.batch_size]
MAX_BATCH_SIZE = 1000

# Hash output size
HASH_BYTES = 32

# Merkle tree depth limit (2^20 = 1M receipts max)
MAX_TREE_DEPTH = 20


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class AccumulatorError(Exception):
    """Base exception for accumulator errors."""
    pass


class MerkleProofError(AccumulatorError):
    """Error in Merkle proof generation or verification."""
    pass


class BatchOverflowError(AccumulatorError):
    """Batch size exceeded maximum."""
    pass


class StateError(AccumulatorError):
    """Invalid accumulator state."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class AccumulatorStatus(Enum):
    """Status of the accumulator."""
    EMPTY = auto()
    ACCUMULATING = auto()
    SEALED = auto()
    PROVEN = auto()
    SETTLED = auto()


class ProofDirection(Enum):
    """Direction in Merkle proof path."""
    LEFT = 0   # Sibling is on the left
    RIGHT = 1  # Sibling is on the right


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MerkleProof:
    """
    Merkle membership proof for a single receipt.
    
    Attributes:
        leaf_hash: Hash of the leaf (receipt circuit input)
        leaf_index: Index of the leaf in the tree
        siblings: List of sibling hashes along the path to root
        directions: Direction of each sibling (LEFT = sibling on left)
        root: Expected Merkle root
    """
    leaf_hash: bytes
    leaf_index: int
    siblings: List[bytes]
    directions: List[ProofDirection]
    root: bytes
    
    def __post_init__(self):
        """Validate proof structure."""
        if len(self.siblings) != len(self.directions):
            raise MerkleProofError(
                f"Siblings/directions mismatch: {len(self.siblings)} vs {len(self.directions)}"
            )
        if len(self.leaf_hash) != HASH_BYTES:
            raise MerkleProofError(f"Invalid leaf_hash length: {len(self.leaf_hash)}")
        if len(self.root) != HASH_BYTES:
            raise MerkleProofError(f"Invalid root length: {len(self.root)}")
        for i, sib in enumerate(self.siblings):
            if len(sib) != HASH_BYTES:
                raise MerkleProofError(f"Invalid sibling[{i}] length: {len(sib)}")
    
    def verify(self) -> bool:
        """
        Verify this proof computes to the expected root.
        
        Returns:
            True if proof is valid
        """
        current = self.leaf_hash
        
        for sibling, direction in zip(self.siblings, self.directions):
            if direction == ProofDirection.LEFT:
                # Sibling on left: H(sibling || current)
                combined = sibling + current
            else:
                # Sibling on right: H(current || sibling)
                combined = current + sibling
            current = hashlib.sha256(combined).digest()
        
        return current == self.root
    
    def to_bytes(self) -> bytes:
        """Serialize proof to bytes."""
        # Format: leaf_index (u32) + path_len (u8) + leaf_hash + root + [(sibling, direction)]
        path_len = len(self.siblings)
        
        header = struct.pack("<IB", self.leaf_index, path_len)
        data = header + self.leaf_hash + self.root
        
        for sibling, direction in zip(self.siblings, self.directions):
            data += sibling + struct.pack("<B", direction.value)
        
        return data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> MerkleProof:
        """Deserialize proof from bytes."""
        if len(data) < 5 + 2 * HASH_BYTES:
            raise MerkleProofError("Proof data too short")
        
        leaf_index, path_len = struct.unpack("<IB", data[:5])
        offset = 5
        
        leaf_hash = data[offset:offset + HASH_BYTES]
        offset += HASH_BYTES
        
        root = data[offset:offset + HASH_BYTES]
        offset += HASH_BYTES
        
        siblings = []
        directions = []
        
        for _ in range(path_len):
            if offset + HASH_BYTES + 1 > len(data):
                raise MerkleProofError("Incomplete proof data")
            
            sibling = data[offset:offset + HASH_BYTES]
            offset += HASH_BYTES
            
            direction = ProofDirection(struct.unpack("<B", data[offset:offset + 1])[0])
            offset += 1
            
            siblings.append(sibling)
            directions.append(direction)
        
        return cls(
            leaf_hash=leaf_hash,
            leaf_index=leaf_index,
            siblings=siblings,
            directions=directions,
            root=root,
        )
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "leaf_hash": self.leaf_hash.hex(),
            "leaf_index": self.leaf_index,
            "siblings": [s.hex() for s in self.siblings],
            "directions": [d.value for d in self.directions],
            "root": self.root.hex(),
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> MerkleProof:
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(
            leaf_hash=bytes.fromhex(data["leaf_hash"]),
            leaf_index=data["leaf_index"],
            siblings=[bytes.fromhex(s) for s in data["siblings"]],
            directions=[ProofDirection(d) for d in data["directions"]],
            root=bytes.fromhex(data["root"]),
        )


@dataclass
class AccumulatorState:
    """
    Immutable snapshot of accumulator state for commitment.
    
    This is what gets bound to the ZK proof and verified on-chain.
    """
    merkle_root: bytes
    leaf_count: int
    constitution_hash: str
    batch_id: str
    sealed_at: datetime
    
    def commitment(self) -> bytes:
        """
        Compute state commitment hash.
        
        This is the value that gets verified on-chain.
        
        Uses length-prefixed encoding for variable-length fields to ensure
        unambiguous parsing and deterministic commitment across all inputs.
        
        Format:
            - merkle_root: 32 bytes (fixed)
            - leaf_count: 8 bytes (u64 little-endian)
            - constitution_hash_len: 4 bytes (u32 little-endian)
            - constitution_hash: variable UTF-8 bytes
            - batch_id_len: 4 bytes (u32 little-endian)
            - batch_id: variable UTF-8 bytes
            - sealed_at: 8 bytes (u64 Unix timestamp little-endian)
        """
        constitution_bytes = self.constitution_hash.encode("utf-8")
        batch_id_bytes = self.batch_id.encode("utf-8")
        timestamp_value = int(self.sealed_at.timestamp())
        
        data = (
            self.merkle_root +
            struct.pack("<Q", self.leaf_count) +
            struct.pack("<I", len(constitution_bytes)) +
            constitution_bytes +
            struct.pack("<I", len(batch_id_bytes)) +
            batch_id_bytes +
            struct.pack("<Q", timestamp_value)
        )
        return hashlib.sha256(data).digest()
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "merkle_root": self.merkle_root.hex(),
            "leaf_count": self.leaf_count,
            "constitution_hash": self.constitution_hash,
            "batch_id": self.batch_id,
            "sealed_at": self.sealed_at.isoformat(),
        }, indent=2)


@dataclass
class RecursiveProofInput:
    """
    Input package for recursive ZK proof generation.
    
    Used to aggregate multiple receipt proofs into a single batch proof.
    """
    receipts: List[Any]  # List[IhsanReceipt]
    merkle_root: bytes
    constitution_hash: str
    batch_size: int
    ihsan_threshold: int = 950  # Fixed-point
    snr_threshold: int = 750    # Fixed-point
    
    def to_risc0_input(self) -> Dict[str, Any]:
        """
        Format for RiscZero recursive proof API.
        """
        return {
            "batch_input": {
                "merkle_root": self.merkle_root.hex(),
                "constitution_hash": self.constitution_hash,
                "batch_size": self.batch_size,
                "ihsan_threshold": self.ihsan_threshold,
                "snr_threshold": self.snr_threshold,
            },
            "receipts": [
                r.to_circuit_input().hex() for r in self.receipts
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
# ZK ACCUMULATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class ZKAccumulator:
    """
    Zero-Knowledge Proof Accumulator with Merkle Tree.
    
    Accumulates IhsanReceipt instances into a Merkle tree structure,
    enabling:
    - Efficient membership proofs
    - Batch proof aggregation
    - On-chain settlement commitment
    
    Thread-safe for concurrent receipt additions.
    
    Usage:
        accumulator = ZKAccumulator(constitution_hash="abc123...")
        index = accumulator.add(receipt)
        proof = accumulator.generate_proof(index)
        assert accumulator.verify_membership(receipt, proof)
        
        # Seal and prepare for proof generation
        state = accumulator.seal()
        recursive_input = accumulator.prepare_recursive_input()
    """
    
    def __init__(
        self,
        constitution_hash: str,
        batch_id: Optional[str] = None,
        max_size: int = MAX_BATCH_SIZE,
    ):
        """
        Initialize accumulator.
        
        Args:
            constitution_hash: Hash of constitution.toml for binding
            batch_id: Optional batch identifier (auto-generated if None)
            max_size: Maximum number of receipts (default from constitution)
        """
        self._constitution_hash = constitution_hash
        self._batch_id = batch_id or self._generate_batch_id()
        self._max_size = max_size
        
        # Receipt storage
        self._receipts: List[Any] = []  # List[IhsanReceipt]
        self._leaf_hashes: List[bytes] = []
        
        # Merkle tree layers (bottom-up)
        self._tree_layers: List[List[bytes]] = []
        self._tree_dirty = True
        
        # State management
        self._status = AccumulatorStatus.EMPTY
        self._sealed_state: Optional[AccumulatorState] = None
        
        # Thread safety (re-entrant for nested access within class methods)
        self._lock = threading.RLock()
        
        # Callbacks for observability
        self._on_add: Optional[Callable[[int, Any], None]] = None
        self._on_seal: Optional[Callable[[AccumulatorState], None]] = None
    
    def _generate_batch_id(self) -> str:
        """Generate unique batch identifier."""
        import uuid
        return f"batch-{uuid.uuid4().hex[:16]}"
    
    @property
    def constitution_hash(self) -> str:
        """Get constitution hash binding."""
        return self._constitution_hash
    
    @property
    def batch_id(self) -> str:
        """Get batch identifier."""
        return self._batch_id
    
    @property
    def status(self) -> AccumulatorStatus:
        """Get current status."""
        return self._status
    
    @property
    def size(self) -> int:
        """Get number of accumulated receipts."""
        return len(self._receipts)
    
    @property
    def is_sealed(self) -> bool:
        """Check if accumulator is sealed."""
        return self._status in (
            AccumulatorStatus.SEALED,
            AccumulatorStatus.PROVEN,
            AccumulatorStatus.SETTLED,
        )
    
    def _compute_leaf_hash(self, receipt: Any) -> bytes:
        """Compute leaf hash from receipt."""
        circuit_input = receipt.to_circuit_input()
        return hashlib.sha256(circuit_input).digest()
    
    def _rebuild_tree(self):
        """Rebuild Merkle tree from leaf hashes."""
        if not self._tree_dirty:
            return
        
        if not self._leaf_hashes:
            self._tree_layers = []
            self._tree_dirty = False
            return
        
        # Start with leaves
        current_layer = list(self._leaf_hashes)
        self._tree_layers = [current_layer]
        
        # Build tree bottom-up
        while len(current_layer) > 1:
            next_layer = []
            
            # Pad with duplicate if odd
            if len(current_layer) % 2 == 1:
                current_layer.append(current_layer[-1])
            
            # Combine pairs
            for i in range(0, len(current_layer), 2):
                combined = current_layer[i] + current_layer[i + 1]
                next_layer.append(hashlib.sha256(combined).digest())
            
            self._tree_layers.append(next_layer)
            current_layer = next_layer
        
        self._tree_dirty = False
    
    def add(self, receipt: Any) -> int:
        """
        Add a receipt to the accumulator.
        
        Args:
            receipt: IhsanReceipt to add
            
        Returns:
            Index of the receipt in the accumulator
            
        Raises:
            StateError: If accumulator is sealed
            BatchOverflowError: If max size exceeded
        """
        with self._lock:
            if self.is_sealed:
                raise StateError("Cannot add to sealed accumulator")
            
            if len(self._receipts) >= self._max_size:
                raise BatchOverflowError(
                    f"Max batch size {self._max_size} exceeded"
                )
            
            # Compute and store leaf hash
            leaf_hash = self._compute_leaf_hash(receipt)
            
            index = len(self._receipts)
            self._receipts.append(receipt)
            self._leaf_hashes.append(leaf_hash)
            self._tree_dirty = True
            
            # Update status
            if self._status == AccumulatorStatus.EMPTY:
                self._status = AccumulatorStatus.ACCUMULATING
            
            # Callback
            if self._on_add:
                self._on_add(index, receipt)
            
            return index
    
    def get_receipt(self, index: int) -> Any:
        """Get receipt by index."""
        if index < 0 or index >= len(self._receipts):
            raise IndexError(f"Receipt index out of range: {index}")
        return self._receipts[index]
    
    def get_merkle_root(self) -> bytes:
        """
        Get current Merkle root.
        
        Returns:
            32-byte Merkle root, or zero bytes if empty
        """
        with self._lock:
            if not self._leaf_hashes:
                return bytes(HASH_BYTES)
            
            self._rebuild_tree()
            return self._tree_layers[-1][0]
    
    def generate_proof(self, index: int) -> MerkleProof:
        """
        Generate Merkle membership proof for receipt at index.
        
        Args:
            index: Index of the receipt
            
        Returns:
            MerkleProof for the receipt
            
        Raises:
            IndexError: If index out of range
            MerkleProofError: If tree is empty
        """
        with self._lock:
            if not self._leaf_hashes:
                raise MerkleProofError("Cannot generate proof from empty tree")
            
            if index < 0 or index >= len(self._leaf_hashes):
                raise IndexError(f"Index out of range: {index}")
            
            self._rebuild_tree()
            
            siblings: List[bytes] = []
            directions: List[ProofDirection] = []
            
            current_index = index
            
            # Walk up the tree
            for layer in self._tree_layers[:-1]:  # Skip root layer
                # Pad layer if needed for indexing
                padded_layer = list(layer)
                if len(padded_layer) % 2 == 1:
                    padded_layer.append(padded_layer[-1])
                
                # Get sibling
                if current_index % 2 == 0:
                    # We're on left, sibling on right
                    sibling_index = current_index + 1
                    direction = ProofDirection.RIGHT
                else:
                    # We're on right, sibling on left
                    sibling_index = current_index - 1
                    direction = ProofDirection.LEFT
                
                siblings.append(padded_layer[sibling_index])
                directions.append(direction)
                
                # Move up
                current_index //= 2
            
            return MerkleProof(
                leaf_hash=self._leaf_hashes[index],
                leaf_index=index,
                siblings=siblings,
                directions=directions,
                root=self.get_merkle_root(),
            )
    
    def verify_membership(self, receipt: Any, proof: MerkleProof) -> bool:
        """
        Verify receipt membership using proof.
        
        Args:
            receipt: IhsanReceipt to verify
            proof: MerkleProof for the receipt
            
        Returns:
            True if receipt is in the accumulator
        """
        # Compute expected leaf hash
        expected_leaf = self._compute_leaf_hash(receipt)
        
        # Check leaf matches
        if expected_leaf != proof.leaf_hash:
            return False
        
        # Check root matches current root
        if proof.root != self.get_merkle_root():
            return False
        
        # Verify proof path
        return proof.verify()
    
    def seal(self) -> AccumulatorState:
        """
        Seal the accumulator, preventing further additions.
        
        Returns:
            AccumulatorState commitment
            
        Raises:
            StateError: If already sealed or empty
        """
        with self._lock:
            if self.is_sealed:
                raise StateError("Accumulator already sealed")
            
            if not self._receipts:
                raise StateError("Cannot seal empty accumulator")
            
            self._rebuild_tree()
            
            self._sealed_state = AccumulatorState(
                merkle_root=self.get_merkle_root(),
                leaf_count=len(self._receipts),
                constitution_hash=self._constitution_hash,
                batch_id=self._batch_id,
                sealed_at=datetime.now(timezone.utc),
            )
            
            self._status = AccumulatorStatus.SEALED
            
            # Callback
            if self._on_seal:
                self._on_seal(self._sealed_state)
            
            return self._sealed_state
    
    def get_sealed_state(self) -> Optional[AccumulatorState]:
        """Get sealed state if available."""
        return self._sealed_state
    
    def prepare_recursive_input(
        self,
        indices: Optional[List[int]] = None,
        ihsan_threshold: int = 950,
        snr_threshold: int = 750,
    ) -> RecursiveProofInput:
        """
        Prepare input for recursive batch proof generation.
        
        Args:
            indices: Specific receipt indices to include (default: all)
            ihsan_threshold: Fixed-point Ihsān threshold
            snr_threshold: Fixed-point SNR threshold
            
        Returns:
            RecursiveProofInput for ZK prover
            
        Raises:
            StateError: If not sealed
        """
        if not self.is_sealed:
            raise StateError("Accumulator must be sealed before preparing recursive input")
        
        if indices is None:
            receipts = list(self._receipts)
        else:
            receipts = [self._receipts[i] for i in indices]
        
        return RecursiveProofInput(
            receipts=receipts,
            merkle_root=self.get_merkle_root(),
            constitution_hash=self._constitution_hash,
            batch_size=len(receipts),
            ihsan_threshold=ihsan_threshold,
            snr_threshold=snr_threshold,
        )
    
    def mark_proven(self):
        """Mark accumulator as proven."""
        with self._lock:
            if self._status != AccumulatorStatus.SEALED:
                raise StateError("Can only mark proven from SEALED state")
            self._status = AccumulatorStatus.PROVEN
    
    def mark_settled(self):
        """Mark accumulator as settled on-chain."""
        with self._lock:
            if self._status != AccumulatorStatus.PROVEN:
                raise StateError("Can only mark settled from PROVEN state")
            self._status = AccumulatorStatus.SETTLED
    
    def on_add(self, callback: Callable[[int, Any], None]):
        """Register callback for receipt addition."""
        self._on_add = callback
    
    def on_seal(self, callback: Callable[[AccumulatorState], None]):
        """Register callback for seal event."""
        self._on_seal = callback
    
    def to_json(self) -> str:
        """Serialize accumulator state to JSON."""
        return json.dumps({
            "batch_id": self._batch_id,
            "constitution_hash": self._constitution_hash,
            "status": self._status.name,
            "receipt_count": len(self._receipts),
            "merkle_root": self.get_merkle_root().hex() if self._receipts else None,
            "sealed_state": self._sealed_state.to_json() if self._sealed_state else None,
        }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# ACCUMULATOR MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class AccumulatorManager:
    """
    Manages multiple accumulators for rolling batch processing.
    
    Provides automatic batch rotation when max size is reached.
    """
    
    def __init__(
        self,
        constitution_hash: str,
        batch_size: int = MAX_BATCH_SIZE,
    ):
        """
        Initialize manager.
        
        Args:
            constitution_hash: Hash for all accumulators
            batch_size: Max receipts per accumulator
        """
        self._constitution_hash = constitution_hash
        self._batch_size = batch_size
        
        self._current: Optional[ZKAccumulator] = None
        self._sealed: List[ZKAccumulator] = []
        self._lock = threading.Lock()
    
    def _ensure_current(self):
        """Ensure current accumulator exists."""
        if self._current is None:
            self._current = ZKAccumulator(
                constitution_hash=self._constitution_hash,
                max_size=self._batch_size,
            )
    
    def add(self, receipt: Any) -> Tuple[str, int]:
        """
        Add receipt to current accumulator.
        
        Returns:
            Tuple of (batch_id, index)
        """
        with self._lock:
            self._ensure_current()
            
            try:
                index = self._current.add(receipt)
                return self._current.batch_id, index
            except BatchOverflowError:
                # Seal current and create new
                self._sealed.append(self._current)
                self._current.seal()
                
                self._current = ZKAccumulator(
                    constitution_hash=self._constitution_hash,
                    max_size=self._batch_size,
                )
                index = self._current.add(receipt)
                return self._current.batch_id, index
    
    def seal_current(self) -> Optional[AccumulatorState]:
        """Seal current accumulator if it has receipts."""
        with self._lock:
            if self._current is None or self._current.size == 0:
                return None
            
            state = self._current.seal()
            self._sealed.append(self._current)
            self._current = None
            return state
    
    def get_sealed_batches(self) -> List[ZKAccumulator]:
        """Get all sealed accumulators."""
        return list(self._sealed)
    
    @property
    def current_batch_id(self) -> Optional[str]:
        """Get current batch ID."""
        return self._current.batch_id if self._current else None
    
    @property
    def current_size(self) -> int:
        """Get current accumulator size."""
        return self._current.size if self._current else 0
    
    @property
    def total_receipts(self) -> int:
        """Get total receipts across all accumulators."""
        total = sum(acc.size for acc in self._sealed)
        if self._current:
            total += self._current.size
        return total


# ══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "ZKAccumulator",
    "AccumulatorManager",
    "AccumulatorState",
    "AccumulatorStatus",
    "AccumulatorError",
    "MerkleProofError",
    "BatchOverflowError",
    "StateError",
    "MerkleProof",
    "ProofDirection",
    "RecursiveProofInput",
    "MAX_BATCH_SIZE",
    "HASH_BYTES",
]
