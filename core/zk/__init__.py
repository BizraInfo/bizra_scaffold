"""
ZK Settlement Layer for BIZRA Sovereignty Framework.

This module provides zero-knowledge proof infrastructure:
  - ZKBridge: PCIEnvelope → IhsanReceipt conversion
  - ZKAccumulator: Merkle tree batch accumulation
  - MerkleProof: Membership proofs for receipts
  - RecursiveProofInput: Batch aggregation for recursive proofs
"""

from core.zk.accumulator import (
    HASH_BYTES,
    MAX_BATCH_SIZE,
    AccumulatorError,
    AccumulatorManager,
    AccumulatorState,
    AccumulatorStatus,
    BatchOverflowError,
    MerkleProof,
    MerkleProofError,
    ProofDirection,
    RecursiveProofInput,
    StateError,
    ZKAccumulator,
)
from core.zk.bridge import (
    FIXED_POINT_SCALE,
    BatchConverter,
    ConversionError,
    IhsanReceipt,
    ProofInput,
    SerializationError,
    ValidationError,
    ZKBridge,
    ZKBridgeError,
)

__all__ = [
    # Bridge exports
    "ZKBridge",
    "ZKBridgeError",
    "ConversionError",
    "SerializationError",
    "ValidationError",
    "IhsanReceipt",
    "ProofInput",
    "BatchConverter",
    "FIXED_POINT_SCALE",
    # Accumulator exports
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
