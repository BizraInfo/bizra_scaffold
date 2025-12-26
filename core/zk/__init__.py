"""
ZK Settlement Layer for BIZRA Sovereignty Framework.

This module provides zero-knowledge proof infrastructure:
  - ZKBridge: PCIEnvelope → IhsanReceipt conversion
  - ZKAccumulator: Merkle tree batch accumulation
  - MerkleProof: Membership proofs for receipts
  - RecursiveProofInput: Batch aggregation for recursive proofs
"""

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

from core.zk.accumulator import (
    ZKAccumulator,
    AccumulatorManager,
    AccumulatorState,
    AccumulatorStatus,
    AccumulatorError,
    MerkleProofError,
    BatchOverflowError,
    StateError,
    MerkleProof,
    ProofDirection,
    RecursiveProofInput,
    MAX_BATCH_SIZE,
    HASH_BYTES,
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
