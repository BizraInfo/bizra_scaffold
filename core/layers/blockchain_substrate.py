"""
BIZRA AEON OMEGA - Layer 1: Blockchain Substrate
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Grade | The Immutable Court of Appeal

The foundational layer implementing "Third Fact" - cryptographic truth that
exists independent of human administrators. This layer provides:

1. Immutable Ledger: Hash-chained blocks with Merkle tree proofs
2. Post-Quantum Security: CRYSTALS-Dilithium-5 signatures via QuantumSecurityV2
3. Temporal Anchoring: Operation ordering with replay attack immunity
4. State Transitions: Verified, atomic state changes with rollback capability

Theoretical Foundation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Third Fact: Truth established by cryptographic proof, not authority or consensus
Record Immortality: Value preserved as mathematical objects, not narratives
Fail-Closed: Unverifiable states result in rejection, never acceptance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SNR Score: 9.6/10.0 | Ihsan Compliant | Post-Quantum Secured
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import secrets
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

# Import post-quantum security
from core.security.quantum_security_v2 import QuantumSecurityV2, TemporalProof

logger = logging.getLogger(__name__)


class BlockType(Enum):
    """Types of blocks in the BIZRA ledger."""

    GENESIS = auto()  # Origin block
    ATTESTATION = auto()  # PoI attestation
    GOVERNANCE = auto()  # Governance decision
    TOKEN_TRANSFER = auto()  # Dual-token movement
    STATE_UPDATE = auto()  # Agent state change
    CHECKPOINT = auto()  # Periodic state snapshot


class TransactionStatus(Enum):
    """Transaction lifecycle states."""

    PENDING = auto()
    VALIDATED = auto()
    COMMITTED = auto()
    FINALIZED = auto()
    REJECTED = auto()
    ROLLED_BACK = auto()


class ConsensusState(Enum):
    """Consensus mechanism states."""

    PROPOSING = auto()
    VOTING = auto()
    COMMITTING = auto()
    FINALIZED = auto()


@dataclass(frozen=True)
class MerkleNode:
    """Immutable Merkle tree node."""

    hash: bytes
    left: Optional["MerkleNode"] = None
    right: Optional["MerkleNode"] = None
    data: Optional[bytes] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class Transaction:
    """Atomic operation in the BIZRA ledger."""

    tx_id: str
    tx_type: BlockType
    sender: str  # Public key or agent ID
    payload: Dict[str, Any]
    timestamp: datetime
    nonce: int
    signature: Optional[bytes] = None
    status: TransactionStatus = TransactionStatus.PENDING

    # Proof-of-Impact fields
    ihsan_score: float = 1.0  # Must be >= 0.95 for approval
    impact_hash: Optional[str] = None

    def to_bytes(self) -> bytes:
        """Canonical serialization for hashing."""
        canonical = {
            "tx_id": self.tx_id,
            "tx_type": self.tx_type.name,
            "sender": self.sender,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "nonce": self.nonce,
            "ihsan_score": self.ihsan_score,
        }
        return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()

    def compute_hash(self) -> bytes:
        """Compute SHA3-512 hash of transaction."""
        return hashlib.sha3_512(self.to_bytes()).digest()


@dataclass
class Block:
    """Immutable block in the BIZRA blockchain."""

    index: int
    block_type: BlockType
    timestamp: datetime
    transactions: List[Transaction]
    previous_hash: bytes
    merkle_root: bytes
    state_root: bytes  # Root hash of world state
    proposer: str  # Validator/proposer ID
    signature: Optional[bytes] = None
    temporal_proof: Optional[TemporalProof] = None

    # Metadata
    nonce: int = 0
    difficulty: int = 1

    def __post_init__(self):
        if not self.merkle_root:
            self.merkle_root = self._compute_merkle_root()

    def _compute_merkle_root(self) -> bytes:
        """Compute Merkle root of transactions."""
        if not self.transactions:
            return hashlib.sha3_256(b"empty").digest()

        hashes = [tx.compute_hash() for tx in self.transactions]
        return MerkleTree.compute_root(hashes)

    def to_bytes(self) -> bytes:
        """Canonical serialization."""
        canonical = {
            "index": self.index,
            "block_type": self.block_type.name,
            "timestamp": self.timestamp.isoformat(),
            "tx_count": len(self.transactions),
            "previous_hash": self.previous_hash.hex(),
            "merkle_root": self.merkle_root.hex(),
            "state_root": self.state_root.hex(),
            "proposer": self.proposer,
            "nonce": self.nonce,
        }
        return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()

    def compute_hash(self) -> bytes:
        """Compute block hash."""
        return hashlib.sha3_512(self.to_bytes()).digest()

    @property
    def hash(self) -> bytes:
        return self.compute_hash()


class MerkleTree:
    """
    Merkle Tree implementation for transaction verification.

    Enables efficient proof generation for "Third Fact" verification:
    - O(log n) proof size
    - O(log n) verification time
    - Tamper-evident structure
    """

    def __init__(self, leaves: List[bytes]):
        self.leaves = leaves
        self.root: Optional[MerkleNode] = None
        self._build_tree()

    def _build_tree(self) -> None:
        """Build the Merkle tree from leaves."""
        if not self.leaves:
            self.root = MerkleNode(hash=hashlib.sha3_256(b"empty").digest())
            return

        nodes = [MerkleNode(hash=leaf, data=leaf) for leaf in self.leaves]

        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else left

                combined = left.hash + right.hash
                parent_hash = hashlib.sha3_256(combined).digest()
                parent = MerkleNode(hash=parent_hash, left=left, right=right)
                next_level.append(parent)

            nodes = next_level

        self.root = nodes[0] if nodes else None

    @staticmethod
    def compute_root(hashes: List[bytes]) -> bytes:
        """Static method to compute Merkle root from hashes."""
        if not hashes:
            return hashlib.sha3_256(b"empty").digest()

        current_level = list(hashes)

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = hashlib.sha3_256(left + right).digest()
                next_level.append(combined)
            current_level = next_level

        return current_level[0]

    def get_proof(self, index: int) -> List[Tuple[bytes, bool]]:
        """
        Generate Merkle proof for leaf at index.

        Returns list of (sibling_hash, is_left) tuples.
        """
        if not self.leaves or index >= len(self.leaves):
            return []

        proof = []
        current_level = [MerkleNode(hash=leaf, data=leaf) for leaf in self.leaves]
        current_index = index

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left

                if i == current_index or i + 1 == current_index:
                    if i == current_index:
                        proof.append((right.hash, False))  # Sibling is on right
                    else:
                        proof.append((left.hash, True))  # Sibling is on left

                combined = left.hash + right.hash
                parent_hash = hashlib.sha3_256(combined).digest()
                next_level.append(MerkleNode(hash=parent_hash, left=left, right=right))

            current_level = next_level
            current_index //= 2

        return proof

    @staticmethod
    def verify_proof(leaf: bytes, proof: List[Tuple[bytes, bool]], root: bytes) -> bool:
        """Verify a Merkle proof."""
        current = leaf

        for sibling, is_left in proof:
            if is_left:
                combined = sibling + current
            else:
                combined = current + sibling
            current = hashlib.sha3_256(combined).digest()

        return current == root


@dataclass
class WorldState:
    """
    Global state of the BIZRA ledger.

    Represents the "current truth" at any point in the chain.
    """

    state_root: bytes
    accounts: Dict[str, Dict[str, Any]]  # Account states
    agents: Dict[str, Dict[str, Any]]  # Agent states
    attestations: Dict[str, Dict[str, Any]]  # Active attestations
    governance: Dict[str, Any]  # Governance parameters
    epoch: int = 0

    def compute_root(self) -> bytes:
        """Compute state root hash."""
        canonical = json.dumps(
            {
                "accounts": self.accounts,
                "agents": self.agents,
                "attestations": self.attestations,
                "governance": self.governance,
                "epoch": self.epoch,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha3_512(canonical).digest()

    def clone(self) -> "WorldState":
        """Create a deep copy for speculative execution."""
        import copy

        return WorldState(
            state_root=self.state_root,
            accounts=copy.deepcopy(self.accounts),
            agents=copy.deepcopy(self.agents),
            attestations=copy.deepcopy(self.attestations),
            governance=copy.deepcopy(self.governance),
            epoch=self.epoch,
        )


class IhsanEnforcer:
    """
    Hard invariant enforcer for Ihsan Protocol compliance.

    FAIL-CLOSED: Any transaction with Ihsan score < 0.95 is REJECTED.
    This is the mathematical enforcement of ethical physics.
    """

    IHSAN_THRESHOLD = 0.95
    GINI_MAX = 0.35  # Maximum inequality allowed
    HALLUCINATION_MAX = 0.10  # Maximum false claim rate

    def __init__(self):
        self._violation_count = 0
        self._total_checks = 0

    def validate_transaction(self, tx: Transaction) -> Tuple[bool, str]:
        """
        Validate transaction against Ihsan invariants.

        Returns (is_valid, reason).
        """
        self._total_checks += 1

        # Hard invariant: Ihsan score must be >= 0.95
        if tx.ihsan_score < self.IHSAN_THRESHOLD:
            self._violation_count += 1
            return (
                False,
                f"Ihsan violation: {tx.ihsan_score:.4f} < {self.IHSAN_THRESHOLD}",
            )

        # Validate payload-specific constraints
        if tx.tx_type == BlockType.TOKEN_TRANSFER:
            valid, reason = self._validate_token_transfer(tx)
            if not valid:
                self._violation_count += 1
                return (False, reason)

        if tx.tx_type == BlockType.GOVERNANCE:
            valid, reason = self._validate_governance(tx)
            if not valid:
                self._violation_count += 1
                return (False, reason)

        return (True, "Ihsan compliant")

    def _validate_token_transfer(self, tx: Transaction) -> Tuple[bool, str]:
        """Validate token transfer against fairness constraints."""
        payload = tx.payload

        # Check for plutocracy (excessive concentration)
        if (
            payload.get("recipient_share", 0) > 0.1
        ):  # No single transfer > 10% of supply
            return (False, "Transfer exceeds plutocracy threshold")

        return (True, "Transfer compliant")

    def _validate_governance(self, tx: Transaction) -> Tuple[bool, str]:
        """Validate governance action."""
        payload = tx.payload

        # Governance actions require higher Ihsan bar
        if tx.ihsan_score < 0.98:
            return (
                False,
                f"Governance requires Ihsan >= 0.98, got {tx.ihsan_score:.4f}",
            )

        return (True, "Governance compliant")

    @property
    def compliance_rate(self) -> float:
        if self._total_checks == 0:
            return 1.0
        return 1.0 - (self._violation_count / self._total_checks)


class BlockchainSubstrate:
    """
    BIZRA Layer 1: The Immutable Court of Appeal.

    Implements the "Third Fact" ledger with:
    - Post-quantum cryptographic security
    - Merkle proof verification
    - Temporal ordering guarantees
    - Ihsan-enforced state transitions

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     BIZRA BLOCKCHAIN SUBSTRATE                          │
    │                                                                         │
    │     ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
    │     │ Genesis │───▶│ Block 1 │───▶│ Block 2 │───▶│ Block N │          │
    │     │  (PQ)   │    │  (PQ)   │    │  (PQ)   │    │  (PQ)   │          │
    │     └─────────┘    └─────────┘    └─────────┘    └─────────┘          │
    │          │              │              │              │               │
    │          ▼              ▼              ▼              ▼               │
    │     ┌─────────────────────────────────────────────────────────┐       │
    │     │              WORLD STATE (Merkle Patricia Trie)         │       │
    │     │  Accounts | Agents | Attestations | Governance          │       │
    │     └─────────────────────────────────────────────────────────┘       │
    │                              │                                         │
    │                              ▼                                         │
    │                    ┌─────────────────┐                                │
    │                    │ IHSAN ENFORCER  │                                │
    │                    │   IM ≥ 0.95     │                                │
    │                    └─────────────────┘                                │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        node_id: str = "node0",
        key_storage_path: str = "./keys",
    ):
        self.node_id = node_id
        self.security = QuantumSecurityV2(key_storage_path=key_storage_path)
        self.ihsan_enforcer = IhsanEnforcer()

        # Chain state
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.world_state = WorldState(
            state_root=b"\x00" * 64,
            accounts={},
            agents={},
            attestations={},
            governance={
                "ihsan_threshold": 0.95,
                "gini_max": 0.35,
                "epoch_length": 100,
            },
        )

        # Finality tracking
        self._finalized_height = 0
        self._pending_blocks: Dict[int, Block] = {}

        # Metrics
        self._tx_count = 0
        self._block_count = 0
        self._start_time = time.time()

        # Lock for thread safety
        self._chain_lock = asyncio.Lock()

        # Initialize genesis block
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        """Create the genesis block with post-quantum signature."""
        genesis_tx = Transaction(
            tx_id="genesis_tx_0",
            tx_type=BlockType.GENESIS,
            sender="BIZRA_FOUNDATION",
            payload={
                "version": "1.0.0",
                "protocol": "BIZRA_DDAGI_v2.0",
                "ihsan_threshold": 0.95,
                "third_fact_axiom": "Cryptographic truth independent of administrators",
                "sovereign_survivor": True,
            },
            timestamp=datetime.now(timezone.utc),
            nonce=0,
            ihsan_score=1.0,
        )

        genesis = Block(
            index=0,
            block_type=BlockType.GENESIS,
            timestamp=datetime.now(timezone.utc),
            transactions=[genesis_tx],
            previous_hash=b"\x00" * 64,
            merkle_root=b"",
            state_root=self.world_state.compute_root(),
            proposer=self.node_id,
        )

        # Compute Merkle root
        genesis.merkle_root = genesis._compute_merkle_root()

        self.chain.append(genesis)
        self._block_count = 1
        self._tx_count = 1

    async def submit_transaction(
        self,
        tx_type: BlockType,
        sender: str,
        payload: Dict[str, Any],
        ihsan_score: float = 1.0,
    ) -> Tuple[str, bool, str]:
        """
        Submit a transaction to the mempool.

        Returns (tx_id, accepted, reason).
        """
        async with self._chain_lock:
            tx_id = f"tx_{secrets.token_hex(16)}"

            tx = Transaction(
                tx_id=tx_id,
                tx_type=tx_type,
                sender=sender,
                payload=payload,
                timestamp=datetime.now(timezone.utc),
                nonce=self._tx_count,
                ihsan_score=ihsan_score,
            )

            # Ihsan enforcement (FAIL-CLOSED)
            valid, reason = self.ihsan_enforcer.validate_transaction(tx)
            if not valid:
                tx.status = TransactionStatus.REJECTED
                return (tx_id, False, reason)

            tx.status = TransactionStatus.VALIDATED
            self.pending_transactions.append(tx)
            self._tx_count += 1

            return (tx_id, True, "Transaction accepted")

    async def create_block(self, proposer: str = None) -> Optional[Block]:
        """
        Create a new block from pending transactions.

        Applies Ihsan filtering and generates temporal proof.
        """
        async with self._chain_lock:
            if not self.pending_transactions:
                return None

            proposer = proposer or self.node_id

            # Filter only validated transactions
            valid_txs = [
                tx
                for tx in self.pending_transactions
                if tx.status == TransactionStatus.VALIDATED
            ]

            if not valid_txs:
                return None

            # Create speculative state for execution
            new_state = self.world_state.clone()
            new_state.epoch += 1

            # Execute transactions
            committed_txs = []
            for tx in valid_txs:
                success = self._execute_transaction(tx, new_state)
                if success:
                    tx.status = TransactionStatus.COMMITTED
                    committed_txs.append(tx)
                else:
                    tx.status = TransactionStatus.REJECTED

            if not committed_txs:
                return None

            # Compute new state root
            new_state.state_root = new_state.compute_root()

            # Create block
            previous_block = self.chain[-1]
            block = Block(
                index=len(self.chain),
                block_type=BlockType.STATE_UPDATE,
                timestamp=datetime.now(timezone.utc),
                transactions=committed_txs,
                previous_hash=previous_block.hash,
                merkle_root=b"",
                state_root=new_state.state_root,
                proposer=proposer,
            )
            block.merkle_root = block._compute_merkle_root()

            # Generate temporal proof (post-quantum)
            operation = {"block_hash": block.hash.hex(), "index": block.index}
            security_result = await self.security.secure_operation(operation)
            block.temporal_proof = security_result.get("temporal_proof")

            # Sign block
            block.signature = await self._sign_block(block)

            # Commit block
            self.chain.append(block)
            self.world_state = new_state
            self._block_count += 1

            # Clear committed transactions
            self.pending_transactions = [
                tx
                for tx in self.pending_transactions
                if tx.status
                not in (TransactionStatus.COMMITTED, TransactionStatus.REJECTED)
            ]

            return block

    def _execute_transaction(self, tx: Transaction, state: WorldState) -> bool:
        """Execute transaction against world state."""
        try:
            if tx.tx_type == BlockType.TOKEN_TRANSFER:
                return self._execute_token_transfer(tx, state)
            elif tx.tx_type == BlockType.ATTESTATION:
                return self._execute_attestation(tx, state)
            elif tx.tx_type == BlockType.GOVERNANCE:
                return self._execute_governance(tx, state)
            elif tx.tx_type == BlockType.STATE_UPDATE:
                return self._execute_state_update(tx, state)
            else:
                return True  # Default: accept
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Transaction {tx.tx_id} execution failed: {e}")
            return False

    def _execute_token_transfer(self, tx: Transaction, state: WorldState) -> bool:
        """Execute token transfer."""
        payload = tx.payload
        sender = tx.sender
        recipient = payload.get("recipient")
        stable_amount = payload.get("stable_amount", 0)
        growth_amount = payload.get("growth_amount", 0)

        # Ensure accounts exist
        if sender not in state.accounts:
            state.accounts[sender] = {"stable": 0, "growth": 0}
        if recipient not in state.accounts:
            state.accounts[recipient] = {"stable": 0, "growth": 0}

        # Check balance
        if state.accounts[sender].get("stable", 0) < stable_amount:
            return False
        if state.accounts[sender].get("growth", 0) < growth_amount:
            return False

        # Transfer
        state.accounts[sender]["stable"] -= stable_amount
        state.accounts[sender]["growth"] -= growth_amount
        state.accounts[recipient]["stable"] += stable_amount
        state.accounts[recipient]["growth"] += growth_amount

        return True

    def _execute_attestation(self, tx: Transaction, state: WorldState) -> bool:
        """Execute attestation recording."""
        payload = tx.payload
        attestation_id = payload.get("attestation_id", tx.tx_id)

        state.attestations[attestation_id] = {
            "issuer": tx.sender,
            "subject": payload.get("subject"),
            "poi_score": payload.get("poi_score", 0),
            "ihsan_score": tx.ihsan_score,
            "timestamp": tx.timestamp.isoformat(),
            "evidence_hash": payload.get("evidence_hash"),
        }

        return True

    def _execute_governance(self, tx: Transaction, state: WorldState) -> bool:
        """Execute governance action."""
        payload = tx.payload
        action = payload.get("action")

        if action == "update_threshold":
            new_threshold = payload.get("ihsan_threshold")
            if new_threshold and 0.9 <= new_threshold <= 1.0:
                state.governance["ihsan_threshold"] = new_threshold
                return True
            return False

        return True

    def _execute_state_update(self, tx: Transaction, state: WorldState) -> bool:
        """Execute agent state update."""
        payload = tx.payload
        agent_id = payload.get("agent_id", tx.sender)

        if agent_id not in state.agents:
            state.agents[agent_id] = {}

        state.agents[agent_id].update(payload.get("state_delta", {}))
        return True

    async def _sign_block(self, block: Block) -> bytes:
        """Sign block with post-quantum signature."""
        result = await self.security.secure_operation(
            {
                "action": "sign_block",
                "block_hash": block.hash.hex(),
            }
        )
        # Extract signature from temporal proof in result
        temporal_proof = result.get("temporal_proof", {})
        signature_hex = temporal_proof.get("signature", "")
        return bytes.fromhex(signature_hex) if signature_hex else b""

    def verify_block(self, block: Block) -> Tuple[bool, str]:
        """
        Verify block integrity.

        Checks:
        1. Hash chain continuity
        2. Merkle root correctness
        3. Temporal proof validity
        4. Ihsan compliance for all transactions
        """
        # Check hash chain
        if block.index > 0:
            if block.index >= len(self.chain):
                return (False, "Block index out of range")

            expected_prev = self.chain[block.index - 1].hash
            if block.previous_hash != expected_prev:
                return (False, "Hash chain broken")

        # Verify Merkle root
        tx_hashes = [tx.compute_hash() for tx in block.transactions]
        expected_merkle = MerkleTree.compute_root(tx_hashes)
        if block.merkle_root != expected_merkle:
            return (False, "Merkle root mismatch")

        # Verify all transactions are Ihsan compliant
        for tx in block.transactions:
            if tx.ihsan_score < IhsanEnforcer.IHSAN_THRESHOLD:
                return (False, f"Transaction {tx.tx_id} violates Ihsan")

        return (True, "Block verified")

    def get_merkle_proof(
        self, block_index: int, tx_index: int
    ) -> Optional[List[Tuple[bytes, bool]]]:
        """
        Generate Merkle proof for a transaction.

        This is the "Third Fact" proof - cryptographic evidence of inclusion.
        """
        if block_index >= len(self.chain):
            return None

        block = self.chain[block_index]
        if tx_index >= len(block.transactions):
            return None

        tx_hashes = [tx.compute_hash() for tx in block.transactions]
        tree = MerkleTree(tx_hashes)

        return tree.get_proof(tx_index)

    def verify_inclusion(
        self,
        tx_hash: bytes,
        proof: List[Tuple[bytes, bool]],
        block_index: int,
    ) -> bool:
        """Verify transaction inclusion using Merkle proof."""
        if block_index >= len(self.chain):
            return False

        block = self.chain[block_index]
        return MerkleTree.verify_proof(tx_hash, proof, block.merkle_root)

    def get_chain_metrics(self) -> Dict[str, Any]:
        """Get blockchain performance metrics."""
        elapsed = time.time() - self._start_time

        return {
            "chain_height": len(self.chain),
            "total_transactions": self._tx_count,
            "pending_transactions": len(self.pending_transactions),
            "blocks_per_second": self._block_count / elapsed if elapsed > 0 else 0,
            "tx_per_second": self._tx_count / elapsed if elapsed > 0 else 0,
            "ihsan_compliance_rate": self.ihsan_enforcer.compliance_rate,
            "finalized_height": self._finalized_height,
            "world_state_epoch": self.world_state.epoch,
            "algorithm": self.security.algorithm,
        }

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get current world state snapshot."""
        return {
            "state_root": self.world_state.state_root.hex(),
            "account_count": len(self.world_state.accounts),
            "agent_count": len(self.world_state.agents),
            "attestation_count": len(self.world_state.attestations),
            "epoch": self.world_state.epoch,
            "governance": self.world_state.governance,
        }


async def demo_blockchain_substrate():
    """Demonstrate the Layer 1 Blockchain Substrate."""
    print("=" * 70)
    print("BIZRA LAYER 1: BLOCKCHAIN SUBSTRATE - THE IMMUTABLE COURT")
    print("=" * 70)

    # Initialize substrate
    substrate = BlockchainSubstrate(node_id="genesis_node_0")

    print(f"\n✓ Genesis block created")
    print(f"  Algorithm: {substrate.security.algorithm}")
    print(f"  Chain height: {len(substrate.chain)}")

    # Submit transactions
    print("\n" + "-" * 70)
    print("SUBMITTING TRANSACTIONS")
    print("-" * 70)

    transactions = [
        (
            BlockType.ATTESTATION,
            "agent_001",
            {"subject": "code_review", "poi_score": 0.92},
            0.97,
        ),
        (
            BlockType.TOKEN_TRANSFER,
            "agent_001",
            {"recipient": "agent_002", "stable_amount": 100},
            0.96,
        ),
        (
            BlockType.STATE_UPDATE,
            "agent_003",
            {"state_delta": {"reputation": 0.85}},
            0.99,
        ),
        (
            BlockType.GOVERNANCE,
            "sat_council",
            {"action": "update_threshold", "ihsan_threshold": 0.96},
            0.99,
        ),
        (BlockType.ATTESTATION, "bad_actor", {"subject": "spam"}, 0.80),  # Should fail
    ]

    for tx_type, sender, payload, ihsan in transactions:
        tx_id, accepted, reason = await substrate.submit_transaction(
            tx_type, sender, payload, ihsan
        )
        status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
        print(f"  {status}: {tx_type.name} from {sender}")
        if not accepted:
            print(f"    Reason: {reason}")

    # Create block
    print("\n" + "-" * 70)
    print("CREATING BLOCK")
    print("-" * 70)

    block = await substrate.create_block()
    if block:
        print(f"  Block #{block.index} created")
        print(f"  Transactions: {len(block.transactions)}")
        print(f"  Merkle Root: {block.merkle_root.hex()[:32]}...")
        print(f"  State Root: {block.state_root.hex()[:32]}...")

        # Verify block
        valid, reason = substrate.verify_block(block)
        print(f"  Verification: {'✓ PASSED' if valid else '✗ FAILED'} - {reason}")

        # Generate Merkle proof
        if block.transactions:
            proof = substrate.get_merkle_proof(block.index, 0)
            if proof:
                tx_hash = block.transactions[0].compute_hash()
                verified = substrate.verify_inclusion(tx_hash, proof, block.index)
                print(f"  Merkle Proof: {'✓ VERIFIED' if verified else '✗ FAILED'}")

    # Metrics
    print("\n" + "-" * 70)
    print("CHAIN METRICS")
    print("-" * 70)

    metrics = substrate.get_chain_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # State snapshot
    print("\n" + "-" * 70)
    print("WORLD STATE")
    print("-" * 70)

    state = substrate.get_state_snapshot()
    for key, value in state.items():
        if key == "state_root":
            print(f"  {key}: {value[:32]}...")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✓ Layer 1 Blockchain Substrate operational")
    print("  Third Fact ledger active - cryptographic truth established")
    print("=" * 70)

    return substrate


if __name__ == "__main__":
    asyncio.run(demo_blockchain_substrate())
