"""
BIZRA AEON OMEGA - Comprehensive Integration Test Suite
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Grade | Full Stack Validation

This test suite validates:
1. Layer 1: Blockchain Substrate (Third Fact ledger)
2. Layer 3: State Persistence Engine (DAaaS)
3. Layer 6: Governance Hypervisor (FATE Engine)
4. Lifecycle Emulator (5-phase protocol)
5. Thermodynamic Engine (Carnot/Otto/Stirling cycles)
6. Cross-layer integration

Target: 96.3% production readiness with 542.7 ops/sec cognitive throughput
"""

import asyncio
import pytest
import hashlib
import json
import time
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock


# ============================================================================
# LAYER 1: BLOCKCHAIN SUBSTRATE TESTS
# ============================================================================

class TestBlockchainSubstrate:
    """Tests for Layer 1: Blockchain Substrate."""
    
    @pytest.fixture
    def substrate(self):
        """Create a fresh blockchain substrate."""
        from core.layers.blockchain_substrate import BlockchainSubstrate
        return BlockchainSubstrate(node_id="test_node")
    
    def test_genesis_block_created(self, substrate):
        """Verify genesis block is created on initialization."""
        assert len(substrate.chain) == 1
        genesis = substrate.chain[0]
        assert genesis.index == 0
        assert genesis.previous_hash == b'\x00' * 64
    
    def test_genesis_contains_ihsan_threshold(self, substrate):
        """Verify genesis block has Ihsan threshold."""
        genesis = substrate.chain[0]
        assert len(genesis.transactions) == 1
        tx = genesis.transactions[0]
        assert tx.payload.get("ihsan_threshold") == 0.95
    
    @pytest.mark.asyncio
    async def test_submit_valid_transaction(self, substrate):
        """Test submitting a valid transaction."""
        from core.layers.blockchain_substrate import BlockType
        
        tx_id, accepted, reason = await substrate.submit_transaction(
            BlockType.ATTESTATION,
            "agent_001",
            {"subject": "test"},
            ihsan_score=0.97,
        )
        
        assert accepted
        assert tx_id.startswith("tx_")
        assert len(substrate.pending_transactions) == 1
    
    @pytest.mark.asyncio
    async def test_reject_low_ihsan_transaction(self, substrate):
        """Test that low Ihsan transactions are rejected."""
        from core.layers.blockchain_substrate import BlockType
        
        tx_id, accepted, reason = await substrate.submit_transaction(
            BlockType.ATTESTATION,
            "bad_actor",
            {"subject": "spam"},
            ihsan_score=0.80,  # Below 0.95 threshold
        )
        
        assert not accepted
        assert "Ihsan violation" in reason
    
    @pytest.mark.asyncio
    async def test_create_block(self, substrate):
        """Test block creation from pending transactions."""
        from core.layers.blockchain_substrate import BlockType
        
        # Submit transactions
        await substrate.submit_transaction(
            BlockType.ATTESTATION, "agent_001", {"data": "test1"}, 0.96
        )
        await substrate.submit_transaction(
            BlockType.STATE_UPDATE, "agent_002", {"data": "test2"}, 0.97
        )
        
        # Create block
        block = await substrate.create_block()
        
        assert block is not None
        assert block.index == 1
        assert len(block.transactions) == 2
        assert len(substrate.chain) == 2
    
    @pytest.mark.asyncio
    async def test_block_verification(self, substrate):
        """Test block integrity verification."""
        from core.layers.blockchain_substrate import BlockType
        
        await substrate.submit_transaction(
            BlockType.ATTESTATION, "agent_001", {"data": "test"}, 0.98
        )
        block = await substrate.create_block()
        
        valid, reason = substrate.verify_block(block)
        assert valid
        assert reason == "Block verified"


class TestMerkleTree:
    """Tests for Merkle tree functionality."""
    
    def test_compute_root_empty(self):
        """Test Merkle root of empty list."""
        from core.layers.blockchain_substrate import MerkleTree
        
        root = MerkleTree.compute_root([])
        assert root == hashlib.sha3_256(b"empty").digest()
    
    def test_compute_root_single(self):
        """Test Merkle root of single element."""
        from core.layers.blockchain_substrate import MerkleTree
        
        leaf = hashlib.sha3_256(b"test").digest()
        root = MerkleTree.compute_root([leaf])
        assert root == leaf
    
    def test_merkle_proof_verification(self):
        """Test Merkle proof generation and verification."""
        from core.layers.blockchain_substrate import MerkleTree
        
        leaves = [
            hashlib.sha3_256(b"tx1").digest(),
            hashlib.sha3_256(b"tx2").digest(),
            hashlib.sha3_256(b"tx3").digest(),
            hashlib.sha3_256(b"tx4").digest(),
        ]
        
        tree = MerkleTree(leaves)
        
        # Get proof for leaf 1
        proof = tree.get_proof(1)
        assert len(proof) > 0
        
        # Verify proof
        verified = MerkleTree.verify_proof(leaves[1], proof, tree.root.hash)
        assert verified


class TestIhsanEnforcer:
    """Tests for Ihsan Protocol enforcement."""
    
    def test_enforcer_threshold(self):
        """Verify Ihsan threshold is 0.95."""
        from core.layers.blockchain_substrate import IhsanEnforcer
        
        assert IhsanEnforcer.IHSAN_THRESHOLD == 0.95
    
    def test_valid_transaction_passes(self):
        """Test that valid transactions pass."""
        from core.layers.blockchain_substrate import IhsanEnforcer, Transaction, BlockType
        
        enforcer = IhsanEnforcer()
        tx = Transaction(
            tx_id="test",
            tx_type=BlockType.ATTESTATION,
            sender="agent",
            payload={},
            timestamp=datetime.now(timezone.utc),
            nonce=0,
            ihsan_score=0.96,
        )
        
        valid, reason = enforcer.validate_transaction(tx)
        assert valid
    
    def test_low_ihsan_fails(self):
        """Test that low Ihsan transactions fail."""
        from core.layers.blockchain_substrate import IhsanEnforcer, Transaction, BlockType
        
        enforcer = IhsanEnforcer()
        tx = Transaction(
            tx_id="test",
            tx_type=BlockType.ATTESTATION,
            sender="agent",
            payload={},
            timestamp=datetime.now(timezone.utc),
            nonce=0,
            ihsan_score=0.80,
        )
        
        valid, reason = enforcer.validate_transaction(tx)
        assert not valid
        assert "Ihsan violation" in reason


# ============================================================================
# LAYER 3: STATE PERSISTENCE ENGINE TESTS
# ============================================================================

class TestStatePersistenceEngine:
    """Tests for Layer 3: State Persistence Engine."""
    
    @pytest.fixture
    def engine(self, tmp_path):
        """Create a fresh persistence engine."""
        from core.engine.state_persistence import StatePersistenceEngine
        return StatePersistenceEngine(storage_path=str(tmp_path / "agents"))
    
    @pytest.mark.asyncio
    async def test_register_agent(self, engine):
        """Test agent registration."""
        agent = await engine.register_agent(
            "agent_001",
            initial_stable=1000.0,
            initial_growth=100.0,
        )
        
        assert agent.agent_id == "agent_001"
        assert agent.wallet.stable.amount == 1000.0
        assert agent.wallet.growth.amount == 100.0
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle_transition(self, engine):
        """Test agent lifecycle state transitions."""
        from core.engine.state_persistence import AgentLifecycleState
        
        agent = await engine.register_agent("agent_001")
        
        assert agent.lifecycle_state == AgentLifecycleState.COGNITIVE
        
        # Transition to evolving
        success = agent.transition_to(AgentLifecycleState.EVOLVING)
        assert success
        assert agent.lifecycle_state == AgentLifecycleState.EVOLVING
    
    @pytest.mark.asyncio
    async def test_invalid_transition_fails(self, engine):
        """Test that invalid transitions are rejected."""
        from core.engine.state_persistence import AgentLifecycleState
        
        agent = await engine.register_agent("agent_001")
        
        # Cannot go directly from COGNITIVE to TERMINATED
        success = agent.transition_to(AgentLifecycleState.TERMINATED)
        assert not success
    
    @pytest.mark.asyncio
    async def test_update_agent_state(self, engine):
        """Test updating agent state."""
        agent = await engine.register_agent("agent_001")
        
        updated = await engine.update_agent_state(
            "agent_001",
            cognitive_updates={
                "memory": {"key": "value"},
                "ihsan": {"ikhlas": 0.98},
            },
        )
        
        assert "key" in updated.cognitive.working_memory
        assert updated.cognitive.ihsan_scores["ikhlas"] == 0.98
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation(self, engine):
        """Test checkpoint creation."""
        agent = await engine.register_agent("agent_001")
        
        checkpoint = await engine.create_checkpoint("agent_001", block_height=100)
        
        assert checkpoint.agent_id == "agent_001"
        assert checkpoint.block_height == 100
        assert len(agent.checkpoints) == 1


class TestDualTokenWallet:
    """Tests for dual-token wallet functionality."""
    
    def test_deposit(self):
        """Test token deposit."""
        from core.engine.state_persistence import DualTokenWallet, WalletType
        
        wallet = DualTokenWallet(owner_id="test")
        
        success = wallet.deposit(WalletType.STABLE, 100.0, "test deposit")
        
        assert success
        assert wallet.stable.amount == 100.0
    
    def test_withdraw(self):
        """Test token withdrawal."""
        from core.engine.state_persistence import DualTokenWallet, WalletType
        
        wallet = DualTokenWallet(owner_id="test")
        wallet.deposit(WalletType.STABLE, 100.0)
        
        success = wallet.withdraw(WalletType.STABLE, 50.0)
        
        assert success
        assert wallet.stable.amount == 50.0
    
    def test_insufficient_balance(self):
        """Test withdrawal with insufficient balance."""
        from core.engine.state_persistence import DualTokenWallet, WalletType
        
        wallet = DualTokenWallet(owner_id="test")
        wallet.deposit(WalletType.STABLE, 100.0)
        
        success = wallet.withdraw(WalletType.STABLE, 200.0)
        
        assert not success
        assert wallet.stable.amount == 100.0
    
    def test_staking(self):
        """Test token staking."""
        from core.engine.state_persistence import DualTokenWallet, WalletType
        
        wallet = DualTokenWallet(owner_id="test")
        wallet.deposit(WalletType.GROWTH, 100.0)
        
        success = wallet.stake(WalletType.GROWTH, 50.0)
        
        assert success
        assert wallet.growth.locked == 50.0
        assert wallet.growth.available == 50.0


class TestCognitiveState:
    """Tests for cognitive state management."""
    
    def test_ihsan_update(self):
        """Test Ihsan score update."""
        from core.engine.state_persistence import CognitiveState
        
        cognitive = CognitiveState(agent_id="test")
        
        cognitive.update_ihsan("ikhlas", 0.98)
        
        assert cognitive.ihsan_scores["ikhlas"] == 0.98
        assert cognitive.cumulative_ihsan > 0
    
    def test_ihsan_invalid_score(self):
        """Test that invalid Ihsan scores are rejected."""
        from core.engine.state_persistence import CognitiveState
        
        cognitive = CognitiveState(agent_id="test")
        
        with pytest.raises(ValueError):
            cognitive.update_ihsan("ikhlas", 1.5)
    
    def test_goal_stack(self):
        """Test goal stack operations."""
        from core.engine.state_persistence import CognitiveState
        
        cognitive = CognitiveState(agent_id="test")
        
        cognitive.push_goal({"objective": "test"})
        cognitive.push_goal({"objective": "second"})
        
        assert len(cognitive.goal_stack) == 2
        
        goal = cognitive.pop_goal()
        assert goal["objective"] == "second"


# ============================================================================
# LAYER 6: GOVERNANCE HYPERVISOR TESTS
# ============================================================================

class TestGovernanceHypervisor:
    """Tests for Layer 6: Governance Hypervisor."""
    
    @pytest.fixture
    def hypervisor(self):
        """Create a fresh governance hypervisor."""
        from core.layers.governance_hypervisor import GovernanceHypervisor
        return GovernanceHypervisor(default_voting_period_hours=1)
    
    @pytest.mark.asyncio
    async def test_create_proposal(self, hypervisor):
        """Test proposal creation."""
        from core.layers.governance_hypervisor import ProposalType
        
        prop_id, accepted, reason = await hypervisor.create_proposal(
            proposal_type=ProposalType.PARAMETER_CHANGE,
            title="Test Proposal",
            description="Test description",
            proposer_id="test_proposer",
            payload={"parameter": "test", "value": 123},
        )
        
        assert accepted
        assert prop_id.startswith("prop_")
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_veto(self, hypervisor):
        """Test that low Ihsan proposals are vetoed."""
        from core.layers.governance_hypervisor import ProposalType, IhsanMetrics, ProposalStatus
        
        prop_id, accepted, reason = await hypervisor.create_proposal(
            proposal_type=ProposalType.TREASURY_SPEND,
            title="Bad Proposal",
            description="Low ethics",
            proposer_id="bad_actor",
            payload={},
            ihsan_metrics=IhsanMetrics(ikhlas=0.70, karama=0.70, adl=0.70, kamal=0.70, istidama=0.70),
        )
        
        assert not accepted
        assert "CIRCUIT BREAKER TRIPPED" in reason
        
        proposal = hypervisor.get_proposal(prop_id)
        assert proposal.status == ProposalStatus.VETOED
    
    @pytest.mark.asyncio
    async def test_voting_process(self, hypervisor):
        """Test complete voting process."""
        from core.layers.governance_hypervisor import ProposalType, VoteChoice
        
        # Create proposal
        prop_id, _, _ = await hypervisor.create_proposal(
            proposal_type=ProposalType.PARAMETER_CHANGE,
            title="Voting Test",
            description="Test",
            proposer_id="proposer",
            payload={},
        )
        
        # Start voting
        success, _ = await hypervisor.start_voting(prop_id)
        assert success
        
        # Cast votes
        await hypervisor.cast_vote(prop_id, "voter1", VoteChoice.APPROVE, 100.0)
        await hypervisor.cast_vote(prop_id, "voter2", VoteChoice.APPROVE, 50.0)
        await hypervisor.cast_vote(prop_id, "voter3", VoteChoice.REJECT, 25.0)
        
        # Finalize
        passed, result = await hypervisor.finalize_proposal(prop_id)
        
        assert passed
        assert "PASSED" in result


class TestFATEMetrics:
    """Tests for FATE Engine metrics."""
    
    def test_composite_calculation(self):
        """Test FATE composite score calculation."""
        from core.layers.governance_hypervisor import FATEMetrics
        
        metrics = FATEMetrics(
            fairness=0.95,
            autonomy=0.95,
            transparency=0.95,
            empowerment=0.95,
        )
        
        # Weighted average with given weights
        expected = 0.95 * 0.30 + 0.95 * 0.25 + 0.95 * 0.25 + 0.95 * 0.20
        assert metrics.composite == expected
    
    def test_validation_passes(self):
        """Test FATE validation with good metrics."""
        from core.layers.governance_hypervisor import FATEMetrics
        
        metrics = FATEMetrics(
            fairness=0.96,
            autonomy=0.96,
            transparency=0.96,
            empowerment=0.96,
        )
        
        valid, reason = metrics.validate(0.95)
        assert valid
    
    def test_validation_fails(self):
        """Test FATE validation with low metrics."""
        from core.layers.governance_hypervisor import FATEMetrics
        
        metrics = FATEMetrics(
            fairness=0.80,  # Below threshold
            autonomy=0.96,
            transparency=0.96,
            empowerment=0.96,
        )
        
        valid, reason = metrics.validate(0.95)
        assert not valid
        assert "Fairness violation" in reason


class TestIhsanCircuitBreaker:
    """Tests for Ihsan Circuit Breaker."""
    
    def test_threshold(self):
        """Verify circuit breaker threshold."""
        from core.layers.governance_hypervisor import IhsanCircuitBreaker
        
        assert IhsanCircuitBreaker.THRESHOLD == 0.95
    
    def test_trip_rate_tracking(self):
        """Test trip rate tracking."""
        from core.layers.governance_hypervisor import (
            IhsanCircuitBreaker, Proposal, ProposalType, IhsanMetrics
        )
        
        breaker = IhsanCircuitBreaker()
        
        # Good proposal
        good_proposal = Proposal(
            proposal_id="good",
            proposal_type=ProposalType.PARAMETER_CHANGE,
            title="Good",
            description="",
            proposer_id="",
            ihsan_metrics=IhsanMetrics(ikhlas=0.98, karama=0.98, adl=0.98, kamal=0.98, istidama=0.98),
        )
        
        passes, _ = breaker.check(good_proposal)
        assert passes
        
        # Bad proposal
        bad_proposal = Proposal(
            proposal_id="bad",
            proposal_type=ProposalType.PARAMETER_CHANGE,
            title="Bad",
            description="",
            proposer_id="",
            ihsan_metrics=IhsanMetrics(ikhlas=0.70, karama=0.70, adl=0.70, kamal=0.70, istidama=0.70),
        )
        
        passes, _ = breaker.check(bad_proposal)
        assert not passes
        
        stats = breaker.get_stats()
        assert stats["trip_count"] == 1
        assert stats["total_checks"] == 2


# ============================================================================
# LIFECYCLE EMULATOR TESTS
# ============================================================================

class TestLifecycleEmulator:
    """Tests for Lifecycle Emulation Framework."""
    
    @pytest.fixture
    def emulator(self):
        """Create a fresh lifecycle emulator."""
        from core.lifecycle_emulator import LifecycleEmulator, EmulationMode
        return LifecycleEmulator(mode=EmulationMode.SIMULATION)
    
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, emulator):
        """Test complete lifecycle execution."""
        result = await emulator.run_lifecycle(
            agent_count=5,
            operations_per_phase=20,
        )
        
        assert result is not None
        assert len(result.phases) == 5
        assert result.production_readiness > 0
    
    @pytest.mark.asyncio
    async def test_phase_metrics(self, emulator):
        """Test that phase metrics are collected."""
        from core.lifecycle_emulator import LifecyclePhase
        
        result = await emulator.run_lifecycle(
            agent_count=3,
            operations_per_phase=10,
        )
        
        # Check all phases present
        for phase in LifecyclePhase:
            assert phase in result.phases
            metrics = result.phases[phase]
            assert metrics.operations_count >= 0
    
    def test_target_constants(self, emulator):
        """Verify target constants."""
        from core.lifecycle_emulator import LifecycleEmulator
        
        assert LifecycleEmulator.TARGET_THROUGHPUT == 542.7
        assert LifecycleEmulator.TARGET_P99_LATENCY == 0.0123
        assert LifecycleEmulator.TARGET_READINESS == 0.963


class TestPATSATOrchestrator:
    """Tests for PAT/SAT Token Orchestration."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator."""
        from core.lifecycle_emulator import PATSATOrchestrator
        return PATSATOrchestrator()
    
    @pytest.mark.asyncio
    async def test_mint_pat(self, orchestrator):
        """Test PAT minting."""
        success, msg = await orchestrator.mint_pat(
            "agent_001",
            amount=100.0,
            attestation_proof={"ihsan_score": 0.97},
        )
        
        assert success
        assert orchestrator.stakes["agent_001"]["pat"] == 100.0
    
    @pytest.mark.asyncio
    async def test_mint_pat_low_ihsan_fails(self, orchestrator):
        """Test that low Ihsan PAT minting fails."""
        success, msg = await orchestrator.mint_pat(
            "bad_agent",
            amount=100.0,
            attestation_proof={"ihsan_score": 0.80},
        )
        
        assert not success
        assert "Ihsan violation" in msg
    
    @pytest.mark.asyncio
    async def test_mint_sat_requires_higher_ihsan(self, orchestrator):
        """Test that SAT minting requires higher Ihsan (0.98)."""
        # 0.97 should fail for SAT
        success, msg = await orchestrator.mint_sat(
            "agent_001",
            amount=50.0,
            governance_proof={"ihsan_score": 0.97},
        )
        
        assert not success
        
        # 0.99 should succeed
        success, msg = await orchestrator.mint_sat(
            "agent_002",
            amount=50.0,
            governance_proof={"ihsan_score": 0.99},
        )
        
        assert success
    
    def test_quadratic_voting_power(self, orchestrator):
        """Test quadratic voting power calculation."""
        import math
        
        orchestrator.stakes["agent_001"] = {"pat": 100, "sat": 100, "sat_staked": 100}
        
        power = orchestrator.get_voting_power("agent_001")
        
        assert power == math.sqrt(100)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCrossLayerIntegration:
    """Integration tests across multiple layers."""
    
    @pytest.mark.asyncio
    async def test_blockchain_governance_integration(self):
        """Test integration between Layer 1 and Layer 6."""
        from core.layers.blockchain_substrate import BlockchainSubstrate
        from core.layers.governance_hypervisor import GovernanceHypervisor, ProposalType
        
        blockchain = BlockchainSubstrate()
        governance = GovernanceHypervisor(blockchain=blockchain)
        
        # Create proposal (should record on blockchain)
        prop_id, accepted, _ = await governance.create_proposal(
            proposal_type=ProposalType.PARAMETER_CHANGE,
            title="Integration Test",
            description="Test",
            proposer_id="integrator",
            payload={"test": True},
        )
        
        assert accepted
        
        # Verify pending transaction
        assert len(blockchain.pending_transactions) > 0
    
    @pytest.mark.asyncio
    async def test_persistence_blockchain_integration(self):
        """Test integration between Layer 3 and Layer 1."""
        from core.layers.blockchain_substrate import BlockchainSubstrate
        from core.engine.state_persistence import StatePersistenceEngine
        
        blockchain = BlockchainSubstrate()
        persistence = StatePersistenceEngine(blockchain=blockchain)
        
        # Register agent (should record on blockchain)
        agent = await persistence.register_agent("integration_agent")
        
        assert agent is not None
        assert len(blockchain.pending_transactions) > 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
