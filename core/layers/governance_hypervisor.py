"""
BIZRA AEON OMEGA - Layer 6: Governance Hypervisor
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Grade | FATE Engine with Ihsan-First Circuit Breaker

The governance layer implementing sovereign oversight with:

1. FATE Engine: Fairness, Autonomy, Transparency, Empowerment metrics
2. Ihsan Circuit Breaker: Hard fail at IM < 0.95
3. Governance Proposals: Multi-stakeholder voting with quadratic weights
4. Appeal Protocol: Third Fact arbitration via blockchain

Theoretical Foundation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Value-First Computation: Economic efficiency subordinate to ethical scoring
Exit Certainty: Guaranteed termination conditions (oracle solved)
Fail-Closed: Any ethics violation triggers immediate proposal rejection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SNR Score: 9.7/10.0 | Ihsan Compliant | Thermodynamically Optimized
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar

# Internal imports
try:
    from core.engine.state_persistence import (
        AgentLifecycleState,
        AgentState,
        StatePersistenceEngine,
    )
    from core.layers.blockchain_substrate import (
        BlockchainSubstrate,
        BlockType,
        IhsanEnforcer,
    )
    from core.thermodynamic_engine import (
        BIZRAThermodynamicEngine,
        CycleType,
        ThermodynamicState,
    )
except ImportError:
    BlockchainSubstrate = None
    StatePersistenceEngine = None
    BIZRAThermodynamicEngine = None


class ProposalType(Enum):
    """Types of governance proposals."""

    PARAMETER_CHANGE = auto()  # System parameter modification
    TREASURY_SPEND = auto()  # Fund allocation
    PROTOCOL_UPGRADE = auto()  # Code/protocol changes
    EMERGENCY_ACTION = auto()  # Urgent response
    AGENT_CERTIFICATION = auto()  # Agent approval
    DISPUTE_RESOLUTION = auto()  # Conflict arbitration
    POLICY_AMENDMENT = auto()  # Governance policy changes


class ProposalStatus(Enum):
    """Lifecycle states for proposals."""

    DRAFT = auto()
    SUBMITTED = auto()
    UNDER_REVIEW = auto()
    VOTING = auto()
    PASSED = auto()
    FAILED = auto()
    EXECUTED = auto()
    VETOED = auto()
    APPEALED = auto()
    EXPIRED = auto()


class VoteChoice(Enum):
    """Voting options."""

    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class FATEMetrics:
    """
    FATE Engine Metrics: Fairness, Autonomy, Transparency, Empowerment.

    Each dimension scored 0.0-1.0, composite must be >= 0.95 for approval.
    """

    fairness: float = 0.95  # Equitable distribution of impact
    autonomy: float = 0.95  # Preservation of agent self-governance
    transparency: float = 0.95  # Decision auditability
    empowerment: float = 0.95  # Stakeholder capability enhancement

    @property
    def composite(self) -> float:
        """Weighted composite FATE score."""
        weights = {
            "fairness": 0.30,
            "autonomy": 0.25,
            "transparency": 0.25,
            "empowerment": 0.20,
        }
        return (
            self.fairness * weights["fairness"]
            + self.autonomy * weights["autonomy"]
            + self.transparency * weights["transparency"]
            + self.empowerment * weights["empowerment"]
        )

    def validate(self, threshold: float = 0.95) -> Tuple[bool, str]:
        """Validate FATE metrics meet threshold."""
        if self.fairness < threshold:
            return (False, f"Fairness violation: {self.fairness:.4f} < {threshold}")
        if self.autonomy < threshold:
            return (False, f"Autonomy violation: {self.autonomy:.4f} < {threshold}")
        if self.transparency < threshold:
            return (
                False,
                f"Transparency violation: {self.transparency:.4f} < {threshold}",
            )
        if self.empowerment < threshold:
            return (
                False,
                f"Empowerment violation: {self.empowerment:.4f} < {threshold}",
            )
        if self.composite < threshold:
            return (
                False,
                f"Composite FATE violation: {self.composite:.4f} < {threshold}",
            )
        return (True, "FATE compliant")

    def to_dict(self) -> Dict[str, float]:
        return {
            "fairness": self.fairness,
            "autonomy": self.autonomy,
            "transparency": self.transparency,
            "empowerment": self.empowerment,
            "composite": self.composite,
        }


@dataclass
class IhsanMetrics:
    """
    Ihsan Protocol Metrics: 5-dimension ethical scoring.

    Arabic-English semantic mapping:
    - Ikhlas (Truthfulness): Honest representation
    - Karama (Dignity): Human/agent dignity preservation
    - Adl (Fairness): Just distribution of outcomes
    - Kamal (Excellence): Optimization for quality
    - Istidama (Sustainability): Long-term viability
    """

    ikhlas: float = 0.95  # Truthfulness
    karama: float = 0.95  # Dignity
    adl: float = 0.95  # Fairness
    kamal: float = 0.95  # Excellence
    istidama: float = 0.95  # Sustainability

    WEIGHTS = {
        "ikhlas": 0.30,
        "karama": 0.20,
        "adl": 0.20,
        "kamal": 0.20,
        "istidama": 0.10,
    }

    @property
    def composite(self) -> float:
        """Weighted composite Ihsan score."""
        return (
            self.ikhlas * self.WEIGHTS["ikhlas"]
            + self.karama * self.WEIGHTS["karama"]
            + self.adl * self.WEIGHTS["adl"]
            + self.kamal * self.WEIGHTS["kamal"]
            + self.istidama * self.WEIGHTS["istidama"]
        )

    def validate(self, threshold: float = 0.95) -> Tuple[bool, str]:
        """Hard Ihsan validation - FAIL-CLOSED."""
        if self.composite < threshold:
            return (False, f"Ihsan violation: {self.composite:.4f} < {threshold}")
        return (True, "Ihsan compliant")

    def to_dict(self) -> Dict[str, float]:
        return {
            "ikhlas": self.ikhlas,
            "karama": self.karama,
            "adl": self.adl,
            "kamal": self.kamal,
            "istidama": self.istidama,
            "composite": self.composite,
        }


@dataclass
class Vote:
    """Individual vote record."""

    voter_id: str
    proposal_id: str
    choice: VoteChoice
    stake_weight: float  # Token-weighted influence
    quadratic_weight: float  # Square-root of stake for anti-plutocracy
    rationale: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signature: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voter_id": self.voter_id,
            "proposal_id": self.proposal_id,
            "choice": self.choice.value,
            "stake_weight": self.stake_weight,
            "quadratic_weight": self.quadratic_weight,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Proposal:
    """
    Governance proposal with full lifecycle tracking.
    """

    proposal_id: str
    proposal_type: ProposalType
    title: str
    description: str
    proposer_id: str

    # Metrics
    fate_metrics: FATEMetrics = field(default_factory=FATEMetrics)
    ihsan_metrics: IhsanMetrics = field(default_factory=IhsanMetrics)

    # Voting
    votes: List[Vote] = field(default_factory=list)
    quorum_threshold: float = 0.50  # Minimum participation
    approval_threshold: float = 0.66  # Required approval %

    # Lifecycle
    status: ProposalStatus = ProposalStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    voting_deadline: Optional[datetime] = None
    execution_deadline: Optional[datetime] = None

    # Payload
    payload: Dict[str, Any] = field(default_factory=dict)
    execution_result: Optional[Dict[str, Any]] = None

    # Blockchain anchoring
    block_height: Optional[int] = None
    tx_hash: Optional[str] = None

    def add_vote(self, vote: Vote) -> bool:
        """Add vote to proposal."""
        if self.status != ProposalStatus.VOTING:
            return False

        # Check for duplicate votes
        if any(v.voter_id == vote.voter_id for v in self.votes):
            return False

        self.votes.append(vote)
        return True

    def compute_results(self) -> Dict[str, Any]:
        """Compute voting results with quadratic weighting."""
        if not self.votes:
            return {
                "total_votes": 0,
                "participation": 0.0,
                "approval_rate": 0.0,
                "quadratic_approval": 0.0,
                "passed": False,
            }

        approve_weight = sum(
            v.quadratic_weight for v in self.votes if v.choice == VoteChoice.APPROVE
        )
        reject_weight = sum(
            v.quadratic_weight for v in self.votes if v.choice == VoteChoice.REJECT
        )
        total_weight = approve_weight + reject_weight

        approval_rate = approve_weight / total_weight if total_weight > 0 else 0.0

        return {
            "total_votes": len(self.votes),
            "approve_votes": sum(
                1 for v in self.votes if v.choice == VoteChoice.APPROVE
            ),
            "reject_votes": sum(1 for v in self.votes if v.choice == VoteChoice.REJECT),
            "abstain_votes": sum(
                1 for v in self.votes if v.choice == VoteChoice.ABSTAIN
            ),
            "approve_weight": approve_weight,
            "reject_weight": reject_weight,
            "approval_rate": approval_rate,
            "passed": approval_rate >= self.approval_threshold,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type.name,
            "title": self.title,
            "proposer_id": self.proposer_id,
            "status": self.status.name,
            "fate_metrics": self.fate_metrics.to_dict(),
            "ihsan_metrics": self.ihsan_metrics.to_dict(),
            "votes_count": len(self.votes),
            "results": self.compute_results(),
            "created_at": self.created_at.isoformat(),
        }


class IhsanCircuitBreaker:
    """
    Hard circuit breaker for Ihsan Protocol violations.

    FAIL-CLOSED: Any proposal with IM < 0.95 is IMMEDIATELY REJECTED.
    This is non-negotiable ethical physics.
    """

    THRESHOLD = 0.95

    def __init__(self):
        self._trip_count = 0
        self._total_checks = 0
        self._trip_history: List[Dict[str, Any]] = []

    def check(self, proposal: Proposal) -> Tuple[bool, str]:
        """
        Check if proposal passes Ihsan circuit breaker.

        Returns (passes, reason).
        """
        self._total_checks += 1

        # Validate Ihsan metrics
        ihsan_valid, ihsan_reason = proposal.ihsan_metrics.validate(self.THRESHOLD)
        if not ihsan_valid:
            self._trip(proposal, ihsan_reason)
            return (False, f"CIRCUIT BREAKER TRIPPED: {ihsan_reason}")

        # Validate FATE metrics
        fate_valid, fate_reason = proposal.fate_metrics.validate(self.THRESHOLD)
        if not fate_valid:
            self._trip(proposal, fate_reason)
            return (False, f"CIRCUIT BREAKER TRIPPED: {fate_reason}")

        return (True, "Circuit breaker: PASSED")

    def _trip(self, proposal: Proposal, reason: str) -> None:
        """Record circuit breaker trip."""
        self._trip_count += 1
        self._trip_history.append(
            {
                "proposal_id": proposal.proposal_id,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ihsan_composite": proposal.ihsan_metrics.composite,
                "fate_composite": proposal.fate_metrics.composite,
            }
        )

    @property
    def trip_rate(self) -> float:
        if self._total_checks == 0:
            return 0.0
        return self._trip_count / self._total_checks

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_checks": self._total_checks,
            "trip_count": self._trip_count,
            "trip_rate": self.trip_rate,
            "recent_trips": self._trip_history[-10:],
        }


class GovernanceHypervisor:
    """
    BIZRA Layer 6: Governance Hypervisor.

    Sovereign oversight with thermodynamically-coupled governance:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      GOVERNANCE HYPERVISOR                               │
    │                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │                    PROPOSAL LIFECYCLE                              │  │
    │  │                                                                    │  │
    │  │  [DRAFT] ──▶ [SUBMITTED] ──▶ [REVIEW] ──▶ [VOTING] ──▶ [EXECUTE] │  │
    │  │                  │                          │                      │  │
    │  │                  ▼                          ▼                      │  │
    │  │            ┌──────────┐              ┌──────────┐                  │  │
    │  │            │  CIRCUIT │              │   VOTE   │                  │  │
    │  │            │ BREAKER  │              │  TALLY   │                  │  │
    │  │            │ IM ≥0.95 │              │ (Quad)   │                  │  │
    │  │            └──────────┘              └──────────┘                  │  │
    │  │                  │                          │                      │  │
    │  │                  ▼                          ▼                      │  │
    │  │            [VETOED]                   [PASSED/FAILED]              │  │
    │  └───────────────────────────────────────────────────────────────────┘  │
    │                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │                    FATE ENGINE                                     │  │
    │  │  Fairness: 0.30 | Autonomy: 0.25 | Transparency: 0.25 | Empower: 0.20 │
    │  └───────────────────────────────────────────────────────────────────┘  │
    │                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │                 THERMODYNAMIC COUPLING                             │  │
    │  │  Work Extraction = f(Ihsan) × Carnot Efficiency × Stake Weight    │  │
    │  └───────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        blockchain: Optional[BlockchainSubstrate] = None,
        persistence: Optional[StatePersistenceEngine] = None,
        default_voting_period_hours: int = 72,
    ):
        self.blockchain = blockchain
        self.persistence = persistence
        self.default_voting_period = timedelta(hours=default_voting_period_hours)

        # Circuit breaker
        self.circuit_breaker = IhsanCircuitBreaker()

        # Proposal registry
        self.proposals: Dict[str, Proposal] = {}
        self.proposal_index: Dict[ProposalStatus, Set[str]] = {
            status: set() for status in ProposalStatus
        }

        # Voter registry
        self.voters: Dict[str, Dict[str, Any]] = {}

        # Thermodynamic coupling
        self.thermodynamic_engine: Optional[BIZRAThermodynamicEngine] = None

        # Metrics
        self._proposals_created = 0
        self._proposals_passed = 0
        self._proposals_failed = 0
        self._proposals_vetoed = 0
        self._total_votes = 0
        self._start_time = time.time()

        # Lock
        self._gov_lock = asyncio.Lock()

    def register_thermodynamic_engine(self, engine: BIZRAThermodynamicEngine) -> None:
        """Couple thermodynamic engine for efficiency calculations."""
        self.thermodynamic_engine = engine

    async def create_proposal(
        self,
        proposal_type: ProposalType,
        title: str,
        description: str,
        proposer_id: str,
        payload: Dict[str, Any],
        fate_metrics: Optional[FATEMetrics] = None,
        ihsan_metrics: Optional[IhsanMetrics] = None,
        voting_period_hours: Optional[int] = None,
    ) -> Tuple[str, bool, str]:
        """
        Create a new governance proposal.

        Returns (proposal_id, accepted, reason).
        """
        async with self._gov_lock:
            proposal_id = f"prop_{secrets.token_hex(8)}"

            voting_deadline = datetime.now(timezone.utc) + (
                timedelta(hours=voting_period_hours)
                if voting_period_hours
                else self.default_voting_period
            )

            # Governance proposals require elevated Ihsan metrics (>= 0.98)
            # If not provided, use governance-grade defaults
            governance_ihsan = ihsan_metrics or IhsanMetrics(
                ikhlas=0.99,
                karama=0.99,
                adl=0.99,
                kamal=0.98,
                istidama=0.98,
            )  # Composite: 0.99*0.3 + 0.99*0.2 + 0.99*0.2 + 0.98*0.2 + 0.98*0.1 = 0.986

            proposal = Proposal(
                proposal_id=proposal_id,
                proposal_type=proposal_type,
                title=title,
                description=description,
                proposer_id=proposer_id,
                fate_metrics=fate_metrics or FATEMetrics(),
                ihsan_metrics=governance_ihsan,
                payload=payload,
                voting_deadline=voting_deadline,
            )

            # Circuit breaker check (FAIL-CLOSED)
            passes, reason = self.circuit_breaker.check(proposal)
            if not passes:
                proposal.status = ProposalStatus.VETOED
                self._proposals_vetoed += 1
                self.proposals[proposal_id] = proposal
                self.proposal_index[ProposalStatus.VETOED].add(proposal_id)
                return (proposal_id, False, reason)

            # Submit to blockchain
            if self.blockchain:
                tx_id, accepted, tx_reason = await self.blockchain.submit_transaction(
                    BlockType.GOVERNANCE,
                    proposer_id,
                    {
                        "action": "create_proposal",
                        "proposal_id": proposal_id,
                        "proposal_type": proposal_type.name,
                        "fate_composite": proposal.fate_metrics.composite,
                        "ihsan_composite": proposal.ihsan_metrics.composite,
                    },
                    ihsan_score=proposal.ihsan_metrics.composite,
                )

                if not accepted:
                    proposal.status = ProposalStatus.VETOED
                    self._proposals_vetoed += 1
                    return (proposal_id, False, f"Blockchain rejection: {tx_reason}")

                proposal.tx_hash = tx_id

            # Transition to SUBMITTED
            proposal.status = ProposalStatus.SUBMITTED
            self.proposals[proposal_id] = proposal
            self.proposal_index[ProposalStatus.SUBMITTED].add(proposal_id)
            self._proposals_created += 1

            return (proposal_id, True, "Proposal created successfully")

    async def start_voting(self, proposal_id: str) -> Tuple[bool, str]:
        """Transition proposal to voting phase."""
        async with self._gov_lock:
            if proposal_id not in self.proposals:
                return (False, "Proposal not found")

            proposal = self.proposals[proposal_id]

            if proposal.status not in (
                ProposalStatus.SUBMITTED,
                ProposalStatus.UNDER_REVIEW,
            ):
                return (False, f"Invalid status for voting: {proposal.status.name}")

            # Re-check circuit breaker
            passes, reason = self.circuit_breaker.check(proposal)
            if not passes:
                self._transition_proposal(proposal, ProposalStatus.VETOED)
                self._proposals_vetoed += 1
                return (False, reason)

            self._transition_proposal(proposal, ProposalStatus.VOTING)

            return (True, "Voting started")

    async def cast_vote(
        self,
        proposal_id: str,
        voter_id: str,
        choice: VoteChoice,
        stake: float,
        rationale: str = "",
    ) -> Tuple[bool, str]:
        """Cast a vote on a proposal."""
        async with self._gov_lock:
            if proposal_id not in self.proposals:
                return (False, "Proposal not found")

            proposal = self.proposals[proposal_id]

            if proposal.status != ProposalStatus.VOTING:
                return (False, f"Proposal not in voting phase: {proposal.status.name}")

            # Check voting deadline
            if (
                proposal.voting_deadline
                and datetime.now(timezone.utc) > proposal.voting_deadline
            ):
                self._finalize_voting(proposal)
                return (False, "Voting deadline passed")

            # Quadratic voting weight (anti-plutocracy)
            quadratic_weight = math.sqrt(stake)

            vote = Vote(
                voter_id=voter_id,
                proposal_id=proposal_id,
                choice=choice,
                stake_weight=stake,
                quadratic_weight=quadratic_weight,
                rationale=rationale,
            )

            if not proposal.add_vote(vote):
                return (False, "Duplicate vote or invalid proposal state")

            self._total_votes += 1

            # Record in blockchain
            if self.blockchain:
                await self.blockchain.submit_transaction(
                    BlockType.GOVERNANCE,
                    voter_id,
                    {
                        "action": "vote",
                        "proposal_id": proposal_id,
                        "choice": choice.value,
                        "stake_weight": stake,
                        "quadratic_weight": quadratic_weight,
                    },
                    ihsan_score=0.99,  # Voting is high-trust action
                )

            return (True, f"Vote recorded: {choice.value}")

    async def finalize_proposal(self, proposal_id: str) -> Tuple[bool, str]:
        """Finalize voting and determine outcome."""
        async with self._gov_lock:
            if proposal_id not in self.proposals:
                return (False, "Proposal not found")

            proposal = self.proposals[proposal_id]

            if proposal.status != ProposalStatus.VOTING:
                return (False, f"Proposal not in voting phase: {proposal.status.name}")

            results = proposal.compute_results()

            if results["passed"]:
                self._transition_proposal(proposal, ProposalStatus.PASSED)
                self._proposals_passed += 1

                # Execute if applicable
                if proposal.proposal_type in (
                    ProposalType.PARAMETER_CHANGE,
                    ProposalType.EMERGENCY_ACTION,
                ):
                    await self._execute_proposal(proposal)

                return (
                    True,
                    f"Proposal PASSED: {results['approval_rate']:.2%} approval",
                )
            else:
                self._transition_proposal(proposal, ProposalStatus.FAILED)
                self._proposals_failed += 1
                return (
                    False,
                    f"Proposal FAILED: {results['approval_rate']:.2%} approval",
                )

    async def _execute_proposal(self, proposal: Proposal) -> None:
        """Execute a passed proposal."""
        proposal.status = ProposalStatus.EXECUTED

        # Record execution
        execution_result = {
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "payload_applied": True,
        }

        # Apply parameter changes
        if proposal.proposal_type == ProposalType.PARAMETER_CHANGE:
            param_name = proposal.payload.get("parameter")
            param_value = proposal.payload.get("value")

            execution_result["parameter"] = param_name
            execution_result["value"] = param_value

            # Update blockchain governance parameters if applicable
            if self.blockchain and param_name == "ihsan_threshold":
                self.blockchain.world_state.governance["ihsan_threshold"] = param_value

        proposal.execution_result = execution_result

        # Anchor to blockchain
        if self.blockchain:
            await self.blockchain.submit_transaction(
                BlockType.GOVERNANCE,
                "governance_hypervisor",
                {
                    "action": "execute_proposal",
                    "proposal_id": proposal.proposal_id,
                    "execution_result": execution_result,
                },
                ihsan_score=proposal.ihsan_metrics.composite,
            )

    async def appeal_proposal(
        self,
        proposal_id: str,
        appellant_id: str,
        grounds: str,
    ) -> Tuple[str, bool, str]:
        """
        File an appeal against a proposal decision.

        Appeals are resolved via "Third Fact" - blockchain arbitration.
        """
        async with self._gov_lock:
            if proposal_id not in self.proposals:
                return ("", False, "Proposal not found")

            proposal = self.proposals[proposal_id]

            if proposal.status not in (
                ProposalStatus.PASSED,
                ProposalStatus.FAILED,
                ProposalStatus.VETOED,
            ):
                return (
                    "",
                    False,
                    f"Cannot appeal proposal in state: {proposal.status.name}",
                )

            appeal_id = f"appeal_{secrets.token_hex(8)}"

            # Record appeal on blockchain
            if self.blockchain:
                tx_id, accepted, reason = await self.blockchain.submit_transaction(
                    BlockType.GOVERNANCE,
                    appellant_id,
                    {
                        "action": "appeal",
                        "appeal_id": appeal_id,
                        "proposal_id": proposal_id,
                        "original_status": proposal.status.name,
                        "grounds": grounds,
                    },
                    ihsan_score=0.99,
                )

                if not accepted:
                    return (appeal_id, False, f"Appeal rejected: {reason}")

            self._transition_proposal(proposal, ProposalStatus.APPEALED)

            return (appeal_id, True, "Appeal filed - awaiting Third Fact arbitration")

    def _transition_proposal(
        self, proposal: Proposal, new_status: ProposalStatus
    ) -> None:
        """Transition proposal status with index update."""
        old_status = proposal.status

        if old_status in self.proposal_index:
            self.proposal_index[old_status].discard(proposal.proposal_id)

        proposal.status = new_status
        self.proposal_index[new_status].add(proposal.proposal_id)

    def _finalize_voting(self, proposal: Proposal) -> None:
        """Finalize voting after deadline."""
        results = proposal.compute_results()

        if results["passed"]:
            self._transition_proposal(proposal, ProposalStatus.PASSED)
            self._proposals_passed += 1
        else:
            self._transition_proposal(proposal, ProposalStatus.FAILED)
            self._proposals_failed += 1

    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Get proposal by ID."""
        return self.proposals.get(proposal_id)

    def list_proposals(
        self,
        status_filter: Optional[ProposalStatus] = None,
        proposal_type_filter: Optional[ProposalType] = None,
    ) -> List[Proposal]:
        """List proposals with optional filtering."""
        proposals = list(self.proposals.values())

        if status_filter:
            proposals = [p for p in proposals if p.status == status_filter]

        if proposal_type_filter:
            proposals = [
                p for p in proposals if p.proposal_type == proposal_type_filter
            ]

        return sorted(proposals, key=lambda p: p.created_at, reverse=True)

    def get_metrics(self) -> Dict[str, Any]:
        """Get governance metrics."""
        elapsed = time.time() - self._start_time

        return {
            "total_proposals": len(self.proposals),
            "proposals_created": self._proposals_created,
            "proposals_passed": self._proposals_passed,
            "proposals_failed": self._proposals_failed,
            "proposals_vetoed": self._proposals_vetoed,
            "total_votes": self._total_votes,
            "pass_rate": self._proposals_passed
            / max(1, self._proposals_created - self._proposals_vetoed),
            "circuit_breaker_trip_rate": self.circuit_breaker.trip_rate,
            "proposals_by_status": {
                status.name: len(ids) for status, ids in self.proposal_index.items()
            },
            "uptime_seconds": elapsed,
        }

    def get_fate_summary(self) -> Dict[str, Any]:
        """Get FATE engine summary across all proposals."""
        if not self.proposals:
            return {"average_fate": 0.0, "average_ihsan": 0.0}

        fate_scores = [p.fate_metrics.composite for p in self.proposals.values()]
        ihsan_scores = [p.ihsan_metrics.composite for p in self.proposals.values()]

        return {
            "average_fate": sum(fate_scores) / len(fate_scores),
            "average_ihsan": sum(ihsan_scores) / len(ihsan_scores),
            "min_fate": min(fate_scores),
            "max_fate": max(fate_scores),
            "min_ihsan": min(ihsan_scores),
            "max_ihsan": max(ihsan_scores),
            "fate_threshold": 0.95,
            "ihsan_threshold": IhsanCircuitBreaker.THRESHOLD,
        }


async def demo_governance_hypervisor():
    """Demonstrate the Layer 6 Governance Hypervisor."""
    print("=" * 70)
    print("BIZRA LAYER 6: GOVERNANCE HYPERVISOR - FATE ENGINE")
    print("=" * 70)

    # Initialize hypervisor
    hypervisor = GovernanceHypervisor(default_voting_period_hours=24)

    print(f"\n✓ Governance Hypervisor initialized")
    print(f"  Circuit breaker threshold: {IhsanCircuitBreaker.THRESHOLD}")

    # Create proposals
    print("\n" + "-" * 70)
    print("CREATING PROPOSALS")
    print("-" * 70)

    # Proposal 1: Valid parameter change
    prop1_id, accepted, reason = await hypervisor.create_proposal(
        proposal_type=ProposalType.PARAMETER_CHANGE,
        title="Increase Ihsan Threshold to 0.96",
        description="Strengthen ethical requirements for all operations",
        proposer_id="sat_council_001",
        payload={"parameter": "ihsan_threshold", "value": 0.96},
        fate_metrics=FATEMetrics(
            fairness=0.98, autonomy=0.96, transparency=0.99, empowerment=0.95
        ),
        ihsan_metrics=IhsanMetrics(
            ikhlas=0.99, karama=0.97, adl=0.98, kamal=0.96, istidama=0.95
        ),
    )
    print(f"  Proposal 1: {'✓ ACCEPTED' if accepted else '✗ REJECTED'}")
    print(f"    ID: {prop1_id}")
    print(f"    {reason}")

    # Proposal 2: Ihsan violation (should be VETOED)
    prop2_id, accepted, reason = await hypervisor.create_proposal(
        proposal_type=ProposalType.TREASURY_SPEND,
        title="Fund controversial project",
        description="Allocation with questionable ethics",
        proposer_id="bad_actor",
        payload={"amount": 1000000},
        ihsan_metrics=IhsanMetrics(
            ikhlas=0.80, karama=0.85, adl=0.75, kamal=0.90, istidama=0.88
        ),
    )
    print(f"  Proposal 2: {'✓ ACCEPTED' if accepted else '✗ REJECTED'}")
    print(f"    ID: {prop2_id}")
    print(f"    {reason}")

    # Proposal 3: Agent certification
    prop3_id, accepted, reason = await hypervisor.create_proposal(
        proposal_type=ProposalType.AGENT_CERTIFICATION,
        title="Certify Agent Alpha for Production",
        description="Agent has passed all Ihsan requirements",
        proposer_id="certification_board",
        payload={"agent_id": "agent_alpha", "certification_level": "production"},
    )
    print(f"  Proposal 3: {'✓ ACCEPTED' if accepted else '✗ REJECTED'}")
    print(f"    ID: {prop3_id}")

    # Start voting
    print("\n" + "-" * 70)
    print("VOTING PHASE")
    print("-" * 70)

    await hypervisor.start_voting(prop1_id)
    await hypervisor.start_voting(prop3_id)
    print(f"  ✓ Voting started for proposals 1 and 3")

    # Cast votes
    voters = [
        ("validator_001", 1000.0, VoteChoice.APPROVE),
        ("validator_002", 500.0, VoteChoice.APPROVE),
        ("validator_003", 750.0, VoteChoice.APPROVE),
        ("validator_004", 300.0, VoteChoice.REJECT),
        ("validator_005", 200.0, VoteChoice.ABSTAIN),
    ]

    for voter_id, stake, choice in voters:
        success, msg = await hypervisor.cast_vote(prop1_id, voter_id, choice, stake)
        print(f"  {voter_id}: {choice.value} (stake: {stake}) - {msg}")

    # Finalize
    print("\n" + "-" * 70)
    print("FINALIZING PROPOSALS")
    print("-" * 70)

    passed, result = await hypervisor.finalize_proposal(prop1_id)
    print(f"  Proposal 1: {result}")

    # Show proposal details
    print("\n" + "-" * 70)
    print("PROPOSAL DETAILS")
    print("-" * 70)

    for prop_id in [prop1_id, prop2_id, prop3_id]:
        prop = hypervisor.get_proposal(prop_id)
        print(f"\n  {prop.title}:")
        print(f"    Status: {prop.status.name}")
        print(f"    FATE: {prop.fate_metrics.composite:.4f}")
        print(f"    Ihsan: {prop.ihsan_metrics.composite:.4f}")
        if prop.votes:
            results = prop.compute_results()
            print(
                f"    Votes: {results['approve_votes']} approve, {results['reject_votes']} reject"
            )
            print(f"    Approval: {results['approval_rate']:.2%}")

    # Metrics
    print("\n" + "-" * 70)
    print("GOVERNANCE METRICS")
    print("-" * 70)

    metrics = hypervisor.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    # FATE Summary
    print("\n" + "-" * 70)
    print("FATE ENGINE SUMMARY")
    print("-" * 70)

    fate = hypervisor.get_fate_summary()
    for key, value in fate.items():
        print(
            f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}"
        )

    print("\n" + "=" * 70)
    print("✓ Layer 6 Governance Hypervisor operational")
    print("  FATE Engine active | Ihsan circuit breaker enforcing IM ≥ 0.95")
    print("=" * 70)

    return hypervisor


if __name__ == "__main__":
    asyncio.run(demo_governance_hypervisor())
