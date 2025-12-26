"""
BIZRA AEON OMEGA - Core Module
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Grade | Complete Cognitive Architecture

This module contains the core components that close the 6% architectural gap:

1. tiered_verification.py   - Addresses latency (2%)
2. consequential_ethics.py  - Addresses ethics gap (1%)
3. narrative_compiler.py    - Addresses interpretability (1%)
4. value_oracle.py          - Addresses oracle problem (2%)
5. ultimate_integration.py  - Complete integration

Together with cognitive_sovereign.py and ihsan_bridge.py, these modules
achieve 100% of the BIZRA-VCC-Node0 architectural vision.

Usage:
    from core.ultimate_integration import BIZRAVCCNode0Ultimate, Observation

    ultimate = BIZRAVCCNode0Ultimate()
    result = await ultimate.process(observation)
"""

from core.consequential_ethics import (
    ConsequentialEthicsEngine,
    EthicalFramework,
    EthicalVerdict,
    VerdictSeverity,
)
from core.narrative_compiler import (
    CognitiveSynthesis,
    CompiledNarrative,
    NarrativeCompiler,
    NarrativeStyle,
)
from core.thermodynamic_engine import (
    BIZRAThermodynamicEngine,
    CycleMetrics,
    CycleType,
    Reservoir,
    ThermodynamicConstants,
    ThermodynamicState,
    WorkingFluid,
    create_engine,
)
from core.tiered_verification import (
    TieredVerificationEngine,
    UrgencyLevel,
    VerificationResult,
    VerificationTier,
)
from core.ultimate_integration import (
    BIZRAVCCNode0Ultimate,
    ConvergenceQuality,
    HealthStatus,
    Observation,
    UltimateResult,
)
from core.value_oracle import (
    Convergence,
    OracleType,
    PluralisticValueOracle,
    ValueAssessment,
)

# Layer imports (with graceful fallback)
try:
    from core.layers.blockchain_substrate import (
        Block,
        BlockchainSubstrate,
        BlockType,
        IhsanEnforcer,
        MerkleTree,
        Transaction,
        WorldState,
    )

    _BLOCKCHAIN_AVAILABLE = True
except ImportError:
    _BLOCKCHAIN_AVAILABLE = False

try:
    from core.engine.state_persistence import (
        AgentCheckpoint,
        AgentLifecycleState,
        AgentState,
        CognitiveState,
        DualTokenWallet,
        StatePersistenceEngine,
        WalletType,
    )

    _PERSISTENCE_AVAILABLE = True
except ImportError:
    _PERSISTENCE_AVAILABLE = False

try:
    from core.layers.governance_hypervisor import (
        FATEMetrics,
        GovernanceHypervisor,
        IhsanCircuitBreaker,
        IhsanMetrics,
        Proposal,
        ProposalStatus,
        ProposalType,
        VoteChoice,
    )

    _GOVERNANCE_AVAILABLE = True
except ImportError:
    _GOVERNANCE_AVAILABLE = False

try:
    from core.lifecycle_emulator import (
        EmulationMode,
        EmulationResult,
        LifecycleEmulator,
        LifecyclePhase,
        PATSATOrchestrator,
    )

    _LIFECYCLE_AVAILABLE = True
except ImportError:
    _LIFECYCLE_AVAILABLE = False

try:
    from core.production_validator import (
        ProductionReadinessValidator,
        ValidationCheck,
        ValidationReport,
        ValidationSeverity,
        ValidationStatus,
    )

    _VALIDATOR_AVAILABLE = True
except ImportError:
    _VALIDATOR_AVAILABLE = False

__all__ = [
    # Tiered Verification
    "TieredVerificationEngine",
    "UrgencyLevel",
    "VerificationTier",
    "VerificationResult",
    # Consequential Ethics
    "ConsequentialEthicsEngine",
    "EthicalFramework",
    "VerdictSeverity",
    "EthicalVerdict",
    # Narrative Compiler
    "NarrativeCompiler",
    "NarrativeStyle",
    "CompiledNarrative",
    "CognitiveSynthesis",
    # Value Oracle
    "PluralisticValueOracle",
    "OracleType",
    "Convergence",
    "ValueAssessment",
    # Ultimate Integration
    "BIZRAVCCNode0Ultimate",
    "Observation",
    "UltimateResult",
    "HealthStatus",
    "ConvergenceQuality",
    # Thermodynamic Engine
    "BIZRAThermodynamicEngine",
    "ThermodynamicState",
    "CycleType",
    "ThermodynamicConstants",
    "Reservoir",
    "WorkingFluid",
    "CycleMetrics",
    "create_engine",
    # Layer 1: Blockchain Substrate
    "BlockchainSubstrate",
    "BlockType",
    "Transaction",
    "Block",
    "MerkleTree",
    "WorldState",
    "IhsanEnforcer",
    # Layer 3: State Persistence
    "StatePersistenceEngine",
    "AgentState",
    "AgentLifecycleState",
    "CognitiveState",
    "DualTokenWallet",
    "WalletType",
    "AgentCheckpoint",
    # Layer 6: Governance Hypervisor
    "GovernanceHypervisor",
    "ProposalType",
    "ProposalStatus",
    "VoteChoice",
    "Proposal",
    "FATEMetrics",
    "IhsanMetrics",
    "IhsanCircuitBreaker",
    # Lifecycle Emulator
    "LifecycleEmulator",
    "LifecyclePhase",
    "EmulationMode",
    "EmulationResult",
    "PATSATOrchestrator",
    # Production Validator
    "ProductionReadinessValidator",
    "ValidationReport",
    "ValidationCheck",
    "ValidationSeverity",
    "ValidationStatus",
]

__version__ = "2.0.0"
__author__ = "BIZRA AEON OMEGA Team"
__license__ = "MIT"

# Feature flags
FEATURES = {
    "blockchain": _BLOCKCHAIN_AVAILABLE if "_BLOCKCHAIN_AVAILABLE" in dir() else False,
    "persistence": (
        _PERSISTENCE_AVAILABLE if "_PERSISTENCE_AVAILABLE" in dir() else False
    ),
    "governance": _GOVERNANCE_AVAILABLE if "_GOVERNANCE_AVAILABLE" in dir() else False,
    "lifecycle": _LIFECYCLE_AVAILABLE if "_LIFECYCLE_AVAILABLE" in dir() else False,
    "validator": _VALIDATOR_AVAILABLE if "_VALIDATOR_AVAILABLE" in dir() else False,
}
