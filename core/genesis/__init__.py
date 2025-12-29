"""
BIZRA Genesis Package
════════════════════════════════════════════════════════════════════════════════

The Genesis module defines the foundational elements of the BIZRA network:

- Node0 (Genesis Node): The first node, running on the First Architect's machine
- Proof of Impact: Calculation of contribution value from ecosystem artifacts
- Genesis Seal: The cryptographic anchor binding Node0 to its 3-year history

This module implements the paradigm shift from "repo-centric" to "ecosystem-centric"
thinking. The entire machine—all repositories, chat data, knowledge graphs, and
crystallized tools—constitutes the Genesis Node's state.

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from core.genesis.node_zero import (
    NodeZeroIdentity,
    NodeRole,
    MachineFingerprint,
    OwnerAttestation,
    ConstitutionBinding,
    TimelineBounds,
    GENESIS_EPOCH,
    GENESIS_EPOCH_ISO,
    NODE_ZERO_ROLE,
    FIRST_ARCHITECT_ROLE,
    GENESIS_IHSAN_THRESHOLD,
)

from core.genesis.proof_of_impact import (
    ProofOfImpact,
    ArtifactImpact,
    TemporalImpact,
    StructuralImpact,
    ImpactCategory,
    IMPACT_MULTIPLIERS,
)

__all__ = [
    # Identity
    "NodeZeroIdentity",
    "NodeRole",
    "MachineFingerprint",
    "OwnerAttestation",
    "ConstitutionBinding",
    "TimelineBounds",
    # Proof of Impact
    "ProofOfImpact",
    "ArtifactImpact",
    "TemporalImpact",
    "StructuralImpact",
    "ImpactCategory",
    "IMPACT_MULTIPLIERS",
    # Constants
    "GENESIS_EPOCH",
    "GENESIS_EPOCH_ISO",
    "NODE_ZERO_ROLE",
    "FIRST_ARCHITECT_ROLE",
    "GENESIS_IHSAN_THRESHOLD",
]
