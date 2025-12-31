"""
BIZRA AEON OMEGA - Layers Package

Elite pattern: lazy exports to avoid import-time coupling to heavy deps.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    # Blockchain Substrate
    "BlockchainSubstrate",
    "BlockType",
    "Transaction",
    "Block",
    "MerkleTree",
    "WorldState",
    "IhsanEnforcer",
    # Governance Hypervisor
    "GovernanceHypervisor",
    "ProposalType",
    "ProposalStatus",
    "VoteChoice",
    "Proposal",
    "FATEMetrics",
    "IhsanMetrics",
    "IhsanCircuitBreaker",
    # Memory Layers
    "L3EpisodicMemoryV2",
    "L4SemanticHyperGraphV2",
]

if TYPE_CHECKING:
    from .blockchain_substrate import (  # noqa: F401
        Block,
        BlockchainSubstrate,
        BlockType,
        IhsanEnforcer,
        MerkleTree,
        Transaction,
        WorldState,
    )
    from .governance_hypervisor import (  # noqa: F401
        FATEMetrics,
        GovernanceHypervisor,
        IhsanCircuitBreaker,
        IhsanMetrics,
        Proposal,
        ProposalStatus,
        ProposalType,
        VoteChoice,
    )
    from .memory_layers_v2 import (  # noqa: F401
        L3EpisodicMemoryV2,
        L4SemanticHyperGraphV2,
    )

_LAZY_IMPORTS = {
    "BlockchainSubstrate": (".blockchain_substrate", "BlockchainSubstrate"),
    "BlockType": (".blockchain_substrate", "BlockType"),
    "Transaction": (".blockchain_substrate", "Transaction"),
    "Block": (".blockchain_substrate", "Block"),
    "MerkleTree": (".blockchain_substrate", "MerkleTree"),
    "WorldState": (".blockchain_substrate", "WorldState"),
    "IhsanEnforcer": (".blockchain_substrate", "IhsanEnforcer"),
    "GovernanceHypervisor": (".governance_hypervisor", "GovernanceHypervisor"),
    "ProposalType": (".governance_hypervisor", "ProposalType"),
    "ProposalStatus": (".governance_hypervisor", "ProposalStatus"),
    "VoteChoice": (".governance_hypervisor", "VoteChoice"),
    "Proposal": (".governance_hypervisor", "Proposal"),
    "FATEMetrics": (".governance_hypervisor", "FATEMetrics"),
    "IhsanMetrics": (".governance_hypervisor", "IhsanMetrics"),
    "IhsanCircuitBreaker": (".governance_hypervisor", "IhsanCircuitBreaker"),
    "L3EpisodicMemoryV2": (".memory_layers_v2", "L3EpisodicMemoryV2"),
    "L4SemanticHyperGraphV2": (".memory_layers_v2", "L4SemanticHyperGraphV2"),
}


def __getattr__(name: str):
    """
    PEP 562: resolve exports lazily to avoid import-time heavy deps.
    """
    if name in _LAZY_IMPORTS:
        module_name, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
