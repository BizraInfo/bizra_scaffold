"""
BIZRA Core Layers - Vanguard Lazy-Load Configuration.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import importlib

__version__ = "5.0.0-vanguard-genesis"

__all__ = [
    "L3EpisodicMemoryV2",
    "L4SemanticHyperGraphV2",
    "ThermodynamicEngine",
    "GovernanceHypervisor",
]

if TYPE_CHECKING:
    from .layers.memory_layers_v2 import L3EpisodicMemoryV2, L4SemanticHyperGraphV2
    from .layers.thermodynamic_engine import ThermodynamicEngine
    from .layers.governance_hypervisor import GovernanceHypervisor

# PEP 562 Lazy Imports
_LAZY_IMPORTS = {
    "L3EpisodicMemoryV2": (".layers.memory_layers_v2", "L3EpisodicMemoryV2"),
    "L4SemanticHyperGraphV2": (".layers.memory_layers_v2", "L4SemanticHyperGraphV2"),
    "ThermodynamicEngine": (".layers.thermodynamic_engine", "ThermodynamicEngine"),
    "GovernanceHypervisor": (".layers.governance_hypervisor", "GovernanceHypervisor"),
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        mod_name, cls_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(mod_name, __package__)
        return getattr(module, cls_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
