"""
BIZRA PILLAR-KERNEL CONTRACT REGISTRY (PKCR)
The executable physics of the Sovereign Organism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class PillarID(str, Enum):
    ADL = "ADL_INVARIANT"
    LIVENESS = "FORMAL_LIVENESS"
    IHSAN = "IHSAN_VECTOR"
    SNR = "SNR_FILTERING"
    HARBERGER = "RESOURCE_TAX"
    INTEGRATION = "PHI_COHERENCE"


@dataclass(frozen=True)
class AdlConfig:
    GINI_THRESHOLD: float = 0.35
    CAUSAL_DRAG_MAX: float = 0.05


@dataclass(frozen=True)
class IhsanConfig:
    MIN_SCORE_THRESHOLD: float = 0.95
    WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "correctness": 0.22,
            "safety": 0.22,
            "user_benefit": 0.14,
            "efficiency": 0.12,
            "auditability": 0.12,
            "anti_centralization": 0.08,
            "robustness": 0.06,
            "fairness": 0.04,
        }
    )


@dataclass(frozen=True)
class SNRConfig:
    MIN_ACCEPTABLE_SNR: float = 8.7
    ELITE_THRESHOLD: float = 0.94


class KernelLaws:
    ADL = AdlConfig()
    IHSAN = IhsanConfig()
    SNR = SNRConfig()
