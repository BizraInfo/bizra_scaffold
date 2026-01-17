"""Ihsan engine (8-dimensional, deterministic)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from .fixed import Fixed64, fixed_dot


@dataclass(frozen=True)
class IhsanDimension:
    key: str
    name: str
    weight: Fixed64


@dataclass(frozen=True)
class IhsanEvaluation:
    dimensions: Dict[str, Fixed64]  # score per dimension
    composite: Fixed64

    def to_dict(self) -> dict:
        return {
            "dimensions": {k: v.to_decimal_str(6) for k, v in self.dimensions.items()},
            "composite": self.composite.to_decimal_str(6),
            "composite_bits": self.composite.as_bits_u64(),
        }


class IhsanEngine:
    def __init__(self, dimensions: List[IhsanDimension], thresholds: Dict[str, Fixed64], artifact_thresholds: Dict[str, Dict[str, Fixed64]] | None = None):
        self.dimensions = dimensions
        self.thresholds = thresholds
        self.artifact_thresholds = artifact_thresholds or {}

        # Validate weights sum ~ 1.0 in Fixed64 space.
        total = sum(d.weight.value for d in self.dimensions)
        if abs(total - Fixed64.one().value) > 5:  # allow tiny rounding error
            raise ValueError(f"Ihsan weights must sum to 1.0; got raw={total}")

    @staticmethod
    def load_from_constitution(path: Path) -> "IhsanEngine":
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

        thresholds = {k: Fixed64.from_decimal_str(v) for k, v in cfg.get("thresholds", {}).items()}

        artifact_thresholds: Dict[str, Dict[str, Fixed64]] = {}
        for art, envmap in (cfg.get("artifact_thresholds") or {}).items():
            artifact_thresholds[art] = {env: Fixed64.from_decimal_str(v) for env, v in envmap.items()}

        dims_cfg = cfg["ihsan_engine"]["dimensions"]
        dims: List[IhsanDimension] = []
        for d in dims_cfg:
            dims.append(
                IhsanDimension(
                    key=str(d["key"]),
                    name=str(d["name"]),
                    weight=Fixed64.from_decimal_str(str(d["weight"])),
                )
            )

        return IhsanEngine(dimensions=dims, thresholds=thresholds, artifact_thresholds=artifact_thresholds)

    def threshold_for(self, env: str, artifact_class: str | None = None) -> Fixed64:
        if artifact_class and artifact_class in self.artifact_thresholds:
            envmap = self.artifact_thresholds[artifact_class]
            if env in envmap:
                return envmap[env]
        if env not in self.thresholds:
            raise KeyError(f"Unknown environment '{env}'")
        return self.thresholds[env]

    def evaluate(self, scores: Dict[str, Fixed64]) -> IhsanEvaluation:
        weights = []
        vals = []
        out_scores: Dict[str, Fixed64] = {}
        for dim in self.dimensions:
            v = scores.get(dim.key, Fixed64.zero()).clamp01()
            out_scores[dim.key] = v
            weights.append(dim.weight)
            vals.append(v)

        composite = fixed_dot(weights, vals).clamp01()
        return IhsanEvaluation(dimensions=out_scores, composite=composite)
