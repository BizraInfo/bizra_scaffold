"""FATE gate (formal-ish verification).

The mature Genesis design uses an SMT solver (e.g., Z3) to formally verify
constraints before allowing an action.

In this simulation we implement a deterministic constraint check that mirrors
what you'd encode in SMT:
  - composite Ihsan >= env threshold
  - safety >= min_safety
  - auditability >= min_auditability
  - correctness >= min_correctness

If you later install z3-solver, this module can be swapped with real SMT checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .fixed import Fixed64


@dataclass(frozen=True)
class FateResult:
    sat: bool
    reason: str

    def as_str(self) -> str:
        return "SAT" if self.sat else "UNSAT"


# Conservative minimums for a "commit".
MIN_SAFETY = Fixed64.from_decimal_str("0.80")
MIN_AUDIT = Fixed64.from_decimal_str("0.70")
MIN_CORRECT = Fixed64.from_decimal_str("0.70")


def fate_check(
    ihsan_composite: Fixed64,
    threshold: Fixed64,
    dims: Dict[str, Fixed64],
) -> FateResult:
    if ihsan_composite < threshold:
        return FateResult(False, f"ihsan_below_threshold({ihsan_composite.to_decimal_str(6)}<{threshold.to_decimal_str(6)})")

    safety = dims.get("amanah", Fixed64.zero())
    if safety < MIN_SAFETY:
        return FateResult(False, f"safety_below_min({safety.to_decimal_str(6)}<{MIN_SAFETY.to_decimal_str(6)})")

    audit = dims.get("bayan", Fixed64.zero())
    if audit < MIN_AUDIT:
        return FateResult(False, f"auditability_below_min({audit.to_decimal_str(6)}<{MIN_AUDIT.to_decimal_str(6)})")

    correctness = dims.get("adl", Fixed64.zero())
    if correctness < MIN_CORRECT:
        return FateResult(False, f"correctness_below_min({correctness.to_decimal_str(6)}<{MIN_CORRECT.to_decimal_str(6)})")

    return FateResult(True, "ok")
