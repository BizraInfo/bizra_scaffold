"""SAT consensus simulation (deterministic).

The mature design describes 6 validators with a 70% threshold and veto roles.
We simulate that with deterministic voting rules based on Ihsan dimensions.

- security (veto): depends on 'amanah' (safety)
- formal (veto): depends on 'adl' (correctness)
- ethics (veto): depends on min('ihsan','mizan')
- performance: depends on 'hikmah'
- reliability: depends on 'sabr'
- ops: depends on 'bayan'

Overall pass requires:
  1) No veto validator fails
  2) Weighted pass ratio >= pass_ratio

Determinism note:
- All weights are stored as integers in "centiweight" units (x100).
- pass_ratio is represented as Fixed64 Q32.32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .fixed import Fixed64


@dataclass(frozen=True)
class Validator:
    role: str
    weight_centi: int  # e.g. 2.0 -> 200
    veto: bool


@dataclass(frozen=True)
class Vote:
    role: str
    passed: bool
    reason: str
    weight_centi: int


@dataclass(frozen=True)
class ConsensusResult:
    passed: bool
    pass_ratio: Fixed64
    total_weight_centi: int
    passed_weight_centi: int
    veto_triggered: bool
    votes: List[Vote]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "pass_ratio": self.pass_ratio.to_decimal_str(6),
            "pass_ratio_bits": self.pass_ratio.as_bits_u64(),
            "total_weight_centi": self.total_weight_centi,
            "passed_weight_centi": self.passed_weight_centi,
            "veto_triggered": self.veto_triggered,
            "votes": [
                {
                    "role": v.role,
                    "passed": v.passed,
                    "reason": v.reason,
                    "weight_centi": v.weight_centi,
                }
                for v in self.votes
            ],
        }


# Validator minimum thresholds (dimension specific)
MIN_SECURITY = Fixed64.from_decimal_str("0.85")
MIN_FORMAL = Fixed64.from_decimal_str("0.80")
MIN_ETHICS = Fixed64.from_decimal_str("0.80")
MIN_PERF = Fixed64.from_decimal_str("0.60")
MIN_RELIABILITY = Fixed64.from_decimal_str("0.70")
MIN_OPS = Fixed64.from_decimal_str("0.70")


def _vote_for(role: str, dims: Dict[str, Fixed64]) -> Tuple[bool, str]:
    if role == "security":
        v = dims.get("amanah", Fixed64.zero())
        return (v >= MIN_SECURITY, f"amanah={v.to_decimal_str(6)}")
    if role == "formal":
        v = dims.get("adl", Fixed64.zero())
        return (v >= MIN_FORMAL, f"adl={v.to_decimal_str(6)}")
    if role == "ethics":
        a = dims.get("ihsan", Fixed64.zero())
        b = dims.get("mizan", Fixed64.zero())
        v = a if a.value < b.value else b
        return (v >= MIN_ETHICS, f"min(ihsan,mizan)={v.to_decimal_str(6)}")
    if role == "performance":
        v = dims.get("hikmah", Fixed64.zero())
        return (v >= MIN_PERF, f"hikmah={v.to_decimal_str(6)}")
    if role == "reliability":
        v = dims.get("sabr", Fixed64.zero())
        return (v >= MIN_RELIABILITY, f"sabr={v.to_decimal_str(6)}")
    if role == "ops":
        v = dims.get("bayan", Fixed64.zero())
        return (v >= MIN_OPS, f"bayan={v.to_decimal_str(6)}")

    return False, "unknown_validator"


def parse_weight_centi(w: object) -> int:
    """Parse a YAML weight like 2.0 into centi-units (x100) deterministically."""
    from decimal import Decimal, ROUND_HALF_UP

    if isinstance(w, (int,)):
        return int(w) * 100
    # YAML may parse 2.0 as float; convert via Decimal(str()) to avoid float quirks.
    d = Decimal(str(w))
    return int((d * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def run_consensus(
    validators: List[Validator],
    dims: Dict[str, Fixed64],
    pass_ratio_required: Fixed64,
) -> ConsensusResult:
    votes: List[Vote] = []
    total = 0
    passed_total = 0
    veto_triggered = False

    for val in validators:
        ok, reason = _vote_for(val.role, dims)
        votes.append(Vote(role=val.role, passed=ok, reason=reason, weight_centi=val.weight_centi))
        total += int(val.weight_centi)
        if ok:
            passed_total += int(val.weight_centi)
        else:
            if val.veto:
                veto_triggered = True

    ratio = Fixed64.from_ratio(passed_total, total) if total > 0 else Fixed64.zero()
    passed = (not veto_triggered) and (ratio >= pass_ratio_required)
    return ConsensusResult(
        passed=passed,
        pass_ratio=ratio,
        total_weight_centi=total,
        passed_weight_centi=passed_total,
        veto_triggered=veto_triggered,
        votes=votes,
    )
