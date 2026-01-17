"""Deterministic fixed-point arithmetic.

We use Q32.32 (signed) fixed-point values represented as Python ints.
All scores and weights in the Genesis simulation are constrained to [0, 1].

This mirrors the "Fixed64 Q32.32" design the BIZRA Genesis docs describe,
without relying on platform-specific floating point.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Any


Q = 32
SCALE = 1 << Q
MASK_64 = (1 << 64) - 1

# Ensure deterministic decimal conversion.
getcontext().prec = 50


@dataclass(frozen=True)
class Fixed64:
    """Signed Q32.32 fixed-point.

    Internally stored as an int representing the raw scaled value.
    """

    value: int

    @staticmethod
    def from_decimal_str(s: str) -> "Fixed64":
        d = Decimal(s)
        raw = int((d * SCALE).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        return Fixed64(raw)

    @staticmethod
    def from_ratio(numer: int, denom: int) -> "Fixed64":
        if denom == 0:
            raise ZeroDivisionError("denom must be non-zero")
        raw = (numer * SCALE) // denom
        return Fixed64(raw)

    @staticmethod
    def from_int(i: int) -> "Fixed64":
        return Fixed64(i * SCALE)

    @staticmethod
    def zero() -> "Fixed64":
        return Fixed64(0)

    @staticmethod
    def one() -> "Fixed64":
        return Fixed64(SCALE)

    def clamp01(self) -> "Fixed64":
        if self.value < 0:
            return Fixed64(0)
        if self.value > SCALE:
            return Fixed64(SCALE)
        return self

    def to_decimal_str(self, places: int = 6) -> str:
        # Deterministic string with fixed number of decimal places.
        sign = "-" if self.value < 0 else ""
        v = abs(self.value)
        integer = v // SCALE
        frac = v % SCALE
        # Convert fractional part to decimal places using integer math.
        # frac/SCALE scaled to 10^places:
        scale10 = 10 ** places
        frac10 = (frac * scale10 + (SCALE // 2)) // SCALE  # round half up
        return f"{sign}{integer}.{frac10:0{places}d}"

    def __add__(self, other: "Fixed64") -> "Fixed64":
        return Fixed64(self.value + other.value)

    def __sub__(self, other: "Fixed64") -> "Fixed64":
        return Fixed64(self.value - other.value)

    def __mul__(self, other: "Fixed64") -> "Fixed64":
        # (a * b) >> 32 to maintain Q32.32
        return Fixed64((self.value * other.value) >> Q)

    def __truediv__(self, other: "Fixed64") -> "Fixed64":
        if other.value == 0:
            raise ZeroDivisionError("division by zero")
        return Fixed64((self.value << Q) // other.value)

    def __lt__(self, other: "Fixed64") -> bool:
        return self.value < other.value

    def __le__(self, other: "Fixed64") -> bool:
        return self.value <= other.value

    def __gt__(self, other: "Fixed64") -> bool:
        return self.value > other.value

    def __ge__(self, other: "Fixed64") -> bool:
        return self.value >= other.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fixed64):
            return False
        return self.value == other.value

    def __repr__(self) -> str:
        return f"Fixed64({self.to_decimal_str(6)})"

    def as_bits_u64(self) -> int:
        """Return a 64-bit two's complement encoding (useful for cross-language interop).

        This is mainly illustrative in this Python prototype.
        """
        return self.value & MASK_64


def fixed_dot(weights: list[Fixed64], scores: list[Fixed64]) -> Fixed64:
    if len(weights) != len(scores):
        raise ValueError("weights and scores length mismatch")
    acc = 0
    for w, s in zip(weights, scores):
        acc += (w.value * s.value) >> Q
    return Fixed64(acc)


def ensure_fixed(obj: Any) -> Fixed64:
    if isinstance(obj, Fixed64):
        return obj
    if isinstance(obj, str):
        return Fixed64.from_decimal_str(obj)
    if isinstance(obj, int):
        return Fixed64.from_int(obj)
    raise TypeError(f"Cannot convert {type(obj)} to Fixed64")
