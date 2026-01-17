"""Deterministic JSON canonicalization.

BIZRA's mature design mentions JCS canonicalization for cross-platform hashing.
RFC 8785 (JCS) is strict about number formatting (IEEE-754) and UTF-8 output.

This prototype intentionally constrains JSON values to:
  - dict[str, ...]
  - list[...]
  - str
  - int
  - bool
  - None

and produces a stable, whitespace-free encoding.

For these value types, Python's json output with sort_keys=True and separators
is deterministic and effectively JCS-compatible for our purposes.
"""

from __future__ import annotations

import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Return a canonical JSON string.

    - dict keys sorted lexicographically
    - no extra whitespace
    - UTF-8 friendly (ensure_ascii=False)
    - disallow NaN/Infinity

    NOTE: We avoid floats entirely in the simulation to keep determinism.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")
