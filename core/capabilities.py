from __future__ import annotations
import importlib.util
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Capabilities:
    numpy: bool
    faiss: bool
    neo4j: bool
    blake3: bool
    z3: bool
    force_lite: bool

def _safe_has(name: str) -> bool:
    """Safely checks module existence without importing/crashing."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False

def detect_capabilities() -> Capabilities:
    # 1. Check Lite Override
    force_lite = os.getenv("BIZRA_LITE", "").lower() in ("1", "true", "yes", "on")

    # 2. Physics Core (Always checked)
    has_blake3 = _safe_has("blake3")
    has_z3 = _safe_has("z3")

    # 3. Heavy Subsystems (Skipped in Lite Mode)
    if force_lite:
        return Capabilities(
            numpy=False,
            faiss=False,
            neo4j=False,
            blake3=has_blake3,
            z3=has_z3,
            force_lite=True,
        )

    return Capabilities(
        numpy=_safe_has("numpy"),
        faiss=_safe_has("faiss"),
        neo4j=_safe_has("neo4j"),
        blake3=has_blake3,
        z3=has_z3,
        force_lite=False,
    )
