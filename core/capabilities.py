"""
BIZRA CAPABILITIES (v5.0.0)
Detects optional dependencies + config readiness without crashing boot.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


def _try_import(module_name: str) -> Tuple[bool, Optional[str]]:
    """
    Returns (available, error_message). Catches all import-time failures,
    including binary/OS errors (common with faiss wheels).
    """
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:  # noqa: BLE001 - intentional: harden boot
        return False, f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class Capabilities:
    # Optional deps
    numpy: bool
    faiss: bool
    neo4j: bool
    z3: bool
    blake3: bool

    # Config / intent
    force_lite: bool
    neo4j_configured: bool

    # Diagnostics (strings so printing never crashes)
    import_errors: Dict[str, str]

    @property
    def l3_mode(self) -> str:
        if self.force_lite:
            return "basic"
        return "faiss" if (self.faiss and self.numpy) else "basic"

    @property
    def l4_mode(self) -> str:
        if self.force_lite:
            return "disabled"
        return "neo4j" if (self.neo4j and self.neo4j_configured) else "disabled"


def detect_capabilities(env: Optional[Dict[str, str]] = None) -> Capabilities:
    env = env or dict(os.environ)

    force_lite = str(env.get("BIZRA_LITE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    neo4j_configured = bool(env.get("NEO4J_PASSWORD")) or bool(env.get("NEO4J_URI"))

    import_errors: Dict[str, str] = {}

    numpy_ok, numpy_err = _try_import("numpy")
    if numpy_err:
        import_errors["numpy"] = numpy_err

    faiss_ok, faiss_err = _try_import("faiss")
    if faiss_err:
        import_errors["faiss"] = faiss_err

    neo4j_ok, neo4j_err = _try_import("neo4j")
    if neo4j_err:
        import_errors["neo4j"] = neo4j_err

    z3_ok, z3_err = _try_import("z3")
    if z3_err:
        import_errors["z3"] = z3_err

    blake3_ok, blake3_err = _try_import("blake3")
    if blake3_err:
        import_errors["blake3"] = blake3_err

    return Capabilities(
        numpy=numpy_ok,
        faiss=faiss_ok,
        neo4j=neo4j_ok,
        z3=z3_ok,
        blake3=blake3_ok,
        force_lite=force_lite,
        neo4j_configured=neo4j_configured,
        import_errors=import_errors,
    )
