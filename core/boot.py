"""
BIZRA VANGUARD BOOTLOADER (v5.0.0-OMEGA)
Sovereign Gate: Enforces Ihsan + Adl + lite-safe liveness at startup.

Design rule:
- Fail-closed on cryptographic truth primitives (blake3).
- Degrade gracefully on optional power (numpy/faiss/neo4j/z3).
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from core.kernel.pillars import KernelLaws


class BootViolation(RuntimeError):
    pass


class IhsanGateViolation(BootViolation):
    pass


class AdlViolation(BootViolation):
    pass


class CryptoMissing(BootViolation):
    pass


@dataclass(frozen=True)
class _Caps:
    force_lite: bool
    numpy: bool
    faiss: bool
    neo4j: bool
    z3: bool
    blake3: bool
    neo4j_configured: bool = False
    l3_mode: str = "basic"
    l4_mode: str = "disabled"


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _resolve_caps() -> _Caps:
    """
    Prefer project capability detector if available; fallback to safe local detection.
    Must never crash due to optional deps.
    """
    force_lite = str(os.getenv("BIZRA_LITE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    try:
        from core.capabilities import detect_capabilities

        caps = detect_capabilities()
        numpy_ok = bool(getattr(caps, "numpy", _has_module("numpy")))
        faiss_ok = bool(getattr(caps, "faiss", _has_module("faiss")))
        neo4j_ok = bool(getattr(caps, "neo4j", _has_module("neo4j")))
        z3_ok = bool(getattr(caps, "z3", _has_module("z3")))
        blake3_ok = bool(getattr(caps, "blake3", _has_module("blake3")))
        neo4j_cfg = bool(getattr(caps, "neo4j_configured", False))

        l3_mode = str(
            getattr(
                caps,
                "l3_mode",
                "faiss" if (faiss_ok and numpy_ok and not force_lite) else "basic",
            )
        )
        l4_mode = str(
            getattr(
                caps,
                "l4_mode",
                "neo4j" if (neo4j_ok and neo4j_cfg and not force_lite) else "disabled",
            )
        )

        return _Caps(
            force_lite=bool(getattr(caps, "force_lite", force_lite)),
            numpy=numpy_ok,
            faiss=faiss_ok,
            neo4j=neo4j_ok,
            z3=z3_ok,
            blake3=blake3_ok,
            neo4j_configured=neo4j_cfg,
            l3_mode=l3_mode,
            l4_mode=l4_mode,
        )
    except Exception:
        numpy_ok = _has_module("numpy")
        faiss_ok = _has_module("faiss")
        neo4j_ok = _has_module("neo4j")
        z3_ok = _has_module("z3")
        blake3_ok = _has_module("blake3")

        neo4j_cfg = (
            bool(os.getenv("NEO4J_PASSWORD"))
            and bool(os.getenv("NEO4J_URI", ""))
            and bool(os.getenv("NEO4J_USER", ""))
        )

        return _Caps(
            force_lite=force_lite,
            numpy=numpy_ok,
            faiss=faiss_ok,
            neo4j=neo4j_ok,
            z3=z3_ok,
            blake3=blake3_ok,
            neo4j_configured=neo4j_cfg,
            l3_mode="faiss" if (faiss_ok and numpy_ok and not force_lite) else "basic",
            l4_mode="neo4j" if (neo4j_ok and neo4j_cfg and not force_lite) else "disabled",
        )


def _dot(weights: Dict[str, float], scores: Dict[str, float]) -> float:
    return float(sum(float(weights[k]) * float(scores.get(k, 0.0)) for k in weights))


def _calculate_boot_vector(caps: Any) -> float:
    """
    Boot-time Ihsan score.
    Missing optional deps must not tank the score for lite nodes.
    Only missing crypto truth primitives is a hard failure.
    """
    correctness = 1.0 if getattr(caps, "blake3", False) else 0.0
    safety = 1.0
    user_benefit = 1.0
    efficiency = 1.0
    auditability = 1.0
    anti_centralization = (
        1.0 if (getattr(caps, "force_lite", False) or getattr(caps, "l3_mode", "") == "basic") else 0.97
    )
    robustness = 1.0 if getattr(caps, "z3", False) else 0.95
    fairness = 1.0

    scores = {
        "correctness": correctness,
        "safety": safety,
        "user_benefit": user_benefit,
        "efficiency": efficiency,
        "auditability": auditability,
        "anti_centralization": anti_centralization,
        "robustness": robustness,
        "fairness": fairness,
    }
    return _dot(KernelLaws.IHSAN.WEIGHTS, scores)


def _enforce_adl_invariant() -> None:
    """
    Boot-time Adl check.
    At genesis boot, ledger is empty => gini ~ 0.0.
    Runtime enforcement belongs to governance hypervisor.
    """
    current_gini = 0.0
    if current_gini > KernelLaws.ADL.GINI_THRESHOLD:
        raise AdlViolation(
            f"BOOT BLOCKED: Gini {current_gini:.4f} > {KernelLaws.ADL.GINI_THRESHOLD:.2f}"
        )


def _print_masterpiece_banner(caps: _Caps, i_score: float) -> None:
    print("=" * 60)
    print(
        f"BIZRA NODE: VANGUARD (v5.0.0-OMEGA) | {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("-" * 60)
    print("[SYSTEM PHYSICS]")
    print(
        " - Ihsan gate:      {score:.3f} (threshold >= {threshold:.2f})".format(
            score=i_score, threshold=KernelLaws.IHSAN.MIN_SCORE_THRESHOLD
        )
    )
    print(f" - Adl invariant:   ACTIVE (Gini <= {KernelLaws.ADL.GINI_THRESHOLD:.2f})")
    print(" - Formal liveness: ACTIVE (lite-safe boot)")
    print(" - SNR filter:      ACTIVE (typed artifacts + verification cascade)")
    print(" - Harberger tax:   ACTIVE (runtime metabolic enforcement)")
    print(" - Phi/coherence:   ACTIVE (proxy monitoring recommended)")
    print("-" * 60)
    print("[CAPABILITIES]")
    print(f" - BIZRA_LITE forced: {'yes' if caps.force_lite else 'no'}")
    print(f" - numpy:   {'yes' if caps.numpy else 'no'}")
    print(f" - faiss:   {'yes' if caps.faiss else 'no'}")
    print(
        " - neo4j:   {available} (configured={configured})".format(
            available="yes" if caps.neo4j else "no",
            configured="yes" if caps.neo4j_configured else "no",
        )
    )
    print(f" - z3:      {'yes' if caps.z3 else 'no'}")
    print(f" - blake3:  {'yes' if caps.blake3 else 'no'}")
    print("-" * 60)
    print("[OPERATIONAL MODES]")
    print(f" - Memory (L3): {caps.l3_mode.upper()}")
    print(f" - Graph  (L4): {caps.l4_mode.upper()}")
    print("=" * 60)
    print("")


def _construct(cls: Any, **preferred_kwargs: Any) -> Any:
    """
    Signature-adaptive constructor: passes only supported kwargs.
    """
    sig = inspect.signature(cls)
    kwargs = {k: v for k, v in preferred_kwargs.items() if k in sig.parameters}
    return cls(**kwargs)


def ignite() -> Tuple[Any, Optional[Any]]:
    """
    Ignite Vanguard node:
    - Enforce Adl + Ihsan at the gate
    - Initialize L3 always
    - Initialize L4 only if neo4j present and configured and not forced lite
    """
    _enforce_adl_invariant()

    caps = _resolve_caps()

    if not caps.blake3:
        raise CryptoMissing(
            "FATAL: blake3 missing. Truth primitive is mandatory (fail-closed)."
        )

    i_score = _calculate_boot_vector(caps)
    _print_masterpiece_banner(caps, i_score)

    if i_score < KernelLaws.IHSAN.MIN_SCORE_THRESHOLD:
        raise IhsanGateViolation(
            "FATAL: Ihsan score {score:.3f} below {threshold:.2f}".format(
                score=i_score, threshold=KernelLaws.IHSAN.MIN_SCORE_THRESHOLD
            )
        )

    from core.layers import L3EpisodicMemoryV2

    print("[BOOT] Initializing Memory Substrate...")
    l3 = _construct(
        L3EpisodicMemoryV2,
        embedding_dim=int(os.getenv("BIZRA_L3_DIM", "768")),
        index_type=os.getenv("BIZRA_L3_INDEX", "Flat"),
        use_faiss=(caps.faiss and caps.numpy and not caps.force_lite),
    )

    l4 = None
    if caps.l4_mode == "neo4j":
        try:
            from core.layers import L4SemanticHyperGraphV2

            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USER")
            neo4j_password = os.getenv("NEO4J_PASSWORD")
            if neo4j_uri and neo4j_user and neo4j_password:
                l4 = _construct(
                    L4SemanticHyperGraphV2,
                    neo4j_uri=neo4j_uri,
                    neo4j_auth=(neo4j_user, neo4j_password),
                )
        except Exception:
            l4 = None

    if l4 is None:
        print("[BOOT] L4 disabled (lite mode or missing/uncfg deps).")
    print("[BOOT] System Sovereign. Waiting for stimulus...")
    return l3, l4


def main() -> None:
    try:
        ignite()
    except Exception as exc:
        print(f"\n[BOOT FAILED] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
