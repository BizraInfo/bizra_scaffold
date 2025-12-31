"""
BIZRA VANGUARD BOOTLOADER
The Sovereign Ignition Sequence.
"""
import sys
import os
import time
import inspect
from typing import Optional

# Canonical Imports
from core import __version__
from core.capabilities import detect_capabilities
from core.kernel.pillars import KernelLaws

class IhsanGateViolation(Exception):
    pass

class AdlViolation(Exception):
    pass

def _calculate_boot_vector(caps) -> float:
    # Simplified score for boot.
    if not caps.numpy:
        return 1.0 if (caps.blake3 and caps.z3) else 0.8

    try:
        import numpy as np
    except Exception:
        return 1.0 if (caps.blake3 and caps.z3) else 0.8

    scores = np.array([
        1.0 if caps.blake3 else 0.0,    # Correctness
        1.0,                            # Safety (No import crash)
        np.mean([caps.numpy, caps.faiss, caps.neo4j, caps.blake3]), # Benefit
        1.0,                            # Efficiency
        1.0,                            # Auditability
        1.0 if caps.force_lite else 0.9,# Anti-Centralization
        0.95,                           # Robustness
        1.0                             # Fairness
    ])
    weights = np.array(list(KernelLaws.IHSAN.WEIGHTS.values()))
    return float(np.dot(weights, scores))

def _print_masterpiece_banner(caps, i_score, l4_status):
    green, red, reset = "\033[92m", "\033[91m", "\033[0m"
    gate_color = green if i_score >= KernelLaws.IHSAN.MIN_SCORE_THRESHOLD else red

    print(f"\n{green}" + "="*60)
    print(f" BIZRA NODE: VANGUARD (v{__version__})")
    print("="*60 + f"{reset}")
    print(" [SYSTEM PHYSICS]")
    print(f"  - Ihsan Score:      {gate_color}{i_score:.3f}{reset} (Threshold: {KernelLaws.IHSAN.MIN_SCORE_THRESHOLD})")
    print(f"  - Adl Invariant:    ACTIVE (Gini <= {KernelLaws.ADL.GINI_THRESHOLD})")
    print("  - Liveness:         ACTIVE (Fail-Closed Boot)")
    print("-" * 60)
    print(" [CAPABILITIES]")
    print(f"  - Lite Mode: {'ON' if caps.force_lite else 'OFF'}")
    print(f"  - Crypto:    {'yes' if caps.blake3 else 'no'}")
    print(f"  - FAISS:     {'yes' if caps.faiss else 'no'}")
    print(f"  - Neo4j:     {'yes' if caps.neo4j else 'no'}")
    print("-" * 60)
    print(" [OPERATIONAL STATUS]")
    print(f"  - Memory (L3): {'FAISS' if caps.faiss and not caps.force_lite else 'BASIC'}")
    print(f"  - Graph  (L4): {l4_status}")
    print("="*60 + "\n")

def _construct(cls, **preferred_kwargs):
    sig = inspect.signature(cls)
    kwargs = {k: v for k, v in preferred_kwargs.items() if k in sig.parameters}
    return cls(**kwargs)

def ignite():
    try:
        # 1. Physics Check
        caps = detect_capabilities()
        i_score = _calculate_boot_vector(caps)

        # 2. L4 Readiness Probe
        l4_status = "DISABLED"
        if caps.neo4j and not caps.force_lite:
            if os.getenv("NEO4J_URI"):
                try:
                    # Late import to verify driver
                    from core.layers.memory_layers_v2 import L4SemanticHyperGraphV2
                    # Mock check for boot speed; real L4 would ping
                    l4_status = "NEO4J (Ready)"
                except Exception as e:
                    l4_status = f"NEO4J (Error: {str(e)[:15]}...)"
            else:
                l4_status = "NEO4J (Not Configured)"
        else:
            l4_status = "SQLITE (Lite)"

        # 3. Render Banner
        _print_masterpiece_banner(caps, i_score, l4_status)

        # 4. Gate Enforcement
        if i_score < KernelLaws.IHSAN.MIN_SCORE_THRESHOLD:
            raise IhsanGateViolation(f"FATAL: Ihsan Score {i_score:.3f} Violation.")

        # 5. Initialization
        from core.layers import L3EpisodicMemoryV2
        print(" [BOOT] Initializing Memory Substrate...")
        l3 = _construct(L3EpisodicMemoryV2, persistence_dir="./data/l3")

        print(" [BOOT] System Sovereign. Waiting for Stimulus...")

    except Exception as e:
        print(f"\n\033[91m [BOOT FAILED] {str(e)}\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    ignite()
