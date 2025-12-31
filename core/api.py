from fastapi import FastAPI
from core import __version__
from core.capabilities import detect_capabilities
from core.boot import _calculate_boot_vector
from core.kernel.pillars import KernelLaws

app = FastAPI(title="BIZRA Node Vanguard", version=__version__)


@app.get("/")
def root():
    return {
        "system": "BIZRA SOVEREIGN NODE",
        "release": "VANGUARD-GENESIS",
        "version": __version__,
        "message": "Ihsan Is All You Need.",
    }


@app.get("/status")
def get_sovereign_status():
    """
    Returns the real-time Ihsan/Physics state of the node.
    Aligned with boot.py telemetry.
    """
    caps = detect_capabilities()
    i_score = _calculate_boot_vector(caps)

    l3_mode = "FAISS" if (caps.faiss and not caps.force_lite) else "BASIC"

    l4_status = "DISABLED"
    if caps.neo4j and not caps.force_lite:
        import os
        if os.getenv("NEO4J_URI"):
            l4_status = "NEO4J (Configured)"
        else:
            l4_status = "NEO4J (Not Configured)"
    else:
        l4_status = "SQLITE (Lite)"

    return {
        "physics": {
            "ihsan_score": i_score,
            "ihsan_threshold": KernelLaws.IHSAN.MIN_SCORE_THRESHOLD,
            "gate_status": "OPEN" if i_score >= 0.95 else "CLOSED",
            "adl_invariant": "ACTIVE",
            "liveness": "ACTIVE",
        },
        "capabilities": {
            "lite_force": caps.force_lite,
            "crypto": caps.blake3,
            "numpy": caps.numpy,
            "faiss": caps.faiss,
            "neo4j": caps.neo4j,
            "z3": caps.z3,
        },
        "modes": {
            "l3_memory": l3_mode,
            "l4_graph": l4_status,
        },
    }
