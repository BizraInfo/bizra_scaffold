from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os

# Canonical Imports
from core import __version__
from core.capabilities import detect_capabilities
from core.boot import _calculate_boot_vector
from core.kernel.pillars import KernelLaws

# LAZY CONFIG LOADING (Fixes High Severity Import Crash)
# We do not import 'config' at module level to prevent validation errors during imports.
def get_config():
    from core.config import BIZRAConfig
    return BIZRAConfig()

# SINGLE ENTRY POINT
app = FastAPI(
    title="BIZRA Node Vanguard",
    version=__version__,
    description="Sovereign Agentic Node API (Genesis Edition)"
)

# --- Data Models ---
class StatusResponse(BaseModel):
    system: str
    release: str
    version: str
    message: str

class TelemetryResponse(BaseModel):
    physics: Dict[str, Any]
    capabilities: Dict[str, bool]
    modes: Dict[str, str]

# --- Routes ---

@app.get("/", response_model=StatusResponse)
def root():
    return {
        "system": "BIZRA SOVEREIGN NODE",
        "release": "VANGUARD-GENESIS",
        "version": __version__,
        "message": "Ihsan Is All You Need."
    }

@app.get("/status", response_model=TelemetryResponse)
def get_sovereign_status():
    """
    Returns the real-time Ihsan/Physics state of the node.
    Aligned with boot.py telemetry and new capabilities.py schema.
    """
    caps = detect_capabilities()
    i_score = _calculate_boot_vector(caps)

    # 1. Resolve L3 Mode
    l3_mode = "FAISS" if (caps.faiss and not caps.force_lite) else "BASIC"

    # 2. Resolve L4 Mode
    l4_status = "DISABLED"
    if caps.neo4j and not caps.force_lite:
        # Check config presence safely
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
            "gate_status": "OPEN" if i_score >= KernelLaws.IHSAN.MIN_SCORE_THRESHOLD else "CLOSED",
            "adl_invariant": "ACTIVE",
            "liveness": "ACTIVE"
        },
        "capabilities": {
            "lite_force": caps.force_lite,
            "crypto": caps.blake3,
            "numpy": caps.numpy,
            "faiss": caps.faiss,
            "neo4j": caps.neo4j,
            "z3": caps.z3
        },
        "modes": {
            "l3_memory": l3_mode,
            "l4_graph": l4_status
        }
    }

# --- Health Check ---
@app.get("/health")
def health_check():
    return {"status": "sovereign", "tick": True}
