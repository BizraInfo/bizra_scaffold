from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
import os

from core import __version__
from core.boot import _calculate_boot_vector
from core.capabilities import detect_capabilities
from core.kernel.pillars import KernelLaws
from core.monitor import PhysicsMonitor


def get_config():
    from core.config import BIZRAConfig
    return BIZRAConfig()


app = FastAPI(
    title="BIZRA Node Vanguard",
    version=__version__,
    description="Sovereign Agentic Node API (Genesis Edition)",
)


class StatusResponse(BaseModel):
    system: str
    release: str
    version: str
    message: str


class TelemetryResponse(BaseModel):
    physics: Dict[str, Any]
    capabilities: Dict[str, bool]
    modes: Dict[str, str]


@app.get("/", response_model=StatusResponse)
def root():
    return {
        "system": "BIZRA SOVEREIGN NODE",
        "release": "VANGUARD-GENESIS",
        "version": __version__,
        "message": "Ihsan Is All You Need.",
    }


@app.get("/status", response_model=TelemetryResponse)
def get_sovereign_status():
    """
    Returns the real-time Ihsan/Physics state of the node.
    Aligned with boot.py telemetry and new capabilities.py schema.
    """
    caps = detect_capabilities()
    i_score = _calculate_boot_vector(caps)
    monitor = PhysicsMonitor()
    snapshot = monitor.capture_physics(i_score)

    l3_mode = "FAISS" if (caps.faiss and not caps.force_lite) else "BASIC"

    l4_status = "DISABLED"
    if caps.neo4j and not caps.force_lite:
        if os.getenv("NEO4J_URI"):
            l4_status = "NEO4J (Configured)"
        else:
            l4_status = "NEO4J (Not Configured)"
    else:
        l4_status = "SQLITE (Lite)"

    return {
        "physics": {
            "ihsan_score": snapshot.ihsan_score,
            "ihsan_threshold": KernelLaws.IHSAN.MIN_SCORE_THRESHOLD,
            "gate_status": "OPEN"
            if snapshot.ihsan_score >= KernelLaws.IHSAN.MIN_SCORE_THRESHOLD
            else "CLOSED",
            "adl_invariant": "ACTIVE",
            "liveness": "ACTIVE",
            "boot_latency_ms": snapshot.boot_latency_ms,
            "memory_usage_mb": snapshot.memory_usage_mb,
            "cpu_usage_percent": snapshot.cpu_usage_percent,
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


@app.get("/health")
def health_check():
    return {"status": "sovereign", "tick": True}
