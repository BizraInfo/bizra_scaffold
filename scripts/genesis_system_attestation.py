#!/usr/bin/env python3
"""
BIZRA Genesis System Attestation

Host-centric discovery for Node0. This script binds:
  - Physical machine fingerprint (privacy-preserving hash)
  - Digital territories (known BIZRA roots on this host)
  - Conceptual declaration (genesis message)

Outputs:
  data/genesis/GENESIS_SYSTEM_MANIFEST.json
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

# Known BIZRA artifact locations on Node0
KNOWN_ROOTS = [
    r"C:\bizra_scaffold",
    r"C:\award-winner-design",
    r"C:\BIZRA-Dual-Agentic-system--main",
    r"C:\BIZRA-NODE0",
    r"C:\BIZRA-OS",
    r"C:\BIZRA-TaskMaster",
]


def _hash_text(text: str, digest_size: int = 32) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=digest_size).hexdigest()


def get_hardware_fingerprint() -> Dict[str, str]:
    """
    Generate a privacy-preserving hardware fingerprint.
    This does not store raw identifiers; it stores a one-way hash.
    """
    mac_raw = uuid.getnode()
    uname = platform.uname()
    sys_info = f"{uname.system}-{uname.node}-{uname.release}-{uname.version}-{uname.machine}"
    proc_info = platform.processor()

    raw_id = f"{mac_raw}|{sys_info}|{proc_info}"
    hardware_hash = _hash_text(raw_id, digest_size=32)

    return {
        "hostname": uname.node,
        "platform": uname.system,
        "architecture": uname.machine,
        "hardware_hash": hardware_hash,
    }


def scan_bizra_universe() -> List[Dict[str, Any]]:
    """
    Scan the host for known BIZRA roots and compute a lightweight territory hash.
    Hashing strategy: directory listing + mtimes (fast, non-invasive).
    """
    universe: List[Dict[str, Any]] = []
    for root_path in KNOWN_ROOTS:
        path = Path(root_path)
        if not path.exists():
            universe.append(
                {
                    "path": root_path,
                    "status": "MISSING",
                    "territory_hash": None,
                    "artifact_count": 0,
                }
            )
            continue

        dir_state = [root_path]
        artifact_count = 0
        try:
            for p in path.glob("*"):
                try:
                    stat = p.stat()
                    dir_state.append(f"{p.name}:{stat.st_mtime}")
                    artifact_count += 1
                except OSError:
                    continue
        except OSError:
            pass

        universe.append(
            {
                "path": root_path,
                "status": "ONLINE",
                "territory_hash": hashlib.sha256("".join(dir_state).encode("utf-8")).hexdigest()[:16],
                "artifact_count": artifact_count,
            }
        )

    return universe


def mint_genesis_block(owner_alias: str = "Momo") -> Path:
    """
    Bind hardware + territories into a genesis system manifest.
    """
    hw = get_hardware_fingerprint()
    territories = scan_bizra_universe()
    artifact_count = sum(t.get("artifact_count", 0) for t in territories)

    genesis_block: Dict[str, Any] = {
        "block_height": 0,
        "timestamp": time.time(),
        "timestamp_human": time.ctime(),
        "node_type": "GENESIS_NODE_ZERO",
        "sovereign_identity": {
            "architect": owner_alias,
            "machine_fingerprint": hw,
        },
        "ecosystem_state": territories,
        "artifact_count": artifact_count,
        "constitution_ver": "1.0",
        "message": "Standing on the shoulders of giants. The whole machine is the node.",
    }

    block_json = json.dumps(genesis_block, indent=2, sort_keys=True)
    genesis_hash = hashlib.blake2b(block_json.encode("utf-8"), digest_size=64).hexdigest()
    genesis_block["genesis_hash"] = genesis_hash

    out_path = Path("data") / "genesis" / "GENESIS_SYSTEM_MANIFEST.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(genesis_block, indent=2, sort_keys=True), encoding="utf-8")

    return out_path


def main() -> int:
    out_path = mint_genesis_block()
    print("[SUCCESS] Genesis system manifest written.")
    print(f"LOCATION: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
