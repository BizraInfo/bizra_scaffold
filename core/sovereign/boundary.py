"""
Sovereign Boundary Awareness
═══════════════════════════════════════════════════════════════════════════════
Knows and controls the physical and digital territory owned by BIZRA.

The sovereign kernel must know its boundaries to be truly sovereign.
This module provides boundary awareness and verification.
"""

from __future__ import annotations

import hashlib
import os
import platform
import psutil
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SovereignBoundary:
    """
    Physical and digital territory owned by MoMo (the sovereign architect).

    This defines what BIZRA owns and controls:
    - Hardware: CPU, GPU, memory, storage
    - Software: Operating system, installed packages
    - Data: All datasets and knowledge bases
    - Network: Local network segment
    """

    # Physical hardware sovereignty
    hardware_fingerprint: str = ""
    cpu_cores: int = 0
    total_memory_gb: float = 0.0
    gpu_devices: Dict[str, Any] = field(default_factory=dict)

    # Digital territory sovereignty
    sovereign_filesystem: str = "C:\\HERMES"  # MoMo's sovereign directory
    sovereign_network: str = "192.168.1.0/24"  # Local network segment

    # Cryptographic sovereignty
    sovereignty_keys: Dict[str, bytes] = field(default_factory=dict)

    # Runtime verification
    boundary_integrity: bool = False
    last_verified: Optional[str] = None

    def __post_init__(self):
        """Auto-discover boundary if not provided."""
        if not self.hardware_fingerprint:
            self._discover_hardware()
        if not self.sovereignty_keys:
            self._initialize_sovereignty_keys()

    def _discover_hardware(self) -> None:
        """Discover and fingerprint the physical hardware."""
        try:
            # CPU information
            self.cpu_cores = os.cpu_count() or 1

            # Memory information
            memory = psutil.virtual_memory()
            self.total_memory_gb = memory.total / (1024**3)

            # GPU detection (simplified)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    self.gpu_devices[f"gpu_{i}"] = {
                        "name": gpu.name,
                        "memory_gb": gpu.memoryTotal / 1024,
                        "uuid": gpu.uuid
                    }
            except ImportError:
                # Fallback GPU detection
                self.gpu_devices = {"status": "detection_unavailable"}

            # Create hardware fingerprint
            hw_data = f"{platform.node()}-{platform.machine()}-{self.cpu_cores}-{self.total_memory_gb}"
            self.hardware_fingerprint = hashlib.sha256(hw_data.encode()).hexdigest()

            logger.info(f"Discovered hardware: {self.cpu_cores} cores, {self.total_memory_gb:.1f}GB RAM")

        except Exception as e:
            logger.warning(f"Hardware discovery failed: {e}")
            self.hardware_fingerprint = "discovery_failed"

    def _initialize_sovereignty_keys(self) -> None:
        """Initialize cryptographic keys for sovereignty verification."""
        # Generate sovereignty verification keys
        import secrets
        self.sovereignty_keys = {
            "boundary_auth": secrets.token_bytes(32),
            "sovereignty_proof": secrets.token_bytes(32),
            "architect_auth": secrets.token_bytes(32),
        }
        logger.info("Initialized sovereignty cryptographic keys")

    def verify_hardware_sovereignty(self) -> bool:
        """Verify that BIZRA controls its declared hardware."""
        try:
            # Check if we can access CPU information
            current_cores = os.cpu_count() or 1
            if current_cores != self.cpu_cores:
                logger.warning(f"CPU core count changed: {self.cpu_cores} → {current_cores}")
                return False

            # Check memory access
            memory = psutil.virtual_memory()
            current_memory = memory.total / (1024**3)
            if abs(current_memory - self.total_memory_gb) > 1.0:  # Allow 1GB variance
                logger.warning(f"Memory changed: {self.total_memory_gb:.1f}GB → {current_memory:.1f}GB")
                return False

            return True

        except Exception as e:
            logger.error(f"Hardware sovereignty verification failed: {e}")
            return False

    def verify_filesystem_sovereignty(self) -> bool:
        """Verify that BIZRA controls its sovereign filesystem."""
        try:
            sovereign_path = Path(self.sovereign_filesystem)

            # Check if sovereign directory exists and is accessible
            if not sovereign_path.exists():
                logger.warning(f"Sovereign filesystem not found: {sovereign_path}")
                return False

            # Check if we can write to sovereign territory
            test_file = sovereign_path / ".sovereignty_test"
            try:
                test_file.write_text("sovereign_test")
                test_file.unlink()  # Clean up
                return True
            except Exception as e:
                logger.error(f"Cannot write to sovereign filesystem: {e}")
                return False

        except Exception as e:
            logger.error(f"Filesystem sovereignty verification failed: {e}")
            return False

    def verify_network_sovereignty(self) -> bool:
        """Verify that BIZRA controls its network segment."""
        try:
            # Get local IP
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)

            # Check if we're in the declared sovereign network
            # This is a simplified check - in production would verify network ownership
            if local_ip.startswith("192.168.1.") or local_ip.startswith("127."):
                return True
            else:
                logger.warning(f"Network sovereignty check: local IP {local_ip} not in sovereign segment")
                return False

        except Exception as e:
            logger.error(f"Network sovereignty verification failed: {e}")
            return False

    def verify_sovereignty(self) -> bool:
        """Comprehensive sovereignty verification."""
        from datetime import datetime, timezone

        hardware_ok = self.verify_hardware_sovereignty()
        filesystem_ok = self.verify_filesystem_sovereignty()
        network_ok = self.verify_network_sovereignty()

        self.boundary_integrity = hardware_ok and filesystem_ok and network_ok
        self.last_verified = datetime.now(timezone.utc).isoformat()

        status = "VERIFIED" if self.boundary_integrity else "VIOLATED"
        logger.info(f"Sovereign boundary {status}: hardware={hardware_ok}, fs={filesystem_ok}, net={network_ok}")

        return self.boundary_integrity

    def get_sovereign_manifest(self) -> Dict[str, Any]:
        """Get the complete sovereign boundary manifest."""
        return {
            "sovereignty_status": "ACTIVE" if self.boundary_integrity else "UNKNOWN",
            "hardware_fingerprint": self.hardware_fingerprint,
            "physical_assets": {
                "cpu_cores": self.cpu_cores,
                "memory_gb": self.total_memory_gb,
                "gpu_devices": self.gpu_devices,
            },
            "digital_territory": {
                "filesystem": self.sovereign_filesystem,
                "network_segment": self.sovereign_network,
            },
            "verification": {
                "last_verified": self.last_verified,
                "boundary_integrity": self.boundary_integrity,
            },
            "owner": "MoMo (Sovereign Architect)",
            "ownership_claim": "All hardware, software, and data in this boundary is owned by Mohamed Ahmed Beshr Elsayed Hassan (MoMo)",
        }


def verify_sovereignty() -> bool:
    """Global sovereignty verification function."""
    boundary = SovereignBoundary()
    return boundary.verify_sovereignty()


def get_sovereign_boundary() -> SovereignBoundary:
    """Get the current sovereign boundary configuration."""
    return SovereignBoundary()
