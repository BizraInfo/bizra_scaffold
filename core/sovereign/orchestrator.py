"""
Unified Resource Orchestrator
═══════════════════════════════════════════════════════════════════════════════
Single point of control for all BIZRA resources.

This orchestrator unifies the fragmented resources (12+ models, CUDA hardware,
vast datasets) under sovereign control, replacing manual operation with
autonomous orchestration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class ModelResource:
    """Represents a single AI model resource."""
    name: str
    provider: str  # "ollama", "llmstudio", "cuda", "api"
    model_path: Optional[str] = None
    status: str = "unknown"  # "available", "busy", "offline", "unknown"
    capabilities: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def is_available(self) -> bool:
        """Check if model is available for use."""
        return self.status == "available"


@dataclass
class HardwareResource:
    """Represents hardware resources."""
    type: str  # "cpu", "gpu", "memory", "storage"
    total: float
    available: float
    unit: str  # "cores", "GB", "TB"

    @property
    def utilization_percent(self) -> float:
        """Calculate utilization percentage."""
        if self.total == 0:
            return 0.0
        return ((self.total - self.available) / self.total) * 100


@dataclass
class DataResource:
    """Represents data/knowledge resources."""
    name: str
    path: str
    size_gb: float
    format: str  # "json", "parquet", "sqlite", etc.
    last_updated: Optional[str] = None
    integrity_hash: Optional[str] = None


class SovereignOrchestrator:
    """
    Unified orchestrator for all BIZRA resources.

    This is the "kernel" that controls:
    - 12+ local models (Ollama, LLM Studio, CUDA)
    - Hardware resources (CPU, GPU, memory)
    - Data pipelines and knowledge bases
    - Autonomous operation replacing manual control
    """

    def __init__(self):
        self.models: Dict[str, ModelResource] = {}
        self.hardware: Dict[str, HardwareResource] = {}
        self.data_resources: Dict[str, DataResource] = {}

        # Orchestration state
        self.is_initialized = False
        self.active_operations: Dict[str, Dict[str, Any]] = {}

        # Resource monitoring thread
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

        logger.info("Sovereign Orchestrator initialized")

    def initialize_sovereign_control(self) -> bool:
        """Initialize sovereign control over all resources."""
        try:
            # Discover and inventory all resources
            self._discover_models()
            self._discover_hardware()
            self._discover_data_resources()

            # Start resource monitoring
            self._start_monitoring()

            self.is_initialized = True
            logger.info("Sovereign control established over all resources")
            return True

        except Exception as e:
            logger.error(f"Failed to establish sovereign control: {e}")
            return False

    def _discover_models(self) -> None:
        """Discover all available AI models."""
        logger.info("Discovering AI models...")

        # Ollama models
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Parse ollama output (simplified)
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 1:
                            model_name = parts[0]
                            self.models[model_name] = ModelResource(
                                name=model_name,
                                provider="ollama",
                                status="available",
                                capabilities=["text_generation", "chat"]
                            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Ollama not available or not found")

        # LLM Studio models (check for running instances)
        try:
            # This is a simplified check - in practice would query LLM Studio API
            import requests
            response = requests.get("http://localhost:1234/v1/models", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                for model_info in models_data.get("data", []):
                    model_name = model_info.get("id", "unknown")
                    self.models[model_name] = ModelResource(
                        name=model_name,
                        provider="llmstudio",
                        status="available",
                        capabilities=["text_generation", "chat", "completion"]
                    )
        except Exception:
            logger.warning("LLM Studio not available")

        # CUDA models (check for PyTorch CUDA availability)
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    device_name = torch.cuda.get_device_name(i)
                    model_name = f"cuda_{i}_{device_name.replace(' ', '_')}"
                    self.models[model_name] = ModelResource(
                        name=model_name,
                        provider="cuda",
                        status="available",
                        capabilities=["gpu_accelerated", "parallel_processing"]
                    )
        except ImportError:
            logger.warning("PyTorch not available for CUDA detection")

        logger.info(f"Discovered {len(self.models)} AI models")

    def _discover_hardware(self) -> None:
        """Discover hardware resources."""
        logger.info("Discovering hardware resources...")

        try:
            import psutil

            # CPU
            cpu_count = os.cpu_count() or 1
            self.hardware["cpu"] = HardwareResource(
                type="cpu",
                total=float(cpu_count),
                available=float(cpu_count),  # Simplified - in practice would check load
                unit="cores"
            )

            # Memory
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            self.hardware["memory"] = HardwareResource(
                type="memory",
                total=total_gb,
                available=available_gb,
                unit="GB"
            )

            # Storage (simplified - check current directory)
            stat = os.statvfs('.')
            total_bytes = stat.f_blocks * stat.f_frsize
            available_bytes = stat.f_available * stat.f_frsize
            total_tb = total_bytes / (1024**4)
            available_tb = available_bytes / (1024**4)

            self.hardware["storage"] = HardwareResource(
                type="storage",
                total=total_tb,
                available=available_tb,
                unit="TB"
            )

            # GPU
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_key = f"gpu_{i}"
                    self.hardware[gpu_key] = HardwareResource(
                        type="gpu",
                        total=gpu.memoryTotal,
                        available=gpu.memoryFree,
                        unit="MB"
                    )
            except ImportError:
                logger.warning("GPUtil not available for GPU detection")

        except Exception as e:
            logger.error(f"Hardware discovery failed: {e}")

        logger.info(f"Discovered {len(self.hardware)} hardware resources")

    def _discover_data_resources(self) -> None:
        """Discover data and knowledge resources."""
        logger.info("Discovering data resources...")

        # Check common data directories
        data_dirs = [
            Path("./data"),
            Path("./chat data sample"),
            Path("./research/vision_artifacts"),
            Path("./evidence"),
        ]

        for data_dir in data_dirs:
            if data_dir.exists():
                try:
                    total_size = sum(
                        f.stat().st_size for f in data_dir.rglob('*') if f.is_file()
                    )
                    size_gb = total_size / (1024**3)

                    if size_gb > 0.001:  # Only include non-empty directories
                        resource_name = data_dir.name
                        self.data_resources[resource_name] = DataResource(
                            name=resource_name,
                            path=str(data_dir),
                            size_gb=round(size_gb, 3),
                            format="mixed",  # Could be more specific
                            last_updated=None,  # Would need to check file timestamps
                        )
                except Exception as e:
                    logger.warning(f"Failed to analyze {data_dir}: {e}")

        logger.info(f"Discovered {len(self.data_resources)} data resources")

    def _start_monitoring(self) -> None:
        """Start background resource monitoring."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")

    def _monitor_resources(self) -> None:
        """Background monitoring of resources."""
        while self.monitoring_active:
            try:
                self._update_hardware_status()
                self._update_model_status()
                time.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(30)

    def _update_hardware_status(self) -> None:
        """Update hardware resource status."""
        try:
            import psutil

            # Update memory
            if "memory" in self.hardware:
                memory = psutil.virtual_memory()
                available_gb = memory.available / (1024**3)
                self.hardware["memory"].available = available_gb

            # Update storage
            if "storage" in self.hardware:
                stat = os.statvfs('.')
                available_bytes = stat.f_available * stat.f_frsize
                available_tb = available_bytes / (1024**4)
                self.hardware["storage"].available = available_tb

        except Exception as e:
            logger.debug(f"Hardware status update failed: {e}")

    def _update_model_status(self) -> None:
        """Update model availability status."""
        # This would ping each model provider to check availability
        # Simplified for now - assume models remain available
        pass

    def execute_sovereign_will(self, command: str, context: Dict[str, Any]) -> Any:
        """
        Execute commands with full resource orchestration.

        This is the main interface for sovereign operations.
        """
        if not self.is_initialized:
            raise RuntimeError("Sovereign orchestrator not initialized")

        logger.info(f"Executing sovereign will: {command}")

        # Route command to appropriate handler
        if command.startswith("model:"):
            return self._handle_model_command(command, context)
        elif command.startswith("data:"):
            return self._handle_data_command(command, context)
        elif command.startswith("system:"):
            return self._handle_system_command(command, context)
        else:
            return self._handle_general_command(command, context)

    def _handle_model_command(self, command: str, context: Dict[str, Any]) -> Any:
        """Handle model-related commands."""
        cmd_parts = command.split(":", 1)
        if len(cmd_parts) < 2:
            return {"error": "Invalid model command format"}

        action = cmd_parts[1]

        if action == "list":
            return {
                "models": [
                    {
                        "name": model.name,
                        "provider": model.provider,
                        "status": model.status,
                        "capabilities": model.capabilities
                    }
                    for model in self.models.values()
                ]
            }
        elif action == "status":
            return {
                "total_models": len(self.models),
                "available_models": sum(1 for m in self.models.values() if m.is_available()),
                "providers": list(set(m.provider for m in self.models.values()))
            }

        return {"error": f"Unknown model action: {action}"}

    def _handle_data_command(self, command: str, context: Dict[str, Any]) -> Any:
        """Handle data-related commands."""
        cmd_parts = command.split(":", 1)
        if len(cmd_parts) < 2:
            return {"error": "Invalid data command format"}

        action = cmd_parts[1]

        if action == "inventory":
            return {
                "data_resources": [
                    {
                        "name": data.name,
                        "path": data.path,
                        "size_gb": data.size_gb,
                        "format": data.format
                    }
                    for data in self.data_resources.values()
                ]
            }

        return {"error": f"Unknown data action: {action}"}

    def _handle_system_command(self, command: str, context: Dict[str, Any]) -> Any:
        """Handle system-related commands."""
        cmd_parts = command.split(":", 1)
        if len(cmd_parts) < 2:
            return {"error": "Invalid system command format"}

        action = cmd_parts[1]

        if action == "resources":
            return {
                "hardware": [
                    {
                        "type": hw.type,
                        "total": hw.total,
                        "available": hw.available,
                        "unit": hw.unit,
                        "utilization_percent": hw.utilization_percent
                    }
                    for hw in self.hardware.values()
                ],
                "models": len(self.models),
                "data_resources": len(self.data_resources)
            }

        return {"error": f"Unknown system action: {action}"}

    def _handle_general_command(self, command: str, context: Dict[str, Any]) -> Any:
        """Handle general sovereign commands."""
        # This would route to the integrated PAT/SAT system
        return {
            "command": command,
            "status": "routed_to_dual_agents",
            "context": context,
            "sovereign_control": "ACTIVE"
        }

    def get_sovereign_manifest(self) -> Dict[str, Any]:
        """Get complete sovereign resource manifest."""
        return {
            "sovereignty_status": "ESTABLISHED" if self.is_initialized else "INITIALIZING",
            "resources_controlled": {
                "models": len(self.models),
                "hardware_components": len(self.hardware),
                "data_resources": len(self.data_resources),
            },
            "monitoring_active": self.monitoring_active,
            "active_operations": len(self.active_operations),
            "sovereign_claim": "All resources unified under MoMo's sovereign control",
        }

    def shutdown(self) -> None:
        """Shutdown the orchestrator gracefully."""
        logger.info("Shutting down sovereign orchestrator")
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.is_initialized = False


# Global orchestrator instance
_orchestrator: Optional[SovereignOrchestrator] = None


def get_sovereign_orchestrator() -> SovereignOrchestrator:
    """Get the global sovereign orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SovereignOrchestrator()
        _orchestrator.initialize_sovereign_control()
    return _orchestrator


def execute_sovereign_command(command: str, context: Optional[Dict[str, Any]] = None) -> Any:
    """Execute a sovereign command through the orchestrator."""
    orchestrator = get_sovereign_orchestrator()
    return orchestrator.execute_sovereign_will(command, context or {})
