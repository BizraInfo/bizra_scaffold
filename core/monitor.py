from __future__ import annotations

import os
import time
from dataclasses import dataclass

import psutil


@dataclass
class NodePhysics:
    boot_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    ihsan_score: float


class PhysicsMonitor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PhysicsMonitor, cls).__new__(cls)
            cls._instance.start_time = time.perf_counter()
            cls._instance.process = psutil.Process(os.getpid())
        return cls._instance

    def capture_physics(self, ihsan_score: float) -> NodePhysics:
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        return NodePhysics(
            boot_latency_ms=(time.perf_counter() - self.start_time) * 1000,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            ihsan_score=ihsan_score,
        )
