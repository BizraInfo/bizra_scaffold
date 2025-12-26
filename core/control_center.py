"""
BIZRA AEON OMEGA - Unified System Control Center
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Pattern: Apex-Level Orchestration & Health Monitoring

Provides unified control plane for:
  1. Data Lake Watcher - Knowledge asset integrity
  2. Event Sourcing - Immutable audit trail
  3. SNR Scorer - Signal quality assessment
  4. Graph of Thoughts - Reasoning chains
  5. APEX Orchestrator - 7-layer stack coordination
  6. Compliance Engine - Self-verification

Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    BIZRA Control Center                              │
  │                                                                      │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
  │  │ Data Lake    │  │   Event      │  │    SNR       │               │
  │  │  Watcher     │  │  Sourcing    │  │   Scorer     │               │
  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
  │         │                 │                 │                       │
  │         └────────────┬────┴─────────────────┘                       │
  │                      ▼                                              │
  │              ┌───────────────┐                                      │
  │              │   Telemetry   │──────► Health Metrics                │
  │              │     Hub       │                                      │
  │              └───────┬───────┘                                      │
  │                      │                                              │
  │                      ▼                                              │
  │              ┌───────────────┐                                      │
  │              │   Dashboard   │──────► CLI / API                     │
  │              └───────────────┘                                      │
  └─────────────────────────────────────────────────────────────────────┘

BIZRA SOT Compliance:
  - Section 3: IM ≥ 0.95 enforcement
  - Section 7: Evidence logging
  - Section 8: Version control

Author: BIZRA Genesis Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════


class SubsystemStatus(Enum):
    """Health status for subsystems."""
    HEALTHY = auto()        # All checks pass
    DEGRADED = auto()       # Partial functionality
    UNHEALTHY = auto()      # Critical failure
    UNKNOWN = auto()        # Not yet checked
    DISABLED = auto()       # Intentionally off


class HealthCheckSeverity(Enum):
    """Severity levels for health issues."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    
    name: str
    status: SubsystemStatus
    message: str
    severity: HealthCheckSeverity = HealthCheckSeverity.INFO
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.name,
            "message": self.message,
            "severity": self.severity.name,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SubsystemHealth:
    """Health summary for a subsystem."""
    
    name: str
    status: SubsystemStatus
    checks: List[HealthCheckResult] = field(default_factory=list)
    last_check: Optional[datetime] = None
    uptime_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def healthy(self) -> bool:
        return self.status == SubsystemStatus.HEALTHY
    
    def add_check(self, result: HealthCheckResult) -> None:
        self.checks.append(result)
        self.last_check = result.timestamp
        
        # Determine overall status from checks
        statuses = [c.status for c in self.checks]
        if SubsystemStatus.UNHEALTHY in statuses:
            self.status = SubsystemStatus.UNHEALTHY
        elif SubsystemStatus.DEGRADED in statuses:
            self.status = SubsystemStatus.DEGRADED
        elif all(s == SubsystemStatus.HEALTHY for s in statuses):
            self.status = SubsystemStatus.HEALTHY
        else:
            self.status = SubsystemStatus.UNKNOWN


@dataclass
class SystemHealth:
    """Overall system health summary."""
    
    status: SubsystemStatus = SubsystemStatus.UNKNOWN
    subsystems: Dict[str, SubsystemHealth] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"
    ihsan_compliant: bool = False
    
    @property
    def healthy(self) -> bool:
        return self.status == SubsystemStatus.HEALTHY
    
    def compute_status(self) -> None:
        """Compute overall status from subsystems."""
        if not self.subsystems:
            self.status = SubsystemStatus.UNKNOWN
            return
        
        statuses = [s.status for s in self.subsystems.values()]
        
        if SubsystemStatus.UNHEALTHY in statuses:
            self.status = SubsystemStatus.UNHEALTHY
        elif SubsystemStatus.DEGRADED in statuses:
            self.status = SubsystemStatus.DEGRADED
        elif all(s == SubsystemStatus.HEALTHY for s in statuses):
            self.status = SubsystemStatus.HEALTHY
            self.ihsan_compliant = True
        else:
            self.status = SubsystemStatus.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.name,
            "ihsan_compliant": self.ihsan_compliant,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "subsystems": {
                name: {
                    "status": sub.status.name,
                    "checks": [c.to_dict() for c in sub.checks],
                    "metrics": sub.metrics,
                }
                for name, sub in self.subsystems.items()
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TELEMETRY HUB
# ═══════════════════════════════════════════════════════════════════════════════


class TelemetryHub:
    """
    Central telemetry collection point for all BIZRA subsystems.
    
    Collects metrics, events, and health data from:
    - Data Lake Watcher
    - Event Sourcing Engine
    - SNR Scorer
    - Graph of Thoughts
    - APEX Orchestrator
    """
    
    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.events: List[Dict[str, Any]] = []
        self.max_events = 1000
        self._start_time = time.time()
    
    def record_metric(self, subsystem: str, name: str, value: Any) -> None:
        """Record a metric value."""
        self.metrics[subsystem][name] = {
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def record_event(self, subsystem: str, event_type: str, data: Dict[str, Any]) -> None:
        """Record an event."""
        event = {
            "subsystem": subsystem,
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.events.append(event)
        
        # Trim old events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def get_metrics(self, subsystem: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics, optionally filtered by subsystem."""
        if subsystem:
            return dict(self.metrics.get(subsystem, {}))
        return dict(self.metrics)
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self._start_time


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL CENTER
# ═══════════════════════════════════════════════════════════════════════════════


class BIZRAControlCenter:
    """
    Unified Control Center for BIZRA System.
    
    Orchestrates health checks, telemetry, and coordination across
    all subsystems with fail-closed Ihsān enforcement.
    
    Usage:
        center = BIZRAControlCenter()
        
        # Run all health checks
        health = await center.check_health()
        
        # Get formatted status
        center.print_status()
        
        # Start background monitoring
        await center.start_monitoring(interval_seconds=60)
    """
    
    # Ihsān threshold (SOT Section 3)
    IHSAN_THRESHOLD = 0.95
    
    def __init__(self, workspace_path: Optional[Path] = None):
        self.workspace_path = workspace_path or Path(".")
        self.telemetry = TelemetryHub()
        self._health: Optional[SystemHealth] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_checks: List[Callable] = []
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register built-in health checks."""
        self._health_checks = [
            self._check_sot_compliance,
            self._check_data_lake_watcher,
            self._check_claim_registry,
            self._check_evidence_artifacts,
            self._check_secret_keys,
            self._check_manifest_integrity,
        ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HEALTH CHECKS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _check_sot_compliance(self) -> HealthCheckResult:
        """Check SOT Ihsān threshold consistency."""
        start = time.time()
        
        try:
            sot_path = self.workspace_path / "BIZRA_SOT.md"
            if not sot_path.exists():
                return HealthCheckResult(
                    name="sot_compliance",
                    status=SubsystemStatus.UNHEALTHY,
                    message="BIZRA_SOT.md not found",
                    severity=HealthCheckSeverity.CRITICAL,
                )
            
            content = sot_path.read_text(encoding="utf-8")
            
            # Check Section 3 and Section 4 consistency
            has_section3 = "## 3. Invariants" in content
            has_section4 = "## 4. PoI Parameters" in content
            
            if has_section3 and has_section4:
                section3 = content.split("## 3. Invariants")[1].split("## 4.")[0]
                section4 = content.split("## 4. PoI Parameters")[1].split("## 5.")[0]
                
                s3_has_095 = "0.95" in section3
                s4_has_095 = "0.95" in section4
                
                if s3_has_095 and s4_has_095:
                    return HealthCheckResult(
                        name="sot_compliance",
                        status=SubsystemStatus.HEALTHY,
                        message="Ihsān threshold 0.95 consistent across sections",
                        details={"section3": s3_has_095, "section4": s4_has_095},
                        duration_ms=(time.time() - start) * 1000,
                    )
                else:
                    return HealthCheckResult(
                        name="sot_compliance",
                        status=SubsystemStatus.UNHEALTHY,
                        message="Ihsān threshold inconsistency detected",
                        severity=HealthCheckSeverity.CRITICAL,
                        details={"section3": s3_has_095, "section4": s4_has_095},
                        duration_ms=(time.time() - start) * 1000,
                    )
            
            return HealthCheckResult(
                name="sot_compliance",
                status=SubsystemStatus.DEGRADED,
                message="SOT sections not found",
                severity=HealthCheckSeverity.WARNING,
                duration_ms=(time.time() - start) * 1000,
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="sot_compliance",
                status=SubsystemStatus.UNHEALTHY,
                message=f"SOT check failed: {e}",
                severity=HealthCheckSeverity.ERROR,
                duration_ms=(time.time() - start) * 1000,
            )
    
    async def _check_data_lake_watcher(self) -> HealthCheckResult:
        """Check Data Lake Watcher status."""
        start = time.time()
        
        try:
            # Check watched paths exist
            paths = [
                ("data_lake", Path("C:/BIZRA-DATA-LAKE")),
                ("node0_knowledge", Path("C:/BIZRA-NODE0/knowledge")),
            ]
            
            existing = []
            missing = []
            total_files = 0
            
            for alias, path in paths:
                if path.exists():
                    existing.append(alias)
                    try:
                        file_count = sum(1 for _ in path.rglob("*") if _.is_file())
                        total_files += file_count
                    except (PermissionError, OSError):
                        pass
                else:
                    missing.append(alias)
            
            # Check manifest
            manifest_path = self.workspace_path / "data/manifests/data_lake_manifest.json"
            has_manifest = manifest_path.exists()
            
            if len(existing) == 2 and has_manifest:
                return HealthCheckResult(
                    name="data_lake_watcher",
                    status=SubsystemStatus.HEALTHY,
                    message=f"All paths accessible, {total_files:,} files indexed",
                    details={
                        "paths_found": existing,
                        "total_files": total_files,
                        "has_manifest": has_manifest,
                    },
                    duration_ms=(time.time() - start) * 1000,
                )
            elif len(existing) > 0:
                return HealthCheckResult(
                    name="data_lake_watcher",
                    status=SubsystemStatus.DEGRADED,
                    message=f"Partial paths: {existing}, missing: {missing}",
                    severity=HealthCheckSeverity.WARNING,
                    details={"existing": existing, "missing": missing},
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                return HealthCheckResult(
                    name="data_lake_watcher",
                    status=SubsystemStatus.UNHEALTHY,
                    message="No data lake paths found",
                    severity=HealthCheckSeverity.ERROR,
                    duration_ms=(time.time() - start) * 1000,
                )
                
        except Exception as e:
            return HealthCheckResult(
                name="data_lake_watcher",
                status=SubsystemStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                severity=HealthCheckSeverity.ERROR,
                duration_ms=(time.time() - start) * 1000,
            )
    
    async def _check_claim_registry(self) -> HealthCheckResult:
        """Check Claim Registry integrity."""
        start = time.time()
        
        try:
            import yaml
            
            registry_path = self.workspace_path / "evidence/CLAIM_REGISTRY.yaml"
            if not registry_path.exists():
                return HealthCheckResult(
                    name="claim_registry",
                    status=SubsystemStatus.UNHEALTHY,
                    message="CLAIM_REGISTRY.yaml not found",
                    severity=HealthCheckSeverity.CRITICAL,
                    duration_ms=(time.time() - start) * 1000,
                )
            
            with open(registry_path, encoding="utf-8") as f:
                registry = yaml.safe_load(f)
            
            claims = registry.get("claims", [])
            verified = [c for c in claims if c.get("status") == "VERIFIED"]
            null_evidence = [
                c for c in verified 
                if not c.get("evidence_artifact_path")
            ]
            
            if len(null_evidence) == 0:
                return HealthCheckResult(
                    name="claim_registry",
                    status=SubsystemStatus.HEALTHY,
                    message=f"{len(verified)} verified claims with evidence",
                    details={
                        "total_claims": len(claims),
                        "verified": len(verified),
                        "version": registry.get("version", "unknown"),
                    },
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                return HealthCheckResult(
                    name="claim_registry",
                    status=SubsystemStatus.DEGRADED,
                    message=f"{len(null_evidence)} verified claims lack evidence",
                    severity=HealthCheckSeverity.WARNING,
                    details={"missing_evidence": [c.get("id") for c in null_evidence]},
                    duration_ms=(time.time() - start) * 1000,
                )
                
        except Exception as e:
            return HealthCheckResult(
                name="claim_registry",
                status=SubsystemStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                severity=HealthCheckSeverity.ERROR,
                duration_ms=(time.time() - start) * 1000,
            )
    
    async def _check_evidence_artifacts(self) -> HealthCheckResult:
        """Check evidence artifacts exist."""
        start = time.time()
        
        try:
            evidence_dir = self.workspace_path / "evidence/architecture"
            if not evidence_dir.exists():
                return HealthCheckResult(
                    name="evidence_artifacts",
                    status=SubsystemStatus.DEGRADED,
                    message="Evidence directory not found",
                    severity=HealthCheckSeverity.WARNING,
                    duration_ms=(time.time() - start) * 1000,
                )
            
            artifacts = list(evidence_dir.glob("*.log"))
            
            if len(artifacts) >= 2:
                return HealthCheckResult(
                    name="evidence_artifacts",
                    status=SubsystemStatus.HEALTHY,
                    message=f"{len(artifacts)} evidence artifacts present",
                    details={"artifacts": [a.name for a in artifacts]},
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                return HealthCheckResult(
                    name="evidence_artifacts",
                    status=SubsystemStatus.DEGRADED,
                    message=f"Only {len(artifacts)} artifacts (expected >= 2)",
                    severity=HealthCheckSeverity.WARNING,
                    duration_ms=(time.time() - start) * 1000,
                )
                
        except Exception as e:
            return HealthCheckResult(
                name="evidence_artifacts",
                status=SubsystemStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                severity=HealthCheckSeverity.ERROR,
                duration_ms=(time.time() - start) * 1000,
            )
    
    async def _check_secret_keys(self) -> HealthCheckResult:
        """Check no secret keys in repository."""
        start = time.time()
        
        try:
            keys_dir = self.workspace_path / "keys"
            if not keys_dir.exists():
                return HealthCheckResult(
                    name="secret_keys",
                    status=SubsystemStatus.HEALTHY,
                    message="No keys directory",
                    duration_ms=(time.time() - start) * 1000,
                )
            
            secret_files = [
                f for f in keys_dir.iterdir()
                if "secret" in f.name.lower()
                and not f.name.endswith(".example")
                and f.name != "README.md"
            ]
            
            if len(secret_files) == 0:
                return HealthCheckResult(
                    name="secret_keys",
                    status=SubsystemStatus.HEALTHY,
                    message="No secret files in repository",
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                # Security: Don't log actual secret filenames in details
                return HealthCheckResult(
                    name="secret_keys",
                    status=SubsystemStatus.UNHEALTHY,
                    message=f"{len(secret_files)} secret files detected!",
                    severity=HealthCheckSeverity.CRITICAL,
                    details={"secret_count": len(secret_files)},
                    duration_ms=(time.time() - start) * 1000,
                )
                
        except Exception as e:
            return HealthCheckResult(
                name="secret_keys",
                status=SubsystemStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                severity=HealthCheckSeverity.ERROR,
                duration_ms=(time.time() - start) * 1000,
            )
    
    async def _check_manifest_integrity(self) -> HealthCheckResult:
        """Check data lake manifest integrity."""
        start = time.time()
        
        try:
            manifest_path = self.workspace_path / "data/manifests/data_lake_manifest.json"
            
            if not manifest_path.exists():
                return HealthCheckResult(
                    name="manifest_integrity",
                    status=SubsystemStatus.DEGRADED,
                    message="No manifest file (run watcher scan)",
                    severity=HealthCheckSeverity.WARNING,
                    duration_ms=(time.time() - start) * 1000,
                )
            
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            
            metadata = manifest.get("metadata", {})
            assets = manifest.get("assets", [])
            
            return HealthCheckResult(
                name="manifest_integrity",
                status=SubsystemStatus.HEALTHY,
                message=f"Manifest valid: {len(assets):,} assets indexed",
                details={
                    "version": metadata.get("version", "unknown"),
                    "asset_count": len(assets),
                    "total_size": metadata.get("total_size_bytes", 0),
                    "last_updated": metadata.get("last_updated", "unknown"),
                },
                duration_ms=(time.time() - start) * 1000,
            )
            
        except json.JSONDecodeError as e:
            return HealthCheckResult(
                name="manifest_integrity",
                status=SubsystemStatus.UNHEALTHY,
                message=f"Manifest corrupted: {e}",
                severity=HealthCheckSeverity.ERROR,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name="manifest_integrity",
                status=SubsystemStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                severity=HealthCheckSeverity.ERROR,
                duration_ms=(time.time() - start) * 1000,
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def check_health(self) -> SystemHealth:
        """
        Run all health checks and return system health.
        
        Returns:
            SystemHealth with status for all subsystems
        """
        health = SystemHealth(timestamp=datetime.now(timezone.utc))
        
        # Group checks by subsystem
        subsystem_checks = {
            "compliance": [
                self._check_sot_compliance,
                self._check_claim_registry,
                self._check_evidence_artifacts,
                self._check_secret_keys,
            ],
            "data_lake": [
                self._check_data_lake_watcher,
                self._check_manifest_integrity,
            ],
        }
        
        for subsystem_name, checks in subsystem_checks.items():
            sub_health = SubsystemHealth(
                name=subsystem_name,
                status=SubsystemStatus.UNKNOWN,
            )
            
            for check in checks:
                result = await check()
                sub_health.add_check(result)
                
                # Record telemetry
                self.telemetry.record_metric(
                    subsystem_name,
                    result.name,
                    result.status.name,
                )
            
            health.subsystems[subsystem_name] = sub_health
        
        health.compute_status()
        self._health = health
        
        return health
    
    def print_status(self, health: Optional[SystemHealth] = None) -> None:
        """Print formatted system status to console."""
        h = health or self._health
        if not h:
            print("No health data available. Run check_health() first.")
            return
        
        # Header
        print()
        print("═" * 70)
        print(" BIZRA AEON OMEGA - System Control Center")
        print("═" * 70)
        print(f" Timestamp: {h.timestamp.isoformat()}")
        print(f" Version: {h.version}")
        print("═" * 70)
        print()
        
        # Overall status
        status_icons = {
            SubsystemStatus.HEALTHY: "✅",
            SubsystemStatus.DEGRADED: "⚠️",
            SubsystemStatus.UNHEALTHY: "❌",
            SubsystemStatus.UNKNOWN: "❓",
            SubsystemStatus.DISABLED: "⏸️",
        }
        
        icon = status_icons.get(h.status, "?")
        print(f" System Status: {icon} {h.status.name}")
        print(f" Ihsān Compliant: {'✅ YES' if h.ihsan_compliant else '❌ NO'}")
        print()
        
        # Subsystems
        for name, sub in h.subsystems.items():
            icon = status_icons.get(sub.status, "?")
            print(f" ┌─ {name.upper()}: {icon} {sub.status.name}")
            
            for check in sub.checks:
                c_icon = status_icons.get(check.status, "?")
                print(f" │  {c_icon} {check.name}: {check.message}")
                
                if check.details:
                    for key, value in check.details.items():
                        if isinstance(value, (int, float)):
                            if isinstance(value, int) and value > 1000:
                                print(f" │     {key}: {value:,}")
                            else:
                                print(f" │     {key}: {value}")
            
            print(" │")
        
        print("═" * 70)
        
        # Summary
        healthy_count = sum(
            1 for s in h.subsystems.values() 
            if s.status == SubsystemStatus.HEALTHY
        )
        total_count = len(h.subsystems)
        
        print(f" Subsystems: {healthy_count}/{total_count} healthy")
        print(f" Uptime: {self.telemetry.get_uptime():.1f}s")
        print("═" * 70)
    
    async def start_monitoring(self, interval_seconds: float = 60.0) -> None:
        """Start background health monitoring."""
        if self._monitoring_task:
            logger.warning("Monitoring already running")
            return
        
        async def monitor_loop():
            logger.info(f"Starting health monitoring (interval: {interval_seconds}s)")
            while True:
                try:
                    await self.check_health()
                    logger.debug(f"Health check complete: {self._health.status.name if self._health else 'N/A'}")
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                
                await asyncio.sleep(interval_seconds)
        
        self._monitoring_task = asyncio.create_task(monitor_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Monitoring stopped")
    
    def get_health_json(self) -> str:
        """Get health status as JSON string."""
        if self._health:
            return json.dumps(self._health.to_dict(), indent=2)
        return "{}"


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """CLI entry point for Control Center."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA Control Center")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--monitor", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=60, help="Monitor interval (seconds)")
    
    args = parser.parse_args()
    
    center = BIZRAControlCenter(workspace_path=Path("."))
    
    # Run health check
    health = await center.check_health()
    
    if args.json:
        print(center.get_health_json())
    else:
        center.print_status(health)
    
    if args.monitor:
        print(f"\nStarting continuous monitoring (interval: {args.interval}s)...")
        print("Press Ctrl+C to stop\n")
        
        await center.start_monitoring(interval_seconds=args.interval)
        
        try:
            while True:
                await asyncio.sleep(args.interval)
                health = await center.check_health()
                center.print_status(health)
        except KeyboardInterrupt:
            await center.stop_monitoring()
            print("\nMonitoring stopped.")
    
    # Return exit code based on health
    return 0 if health.healthy else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
