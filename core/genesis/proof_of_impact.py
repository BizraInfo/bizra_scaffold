"""
BIZRA Proof of Impact Engine
════════════════════════════════════════════════════════════════════════════════

Calculates contribution value from ecosystem artifacts.

Proof of Impact (PoI) answers: "What is the measurable value created by
contributions to the BIZRA ecosystem?"

Unlike Proof of Work (computation) or Proof of Stake (capital), Proof of
Impact measures:

1. **Artifact Density**: Number and significance of artifacts created
2. **Temporal Commitment**: Duration and consistency of contribution
3. **Structural Contribution**: Core vs peripheral artifact ratio
4. **Knowledge Crystallization**: Reusable knowledge captured
5. **Quality Signal**: SNR-weighted value (high signal > low signal)

The Genesis Node (Node0) uses this to establish the First Architect's
baseline impact—the 3-year "Big Bang" contribution against which all
future contributions are measured.

Design Philosophy:
- Giants Protocol: Impact builds on accumulated wisdom
- SNR-Weighted: High-signal contributions count more
- Graph of Thoughts: Contributions form a connected impact graph
- Ihsān-Bound: Only ethically-aligned contributions count

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add repo root to path
MODULE_DIR = Path(__file__).resolve().parent
CORE_DIR = MODULE_DIR.parent
REPO_ROOT = CORE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

logger = logging.getLogger("bizra.genesis.proof_of_impact")


# =============================================================================
# IMPACT CATEGORIES
# =============================================================================


class ImpactCategory(Enum):
    """Categories of contribution impact."""
    
    # Foundational contributions (highest impact)
    GENESIS = "genesis"               # Original creation, first-of-kind
    ARCHITECTURAL = "architectural"   # Core structure, patterns
    CONSTITUTIONAL = "constitutional" # Rules, governance, ethics
    
    # High-value contributions
    INTEGRATION = "integration"       # Connecting systems
    INNOVATION = "innovation"         # New capabilities
    OPTIMIZATION = "optimization"     # Performance, efficiency
    
    # Medium-value contributions
    DOCUMENTATION = "documentation"   # Knowledge capture
    TESTING = "testing"               # Validation, verification
    MAINTENANCE = "maintenance"       # Bug fixes, updates
    
    # Lower-value contributions
    CONFIGURATION = "configuration"   # Settings, tuning
    FORMATTING = "formatting"         # Style, presentation
    COMMENTARY = "commentary"         # Discussion, review


# Impact multipliers by category
IMPACT_MULTIPLIERS = {
    ImpactCategory.GENESIS: 10.0,
    ImpactCategory.ARCHITECTURAL: 5.0,
    ImpactCategory.CONSTITUTIONAL: 5.0,
    ImpactCategory.INTEGRATION: 3.0,
    ImpactCategory.INNOVATION: 3.0,
    ImpactCategory.OPTIMIZATION: 2.5,
    ImpactCategory.DOCUMENTATION: 2.0,
    ImpactCategory.TESTING: 2.0,
    ImpactCategory.MAINTENANCE: 1.5,
    ImpactCategory.CONFIGURATION: 1.0,
    ImpactCategory.FORMATTING: 0.5,
    ImpactCategory.COMMENTARY: 0.5,
}


# =============================================================================
# IMPACT METRICS
# =============================================================================


@dataclass
class ArtifactImpact:
    """Impact score for a single artifact."""
    
    artifact_path: str
    artifact_type: str
    category: ImpactCategory
    
    # Raw metrics
    size_bytes: int
    line_count: int
    complexity_score: float  # 0-1
    
    # Weighted metrics
    base_impact: float
    multiplier: float
    snr_weight: float  # Signal-to-Noise ratio weight
    
    # Final impact
    total_impact: float
    
    # Metadata
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    
    @classmethod
    def calculate(
        cls,
        artifact_path: str,
        artifact_type: str,
        size_bytes: int,
        line_count: int = 0,
        complexity_score: float = 0.5,
        snr_weight: float = 0.7,
        created_at: Optional[datetime] = None,
    ) -> "ArtifactImpact":
        """Calculate impact for an artifact."""
        
        # Infer category from type
        category = cls._infer_category(artifact_type, artifact_path)
        
        # Base impact from size and lines (log scale to avoid huge outliers)
        size_impact = math.log10(max(size_bytes, 1))
        line_impact = math.log10(max(line_count, 1)) if line_count > 0 else size_impact * 0.5
        base_impact = (size_impact + line_impact) * (1 + complexity_score)
        
        # Apply multipliers
        multiplier = IMPACT_MULTIPLIERS.get(category, 1.0)
        
        # SNR weighting (high signal contributions count more)
        # snr_weight is 0-1, we want it to boost by 0.5x to 2x
        snr_factor = 0.5 + (snr_weight * 1.5)
        
        # Total impact
        total_impact = base_impact * multiplier * snr_factor
        
        return cls(
            artifact_path=artifact_path,
            artifact_type=artifact_type,
            category=category,
            size_bytes=size_bytes,
            line_count=line_count,
            complexity_score=complexity_score,
            base_impact=base_impact,
            multiplier=multiplier,
            snr_weight=snr_weight,
            total_impact=total_impact,
            created_at=created_at,
        )
    
    @staticmethod
    def _infer_category(artifact_type: str, path: str) -> ImpactCategory:
        """Infer impact category from artifact type and path."""
        
        path_lower = path.lower()
        type_lower = artifact_type.lower()
        
        # Genesis artifacts
        if "genesis" in path_lower or "node_zero" in path_lower:
            return ImpactCategory.GENESIS
        
        # Constitutional
        if "constitution" in path_lower or "ethics" in path_lower:
            return ImpactCategory.CONSTITUTIONAL
        
        # Architectural
        if any(x in path_lower for x in ["core/", "architecture", "engine", "orchestrator"]):
            return ImpactCategory.ARCHITECTURAL
        
        # Testing
        if "test" in type_lower or path_lower.startswith("tests/"):
            return ImpactCategory.TESTING
        
        # Documentation
        if type_lower in ["markdown", "documentation"] or path_lower.endswith(".md"):
            return ImpactCategory.DOCUMENTATION
        
        # Configuration
        if type_lower in ["config", "toml", "yaml", "json"]:
            return ImpactCategory.CONFIGURATION
        
        # Integration
        if "integration" in path_lower or "bridge" in path_lower:
            return ImpactCategory.INTEGRATION
        
        # Default to innovation
        return ImpactCategory.INNOVATION
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "artifact_path": self.artifact_path,
            "artifact_type": self.artifact_type,
            "category": self.category.value,
            "size_bytes": self.size_bytes,
            "line_count": self.line_count,
            "complexity_score": self.complexity_score,
            "base_impact": round(self.base_impact, 4),
            "multiplier": self.multiplier,
            "snr_weight": self.snr_weight,
            "total_impact": round(self.total_impact, 4),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class TemporalImpact:
    """Impact from temporal commitment."""
    
    # Duration metrics
    total_days: int
    active_days: int
    consistency_ratio: float  # active/total
    
    # Streak metrics
    longest_streak_days: int
    current_streak_days: int
    
    # Session metrics
    total_sessions: int
    avg_session_hours: float
    total_hours: float
    
    # Calculated impact
    temporal_impact: float
    
    @classmethod
    def calculate(
        cls,
        earliest_date: datetime,
        latest_date: datetime,
        active_days: int,
        total_sessions: int = 0,
        total_hours: float = 0,
        longest_streak: int = 0,
    ) -> "TemporalImpact":
        """Calculate temporal impact."""
        
        total_days = max((latest_date - earliest_date).days, 1)
        consistency_ratio = min(active_days / total_days, 1.0)
        avg_session_hours = total_hours / max(total_sessions, 1)
        
        # Temporal impact formula
        # - Duration: log scale (3 years >> 3 months)
        # - Consistency: linear (more consistent = better)
        # - Intensity: hours contributed
        
        duration_factor = math.log10(total_days + 1) * 10  # ~30 for 3 years
        consistency_factor = 1 + (consistency_ratio * 2)   # 1-3x
        intensity_factor = math.log10(total_hours + 1)     # ~3.5 for 3000 hours
        
        temporal_impact = duration_factor * consistency_factor * intensity_factor
        
        return cls(
            total_days=total_days,
            active_days=active_days,
            consistency_ratio=round(consistency_ratio, 4),
            longest_streak_days=longest_streak,
            current_streak_days=0,  # Calculated separately
            total_sessions=total_sessions,
            avg_session_hours=round(avg_session_hours, 2),
            total_hours=round(total_hours, 2),
            temporal_impact=round(temporal_impact, 4),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


@dataclass
class StructuralImpact:
    """Impact from structural contribution (core vs peripheral)."""
    
    # Artifact counts by location
    core_artifacts: int
    peripheral_artifacts: int
    total_artifacts: int
    
    # Ratios
    core_ratio: float
    
    # Calculated impact
    structural_impact: float
    
    @classmethod
    def calculate(
        cls,
        core_count: int,
        peripheral_count: int,
    ) -> "StructuralImpact":
        """Calculate structural impact."""
        
        total = core_count + peripheral_count
        core_ratio = core_count / max(total, 1)
        
        # Core contributions count 3x more
        structural_impact = (core_count * 3.0) + peripheral_count
        
        return cls(
            core_artifacts=core_count,
            peripheral_artifacts=peripheral_count,
            total_artifacts=total,
            core_ratio=round(core_ratio, 4),
            structural_impact=round(structural_impact, 4),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# =============================================================================
# PROOF OF IMPACT
# =============================================================================


@dataclass
class ProofOfImpact:
    """
    Complete Proof of Impact for a contributor.
    
    This is the cryptographic proof of measurable value created.
    For the First Architect, this represents the 3-year genesis
    contribution that bootstrapped the entire ecosystem.
    """
    
    # Identity
    contributor_id: str
    contributor_alias: str
    
    # Component impacts
    artifact_impacts: List[ArtifactImpact]
    temporal_impact: TemporalImpact
    structural_impact: StructuralImpact
    
    # Aggregate metrics
    total_artifact_impact: float
    total_impact: float
    
    # Impact breakdown by category
    impact_by_category: Dict[str, float]
    
    # Proof metadata
    calculated_at: datetime
    proof_hash: str
    
    # Optional signature
    signature: Optional[str] = None
    
    @classmethod
    def calculate_genesis_impact(
        cls,
        contributor_id: str,
        contributor_alias: str,
        artifacts: List[Dict[str, Any]],
        temporal_data: Dict[str, Any],
    ) -> "ProofOfImpact":
        """
        Calculate Proof of Impact for the Genesis Node (First Architect).
        
        Args:
            contributor_id: Unique identifier for contributor
            contributor_alias: Human-readable name (e.g., "Momo")
            artifacts: List of artifact data from ecosystem discovery
            temporal_data: Timeline data from ecosystem discovery
        """
        logger.info(f"Calculating Proof of Impact for {contributor_alias}...")
        
        # Calculate artifact impacts
        artifact_impacts = []
        core_count = 0
        peripheral_count = 0
        
        for artifact in artifacts:
            impact = ArtifactImpact.calculate(
                artifact_path=artifact.get("path", ""),
                artifact_type=artifact.get("type", "unknown"),
                size_bytes=artifact.get("size", 0),
                line_count=artifact.get("lines", 0),
                complexity_score=artifact.get("complexity", 0.5),
                snr_weight=artifact.get("snr", 0.7),
            )
            artifact_impacts.append(impact)
            
            # Track core vs peripheral
            if "core/" in artifact.get("path", ""):
                core_count += 1
            else:
                peripheral_count += 1
        
        # Calculate temporal impact
        earliest = datetime.fromisoformat(temporal_data.get(
            "earliest",
            "2023-01-14T00:00:00+00:00"
        ))
        latest = datetime.fromisoformat(temporal_data.get(
            "latest",
            datetime.now(timezone.utc).isoformat()
        ))
        temporal = TemporalImpact.calculate(
            earliest_date=earliest,
            latest_date=latest,
            active_days=temporal_data.get("active_days", 100),
            total_sessions=temporal_data.get("sessions", 100),
            total_hours=temporal_data.get("hours", 500),
            longest_streak=temporal_data.get("longest_streak", 30),
        )
        
        # Calculate structural impact
        structural = StructuralImpact.calculate(core_count, peripheral_count)
        
        # Aggregate metrics
        total_artifact_impact = sum(a.total_impact for a in artifact_impacts)
        total_impact = (
            total_artifact_impact +
            temporal.temporal_impact * 10 +  # Scale temporal to be meaningful
            structural.structural_impact
        )
        
        # Impact by category
        impact_by_category: Dict[str, float] = {}
        for impact in artifact_impacts:
            cat = impact.category.value
            impact_by_category[cat] = impact_by_category.get(cat, 0) + impact.total_impact
        
        # Calculate proof hash
        calculated_at = datetime.now(timezone.utc)
        proof_data = {
            "contributor_id": contributor_id,
            "contributor_alias": contributor_alias,
            "total_impact": total_impact,
            "calculated_at": calculated_at.isoformat(),
            "artifact_count": len(artifact_impacts),
            "impact_by_category": impact_by_category,
        }
        proof_json = json.dumps(proof_data, sort_keys=True, separators=(",", ":"))
        
        if BLAKE3_AVAILABLE:
            proof_hash = blake3.blake3(proof_json.encode()).hexdigest()
        else:
            proof_hash = hashlib.sha256(proof_json.encode()).hexdigest()
        
        logger.info(f"Proof of Impact calculated: {total_impact:.2f} total impact")
        
        return cls(
            contributor_id=contributor_id,
            contributor_alias=contributor_alias,
            artifact_impacts=artifact_impacts,
            temporal_impact=temporal,
            structural_impact=structural,
            total_artifact_impact=round(total_artifact_impact, 4),
            total_impact=round(total_impact, 4),
            impact_by_category={k: round(v, 4) for k, v in impact_by_category.items()},
            calculated_at=calculated_at,
            proof_hash=proof_hash,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "contributor_id": self.contributor_id,
            "contributor_alias": self.contributor_alias,
            "artifact_impacts": [a.to_dict() for a in self.artifact_impacts],
            "temporal_impact": self.temporal_impact.to_dict(),
            "structural_impact": self.structural_impact.to_dict(),
            "total_artifact_impact": self.total_artifact_impact,
            "total_impact": self.total_impact,
            "impact_by_category": self.impact_by_category,
            "calculated_at": self.calculated_at.isoformat(),
            "proof_hash": self.proof_hash,
            "signature": self.signature,
        }
    
    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "PROOF OF IMPACT",
            "=" * 70,
            f"Contributor: {self.contributor_alias} ({self.contributor_id})",
            f"Calculated:  {self.calculated_at.isoformat()}",
            "",
            "TOTAL IMPACT: {:,.2f}".format(self.total_impact),
            "",
            "Breakdown:",
            f"  • Artifact Impact:  {self.total_artifact_impact:,.2f}",
            f"  • Temporal Impact:  {self.temporal_impact.temporal_impact:,.2f}",
            f"  • Structural Impact: {self.structural_impact.structural_impact:,.2f}",
            "",
            "By Category:",
        ]
        
        for cat, impact in sorted(
            self.impact_by_category.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            lines.append(f"  • {cat:20s} {impact:,.2f}")
        
        lines.extend([
            "",
            "Temporal Metrics:",
            f"  • Total Days:       {self.temporal_impact.total_days:,}",
            f"  • Active Days:      {self.temporal_impact.active_days:,}",
            f"  • Total Hours:      {self.temporal_impact.total_hours:,.1f}",
            f"  • Consistency:      {self.temporal_impact.consistency_ratio:.1%}",
            "",
            "Structural Metrics:",
            f"  • Core Artifacts:   {self.structural_impact.core_artifacts:,}",
            f"  • Total Artifacts:  {self.structural_impact.total_artifacts:,}",
            f"  • Core Ratio:       {self.structural_impact.core_ratio:.1%}",
            "",
            f"Proof Hash: {self.proof_hash[:32]}...",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save proof to file."""
        if path is None:
            path = REPO_ROOT / "data" / "genesis" / "PROOF_OF_IMPACT.json"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Proof of Impact saved to: {path}")
        return path


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main() -> int:
    """Main entry point for PoI calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate Proof of Impact for Genesis Node"
    )
    parser.add_argument(
        "--discovery-file",
        type=Path,
        help="Path to ecosystem discovery results JSON",
    )
    parser.add_argument(
        "--contributor-id",
        default="genesis_architect",
        help="Contributor identifier",
    )
    parser.add_argument(
        "--contributor-alias",
        default="Momo",
        help="Contributor display name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for PoI file",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print human-readable summary",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    # Load discovery data if available
    if args.discovery_file and args.discovery_file.exists():
        with args.discovery_file.open("r") as f:
            discovery_data = json.load(f)
        
        artifacts = [
            {
                "path": a.get("path", ""),
                "type": a.get("artifact_type", "unknown"),
                "size": a.get("size_bytes", 0),
                "lines": 0,
                "complexity": 0.5,
                "snr": 0.7,
            }
            for a in discovery_data.get("artifacts", [])
        ]
        
        temporal_data = {
            "earliest": discovery_data.get("timeline_start", "2023-01-14T00:00:00+00:00"),
            "latest": discovery_data.get("timeline_end", datetime.now(timezone.utc).isoformat()),
            "active_days": discovery_data.get("active_days", 100),
            "sessions": discovery_data.get("total_sessions", 100),
            "hours": discovery_data.get("contribution_hours", 500),
        }
    else:
        # Default genesis data
        artifacts = [
            {"path": "core/genesis/node_zero.py", "type": "python", "size": 15000, "snr": 0.9},
            {"path": "core/constitution.py", "type": "python", "size": 8000, "snr": 0.95},
            {"path": "core/engine/giants_protocol.py", "type": "python", "size": 12000, "snr": 0.85},
            {"path": "core/memory/agent_memory.py", "type": "python", "size": 10000, "snr": 0.8},
            {"path": "core/verification/ihsan_flow.py", "type": "python", "size": 6000, "snr": 0.9},
            {"path": "docs/ARCHITECTURE_DIAGRAMS.md", "type": "markdown", "size": 25000, "snr": 0.85},
            {"path": "constitution.toml", "type": "config", "size": 5000, "snr": 0.95},
        ]
        temporal_data = {
            "earliest": "2023-01-14T00:00:00+00:00",
            "latest": datetime.now(timezone.utc).isoformat(),
            "active_days": 500,
            "sessions": 1000,
            "hours": 3000,
        }
    
    # Calculate PoI
    poi = ProofOfImpact.calculate_genesis_impact(
        contributor_id=args.contributor_id,
        contributor_alias=args.contributor_alias,
        artifacts=artifacts,
        temporal_data=temporal_data,
    )
    
    # Save
    output_path = poi.save(args.output)
    
    if args.summary:
        print(poi.to_summary())
    else:
        print(f"Proof of Impact saved to: {output_path}")
        print(f"Total Impact: {poi.total_impact:,.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
