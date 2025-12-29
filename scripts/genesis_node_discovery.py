#!/usr/bin/env python3
"""
BIZRA Genesis Node Discovery Engine
════════════════════════════════════════════════════════════════════════════════

Scans the Genesis Node (this machine) to discover all BIZRA ecosystem artifacts:
- Repositories and worktrees
- Chat data exports (OpenAI, Claude, multi-model)
- Configuration files and keys
- Evidence packs and attestations
- Knowledge graphs and derived artifacts

This is NOT a codebase scanner—it's an ECOSYSTEM mapper.
The Genesis Node is the physical home of BIZRA: hardware, software, data.

Design Philosophy:
- Giants Protocol: Extract wisdom from the full 3-year corpus
- SNR-Weighted: Prioritize high-signal artifacts, deprioritize noise
- Graph of Thoughts: Map relationships between artifacts, not just list them
- Proof of Impact: Every artifact contributes to the First Architect's PoI

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

logger = logging.getLogger("bizra.genesis.discovery")


# =============================================================================
# ARTIFACT TAXONOMY
# =============================================================================


class ArtifactType(Enum):
    """Classification of BIZRA ecosystem artifacts."""
    
    # Source artifacts
    REPOSITORY = auto()          # Git repository
    WORKTREE = auto()            # Git worktree
    PYTHON_MODULE = auto()       # Python source file
    RUST_CRATE = auto()          # Rust crate
    
    # Data artifacts
    CHAT_EXPORT = auto()         # LLM conversation export
    KNOWLEDGE_GRAPH = auto()     # Concept/relationship graph
    EVIDENCE_PACK = auto()       # Cryptographic evidence bundle
    CHECKPOINT = auto()          # Model/state checkpoint
    
    # Configuration artifacts
    CONSTITUTION = auto()        # constitution.toml
    POLICY = auto()              # Policy definition
    KEY_MATERIAL = auto()        # Cryptographic keys (SENSITIVE)
    CONFIG = auto()              # Configuration file
    
    # Derived artifacts
    ATTESTATION = auto()         # Signed attestation
    RECEIPT = auto()             # Commit/verification receipt
    MANIFEST = auto()            # File manifest
    
    # Documentation
    SPECIFICATION = auto()       # Technical specification
    ROADMAP = auto()             # Planning document
    EVIDENCE_DOC = auto()        # Evidence documentation


class ArtifactSignificance(Enum):
    """SNR-weighted significance of artifacts."""
    
    GENESIS = 5      # Core genesis artifacts (constitution, first commits)
    CRITICAL = 4     # Critical infrastructure (crypto, attestations)
    HIGH = 3         # High-value artifacts (knowledge graphs, evidence)
    MEDIUM = 2       # Standard artifacts (source code, configs)
    LOW = 1          # Low-signal artifacts (logs, temp files)
    NOISE = 0        # Noise (should be filtered)


@dataclass
class DiscoveredArtifact:
    """A discovered artifact in the BIZRA ecosystem."""
    
    path: str
    artifact_type: ArtifactType
    significance: ArtifactSignificance
    size_bytes: int
    modified_at: datetime
    created_at: Optional[datetime] = None
    
    # Content fingerprint
    blake3_hash: Optional[str] = None
    sha256_hash: Optional[str] = None
    
    # Relationships
    parent_repo: Optional[str] = None
    related_artifacts: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "path": self.path,
            "artifact_type": self.artifact_type.name,
            "significance": self.significance.name,
            "significance_score": self.significance.value,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at.isoformat(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "blake3_hash": self.blake3_hash,
            "sha256_hash": self.sha256_hash,
            "parent_repo": self.parent_repo,
            "related_artifacts": self.related_artifacts,
            "metadata": self.metadata,
        }


@dataclass
class GenesisNodeProfile:
    """Complete profile of the Genesis Node ecosystem."""
    
    node_id: str
    hostname: str
    platform: str
    discovery_timestamp: datetime
    
    # Discovered artifacts
    artifacts: List[DiscoveredArtifact] = field(default_factory=list)
    
    # Aggregate statistics
    total_artifacts: int = 0
    total_size_bytes: int = 0
    total_repositories: int = 0
    total_chat_messages: int = 0
    
    # Timeline bounds
    earliest_artifact: Optional[datetime] = None
    latest_artifact: Optional[datetime] = None
    
    # Ecosystem hash (covers all artifacts)
    ecosystem_root_hash: Optional[str] = None
    
    # Proof of Impact inputs
    contribution_hours_estimated: float = 0.0
    artifact_density: float = 0.0
    conceptual_nodes: int = 0
    conceptual_edges: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "platform": self.platform,
            "discovery_timestamp": self.discovery_timestamp.isoformat(),
            "total_artifacts": self.total_artifacts,
            "total_size_bytes": self.total_size_bytes,
            "total_repositories": self.total_repositories,
            "total_chat_messages": self.total_chat_messages,
            "earliest_artifact": self.earliest_artifact.isoformat() if self.earliest_artifact else None,
            "latest_artifact": self.latest_artifact.isoformat() if self.latest_artifact else None,
            "ecosystem_root_hash": self.ecosystem_root_hash,
            "contribution_hours_estimated": self.contribution_hours_estimated,
            "artifact_density": self.artifact_density,
            "conceptual_nodes": self.conceptual_nodes,
            "conceptual_edges": self.conceptual_edges,
            "artifacts": [a.to_dict() for a in self.artifacts],
        }


# =============================================================================
# DISCOVERY PATTERNS
# =============================================================================


# Known BIZRA-related directory patterns
BIZRA_PATTERNS = [
    r"bizra",
    r"scaffold",
    r"genesis",
    r"pat[-_]?sat",
    r"pci[-_]?envelope",
    r"ihsan",
    r"sot",  # Source of Truth
]

# Known chat data patterns
CHAT_DATA_PATTERNS = [
    r"chat[-_]?data",
    r"openai[-_]?export",
    r"claude[-_]?export",
    r"conversation",
    r"message",
    r"chat[-_]?history",
]

# File patterns by type
ARTIFACT_PATTERNS: Dict[ArtifactType, List[str]] = {
    ArtifactType.CONSTITUTION: [r"constitution\.toml$"],
    ArtifactType.KNOWLEDGE_GRAPH: [r".*graph.*\.json$", r".*graph.*\.graphml$"],
    ArtifactType.EVIDENCE_PACK: [r"EVIDENCE.*\.json$", r"PACK-.*"],
    ArtifactType.ATTESTATION: [r"attestation.*\.json$", r".*\.attestation$"],
    ArtifactType.RECEIPT: [r"receipt.*\.json$", r"commit[-_]?receipt"],
    ArtifactType.MANIFEST: [r"manifest.*\.json$", r"MANIFEST"],
    ArtifactType.SPECIFICATION: [r"PROTOCOL\.md$", r".*_SPEC.*\.md$"],
    ArtifactType.ROADMAP: [r"ROADMAP.*\.md$", r".*_ROADMAP.*\.md$"],
    ArtifactType.KEY_MATERIAL: [r".*\.pem$", r".*\.key$", r"keys/"],
    ArtifactType.CHECKPOINT: [r"checkpoint.*", r"\.ckpt$"],
    ArtifactType.CHAT_EXPORT: [r".*messages.*\.json$", r".*conversations.*\.json$"],
}

# Directories to skip (noise reduction)
SKIP_PATTERNS = [
    r"__pycache__",
    r"\.git/objects",
    r"node_modules",
    r"\.venv",
    r"venv",
    r"\.mypy_cache",
    r"\.pytest_cache",
    r"\.ruff_cache",
    r"target/debug",
    r"target/release",
    r"dist",
    r"build",
    r"\.eggs",
]


# =============================================================================
# HASH COMPUTATION
# =============================================================================


def compute_file_hash(path: Path, use_blake3: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    Compute cryptographic hash of file.
    
    Returns:
        Tuple of (blake3_hash, sha256_hash)
    """
    blake3_hash = None
    sha256_hash = None
    
    try:
        content = path.read_bytes()
        
        # BLAKE3 (preferred)
        if use_blake3 and BLAKE3_AVAILABLE:
            blake3_hash = blake3.blake3(content).hexdigest()
        
        # SHA-256 (fallback/additional)
        sha256_hash = hashlib.sha256(content).hexdigest()
        
    except Exception as e:
        logger.debug(f"Failed to hash {path}: {e}")
    
    return blake3_hash, sha256_hash


def compute_manifest_hash(artifacts: List[DiscoveredArtifact]) -> str:
    """
    Compute root hash of artifact manifest.
    
    Uses deterministic ordering and canonical JSON for reproducibility.
    """
    # Sort artifacts by path for determinism
    sorted_artifacts = sorted(artifacts, key=lambda a: a.path)
    
    # Build canonical representation
    canonical_entries = []
    for artifact in sorted_artifacts:
        entry = {
            "path": artifact.path,
            "size": artifact.size_bytes,
            "blake3": artifact.blake3_hash or artifact.sha256_hash,
        }
        canonical_entries.append(entry)
    
    # Serialize with sorted keys
    canonical_json = json.dumps(canonical_entries, sort_keys=True, separators=(",", ":"))
    
    # Compute root hash
    if BLAKE3_AVAILABLE:
        return blake3.blake3(canonical_json.encode()).hexdigest()
    else:
        return hashlib.sha256(canonical_json.encode()).hexdigest()


# =============================================================================
# DISCOVERY ENGINE
# =============================================================================


class GenesisNodeDiscovery:
    """
    Discovers and catalogs all BIZRA ecosystem artifacts on the Genesis Node.
    
    This is the "eyes" of the Genesis attestation—it sees the full ecosystem,
    not just one repository. Every artifact contributes to Proof of Impact.
    """
    
    def __init__(
        self,
        search_roots: Optional[List[Path]] = None,
        max_file_size_mb: float = 100.0,
        include_hashes: bool = True,
        skip_sensitive: bool = True,
    ):
        """
        Initialize discovery engine.
        
        Args:
            search_roots: Root directories to search (defaults to common locations)
            max_file_size_mb: Maximum file size to hash (larger files get size only)
            include_hashes: Whether to compute content hashes
            skip_sensitive: Whether to skip sensitive directories (keys, etc.)
        """
        self.search_roots = search_roots or self._default_search_roots()
        self.max_file_size = int(max_file_size_mb * 1024 * 1024)
        self.include_hashes = include_hashes
        self.skip_sensitive = skip_sensitive
        
        # Compiled patterns
        self._bizra_re = re.compile("|".join(BIZRA_PATTERNS), re.IGNORECASE)
        self._chat_re = re.compile("|".join(CHAT_DATA_PATTERNS), re.IGNORECASE)
        self._skip_re = re.compile("|".join(SKIP_PATTERNS))
        
        # Discovery state
        self._discovered: List[DiscoveredArtifact] = []
        self._repos: Set[str] = set()
        self._chat_message_count = 0
    
    def _default_search_roots(self) -> List[Path]:
        """Get default search roots based on OS."""
        roots = []
        
        if platform.system() == "Windows":
            # Windows: Look in common development locations
            roots.extend([
                Path("C:/bizra_scaffold"),
                Path("C:/bizra_scaffold.worktrees"),
                Path("C:/Users"),  # Will be filtered by BIZRA patterns
            ])
            # Add current user's home
            home = Path.home()
            roots.append(home / "Documents")
            roots.append(home / "Desktop")
            roots.append(home / "Downloads")
        else:
            # Unix: Common locations
            roots.extend([
                Path.home(),
                Path("/opt/bizra"),
                Path("/var/lib/bizra"),
            ])
        
        return [r for r in roots if r.exists()]
    
    def _should_skip(self, path: Path) -> bool:
        """Check if path should be skipped."""
        path_str = str(path)
        
        # Skip noise directories
        if self._skip_re.search(path_str):
            return True
        
        # Skip sensitive if configured
        if self.skip_sensitive:
            if "keys" in path_str.lower() or ".pem" in path_str or ".key" in path_str:
                # Still discover but don't hash
                pass
        
        return False
    
    def _is_bizra_related(self, path: Path) -> bool:
        """Check if path is BIZRA-related."""
        path_str = str(path).lower()
        return bool(self._bizra_re.search(path_str))
    
    def _classify_artifact(self, path: Path) -> Tuple[ArtifactType, ArtifactSignificance]:
        """Classify artifact type and significance."""
        path_str = str(path)
        name = path.name.lower()
        
        # Check specific patterns
        for artifact_type, patterns in ARTIFACT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, path_str, re.IGNORECASE):
                    # Determine significance
                    if artifact_type == ArtifactType.CONSTITUTION:
                        return artifact_type, ArtifactSignificance.GENESIS
                    elif artifact_type in (ArtifactType.ATTESTATION, ArtifactType.KEY_MATERIAL):
                        return artifact_type, ArtifactSignificance.CRITICAL
                    elif artifact_type in (ArtifactType.KNOWLEDGE_GRAPH, ArtifactType.EVIDENCE_PACK):
                        return artifact_type, ArtifactSignificance.HIGH
                    else:
                        return artifact_type, ArtifactSignificance.MEDIUM
        
        # Check for repository
        if (path / ".git").exists():
            return ArtifactType.REPOSITORY, ArtifactSignificance.HIGH
        
        # Check file extensions
        if path.suffix == ".py":
            return ArtifactType.PYTHON_MODULE, ArtifactSignificance.MEDIUM
        elif path.suffix == ".rs":
            return ArtifactType.RUST_CRATE, ArtifactSignificance.MEDIUM
        elif path.suffix in (".toml", ".yaml", ".yml", ".json"):
            return ArtifactType.CONFIG, ArtifactSignificance.MEDIUM
        elif path.suffix == ".md":
            if "spec" in name or "protocol" in name:
                return ArtifactType.SPECIFICATION, ArtifactSignificance.HIGH
            elif "roadmap" in name:
                return ArtifactType.ROADMAP, ArtifactSignificance.MEDIUM
            else:
                return ArtifactType.EVIDENCE_DOC, ArtifactSignificance.MEDIUM
        
        return ArtifactType.CONFIG, ArtifactSignificance.LOW
    
    def _extract_chat_metadata(self, path: Path) -> Dict[str, Any]:
        """Extract metadata from chat export files."""
        metadata = {"is_chat_data": True}
        
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(content)
            
            # Count messages
            if isinstance(data, list):
                metadata["message_count"] = len(data)
                self._chat_message_count += len(data)
            elif isinstance(data, dict):
                if "messages" in data:
                    metadata["message_count"] = len(data["messages"])
                    self._chat_message_count += len(data["messages"])
                if "conversations" in data:
                    metadata["conversation_count"] = len(data["conversations"])
            
            # Extract date range
            timestamps = []
            for item in (data if isinstance(data, list) else data.get("messages", [])):
                if isinstance(item, dict):
                    for key in ("create_time", "timestamp", "created_at", "date"):
                        if key in item:
                            ts = item[key]
                            if isinstance(ts, (int, float)):
                                timestamps.append(datetime.fromtimestamp(ts, tz=timezone.utc))
                            elif isinstance(ts, str):
                                try:
                                    timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                                except ValueError:
                                    pass
                            break
            
            if timestamps:
                metadata["earliest_message"] = min(timestamps).isoformat()
                metadata["latest_message"] = max(timestamps).isoformat()
        
        except Exception as e:
            logger.debug(f"Failed to extract chat metadata from {path}: {e}")
        
        return metadata
    
    def _discover_directory(self, root: Path, depth: int = 0, max_depth: int = 10) -> None:
        """Recursively discover artifacts in directory."""
        if depth > max_depth:
            return
        
        if not root.exists() or self._should_skip(root):
            return
        
        try:
            for entry in root.iterdir():
                try:
                    if entry.is_dir():
                        # Check if it's a git repo
                        if (entry / ".git").exists():
                            self._repos.add(str(entry))
                            self._discover_artifact(entry)
                        
                        # Recurse if BIZRA-related or at shallow depth
                        if self._is_bizra_related(entry) or depth < 3:
                            self._discover_directory(entry, depth + 1, max_depth)
                    
                    elif entry.is_file():
                        # Only process BIZRA-related files or known patterns
                        if self._is_bizra_related(entry) or self._matches_artifact_pattern(entry):
                            self._discover_artifact(entry)
                
                except PermissionError:
                    continue
                except Exception as e:
                    logger.debug(f"Error processing {entry}: {e}")
        
        except PermissionError:
            pass
    
    def _matches_artifact_pattern(self, path: Path) -> bool:
        """Check if file matches any artifact pattern."""
        path_str = str(path)
        for patterns in ARTIFACT_PATTERNS.values():
            for pattern in patterns:
                if re.search(pattern, path_str, re.IGNORECASE):
                    return True
        return False
    
    def _discover_artifact(self, path: Path) -> None:
        """Discover and catalog a single artifact."""
        try:
            stat = path.stat()
            
            artifact_type, significance = self._classify_artifact(path)
            
            # Get timestamps
            modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            created_at = None
            if hasattr(stat, "st_birthtime"):
                created_at = datetime.fromtimestamp(stat.st_birthtime, tz=timezone.utc)
            
            # Compute hashes for files under size limit
            blake3_hash = None
            sha256_hash = None
            if path.is_file() and stat.st_size <= self.max_file_size and self.include_hashes:
                # Skip key material
                if artifact_type != ArtifactType.KEY_MATERIAL:
                    blake3_hash, sha256_hash = compute_file_hash(path)
            
            # Extract metadata for chat data
            metadata = {}
            if artifact_type == ArtifactType.CHAT_EXPORT or self._chat_re.search(str(path)):
                metadata = self._extract_chat_metadata(path)
            
            # Find parent repo
            parent_repo = None
            for repo in self._repos:
                if str(path).startswith(repo):
                    parent_repo = repo
                    break
            
            artifact = DiscoveredArtifact(
                path=str(path),
                artifact_type=artifact_type,
                significance=significance,
                size_bytes=stat.st_size if path.is_file() else 0,
                modified_at=modified_at,
                created_at=created_at,
                blake3_hash=blake3_hash,
                sha256_hash=sha256_hash,
                parent_repo=parent_repo,
                metadata=metadata,
            )
            
            self._discovered.append(artifact)
            
        except Exception as e:
            logger.debug(f"Failed to discover {path}: {e}")
    
    def discover(self) -> GenesisNodeProfile:
        """
        Execute full ecosystem discovery.
        
        Returns:
            GenesisNodeProfile with all discovered artifacts
        """
        logger.info("Starting Genesis Node discovery...")
        
        # Clear state
        self._discovered = []
        self._repos = set()
        self._chat_message_count = 0
        
        # Discover from all roots
        for root in self.search_roots:
            logger.info(f"Scanning: {root}")
            self._discover_directory(root)
        
        # Build profile
        profile = GenesisNodeProfile(
            node_id=self._generate_node_id(),
            hostname=platform.node(),
            platform=f"{platform.system()} {platform.release()}",
            discovery_timestamp=datetime.now(timezone.utc),
            artifacts=self._discovered,
            total_artifacts=len(self._discovered),
            total_size_bytes=sum(a.size_bytes for a in self._discovered),
            total_repositories=len(self._repos),
            total_chat_messages=self._chat_message_count,
        )
        
        # Compute timeline bounds
        timestamps = [a.modified_at for a in self._discovered if a.modified_at]
        if timestamps:
            profile.earliest_artifact = min(timestamps)
            profile.latest_artifact = max(timestamps)
        
        # Compute ecosystem root hash
        profile.ecosystem_root_hash = compute_manifest_hash(self._discovered)
        
        # Estimate contribution metrics
        profile.contribution_hours_estimated = self._estimate_contribution_hours()
        profile.artifact_density = len(self._discovered) / max(1, len(self._repos))
        
        logger.info(
            f"Discovery complete: {profile.total_artifacts} artifacts, "
            f"{profile.total_repositories} repos, "
            f"{profile.total_chat_messages} chat messages"
        )
        
        return profile
    
    def _generate_node_id(self) -> str:
        """Generate deterministic node ID from machine characteristics."""
        components = [
            platform.node(),
            platform.machine(),
            platform.processor(),
        ]
        combined = ":".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _estimate_contribution_hours(self) -> float:
        """
        Estimate contribution hours from artifact timeline.
        
        Uses a heuristic based on file modifications and chat activity.
        """
        # Get all modification timestamps
        timestamps = sorted([a.modified_at for a in self._discovered if a.modified_at])
        
        if len(timestamps) < 2:
            return 0.0
        
        # Cluster modifications into sessions (30-minute gaps = new session)
        sessions = []
        session_start = timestamps[0]
        session_end = timestamps[0]
        
        for ts in timestamps[1:]:
            gap = (ts - session_end).total_seconds()
            if gap > 1800:  # 30 minutes
                # End current session
                duration = (session_end - session_start).total_seconds() / 3600
                sessions.append(max(0.25, duration))  # Minimum 15 minutes per session
                session_start = ts
            session_end = ts
        
        # Add final session
        duration = (session_end - session_start).total_seconds() / 3600
        sessions.append(max(0.25, duration))
        
        return sum(sessions)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main() -> int:
    """Main entry point for Genesis Node discovery."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Discover all BIZRA ecosystem artifacts on the Genesis Node"
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=None,
        help="Root directories to search (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/genesis/ECOSYSTEM_PROFILE.json"),
        help="Output file for ecosystem profile",
    )
    parser.add_argument(
        "--no-hashes",
        action="store_true",
        help="Skip content hash computation (faster but less complete)",
    )
    parser.add_argument(
        "--include-sensitive",
        action="store_true",
        help="Include sensitive directories (keys, etc.)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    # Run discovery
    discovery = GenesisNodeDiscovery(
        search_roots=args.roots,
        include_hashes=not args.no_hashes,
        skip_sensitive=not args.include_sensitive,
    )
    
    profile = discovery.discover()
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Ecosystem profile written to: {args.output}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("GENESIS NODE ECOSYSTEM DISCOVERY SUMMARY")
    print("=" * 70)
    print(f"Node ID:           {profile.node_id}")
    print(f"Hostname:          {profile.hostname}")
    print(f"Platform:          {profile.platform}")
    print(f"Discovery Time:    {profile.discovery_timestamp.isoformat()}")
    print("-" * 70)
    print(f"Total Artifacts:   {profile.total_artifacts}")
    print(f"Total Size:        {profile.total_size_bytes / 1024 / 1024:.2f} MB")
    print(f"Repositories:      {profile.total_repositories}")
    print(f"Chat Messages:     {profile.total_chat_messages}")
    print("-" * 70)
    print(f"Earliest Artifact: {profile.earliest_artifact}")
    print(f"Latest Artifact:   {profile.latest_artifact}")
    print(f"Contribution Est:  {profile.contribution_hours_estimated:.1f} hours")
    print("-" * 70)
    print(f"ECOSYSTEM ROOT HASH: {profile.ecosystem_root_hash}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
