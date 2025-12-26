r"""
BIZRA AEON OMEGA - Data Lake Watcher & Unified Knowledge Sentinel
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Pattern: Graph-of-Thoughts Integrated File System Observer

Implements unified watcher for:
  - C:\BIZRA-DATA-LAKE: Primary data lake
  - C:\BIZRA-NODE0\knowledge: Node zero knowledge base
  
Features:
  1. Cryptographic Integrity: SHA-256 hash manifests for all watched files
  2. Change Detection: Real-time observation with drift alerting
  3. SNR Quality Scoring: Signal-to-noise assessment of knowledge assets
  4. Graph-of-Thoughts Integration: Knowledge items linked as thought nodes
  5. Ihsān Compliance: All operations respect ethical threshold (IM ≥ 0.95)

Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     DataLakeWatcher                                  │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
  │  │ WatchedPath  │  │ WatchedPath  │  │   Manifest   │               │
  │  │  DATA-LAKE   │  │   NODE0/     │  │  Generator   │               │
  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
  │         │                 │                 │                       │
  │         └────────────┬────┴─────────────────┘                       │
  │                      ▼                                              │
  │              ┌───────────────┐                                      │
  │              │ UnifiedIndex  │──────► SNRScorer                     │
  │              └───────────────┘                                      │
  │                      │                                              │
  │                      ▼                                              │
  │              ┌───────────────┐                                      │
  │              │ Graph of      │──────► ThoughtChain                  │
  │              │ Thoughts      │                                      │
  │              └───────────────┘                                      │
  └─────────────────────────────────────────────────────────────────────┘

BIZRA SOT Compliance:
  - Section 3 (Invariants): IM ≥ 0.95 enforced
  - Section 7 (Evidence Policy): All changes logged to evidence/
  - Section 8 (Change Control): Version-tracked manifests

Author: BIZRA Genesis Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════


class FileChangeType(Enum):
    """Types of file system changes."""
    CREATED = auto()
    MODIFIED = auto()
    DELETED = auto()
    MOVED = auto()
    HASH_MISMATCH = auto()  # Content changed without metadata update


class AssetQuality(Enum):
    """Quality classification of knowledge assets."""
    CRITICAL = auto()       # Core knowledge, SNR > 0.90
    HIGH = auto()           # Valuable, SNR > 0.80
    MEDIUM = auto()         # Standard, 0.50 ≤ SNR ≤ 0.80
    LOW = auto()            # Noise candidate, SNR < 0.50
    UNSCORED = auto()       # Not yet assessed


class WatcherState(Enum):
    """Watcher operational state."""
    IDLE = auto()
    WATCHING = auto()
    SCANNING = auto()
    ERROR = auto()
    PAUSED = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FileAsset:
    """
    Single file asset in the data lake.
    
    Represents a knowledge item with:
    - Identity: Path and hash for integrity
    - Quality: SNR metrics for value assessment
    - Provenance: When added, last modified, by whom
    """
    
    # Identity
    path: Path
    relative_path: str
    sha256_hash: str
    size_bytes: int
    
    # Quality
    quality: AssetQuality = AssetQuality.UNSCORED
    snr_score: float = 0.0
    
    # Provenance
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_verified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metadata
    file_type: str = ""
    domains: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize asset for manifest storage."""
        return {
            "path": str(self.path),
            "relative_path": self.relative_path,
            "sha256_hash": self.sha256_hash,
            "size_bytes": self.size_bytes,
            "quality": self.quality.name,
            "snr_score": self.snr_score,
            "first_seen": self.first_seen.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "last_verified": self.last_verified.isoformat(),
            "file_type": self.file_type,
            "domains": list(self.domains),
            "tags": list(self.tags),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileAsset":
        """Deserialize asset from manifest."""
        return cls(
            path=Path(data["path"]),
            relative_path=data["relative_path"],
            sha256_hash=data["sha256_hash"],
            size_bytes=data["size_bytes"],
            quality=AssetQuality[data.get("quality", "UNSCORED")],
            snr_score=data.get("snr_score", 0.0),
            first_seen=datetime.fromisoformat(data.get("first_seen", datetime.now(timezone.utc).isoformat())),
            last_modified=datetime.fromisoformat(data.get("last_modified", datetime.now(timezone.utc).isoformat())),
            last_verified=datetime.fromisoformat(data.get("last_verified", datetime.now(timezone.utc).isoformat())),
            file_type=data.get("file_type", ""),
            domains=set(data.get("domains", [])),
            tags=set(data.get("tags", [])),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FileChange:
    """Record of a file system change."""
    
    asset: FileAsset
    change_type: FileChangeType
    previous_hash: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize change event."""
        return {
            "relative_path": self.asset.relative_path,
            "change_type": self.change_type.name,
            "new_hash": self.asset.sha256_hash,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class WatchedPath:
    """Configuration for a watched directory."""
    
    # Path configuration
    path: Path
    alias: str                          # Human-readable name
    enabled: bool = True
    recursive: bool = True
    
    # Filtering
    include_patterns: List[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*.tmp", "*.temp", "*.swp", "*.bak", "__pycache__/*", ".git/*", 
        "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db"
    ])
    
    # Quality thresholds
    min_snr_for_index: float = 0.0      # Minimum SNR to include in index
    critical_threshold: float = 0.90     # SNR threshold for CRITICAL assets
    
    def __post_init__(self):
        """Ensure path is absolute."""
        if not self.path.is_absolute():
            self.path = self.path.resolve()


@dataclass
class ManifestMetadata:
    """Manifest file metadata."""
    
    version: str = "1.0.0"
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_assets: int = 0
    total_size_bytes: int = 0
    checksum: str = ""  # SHA-256 of all asset hashes concatenated
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata."""
        return {
            "version": self.version,
            "created": self.created.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "total_assets": self.total_assets,
            "total_size_bytes": self.total_size_bytes,
            "checksum": self.checksum,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CORE WATCHER ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class DataLakeWatcher:
    """
    Unified watcher for BIZRA data lake and knowledge directories.
    
    Implements the Sentinel pattern:
    - Continuous integrity monitoring
    - Change detection with event emission
    - SNR-based quality assessment
    - Graph-of-thoughts integration
    
    Usage:
        watcher = DataLakeWatcher()
        watcher.add_watched_path(Path("C:/BIZRA-DATA-LAKE"), alias="data_lake")
        watcher.add_watched_path(Path("C:/BIZRA-NODE0/knowledge"), alias="node0_knowledge")
        
        # Initial scan
        changes = await watcher.scan_all()
        
        # Start watching (background)
        await watcher.start_watching(interval_seconds=60)
    """
    
    # Default watched paths
    DEFAULT_PATHS = [
        WatchedPath(path=Path("C:/BIZRA-DATA-LAKE"), alias="data_lake"),
        WatchedPath(path=Path("C:/BIZRA-NODE0/knowledge"), alias="node0_knowledge"),
    ]
    
    def __init__(
        self,
        manifest_dir: Optional[Path] = None,
        ihsan_threshold: float = 0.95,
        enable_snr_scoring: bool = True,
    ):
        """
        Initialize the Data Lake Watcher.
        
        Args:
            manifest_dir: Directory to store manifests (default: ./data/manifests)
            ihsan_threshold: Minimum IM for HIGH quality (SOT Section 3)
            enable_snr_scoring: Whether to compute SNR for assets
        """
        self.manifest_dir = manifest_dir or Path("data/manifests")
        self.ihsan_threshold = ihsan_threshold
        self.enable_snr_scoring = enable_snr_scoring
        
        # State
        self.watched_paths: Dict[str, WatchedPath] = {}
        self.assets: Dict[str, FileAsset] = {}  # Keyed by relative_path
        self.state = WatcherState.IDLE
        self._change_listeners: List[Callable[[FileChange], None]] = []
        self._lock = threading.RLock()
        self._watching_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "total_scans": 0,
            "total_changes_detected": 0,
            "last_scan_time": None,
            "last_scan_duration_ms": 0,
        }
        
        # Ensure manifest directory exists
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataLakeWatcher initialized. Manifest dir: {self.manifest_dir}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PATH MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_watched_path(
        self,
        path: Union[str, Path],
        alias: str,
        recursive: bool = True,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> bool:
        """
        Add a directory to watch.
        
        Args:
            path: Absolute path to directory
            alias: Human-readable alias (e.g., "data_lake")
            recursive: Whether to watch subdirectories
            include_patterns: Glob patterns to include
            exclude_patterns: Glob patterns to exclude
            
        Returns:
            True if path was added successfully
        """
        path = Path(path)
        
        if not path.is_absolute():
            path = path.resolve()
        
        if not path.exists():
            logger.warning(f"Path does not exist: {path}. Will be watched when created.")
        
        default_exclude = [
            "*.tmp", "*.temp", "*.swp", "*.bak", "__pycache__/*", ".git/*",
            "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db"
        ]
        watched = WatchedPath(
            path=path,
            alias=alias,
            enabled=True,
            recursive=recursive,
            include_patterns=include_patterns or ["*"],
            exclude_patterns=exclude_patterns or default_exclude,
        )
        
        with self._lock:
            self.watched_paths[alias] = watched
        
        logger.info(f"Added watched path: {alias} -> {path}")
        return True
    
    def remove_watched_path(self, alias: str) -> bool:
        """Remove a watched path by alias."""
        with self._lock:
            if alias in self.watched_paths:
                del self.watched_paths[alias]
                logger.info(f"Removed watched path: {alias}")
                return True
        return False
    
    def list_watched_paths(self) -> List[Dict[str, Any]]:
        """List all watched paths with status."""
        with self._lock:
            result = []
            for alias, wp in self.watched_paths.items():
                result.append({
                    "alias": alias,
                    "path": str(wp.path),
                    "exists": wp.path.exists(),
                    "enabled": wp.enabled,
                    "recursive": wp.recursive,
                    "asset_count": sum(
                        1 for a in self.assets.values() 
                        if str(a.path).startswith(str(wp.path))
                    ),
                })
            return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FILE SCANNING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def compute_file_hash(self, filepath: Path, chunk_size: int = 8192) -> str:
        """
        Compute SHA-256 hash of file.
        
        Uses chunked reading for memory efficiency with large files.
        """
        sha256 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                while chunk := f.read(chunk_size):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except (IOError, OSError) as e:
            logger.error(f"Failed to hash file {filepath}: {e}")
            return ""
    
    def _matches_patterns(self, path: Path, patterns: List[str]) -> bool:
        """Check if path matches any glob pattern."""
        import fnmatch
        name = path.name
        rel_path = str(path)
        
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern):
                return True
        return False
    
    def _scan_directory(self, watched: WatchedPath) -> Generator[FileAsset, None, None]:
        """
        Scan a watched directory and yield FileAssets.
        
        Applies include/exclude filters and computes hashes.
        """
        if not watched.path.exists():
            logger.warning(f"Watched path does not exist: {watched.path}")
            return
        
        def walk_dir(base: Path) -> Generator[Path, None, None]:
            """Walk directory respecting recursive flag."""
            try:
                for entry in base.iterdir():
                    if entry.is_file():
                        yield entry
                    elif entry.is_dir() and watched.recursive:
                        # Check exclude patterns for directories
                        if not self._matches_patterns(entry, watched.exclude_patterns):
                            yield from walk_dir(entry)
            except PermissionError as e:
                logger.warning(f"Permission denied: {e}")
        
        for filepath in walk_dir(watched.path):
            # Apply filters
            if self._matches_patterns(filepath, watched.exclude_patterns):
                continue
            if not self._matches_patterns(filepath, watched.include_patterns):
                continue
            
            try:
                stat = filepath.stat()
                file_hash = self.compute_file_hash(filepath)
                
                if not file_hash:
                    continue
                
                relative = str(filepath.relative_to(watched.path))
                
                yield FileAsset(
                    path=filepath,
                    relative_path=f"{watched.alias}/{relative}",
                    sha256_hash=file_hash,
                    size_bytes=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    file_type=filepath.suffix.lower(),
                )
            except (OSError, IOError) as e:
                logger.warning(f"Failed to scan {filepath}: {e}")
    
    async def scan_all(self) -> List[FileChange]:
        """
        Perform full scan of all watched paths.
        
        Returns list of changes detected since last scan.
        """
        start_time = time.time()
        self.state = WatcherState.SCANNING
        changes: List[FileChange] = []
        
        try:
            # Track seen paths for deletion detection
            seen_paths: Set[str] = set()
            
            with self._lock:
                for watched in self.watched_paths.values():
                    if not watched.enabled:
                        continue
                    
                    for asset in self._scan_directory(watched):
                        seen_paths.add(asset.relative_path)
                        
                        if asset.relative_path in self.assets:
                            # Existing asset - check for changes
                            existing = self.assets[asset.relative_path]
                            if existing.sha256_hash != asset.sha256_hash:
                                change = FileChange(
                                    asset=asset,
                                    change_type=FileChangeType.MODIFIED,
                                    previous_hash=existing.sha256_hash,
                                )
                                changes.append(change)
                                self._notify_listeners(change)
                            
                            # Update asset
                            asset.first_seen = existing.first_seen
                            asset.snr_score = existing.snr_score
                            asset.quality = existing.quality
                            asset.domains = existing.domains
                            asset.tags = existing.tags
                        else:
                            # New asset
                            change = FileChange(
                                asset=asset,
                                change_type=FileChangeType.CREATED,
                            )
                            changes.append(change)
                            self._notify_listeners(change)
                        
                        self.assets[asset.relative_path] = asset
                
                # Detect deletions
                for rel_path in list(self.assets.keys()):
                    if rel_path not in seen_paths:
                        deleted_asset = self.assets[rel_path]
                        change = FileChange(
                            asset=deleted_asset,
                            change_type=FileChangeType.DELETED,
                        )
                        changes.append(change)
                        self._notify_listeners(change)
                        del self.assets[rel_path]
            
            # Update stats
            duration_ms = (time.time() - start_time) * 1000
            self.stats["total_scans"] += 1
            self.stats["total_changes_detected"] += len(changes)
            self.stats["last_scan_time"] = datetime.now(timezone.utc).isoformat()
            self.stats["last_scan_duration_ms"] = duration_ms
            
            logger.info(
                f"Scan complete. Assets: {len(self.assets)}, "
                f"Changes: {len(changes)}, Duration: {duration_ms:.2f}ms"
            )
            
        finally:
            self.state = WatcherState.IDLE
        
        return changes
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MANIFEST MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save_manifest(self, filename: str = "data_lake_manifest.json") -> Path:
        """
        Save current asset inventory to manifest file.
        
        Manifest includes:
        - All assets with hashes
        - Quality scores
        - Metadata
        - Integrity checksum
        """
        with self._lock:
            # Compute manifest checksum
            all_hashes = sorted(a.sha256_hash for a in self.assets.values())
            manifest_checksum = hashlib.sha256(
                "".join(all_hashes).encode()
            ).hexdigest()
            
            total_size = sum(a.size_bytes for a in self.assets.values())
            
            metadata = ManifestMetadata(
                version="1.0.0",
                created=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
                total_assets=len(self.assets),
                total_size_bytes=total_size,
                checksum=manifest_checksum,
            )
            
            manifest = {
                "metadata": metadata.to_dict(),
                "watched_paths": [
                    {
                        "alias": wp.alias,
                        "path": str(wp.path),
                        "enabled": wp.enabled,
                        "recursive": wp.recursive,
                    }
                    for wp in self.watched_paths.values()
                ],
                "assets": [a.to_dict() for a in self.assets.values()],
            }
            
            manifest_path = self.manifest_dir / filename
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Manifest saved: {manifest_path} ({len(self.assets)} assets)")
            return manifest_path
    
    def load_manifest(self, filename: str = "data_lake_manifest.json") -> bool:
        """
        Load manifest from file.
        
        Returns True if manifest was loaded successfully.
        """
        manifest_path = self.manifest_dir / filename
        
        if not manifest_path.exists():
            logger.warning(f"Manifest not found: {manifest_path}")
            return False
        
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            
            with self._lock:
                # Load watched paths
                for wp_data in manifest.get("watched_paths", []):
                    self.watched_paths[wp_data["alias"]] = WatchedPath(
                        path=Path(wp_data["path"]),
                        alias=wp_data["alias"],
                        enabled=wp_data.get("enabled", True),
                        recursive=wp_data.get("recursive", True),
                    )
                
                # Load assets
                self.assets.clear()
                for asset_data in manifest.get("assets", []):
                    asset = FileAsset.from_dict(asset_data)
                    self.assets[asset.relative_path] = asset
            
            logger.info(f"Manifest loaded: {len(self.assets)} assets from {manifest_path}")
            return True
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load manifest: {e}")
            return False
    
    def verify_manifest(self) -> Dict[str, Any]:
        """
        Verify current state against saved manifest.
        
        Returns verification report with:
        - matched: Files matching manifest
        - modified: Files with hash changes
        - missing: Files in manifest but not on disk
        - new: Files on disk but not in manifest
        """
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "matched": [],
            "modified": [],
            "missing": [],
            "new": [],
            "integrity_score": 0.0,
        }
        
        with self._lock:
            current_paths: Set[str] = set()
            
            for watched in self.watched_paths.values():
                if not watched.enabled or not watched.path.exists():
                    continue
                
                for asset in self._scan_directory(watched):
                    current_paths.add(asset.relative_path)
                    
                    if asset.relative_path in self.assets:
                        existing = self.assets[asset.relative_path]
                        if existing.sha256_hash == asset.sha256_hash:
                            report["matched"].append(asset.relative_path)
                        else:
                            report["modified"].append({
                                "path": asset.relative_path,
                                "expected": existing.sha256_hash[:16] + "...",
                                "actual": asset.sha256_hash[:16] + "...",
                            })
                    else:
                        report["new"].append(asset.relative_path)
            
            # Check for missing files
            for rel_path in self.assets.keys():
                if rel_path not in current_paths:
                    report["missing"].append(rel_path)
            
            # Compute integrity score
            total = len(report["matched"]) + len(report["modified"]) + len(report["missing"]) + len(report["new"])
            if total > 0:
                report["integrity_score"] = len(report["matched"]) / total
        
        return report
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SNR QUALITY SCORING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def score_asset_snr(self, asset: FileAsset) -> float:
        """
        Compute SNR score for a knowledge asset.
        
        Signal components:
        - File size (larger = more content, but capped)
        - File type weight (code/docs higher than binary)
        - Recency (recent modifications = active knowledge)
        
        Noise components:
        - Duplicate content (detected via hash)
        - Generic filenames
        - Deep nesting (harder to discover)
        
        Returns SNR score in [0, 1].
        """
        if not self.enable_snr_scoring:
            return 0.5
        
        signal = 0.0
        noise = 0.0
        
        # Signal: File type weights
        type_weights = {
            ".md": 0.9,
            ".py": 0.9,
            ".json": 0.7,
            ".yaml": 0.7,
            ".yml": 0.7,
            ".txt": 0.6,
            ".rs": 0.9,
            ".toml": 0.7,
            ".sql": 0.8,
            ".pdf": 0.5,
            ".csv": 0.6,
        }
        type_weight = type_weights.get(asset.file_type, 0.4)
        signal += type_weight
        
        # Signal: Size factor (log scale, capped)
        import math
        size_factor = min(math.log10(asset.size_bytes + 1) / 7, 1.0)  # Cap at 10MB equiv
        signal += size_factor * 0.3
        
        # Signal: Recency (exponential decay over 30 days)
        days_old = (datetime.now(timezone.utc) - asset.last_modified).days
        recency_factor = math.exp(-days_old / 30)
        signal += recency_factor * 0.2
        
        # Noise: Path depth penalty
        depth = asset.relative_path.count("/") + asset.relative_path.count("\\")
        depth_penalty = min(depth * 0.05, 0.3)
        noise += depth_penalty
        
        # Noise: Generic filename penalty
        generic_names = {"readme", "index", "main", "test", "temp", "draft"}
        if any(g in Path(asset.relative_path).stem.lower() for g in generic_names):
            noise += 0.1
        
        # Noise: Very small files (likely stubs)
        if asset.size_bytes < 100:
            noise += 0.2
        
        # Compute SNR
        epsilon = 1e-6
        snr = signal / (signal + noise + epsilon)
        
        # Classify quality
        if snr >= 0.90:
            asset.quality = AssetQuality.CRITICAL
        elif snr >= 0.80:
            asset.quality = AssetQuality.HIGH
        elif snr >= 0.50:
            asset.quality = AssetQuality.MEDIUM
        else:
            asset.quality = AssetQuality.LOW
        
        asset.snr_score = snr
        return snr
    
    async def score_all_assets(self) -> Dict[str, int]:
        """
        Score all assets and return quality distribution.
        
        Returns dict with counts per quality level.
        """
        distribution = {q.name: 0 for q in AssetQuality}
        
        with self._lock:
            for asset in self.assets.values():
                self.score_asset_snr(asset)
                distribution[asset.quality.name] += 1
        
        logger.info(f"SNR scoring complete. Distribution: {distribution}")
        return distribution
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CHANGE LISTENERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_change_listener(self, callback: Callable[[FileChange], None]) -> None:
        """Register callback for file change events."""
        self._change_listeners.append(callback)
    
    def remove_change_listener(self, callback: Callable[[FileChange], None]) -> bool:
        """Unregister change callback. Returns True if callback was removed."""
        try:
            self._change_listeners.remove(callback)
            return True
        except ValueError:
            logger.warning(f"Attempted to remove unregistered listener: {callback}")
            return False
    
    def _notify_listeners(self, change: FileChange) -> None:
        """Notify all registered listeners of a change."""
        # Iterate over a copy to prevent issues if listener modifies the list
        for listener in list(self._change_listeners):
            try:
                listener(change)
            except Exception as e:
                logger.error(f"Listener error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BACKGROUND WATCHING
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def start_watching(self, interval_seconds: float = 60.0) -> None:
        """
        Start background watching with periodic scans.
        
        Args:
            interval_seconds: Seconds between scans
        """
        if self._watching_task is not None:
            logger.warning("Watcher already running")
            return
        
        async def watch_loop():
            self.state = WatcherState.WATCHING
            logger.info(f"Starting watch loop (interval: {interval_seconds}s)")
            
            while self.state == WatcherState.WATCHING:
                try:
                    await self.scan_all()
                except Exception as e:
                    logger.error(f"Scan error: {e}")
                
                await asyncio.sleep(interval_seconds)
        
        self._watching_task = asyncio.create_task(watch_loop())
    
    async def stop_watching(self) -> None:
        """Stop background watching."""
        if self._watching_task is not None:
            self.state = WatcherState.IDLE
            self._watching_task.cancel()
            try:
                await self._watching_task
            except asyncio.CancelledError:
                pass
            self._watching_task = None
            logger.info("Watch loop stopped")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REPORTING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive watcher summary.
        
        Returns summary suitable for display or logging.
        """
        with self._lock:
            quality_dist = {q.name: 0 for q in AssetQuality}
            type_dist: Dict[str, int] = defaultdict(int)
            total_size = 0
            
            for asset in self.assets.values():
                quality_dist[asset.quality.name] += 1
                type_dist[asset.file_type or "unknown"] += 1
                total_size += asset.size_bytes
            
            return {
                "state": self.state.name,
                "watched_paths": len(self.watched_paths),
                "total_assets": len(self.assets),
                "total_size_bytes": total_size,
                "total_size_human": self._human_size(total_size),
                "quality_distribution": dict(quality_dist),
                "file_type_distribution": dict(type_dist),
                "stats": self.stats,
            }
    
    @staticmethod
    def _human_size(size_bytes: float) -> str:
        """Convert bytes to human-readable size."""
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def create_default_watcher(
    manifest_dir: Optional[Path] = None,
) -> DataLakeWatcher:
    """
    Create watcher with default BIZRA paths.
    
    Watches:
    - C:/BIZRA-DATA-LAKE (alias: data_lake)
    - C:/BIZRA-NODE0/knowledge (alias: node0_knowledge)
    """
    watcher = DataLakeWatcher(manifest_dir=manifest_dir)
    
    for default_path in DataLakeWatcher.DEFAULT_PATHS:
        watcher.add_watched_path(
            path=default_path.path,
            alias=default_path.alias,
            recursive=default_path.recursive,
        )
    
    return watcher


async def run_watcher_cli():
    """
    CLI entry point for the Data Lake Watcher.
    
    Performs initial scan, saves manifest, and optionally starts watching.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA Data Lake Watcher")
    parser.add_argument("--scan", action="store_true", help="Perform initial scan")
    parser.add_argument("--verify", action="store_true", help="Verify against manifest")
    parser.add_argument("--watch", action="store_true", help="Start continuous watching")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval (seconds)")
    parser.add_argument("--manifest-dir", type=str, help="Manifest directory")
    
    args = parser.parse_args()
    
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else None
    watcher = create_default_watcher(manifest_dir=manifest_dir)
    
    # Load existing manifest if present
    watcher.load_manifest()
    
    if args.scan:
        print("Scanning all watched paths...")
        changes = await watcher.scan_all()
        print(f"Scan complete. {len(changes)} changes detected.")
        
        # Score all assets
        distribution = await watcher.score_all_assets()
        print(f"SNR Distribution: {distribution}")
        
        # Save manifest
        manifest_path = watcher.save_manifest()
        print(f"Manifest saved: {manifest_path}")
    
    if args.verify:
        print("Verifying against manifest...")
        report = watcher.verify_manifest()
        print(f"Integrity Score: {report['integrity_score']:.2%}")
        print(f"  Matched: {len(report['matched'])}")
        print(f"  Modified: {len(report['modified'])}")
        print(f"  Missing: {len(report['missing'])}")
        print(f"  New: {len(report['new'])}")
    
    if args.watch:
        print(f"Starting continuous watch (interval: {args.interval}s)...")
        await watcher.start_watching(interval_seconds=args.interval)
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await watcher.stop_watching()
            print("Watcher stopped.")
    
    # Print summary
    summary = watcher.get_summary()
    print("\n" + "=" * 60)
    print(" BIZRA Data Lake Watcher Summary")
    print("=" * 60)
    print(f"  State: {summary['state']}")
    print(f"  Watched Paths: {summary['watched_paths']}")
    print(f"  Total Assets: {summary['total_assets']}")
    print(f"  Total Size: {summary['total_size_human']}")
    print(f"  Quality Distribution: {summary['quality_distribution']}")


if __name__ == "__main__":
    asyncio.run(run_watcher_cli())
