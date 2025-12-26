r"""
BIZRA AEON OMEGA - Data Lake Configuration
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Pattern: Declarative Configuration for Knowledge Management

Defines watched paths, quality thresholds, and SNR parameters for:
  - C:\BIZRA-DATA-LAKE: Primary data lake
  - C:\BIZRA-NODE0\knowledge: Node zero knowledge base

BIZRA SOT Compliance:
  - Section 3 (Invariants): IM ≥ 0.95 enforced
  - Section 4 (PoI Parameters): Aligned thresholds

Author: BIZRA Genesis Team
Version: 1.0.0
"""

import os
from pathlib import Path
from typing import Any, Dict, List

# ═══════════════════════════════════════════════════════════════════════════════
# WATCHED PATHS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

WATCHED_PATHS = [
    {
        "alias": "data_lake",
        "path": os.environ.get("BIZRA_DATA_LAKE_PATH", "C:/BIZRA-DATA-LAKE"),
        "enabled": True,
        "recursive": True,
        "description": "Primary BIZRA data lake for persistent knowledge storage",
        "include_patterns": ["*"],
        "exclude_patterns": [
            "*.tmp",
            "*.temp",
            "*.swp",
            "*.bak",
            "__pycache__/*",
            ".git/*",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "Thumbs.db",
            "*.log",
            ".venv/*",
        ],
    },
    {
        "alias": "node0_knowledge",
        "path": os.environ.get(
            "BIZRA_NODE0_KNOWLEDGE_PATH", "C:/BIZRA-NODE0/knowledge"
        ),
        "enabled": True,
        "recursive": True,
        "description": "Node Zero knowledge base for genesis attestation",
        "include_patterns": ["*"],
        "exclude_patterns": [
            "*.tmp",
            "*.temp",
            "*.swp",
            "*.bak",
            "__pycache__/*",
            ".git/*",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "Thumbs.db",
        ],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# SNR QUALITY THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

SNR_THRESHOLDS = {
    # Quality classification thresholds
    "critical_threshold": 0.90,  # SNR > 0.90 = CRITICAL (top-tier knowledge)
    "high_threshold": 0.80,  # SNR > 0.80 = HIGH (valuable)
    "medium_threshold": 0.50,  # SNR ≥ 0.50 = MEDIUM (standard)
    # Below 0.50 = LOW (noise candidate)
    # Ihsān compliance (SOT Section 3)
    "ihsan_threshold": 0.95,  # IM ≥ 0.95 required for HIGH classification
    # Confidence requirements
    "min_confidence": 0.70,  # Minimum statistical confidence for classification
}


# ═══════════════════════════════════════════════════════════════════════════════
# FILE TYPE WEIGHTS (Signal Contribution)
# ═══════════════════════════════════════════════════════════════════════════════

FILE_TYPE_WEIGHTS = {
    # High-signal knowledge assets
    ".md": 0.95,  # Markdown documentation
    ".py": 0.90,  # Python code
    ".rs": 0.90,  # Rust code
    ".yaml": 0.85,  # Configuration/structured data
    ".yml": 0.85,
    ".json": 0.80,  # Structured data
    ".toml": 0.80,  # Configuration
    ".sql": 0.85,  # Database schemas/queries
    # Medium-signal assets
    ".txt": 0.65,  # Plain text
    ".csv": 0.70,  # Tabular data
    ".xml": 0.60,  # Structured markup
    ".html": 0.55,  # Web content
    # Lower-signal (often generated/binary)
    ".pdf": 0.50,  # Documents (can't parse content)
    ".png": 0.30,  # Images
    ".jpg": 0.30,
    ".jpeg": 0.30,
    ".gif": 0.25,
    ".mp4": 0.20,  # Video
    ".mp3": 0.20,  # Audio
    # Default for unknown types
    "default": 0.40,
}


# ═══════════════════════════════════════════════════════════════════════════════
# MANIFEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MANIFEST_CONFIG = {
    "manifest_dir": "data/manifests",
    "manifest_filename": "data_lake_manifest.json",
    "backup_count": 3,  # Keep N backup manifests
    "auto_backup": True,  # Create backup before overwriting
}


# ═══════════════════════════════════════════════════════════════════════════════
# WATCHER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

WATCHER_CONFIG = {
    "scan_interval_seconds": 60,  # Seconds between scans in watch mode
    "hash_chunk_size": 8192,  # Bytes per chunk for hashing
    "enable_snr_scoring": True,  # Compute SNR for all assets
    "auto_save_manifest": True,  # Save manifest after each scan
    "log_changes": True,  # Log all detected changes
}


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH OF THOUGHTS INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

GOT_INTEGRATION = {
    "enabled": True,
    "link_threshold": 0.70,  # Minimum SNR to create thought node
    "max_thought_depth": 5,  # Maximum chain depth for file relationships
    "domain_extraction": True,  # Extract domains from file paths
    "tag_extraction": True,  # Extract tags from file content (if parseable)
}


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLIANCE RULES
# ═══════════════════════════════════════════════════════════════════════════════

COMPLIANCE_RULES = {
    # Evidence Policy (SOT Section 7)
    "require_evidence_for_changes": True,
    "evidence_directory": "evidence/data_lake",
    # Change Control (SOT Section 8)
    "version_manifests": True,
    "require_change_notes": True,
    # Security
    "exclude_secrets_patterns": [
        "*.key",
        "*.pem",
        "*.p12",
        "*secret*",
        "*password*",
        "*.env",
        ".env.*",
        "credentials.*",
    ],
    "alert_on_secret_detection": True,
}


def get_config() -> Dict[str, Any]:
    """
    Get complete watcher configuration.

    Returns merged configuration suitable for DataLakeWatcher initialization.
    """
    return {
        "watched_paths": WATCHED_PATHS,
        "snr_thresholds": SNR_THRESHOLDS,
        "file_type_weights": FILE_TYPE_WEIGHTS,
        "manifest": MANIFEST_CONFIG,
        "watcher": WATCHER_CONFIG,
        "got_integration": GOT_INTEGRATION,
        "compliance": COMPLIANCE_RULES,
    }


def validate_paths() -> Dict[str, bool]:
    """
    Validate that configured paths exist.

    Returns dict mapping alias to existence status.
    """
    result = {}
    for wp in WATCHED_PATHS:
        path = Path(wp["path"])
        result[wp["alias"]] = path.exists()
    return result
