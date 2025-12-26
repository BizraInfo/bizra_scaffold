"""
BIZRA AEON OMEGA - Data Lake Watcher Tests
═══════════════════════════════════════════════════════════════════════════════
Unit tests for DataLakeWatcher with comprehensive coverage.
"""

import asyncio
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.data_lake_watcher import (
    AssetQuality,
    DataLakeWatcher,
    FileAsset,
    FileChange,
    FileChangeType,
    ManifestMetadata,
    WatchedPath,
    WatcherState,
    create_default_watcher,
)

# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def watcher(temp_dir):
    """Create watcher with temp directory."""
    manifest_dir = temp_dir / "manifests"
    manifest_dir.mkdir()
    return DataLakeWatcher(manifest_dir=manifest_dir)


@pytest.fixture
def sample_files(temp_dir):
    """Create sample files for testing."""
    # Create test directory structure
    data_dir = temp_dir / "data"
    data_dir.mkdir()

    # Create files with various types
    (data_dir / "readme.md").write_text("# Test Readme\n\nThis is a test file.")
    (data_dir / "config.yaml").write_text("key: value\nthreshold: 0.95")
    (data_dir / "script.py").write_text("def hello():\n    return 'world'")
    (data_dir / "data.json").write_text('{"items": [1, 2, 3]}')
    (data_dir / "notes.txt").write_text("Some notes here")

    # Create nested directory
    nested = data_dir / "nested"
    nested.mkdir()
    (nested / "deep_file.md").write_text("# Deep file")

    return data_dir


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS - FileAsset
# ═══════════════════════════════════════════════════════════════════════════════


class TestFileAsset:
    """Tests for FileAsset dataclass."""

    def test_create_asset(self):
        """Test basic asset creation."""
        asset = FileAsset(
            path=Path("/test/file.md"),
            relative_path="test/file.md",
            sha256_hash="abc123",
            size_bytes=100,
        )

        assert asset.path == Path("/test/file.md")
        assert asset.relative_path == "test/file.md"
        assert asset.sha256_hash == "abc123"
        assert asset.size_bytes == 100
        assert asset.quality == AssetQuality.UNSCORED
        assert asset.snr_score == 0.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        asset = FileAsset(
            path=Path("/test/file.md"),
            relative_path="test/file.md",
            sha256_hash="abc123",
            size_bytes=100,
            quality=AssetQuality.HIGH,
            snr_score=0.85,
        )

        data = asset.to_dict()

        # Path string format varies by OS (Windows uses backslashes)
        assert Path(data["path"]) == Path("/test/file.md")
        assert data["relative_path"] == "test/file.md"
        assert data["sha256_hash"] == "abc123"
        assert data["size_bytes"] == 100
        assert data["quality"] == "HIGH"
        assert data["snr_score"] == 0.85

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "path": "/test/file.md",
            "relative_path": "test/file.md",
            "sha256_hash": "abc123",
            "size_bytes": 100,
            "quality": "HIGH",
            "snr_score": 0.85,
        }

        asset = FileAsset.from_dict(data)

        assert asset.path == Path("/test/file.md")
        assert asset.quality == AssetQuality.HIGH
        assert asset.snr_score == 0.85


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS - DataLakeWatcher
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataLakeWatcher:
    """Tests for DataLakeWatcher class."""

    def test_init(self, watcher):
        """Test watcher initialization."""
        assert watcher.state == WatcherState.IDLE
        assert watcher.ihsan_threshold == 0.95
        assert watcher.enable_snr_scoring is True
        assert len(watcher.watched_paths) == 0
        assert len(watcher.assets) == 0

    def test_add_watched_path(self, watcher, temp_dir):
        """Test adding watched path."""
        result = watcher.add_watched_path(
            path=temp_dir / "data",
            alias="test_data",
            recursive=True,
        )

        assert result is True
        assert "test_data" in watcher.watched_paths
        assert watcher.watched_paths["test_data"].alias == "test_data"

    def test_remove_watched_path(self, watcher, temp_dir):
        """Test removing watched path."""
        watcher.add_watched_path(temp_dir, "test")

        result = watcher.remove_watched_path("test")

        assert result is True
        assert "test" not in watcher.watched_paths

    def test_remove_nonexistent_path(self, watcher):
        """Test removing path that doesn't exist."""
        result = watcher.remove_watched_path("nonexistent")
        assert result is False

    def test_list_watched_paths(self, watcher, temp_dir):
        """Test listing watched paths."""
        watcher.add_watched_path(temp_dir, "test1")
        watcher.add_watched_path(temp_dir, "test2")

        paths = watcher.list_watched_paths()

        assert len(paths) == 2
        aliases = [p["alias"] for p in paths]
        assert "test1" in aliases
        assert "test2" in aliases


class TestFileHashing:
    """Tests for file hashing functionality."""

    def test_compute_file_hash(self, watcher, temp_dir):
        """Test computing file hash."""
        test_file = temp_dir / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        expected_hash = hashlib.sha256(test_content).hexdigest()
        actual_hash = watcher.compute_file_hash(test_file)

        assert actual_hash == expected_hash

    def test_compute_hash_nonexistent_file(self, watcher, temp_dir):
        """Test hashing nonexistent file returns empty string."""
        result = watcher.compute_file_hash(temp_dir / "nonexistent.txt")
        assert result == ""


class TestScanning:
    """Tests for directory scanning."""

    @pytest.mark.asyncio
    async def test_scan_all(self, watcher, sample_files):
        """Test scanning all watched paths."""
        watcher.add_watched_path(sample_files, "test_data")

        changes = await watcher.scan_all()

        # All files should be CREATED on first scan
        assert len(changes) > 0
        assert all(c.change_type == FileChangeType.CREATED for c in changes)
        assert len(watcher.assets) == 6  # 5 files + 1 nested

    @pytest.mark.asyncio
    async def test_detect_modification(self, watcher, sample_files):
        """Test detecting file modification."""
        watcher.add_watched_path(sample_files, "test_data")

        # Initial scan
        await watcher.scan_all()

        # Modify a file
        readme = sample_files / "readme.md"
        readme.write_text("# Modified Content")

        # Second scan
        changes = await watcher.scan_all()

        modified = [c for c in changes if c.change_type == FileChangeType.MODIFIED]
        assert len(modified) == 1
        assert "readme.md" in modified[0].asset.relative_path

    @pytest.mark.asyncio
    async def test_detect_deletion(self, watcher, sample_files):
        """Test detecting file deletion."""
        watcher.add_watched_path(sample_files, "test_data")

        # Initial scan
        await watcher.scan_all()

        # Delete a file
        (sample_files / "notes.txt").unlink()

        # Second scan
        changes = await watcher.scan_all()

        deleted = [c for c in changes if c.change_type == FileChangeType.DELETED]
        assert len(deleted) == 1
        assert "notes.txt" in deleted[0].asset.relative_path


class TestManifest:
    """Tests for manifest management."""

    @pytest.mark.asyncio
    async def test_save_manifest(self, watcher, sample_files):
        """Test saving manifest."""
        watcher.add_watched_path(sample_files, "test_data")
        await watcher.scan_all()

        manifest_path = watcher.save_manifest()

        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "metadata" in manifest
        assert "assets" in manifest
        assert len(manifest["assets"]) == 6

    @pytest.mark.asyncio
    async def test_load_manifest(self, watcher, sample_files):
        """Test loading manifest."""
        watcher.add_watched_path(sample_files, "test_data")
        await watcher.scan_all()
        watcher.save_manifest()

        # Clear assets
        watcher.assets.clear()
        assert len(watcher.assets) == 0

        # Load manifest
        result = watcher.load_manifest()

        assert result is True
        assert len(watcher.assets) == 6

    @pytest.mark.asyncio
    async def test_verify_manifest(self, watcher, sample_files):
        """Test manifest verification."""
        watcher.add_watched_path(sample_files, "test_data")
        await watcher.scan_all()
        watcher.save_manifest()

        # Verify (no changes)
        report = watcher.verify_manifest()

        assert report["integrity_score"] == 1.0
        assert len(report["matched"]) == 6
        assert len(report["modified"]) == 0
        assert len(report["missing"]) == 0


class TestSNRScoring:
    """Tests for SNR quality scoring."""

    def test_score_markdown_file(self, watcher):
        """Test scoring markdown file."""
        asset = FileAsset(
            path=Path("/test/docs/readme.md"),
            relative_path="docs/readme.md",
            sha256_hash="abc123",
            size_bytes=5000,
            file_type=".md",
        )

        score = watcher.score_asset_snr(asset)

        assert score > 0.7  # Markdown files should score high
        assert asset.quality in (AssetQuality.HIGH, AssetQuality.CRITICAL)

    def test_score_tiny_file(self, watcher):
        """Test scoring very small file."""
        asset = FileAsset(
            path=Path("/test/stub.txt"),
            relative_path="stub.txt",
            sha256_hash="abc123",
            size_bytes=10,  # Very small
            file_type=".txt",
        )

        score = watcher.score_asset_snr(asset)

        # Small files still have type-based signal, but less than larger files
        assert score < 0.85  # Small files should score lower than optimal

    def test_score_deeply_nested_file(self, watcher):
        """Test scoring deeply nested file."""
        asset = FileAsset(
            path=Path("/test/a/b/c/d/e/file.md"),
            relative_path="a/b/c/d/e/file.md",
            sha256_hash="abc123",
            size_bytes=1000,
            file_type=".md",
        )

        score = watcher.score_asset_snr(asset)

        # Deep nesting adds noise penalty
        assert asset.snr_score < 0.9

    @pytest.mark.asyncio
    async def test_score_all_assets(self, watcher, sample_files):
        """Test scoring all assets."""
        watcher.add_watched_path(sample_files, "test_data")
        await watcher.scan_all()

        distribution = await watcher.score_all_assets()

        assert isinstance(distribution, dict)
        assert sum(distribution.values()) == 6  # All assets scored


class TestChangeListeners:
    """Tests for change listener functionality."""

    @pytest.mark.asyncio
    async def test_listener_receives_changes(self, watcher, sample_files):
        """Test that listeners receive change events."""
        watcher.add_watched_path(sample_files, "test_data")

        received_changes = []

        def listener(change: FileChange):
            received_changes.append(change)

        watcher.add_change_listener(listener)

        await watcher.scan_all()

        assert len(received_changes) == 6  # All files as CREATED

    def test_remove_listener(self, watcher):
        """Test removing a listener."""
        listener = MagicMock()
        watcher.add_change_listener(listener)

        watcher.remove_change_listener(listener)

        assert listener not in watcher._change_listeners


class TestSummary:
    """Tests for summary generation."""

    @pytest.mark.asyncio
    async def test_get_summary(self, watcher, sample_files):
        """Test getting watcher summary."""
        watcher.add_watched_path(sample_files, "test_data")
        await watcher.scan_all()
        await watcher.score_all_assets()

        summary = watcher.get_summary()

        assert summary["state"] == "IDLE"
        assert summary["watched_paths"] == 1
        assert summary["total_assets"] == 6
        assert summary["total_size_bytes"] > 0
        assert "quality_distribution" in summary
        assert "file_type_distribution" in summary


class TestFactory:
    """Tests for factory functions."""

    def test_create_default_watcher(self, temp_dir):
        """Test default watcher creation."""
        watcher = create_default_watcher(manifest_dir=temp_dir)

        assert len(watcher.watched_paths) == 2
        assert "data_lake" in watcher.watched_paths
        assert "node0_knowledge" in watcher.watched_paths


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for full workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, temp_dir):
        """Test complete scan -> score -> save -> load -> verify workflow."""
        # Setup
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        (data_dir / "file1.md").write_text("# File 1")
        (data_dir / "file2.py").write_text("def main(): pass")

        manifest_dir = temp_dir / "manifests"
        manifest_dir.mkdir()

        watcher = DataLakeWatcher(manifest_dir=manifest_dir)
        watcher.add_watched_path(data_dir, "test")

        # Scan
        changes = await watcher.scan_all()
        assert len(changes) == 2

        # Score
        distribution = await watcher.score_all_assets()
        assert sum(distribution.values()) == 2

        # Save
        manifest_path = watcher.save_manifest()
        assert manifest_path.exists()

        # Create new watcher and load
        watcher2 = DataLakeWatcher(manifest_dir=manifest_dir)
        watcher2.add_watched_path(data_dir, "test")
        assert watcher2.load_manifest()
        assert len(watcher2.assets) == 2

        # Verify
        report = watcher2.verify_manifest()
        assert report["integrity_score"] == 1.0
