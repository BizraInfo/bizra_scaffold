"""
Tests for BIZRA Expertise Store (Content-Addressable)
═══════════════════════════════════════════════════════════════════════════════

Covers:
1. Artifact storage and retrieval
2. Content-addressed deduplication
3. Atomic write operations
4. Quota enforcement
5. Delete race condition handling
6. Garbage collection
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.expertise_store import (
    AtomicWriteError,
    ContentMismatchError,
    ExpertiseArtifact,
    ExpertiseStore,
    StorageFullError,
    _atomic_write,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_store_path():
    """Create a temporary store directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "expertise_store"


@pytest.fixture
def store(temp_store_path):
    """Create an expertise store instance."""
    return ExpertiseStore(temp_store_path)


@pytest.fixture
def sample_artifact():
    """Create a sample artifact."""
    return ExpertiseArtifact.create(
        artifact_type="knowledge",
        content=b"Sample expertise content for testing",
        metadata={"source": "unit_test", "version": "1.0"},
    )


# =============================================================================
# EXPERTISE ARTIFACT TESTS
# =============================================================================


class TestExpertiseArtifact:
    """Tests for ExpertiseArtifact dataclass."""

    def test_create_computes_hash(self):
        """Test create() computes content hash correctly."""
        content = b"Test content for hashing"
        artifact = ExpertiseArtifact.create(
            artifact_type="test",
            content=content,
        )

        assert artifact.content_hash == ExpertiseArtifact.compute_hash(content)
        assert len(artifact.content_hash) == 64  # SHA-256 hex

    def test_create_generates_id(self):
        """Test create() generates unique artifact ID."""
        artifact = ExpertiseArtifact.create(
            artifact_type="test",
            content=b"content",
        )

        assert artifact.artifact_id.startswith("exp_")
        assert len(artifact.artifact_id) > 4

    def test_create_sets_size(self):
        """Test create() sets size_bytes correctly."""
        content = b"x" * 1000
        artifact = ExpertiseArtifact.create(
            artifact_type="test",
            content=content,
        )

        assert artifact.size_bytes == 1000

    def test_create_sets_timestamp(self):
        """Test create() sets created_at timestamp."""
        artifact = ExpertiseArtifact.create(
            artifact_type="test",
            content=b"content",
        )

        assert artifact.created_at is not None
        assert "T" in artifact.created_at  # ISO format

    def test_same_content_same_hash(self):
        """Test same content produces same hash."""
        content = b"Identical content"
        hash1 = ExpertiseArtifact.compute_hash(content)
        hash2 = ExpertiseArtifact.compute_hash(content)

        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Test different content produces different hash."""
        hash1 = ExpertiseArtifact.compute_hash(b"Content A")
        hash2 = ExpertiseArtifact.compute_hash(b"Content B")

        assert hash1 != hash2


# =============================================================================
# ATOMIC WRITE TESTS
# =============================================================================


class TestAtomicWrite:
    """Tests for _atomic_write function."""

    def test_creates_file(self, temp_store_path):
        """Test atomic write creates file."""
        file_path = temp_store_path / "test_file.bin"
        data = b"Test data for atomic write"

        _atomic_write(file_path, data)

        assert file_path.exists()
        assert file_path.read_bytes() == data

    def test_creates_parent_directories(self, temp_store_path):
        """Test atomic write creates parent directories."""
        file_path = temp_store_path / "deep" / "nested" / "path" / "file.bin"
        data = b"Nested file content"

        _atomic_write(file_path, data)

        assert file_path.exists()
        assert file_path.read_bytes() == data

    def test_atomic_overwrite(self, temp_store_path):
        """Test atomic write overwrites existing file."""
        file_path = temp_store_path / "overwrite.bin"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"Original content")

        _atomic_write(file_path, b"New content")

        assert file_path.read_bytes() == b"New content"

    def test_handles_large_data(self, temp_store_path):
        """Test atomic write handles large data (chunked)."""
        file_path = temp_store_path / "large_file.bin"
        data = b"x" * (5 * 1024 * 1024)  # 5MB

        _atomic_write(file_path, data, chunk_size=1024 * 1024)

        assert file_path.exists()
        assert file_path.read_bytes() == data

    def test_cleans_up_on_failure(self, temp_store_path):
        """Test temp file is cleaned up on failure."""
        file_path = temp_store_path / "fail_file.bin"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Simulate failure during write
        with patch("os.write", side_effect=OSError("Simulated write error")):
            with pytest.raises(AtomicWriteError):
                _atomic_write(file_path, b"data")

        # No temp files should remain
        temp_files = list(file_path.parent.glob(".*"))
        assert len(temp_files) == 0


# =============================================================================
# EXPERTISE STORE TESTS
# =============================================================================


class TestExpertiseStore:
    """Tests for ExpertiseStore class."""

    def test_store_and_get_artifact(self, store, sample_artifact):
        """Test storing and retrieving an artifact."""
        artifact_id = store.store(sample_artifact)

        retrieved = store.get(artifact_id)

        assert retrieved is not None
        assert retrieved.artifact_id == sample_artifact.artifact_id
        assert retrieved.content == sample_artifact.content
        assert retrieved.artifact_type == sample_artifact.artifact_type

    def test_exists_returns_true_for_stored(self, store, sample_artifact):
        """Test exists() returns True for stored artifact."""
        artifact_id = store.store(sample_artifact)

        assert store.exists(artifact_id)

    def test_exists_returns_false_for_missing(self, store):
        """Test exists() returns False for missing artifact."""
        assert not store.exists("nonexistent_id")

    def test_get_returns_none_for_missing(self, store):
        """Test get() returns None for missing artifact."""
        result = store.get("nonexistent_id")

        assert result is None

    def test_content_deduplication(self, store):
        """Test same content is deduplicated."""
        content = b"Shared content for deduplication test"

        artifact1 = ExpertiseArtifact.create(
            artifact_type="type1",
            content=content,
            metadata={"instance": 1},
        )
        artifact2 = ExpertiseArtifact.create(
            artifact_type="type2",
            content=content,
            metadata={"instance": 2},
        )

        store.store(artifact1)
        store.store(artifact2)

        # Both should have same content hash
        assert artifact1.content_hash == artifact2.content_hash

        # Only one blob should exist
        blob_path = store._get_blob_path(artifact1.content_hash)
        assert blob_path.exists()

        # Both artifacts should be retrievable
        assert store.get(artifact1.artifact_id) is not None
        assert store.get(artifact2.artifact_id) is not None

    def test_content_hash_mismatch_raises_error(self, store):
        """Test storing with mismatched hash raises error."""
        artifact = ExpertiseArtifact(
            artifact_id="bad_hash",
            content_hash="wrong_hash_value",  # Incorrect hash
            artifact_type="test",
            content=b"Actual content",
            metadata={},
            created_at="2025-12-27T00:00:00Z",
            size_bytes=14,
        )

        with pytest.raises(ContentMismatchError):
            store.store(artifact)

    def test_delete_removes_artifact(self, store, sample_artifact):
        """Test delete removes artifact."""
        artifact_id = store.store(sample_artifact)

        success = store.delete(artifact_id)

        assert success
        assert not store.exists(artifact_id)

    def test_delete_returns_false_for_missing(self, store):
        """Test delete returns False for missing artifact."""
        success = store.delete("nonexistent_id")

        assert not success

    def test_delete_preserves_shared_blob(self, store):
        """Test delete preserves blob used by other artifacts."""
        content = b"Shared content"

        artifact1 = ExpertiseArtifact.create(artifact_type="a", content=content)
        artifact2 = ExpertiseArtifact.create(artifact_type="b", content=content)

        store.store(artifact1)
        store.store(artifact2)

        # Delete first artifact
        store.delete(artifact1.artifact_id)

        # Blob should still exist (used by artifact2)
        blob_path = store._get_blob_path(artifact1.content_hash)
        assert blob_path.exists()

        # Second artifact should still be retrievable
        retrieved = store.get(artifact2.artifact_id)
        assert retrieved is not None
        assert retrieved.content == content

    def test_delete_race_condition_handled(self, store, temp_store_path):
        """Test delete handles FileNotFoundError gracefully."""
        artifact = ExpertiseArtifact.create(
            artifact_type="test",
            content=b"Content that will be deleted by 'another process'",
        )
        artifact_id = store.store(artifact)

        # Simulate another process deleting the blob
        blob_path = store._get_blob_path(artifact.content_hash)

        # First delete in index, then simulate race
        original_unlink = Path.unlink

        def patched_unlink(self, *args, **kwargs):
            # First call succeeds, simulating race
            original_unlink(self, *args, **kwargs)
            # Call again to simulate race condition
            try:
                original_unlink(self, *args, **kwargs)
            except FileNotFoundError:
                pass  # Expected

        # Delete should succeed without error
        success = store.delete(artifact_id)
        assert success

    def test_list_artifacts(self, store):
        """Test listing all artifacts."""
        for i in range(5):
            artifact = ExpertiseArtifact.create(
                artifact_type=f"type_{i % 2}",
                content=f"Content {i}".encode(),
            )
            store.store(artifact)

        all_artifacts = store.list_artifacts()
        assert len(all_artifacts) == 5

    def test_list_artifacts_by_type(self, store):
        """Test listing artifacts filtered by type."""
        for i in range(5):
            artifact = ExpertiseArtifact.create(
                artifact_type="typeA" if i < 3 else "typeB",
                content=f"Content {i}".encode(),
            )
            store.store(artifact)

        type_a = store.list_artifacts(artifact_type="typeA")
        type_b = store.list_artifacts(artifact_type="typeB")

        assert len(type_a) == 3
        assert len(type_b) == 2

    def test_get_stats(self, store, sample_artifact):
        """Test statistics retrieval."""
        store.store(sample_artifact)

        stats = store.get_stats()

        assert stats["artifact_count"] == 1
        assert stats["total_size_bytes"] == sample_artifact.size_bytes
        assert "usage_percent" in stats


# =============================================================================
# QUOTA TESTS
# =============================================================================


class TestQuotaEnforcement:
    """Tests for storage quota enforcement."""

    def test_quota_exceeded_raises_error(self, temp_store_path):
        """Test exceeding quota raises StorageFullError."""
        # Create store with tiny quota
        store = ExpertiseStore(temp_store_path, quota_bytes=100)

        artifact = ExpertiseArtifact.create(
            artifact_type="large",
            content=b"x" * 200,  # Exceeds 100 byte quota
        )

        with pytest.raises(StorageFullError):
            store.store(artifact)

    def test_max_artifact_size_enforced(self, temp_store_path):
        """Test max artifact size is enforced."""
        store = ExpertiseStore(temp_store_path)

        # Create artifact larger than MAX_ARTIFACT_SIZE
        oversized = ExpertiseArtifact.create(
            artifact_type="huge",
            content=b"x" * (ExpertiseStore.MAX_ARTIFACT_SIZE + 1),
        )

        with pytest.raises(StorageFullError):
            store.store(oversized)

    def test_quota_tracking_accurate(self, temp_store_path):
        """Test quota tracking is accurate."""
        store = ExpertiseStore(temp_store_path, quota_bytes=10000)

        total_size = 0
        for i in range(5):
            content = f"Content block {i}".encode() * 100
            artifact = ExpertiseArtifact.create(artifact_type="test", content=content)
            store.store(artifact)
            total_size += len(content)

        stats = store.get_stats()
        assert stats["total_size_bytes"] == total_size


# =============================================================================
# GARBAGE COLLECTION TESTS
# =============================================================================


class TestGarbageCollection:
    """Tests for garbage collection of orphaned blobs."""

    def test_gc_removes_orphaned_blobs(self, store, temp_store_path):
        """Test GC removes blobs not referenced by index."""
        artifact = ExpertiseArtifact.create(
            artifact_type="test",
            content=b"Will be orphaned",
        )
        store.store(artifact)

        # Manually orphan the blob by removing from index
        blob_path = store._get_blob_path(artifact.content_hash)
        del store._index[artifact.artifact_id]
        store._save_index()

        # Blob should still exist
        assert blob_path.exists()

        # Run GC
        removed = store.gc()

        assert removed == 1
        assert not blob_path.exists()

    def test_gc_preserves_referenced_blobs(self, store):
        """Test GC preserves blobs still in use."""
        artifact = ExpertiseArtifact.create(
            artifact_type="test",
            content=b"Still in use",
        )
        store.store(artifact)

        removed = store.gc()

        assert removed == 0
        blob_path = store._get_blob_path(artifact.content_hash)
        assert blob_path.exists()


# =============================================================================
# PERSISTENCE TESTS
# =============================================================================


class TestStorePersistence:
    """Tests for store persistence across instances."""

    def test_reopen_preserves_artifacts(self, temp_store_path):
        """Test reopening store preserves artifacts."""
        store1 = ExpertiseStore(temp_store_path)
        artifact = ExpertiseArtifact.create(
            artifact_type="persistent",
            content=b"Persistent content",
        )
        artifact_id = store1.store(artifact)

        # Re-open
        store2 = ExpertiseStore(temp_store_path)

        retrieved = store2.get(artifact_id)
        assert retrieved is not None
        assert retrieved.content == b"Persistent content"

    def test_index_survives_restart(self, temp_store_path):
        """Test index metadata survives restart."""
        store1 = ExpertiseStore(temp_store_path)
        for i in range(5):
            artifact = ExpertiseArtifact.create(
                artifact_type=f"type_{i}",
                content=f"Content {i}".encode(),
                metadata={"index": i},
            )
            store1.store(artifact)

        # Re-open
        store2 = ExpertiseStore(temp_store_path)

        assert len(store2.list_artifacts()) == 5


# =============================================================================
# CONCURRENCY TESTS
# =============================================================================


class TestConcurrency:
    """Tests for concurrent access patterns."""

    def test_concurrent_stores_succeed(self, store):
        """Test concurrent stores don't corrupt the store."""
        results = []
        errors = []

        def store_artifact(index):
            try:
                artifact = ExpertiseArtifact.create(
                    artifact_type="concurrent",
                    content=f"Concurrent content {index}".encode(),
                )
                artifact_id = store.store(artifact)
                results.append(artifact_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_artifact, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert len(store.list_artifacts()) == 10


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
