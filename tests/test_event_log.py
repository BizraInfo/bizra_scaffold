"""
Tests for BIZRA Append-Only Event Log
═══════════════════════════════════════════════════════════════════════════════

Covers:
1. Basic append and read operations
2. Hash chain integrity
3. Concurrent write detection (fail-closed)
4. Long JSONL entry handling
5. Corrupted log detection
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory.event_log import (
    AppendOnlyEventLog,
    ChainIntegrityError,
    ConcurrentWriteError,
    CorruptedLogError,
    EventEntry,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_log_path():
    """Create a temporary log file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_events.jsonl"


@pytest.fixture
def event_log(temp_log_path):
    """Create an event log instance."""
    return AppendOnlyEventLog(temp_log_path)


# =============================================================================
# EVENT ENTRY TESTS
# =============================================================================


class TestEventEntry:
    """Tests for EventEntry dataclass."""

    def test_to_json_deterministic(self):
        """Test JSON serialization is deterministic (sorted keys)."""
        entry = EventEntry(
            event_id="evt_123",
            timestamp="2025-12-27T00:00:00Z",
            event_type="test",
            payload={"z": 1, "a": 2, "m": 3},
            previous_hash="0" * 64,
            entry_hash="abc123",
            sequence=1,
        )

        json1 = entry.to_json()
        json2 = entry.to_json()

        assert json1 == json2
        # Keys should be sorted
        assert json1.index('"a"') < json1.index('"m"') < json1.index('"z"')

    def test_from_json_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        original = EventEntry(
            event_id="evt_456",
            timestamp="2025-12-27T12:00:00Z",
            event_type="user_action",
            payload={"action": "click", "target": "button"},
            previous_hash="a" * 64,
            entry_hash="b" * 64,
            sequence=42,
        )

        json_str = original.to_json()
        restored = EventEntry.from_json(json_str)

        assert restored.event_id == original.event_id
        assert restored.event_type == original.event_type
        assert restored.payload == original.payload
        assert restored.sequence == original.sequence

    def test_compute_hash_deterministic(self):
        """Test hash computation is deterministic."""
        hash1 = EventEntry.compute_hash(
            event_type="test",
            payload={"key": "value"},
            previous_hash="0" * 64,
            sequence=1,
            timestamp="2025-12-27T00:00:00Z",
        )

        hash2 = EventEntry.compute_hash(
            event_type="test",
            payload={"key": "value"},
            previous_hash="0" * 64,
            sequence=1,
            timestamp="2025-12-27T00:00:00Z",
        )

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_compute_hash_changes_with_input(self):
        """Test hash changes with different inputs."""
        base_args = {
            "event_type": "test",
            "payload": {"key": "value"},
            "previous_hash": "0" * 64,
            "sequence": 1,
            "timestamp": "2025-12-27T00:00:00Z",
        }

        hash1 = EventEntry.compute_hash(**base_args)

        # Change payload
        modified_args = {**base_args, "payload": {"key": "different"}}
        hash2 = EventEntry.compute_hash(**modified_args)

        assert hash1 != hash2


# =============================================================================
# APPEND-ONLY EVENT LOG TESTS
# =============================================================================


class TestAppendOnlyEventLog:
    """Tests for AppendOnlyEventLog class."""

    def test_creates_empty_log(self, temp_log_path):
        """Test log file is created on initialization."""
        log = AppendOnlyEventLog(temp_log_path)

        assert temp_log_path.exists()
        assert log.count() == 0

    def test_append_single_event(self, event_log):
        """Test appending a single event."""
        entry = event_log.append(
            event_type="test_event",
            payload={"message": "Hello, World!"},
        )

        assert entry.event_id.startswith("evt_")
        assert entry.event_type == "test_event"
        assert entry.payload["message"] == "Hello, World!"
        assert entry.sequence == 1
        assert entry.previous_hash == "0" * 64  # Genesis

    def test_append_multiple_events_chains_hashes(self, event_log):
        """Test multiple appends form a hash chain."""
        entry1 = event_log.append(event_type="first", payload={})
        entry2 = event_log.append(event_type="second", payload={})
        entry3 = event_log.append(event_type="third", payload={})

        assert entry2.previous_hash == entry1.entry_hash
        assert entry3.previous_hash == entry2.entry_hash
        assert event_log.count() == 3

    def test_custom_event_id(self, event_log):
        """Test custom event ID is used."""
        entry = event_log.append(
            event_type="custom",
            payload={},
            event_id="my_custom_id_123",
        )

        assert entry.event_id == "my_custom_id_123"

    def test_verify_chain_valid(self, event_log):
        """Test chain verification passes for valid log."""
        for i in range(5):
            event_log.append(event_type=f"event_{i}", payload={"index": i})

        is_valid, error = event_log.verify_chain()

        assert is_valid
        assert error is None

    def test_verify_chain_detects_tampering(self, event_log, temp_log_path):
        """Test chain verification detects tampering."""
        for i in range(3):
            event_log.append(event_type=f"event_{i}", payload={"index": i})

        # Tamper with the log file
        lines = temp_log_path.read_text().splitlines()
        entry = json.loads(lines[1])
        entry["payload"]["index"] = 999  # Modify payload
        lines[1] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        temp_log_path.write_text("\n".join(lines) + "\n")

        # Re-load and verify
        log2 = AppendOnlyEventLog(temp_log_path)
        is_valid, error = log2.verify_chain()

        assert not is_valid
        assert "Hash mismatch" in error or "Chain broken" in error

    def test_iter_entries(self, event_log):
        """Test iterating over entries."""
        for i in range(5):
            event_log.append(event_type=f"event_{i}", payload={"i": i})

        entries = list(event_log.iter_entries())

        assert len(entries) == 5
        assert entries[0].event_type == "event_0"
        assert entries[4].event_type == "event_4"

    def test_get_stats(self, event_log):
        """Test statistics retrieval."""
        event_log.append(event_type="test", payload={"data": "x" * 100})

        stats = event_log.get_stats()

        assert stats["entries"] == 1
        assert stats["file_size_bytes"] > 0
        assert "last_hash" in stats

    def test_callback_invoked_on_append(self, temp_log_path):
        """Test on_append callback is invoked."""
        appended_events = []

        def on_append(entry):
            appended_events.append(entry)

        log = AppendOnlyEventLog(temp_log_path, on_append=on_append)
        log.append(event_type="callback_test", payload={})

        assert len(appended_events) == 1
        assert appended_events[0].event_type == "callback_test"


# =============================================================================
# LONG JSONL ENTRY TESTS
# =============================================================================


class TestLongJSONLEntries:
    """Tests for handling long JSONL entries (> 4KB)."""

    def test_handles_entry_larger_than_chunk_size(self, event_log, temp_log_path):
        """Test _get_last_hash handles entries larger than read chunk."""
        # Create a payload larger than 8KB (default chunk size)
        large_payload = {"data": "x" * 20000}  # ~20KB

        entry = event_log.append(event_type="large_event", payload=large_payload)

        # Verify we can read the last hash correctly
        last_hash = event_log._get_last_hash()
        assert last_hash == entry.entry_hash

    def test_multiple_large_entries(self, event_log):
        """Test multiple large entries are chained correctly."""
        entries = []
        for i in range(3):
            payload = {"data": f"{'x' * 15000}_{i}"}  # ~15KB each
            entry = event_log.append(event_type=f"large_{i}", payload=payload)
            entries.append(entry)

        # Verify chain
        assert entries[1].previous_hash == entries[0].entry_hash
        assert entries[2].previous_hash == entries[1].entry_hash

        # Verify chain integrity
        is_valid, error = event_log.verify_chain()
        assert is_valid, f"Chain validation failed: {error}"

    def test_single_line_file(self, event_log):
        """Test file with single entry (no newline to find)."""
        entry = event_log.append(event_type="only_entry", payload={"solo": True})

        last_hash = event_log._get_last_hash()
        assert last_hash == entry.entry_hash


# =============================================================================
# CONCURRENT WRITE TESTS
# =============================================================================


class TestConcurrentWriteDetection:
    """Tests for concurrent write detection (fail-closed)."""

    def test_concurrent_write_raises_error(self, temp_log_path):
        """Test concurrent write detection raises ConcurrentWriteError."""
        log1 = AppendOnlyEventLog(temp_log_path)
        log1.append(event_type="initial", payload={})

        # Simulate another process writing between hash read and lock acquire
        original_get_last_hash = log1._get_last_hash

        call_count = [0]

        def patched_get_last_hash():
            result = original_get_last_hash()
            call_count[0] += 1
            # After first call (before lock), inject a write
            if call_count[0] == 1:
                # Directly write to simulate concurrent modification
                with open(temp_log_path, "a") as f:
                    fake_entry = EventEntry(
                        event_id="concurrent_evt",
                        timestamp="2025-12-27T00:00:00Z",
                        event_type="concurrent",
                        payload={},
                        previous_hash=result,
                        entry_hash="c" * 64,
                        sequence=2,
                    )
                    f.write(fake_entry.to_json() + "\n")
            return result

        log1._get_last_hash = patched_get_last_hash

        with pytest.raises(ConcurrentWriteError) as exc_info:
            log1.append(event_type="should_fail", payload={})

        assert "Concurrent write detected" in str(exc_info.value)

    def test_sequential_writes_succeed(self, event_log):
        """Test sequential writes from same instance succeed."""
        for i in range(10):
            entry = event_log.append(event_type=f"seq_{i}", payload={"i": i})
            assert entry.sequence == i + 1


# =============================================================================
# CORRUPTED LOG TESTS
# =============================================================================


class TestCorruptedLogHandling:
    """Tests for corrupted log detection."""

    def test_corrupted_last_line_raises_error(self, temp_log_path):
        """Test corrupted last line raises CorruptedLogError."""
        log = AppendOnlyEventLog(temp_log_path)
        log.append(event_type="valid", payload={})

        # Corrupt the file
        with open(temp_log_path, "a") as f:
            f.write("not valid json\n")

        # Re-initialize should fail
        with pytest.raises(CorruptedLogError):
            AppendOnlyEventLog(temp_log_path)

    def test_empty_log_returns_genesis_hash(self, temp_log_path):
        """Test empty log returns genesis hash."""
        log = AppendOnlyEventLog(temp_log_path)

        assert log._get_last_hash() == "0" * 64


# =============================================================================
# PERSISTENCE TESTS
# =============================================================================


class TestPersistence:
    """Tests for log persistence across instances."""

    def test_reopen_preserves_entries(self, temp_log_path):
        """Test reopening log preserves entries."""
        log1 = AppendOnlyEventLog(temp_log_path)
        for i in range(5):
            log1.append(event_type=f"event_{i}", payload={})

        # Re-open
        log2 = AppendOnlyEventLog(temp_log_path)

        assert log2.count() == 5
        entries = list(log2.iter_entries())
        assert len(entries) == 5

    def test_append_after_reopen_chains_correctly(self, temp_log_path):
        """Test appending after reopen maintains chain."""
        log1 = AppendOnlyEventLog(temp_log_path)
        entry1 = log1.append(event_type="before_close", payload={})

        # Re-open and append
        log2 = AppendOnlyEventLog(temp_log_path)
        entry2 = log2.append(event_type="after_reopen", payload={})

        assert entry2.previous_hash == entry1.entry_hash
        assert entry2.sequence == 2


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
