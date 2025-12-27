"""
BIZRA AEON OMEGA - Append-Only Event Log (Receipt-First)
==========================================================
File-based append-only event log with hash chaining for integrity.

Implements SOT invariants:
- I2: Receipt-first mutation (all appends require hash chain validation)
- M2: Atomic + versioned (WAL appends are atomic)
- M3: Receipt-first (SAT must sign any durable memory event)

Security Considerations:
- Long JSONL entries: _get_last_hash reads backwards until newline found
- Concurrent writes: Fail-closed on hash mismatch (no silent corruption)
- Partial writes: Uses fsync and atomic append patterns

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Generator

# Cross-platform file locking
if sys.platform == "win32":
    import msvcrt

    @contextmanager
    def file_lock(f, exclusive: bool = True) -> Generator[None, None, None]:
        """Windows file locking using msvcrt."""
        try:
            # Lock the file (blocking)
            if exclusive:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            else:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass

else:
    import fcntl

    @contextmanager
    def file_lock(f, exclusive: bool = True) -> Generator[None, None, None]:
        """Unix file locking using fcntl."""
        try:
            if exclusive:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            yield
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


logger = logging.getLogger("bizra.memory.event_log")


# =============================================================================
# EXCEPTIONS
# =============================================================================


class EventLogError(Exception):
    """Base exception for event log operations."""

    pass


class ChainIntegrityError(EventLogError):
    """Hash chain integrity violation - fail-closed."""

    pass


class ConcurrentWriteError(EventLogError):
    """Concurrent write detected during append - fail-closed."""

    pass


class CorruptedLogError(EventLogError):
    """Log file is corrupted and cannot be read."""

    pass


# =============================================================================
# EVENT ENTRY
# =============================================================================


@dataclass
class EventEntry:
    """Immutable event log entry with hash chain linkage."""

    event_id: str
    timestamp: str
    event_type: str
    payload: Dict[str, Any]
    previous_hash: str  # Hash of previous entry (chain linkage)
    entry_hash: str  # Hash of this entry (computed from content)
    sequence: int

    def to_json(self) -> str:
        """Serialize to JSON (deterministic, sorted keys)."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, json_str: str) -> "EventEntry":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)

    @staticmethod
    def compute_hash(
        event_type: str,
        payload: Dict[str, Any],
        previous_hash: str,
        sequence: int,
        timestamp: str,
    ) -> str:
        """Compute SHA-256 hash of entry content."""
        content = json.dumps(
            {
                "event_type": event_type,
                "payload": payload,
                "previous_hash": previous_hash,
                "sequence": sequence,
                "timestamp": timestamp,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


# =============================================================================
# APPEND-ONLY EVENT LOG
# =============================================================================


class AppendOnlyEventLog:
    """
    File-based append-only event log with hash chain integrity.

    Security Properties:
    - Append-only: No modification or deletion of existing entries
    - Hash-chained: Each entry includes hash of previous entry
    - Fail-closed: Any integrity violation raises exception
    - Concurrent-safe: File locking prevents race conditions

    Performance Considerations:
    - Uses file-based locking (fcntl) for concurrency control
    - Reads last hash by scanning backwards from EOF
    - Handles arbitrarily long JSONL entries safely
    """

    # Genesis hash for first entry
    GENESIS_HASH = "0" * 64

    # Maximum chunk size for backward reading
    _READ_CHUNK_SIZE = 8192  # 8KB chunks for efficient backward reads

    def __init__(
        self,
        log_path: Path,
        on_append: Optional[Callable[[EventEntry], None]] = None,
    ):
        """
        Initialize event log.

        Args:
            log_path: Path to the JSONL log file
            on_append: Optional callback invoked after successful append
        """
        self._log_path = Path(log_path)
        self._on_append = on_append
        self._lock = threading.RLock()
        self._sequence = 0

        # Ensure parent directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize or validate existing log
        if self._log_path.exists():
            self._validate_and_load()
        else:
            # Create empty log file
            self._log_path.touch()
            logger.info(f"Created new event log: {self._log_path}")

    def _validate_and_load(self) -> None:
        """Validate existing log and load sequence counter."""
        last_entry = self._read_last_entry()
        if last_entry:
            self._sequence = last_entry.sequence
            logger.info(
                f"Loaded event log with {self._sequence} entries: {self._log_path}"
            )
        else:
            logger.info(f"Event log is empty: {self._log_path}")

    def _get_last_hash(self) -> str:
        """
        Get the hash of the last entry in the log.

        SECURITY FIX: Reads backwards from EOF until complete last line is found.
        This handles arbitrarily long JSONL entries without truncation.

        Returns:
            Hash of last entry, or GENESIS_HASH if log is empty
        """
        if not self._log_path.exists():
            return self.GENESIS_HASH

        file_size = self._log_path.stat().st_size
        if file_size == 0:
            return self.GENESIS_HASH

        with open(self._log_path, "rb") as f:
            return self._get_last_hash_from_handle(f, file_size)

    def _get_last_hash_from_handle(self, f, file_size: int) -> str:
        """
        Get the hash of the last entry from an open file handle.

        This allows reading while holding a lock on the file (Windows compatibility).

        Args:
            f: Open file handle in binary read mode
            file_size: Size of the file in bytes

        Returns:
            Hash of last entry, or GENESIS_HASH if log is empty
        """
        if file_size == 0:
            return self.GENESIS_HASH

        # Read backwards to find the last complete line
        # This handles entries of ANY length, not just 4096 bytes
        chunk_end = file_size
        buffer = b""

        while chunk_end > 0:
            # Calculate chunk start
            chunk_start = max(0, chunk_end - self._READ_CHUNK_SIZE)
            chunk_size = chunk_end - chunk_start

            # Read chunk
            f.seek(chunk_start)
            chunk = f.read(chunk_size)
            buffer = chunk + buffer
            chunk_end = chunk_start

            # Look for complete lines in buffer
            # Skip trailing newline if present
            search_buffer = buffer.rstrip(b"\n")
            if b"\n" in search_buffer:
                # Found a newline - extract last complete line
                last_newline = search_buffer.rfind(b"\n")
                last_line = search_buffer[last_newline + 1 :]
                break
            elif chunk_start == 0:
                # Reached start of file - entire file is one line
                last_line = search_buffer
                break
        else:
            # Buffer contains the entire file as one line
            last_line = buffer.rstrip(b"\n")

        if not last_line:
            return self.GENESIS_HASH

        try:
            entry = EventEntry.from_json(last_line.decode("utf-8"))
            return entry.entry_hash
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
            raise CorruptedLogError(
                f"Failed to parse last log entry: {e}. "
                f"Last line (first 100 chars): {last_line[:100]!r}"
            )

    def _read_last_entry(self) -> Optional[EventEntry]:
        """Read the last entry from the log."""
        if not self._log_path.exists():
            return None

        file_size = self._log_path.stat().st_size
        if file_size == 0:
            return None

        with open(self._log_path, "rb") as f:
            # Same backward reading logic as _get_last_hash
            chunk_end = file_size
            buffer = b""

            while chunk_end > 0:
                chunk_start = max(0, chunk_end - self._READ_CHUNK_SIZE)
                chunk_size = chunk_end - chunk_start

                f.seek(chunk_start)
                chunk = f.read(chunk_size)
                buffer = chunk + buffer
                chunk_end = chunk_start

                search_buffer = buffer.rstrip(b"\n")
                if b"\n" in search_buffer:
                    last_newline = search_buffer.rfind(b"\n")
                    last_line = search_buffer[last_newline + 1 :]
                    break
                elif chunk_start == 0:
                    last_line = search_buffer
                    break
            else:
                last_line = buffer.rstrip(b"\n")

        if not last_line:
            return None

        try:
            return EventEntry.from_json(last_line.decode("utf-8"))
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
            raise CorruptedLogError(f"Failed to parse last log entry: {e}")

    def append(
        self,
        event_type: str,
        payload: Dict[str, Any],
        event_id: Optional[str] = None,
    ) -> EventEntry:
        """
        Append a new event to the log.

        FAIL-CLOSED SEMANTICS:
        - If concurrent write is detected (hash mismatch), raises ConcurrentWriteError
        - If integrity check fails, raises ChainIntegrityError
        - No silent corruption - all failures are explicit

        Args:
            event_type: Type/category of the event
            payload: Event data (must be JSON-serializable)
            event_id: Optional event ID (generated if not provided)

        Returns:
            The appended EventEntry

        Raises:
            ConcurrentWriteError: Another process modified the log
            ChainIntegrityError: Hash chain validation failed
        """
        with self._lock:
            # Generate event ID if not provided
            if event_id is None:
                import secrets

                event_id = f"evt_{secrets.token_hex(8)}"

            timestamp = datetime.now(timezone.utc).isoformat()

            # Get expected previous hash BEFORE acquiring file lock
            expected_previous_hash = self._get_last_hash()

            # Acquire exclusive file lock for atomic append
            # Open in r+b mode to allow reading the current content before appending
            # Create the file if it doesn't exist
            if not self._log_path.exists():
                self._log_path.touch()

            with open(self._log_path, "r+b") as f:
                with file_lock(f, exclusive=True):
                    # SECURITY FIX: Re-read last hash AFTER acquiring lock
                    # to detect concurrent writes. Use file handle to avoid
                    # Windows permission issues with locked files.
                    f.seek(0, os.SEEK_END)
                    file_size = f.tell()
                    actual_previous_hash = self._get_last_hash_from_handle(f, file_size)

                    if actual_previous_hash != expected_previous_hash:
                        # FAIL-CLOSED: Concurrent write detected
                        # Do NOT silently proceed - this would corrupt the chain
                        raise ConcurrentWriteError(
                            f"Concurrent write detected. "
                            f"Expected previous_hash: {expected_previous_hash[:16]}..., "
                            f"Actual: {actual_previous_hash[:16]}... "
                            f"Retry the operation."
                        )

                    # Compute entry hash
                    self._sequence += 1
                    entry_hash = EventEntry.compute_hash(
                        event_type=event_type,
                        payload=payload,
                        previous_hash=actual_previous_hash,
                        sequence=self._sequence,
                        timestamp=timestamp,
                    )

                    # Create entry
                    entry = EventEntry(
                        event_id=event_id,
                        timestamp=timestamp,
                        event_type=event_type,
                        payload=payload,
                        previous_hash=actual_previous_hash,
                        entry_hash=entry_hash,
                        sequence=self._sequence,
                    )

                    # Seek to end and write entry atomically
                    f.seek(0, os.SEEK_END)
                    line = entry.to_json() + "\n"
                    f.write(line.encode("utf-8"))
                    f.flush()
                    os.fsync(f.fileno())

            # Invoke callback outside lock
            if self._on_append:
                try:
                    self._on_append(entry)
                except Exception as e:
                    logger.warning(f"on_append callback failed: {e}")

            return entry

    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Verify the integrity of the entire hash chain.

        Returns:
            (is_valid, error_message) tuple
        """
        if not self._log_path.exists():
            return True, None

        previous_hash = self.GENESIS_HASH
        line_number = 0

        with open(self._log_path, "r") as f:
            for line in f:
                line_number += 1
                if not line.strip():
                    continue

                try:
                    entry = EventEntry.from_json(line)
                except (json.JSONDecodeError, KeyError) as e:
                    return False, f"Line {line_number}: Parse error: {e}"

                # Verify previous hash linkage
                if entry.previous_hash != previous_hash:
                    return False, (
                        f"Line {line_number}: Chain broken. "
                        f"Expected previous_hash: {previous_hash[:16]}..., "
                        f"Got: {entry.previous_hash[:16]}..."
                    )

                # Verify entry hash
                expected_hash = EventEntry.compute_hash(
                    event_type=entry.event_type,
                    payload=entry.payload,
                    previous_hash=entry.previous_hash,
                    sequence=entry.sequence,
                    timestamp=entry.timestamp,
                )

                if entry.entry_hash != expected_hash:
                    return False, (
                        f"Line {line_number}: Hash mismatch. "
                        f"Stored: {entry.entry_hash[:16]}..., "
                        f"Computed: {expected_hash[:16]}..."
                    )

                previous_hash = entry.entry_hash

        return True, None

    def iter_entries(self) -> Iterator[EventEntry]:
        """Iterate over all entries in the log."""
        if not self._log_path.exists():
            return

        with open(self._log_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                yield EventEntry.from_json(line)

    def count(self) -> int:
        """Get the number of entries in the log."""
        return self._sequence

    def get_stats(self) -> Dict[str, Any]:
        """Get log statistics."""
        file_size = self._log_path.stat().st_size if self._log_path.exists() else 0
        return {
            "path": str(self._log_path),
            "entries": self._sequence,
            "file_size_bytes": file_size,
            "last_hash": self._get_last_hash()[:16] + "...",
        }
