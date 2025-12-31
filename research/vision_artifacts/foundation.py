"""
BIZRA LAYER 0: THE ATOMIC FOUNDATION
Implementation of the 'AtomicElite' pattern for absolute reliability.
Ensures Data Sovereignty through POSIX-atomic writes and linked evidence.
"""

import os
import time
import tempfile
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class Evidence:
    """Cryptographic proof of an atomic operation."""
    pre_hash: str
    timestamp_ns: int
    sequence: int
    operation_type: str = "ATOMIC_WRITE"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)

class BIZRAViolation(Exception):
    """Raised when a core BIZRA invariant is violated."""
    pass

class AtomicElite:
    """
    Non-negotiable BIZRA Layer 0 Implementation.
    Patterns extracted from the OMNI-SYNTHESIS v5.0.0.
    """
    
    _sequence_counter: int = 0

    @classmethod
    def _get_next_sequence(cls) -> int:
        cls._sequence_counter += 1
        return cls._sequence_counter

    @classmethod
    def atomic_write(cls, path: str, content: bytes, metadata: Optional[Dict[str, Any]] = None) -> Evidence:
        """
        Elite pattern: atomic + evidence + deterministic.
        Ensures that either the file is written completely or not at all.
        Returns: cryptographic proof of operation.
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Create temporary file with content
        # We use the same directory as the target to ensure same-filesystem atomic rename
        fd, tmp_name = tempfile.mkstemp(dir=target_path.parent, prefix=".tmp_bizra_")
        try:
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(content)
                tmp.flush()
                # Ensure physical write to disk
                os.fsync(tmp.fileno())
            
            # 2. Generate evidence (SHA-3-512 + timestamp + sequence)
            # Upgraded to SHA-3-512 based on the Post-Quantum Roadmap
            content_hash = hashlib.sha3_512(content).hexdigest()
            
            evidence = Evidence(
                pre_hash=content_hash,
                timestamp_ns=time.time_ns(),
                sequence=cls._get_next_sequence(),
                metadata=metadata or {}
            )
            
            # 3. Write evidence to separate atomic file (recursive call)
            # Note: For evidence files themselves, we skip nested evidence to prevent infinite recursion
            evidence_path = f"{path}.evidence"
            if not path.endswith(".evidence"):
                cls._write_evidence_metadata(evidence_path, evidence)
            
            # 4. Atomic commit with POSIX rename
            # This is the single point of truth in POSIX systems
            os.replace(tmp_name, path)
            
            # 5. Verify Invariant (The 'No Silent Failure' Rule)
            if not cls._verify_atomic_write(path, evidence):
                raise BIZRAViolation(f"Atomic write verification failed for {path}")
            
            return evidence

        except Exception as e:
            # Cleanup temp file on failure
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise BIZRAViolation(f"Atomic write failed: {str(e)}")

    @staticmethod
    def _write_evidence_metadata(path: str, evidence: Evidence):
        """Internal helper for writing evidence without recursion."""
        content = evidence.to_json().encode('utf-8')
        fd, tmp_name = tempfile.mkstemp(dir=Path(path).parent, prefix=".tmp_ev_")
        try:
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    @staticmethod
    def _verify_atomic_write(path: str, evidence: Evidence) -> bool:
        """BIZRA verification: what was written matches evidence."""
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            content = f.read()
        return hashlib.sha3_512(content).hexdigest() == evidence.pre_hash

    @classmethod
    def load_with_evidence(cls, path: str) -> tuple[bytes, Evidence]:
        """Load a file and verify its evidence chain integrity."""
        evidence_path = f"{path}.evidence"
        if not os.path.exists(path) or not os.path.exists(evidence_path):
            raise BIZRAViolation(f"Missing data or evidence for {path}")
            
        with open(evidence_path, 'r') as f:
            ev_data = json.load(f)
            evidence = Evidence(**ev_data)
            
        with open(path, 'rb') as f:
            content = f.read()
            
        if hashlib.sha3_512(content).hexdigest() != evidence.pre_hash:
            raise BIZRAViolation(f"Evidence mismatch for {path}: corruption detected.")
            
        return content, evidence
