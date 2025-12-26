"""
BIZRA PCI Envelope
═══════════════════════════════════════════════════════════════════════════════
Proof-Carrying Inference envelope with canonical JSON and domain-separated digest.

PROTOCOL.md Section 2: Wire Format
- Canonical JSON (RFC 8785 JCS)
- Domain-separated digest: BLAKE3("bizra-pci-v1:" || canonical_bytes)
- Ed25519 signature with PQ migration path

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

# Use blake3 if available, fallback to hashlib
try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    import hashlib

    HAS_BLAKE3 = False


# Domain separation prefix per PROTOCOL.md Section 2.2
DOMAIN_PREFIX = b"bizra-pci-v1:"

# Protocol version
PROTOCOL_VERSION = "1.0.0"

# Signed fields per PROTOCOL.md Section 2.3
SIGNED_FIELDS = [
    "version",
    "envelope_id",
    "timestamp",
    "nonce",
    "sender",
    "payload",
    "metadata",
]


def canonical_json(data: Dict[str, Any]) -> bytes:
    """
    Produce canonical JSON matching RFC 8785 JCS.

    - Keys sorted lexicographically
    - No whitespace between tokens
    - UTF-8 encoded
    """
    return json.dumps(
        data,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def compute_digest(data: bytes, domain_separated: bool = True) -> str:
    """
    Compute BLAKE3 digest with optional domain separation.

    Args:
        data: Raw bytes to hash
        domain_separated: If True, prepend DOMAIN_PREFIX

    Returns:
        Hex-encoded 256-bit digest
    """
    if domain_separated:
        data = DOMAIN_PREFIX + data

    if HAS_BLAKE3:
        return blake3.blake3(data).hexdigest()
    else:
        # Fallback to SHA-256 if blake3 not available
        import hashlib

        return hashlib.sha256(data).hexdigest()


@dataclass
class Sender:
    """Agent sender information."""

    agent_type: str  # "PAT" or "SAT"
    agent_id: str
    public_key: str  # Hex-encoded 32-byte Ed25519 public key

    def to_dict(self) -> Dict[str, str]:
        return {
            "agent_type": self.agent_type,
            "agent_id": self.agent_id,
            "public_key": self.public_key,
        }

    @staticmethod
    def from_dict(data: Dict[str, str]) -> Sender:
        return Sender(
            agent_type=data["agent_type"],
            agent_id=data["agent_id"],
            public_key=data["public_key"],
        )


@dataclass
class Payload:
    """Envelope payload with action and policy binding."""

    action: str
    data: Dict[str, Any]
    policy_hash: str  # BLAKE3 of constitution
    state_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "action": self.action,
            "data": self.data,
            "policy_hash": self.policy_hash,
        }
        if self.state_hash:
            result["state_hash"] = self.state_hash
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Payload:
        return Payload(
            action=data["action"],
            data=data.get("data", {}),
            policy_hash=data["policy_hash"],
            state_hash=data.get("state_hash"),
        )


@dataclass
class Metadata:
    """Envelope metadata with Ihsān and SNR scores."""

    ihsan_score: float
    snr_score: Optional[float] = None
    urgency: str = "BATCH"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate ihsan_score range
        if not 0.0 <= self.ihsan_score <= 1.0:
            raise ValueError(
                f"ihsan_score must be in [0.0, 1.0], got {self.ihsan_score}"
            )
        if self.snr_score is not None and not 0.0 <= self.snr_score <= 1.0:
            raise ValueError(f"snr_score must be in [0.0, 1.0], got {self.snr_score}")

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "ihsan_score": self.ihsan_score,
            "urgency": self.urgency,
        }
        if self.snr_score is not None:
            result["snr_score"] = self.snr_score
        result.update(self.extra)
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Metadata:
        return Metadata(
            ihsan_score=data["ihsan_score"],
            snr_score=data.get("snr_score"),
            urgency=data.get("urgency", "BATCH"),
            extra={
                k: v
                for k, v in data.items()
                if k not in ("ihsan_score", "snr_score", "urgency")
            },
        )


@dataclass
class Signature:
    """Cryptographic signature with algorithm negotiation."""

    algorithm: str  # "ed25519" or "dilithium5"
    value: str  # Hex-encoded signature
    signed_fields: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "value": self.value,
            "signed_fields": self.signed_fields,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Signature:
        return Signature(
            algorithm=data["algorithm"],
            value=data["value"],
            signed_fields=data["signed_fields"],
        )


@dataclass
class PCIEnvelope:
    """
    Proof-Carrying Inference Envelope per PROTOCOL.md Section 2.

    Wire format for dual-agent protocol with:
    - Canonical JSON serialization
    - Domain-separated BLAKE3 digest
    - Ed25519 signature (PQ migration path supported)
    """

    version: str
    envelope_id: str
    timestamp: datetime
    nonce: str  # Hex-encoded 32-byte nonce
    sender: Sender
    payload: Payload
    metadata: Metadata
    signature: Optional[Signature] = None

    @staticmethod
    def create(
        sender: Sender,
        payload: Payload,
        metadata: Metadata,
    ) -> PCIEnvelope:
        """
        Factory method to create a new unsigned envelope.

        Generates unique envelope_id, timestamp, and nonce.
        """
        return PCIEnvelope(
            version=PROTOCOL_VERSION,
            envelope_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            nonce=secrets.token_hex(32),  # 256-bit nonce
            sender=sender,
            payload=payload,
            metadata=metadata,
            signature=None,
        )

    def signable_content(self) -> Dict[str, Any]:
        """Extract fields that are signed."""
        return {
            "version": self.version,
            "envelope_id": self.envelope_id,
            "timestamp": self.timestamp.isoformat(),
            "nonce": self.nonce,
            "sender": self.sender.to_dict(),
            "payload": self.payload.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        """Get canonical JSON bytes for signing/hashing."""
        return canonical_json(self.signable_content())

    def digest(self) -> str:
        """Compute domain-separated BLAKE3 digest."""
        return compute_digest(self.canonical_bytes(), domain_separated=True)

    def sign(self, private_key: ed25519.Ed25519PrivateKey) -> PCIEnvelope:
        """
        Sign the envelope with Ed25519 private key.

        Returns a new envelope with signature attached.
        """
        # Compute signature over canonical bytes
        canonical = self.canonical_bytes()
        sig_bytes = private_key.sign(canonical)

        # Extract public key for verification
        public_key = private_key.public_key()
        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        # Update sender public key if not set
        sender = Sender(
            agent_type=self.sender.agent_type,
            agent_id=self.sender.agent_id,
            public_key=public_key_bytes.hex(),
        )

        signature = Signature(
            algorithm="ed25519",
            value=sig_bytes.hex(),
            signed_fields=SIGNED_FIELDS,
        )

        return PCIEnvelope(
            version=self.version,
            envelope_id=self.envelope_id,
            timestamp=self.timestamp,
            nonce=self.nonce,
            sender=sender,
            payload=self.payload,
            metadata=self.metadata,
            signature=signature,
        )

    def verify_signature(self) -> Tuple[bool, Optional[str]]:
        """
        Verify the envelope signature.

        Returns:
            (True, None) if valid
            (False, error_message) if invalid
        """
        if self.signature is None:
            return (False, "Envelope is not signed")

        if self.signature.signed_fields != SIGNED_FIELDS:
            return (False, "Signature signed_fields mismatch")

        if self.signature.algorithm != "ed25519":
            return (False, f"Unsupported algorithm: {self.signature.algorithm}")

        try:
            # Reconstruct public key
            public_key_bytes = bytes.fromhex(self.sender.public_key)
            if len(public_key_bytes) != 32:
                return (False, "Invalid Ed25519 public key length")
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)

            # Reconstruct signature
            sig_bytes = bytes.fromhex(self.signature.value)
            if len(sig_bytes) != 64:
                return (False, "Invalid Ed25519 signature length")

            # Verify against canonical bytes
            canonical = self.canonical_bytes()
            public_key.verify(sig_bytes, canonical)

            return (True, None)
        except InvalidSignature:
            return (False, "Signature verification failed")
        except Exception as e:
            return (False, f"Signature verification error: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON encoding."""
        result = {
            "version": self.version,
            "envelope_id": self.envelope_id,
            "timestamp": self.timestamp.isoformat(),
            "nonce": self.nonce,
            "sender": self.sender.to_dict(),
            "payload": self.payload.to_dict(),
            "metadata": self.metadata.to_dict(),
        }
        if self.signature:
            result["signature"] = self.signature.to_dict()
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> PCIEnvelope:
        """Deserialize from dictionary."""
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            # Parse ISO8601 timestamp
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        signature = None
        if "signature" in data:
            signature = Signature.from_dict(data["signature"])

        return PCIEnvelope(
            version=data["version"],
            envelope_id=data["envelope_id"],
            timestamp=timestamp,
            nonce=data["nonce"],
            sender=Sender.from_dict(data["sender"]),
            payload=Payload.from_dict(data["payload"]),
            metadata=Metadata.from_dict(data["metadata"]),
            signature=signature,
        )

    def to_json(self) -> str:
        """Serialize to canonical JSON string."""
        return canonical_json(self.to_dict()).decode("utf-8")

    @staticmethod
    def from_json(json_str: str) -> PCIEnvelope:
        """Deserialize from JSON string."""
        return PCIEnvelope.from_dict(json.loads(json_str))


__all__ = [
    "PCIEnvelope",
    "Sender",
    "Payload",
    "Metadata",
    "Signature",
    "canonical_json",
    "compute_digest",
    "DOMAIN_PREFIX",
    "PROTOCOL_VERSION",
    "SIGNED_FIELDS",
]
