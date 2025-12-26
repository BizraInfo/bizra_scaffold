#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      BIZRA Key Rotation Script v1.0.0                         ║
║                        SEC-H-001 Compliance Implementation                    ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  This script implements cryptographic key rotation with:                      ║
║    - Zero-downtime key transition                                             ║
║    - Audit trail generation                                                   ║
║    - Secure key destruction                                                   ║
║    - Constitution-compliant key parameters                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python scripts/rotate_keys.py --role pat --output keys/
    python scripts/rotate_keys.py --role sat --output keys/ --force
    python scripts/rotate_keys.py --list-active
    python scripts/rotate_keys.py --revoke <key_id>
"""

import argparse
import hashlib
import json
import os
import secrets
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Cryptographic imports
try:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print(
        "⚠️ cryptography library not installed. Install with: pip install cryptography"
    )

try:
    import toml

    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (from constitution.toml)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_KEY_VALIDITY_DAYS = 90
GRACE_PERIOD_DAYS = 7
KEY_SIZE_BYTES = 32
SIGNATURE_ALGORITHM = "ed25519"
PQ_MIGRATION_TARGET = "dilithium5"

import logging
import tempfile

logger = logging.getLogger(__name__)


def _atomic_write_bytes(path: Path, data: bytes, mode: int = 0o644) -> None:
    """
    Atomically write bytes to a file using write-then-rename pattern.

    This prevents partial writes on crash - either the old file exists
    or the new complete file exists, never a partial file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory (ensures same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    tmp_path = Path(tmp_path)

    try:
        os.write(fd, data)
        os.fsync(fd)  # Ensure data hits disk before rename
        os.close(fd)
        fd = -1

        # Set permissions before rename
        os.chmod(tmp_path, mode)

        # Atomic rename
        tmp_path.replace(path)
    except Exception:
        # Clean up temp file on failure
        if fd >= 0:
            os.close(fd)
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _atomic_write_json(path: Path, data: dict, indent: int = 2) -> None:
    """Atomically write JSON data to a file."""
    content = json.dumps(data, indent=indent).encode("utf-8")
    _atomic_write_bytes(path, content, mode=0o644)


@dataclass
class KeyMetadata:
    """Metadata for a cryptographic key."""

    key_id: str
    role: str  # "pat" or "sat"
    algorithm: str
    created_at: str
    expires_at: str
    revoked_at: Optional[str] = None
    revocation_reason: Optional[str] = None
    fingerprint: str = ""
    status: str = "active"  # active, grace, revoked, expired
    predecessor_id: Optional[str] = None
    successor_id: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if key is currently valid for use."""
        if self.status in ("revoked", "expired"):
            return False
        now = datetime.now(timezone.utc)
        expires = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        return now < expires


@dataclass
class KeyPair:
    """A cryptographic key pair with metadata."""

    private_key_pem: bytes
    public_key_pem: bytes
    metadata: KeyMetadata


@dataclass
class RotationEvent:
    """Audit event for key rotation."""

    event_id: str
    event_type: str  # "generate", "rotate", "revoke", "expire"
    timestamp: str
    role: str
    old_key_id: Optional[str]
    new_key_id: Optional[str]
    reason: str
    operator: str
    constitution_hash: str


class KeyRotationManager:
    """
    Manages cryptographic key lifecycle for BIZRA agents.

    Implements SEC-H-001: Key Rotation Compliance
    - Maximum key validity: 90 days
    - Grace period for transition: 7 days
    - Secure key destruction after revocation
    - Full audit trail
    """

    def __init__(self, keys_dir: Path, constitution_path: Optional[Path] = None):
        self.keys_dir = Path(keys_dir)
        self.keys_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.keys_dir / "key_registry.json"
        self.audit_file = self.keys_dir / "rotation_audit.json"

        self.constitution_hash = self._load_constitution_hash(constitution_path)
        self.registry = self._load_registry()
        self.audit_log = self._load_audit_log()

    def _load_constitution_hash(self, path: Optional[Path]) -> str:
        """Compute BLAKE3 hash of constitution for binding."""
        if path is None:
            path = Path("constitution.toml")

        if not path.exists():
            return "NO_CONSTITUTION"

        try:
            import hashlib

            # Use SHA256 as fallback if BLAKE3 not available
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:32]
        except (IOError, OSError) as e:
            logger.warning(f"Failed to hash constitution at {path}: {e}")
            return "HASH_ERROR"

    def _load_registry(self) -> dict:
        """Load key registry from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"keys": {}, "active_pat": None, "active_sat": None}

    def _save_registry(self):
        """Persist key registry to disk atomically."""
        _atomic_write_json(self.metadata_file, self.registry)

    def _load_audit_log(self) -> list:
        """Load audit log from disk."""
        if self.audit_file.exists():
            with open(self.audit_file, "r") as f:
                return json.load(f)
        return []

    def _save_audit_log(self):
        """Persist audit log to disk atomically."""
        _atomic_write_json(self.audit_file, self.audit_log)

    def _generate_key_id(self) -> str:
        """Generate a unique key identifier."""
        return f"bizra-key-{secrets.token_hex(8)}"

    def _generate_event_id(self) -> str:
        """Generate a unique event identifier."""
        return f"evt-{secrets.token_hex(8)}"

    def _compute_fingerprint(self, public_key_pem: bytes) -> str:
        """Compute SHA256 fingerprint of public key."""
        return hashlib.sha256(public_key_pem).hexdigest()[:16]

    def _log_event(self, event: RotationEvent):
        """Append event to audit log."""
        self.audit_log.append(asdict(event))
        self._save_audit_log()
        print(f"📝 Audit: {event.event_type} - {event.reason}")

    def generate_key_pair(
        self, role: str, validity_days: int = DEFAULT_KEY_VALIDITY_DAYS
    ) -> KeyPair:
        """
        Generate a new Ed25519 key pair.

        Args:
            role: "pat" or "sat"
            validity_days: Key validity period

        Returns:
            KeyPair with private/public keys and metadata
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for key generation")

        if role not in ("pat", "sat"):
            raise ValueError(f"Invalid role: {role}. Must be 'pat' or 'sat'")

        # Generate Ed25519 key pair
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        # Create metadata
        now = datetime.now(timezone.utc)
        key_id = self._generate_key_id()

        metadata = KeyMetadata(
            key_id=key_id,
            role=role,
            algorithm=SIGNATURE_ALGORITHM,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(days=validity_days)).isoformat(),
            fingerprint=self._compute_fingerprint(public_pem),
            status="active",
        )

        return KeyPair(
            private_key_pem=private_pem, public_key_pem=public_pem, metadata=metadata
        )

    def store_key_pair(self, key_pair: KeyPair, set_active: bool = True) -> str:
        """
        Store a key pair securely.

        Args:
            key_pair: The key pair to store
            set_active: Whether to make this the active key for the role

        Returns:
            Key ID
        """
        key_id = key_pair.metadata.key_id
        role = key_pair.metadata.role

        # Create role directory
        role_dir = self.keys_dir / role
        role_dir.mkdir(exist_ok=True)

        # Store private key atomically (with restrictive permissions)
        private_path = role_dir / f"{key_id}.private.pem"
        _atomic_write_bytes(private_path, key_pair.private_key_pem, mode=0o600)

        # Store public key atomically
        public_path = role_dir / f"{key_id}.public.pem"
        _atomic_write_bytes(public_path, key_pair.public_key_pem, mode=0o644)

        # Update registry
        self.registry["keys"][key_id] = asdict(key_pair.metadata)

        if set_active:
            active_key = f"active_{role}"
            old_active = self.registry.get(active_key)

            # Link old → new
            if old_active and old_active in self.registry["keys"]:
                self.registry["keys"][old_active]["successor_id"] = key_id
                key_pair.metadata.predecessor_id = old_active
                self.registry["keys"][key_id]["predecessor_id"] = old_active

            self.registry[active_key] = key_id

        self._save_registry()

        # Log event
        event = RotationEvent(
            event_id=self._generate_event_id(),
            event_type="generate" if not key_pair.metadata.predecessor_id else "rotate",
            timestamp=datetime.now(timezone.utc).isoformat(),
            role=role,
            old_key_id=key_pair.metadata.predecessor_id,
            new_key_id=key_id,
            reason=(
                "Key generation"
                if not key_pair.metadata.predecessor_id
                else "Key rotation"
            ),
            operator=os.environ.get("USER", "system"),
            constitution_hash=self.constitution_hash,
        )
        self._log_event(event)

        return key_id

    def rotate_key(self, role: str, reason: str = "Scheduled rotation") -> str:
        """
        Rotate the active key for a role.

        Args:
            role: "pat" or "sat"
            reason: Reason for rotation

        Returns:
            New key ID
        """
        active_key = f"active_{role}"
        old_key_id = self.registry.get(active_key)

        # Generate new key
        key_pair = self.generate_key_pair(role)

        if old_key_id:
            # Mark old key for grace period
            self.registry["keys"][old_key_id]["status"] = "grace"
            grace_end = datetime.now(timezone.utc) + timedelta(days=GRACE_PERIOD_DAYS)
            self.registry["keys"][old_key_id]["grace_ends_at"] = grace_end.isoformat()
            key_pair.metadata.predecessor_id = old_key_id

        # Store new key
        new_key_id = self.store_key_pair(key_pair, set_active=True)

        # Log rotation event
        event = RotationEvent(
            event_id=self._generate_event_id(),
            event_type="rotate",
            timestamp=datetime.now(timezone.utc).isoformat(),
            role=role,
            old_key_id=old_key_id,
            new_key_id=new_key_id,
            reason=reason,
            operator=os.environ.get("USER", "system"),
            constitution_hash=self.constitution_hash,
        )
        self._log_event(event)

        print(f"✅ Key rotated: {old_key_id} → {new_key_id}")
        return new_key_id

    def revoke_key(self, key_id: str, reason: str) -> bool:
        """
        Revoke a key immediately.

        Args:
            key_id: Key to revoke
            reason: Reason for revocation

        Returns:
            Success status
        """
        if key_id not in self.registry["keys"]:
            print(f"❌ Key not found: {key_id}")
            return False

        key_meta = self.registry["keys"][key_id]
        role = key_meta["role"]

        # Mark as revoked
        key_meta["status"] = "revoked"
        key_meta["revoked_at"] = datetime.now(timezone.utc).isoformat()
        key_meta["revocation_reason"] = reason

        # If this was the active key, clear it
        active_key = f"active_{role}"
        if self.registry.get(active_key) == key_id:
            self.registry[active_key] = None
            print(f"⚠️ Active key revoked for {role}. Generate a new key.")

        self._save_registry()

        # Securely delete private key
        role_dir = self.keys_dir / role
        private_path = role_dir / f"{key_id}.private.pem"
        if private_path.exists():
            # Overwrite with random data before deletion
            with open(private_path, "wb") as f:
                f.write(secrets.token_bytes(1024))
            private_path.unlink()
            print(f"🔥 Private key securely destroyed: {key_id}")

        # Log revocation
        event = RotationEvent(
            event_id=self._generate_event_id(),
            event_type="revoke",
            timestamp=datetime.now(timezone.utc).isoformat(),
            role=role,
            old_key_id=key_id,
            new_key_id=None,
            reason=reason,
            operator=os.environ.get("USER", "system"),
            constitution_hash=self.constitution_hash,
        )
        self._log_event(event)

        return True

    def list_keys(
        self, role: Optional[str] = None, include_revoked: bool = False
    ) -> list:
        """List all keys, optionally filtered by role."""
        keys = []
        for key_id, meta in self.registry["keys"].items():
            if role and meta["role"] != role:
                continue
            if not include_revoked and meta["status"] == "revoked":
                continue

            # Check expiration
            expires = datetime.fromisoformat(meta["expires_at"].replace("Z", "+00:00"))
            if expires < datetime.now(timezone.utc) and meta["status"] not in (
                "revoked",
                "expired",
            ):
                meta["status"] = "expired"

            keys.append(meta)

        return keys

    def get_active_key(self, role: str) -> Optional[dict]:
        """Get the currently active key for a role."""
        active_key = f"active_{role}"
        key_id = self.registry.get(active_key)
        if key_id and key_id in self.registry["keys"]:
            return self.registry["keys"][key_id]
        return None

    def check_rotation_needed(self) -> list:
        """Check which keys need rotation."""
        needs_rotation = []
        warning_threshold = timedelta(days=14)  # Warn 14 days before expiry

        for role in ("pat", "sat"):
            key = self.get_active_key(role)
            if not key:
                needs_rotation.append(
                    {"role": role, "reason": "No active key", "urgency": "critical"}
                )
                continue

            expires = datetime.fromisoformat(key["expires_at"].replace("Z", "+00:00"))
            remaining = expires - datetime.now(timezone.utc)

            if remaining <= timedelta(days=0):
                needs_rotation.append(
                    {
                        "role": role,
                        "key_id": key["key_id"],
                        "reason": "Key expired",
                        "urgency": "critical",
                    }
                )
            elif remaining <= warning_threshold:
                needs_rotation.append(
                    {
                        "role": role,
                        "key_id": key["key_id"],
                        "reason": f"Expires in {remaining.days} days",
                        "urgency": "warning",
                    }
                )

        return needs_rotation


def main():
    parser = argparse.ArgumentParser(
        description="BIZRA Key Rotation Manager - SEC-H-001 Compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--role", choices=["pat", "sat"], help="Agent role for key operation"
    )
    parser.add_argument(
        "--output", default="keys/", help="Directory for key storage (default: keys/)"
    )
    parser.add_argument(
        "--generate", action="store_true", help="Generate a new key pair"
    )
    parser.add_argument("--rotate", action="store_true", help="Rotate the active key")
    parser.add_argument("--revoke", metavar="KEY_ID", help="Revoke a specific key")
    parser.add_argument(
        "--reason", default="Manual operation", help="Reason for rotation/revocation"
    )
    parser.add_argument("--list-active", action="store_true", help="List active keys")
    parser.add_argument(
        "--list-all", action="store_true", help="List all keys including revoked"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check if rotation is needed"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force operation without confirmation"
    )
    parser.add_argument(
        "--constitution", default="constitution.toml", help="Path to constitution.toml"
    )

    args = parser.parse_args()

    # Initialize manager
    manager = KeyRotationManager(
        keys_dir=Path(args.output),
        constitution_path=Path(args.constitution) if args.constitution else None,
    )

    print("═" * 60)
    print("  BIZRA Key Rotation Manager v1.0.0")
    print(f"  Constitution Hash: {manager.constitution_hash}")
    print("═" * 60)

    # Handle commands
    if args.generate:
        if not args.role:
            print("❌ --role required for key generation")
            sys.exit(1)

        key_pair = manager.generate_key_pair(args.role)
        key_id = manager.store_key_pair(key_pair)
        print(f"\n✅ Generated new {args.role.upper()} key: {key_id}")
        print(f"   Fingerprint: {key_pair.metadata.fingerprint}")
        print(f"   Expires: {key_pair.metadata.expires_at}")

    elif args.rotate:
        if not args.role:
            print("❌ --role required for key rotation")
            sys.exit(1)

        if not args.force:
            confirm = input(f"⚠️ Rotate {args.role.upper()} key? [y/N]: ")
            if confirm.lower() != "y":
                print("Aborted.")
                sys.exit(0)

        new_key_id = manager.rotate_key(args.role, args.reason)
        print(f"\n✅ Rotation complete: {new_key_id}")

    elif args.revoke:
        if not args.force:
            confirm = input(
                f"⚠️ REVOKE key {args.revoke}? This is IRREVERSIBLE! [y/N]: "
            )
            if confirm.lower() != "y":
                print("Aborted.")
                sys.exit(0)

        success = manager.revoke_key(args.revoke, args.reason)
        if success:
            print(f"\n✅ Key revoked: {args.revoke}")
        else:
            sys.exit(1)

    elif args.list_active:
        print("\n📋 Active Keys:")
        for role in ("pat", "sat"):
            key = manager.get_active_key(role)
            if key:
                print(f"\n  [{role.upper()}]")
                print(f"    ID: {key['key_id']}")
                print(f"    Fingerprint: {key['fingerprint']}")
                print(f"    Status: {key['status']}")
                print(f"    Expires: {key['expires_at']}")
            else:
                print(f"\n  [{role.upper()}] ⚠️ No active key!")

    elif args.list_all:
        print("\n📋 All Keys:")
        keys = manager.list_keys(include_revoked=True)
        for key in keys:
            status_icon = {
                "active": "✅",
                "grace": "⏳",
                "revoked": "❌",
                "expired": "💀",
            }.get(key["status"], "❓")

            print(f"\n  {status_icon} {key['key_id']}")
            print(f"     Role: {key['role'].upper()}")
            print(f"     Status: {key['status']}")
            print(f"     Created: {key['created_at']}")
            print(f"     Expires: {key['expires_at']}")

    elif args.check:
        print("\n🔍 Checking rotation status...")
        issues = manager.check_rotation_needed()

        if not issues:
            print("✅ All keys are healthy")
        else:
            for issue in issues:
                icon = "🔴" if issue["urgency"] == "critical" else "🟡"
                print(f"\n  {icon} {issue['role'].upper()}: {issue['reason']}")
                if "key_id" in issue:
                    print(f"     Key: {issue['key_id']}")

            if any(i["urgency"] == "critical" for i in issues):
                sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
