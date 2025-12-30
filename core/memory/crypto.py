"""
BIZRA Memory Scaffold v0.2 - Cryptographic Security Layer
==========================================================
Provides authenticated encryption for memory persistence.

Invariant M4 (Provenance): All stored data is cryptographically
bound to its context, preventing unauthorized modification.

Security Model:
    - Master Key: Never used directly for encryption
    - Derived Keys: HKDF-SHA256 generates context-specific keys
    - Encryption: AES-256-GCM (authenticated encryption with AAD)
    - Nonce: Cryptographically random, never reused

Key Hierarchy:
    master_key
        └── derive_key("expertise_store")
        └── derive_key("event_log")
        └── derive_key("session_xyz")

Author: BIZRA Core Team
Version: 0.2.0
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.exceptions import InvalidTag


# =============================================================================
# CONSTANTS
# =============================================================================

# Key sizes (bytes)
MASTER_KEY_SIZE = 32  # 256 bits
DERIVED_KEY_SIZE = 32  # 256 bits for AES-256
SALT_SIZE = 16  # 128 bits
NONCE_SIZE = 12  # 96 bits (GCM recommended)

# Algorithm identifiers
ALGORITHM_VERSION = "BIZRA-CRYPTO-V1"


# =============================================================================
# EXCEPTIONS
# =============================================================================


class CryptoError(Exception):
    """Base exception for cryptographic operations."""

    pass


class DecryptionError(CryptoError):
    """Raised when decryption fails (invalid key, tampered data, etc.)."""

    pass


class KeyDerivationError(CryptoError):
    """Raised when key derivation fails."""

    pass


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class EncryptedBlob:
    """
    Immutable container for encrypted data with all required metadata.

    Attributes:
        ciphertext: The encrypted data
        nonce: Unique value for this encryption (never reuse!)
        salt: Salt used in key derivation
        context: Domain separation string
        version: Algorithm version for forward compatibility
    """

    ciphertext: bytes
    nonce: bytes
    salt: bytes
    context: str
    version: str = ALGORITHM_VERSION

    def to_dict(self) -> Dict[str, str]:
        """Serialize to JSON-compatible dict with base64 encoding."""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode("ascii"),
            "nonce": base64.b64encode(self.nonce).decode("ascii"),
            "salt": base64.b64encode(self.salt).decode("ascii"),
            "context": self.context,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "EncryptedBlob":
        """Deserialize from JSON-compatible dict."""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            nonce=base64.b64decode(data["nonce"]),
            salt=base64.b64decode(data["salt"]),
            context=data["context"],
            version=data.get("version", ALGORITHM_VERSION),
        )

    def to_bytes(self) -> bytes:
        """Serialize to compact binary format."""
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedBlob":
        """Deserialize from compact binary format."""
        return cls.from_dict(json.loads(data.decode("utf-8")))


# =============================================================================
# KEY MANAGEMENT
# =============================================================================


def generate_master_key() -> bytes:
    """
    Generate a cryptographically secure master key.

    Returns:
        32 bytes of random data suitable for use as a master key

    Example:
        >>> key = generate_master_key()
        >>> len(key)
        32
    """
    return secrets.token_bytes(MASTER_KEY_SIZE)


def derive_key(
    master_key: bytes,
    context: str,
    salt: Optional[bytes] = None,
) -> Tuple[bytes, bytes]:
    """
    Derive a context-specific encryption key using HKDF.

    Uses HKDF (HMAC-based Key Derivation Function) with SHA-256
    to derive sub-keys from the master key. Each context gets
    a unique key, providing domain separation.

    Args:
        master_key: The root key (must be 32 bytes)
        context: Domain separation string (e.g., "expertise_store")
        salt: Optional salt (random bytes generated if not provided)

    Returns:
        Tuple of (derived_key, salt)

    Raises:
        KeyDerivationError: If master_key is invalid

    Example:
        >>> master = generate_master_key()
        >>> key, salt = derive_key(master, "my_context")
        >>> len(key)
        32
    """
    if len(master_key) != MASTER_KEY_SIZE:
        raise KeyDerivationError(
            f"Master key must be {MASTER_KEY_SIZE} bytes, got {len(master_key)}"
        )

    if salt is None:
        salt = secrets.token_bytes(SALT_SIZE)

    # HKDF with SHA-256
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=DERIVED_KEY_SIZE,
        salt=salt,
        info=context.encode("utf-8"),
    )

    derived = hkdf.derive(master_key)
    return derived, salt


# =============================================================================
# CRYPTO MANAGER
# =============================================================================


class CryptoManager:
    """
    High-level interface for symmetric encryption operations.

    This class manages key derivation and provides encrypt/decrypt
    operations with proper domain separation and authentication.

    The master key is never used directly for encryption. Instead,
    context-specific keys are derived using HKDF.

    Attributes:
        _master_key: The root key (kept in memory, never written to disk)

    Example:
        >>> crypto = CryptoManager(generate_master_key())
        >>> blob = crypto.encrypt(b"secret data", "my_context")
        >>> plaintext = crypto.decrypt(blob)
        >>> plaintext
        b'secret data'
    """

    __slots__ = ("_master_key", "_key_cache")

    def __init__(self, master_key: bytes) -> None:
        """
        Initialize the crypto manager.

        Args:
            master_key: The 32-byte master key

        Raises:
            KeyDerivationError: If master_key is invalid
        """
        if len(master_key) != MASTER_KEY_SIZE:
            raise KeyDerivationError(
                f"Master key must be {MASTER_KEY_SIZE} bytes"
            )
        self._master_key = master_key
        # Cache derived keys to avoid repeated HKDF operations
        # Key: (context, salt_hex) -> Value: derived_key
        self._key_cache: Dict[Tuple[str, str], bytes] = {}

    def _get_derived_key(
        self, context: str, salt: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """Get or derive a context-specific key."""
        if salt is not None:
            cache_key = (context, salt.hex())
            if cache_key in self._key_cache:
                return self._key_cache[cache_key], salt

        derived, used_salt = derive_key(self._master_key, context, salt)

        # Cache the result
        cache_key = (context, used_salt.hex())
        self._key_cache[cache_key] = derived

        return derived, used_salt

    def _get_stable_salt(self, context: str) -> bytes:
        """Derive a deterministic salt for stable context keys."""
        seed = f"bizra-stable-salt:{context}".encode("utf-8")
        return hashlib.sha256(seed).digest()[:SALT_SIZE]

    def encrypt(
        self,
        plaintext: bytes,
        context: str,
        associated_data: Optional[bytes] = None,
    ) -> EncryptedBlob:
        """
        Encrypt data with authenticated encryption.

        Uses AES-256-GCM which provides both confidentiality and
        integrity. The context is used to derive a unique key,
        ensuring domain separation.

        Args:
            plaintext: Data to encrypt
            context: Domain separation string
            associated_data: Optional AAD (authenticated but not encrypted)

        Returns:
            EncryptedBlob containing ciphertext and metadata

        Example:
            >>> blob = crypto.encrypt(b"secret", "messages")
        """
        # Derive a fresh key with random salt
        derived_key, salt = self._get_derived_key(context)

        # Generate random nonce (CRITICAL: never reuse with same key!)
        nonce = secrets.token_bytes(NONCE_SIZE)

        # Create cipher and encrypt
        aesgcm = AESGCM(derived_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)

        return EncryptedBlob(
            ciphertext=ciphertext,
            nonce=nonce,
            salt=salt,
            context=context,
            version=ALGORITHM_VERSION,
        )

    def decrypt(
        self,
        blob: EncryptedBlob,
        associated_data: Optional[bytes] = None,
    ) -> bytes:
        """
        Decrypt data and verify authenticity.

        Decrypts the ciphertext and verifies the GCM authentication tag.
        If the data has been tampered with, decryption will fail.

        Args:
            blob: The encrypted blob to decrypt
            associated_data: Must match AAD used during encryption

        Returns:
            Decrypted plaintext

        Raises:
            DecryptionError: If decryption fails (wrong key, tampered data)

        Example:
            >>> plaintext = crypto.decrypt(blob)
        """
        # Check version compatibility
        if blob.version != ALGORITHM_VERSION:
            raise DecryptionError(
                f"Unsupported algorithm version: {blob.version}"
            )

        # Re-derive the same key using stored salt
        derived_key, _ = self._get_derived_key(blob.context, blob.salt)

        # Decrypt and verify
        aesgcm = AESGCM(derived_key)
        try:
            plaintext = aesgcm.decrypt(blob.nonce, blob.ciphertext, associated_data)
            return plaintext
        except InvalidTag:
            raise DecryptionError(
                "Decryption failed: data may be corrupted or tampered"
            )

    def compute_hmac(self, data: bytes, context: str) -> bytes:
        """
        Compute HMAC-SHA256 for data integrity verification.

        Uses a derived key specific to the context for HMAC.

        Args:
            data: Data to authenticate
            context: Domain separation string

        Returns:
            32-byte HMAC digest
        """
        # Use a different context suffix for HMAC keys
        hmac_context = f"{context}:hmac"
        stable_salt = self._get_stable_salt(hmac_context)
        derived_key, _ = self._get_derived_key(hmac_context, stable_salt)

        return hmac.new(derived_key, data, hashlib.sha256).digest()

    def verify_hmac(
        self, data: bytes, expected_hmac: bytes, context: str
    ) -> bool:
        """
        Verify HMAC-SHA256 for data integrity.

        Uses constant-time comparison to prevent timing attacks.

        Args:
            data: Data to verify
            expected_hmac: Expected HMAC value
            context: Domain separation string

        Returns:
            True if HMAC is valid, False otherwise
        """
        computed = self.compute_hmac(data, context)
        return hmac.compare_digest(computed, expected_hmac)

    def derive_key(self, context: str) -> bytes:
        """
        Derive a context-specific key (public interface).
        
        This is useful for getting a key to use with external
        components like TamperEvidentLog.
        Uses a deterministic salt so keys are stable across runs.
        
        Args:
            context: Domain separation string
        
        Returns:
            32-byte derived key
        """
        stable_salt = self._get_stable_salt(context)
        key, _ = self._get_derived_key(context, stable_salt)
        return key

    def clear_cache(self) -> None:
        """Clear the derived key cache (use when context is no longer needed)."""
        self._key_cache.clear()

    def __repr__(self) -> str:
        return f"CryptoManager(key_id={hashlib.sha256(self._master_key).hexdigest()[:16]})"


# =============================================================================
# KEY STORAGE UTILITIES
# =============================================================================


def save_key_to_file(
    key: bytes,
    file_path: Union[str, Path],
    password: Optional[str] = None,
) -> None:
    """
    Save a key to a file with optional password protection.

    If password is provided, the key is encrypted before saving.
    Otherwise, it's saved as base64-encoded text.

    Args:
        key: The key to save
        file_path: Destination path
        password: Optional password for encryption

    Warning:
        Unencrypted keys should only be stored on secure filesystems
        with appropriate permissions!
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if password:
        # Derive encryption key from password
        salt = secrets.token_bytes(SALT_SIZE)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=DERIVED_KEY_SIZE,
            salt=salt,
            info=b"bizra-key-encryption",
        )
        enc_key = hkdf.derive(password.encode("utf-8"))

        # Encrypt the key
        nonce = secrets.token_bytes(NONCE_SIZE)
        aesgcm = AESGCM(enc_key)
        ciphertext = aesgcm.encrypt(nonce, key, None)

        # Save as JSON
        data = {
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "version": ALGORITHM_VERSION,
        }
        path.write_text(json.dumps(data, indent=2))
    else:
        # Save as base64 (WARNING: unencrypted!)
        path.write_text(base64.b64encode(key).decode("ascii"))

    # Set restrictive permissions (Unix only)
    try:
        os.chmod(path, 0o600)
    except (OSError, AttributeError):
        pass  # Windows doesn't support chmod


def load_key_from_file(
    file_path: Union[str, Path],
    password: Optional[str] = None,
) -> bytes:
    """
    Load a key from a file.

    Args:
        file_path: Source path
        password: Password if file is encrypted

    Returns:
        The loaded key

    Raises:
        DecryptionError: If password is wrong or file is corrupted
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    content = path.read_text().strip()

    try:
        data = json.loads(content)
        # File is encrypted
        if password is None:
            raise DecryptionError("Key file is encrypted but no password provided")

        salt = base64.b64decode(data["salt"])
        nonce = base64.b64decode(data["nonce"])
        ciphertext = base64.b64decode(data["ciphertext"])

        # Derive decryption key from password
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=DERIVED_KEY_SIZE,
            salt=salt,
            info=b"bizra-key-encryption",
        )
        dec_key = hkdf.derive(password.encode("utf-8"))

        # Decrypt
        aesgcm = AESGCM(dec_key)
        try:
            return aesgcm.decrypt(nonce, ciphertext, None)
        except InvalidTag:
            raise DecryptionError("Wrong password or corrupted key file")

    except (json.JSONDecodeError, KeyError):
        # File is unencrypted base64
        return base64.b64decode(content)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "CryptoManager",
    "EncryptedBlob",
    # Functions
    "generate_master_key",
    "derive_key",
    "save_key_to_file",
    "load_key_from_file",
    # Exceptions
    "CryptoError",
    "DecryptionError",
    "KeyDerivationError",
    # Constants
    "MASTER_KEY_SIZE",
    "ALGORITHM_VERSION",
]
