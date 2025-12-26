"""
BIZRA AEON OMEGA - HSM Key Management Abstraction Layer
Hardware Security Module Integration for Production Deployments

Supports:
- AWS CloudHSM
- Azure Key Vault (HSM-backed)
- HashiCorp Vault (Transit secrets engine)
- Google Cloud HSM
- Local PKCS#11 HSMs
- Software fallback for development

Author: BIZRA Security Team
Version: 1.0.0
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Cryptography
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger("bizra.security.hsm")


# =============================================================================
# EXCEPTIONS
# =============================================================================

class HSMError(Exception):
    """Base exception for HSM operations."""
    pass


class HSMConnectionError(HSMError):
    """Failed to connect to HSM."""
    pass


class HSMKeyNotFoundError(HSMError):
    """Requested key not found in HSM."""
    pass


class HSMOperationError(HSMError):
    """HSM operation failed."""
    pass


class HSMConfigurationError(HSMError):
    """HSM configuration is invalid."""
    pass


# =============================================================================
# KEY TYPES AND ALGORITHMS
# =============================================================================

class KeyType(Enum):
    """Supported key types."""
    AES_256 = "aes-256"
    AES_128 = "aes-128"
    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"
    EC_P256 = "ec-p256"
    EC_P384 = "ec-p384"
    EC_P521 = "ec-p521"
    ED25519 = "ed25519"
    HMAC_SHA256 = "hmac-sha256"
    HMAC_SHA512 = "hmac-sha512"


class KeyUsage(Enum):
    """Key usage purposes."""
    ENCRYPT = auto()
    DECRYPT = auto()
    SIGN = auto()
    VERIFY = auto()
    WRAP = auto()
    UNWRAP = auto()
    DERIVE = auto()


@dataclass
class KeyMetadata:
    """Metadata for a managed key."""
    key_id: str
    key_type: KeyType
    created_at: datetime
    expires_at: Optional[datetime]
    version: int
    usages: List[KeyUsage]
    tags: Dict[str, str] = field(default_factory=dict)
    is_exportable: bool = False
    is_enabled: bool = True
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


@dataclass
class EncryptionResult:
    """Result of an encryption operation."""
    ciphertext: bytes
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    key_version: int = 1
    algorithm: str = "AES-256-GCM"


@dataclass
class SignatureResult:
    """Result of a signing operation."""
    signature: bytes
    key_version: int = 1
    algorithm: str = "ECDSA-P256-SHA256"


# =============================================================================
# ABSTRACT HSM PROVIDER
# =============================================================================

class HSMProvider(ABC):
    """
    Abstract base class for HSM providers.
    
    Implementations must be thread-safe.
    """
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to HSM."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to HSM."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to HSM."""
        pass
    
    @abstractmethod
    def create_key(
        self,
        key_id: str,
        key_type: KeyType,
        usages: List[KeyUsage],
        expires_in: Optional[timedelta] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> KeyMetadata:
        """Create a new key in the HSM."""
        pass
    
    @abstractmethod
    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        """Get metadata for a key."""
        pass
    
    @abstractmethod
    def delete_key(self, key_id: str) -> None:
        """Delete a key from the HSM."""
        pass
    
    @abstractmethod
    def rotate_key(self, key_id: str) -> KeyMetadata:
        """Rotate a key to a new version."""
        pass
    
    @abstractmethod
    def encrypt(
        self,
        key_id: str,
        plaintext: bytes,
        context: Optional[Dict[str, str]] = None
    ) -> EncryptionResult:
        """Encrypt data using a key."""
        pass
    
    @abstractmethod
    def decrypt(
        self,
        key_id: str,
        ciphertext: bytes,
        iv: Optional[bytes] = None,
        tag: Optional[bytes] = None,
        context: Optional[Dict[str, str]] = None,
        key_version: Optional[int] = None
    ) -> bytes:
        """Decrypt data using a key."""
        pass
    
    @abstractmethod
    def sign(
        self,
        key_id: str,
        data: bytes,
        prehashed: bool = False
    ) -> SignatureResult:
        """Sign data using a key."""
        pass
    
    @abstractmethod
    def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        prehashed: bool = False,
        key_version: Optional[int] = None
    ) -> bool:
        """Verify a signature."""
        pass
    
    @abstractmethod
    def generate_random(self, num_bytes: int) -> bytes:
        """Generate random bytes from HSM RNG."""
        pass


# =============================================================================
# SOFTWARE HSM (Development/Testing)
# =============================================================================

class SoftwareHSM(HSMProvider):
    """
    Software-based HSM implementation for development and testing.
    
    WARNING: Not suitable for production use. Keys are stored in memory
    and optionally persisted to encrypted files.
    
    For production, use a real HSM provider (AWS CloudHSM, Azure Key Vault, etc.)
    """
    
    # File to store the master key alongside the encrypted storage
    _MASTER_KEY_FILE = ".master_key"
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        master_key: Optional[bytes] = None
    ):
        """
        Initialize software HSM.
        
        Args:
            storage_path: Optional path for encrypted key storage. When set,
                keys are persisted and a stable master key is required.
            master_key: Optional master key for storage encryption (32 bytes).
                If storage_path is set and no master_key provided, will attempt
                to load from disk or raise an error.
                
        Raises:
            ValueError: If storage_path is set but no master_key is available
                (either provided or loadable from disk).
        """
        self._storage_path = storage_path
        self._keys: Dict[str, Dict] = {}  # key_id -> {metadata, key_material}
        self._connected = False
        self._lock = threading.RLock()
        
        # Determine master key (stable for persistence)
        if storage_path:
            self._master_key = self._resolve_master_key(storage_path, master_key)
        else:
            # Ephemeral mode - random key is fine
            self._master_key = master_key or secrets.token_bytes(32)
        
        if storage_path:
            self._load_from_storage()
    
    def _resolve_master_key(
        self,
        storage_path: Path,
        provided_key: Optional[bytes]
    ) -> bytes:
        """Resolve master key for persistent storage.
        
        Priority:
        1. Use provided_key if given
        2. Load from environment variable BIZRA_HSM_MASTER_KEY
        3. Load from .master_key file next to storage
        4. Raise error (don't generate random - would be unrecoverable)
        """
        import os
        import base64
        
        # 1. Use provided key
        if provided_key:
            if len(provided_key) != 32:
                raise ValueError("Master key must be exactly 32 bytes")
            self._persist_master_key(storage_path, provided_key)
            return provided_key
        
        # 2. Check environment variable
        env_key = os.environ.get("BIZRA_HSM_MASTER_KEY")
        if env_key:
            try:
                key = base64.b64decode(env_key)
                if len(key) == 32:
                    logger.info("Using master key from BIZRA_HSM_MASTER_KEY environment")
                    return key
            except Exception:
                pass
        
        # 3. Load from file
        key_file = storage_path.parent / self._MASTER_KEY_FILE
        if key_file.exists():
            try:
                key = key_file.read_bytes()
                if len(key) == 32:
                    logger.info(f"Loaded master key from {key_file}")
                    return key
            except Exception as e:
                logger.warning(f"Failed to load master key from file: {e}")
        
        # 4. Fail - don't generate random key for persistent storage
        raise ValueError(
            f"SoftwareHSM with storage_path requires a stable master_key. "
            f"Either provide master_key parameter, set BIZRA_HSM_MASTER_KEY "
            f"environment variable (base64-encoded 32 bytes), or create "
            f"{key_file} with 32 random bytes. Generating a random key "
            f"would make stored keys unrecoverable on restart."
        )
    
    def _persist_master_key(self, storage_path: Path, key: bytes) -> None:
        """Persist master key to disk for future use.
        
        Safety: Will not overwrite an existing master key file to prevent
        accidental key loss. If overwrite is needed, delete the file first.
        """
        key_file = storage_path.parent / self._MASTER_KEY_FILE
        try:
            # Safety check: don't silently overwrite existing key file
            if key_file.exists():
                existing_key = key_file.read_bytes()
                if existing_key == key:
                    logger.debug("Master key file already exists with same key")
                    return
                else:
                    logger.warning(
                        f"Master key file {key_file} already exists with different content. "
                        f"Not overwriting to prevent key loss. Delete manually if intended."
                    )
                    return
            
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.write_bytes(key)
            # Restrict permissions (best effort on Windows)
            import stat
            key_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
            logger.info(f"Persisted master key to {key_file}")
        except Exception as e:
            logger.warning(f"Could not persist master key: {e}")
    
    def connect(self) -> None:
        """Connect (no-op for software HSM)."""
        self._connected = True
        logger.info("Software HSM connected (development mode)")
    
    def disconnect(self) -> None:
        """Disconnect and optionally save keys."""
        if self._storage_path:
            self._save_to_storage()
        self._connected = False
        logger.info("Software HSM disconnected")
    
    def is_connected(self) -> bool:
        return self._connected
    
    def create_key(
        self,
        key_id: str,
        key_type: KeyType,
        usages: List[KeyUsage],
        expires_in: Optional[timedelta] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> KeyMetadata:
        """Create a new key."""
        with self._lock:
            if key_id in self._keys:
                raise HSMOperationError(f"Key {key_id} already exists")
            
            # Generate key material
            key_material = self._generate_key_material(key_type)
            
            now = datetime.now(timezone.utc)
            expires_at = now + expires_in if expires_in else None
            
            metadata = KeyMetadata(
                key_id=key_id,
                key_type=key_type,
                created_at=now,
                expires_at=expires_at,
                version=1,
                usages=usages,
                tags=tags or {},
                is_exportable=False,
                is_enabled=True
            )
            
            self._keys[key_id] = {
                "metadata": metadata,
                "versions": {1: key_material}
            }
            
            logger.info(f"Created key: {key_id} ({key_type.value})")
            return metadata
    
    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        """Get key metadata."""
        with self._lock:
            if key_id not in self._keys:
                raise HSMKeyNotFoundError(f"Key {key_id} not found")
            return self._keys[key_id]["metadata"]
    
    def delete_key(self, key_id: str) -> None:
        """Delete a key."""
        with self._lock:
            if key_id not in self._keys:
                raise HSMKeyNotFoundError(f"Key {key_id} not found")
            del self._keys[key_id]
            logger.info(f"Deleted key: {key_id}")
    
    def rotate_key(self, key_id: str) -> KeyMetadata:
        """Rotate key to new version."""
        with self._lock:
            if key_id not in self._keys:
                raise HSMKeyNotFoundError(f"Key {key_id} not found")
            
            key_data = self._keys[key_id]
            metadata = key_data["metadata"]
            
            # Generate new version
            new_version = metadata.version + 1
            new_material = self._generate_key_material(metadata.key_type)
            key_data["versions"][new_version] = new_material
            
            # Update metadata
            metadata.version = new_version
            
            logger.info(f"Rotated key: {key_id} to version {new_version}")
            return metadata
    
    def encrypt(
        self,
        key_id: str,
        plaintext: bytes,
        context: Optional[Dict[str, str]] = None
    ) -> EncryptionResult:
        """Encrypt data."""
        with self._lock:
            if key_id not in self._keys:
                raise HSMKeyNotFoundError(f"Key {key_id} not found")
            
            key_data = self._keys[key_id]
            metadata = key_data["metadata"]
            
            if KeyUsage.ENCRYPT not in metadata.usages:
                raise HSMOperationError(f"Key {key_id} not authorized for encryption")
            
            key_material = key_data["versions"][metadata.version]
            
            # AES-256-GCM encryption
            iv = secrets.token_bytes(12)
            cipher = Cipher(
                algorithms.AES(key_material[:32]),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Add context as AAD if provided
            if context:
                aad = json.dumps(context, sort_keys=True).encode()
                encryptor.authenticate_additional_data(aad)
            
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            
            return EncryptionResult(
                ciphertext=ciphertext,
                iv=iv,
                tag=encryptor.tag,
                key_version=metadata.version,
                algorithm="AES-256-GCM"
            )
    
    def decrypt(
        self,
        key_id: str,
        ciphertext: bytes,
        iv: Optional[bytes] = None,
        tag: Optional[bytes] = None,
        context: Optional[Dict[str, str]] = None,
        key_version: Optional[int] = None
    ) -> bytes:
        """Decrypt data."""
        with self._lock:
            if key_id not in self._keys:
                raise HSMKeyNotFoundError(f"Key {key_id} not found")
            
            key_data = self._keys[key_id]
            metadata = key_data["metadata"]
            
            if KeyUsage.DECRYPT not in metadata.usages:
                raise HSMOperationError(f"Key {key_id} not authorized for decryption")
            
            version = key_version or metadata.version
            if version not in key_data["versions"]:
                raise HSMOperationError(f"Key version {version} not found")
            
            key_material = key_data["versions"][version]
            
            if iv is None or tag is None:
                raise HSMOperationError("IV and tag required for decryption")
            
            cipher = Cipher(
                algorithms.AES(key_material[:32]),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            if context:
                aad = json.dumps(context, sort_keys=True).encode()
                decryptor.authenticate_additional_data(aad)
            
            return decryptor.update(ciphertext) + decryptor.finalize()
    
    def sign(
        self,
        key_id: str,
        data: bytes,
        prehashed: bool = False
    ) -> SignatureResult:
        """Sign data using HMAC-SHA256."""
        with self._lock:
            if key_id not in self._keys:
                raise HSMKeyNotFoundError(f"Key {key_id} not found")
            
            key_data = self._keys[key_id]
            metadata = key_data["metadata"]
            
            if KeyUsage.SIGN not in metadata.usages:
                raise HSMOperationError(f"Key {key_id} not authorized for signing")
            
            key_material = key_data["versions"][metadata.version]
            
            import hmac as hmac_lib
            signature = hmac_lib.new(
                key_material,
                data,
                hashlib.sha256
            ).digest()
            
            return SignatureResult(
                signature=signature,
                key_version=metadata.version,
                algorithm="HMAC-SHA256"
            )
    
    def verify(
        self,
        key_id: str,
        data: bytes,
        signature: bytes,
        prehashed: bool = False,
        key_version: Optional[int] = None
    ) -> bool:
        """Verify a signature."""
        with self._lock:
            if key_id not in self._keys:
                raise HSMKeyNotFoundError(f"Key {key_id} not found")
            
            key_data = self._keys[key_id]
            metadata = key_data["metadata"]
            
            if KeyUsage.VERIFY not in metadata.usages:
                raise HSMOperationError(f"Key {key_id} not authorized for verification")
            
            version = key_version or metadata.version
            if version not in key_data["versions"]:
                raise HSMOperationError(f"Key version {version} not found")
            
            key_material = key_data["versions"][version]
            
            import hmac as hmac_lib
            expected = hmac_lib.new(
                key_material,
                data,
                hashlib.sha256
            ).digest()
            
            return hmac_lib.compare_digest(signature, expected)
    
    def generate_random(self, num_bytes: int) -> bytes:
        """Generate random bytes."""
        return secrets.token_bytes(num_bytes)
    
    def _generate_key_material(self, key_type: KeyType) -> bytes:
        """Generate key material for a key type."""
        if key_type in (KeyType.AES_256, KeyType.HMAC_SHA256):
            return secrets.token_bytes(32)
        elif key_type == KeyType.AES_128:
            return secrets.token_bytes(16)
        elif key_type == KeyType.HMAC_SHA512:
            return secrets.token_bytes(64)
        else:
            # For asymmetric keys, we'd generate key pairs
            # Simplified: just generate symmetric key material
            return secrets.token_bytes(32)
    
    def _save_to_storage(self) -> None:
        """Save encrypted keys to storage."""
        if not self._storage_path:
            return
        
        # Serialize keys
        data = {}
        for key_id, key_data in self._keys.items():
            metadata = key_data["metadata"]
            data[key_id] = {
                "metadata": {
                    "key_id": metadata.key_id,
                    "key_type": metadata.key_type.value,
                    "created_at": metadata.created_at.isoformat(),
                    "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                    "version": metadata.version,
                    "usages": [u.name for u in metadata.usages],
                    "tags": metadata.tags,
                },
                "versions": {
                    str(v): base64.b64encode(k).decode()
                    for v, k in key_data["versions"].items()
                }
            }
        
        plaintext = json.dumps(data).encode()
        
        # Encrypt with master key
        iv = secrets.token_bytes(12)
        cipher = Cipher(
            algorithms.AES(self._master_key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Write to file
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, "wb") as f:
            f.write(iv + encryptor.tag + ciphertext)
        
        logger.info(f"Saved {len(self._keys)} keys to storage")
    
    def _load_from_storage(self) -> None:
        """Load encrypted keys from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return
        
        try:
            with open(self._storage_path, "rb") as f:
                encrypted = f.read()
            
            iv = encrypted[:12]
            tag = encrypted[12:28]
            ciphertext = encrypted[28:]
            
            cipher = Cipher(
                algorithms.AES(self._master_key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            data = json.loads(plaintext.decode())
            
            for key_id, key_data in data.items():
                meta = key_data["metadata"]
                metadata = KeyMetadata(
                    key_id=meta["key_id"],
                    key_type=KeyType(meta["key_type"]),
                    created_at=datetime.fromisoformat(meta["created_at"]),
                    expires_at=datetime.fromisoformat(meta["expires_at"]) if meta["expires_at"] else None,
                    version=meta["version"],
                    usages=[KeyUsage[u] for u in meta["usages"]],
                    tags=meta["tags"]
                )
                
                versions = {
                    int(v): base64.b64decode(k)
                    for v, k in key_data["versions"].items()
                }
                
                self._keys[key_id] = {
                    "metadata": metadata,
                    "versions": versions
                }
            
            logger.info(f"Loaded {len(self._keys)} keys from storage")
        except Exception as e:
            logger.error(f"Failed to load keys from storage: {e}")


# =============================================================================
# AWS CloudHSM PROVIDER (Stub)
# =============================================================================

class AWSCloudHSMProvider(HSMProvider):
    """
    AWS CloudHSM provider stub.
    
    Requires AWS CloudHSM client library and configured cluster.
    """
    
    def __init__(
        self,
        cluster_id: str,
        hsm_user: str,
        hsm_password: str,
        region: str = "us-east-1"
    ):
        self._cluster_id = cluster_id
        self._hsm_user = hsm_user
        self._hsm_password = hsm_password
        self._region = region
        self._connected = False
        
        logger.warning(
            "AWSCloudHSMProvider is a stub. Implement with AWS CloudHSM client."
        )
    
    def connect(self) -> None:
        raise NotImplementedError("Implement with AWS CloudHSM client")
    
    def disconnect(self) -> None:
        raise NotImplementedError("Implement with AWS CloudHSM client")
    
    def is_connected(self) -> bool:
        return self._connected
    
    def create_key(self, key_id: str, key_type: KeyType, usages: List[KeyUsage], 
                   expires_in: Optional[timedelta] = None, tags: Optional[Dict[str, str]] = None) -> KeyMetadata:
        raise NotImplementedError("Implement with AWS CloudHSM client")
    
    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        raise NotImplementedError("Implement with AWS CloudHSM client")
    
    def delete_key(self, key_id: str) -> None:
        raise NotImplementedError("Implement with AWS CloudHSM client")
    
    def rotate_key(self, key_id: str) -> KeyMetadata:
        raise NotImplementedError("Implement with AWS CloudHSM client")
    
    def encrypt(self, key_id: str, plaintext: bytes, context: Optional[Dict[str, str]] = None) -> EncryptionResult:
        raise NotImplementedError("Implement with AWS CloudHSM client")
    
    def decrypt(self, key_id: str, ciphertext: bytes, iv: Optional[bytes] = None, 
                tag: Optional[bytes] = None, context: Optional[Dict[str, str]] = None,
                key_version: Optional[int] = None) -> bytes:
        raise NotImplementedError("Implement with AWS CloudHSM client")
    
    def sign(self, key_id: str, data: bytes, prehashed: bool = False) -> SignatureResult:
        raise NotImplementedError("Implement with AWS CloudHSM client")
    
    def verify(self, key_id: str, data: bytes, signature: bytes, prehashed: bool = False,
               key_version: Optional[int] = None) -> bool:
        raise NotImplementedError("Implement with AWS CloudHSM client")
    
    def generate_random(self, num_bytes: int) -> bytes:
        raise NotImplementedError("Implement with AWS CloudHSM client")


# =============================================================================
# AZURE KEY VAULT PROVIDER (Stub)
# =============================================================================

class AzureKeyVaultProvider(HSMProvider):
    """
    Azure Key Vault (HSM-backed) provider stub.
    
    Requires azure-keyvault-keys and azure-identity packages.
    """
    
    def __init__(
        self,
        vault_url: str,
        credential: Optional[Any] = None
    ):
        self._vault_url = vault_url
        self._credential = credential
        self._connected = False
        
        logger.warning(
            "AzureKeyVaultProvider is a stub. Implement with azure-keyvault-keys."
        )
    
    def connect(self) -> None:
        raise NotImplementedError("Implement with azure-keyvault-keys")
    
    def disconnect(self) -> None:
        raise NotImplementedError("Implement with azure-keyvault-keys")
    
    def is_connected(self) -> bool:
        return self._connected
    
    def create_key(self, key_id: str, key_type: KeyType, usages: List[KeyUsage],
                   expires_in: Optional[timedelta] = None, tags: Optional[Dict[str, str]] = None) -> KeyMetadata:
        raise NotImplementedError("Implement with azure-keyvault-keys")
    
    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        raise NotImplementedError("Implement with azure-keyvault-keys")
    
    def delete_key(self, key_id: str) -> None:
        raise NotImplementedError("Implement with azure-keyvault-keys")
    
    def rotate_key(self, key_id: str) -> KeyMetadata:
        raise NotImplementedError("Implement with azure-keyvault-keys")
    
    def encrypt(self, key_id: str, plaintext: bytes, context: Optional[Dict[str, str]] = None) -> EncryptionResult:
        raise NotImplementedError("Implement with azure-keyvault-keys")
    
    def decrypt(self, key_id: str, ciphertext: bytes, iv: Optional[bytes] = None,
                tag: Optional[bytes] = None, context: Optional[Dict[str, str]] = None,
                key_version: Optional[int] = None) -> bytes:
        raise NotImplementedError("Implement with azure-keyvault-keys")
    
    def sign(self, key_id: str, data: bytes, prehashed: bool = False) -> SignatureResult:
        raise NotImplementedError("Implement with azure-keyvault-keys")
    
    def verify(self, key_id: str, data: bytes, signature: bytes, prehashed: bool = False,
               key_version: Optional[int] = None) -> bool:
        raise NotImplementedError("Implement with azure-keyvault-keys")
    
    def generate_random(self, num_bytes: int) -> bytes:
        raise NotImplementedError("Implement with azure-keyvault-keys")


# =============================================================================
# HASHICORP VAULT PROVIDER (Stub)
# =============================================================================

class HashiCorpVaultProvider(HSMProvider):
    """
    HashiCorp Vault Transit secrets engine provider stub.
    
    Requires hvac package.
    """
    
    def __init__(
        self,
        vault_addr: str,
        token: Optional[str] = None,
        mount_point: str = "transit"
    ):
        self._vault_addr = vault_addr
        self._token = token
        self._mount_point = mount_point
        self._connected = False
        
        logger.warning(
            "HashiCorpVaultProvider is a stub. Implement with hvac."
        )
    
    def connect(self) -> None:
        raise NotImplementedError("Implement with hvac")
    
    def disconnect(self) -> None:
        raise NotImplementedError("Implement with hvac")
    
    def is_connected(self) -> bool:
        return self._connected
    
    def create_key(self, key_id: str, key_type: KeyType, usages: List[KeyUsage],
                   expires_in: Optional[timedelta] = None, tags: Optional[Dict[str, str]] = None) -> KeyMetadata:
        raise NotImplementedError("Implement with hvac")
    
    def get_key_metadata(self, key_id: str) -> KeyMetadata:
        raise NotImplementedError("Implement with hvac")
    
    def delete_key(self, key_id: str) -> None:
        raise NotImplementedError("Implement with hvac")
    
    def rotate_key(self, key_id: str) -> KeyMetadata:
        raise NotImplementedError("Implement with hvac")
    
    def encrypt(self, key_id: str, plaintext: bytes, context: Optional[Dict[str, str]] = None) -> EncryptionResult:
        raise NotImplementedError("Implement with hvac")
    
    def decrypt(self, key_id: str, ciphertext: bytes, iv: Optional[bytes] = None,
                tag: Optional[bytes] = None, context: Optional[Dict[str, str]] = None,
                key_version: Optional[int] = None) -> bytes:
        raise NotImplementedError("Implement with hvac")
    
    def sign(self, key_id: str, data: bytes, prehashed: bool = False) -> SignatureResult:
        raise NotImplementedError("Implement with hvac")
    
    def verify(self, key_id: str, data: bytes, signature: bytes, prehashed: bool = False,
               key_version: Optional[int] = None) -> bool:
        raise NotImplementedError("Implement with hvac")
    
    def generate_random(self, num_bytes: int) -> bytes:
        raise NotImplementedError("Implement with hvac")


# =============================================================================
# KEY MANAGEMENT SERVICE FACADE
# =============================================================================

class KeyManagementService:
    """
    High-level key management facade.
    
    Provides a unified interface for key operations across different
    HSM backends, with automatic provider selection based on configuration.
    """
    
    def __init__(self, provider: HSMProvider):
        self._provider = provider
        self._lock = threading.RLock()
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> KeyManagementService:
        """
        Create KMS from configuration dictionary.
        
        Config format:
        {
            "provider": "software" | "aws" | "azure" | "hashicorp",
            "aws": { "cluster_id": "...", ... },
            "azure": { "vault_url": "...", ... },
            "hashicorp": { "vault_addr": "...", ... },
            "software": { "storage_path": "...", ... }
        }
        """
        provider_type = config.get("provider", "software")
        
        if provider_type == "software":
            sw_config = config.get("software", {})
            storage_path = sw_config.get("storage_path")
            provider = SoftwareHSM(
                storage_path=Path(storage_path) if storage_path else None
            )
        elif provider_type == "aws":
            aws_config = config.get("aws", {})
            provider = AWSCloudHSMProvider(**aws_config)
        elif provider_type == "azure":
            azure_config = config.get("azure", {})
            provider = AzureKeyVaultProvider(**azure_config)
        elif provider_type == "hashicorp":
            hc_config = config.get("hashicorp", {})
            provider = HashiCorpVaultProvider(**hc_config)
        else:
            raise HSMConfigurationError(f"Unknown provider: {provider_type}")
        
        provider.connect()
        return cls(provider)
    
    def create_jwt_signing_key(
        self,
        key_id: str = "bizra-jwt-signing",
        expires_in: Optional[timedelta] = None
    ) -> KeyMetadata:
        """Create a dedicated JWT signing key."""
        return self._provider.create_key(
            key_id=key_id,
            key_type=KeyType.HMAC_SHA256,
            usages=[KeyUsage.SIGN, KeyUsage.VERIFY],
            expires_in=expires_in,
            tags={"purpose": "jwt-signing", "system": "bizra"}
        )
    
    def create_encryption_key(
        self,
        key_id: str,
        expires_in: Optional[timedelta] = None
    ) -> KeyMetadata:
        """Create an AES-256 encryption key."""
        return self._provider.create_key(
            key_id=key_id,
            key_type=KeyType.AES_256,
            usages=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
            expires_in=expires_in,
            tags={"purpose": "data-encryption", "system": "bizra"}
        )
    
    def sign_jwt_payload(self, key_id: str, payload: bytes) -> bytes:
        """Sign a JWT payload."""
        result = self._provider.sign(key_id, payload)
        return result.signature
    
    def verify_jwt_signature(
        self,
        key_id: str,
        payload: bytes,
        signature: bytes
    ) -> bool:
        """Verify a JWT signature."""
        return self._provider.verify(key_id, payload, signature)
    
    def encrypt_sensitive_data(
        self,
        key_id: str,
        data: bytes,
        context: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, bytes, bytes, int]:
        """
        Encrypt sensitive data.
        
        Returns: (ciphertext, iv, tag, key_version)
        """
        result = self._provider.encrypt(key_id, data, context)
        return result.ciphertext, result.iv, result.tag, result.key_version
    
    def decrypt_sensitive_data(
        self,
        key_id: str,
        ciphertext: bytes,
        iv: bytes,
        tag: bytes,
        context: Optional[Dict[str, str]] = None,
        key_version: Optional[int] = None
    ) -> bytes:
        """Decrypt sensitive data."""
        return self._provider.decrypt(
            key_id, ciphertext, iv, tag, context, key_version
        )
    
    def rotate_key(self, key_id: str) -> KeyMetadata:
        """Rotate a key to a new version."""
        return self._provider.rotate_key(key_id)
    
    def get_random_bytes(self, num_bytes: int) -> bytes:
        """Get random bytes from HSM RNG."""
        return self._provider.generate_random(num_bytes)
    
    def close(self) -> None:
        """Close the provider connection."""
        self._provider.disconnect()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_development_kms() -> KeyManagementService:
    """Create a development KMS using software HSM."""
    provider = SoftwareHSM()
    provider.connect()
    return KeyManagementService(provider)


def create_production_kms_from_env() -> KeyManagementService:
    """
    Create production KMS from environment variables.
    
    Environment variables:
    - BIZRA_HSM_PROVIDER: "aws" | "azure" | "hashicorp"
    - AWS_HSM_CLUSTER_ID, AWS_HSM_USER, AWS_HSM_PASSWORD
    - AZURE_VAULT_URL
    - VAULT_ADDR, VAULT_TOKEN
    """
    provider_type = os.environ.get("BIZRA_HSM_PROVIDER", "software")
    
    config = {"provider": provider_type}
    
    if provider_type == "aws":
        config["aws"] = {
            "cluster_id": os.environ["AWS_HSM_CLUSTER_ID"],
            "hsm_user": os.environ["AWS_HSM_USER"],
            "hsm_password": os.environ["AWS_HSM_PASSWORD"],
            "region": os.environ.get("AWS_REGION", "us-east-1")
        }
    elif provider_type == "azure":
        config["azure"] = {
            "vault_url": os.environ["AZURE_VAULT_URL"]
        }
    elif provider_type == "hashicorp":
        config["hashicorp"] = {
            "vault_addr": os.environ["VAULT_ADDR"],
            "token": os.environ.get("VAULT_TOKEN")
        }
    
    return KeyManagementService.create_from_config(config)


__all__ = [
    # Exceptions
    "HSMError",
    "HSMConnectionError",
    "HSMKeyNotFoundError",
    "HSMOperationError",
    "HSMConfigurationError",
    
    # Types
    "KeyType",
    "KeyUsage",
    "KeyMetadata",
    "EncryptionResult",
    "SignatureResult",
    
    # Providers
    "HSMProvider",
    "SoftwareHSM",
    "AWSCloudHSMProvider",
    "AzureKeyVaultProvider",
    "HashiCorpVaultProvider",
    
    # Service
    "KeyManagementService",
    
    # Factory functions
    "create_development_kms",
    "create_production_kms_from_env",
]
