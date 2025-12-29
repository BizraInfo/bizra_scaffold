"""
BIZRA Node Zero Identity Module
════════════════════════════════════════════════════════════════════════════════

Defines the Genesis Node (Node0) identity—the first node in the BIZRA network.
This is YOUR machine, running as the home base of the entire ecosystem.

Node0 is special because:
- It is the FIRST node (genesis)
- It hosts the FIRST architect (Momo)
- It contains the FIRST 3 years of data (genesis corpus)
- It must be provably authentic and reproducible

The Node0 identity is:
1. Machine-bound (hardware fingerprint)
2. Owner-bound (your public key)
3. Constitution-bound (genesis constitution hash)
4. Timeline-bound (2023-01-14 → now)

Design Philosophy:
- Giants Protocol: Node0 stands on the shoulders of 3 years of accumulated wisdom
- SNR-Weighted: Only high-signal identity components matter
- Graph of Thoughts: Node0 is the root of the BIZRA graph
- Proof of Impact: Node0 proves the First Architect's contribution

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import secrets
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add repo root to path
MODULE_DIR = Path(__file__).resolve().parent
CORE_DIR = MODULE_DIR.parent
REPO_ROOT = CORE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger("bizra.genesis.node_zero")


# =============================================================================
# CONSTANTS
# =============================================================================

# Genesis timeline (provable from chat exports)
GENESIS_EPOCH = datetime(2023, 1, 14, 0, 0, 0, tzinfo=timezone.utc)
GENESIS_EPOCH_ISO = "2023-01-14T00:00:00+00:00"

# Node0 role identifiers
NODE_ZERO_ROLE = "GENESIS_NODE"
FIRST_ARCHITECT_ROLE = "FIRST_ARCHITECT"

# Minimum Ihsān for genesis operations
GENESIS_IHSAN_THRESHOLD = 0.95

# Host-centric system manifest (Node0 attestation)
SYSTEM_MANIFEST_PATH = REPO_ROOT / "data" / "genesis" / "GENESIS_SYSTEM_MANIFEST.json"


def _hash_bytes(data: bytes) -> str:
    """Hash bytes with BLAKE3 when available, SHA-256 otherwise."""
    if BLAKE3_AVAILABLE:
        return blake3.blake3(data).hexdigest()
    return hashlib.sha256(data).hexdigest()


def load_system_manifest(
    path: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Path]]:
    """
    Load the host-centric system manifest if present.

    Returns:
        (manifest_data, manifest_hash, manifest_path)
    """
    if path is None:
        path = SYSTEM_MANIFEST_PATH
    if not path.exists():
        return None, None, None
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
        return data, _hash_bytes(raw), path
    except Exception as exc:
        logger.warning(f"Failed to load system manifest: {exc}")
        return None, None, None


# =============================================================================
# NODE IDENTITY STRUCTURES
# =============================================================================


class NodeRole(Enum):
    """Roles a node can have in the BIZRA network."""
    
    GENESIS = auto()      # The first node (Node0)
    VALIDATOR = auto()    # Validates attestations
    PRODUCER = auto()     # Produces artifacts
    RELAY = auto()        # Relays messages
    OBSERVER = auto()     # Read-only observer


@dataclass
class MachineFingerprint:
    """
    Privacy-preserving machine fingerprint.
    
    Identifies the machine without exposing sensitive details.
    Uses one-way hashes of hardware identifiers.
    """
    
    # Hashed identifiers (one-way, privacy-preserving)
    hostname_hash: str
    platform_hash: str
    processor_hash: str
    machine_hash: str
    
    # Combined fingerprint
    fingerprint: str
    
    # Metadata (non-identifying)
    platform_type: str  # "Windows", "Linux", "Darwin"
    architecture: str   # "x86_64", "arm64"
    
    @classmethod
    def from_current_machine(cls) -> "MachineFingerprint":
        """Generate fingerprint from current machine."""
        
        # Hash individual components
        hostname_hash = hashlib.sha256(platform.node().encode()).hexdigest()[:16]
        platform_hash = hashlib.sha256(platform.platform().encode()).hexdigest()[:16]
        processor_hash = hashlib.sha256(platform.processor().encode()).hexdigest()[:16]
        machine_hash = hashlib.sha256(platform.machine().encode()).hexdigest()[:16]
        
        # Combined fingerprint
        combined = f"{hostname_hash}:{platform_hash}:{processor_hash}:{machine_hash}"
        fingerprint = hashlib.sha256(combined.encode()).hexdigest()
        
        return cls(
            hostname_hash=hostname_hash,
            platform_hash=platform_hash,
            processor_hash=processor_hash,
            machine_hash=machine_hash,
            fingerprint=fingerprint,
            platform_type=platform.system(),
            architecture=platform.machine(),
        )
    
    def to_dict(self) -> Dict[str, str]:
        """Serialize to dictionary."""
        return asdict(self)


@dataclass
class OwnerAttestation:
    """
    Attestation binding an owner to a node.
    
    The First Architect (Momo) is the genesis owner of Node0.
    This attestation is signed and irrevocable.
    """
    
    owner_id: str
    owner_alias: str
    owner_public_key: str
    
    # Role and capabilities
    role: str
    capabilities: List[str]
    
    # Attestation metadata
    attested_at: datetime
    attestation_hash: str
    signature: Optional[str] = None
    
    @classmethod
    def create_genesis_owner(
        cls,
        owner_alias: str = "Momo",
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
    ) -> Tuple["OwnerAttestation", Optional[ed25519.Ed25519PrivateKey]]:
        """
        Create attestation for the First Architect (genesis owner).
        
        Returns:
            Tuple of (attestation, private_key)
        """
        # Generate or use provided key
        if private_key is None and CRYPTO_AVAILABLE:
            private_key = ed25519.Ed25519PrivateKey.generate()
        
        # Extract public key
        if private_key and CRYPTO_AVAILABLE:
            public_key = private_key.public_key()
            public_key_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            public_key_hex = public_key_bytes.hex()
        else:
            # Fallback: generate deterministic placeholder
            public_key_hex = hashlib.sha256(owner_alias.encode()).hexdigest()
        
        # Generate owner ID
        owner_id = f"owner_{hashlib.sha256(public_key_hex.encode()).hexdigest()[:12]}"
        
        # Genesis capabilities
        capabilities = [
            "GENESIS_AUTHORITY",
            "CONSTITUTION_AMEND",
            "NODE_PROVISION",
            "ATTESTATION_ISSUE",
            "REWARD_CLAIM",
            "PROOF_OF_IMPACT",
        ]
        
        attested_at = datetime.now(timezone.utc)
        
        # Compute attestation hash
        attestation_data = {
            "owner_id": owner_id,
            "owner_alias": owner_alias,
            "owner_public_key": public_key_hex,
            "role": FIRST_ARCHITECT_ROLE,
            "capabilities": capabilities,
            "attested_at": attested_at.isoformat(),
        }
        attestation_json = json.dumps(attestation_data, sort_keys=True, separators=(",", ":"))
        
        if BLAKE3_AVAILABLE:
            attestation_hash = blake3.blake3(attestation_json.encode()).hexdigest()
        else:
            attestation_hash = hashlib.sha256(attestation_json.encode()).hexdigest()
        
        # Sign if key available
        signature = None
        if private_key and CRYPTO_AVAILABLE:
            sig_bytes = private_key.sign(attestation_hash.encode())
            signature = sig_bytes.hex()
        
        attestation = cls(
            owner_id=owner_id,
            owner_alias=owner_alias,
            owner_public_key=public_key_hex,
            role=FIRST_ARCHITECT_ROLE,
            capabilities=capabilities,
            attested_at=attested_at,
            attestation_hash=attestation_hash,
            signature=signature,
        )
        
        return attestation, private_key
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "owner_id": self.owner_id,
            "owner_alias": self.owner_alias,
            "owner_public_key": self.owner_public_key,
            "role": self.role,
            "capabilities": self.capabilities,
            "attested_at": self.attested_at.isoformat(),
            "attestation_hash": self.attestation_hash,
            "signature": self.signature,
        }


@dataclass
class ConstitutionBinding:
    """
    Binding to the genesis constitution.
    
    The constitution hash at genesis time is immutable and
    defines the foundational rules of the BIZRA network.
    """
    
    constitution_hash: str
    constitution_version: str
    binding_timestamp: datetime
    
    # Key values from constitution
    ihsan_threshold: float
    snr_high_threshold: float
    snr_medium_threshold: float
    
    @classmethod
    def from_constitution_file(
        cls,
        constitution_path: Optional[Path] = None,
    ) -> "ConstitutionBinding":
        """Load binding from constitution file."""
        
        if constitution_path is None:
            constitution_path = REPO_ROOT / "constitution.toml"
        
        # Compute hash
        if constitution_path.exists():
            content = constitution_path.read_bytes()
            if BLAKE3_AVAILABLE:
                constitution_hash = blake3.blake3(content).hexdigest()
            else:
                constitution_hash = hashlib.sha256(content).hexdigest()
            
            # Parse TOML for key values
            try:
                import tomllib
                config = tomllib.loads(content.decode())
                ihsan_threshold = config.get("ethics", {}).get("ihsan_threshold", 0.95)
                snr_config = config.get("snr", {})
                snr_high = snr_config.get("high_threshold", 0.8)
                snr_medium = snr_config.get("medium_threshold", 0.5)
                version = config.get("metadata", {}).get("version", "1.0.0")
            except Exception:
                ihsan_threshold = 0.95
                snr_high = 0.8
                snr_medium = 0.5
                version = "1.0.0"
        else:
            # Fallback: use defaults
            constitution_hash = hashlib.sha256(b"genesis").hexdigest()
            ihsan_threshold = 0.95
            snr_high = 0.8
            snr_medium = 0.5
            version = "1.0.0"
        
        return cls(
            constitution_hash=constitution_hash,
            constitution_version=version,
            binding_timestamp=datetime.now(timezone.utc),
            ihsan_threshold=ihsan_threshold,
            snr_high_threshold=snr_high,
            snr_medium_threshold=snr_medium,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "constitution_hash": self.constitution_hash,
            "constitution_version": self.constitution_version,
            "binding_timestamp": self.binding_timestamp.isoformat(),
            "ihsan_threshold": self.ihsan_threshold,
            "snr_high_threshold": self.snr_high_threshold,
            "snr_medium_threshold": self.snr_medium_threshold,
        }


@dataclass
class TimelineBounds:
    """
    Timeline bounds for the Genesis Node.
    
    Proves when BIZRA work began and establishes the
    provenance of the 3-year development history.
    """
    
    genesis_epoch: datetime
    current_timestamp: datetime
    
    # Provable bounds from data
    earliest_chat: Optional[datetime] = None
    latest_chat: Optional[datetime] = None
    earliest_commit: Optional[datetime] = None
    latest_commit: Optional[datetime] = None
    
    # Derived metrics
    total_days: int = 0
    active_months: int = 0
    
    @classmethod
    def from_ecosystem_profile(
        cls,
        earliest_artifact: Optional[datetime] = None,
        latest_artifact: Optional[datetime] = None,
    ) -> "TimelineBounds":
        """Create bounds from ecosystem discovery."""
        
        now = datetime.now(timezone.utc)
        
        bounds = cls(
            genesis_epoch=GENESIS_EPOCH,
            current_timestamp=now,
            earliest_chat=earliest_artifact,
            latest_chat=latest_artifact,
        )
        
        # Calculate total days
        if earliest_artifact:
            bounds.total_days = (now - earliest_artifact).days
        else:
            bounds.total_days = (now - GENESIS_EPOCH).days
        
        # Estimate active months (rough)
        bounds.active_months = bounds.total_days // 30
        
        return bounds
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "genesis_epoch": self.genesis_epoch.isoformat(),
            "current_timestamp": self.current_timestamp.isoformat(),
            "earliest_chat": self.earliest_chat.isoformat() if self.earliest_chat else None,
            "latest_chat": self.latest_chat.isoformat() if self.latest_chat else None,
            "earliest_commit": self.earliest_commit.isoformat() if self.earliest_commit else None,
            "latest_commit": self.latest_commit.isoformat() if self.latest_commit else None,
            "total_days": self.total_days,
            "active_months": self.active_months,
        }


# =============================================================================
# NODE ZERO IDENTITY
# =============================================================================


@dataclass
class NodeZeroIdentity:
    """
    Complete identity of the Genesis Node (Node0).
    
    This is the cryptographic anchor for the entire BIZRA ecosystem.
    It binds:
    - Machine (hardware fingerprint)
    - Owner (First Architect attestation)
    - Constitution (genesis rules)
    - Timeline (3-year history)
    - Ecosystem (all artifacts)
    
    Once sealed, this identity is immutable and forms the
    root of trust for all subsequent nodes.
    """
    
    # Node identification
    node_id: str
    node_role: NodeRole
    
    # Components
    machine: MachineFingerprint
    owner: OwnerAttestation
    constitution: ConstitutionBinding
    timeline: TimelineBounds
    
    # Ecosystem binding (from discovery)
    ecosystem_root_hash: Optional[str] = None
    artifact_count: int = 0
    repository_count: int = 0

    # Host-centric manifest binding (optional)
    system_manifest_path: Optional[str] = None
    system_manifest_hash: Optional[str] = None
    
    # Seal metadata
    sealed_at: Optional[datetime] = None
    seal_hash: Optional[str] = None
    seal_signature: Optional[str] = None
    
    @classmethod
    def create_genesis_node(
        cls,
        owner_alias: str = "Momo",
        ecosystem_root_hash: Optional[str] = None,
        artifact_count: int = 0,
        repository_count: int = 0,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
    ) -> Tuple["NodeZeroIdentity", Optional[ed25519.Ed25519PrivateKey]]:
        """
        Create the Genesis Node identity.
        
        This is called ONCE to establish Node0. Subsequent calls
        should load from the sealed identity file.
        
        Returns:
            Tuple of (identity, private_key)
        """
        logger.info("Creating Genesis Node (Node0) identity...")

        # Load host-centric system manifest if present
        manifest_data, manifest_hash, manifest_path = load_system_manifest()
        if manifest_data:
            if ecosystem_root_hash is None:
                ecosystem_root_hash = (
                    manifest_data.get("ecosystem_root_hash")
                    or manifest_data.get("genesis_hash")
                )
            if artifact_count == 0:
                artifact_count = manifest_data.get("artifact_count", artifact_count)
            if repository_count == 0:
                repository_count = manifest_data.get("repository_count", repository_count)
        
        # Generate components
        machine = MachineFingerprint.from_current_machine()
        owner, private_key = OwnerAttestation.create_genesis_owner(
            owner_alias=owner_alias,
            private_key=private_key,
        )
        constitution = ConstitutionBinding.from_constitution_file()
        timeline = TimelineBounds.from_ecosystem_profile()
        
        # Generate node ID from components
        node_id_input = (
            f"{machine.fingerprint}:"
            f"{owner.owner_id}:"
            f"{constitution.constitution_hash[:16]}"
        )
        node_id = f"node0_{hashlib.sha256(node_id_input.encode()).hexdigest()[:12]}"
        
        identity = cls(
            node_id=node_id,
            node_role=NodeRole.GENESIS,
            machine=machine,
            owner=owner,
            constitution=constitution,
            timeline=timeline,
            ecosystem_root_hash=ecosystem_root_hash,
            artifact_count=artifact_count,
            repository_count=repository_count,
            system_manifest_path=str(manifest_path) if manifest_path else None,
            system_manifest_hash=manifest_hash,
        )
        
        logger.info(f"Genesis Node created: {node_id}")
        
        return identity, private_key
    
    def seal(
        self,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
    ) -> None:
        """
        Seal the identity (make it immutable).
        
        Once sealed, the identity hash cannot change without
        invalidating the seal.
        """
        self.sealed_at = datetime.now(timezone.utc)
        
        # Compute seal hash
        seal_data = self.to_dict()
        del seal_data["seal_hash"]
        del seal_data["seal_signature"]
        seal_json = json.dumps(seal_data, sort_keys=True, separators=(",", ":"))
        
        if BLAKE3_AVAILABLE:
            self.seal_hash = blake3.blake3(seal_json.encode()).hexdigest()
        else:
            self.seal_hash = hashlib.sha256(seal_json.encode()).hexdigest()
        
        # Sign seal
        if private_key and CRYPTO_AVAILABLE:
            sig_bytes = private_key.sign(self.seal_hash.encode())
            self.seal_signature = sig_bytes.hex()
        
        logger.info(f"Node0 identity sealed: {self.seal_hash[:16]}...")
    
    def verify_seal(self) -> bool:
        """Verify the identity seal is valid."""
        if not self.seal_hash or not self.sealed_at:
            return False
        
        # Recompute seal hash
        seal_data = self.to_dict()
        stored_hash = seal_data.pop("seal_hash")
        seal_data.pop("seal_signature", None)
        seal_json = json.dumps(seal_data, sort_keys=True, separators=(",", ":"))
        
        if BLAKE3_AVAILABLE:
            computed_hash = blake3.blake3(seal_json.encode()).hexdigest()
        else:
            computed_hash = hashlib.sha256(seal_json.encode()).hexdigest()
        
        return computed_hash == stored_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "node_role": self.node_role.name,
            "machine": self.machine.to_dict(),
            "owner": self.owner.to_dict(),
            "constitution": self.constitution.to_dict(),
            "timeline": self.timeline.to_dict(),
            "ecosystem_root_hash": self.ecosystem_root_hash,
            "artifact_count": self.artifact_count,
            "repository_count": self.repository_count,
            "system_manifest_path": self.system_manifest_path,
            "system_manifest_hash": self.system_manifest_hash,
            "sealed_at": self.sealed_at.isoformat() if self.sealed_at else None,
            "seal_hash": self.seal_hash,
            "seal_signature": self.seal_signature,
        }
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save identity to file."""
        if path is None:
            path = REPO_ROOT / "data" / "genesis" / "NODE_ZERO_IDENTITY.json"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Node0 identity saved to: {path}")
        return path
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NodeZeroIdentity":
        """Load identity from file."""
        if path is None:
            path = REPO_ROOT / "data" / "genesis" / "NODE_ZERO_IDENTITY.json"
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Reconstruct components
        machine = MachineFingerprint(**data["machine"])
        
        owner = OwnerAttestation(
            owner_id=data["owner"]["owner_id"],
            owner_alias=data["owner"]["owner_alias"],
            owner_public_key=data["owner"]["owner_public_key"],
            role=data["owner"]["role"],
            capabilities=data["owner"]["capabilities"],
            attested_at=datetime.fromisoformat(data["owner"]["attested_at"]),
            attestation_hash=data["owner"]["attestation_hash"],
            signature=data["owner"].get("signature"),
        )
        
        constitution = ConstitutionBinding(
            constitution_hash=data["constitution"]["constitution_hash"],
            constitution_version=data["constitution"]["constitution_version"],
            binding_timestamp=datetime.fromisoformat(data["constitution"]["binding_timestamp"]),
            ihsan_threshold=data["constitution"]["ihsan_threshold"],
            snr_high_threshold=data["constitution"]["snr_high_threshold"],
            snr_medium_threshold=data["constitution"]["snr_medium_threshold"],
        )
        
        timeline = TimelineBounds(
            genesis_epoch=datetime.fromisoformat(data["timeline"]["genesis_epoch"]),
            current_timestamp=datetime.fromisoformat(data["timeline"]["current_timestamp"]),
            earliest_chat=datetime.fromisoformat(data["timeline"]["earliest_chat"]) if data["timeline"]["earliest_chat"] else None,
            latest_chat=datetime.fromisoformat(data["timeline"]["latest_chat"]) if data["timeline"]["latest_chat"] else None,
            earliest_commit=datetime.fromisoformat(data["timeline"]["earliest_commit"]) if data["timeline"]["earliest_commit"] else None,
            latest_commit=datetime.fromisoformat(data["timeline"]["latest_commit"]) if data["timeline"]["latest_commit"] else None,
            total_days=data["timeline"]["total_days"],
            active_months=data["timeline"]["active_months"],
        )
        
        identity = cls(
            node_id=data["node_id"],
            node_role=NodeRole[data["node_role"]],
            machine=machine,
            owner=owner,
            constitution=constitution,
            timeline=timeline,
            ecosystem_root_hash=data.get("ecosystem_root_hash"),
            artifact_count=data.get("artifact_count", 0),
            repository_count=data.get("repository_count", 0),
            system_manifest_path=data.get("system_manifest_path"),
            system_manifest_hash=data.get("system_manifest_hash"),
            sealed_at=datetime.fromisoformat(data["sealed_at"]) if data.get("sealed_at") else None,
            seal_hash=data.get("seal_hash"),
            seal_signature=data.get("seal_signature"),
        )
        
        return identity


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main() -> int:
    """Main entry point for Node0 identity creation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create or verify the Genesis Node (Node0) identity"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a new Node0 identity (first run only)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing Node0 identity seal",
    )
    parser.add_argument(
        "--owner",
        default="Momo",
        help="Owner alias for genesis attestation",
    )
    parser.add_argument(
        "--ecosystem-hash",
        help="Ecosystem root hash from discovery",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for identity file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    if args.create:
        # Create new identity
        identity, private_key = NodeZeroIdentity.create_genesis_node(
            owner_alias=args.owner,
            ecosystem_root_hash=args.ecosystem_hash,
        )
        
        # Seal the identity
        identity.seal(private_key)
        
        # Save
        output_path = identity.save(args.output)
        
        # Save private key separately (SENSITIVE)
        if private_key and CRYPTO_AVAILABLE:
            key_path = output_path.parent / "NODE_ZERO_PRIVATE_KEY.pem"
            key_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            key_path.write_bytes(key_bytes)
            logger.warning(f"Private key saved to: {key_path} (KEEP THIS SECURE)")
        
        # Print summary
        print("\n" + "=" * 70)
        print("GENESIS NODE (NODE0) IDENTITY CREATED")
        print("=" * 70)
        print(f"Node ID:         {identity.node_id}")
        print(f"Owner:           {identity.owner.owner_alias}")
        print(f"Role:            {identity.node_role.name}")
        print(f"Constitution:    {identity.constitution.constitution_hash[:16]}...")
        print(f"Machine:         {identity.machine.fingerprint[:16]}...")
        print(f"Seal Hash:       {identity.seal_hash[:16]}...")
        print(f"Sealed At:       {identity.sealed_at}")
        print("=" * 70)
        print(f"\nIdentity saved to: {output_path}")
        
    elif args.verify:
        # Load and verify
        identity = NodeZeroIdentity.load(args.output)
        
        if identity.verify_seal():
            print("\n✅ Node0 identity seal is VALID")
            print(f"   Node ID: {identity.node_id}")
            print(f"   Sealed:  {identity.sealed_at}")
            return 0
        else:
            print("\n❌ Node0 identity seal is INVALID")
            return 1
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
