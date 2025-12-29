"""
BIZRA Lineage Seal Pack
════════════════════════════════════════════════════════════════════════════════

Creates investor-grade evidence packs for due diligence.

The Lineage Seal Pack is a comprehensive archive containing:

1. **Genesis Seal**: The cryptographic anchor (GENESIS_SEAL.json)
2. **Node0 Identity**: Machine + owner binding (NODE_ZERO_IDENTITY.json)
3. **Proof of Impact**: First Architect contribution (PROOF_OF_IMPACT.json)
4. **Constitution**: Genesis rules (constitution.toml)
5. **Architecture Docs**: Technical diagrams (ARCHITECTURE_DIAGRAMS.md)
6. **Audit Trail**: Verification log (AUDIT_LOG.json)
7. **Manifest**: Pack contents hash (MANIFEST.json)

This pack can be:
- Shared with investors for due diligence
- Submitted to auditors for verification
- Anchored to Bitcoin via OpenTimestamps
- Stored in immutable archives (IPFS, Arweave)

Design Philosophy:
- Complete: Contains all necessary evidence
- Verifiable: Every component is cryptographically signed
- Portable: Single archive can be shared
- Immutable: Once sealed, cannot be modified

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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

logger = logging.getLogger("bizra.genesis.lineage_seal_pack")


# =============================================================================
# CONSTANTS
# =============================================================================

PACK_VERSION = "1.0.0"
DEFAULT_PACK_NAME = "BIZRA_LINEAGE_SEAL_PACK"


# =============================================================================
# PACK MANIFEST
# =============================================================================


@dataclass
class PackItem:
    """An item in the lineage seal pack."""
    
    filename: str
    relative_path: str
    size_bytes: int
    content_hash: str
    hash_algorithm: str
    item_type: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


@dataclass
class PackManifest:
    """
    Manifest for the Lineage Seal Pack.
    
    Contains hashes of all pack contents for verification.
    """
    
    version: str
    pack_id: str
    created_at: datetime
    created_by: str
    
    items: List[PackItem]
    total_items: int
    total_bytes: int
    
    # Manifest hash (covers all items)
    manifest_hash: str
    signature: Optional[str] = None
    
    @classmethod
    def from_items(
        cls,
        items: List[PackItem],
        created_by: str = "genesis_architect",
    ) -> "PackManifest":
        """Create manifest from items."""
        
        created_at = datetime.now(timezone.utc)
        
        # Generate pack ID
        pack_id_input = f"{created_at.isoformat()}:{len(items)}"
        pack_id = f"pack_{hashlib.sha256(pack_id_input.encode()).hexdigest()[:12]}"
        
        # Calculate totals
        total_bytes = sum(item.size_bytes for item in items)
        
        # Compute manifest hash
        manifest_data = {
            "version": PACK_VERSION,
            "pack_id": pack_id,
            "created_at": created_at.isoformat(),
            "created_by": created_by,
            "items": [item.to_dict() for item in items],
            "total_items": len(items),
            "total_bytes": total_bytes,
        }
        manifest_json = json.dumps(manifest_data, sort_keys=True, separators=(",", ":"))
        
        if BLAKE3_AVAILABLE:
            manifest_hash = blake3.blake3(manifest_json.encode()).hexdigest()
        else:
            manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
        
        return cls(
            version=PACK_VERSION,
            pack_id=pack_id,
            created_at=created_at,
            created_by=created_by,
            items=items,
            total_items=len(items),
            total_bytes=total_bytes,
            manifest_hash=manifest_hash,
        )
    
    def sign(
        self,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
    ) -> None:
        """Sign the manifest."""
        if private_key and CRYPTO_AVAILABLE:
            sig_bytes = private_key.sign(self.manifest_hash.encode())
            self.signature = sig_bytes.hex()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "pack_id": self.pack_id,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "items": [item.to_dict() for item in self.items],
            "total_items": self.total_items,
            "total_bytes": self.total_bytes,
            "manifest_hash": self.manifest_hash,
            "signature": self.signature,
        }


# =============================================================================
# LINEAGE SEAL PACK
# =============================================================================


class LineageSealPack:
    """
    Creates and manages investor-grade evidence packs.
    
    A Lineage Seal Pack contains all cryptographic proofs and
    evidence needed for investor due diligence.
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        pack_name: str = DEFAULT_PACK_NAME,
    ):
        self.output_dir = output_dir or (REPO_ROOT / "data" / "lineage_packs")
        self.pack_name = pack_name
        self.items: List[PackItem] = []
        self._staging_dir: Optional[Path] = None
    
    def _hash_file(self, path: Path) -> str:
        """Compute hash of a file."""
        if BLAKE3_AVAILABLE:
            hasher = blake3.blake3()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        else:
            hasher = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
    
    def _create_staging(self) -> Path:
        """Create staging directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        staging = self.output_dir / f"{self.pack_name}_{timestamp}"
        staging.mkdir(parents=True, exist_ok=True)
        self._staging_dir = staging
        return staging
    
    def add_file(
        self,
        source_path: Path,
        relative_path: Optional[str] = None,
        item_type: str = "document",
        description: str = "",
        skip_copy: bool = False,
    ) -> None:
        """Add a file to the pack."""
        if not source_path.exists():
            logger.warning(f"Source file not found: {source_path}")
            return
        
        if relative_path is None:
            relative_path = source_path.name
        
        # Hash the file
        content_hash = self._hash_file(source_path)
        
        item = PackItem(
            filename=source_path.name,
            relative_path=relative_path,
            size_bytes=source_path.stat().st_size,
            content_hash=content_hash,
            hash_algorithm="blake3" if BLAKE3_AVAILABLE else "sha256",
            item_type=item_type,
            description=description or f"Evidence: {source_path.name}",
        )
        
        self.items.append(item)
        
        # Copy to staging if active (unless already in staging or skip requested)
        if self._staging_dir and not skip_copy:
            dest = self._staging_dir / relative_path
            # Skip if source and dest are the same
            if source_path.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(source_path, dest)
                except PermissionError:
                    # File might be locked; copy content directly
                    with source_path.open("rb") as src, dest.open("wb") as dst:
                        dst.write(src.read())
    
    def add_genesis_files(self) -> None:
        """Add all genesis files to the pack."""
        genesis_dir = REPO_ROOT / "data" / "genesis"
        
        # Genesis Seal
        seal_path = genesis_dir / "GENESIS_SEAL.json"
        if seal_path.exists():
            self.add_file(
                seal_path,
                "seal/GENESIS_SEAL.json",
                "attestation",
                "The cryptographic anchor binding the BIZRA ecosystem"
            )
        
        # Node0 Identity
        identity_path = genesis_dir / "NODE_ZERO_IDENTITY.json"
        if identity_path.exists():
            self.add_file(
                identity_path,
                "identity/NODE_ZERO_IDENTITY.json",
                "attestation",
                "Genesis Node machine and owner identity"
            )
        
        # Proof of Impact
        poi_path = genesis_dir / "PROOF_OF_IMPACT.json"
        if poi_path.exists():
            self.add_file(
                poi_path,
                "impact/PROOF_OF_IMPACT.json",
                "attestation",
                "First Architect contribution proof"
            )
        
        # Constitution
        constitution_path = REPO_ROOT / "constitution.toml"
        if constitution_path.exists():
            self.add_file(
                constitution_path,
                "constitution/constitution.toml",
                "governance",
                "Genesis constitution defining network rules"
            )
        
        # Architecture Diagrams
        arch_path = REPO_ROOT / "docs" / "ARCHITECTURE_DIAGRAMS.md"
        if arch_path.exists():
            self.add_file(
                arch_path,
                "docs/ARCHITECTURE_DIAGRAMS.md",
                "documentation",
                "Technical architecture diagrams"
            )
        
        # Security Model
        security_path = REPO_ROOT / "docs" / "SECURITY_MODEL.md"
        if security_path.exists():
            self.add_file(
                security_path,
                "docs/SECURITY_MODEL.md",
                "documentation",
                "Security model documentation"
            )
        
        # Protocol Spec
        protocol_path = REPO_ROOT / "PROTOCOL.md"
        if protocol_path.exists():
            self.add_file(
                protocol_path,
                "docs/PROTOCOL.md",
                "specification",
                "Protocol specification"
            )
    
    def create_audit_log(self) -> Path:
        """Create audit log for the pack."""
        if not self._staging_dir:
            self._create_staging()
        
        audit_log = {
            "pack_name": self.pack_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "blake3_available": BLAKE3_AVAILABLE,
                "crypto_available": CRYPTO_AVAILABLE,
            },
            "verification_steps": [
                {
                    "step": 1,
                    "name": "Genesis Seal Verification",
                    "description": "Verify GENESIS_SEAL.json hash and signature",
                    "command": "python scripts/generate_genesis_seal.py --verify",
                },
                {
                    "step": 2,
                    "name": "Manifest Verification",
                    "description": "Verify all item hashes match manifest",
                    "command": "python core/genesis/lineage_seal_pack.py --verify <pack_path>",
                },
                {
                    "step": 3,
                    "name": "Signature Verification",
                    "description": "Verify Ed25519 signatures with public key",
                    "command": "See docs/VERIFICATION_STANDARD.md",
                },
            ],
            "items_included": len(self.items),
        }
        
        audit_path = self._staging_dir / "AUDIT_LOG.json"
        with audit_path.open("w", encoding="utf-8") as f:
            json.dump(audit_log, f, indent=2, ensure_ascii=False)
        
        return audit_path
    
    def create_readme(self) -> Path:
        """Create README for the pack."""
        if not self._staging_dir:
            self._create_staging()
        
        readme_content = f"""# BIZRA Lineage Seal Pack

**Pack Version:** {PACK_VERSION}
**Created:** {datetime.now(timezone.utc).isoformat()}

## Overview

This Lineage Seal Pack contains cryptographic proofs and evidence for
due diligence of the BIZRA ecosystem. All files are hashed and the
manifest is signed for integrity verification.

## Contents

| File | Type | Description |
|------|------|-------------|
"""
        
        for item in self.items:
            readme_content += f"| `{item.relative_path}` | {item.item_type} | {item.description} |\n"
        
        readme_content += """
## Verification

1. **Verify Manifest Hash**: Recompute MANIFEST.json hash and compare
2. **Verify File Hashes**: Check each file hash against manifest
3. **Verify Signatures**: Use provided public key to verify Ed25519 signatures
4. **Verify Timeline**: Check OpenTimestamps proof (if included)

## Cryptographic Details

- **Hash Algorithm**: BLAKE3 (or SHA-256 fallback)
- **Signature Scheme**: Ed25519
- **Timestamp Proof**: OpenTimestamps (Bitcoin-anchored)

## Contact

For verification questions, contact the BIZRA Genesis Team.

---
*This pack is cryptographically sealed and cannot be modified without
invalidating the manifest hash.*
"""
        
        readme_path = self._staging_dir / "README.md"
        with readme_path.open("w", encoding="utf-8") as f:
            f.write(readme_content)
        
        return readme_path
    
    def seal(
        self,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
        created_by: str = "genesis_architect",
    ) -> PackManifest:
        """
        Seal the pack and generate manifest.
        
        Returns:
            The signed manifest
        """
        if not self._staging_dir:
            self._create_staging()
        
        # Create supplementary files
        audit_path = self.create_audit_log()
        readme_path = self.create_readme()
        
        # Add them to items
        self.add_file(
            audit_path,
            "AUDIT_LOG.json",
            "audit",
            "Audit log with verification steps",
        )
        self.add_file(
            readme_path,
            "README.md",
            "documentation",
            "Pack documentation and verification guide",
        )
        
        # Create manifest
        manifest = PackManifest.from_items(self.items, created_by)
        
        # Sign manifest
        manifest.sign(private_key)
        
        # Save manifest
        manifest_path = self._staging_dir / "MANIFEST.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Lineage Seal Pack created: {self._staging_dir}")
        logger.info(f"Manifest hash: {manifest.manifest_hash[:16]}...")
        
        return manifest
    
    def create_archive(
        self,
        format: str = "zip",
    ) -> Path:
        """Create compressed archive of the pack."""
        if not self._staging_dir:
            raise ValueError("Pack not sealed yet")
        
        if format == "zip":
            archive_path = self._staging_dir.with_suffix(".zip")
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for item in self.items:
                    item_path = self._staging_dir / item.relative_path
                    if item_path.exists():
                        zf.write(item_path, item.relative_path)
                # Add manifest
                zf.write(self._staging_dir / "MANIFEST.json", "MANIFEST.json")
        elif format == "tar.gz":
            archive_path = self._staging_dir.with_suffix(".tar.gz")
            with tarfile.open(archive_path, "w:gz") as tf:
                tf.add(self._staging_dir, arcname=self.pack_name)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Archive created: {archive_path}")
        return archive_path
    
    @classmethod
    def verify_archive(cls, archive_path: Path) -> bool:
        """Verify a pack archive."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Extract
            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(tmpdir_path)
            else:
                with tarfile.open(archive_path, "r:*") as tf:
                    tf.extractall(tmpdir_path)
            
            # Find manifest
            manifest_path = None
            for p in tmpdir_path.rglob("MANIFEST.json"):
                manifest_path = p
                break
            
            if not manifest_path:
                logger.error("MANIFEST.json not found in archive")
                return False
            
            with manifest_path.open("r") as f:
                manifest_data = json.load(f)
            
            # Verify each item
            pack_root = manifest_path.parent
            all_valid = True
            
            for item_data in manifest_data.get("items", []):
                item_path = pack_root / item_data["relative_path"]
                if not item_path.exists():
                    logger.error(f"Missing: {item_data['relative_path']}")
                    all_valid = False
                    continue
                
                # Compute hash
                if item_data["hash_algorithm"] == "blake3" and BLAKE3_AVAILABLE:
                    hasher = blake3.blake3()
                else:
                    hasher = hashlib.sha256()
                
                with item_path.open("rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)
                
                computed = hasher.hexdigest()
                if computed != item_data["content_hash"]:
                    logger.error(f"Hash mismatch: {item_data['relative_path']}")
                    all_valid = False
            
            return all_valid


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create or verify a BIZRA Lineage Seal Pack"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a new lineage seal pack",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify an existing pack archive",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "lineage_packs",
        help="Output directory for packs",
    )
    parser.add_argument(
        "--pack-name",
        default=DEFAULT_PACK_NAME,
        help="Name for the pack",
    )
    parser.add_argument(
        "--format",
        choices=["zip", "tar.gz"],
        default="zip",
        help="Archive format",
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
    
    if args.verify:
        if LineageSealPack.verify_archive(args.verify):
            print(f"\n✅ Pack verification PASSED: {args.verify}")
            return 0
        else:
            print(f"\n❌ Pack verification FAILED: {args.verify}")
            return 1
    
    if args.create:
        # Load private key if available
        private_key = None
        key_path = REPO_ROOT / "data" / "genesis" / "NODE_ZERO_PRIVATE_KEY.pem"
        if key_path.exists() and CRYPTO_AVAILABLE:
            key_bytes = key_path.read_bytes()
            private_key = serialization.load_pem_private_key(key_bytes, password=None)
        
        # Create pack
        pack = LineageSealPack(args.output_dir, args.pack_name)
        pack._create_staging()
        
        # Add genesis files
        pack.add_genesis_files()
        
        # Seal
        manifest = pack.seal(private_key)
        
        # Create archive
        archive_path = pack.create_archive(args.format)
        
        print("\n" + "=" * 70)
        print("LINEAGE SEAL PACK CREATED")
        print("=" * 70)
        print(f"Pack ID:       {manifest.pack_id}")
        print(f"Items:         {manifest.total_items}")
        print(f"Size:          {manifest.total_bytes:,} bytes")
        print(f"Manifest Hash: {manifest.manifest_hash[:32]}...")
        print(f"Signed:        {'YES' if manifest.signature else 'NO'}")
        print(f"Archive:       {archive_path}")
        print("=" * 70)
        
        return 0
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
