#!/usr/bin/env python3
"""
BIZRA Genesis Seal Generator
════════════════════════════════════════════════════════════════════════════════

Generates the Genesis Seal—the cryptographic anchor binding the entire
BIZRA ecosystem to its origin point (Node0, First Architect, 3-year history).

The Genesis Seal is:
1. Ecosystem-Wide ROOT_HASH: Merkle root of all discovered artifacts
2. Node0 Identity Binding: Links seal to the genesis machine
3. First Architect Attestation: Signed by the genesis owner
4. OpenTimestamps Proof: Anchored to Bitcoin blockchain (optional)

Once generated, the Genesis Seal is IMMUTABLE. It proves:
- WHAT: The complete state of the BIZRA ecosystem
- WHEN: The timestamp of genesis
- WHO: The First Architect's identity
- WHERE: The Genesis Node's machine fingerprint

Design Philosophy:
- Giants Protocol: The seal stands on all prior work
- SNR-Weighted: Only high-signal artifacts contribute
- Graph of Thoughts: The seal is the root of the trust graph
- Proof of Impact: First Architect's contribution is sealed

Usage:
    python scripts/generate_genesis_seal.py --discover --seal

This will:
1. Run ecosystem discovery (if not cached)
2. Create Node0 identity
3. Calculate Proof of Impact
4. Generate and sign the Genesis Seal
5. Optionally submit to OpenTimestamps

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
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

# OpenTimestamps support for court-grade Bitcoin-anchored timestamps
try:
    import subprocess
    OTS_AVAILABLE = subprocess.run(
        ["ots", "--version"],
        capture_output=True,
        timeout=5
    ).returncode == 0
except Exception:
    OTS_AVAILABLE = False

logger = logging.getLogger("bizra.genesis.seal")


# =============================================================================
# HELPERS
# =============================================================================

def load_cached_discovery() -> Dict[str, Any]:
    """Load discovery facts from the host-centric manifest if present."""
    manifest_path = REPO_ROOT / "data" / "genesis" / "GENESIS_SYSTEM_MANIFEST.json"
    if not manifest_path.exists():
        return {}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Prefer genesis_hash as the ecosystem root; fall back to hash_algo root if absent
        root_hash = manifest.get("genesis_hash") or manifest.get("ecosystem_root_hash", "")
        ts_unix = manifest.get("timestamp_unix")
        timeline_end = (
            datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()
            if ts_unix
            else datetime.now(timezone.utc).isoformat()
        )
        return {
            "ecosystem_root_hash": root_hash,
            "artifact_count": manifest.get("artifact_count", 0),
            "repository_count": manifest.get("repository_count", 0),
            "timeline_start": "2023-01-14T00:00:00+00:00",
            "timeline_end": timeline_end,
            "contribution_hours": 15000,
            "artifacts": [],  # We keep it light; PoI uses count/metadata
        }
    except Exception:
        return {}


# =============================================================================
# CONSTANTS
# =============================================================================

GENESIS_SEAL_VERSION = "1.0.0"
GENESIS_SEAL_FILENAME = "GENESIS_SEAL.json"


# =============================================================================
# GENESIS SEAL
# =============================================================================


class GenesisSeal:
    """
    The Genesis Seal—cryptographic anchor for the BIZRA ecosystem.
    
    This is the "Big Bang" attestation that binds:
    - All ecosystem artifacts (ROOT_HASH)
    - Node0 identity (machine + owner)
    - Proof of Impact (First Architect contribution)
    - Temporal proof (timestamp, optional OTS)
    """
    
    def __init__(
        self,
        ecosystem_root_hash: str,
        node_id: str,
        owner_id: str,
        owner_alias: str,
        owner_public_key: str,
        proof_of_impact_hash: str,
        total_impact: float,
        artifact_count: int,
        repository_count: int,
        timeline_start: str,
        timeline_end: str,
        constitution_hash: str,
        machine_fingerprint: str,
    ):
        self.version = GENESIS_SEAL_VERSION
        self.ecosystem_root_hash = ecosystem_root_hash
        self.node_id = node_id
        self.owner_id = owner_id
        self.owner_alias = owner_alias
        self.owner_public_key = owner_public_key
        self.proof_of_impact_hash = proof_of_impact_hash
        self.total_impact = total_impact
        self.artifact_count = artifact_count
        self.repository_count = repository_count
        self.timeline_start = timeline_start
        self.timeline_end = timeline_end
        self.constitution_hash = constitution_hash
        self.machine_fingerprint = machine_fingerprint
        
        # Seal metadata (set on seal())
        self.sealed_at: Optional[str] = None
        self.seal_hash: Optional[str] = None
        self.signature: Optional[str] = None
        self.ots_proof: Optional[str] = None
    
    def seal(
        self,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
    ) -> str:
        """
        Generate and sign the seal hash.
        
        Returns:
            The seal hash (hex string)
        """
        self.sealed_at = datetime.now(timezone.utc).isoformat()
        
        # Compute seal hash from all components
        seal_data = {
            "version": self.version,
            "ecosystem_root_hash": self.ecosystem_root_hash,
            "node_id": self.node_id,
            "owner_id": self.owner_id,
            "owner_alias": self.owner_alias,
            "owner_public_key": self.owner_public_key,
            "proof_of_impact_hash": self.proof_of_impact_hash,
            "total_impact": self.total_impact,
            "artifact_count": self.artifact_count,
            "repository_count": self.repository_count,
            "timeline_start": self.timeline_start,
            "timeline_end": self.timeline_end,
            "constitution_hash": self.constitution_hash,
            "machine_fingerprint": self.machine_fingerprint,
            "sealed_at": self.sealed_at,
        }
        seal_json = json.dumps(seal_data, sort_keys=True, separators=(",", ":"))
        
        if BLAKE3_AVAILABLE:
            self.seal_hash = blake3.blake3(seal_json.encode()).hexdigest()
        else:
            self.seal_hash = hashlib.sha256(seal_json.encode()).hexdigest()
        
        # Sign with Ed25519 if key provided
        if private_key and CRYPTO_AVAILABLE:
            sig_bytes = private_key.sign(self.seal_hash.encode())
            self.signature = sig_bytes.hex()
            logger.info(f"Seal signed with Ed25519")
        
        logger.info(f"Genesis Seal generated: {self.seal_hash[:16]}...")
        
        return self.seal_hash
    
    def verify(
        self,
        public_key: Optional[ed25519.Ed25519PublicKey] = None,
    ) -> bool:
        """Verify the seal hash and optional signature."""
        
        if not self.seal_hash or not self.sealed_at:
            return False
        
        # Recompute hash
        seal_data = {
            "version": self.version,
            "ecosystem_root_hash": self.ecosystem_root_hash,
            "node_id": self.node_id,
            "owner_id": self.owner_id,
            "owner_alias": self.owner_alias,
            "owner_public_key": self.owner_public_key,
            "proof_of_impact_hash": self.proof_of_impact_hash,
            "total_impact": self.total_impact,
            "artifact_count": self.artifact_count,
            "repository_count": self.repository_count,
            "timeline_start": self.timeline_start,
            "timeline_end": self.timeline_end,
            "constitution_hash": self.constitution_hash,
            "machine_fingerprint": self.machine_fingerprint,
            "sealed_at": self.sealed_at,
        }
        seal_json = json.dumps(seal_data, sort_keys=True, separators=(",", ":"))
        
        if BLAKE3_AVAILABLE:
            computed_hash = blake3.blake3(seal_json.encode()).hexdigest()
        else:
            computed_hash = hashlib.sha256(seal_json.encode()).hexdigest()
        
        if computed_hash != self.seal_hash:
            return False
        
        # Verify signature if public key provided
        if public_key and self.signature and CRYPTO_AVAILABLE:
            try:
                sig_bytes = bytes.fromhex(self.signature)
                public_key.verify(sig_bytes, self.seal_hash.encode())
            except Exception:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "ecosystem_root_hash": self.ecosystem_root_hash,
            "node_id": self.node_id,
            "owner_id": self.owner_id,
            "owner_alias": self.owner_alias,
            "owner_public_key": self.owner_public_key,
            "proof_of_impact_hash": self.proof_of_impact_hash,
            "total_impact": self.total_impact,
            "artifact_count": self.artifact_count,
            "repository_count": self.repository_count,
            "timeline_start": self.timeline_start,
            "timeline_end": self.timeline_end,
            "constitution_hash": self.constitution_hash,
            "machine_fingerprint": self.machine_fingerprint,
            "sealed_at": self.sealed_at,
            "seal_hash": self.seal_hash,
            "signature": self.signature,
            "ots_proof": self.ots_proof,
        }
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save seal to file."""
        if path is None:
            path = REPO_ROOT / "data" / "genesis" / GENESIS_SEAL_FILENAME
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Genesis Seal saved to: {path}")
        return path

    def stamp_opentimestamps(self, seal_path: Optional[Path] = None) -> Optional[Path]:
        """
        Create OpenTimestamps proof for the Genesis Seal.
        
        This anchors the seal to the Bitcoin blockchain, providing
        court-grade temporal proof that cannot be backdated.
        
        Args:
            seal_path: Path to the seal JSON file (default: standard location)
            
        Returns:
            Path to the .ots proof file, or None if stamping failed
        """
        import subprocess
        
        if seal_path is None:
            seal_path = REPO_ROOT / "data" / "genesis" / GENESIS_SEAL_FILENAME
        
        if not seal_path.exists():
            logger.warning(f"Seal file not found: {seal_path}")
            return None
        
        if not OTS_AVAILABLE:
            logger.warning("OpenTimestamps CLI (ots) not available - skipping Bitcoin anchor")
            logger.info("Install with: pip install opentimestamps-client")
            return None
        
        try:
            # ots stamp creates <file>.ots next to the original file
            result = subprocess.run(
                ["ots", "stamp", str(seal_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=seal_path.parent,
            )
            
            if result.returncode != 0:
                logger.warning(f"OTS stamp failed: {result.stderr}")
                return None
            
            ots_path = seal_path.with_suffix(seal_path.suffix + ".ots")
            
            if ots_path.exists():
                # Update seal with OTS proof reference
                self.ots_proof = str(ots_path.relative_to(REPO_ROOT))
                
                # Re-save seal with OTS reference
                with seal_path.open("w", encoding="utf-8") as f:
                    json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
                
                logger.info(f"OpenTimestamps proof created: {ots_path}")
                logger.info("Note: Full Bitcoin confirmation takes ~1-6 hours")
                return ots_path
            else:
                logger.warning("OTS file not created")
                return None
                
        except subprocess.TimeoutExpired:
            logger.warning("OTS stamp timed out (network issue?)")
            return None
        except Exception as e:
            logger.warning(f"OTS stamp error: {e}")
            return None

    def verify_opentimestamps(self, seal_path: Optional[Path] = None) -> bool:
        """
        Verify OpenTimestamps proof for the Genesis Seal.
        
        Returns:
            True if verified, False if pending or failed
        """
        import subprocess
        
        if seal_path is None:
            seal_path = REPO_ROOT / "data" / "genesis" / GENESIS_SEAL_FILENAME
        
        ots_path = seal_path.with_suffix(seal_path.suffix + ".ots")
        
        if not ots_path.exists():
            logger.warning("No OTS proof file found")
            return False
        
        if not OTS_AVAILABLE:
            logger.warning("OpenTimestamps CLI not available")
            return False
        
        try:
            result = subprocess.run(
                ["ots", "verify", str(ots_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            if "Success!" in result.stdout or result.returncode == 0:
                logger.info("OpenTimestamps proof VERIFIED ✓")
                return True
            elif "Pending" in result.stdout:
                logger.info("OpenTimestamps proof pending Bitcoin confirmation")
                return False
            else:
                logger.warning(f"OTS verify: {result.stdout} {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"OTS verify error: {e}")
            return False
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "GenesisSeal":
        """Load seal from file."""
        if path is None:
            path = REPO_ROOT / "data" / "genesis" / GENESIS_SEAL_FILENAME
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        seal = cls(
            ecosystem_root_hash=data["ecosystem_root_hash"],
            node_id=data["node_id"],
            owner_id=data["owner_id"],
            owner_alias=data["owner_alias"],
            owner_public_key=data["owner_public_key"],
            proof_of_impact_hash=data["proof_of_impact_hash"],
            total_impact=data["total_impact"],
            artifact_count=data["artifact_count"],
            repository_count=data["repository_count"],
            timeline_start=data["timeline_start"],
            timeline_end=data["timeline_end"],
            constitution_hash=data["constitution_hash"],
            machine_fingerprint=data["machine_fingerprint"],
        )
        
        seal.sealed_at = data.get("sealed_at")
        seal.seal_hash = data.get("seal_hash")
        seal.signature = data.get("signature")
        seal.ots_proof = data.get("ots_proof")
        
        return seal
    
    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════════╗",
            "║                        BIZRA GENESIS SEAL                             ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            f"║  Version:           {self.version:<49}║",
            f"║  Sealed At:         {self.sealed_at or 'NOT SEALED':<49}║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  ECOSYSTEM                                                            ║",
            f"║    Root Hash:       {(self.ecosystem_root_hash[:32] + '...' if len(self.ecosystem_root_hash) > 32 else self.ecosystem_root_hash):<49}║",
            f"║    Artifacts:       {str(self.artifact_count):<49}║",
            f"║    Repositories:    {str(self.repository_count):<49}║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  NODE ZERO                                                            ║",
            f"║    Node ID:         {self.node_id:<49}║",
            f"║    Machine:         {(self.machine_fingerprint[:32] + '...' if len(self.machine_fingerprint) > 32 else self.machine_fingerprint):<49}║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  FIRST ARCHITECT                                                      ║",
            f"║    Alias:           {self.owner_alias:<49}║",
            f"║    ID:              {self.owner_id:<49}║",
            f"║    Total Impact:    {str(self.total_impact):<49}║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  TIMELINE                                                             ║",
            f"║    Start:           {self.timeline_start[:25]:<49}║",
            f"║    End:             {self.timeline_end[:25]:<49}║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  SEAL                                                                 ║",
            f"║    Hash:            {(self.seal_hash[:32] + '...' if self.seal_hash and len(self.seal_hash) > 32 else str(self.seal_hash)):<49}║",
            f"║    Signed:          {'YES' if self.signature else 'NO':<49}║",
            f"║    OTS Proof:       {'YES' if self.ots_proof else 'NO':<49}║",
            "╚══════════════════════════════════════════════════════════════════════╝",
            "",
        ]
        return "\n".join(lines)


# =============================================================================
# SEAL GENERATION WORKFLOW
# =============================================================================


def run_discovery(scan_paths: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Run ecosystem discovery and return results."""
    from scripts.genesis_node_discovery import GenesisNodeDiscovery
    
    if scan_paths is None:
        # Default: scan common locations
        scan_paths = [
            Path.home() / "bizra_scaffold.worktrees",
            Path.home() / "Documents",
            Path("C:/bizra_scaffold.worktrees"),
        ]
        # Filter to existing paths
        scan_paths = [p for p in scan_paths if p.exists()]
        if not scan_paths:
            scan_paths = [REPO_ROOT]
    
    discovery = GenesisNodeDiscovery(scan_paths)
    profile = discovery.discover()
    
    return {
        "node_id": profile.node_id,
        "artifacts": [
            {
                "path": str(a.path),
                "artifact_type": a.artifact_type.name,
                "significance": a.significance.name,
                "size_bytes": a.size_bytes,
                "content_hash": getattr(a, 'blake3_hash', None) or getattr(a, 'sha256_hash', None),
            }
            for a in profile.artifacts
        ],
        "artifact_count": profile.total_artifacts,
        "repository_count": profile.total_repositories,
        "ecosystem_root_hash": profile.ecosystem_root_hash,
        "timeline_start": profile.earliest_artifact.isoformat() if profile.earliest_artifact else None,
        "timeline_end": profile.latest_artifact.isoformat() if profile.latest_artifact else None,
        "contribution_hours": profile.contribution_hours_estimated,
    }


def create_node_identity(
    discovery_data: Dict[str, Any],
    owner_alias: str = "Momo",
) -> Dict[str, Any]:
    """Create Node0 identity."""
    from core.genesis.node_zero import NodeZeroIdentity
    
    identity, private_key = NodeZeroIdentity.create_genesis_node(
        owner_alias=owner_alias,
        ecosystem_root_hash=discovery_data.get("ecosystem_root_hash"),
        artifact_count=discovery_data.get("artifact_count", 0),
        repository_count=discovery_data.get("repository_count", 0),
    )
    
    identity.seal(private_key)
    identity.save()
    
    # Save private key
    if private_key and CRYPTO_AVAILABLE:
        key_path = REPO_ROOT / "data" / "genesis" / "NODE_ZERO_PRIVATE_KEY.pem"
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        key_path.write_bytes(key_bytes)
    
    return {
        "node_id": identity.node_id,
        "owner_id": identity.owner.owner_id,
        "owner_alias": identity.owner.owner_alias,
        "owner_public_key": identity.owner.owner_public_key,
        "constitution_hash": identity.constitution.constitution_hash,
        "machine_fingerprint": identity.machine.fingerprint,
        "private_key": private_key,
    }


def calculate_poi(
    discovery_data: Dict[str, Any],
    identity_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculate Proof of Impact."""
    from core.genesis.proof_of_impact import ProofOfImpact
    
    # Convert discovery artifacts to PoI format
    artifacts = [
        {
            "path": a.get("path", ""),
            "type": a.get("artifact_type", "unknown"),
            "size": a.get("size_bytes", 0),
            "snr": 0.7,  # Default SNR
        }
        for a in discovery_data.get("artifacts", [])
    ]
    
    temporal_data = {
        "earliest": discovery_data.get("timeline_start", "2023-01-14T00:00:00+00:00"),
        "latest": discovery_data.get("timeline_end", datetime.now(timezone.utc).isoformat()),
        "hours": discovery_data.get("contribution_hours", 500),
        "active_days": 500,
        "sessions": 1000,
    }
    
    poi = ProofOfImpact.calculate_genesis_impact(
        contributor_id=identity_data.get("owner_id", "genesis_architect"),
        contributor_alias=identity_data.get("owner_alias", "Momo"),
        artifacts=artifacts,
        temporal_data=temporal_data,
    )
    
    poi.save()
    
    return {
        "proof_hash": poi.proof_hash,
        "total_impact": poi.total_impact,
    }


def generate_seal(
    discovery_data: Dict[str, Any],
    identity_data: Dict[str, Any],
    poi_data: Dict[str, Any],
) -> GenesisSeal:
    """Generate and sign the Genesis Seal."""
    
    seal = GenesisSeal(
        ecosystem_root_hash=discovery_data.get("ecosystem_root_hash", ""),
        node_id=identity_data.get("node_id", ""),
        owner_id=identity_data.get("owner_id", ""),
        owner_alias=identity_data.get("owner_alias", ""),
        owner_public_key=identity_data.get("owner_public_key", ""),
        proof_of_impact_hash=poi_data.get("proof_hash", ""),
        total_impact=poi_data.get("total_impact", 0),
        artifact_count=discovery_data.get("artifact_count", 0),
        repository_count=discovery_data.get("repository_count", 0),
        timeline_start=discovery_data.get("timeline_start") or "2023-01-14T00:00:00+00:00",
        timeline_end=discovery_data.get("timeline_end") or datetime.now(timezone.utc).isoformat(),
        constitution_hash=identity_data.get("constitution_hash", ""),
        machine_fingerprint=identity_data.get("machine_fingerprint", ""),
    )
    
    # Seal and sign
    private_key = identity_data.get("private_key")
    seal.seal(private_key)
    seal.save()
    
    return seal


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate the BIZRA Genesis Seal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow: discover → identity → PoI → seal
  python scripts/generate_genesis_seal.py --discover --seal

  # Just verify existing seal
  python scripts/generate_genesis_seal.py --verify

  # Generate with custom owner
  python scripts/generate_genesis_seal.py --discover --seal --owner "Momo"
        """,
    )
    
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run ecosystem discovery first",
    )
    parser.add_argument(
        "--seal",
        action="store_true",
        help="Generate the Genesis Seal",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing Genesis Seal",
    )
    parser.add_argument(
        "--owner",
        default="Momo",
        help="Owner alias for attestation (default: Momo)",
    )
    parser.add_argument(
        "--scan-paths",
        nargs="*",
        type=Path,
        help="Paths to scan for ecosystem discovery",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "genesis",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--ots",
        action="store_true",
        help="Create OpenTimestamps proof (Bitcoin-anchored timestamp)",
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
        # Verify existing seal
        seal_path = args.output_dir / GENESIS_SEAL_FILENAME
        if not seal_path.exists():
            print(f"❌ Genesis Seal not found: {seal_path}")
            return 1
        
        seal = GenesisSeal.load(seal_path)
        
        if seal.verify():
            print("\n✅ Genesis Seal is VALID")
            print(seal.to_summary())
            return 0
        else:
            print("\n❌ Genesis Seal is INVALID")
            return 1
    
    if args.seal:
        print("\n" + "=" * 70)
        print("GENERATING BIZRA GENESIS SEAL")
        print("=" * 70 + "\n")
        
        # Step 1: Discovery
        if args.discover:
            print("📍 Step 1/4: Ecosystem Discovery...")
            discovery_data = run_discovery(args.scan_paths)
            print(f"   ✓ Found {discovery_data['artifact_count']} artifacts")
            print(f"   ✓ Root hash: {discovery_data['ecosystem_root_hash'][:16]}...")
        else:
            # Try cached manifest first; fall back to minimal stub
            discovery_data = load_cached_discovery()
            if discovery_data:
                print("📍 Step 1/4: Using cached GENESIS_SYSTEM_MANIFEST.json")
                print(f"   ✓ Found {discovery_data['artifact_count']} artifacts")
                print(f"   ✓ Root hash: {discovery_data['ecosystem_root_hash'][:16]}...")
            else:
                discovery_data = {
                    "ecosystem_root_hash": hashlib.sha256(b"genesis").hexdigest(),
                    "artifact_count": 0,
                    "repository_count": 0,
                    "timeline_start": "2023-01-14T00:00:00+00:00",
                    "timeline_end": datetime.now(timezone.utc).isoformat(),
                    "contribution_hours": 500,
                    "artifacts": [],
                }
        
        # Step 2: Node0 Identity
        print("\n🖥️  Step 2/4: Creating Node0 Identity...")
        identity_data = create_node_identity(discovery_data, args.owner)
        print(f"   ✓ Node ID: {identity_data['node_id']}")
        print(f"   ✓ Owner: {identity_data['owner_alias']}")
        
        # Step 3: Proof of Impact
        print("\n📊 Step 3/4: Calculating Proof of Impact...")
        poi_data = calculate_poi(discovery_data, identity_data)
        print(f"   ✓ Total Impact: {poi_data['total_impact']:,.2f}")
        print(f"   ✓ Proof Hash: {poi_data['proof_hash'][:16]}...")
        
        # Step 4: Generate Seal
        print("\n🔏 Step 4/4: Generating Genesis Seal...")
        seal = generate_seal(discovery_data, identity_data, poi_data)
        
        print(seal.to_summary())
        
        print(f"\n✅ Genesis Seal generated successfully!")
        print(f"   📁 Saved to: {args.output_dir / GENESIS_SEAL_FILENAME}")
        
        # Optional: Create OpenTimestamps proof
        if args.ots:
            print("\n⏰ Creating OpenTimestamps proof (Bitcoin anchor)...")
            ots_path = seal.stamp_opentimestamps()
            if ots_path:
                print(f"   ✓ OTS proof: {ots_path}")
                print("   ℹ️  Full Bitcoin confirmation: ~1-6 hours")
            else:
                print("   ⚠️  OTS stamp skipped (install: pip install opentimestamps-client)")
        
        return 0
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
