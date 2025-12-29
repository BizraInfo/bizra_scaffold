"""
Test Suite for Genesis Chain Verification
═══════════════════════════════════════════════════════════════════════════════
Validates the complete host-centric attestation chain.

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestGenesisChainVerification:
    """Tests for the Genesis Chain verification module."""
    
    def test_verify_genesis_chain_module_import(self):
        """Verify the genesis chain module can be imported."""
        import scripts.verify_genesis_chain as vgc
        
        assert hasattr(vgc, 'verify_genesis_chain')
        assert hasattr(vgc, 'verify_system_manifest')
        assert hasattr(vgc, 'verify_node_zero_identity')
        assert hasattr(vgc, 'verify_genesis_seal')
    
    def test_verification_step_dataclass(self):
        """Test VerificationStep dataclass."""
        from scripts.verify_genesis_chain import VerificationStep
        
        step = VerificationStep(
            name="Test Step",
            passed=True,
            message="Test passed",
            details={"key": "value"}
        )
        
        assert step.name == "Test Step"
        assert step.passed is True
        assert step.message == "Test passed"
        assert step.details == {"key": "value"}
    
    def test_chain_verification_result_dataclass(self):
        """Test ChainVerificationResult dataclass."""
        from datetime import datetime
        from scripts.verify_genesis_chain import ChainVerificationResult, VerificationStep
        
        steps = [
            VerificationStep(name="Step1", passed=True, message="OK"),
            VerificationStep(name="Step2", passed=True, message="OK"),
        ]
        
        result = ChainVerificationResult(
            passed=True,
            steps=steps,
            chain_integrity="COMPLETE",
            timestamp=datetime.utcnow(),
        )
        
        assert result.passed is True
        assert len(result.steps) == 2
        assert result.chain_integrity == "COMPLETE"
    
    def test_hash_bytes_blake3_fallback(self):
        """Test hash function with fallback."""
        from scripts.verify_genesis_chain import hash_bytes_blake3
        
        data = b"test data"
        hash_result = hash_bytes_blake3(data)
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex or BLAKE3 hex
    
    def test_hash_bytes_blake2b(self):
        """Test BLAKE2b hash function."""
        from scripts.verify_genesis_chain import hash_bytes_blake2b
        
        data = b"test data"
        hash_result = hash_bytes_blake2b(data, digest_size=32)
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # 32 bytes = 64 hex chars


class TestGenesisSystemAttestation:
    """Tests for the host-centric system attestation module."""
    
    def test_attestation_module_import(self):
        """Verify the attestation module can be imported."""
        import scripts.genesis_system_attestation as gsa
        
        assert hasattr(gsa, 'get_hardware_fingerprint')
        assert hasattr(gsa, 'scan_bizra_universe')
        assert hasattr(gsa, 'mint_genesis_system_manifest')
    
    def test_hardware_fingerprint_structure(self):
        """Test hardware fingerprint has required fields."""
        from scripts.genesis_system_attestation import get_hardware_fingerprint
        
        fingerprint = get_hardware_fingerprint()
        
        assert "hostname" in fingerprint
        assert "platform" in fingerprint
        assert "architecture" in fingerprint
        assert "hardware_hash" in fingerprint
        
        # Hardware hash should be hex string
        assert len(fingerprint["hardware_hash"]) == 64
    
    def test_scan_bizra_universe_returns_territories(self):
        """Test universe scanning returns territory list."""
        from scripts.genesis_system_attestation import scan_bizra_universe
        
        territories, artifact_count, repo_count = scan_bizra_universe()
        
        assert isinstance(territories, list)
        assert isinstance(artifact_count, int)
        assert isinstance(repo_count, int)
        
        for t in territories:
            assert "path" in t
            assert "status" in t
            assert t["status"] in ("ONLINE", "MISSING")
    
    def test_mint_genesis_manifest_creates_file(self):
        """Test that minting creates manifest file."""
        from scripts.genesis_system_attestation import mint_genesis_system_manifest
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_manifest.json"
            
            result_path = mint_genesis_system_manifest(
                output_path=output_path,
                owner_alias="TestOwner"
            )
            
            assert result_path.exists()
            
            data = json.loads(result_path.read_text(encoding="utf-8"))
            assert data["sovereign_identity"]["architect"] == "TestOwner"
            assert data["block_height"] == 0
            assert data["node_type"] == "GENESIS_NODE_ZERO"
            assert "genesis_hash" in data


class TestNodeZeroManifestBinding:
    """Tests for Node0 identity and manifest binding."""
    
    def test_load_system_manifest_missing(self):
        """Test loading non-existent manifest returns None."""
        from core.genesis.node_zero import load_system_manifest
        
        data, hash_val, path = load_system_manifest(Path("/nonexistent/path.json"))
        
        assert data is None
        assert hash_val is None
        assert path is None
    
    def test_node_zero_identity_has_manifest_fields(self):
        """Test NodeZeroIdentity has manifest binding fields."""
        from core.genesis.node_zero import NodeZeroIdentity
        
        import inspect
        sig = inspect.signature(NodeZeroIdentity.__init__)
        param_names = list(sig.parameters.keys())
        
        # Check dataclass has the new fields
        assert hasattr(NodeZeroIdentity, '__dataclass_fields__')
        fields = NodeZeroIdentity.__dataclass_fields__
        
        assert "system_manifest_path" in fields
        assert "system_manifest_hash" in fields
    
    def test_node_zero_to_dict_includes_manifest(self):
        """Test NodeZeroIdentity.to_dict includes manifest fields."""
        from core.genesis.node_zero import NodeZeroIdentity
        from datetime import datetime, timezone
        
        # Create minimal identity for testing
        identity, _ = NodeZeroIdentity.create_genesis_node(
            owner_alias="TestOwner",
            ecosystem_root_hash="abc123",
        )
        
        data = identity.to_dict()
        
        # Verify manifest fields are serialized
        assert "system_manifest_path" in data
        assert "system_manifest_hash" in data
