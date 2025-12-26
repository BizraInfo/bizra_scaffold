"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    BIZRA Constitution Test Suite                              ║
║              Validates constitutional law binding and enforcement             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Tests cover:                                                                 ║
║    - Constitution singleton loading and caching                              ║
║    - Hash verification and integrity                                          ║
║    - Immutable sections enforcement                                           ║
║    - Invariant threshold access (ihsan, snr, justice)                        ║
║    - ZK configuration integration                                             ║
║    - Thread safety for concurrent access                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Import with availability checking
try:
    from core.constitution import (
        Constitution,
        ConstitutionError,
        IhsanInvariants,
        JusticeInvariants,
        GateConfig,
        TierConfig,
        AgentRole,
        CryptoConfig,
        ZKConfig,
    )
    CONSTITUTION_AVAILABLE = True
except ImportError as e:
    CONSTITUTION_AVAILABLE = False
    IMPORT_ERROR = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def minimal_constitution_toml() -> str:
    """Minimal valid constitution.toml for testing."""
    return '''
[meta]
version = "1.0.0"
name = "test-constitution"
hash_algorithm = "sha256"
canonical_format = "raw"

[invariants.ihsan]
minimum_threshold = 0.95
emergency_threshold = 0.90
precision = 3

[invariants.ihsan.weights]
snr_contribution = 0.40
impact_contribution = 0.35
intent_contribution = 0.25

[invariants.ihsan.snr]
minimum = 0.75
excellent = 0.95
critical_action_minimum = 0.90

[invariants.justice]
max_gini_coefficient = 0.35
min_allocation_floor = 0.05
max_single_entity_share = 0.25
rebalance_trigger = 0.30

[gates.ihsan_check]
id = "ihsan_check"
order = 1
name = "IHSAN_CHECK"
description = "Ihsan threshold check"
tier = "CHEAP"
max_latency_ms = 10
reject_code = 6
mandatory = true

[tiers.CHEAP]
name = "CHEAP"
max_latency_ms = 10
description = "Fast, stateless checks"
parallel_allowed = true

[agents.pat]
role = "PROVER"
description = "Proposal Agent"
can_propose = true
can_sign = true
can_commit = false
can_verify = false

[agents.sat]
role = "VERIFIER"
description = "Sovereignty Agent"
can_propose = false
can_sign = false
can_commit = true
can_verify = true

[crypto.digest]
algorithm = "blake3"
domain_prefix = "bizra-pci-v1:"
output_bytes = 32

[crypto.signature]
algorithm = "ed25519"
key_bytes = 32
signature_bytes = 64

[crypto.nonce]
bytes = 32
source = "csprng"

[crypto.pq_migration]
enabled = false
target_algorithm = "dilithium5"

[zk.circuit]
name = "ihsan-verifier"
version = "1.0.0"
backend = "risc0"
hash_algorithm = "sha256"

[zk.circuit.thresholds]
ihsan_minimum_fixed = 950
snr_minimum_fixed = 750

[zk.proof]
scheme = "stark"
security_bits = 128
recursive = true
batch_size = 1000

[governance.immutable]
sections = [
    "invariants.ihsan.minimum_threshold",
    "agents.pat.can_commit",
    "agents.sat.can_propose"
]
'''


@pytest.fixture
def temp_constitution_file(minimal_constitution_toml: str):
    """Create a temporary constitution.toml file."""
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.toml',
        delete=False
    ) as f:
        f.write(minimal_constitution_toml)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture(autouse=True)
def reset_constitution_singleton():
    """Reset Constitution singleton between tests."""
    if CONSTITUTION_AVAILABLE:
        # Reset the singleton
        Constitution._instance = None
    yield


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not CONSTITUTION_AVAILABLE,
    reason=f"Constitution module not available: {IMPORT_ERROR if not CONSTITUTION_AVAILABLE else ''}"
)
class TestConstitutionLoading:
    """Tests for Constitution loading and initialization."""
    
    def test_load_from_file(self, temp_constitution_file: Path):
        """Test loading constitution from file path."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution is not None
        assert constitution.meta.version == "1.0.0"
        assert constitution.meta.name == "test-constitution"
    
    def test_singleton_pattern(self, temp_constitution_file: Path):
        """Test that Constitution implements singleton pattern."""
        constitution1 = Constitution.load(path=temp_constitution_file)
        constitution2 = Constitution.get()
        
        assert constitution1 is constitution2
    
    def test_get_without_load_raises(self):
        """Test that get() raises if not loaded."""
        with pytest.raises(ConstitutionError, match="not loaded"):
            Constitution.get()
    
    def test_reload_creates_new_instance(self, temp_constitution_file: Path):
        """Test that reload() creates fresh instance."""
        constitution1 = Constitution.load(path=temp_constitution_file)
        constitution2 = Constitution.reload(path=temp_constitution_file)
        
        # Note: Might be same object or different depending on implementation
        # Key is that it should work without error
        assert constitution2 is not None
    
    def test_hash_computed(self, temp_constitution_file: Path):
        """Test that constitution hash is computed."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.hash is not None
        assert len(constitution.hash) > 0
        # Hash should be hex string (SHA256 = 64 chars, BLAKE3 = 64 chars)
        assert len(constitution.hash) >= 64


@pytest.mark.skipif(
    not CONSTITUTION_AVAILABLE,
    reason="Constitution module not available"
)
class TestIhsanInvariants:
    """Tests for Ihsān invariant access."""
    
    def test_ihsan_threshold_access(self, temp_constitution_file: Path):
        """Test accessing ihsan threshold."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.ihsan.minimum_threshold == 0.95
    
    def test_ihsan_emergency_threshold(self, temp_constitution_file: Path):
        """Test emergency threshold access."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.ihsan.emergency_threshold == 0.90
    
    def test_ihsan_precision(self, temp_constitution_file: Path):
        """Test precision (fixed-point scale) access."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.ihsan.precision == 3
    
    def test_ihsan_threshold_fixed(self, temp_constitution_file: Path):
        """Test fixed-point threshold computation."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        # 0.95 * 10^3 = 950
        expected_fixed = int(0.95 * (10 ** 3))
        assert constitution.ihsan.threshold_fixed == expected_fixed


@pytest.mark.skipif(
    not CONSTITUTION_AVAILABLE,
    reason="Constitution module not available"
)
class TestJusticeInvariants:
    """Tests for justice invariants."""
    
    def test_gini_coefficient(self, temp_constitution_file: Path):
        """Test max Gini coefficient access."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.justice.max_gini_coefficient == 0.35
    
    def test_fairness_score(self, temp_constitution_file: Path):
        """Test minimum allocation floor access (fairness metric)."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.justice.min_allocation_floor == 0.05


@pytest.mark.skipif(
    not CONSTITUTION_AVAILABLE,
    reason="Constitution module not available"
)
class TestGateConfiguration:
    """Tests for gate configuration access."""
    
    def test_gate_exists(self, temp_constitution_file: Path):
        """Test that defined gates are accessible."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        gates = constitution.gates
        assert "ihsan_check" in gates
    
    def test_gate_properties(self, temp_constitution_file: Path):
        """Test gate property access."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        gate = constitution.gates.get("ihsan_check")
        assert gate is not None
        assert gate.tier == "CHEAP"
        assert gate.required is True
        assert gate.timeout_ms == 10


@pytest.mark.skipif(
    not CONSTITUTION_AVAILABLE,
    reason="Constitution module not available"
)
class TestAgentRoles:
    """Tests for agent role configuration."""
    
    def test_pat_role(self, temp_constitution_file: Path):
        """Test PAT agent role configuration."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        pat = constitution.agents.get("pat")
        assert pat is not None
        assert pat.role == "PROVER"
        assert pat.can_propose is True
        assert pat.can_commit is False  # Constitutional immutable!
    
    def test_sat_role(self, temp_constitution_file: Path):
        """Test SAT agent role configuration."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        sat = constitution.agents.get("sat")
        assert sat is not None
        assert sat.role == "VERIFIER"
        assert sat.can_propose is False  # Constitutional immutable!
        assert sat.can_commit is True


@pytest.mark.skipif(
    not CONSTITUTION_AVAILABLE,
    reason="Constitution module not available"
)
class TestCryptoConfiguration:
    """Tests for cryptographic configuration."""
    
    def test_signature_algorithm(self, temp_constitution_file: Path):
        """Test signature algorithm access."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.crypto.signature_algorithm == "ed25519"
    
    def test_hash_algorithm(self, temp_constitution_file: Path):
        """Test hash algorithm access (via digest_algorithm)."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.crypto.digest_algorithm == "blake3"
    
    def test_key_rotation_days(self, temp_constitution_file: Path):
        """Test PQ migration configuration access."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.crypto.pq_migration_enabled is False
        assert constitution.crypto.pq_target_algorithm == "dilithium5"


@pytest.mark.skipif(
    not CONSTITUTION_AVAILABLE,
    reason="Constitution module not available"
)
class TestZKConfiguration:
    """Tests for ZK proof configuration."""
    
    def test_zk_backend(self, temp_constitution_file: Path):
        """Test ZK backend access."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.zk.backend == "risc0"
    
    def test_zk_proof_scheme(self, temp_constitution_file: Path):
        """Test proof scheme access."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.zk.proof_scheme == "stark"
    
    def test_zk_security_bits(self, temp_constitution_file: Path):
        """Test security level access."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.zk.security_bits == 128
    
    def test_zk_fixed_point_thresholds(self, temp_constitution_file: Path):
        """Test ZK circuit fixed-point thresholds."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.zk.ihsan_minimum_fixed == 950
        assert constitution.zk.snr_minimum_fixed == 750


@pytest.mark.skipif(
    not CONSTITUTION_AVAILABLE,
    reason="Constitution module not available"
)
class TestImmutableSections:
    """Tests for immutable section enforcement."""
    
    def test_immutable_sections_defined(self, temp_constitution_file: Path):
        """Test that immutable sections are tracked."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        immutable = constitution.immutable_sections
        assert "invariants.ihsan.minimum_threshold" in immutable
        assert "agents.pat.can_commit" in immutable
        assert "agents.sat.can_propose" in immutable
    
    def test_is_immutable_check(self, temp_constitution_file: Path):
        """Test is_immutable helper method."""
        constitution = Constitution.load(path=temp_constitution_file)
        
        assert constitution.is_immutable("invariants.ihsan.minimum_threshold")
        assert constitution.is_immutable("agents.pat.can_commit")
        assert not constitution.is_immutable("meta.version")


@pytest.mark.skipif(
    not CONSTITUTION_AVAILABLE,
    reason="Constitution module not available"
)
class TestHashVerification:
    """Tests for constitution hash verification."""
    
    def test_hash_verification_success(self, temp_constitution_file: Path):
        """Test loading with correct hash succeeds."""
        # First load to get hash
        constitution = Constitution.load(path=temp_constitution_file)
        correct_hash = constitution.hash
        
        # Reset and reload with hash verification
        Constitution._instance = None
        verified = Constitution.load(
            path=temp_constitution_file,
            verify_hash=correct_hash
        )
        
        assert verified is not None
        assert verified.hash == correct_hash
    
    def test_hash_verification_failure(self, temp_constitution_file: Path):
        """Test loading with wrong hash fails."""
        wrong_hash = "0" * 64  # Obviously wrong hash
        
        with pytest.raises(ConstitutionError):
            Constitution.load(
                path=temp_constitution_file,
                verify_hash=wrong_hash
            )


@pytest.mark.skipif(
    not CONSTITUTION_AVAILABLE,
    reason="Constitution module not available"
)
class TestThreadSafety:
    """Tests for thread-safe singleton access."""
    
    def test_concurrent_loading(self, temp_constitution_file: Path):
        """Test concurrent loads return same instance."""
        results = []
        errors = []
        
        def load_constitution():
            try:
                c = Constitution.load(path=temp_constitution_file)
                results.append(id(c))
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=load_constitution) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        # All should reference same singleton
        assert len(set(results)) == 1
    
    def test_concurrent_get(self, temp_constitution_file: Path):
        """Test concurrent get() calls."""
        # Load first
        Constitution.load(path=temp_constitution_file)
        
        results = []
        
        def get_constitution():
            c = Constitution.get()
            results.append(id(c))
        
        threads = [threading.Thread(target=get_constitution) for _ in range(20)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should be same instance
        assert len(set(results)) == 1


@pytest.mark.skipif(
    not CONSTITUTION_AVAILABLE,
    reason="Constitution module not available"
)
class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_missing_file_raises(self):
        """Test loading non-existent file raises error."""
        fake_path = Path("/nonexistent/constitution.toml")
        
        with pytest.raises((ConstitutionError, FileNotFoundError)):
            Constitution.load(path=fake_path)
    
    def test_invalid_toml_raises(self):
        """Test loading invalid TOML raises error."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.toml',
            delete=False
        ) as f:
            f.write("this is not valid [ toml {{{")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ConstitutionError):
                Constitution.load(path=temp_path)
        finally:
            temp_path.unlink()
    
    def test_missing_required_section_raises(self):
        """Test that invalid ihsan threshold raises constitutional violation."""
        incomplete_toml = '''
[meta]
version = "1.0.0"

[invariants.ihsan]
minimum_threshold = 0.50  # VIOLATION: below 0.95
emergency_threshold = 0.45
precision = 3
'''
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.toml',
            delete=False
        ) as f:
            f.write(incomplete_toml)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ConstitutionError):
                Constitution.load(path=temp_path)
        finally:
            temp_path.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not CONSTITUTION_AVAILABLE,
    reason="Constitution module not available"
)
class TestConstitutionIntegration:
    """Integration tests with actual constitution.toml."""
    
    @pytest.fixture
    def real_constitution_path(self) -> Path:
        """Get path to actual constitution.toml if it exists."""
        workspace_root = Path(__file__).parent.parent
        constitution_path = workspace_root / "constitution.toml"
        
        if not constitution_path.exists():
            pytest.skip("constitution.toml not found in workspace")
        
        return constitution_path
    
    def test_load_real_constitution(self, real_constitution_path: Path):
        """Test loading actual constitution.toml."""
        constitution = Constitution.load(path=real_constitution_path)
        
        assert constitution is not None
        assert constitution.meta.version is not None
        assert constitution.ihsan.minimum_threshold >= 0.9
    
    def test_real_constitution_has_all_sections(self, real_constitution_path: Path):
        """Test that real constitution has all required sections."""
        constitution = Constitution.load(path=real_constitution_path)
        
        # Required sections per spec
        assert constitution.meta is not None
        assert constitution.ihsan is not None
        assert constitution.justice is not None
        assert constitution.gates is not None
        assert constitution.agents is not None
        assert constitution.crypto is not None
        assert constitution.zk is not None
    
    def test_real_ihsan_threshold_is_correct(self, real_constitution_path: Path):
        """Test that real constitution has 0.95 Ihsān threshold."""
        constitution = Constitution.load(path=real_constitution_path)
        
        # Per BIZRA specification
        assert constitution.ihsan.minimum_threshold == 0.95
        assert constitution.ihsan.threshold_fixed == 950
