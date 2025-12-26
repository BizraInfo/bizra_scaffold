"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║               BIZRA Constitution Loader and Validator v1.0.0                  ║
║                  Runtime Binding to constitution.toml                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  This module provides:                                                        ║
║    - Atomic loading and validation of constitution.toml                       ║
║    - Runtime access to invariants and thresholds                              ║
║    - Constitution hash for policy binding                                     ║
║    - Immutable enforcement of critical sections                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

try:
    import toml

    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

try:
    import blake3

    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

try:
    import json

    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL ENCODING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def _canonical_json_encode(obj: Any) -> bytes:
    """
    Encode object to canonical JSON per RFC 8785 (JCS).

    Key rules:
    - Keys sorted lexicographically
    - No whitespace
    - UTF-8 encoding
    - Numbers without unnecessary precision
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class HashAlgorithmError(Exception):
    """Raised when required hash algorithm is unavailable."""

    pass


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════


class ConstitutionError(Exception):
    """Base exception for constitution-related errors."""

    pass


class ConstitutionNotFoundError(ConstitutionError):
    """Raised when constitution.toml cannot be found."""

    pass


class ConstitutionValidationError(ConstitutionError):
    """Raised when constitution fails validation."""

    pass


class ConstitutionTamperingError(ConstitutionError):
    """Raised when constitution hash mismatch detected (potential tampering)."""

    pass


class ImmutableViolationError(ConstitutionError):
    """Raised when attempting to modify immutable constitution section."""

    pass


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class IhsanInvariants:
    """Immutable Ihsān invariant values."""

    minimum_threshold: float
    emergency_threshold: float
    precision: int
    snr_contribution: float
    impact_contribution: float
    intent_contribution: float
    snr_minimum: float
    snr_excellent: float
    snr_critical_minimum: float

    def __post_init__(self):
        # Validation
        if not 0.0 <= self.minimum_threshold <= 1.0:
            raise ConstitutionValidationError(
                f"Invalid minimum_threshold: {self.minimum_threshold}"
            )
        if self.minimum_threshold < 0.95:
            raise ConstitutionValidationError(
                f"CONSTITUTIONAL VIOLATION: minimum_threshold {self.minimum_threshold} < 0.95"
            )

    @property
    def minimum_fixed(self) -> int:
        """Fixed-point representation for ZK circuits."""
        return int(self.minimum_threshold * (10**self.precision))

    @property
    def threshold_fixed(self) -> int:
        """Alias for minimum_fixed for API compatibility."""
        return self.minimum_fixed


@dataclass(frozen=True)
class JusticeInvariants:
    """Immutable justice invariant values."""

    max_gini_coefficient: float
    min_allocation_floor: float
    max_single_entity_share: float
    rebalance_trigger: float


@dataclass(frozen=True)
class GateConfig:
    """Configuration for a verification gate."""

    id: str  # Gate identifier
    order: int
    name: str
    description: str
    tier: str
    max_latency_ms: int
    reject_code: int
    mandatory: bool
    timeout_ms: int = 100  # Alias for max_latency_ms for API compatibility

    @property
    def required(self) -> bool:
        """Alias for mandatory for API compatibility."""
        return self.mandatory


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a verification tier."""

    name: str
    max_latency_ms: int
    description: str
    parallel_allowed: bool


@dataclass(frozen=True)
class AgentRole:
    """Configuration for an agent role."""

    role: str
    description: str
    can_propose: bool
    can_sign: bool
    can_commit: bool
    can_verify: bool


@dataclass(frozen=True)
class CryptoConfig:
    """Cryptographic configuration."""

    digest_algorithm: str
    digest_domain_prefix: str
    digest_output_bytes: int
    signature_algorithm: str
    signature_key_bytes: int
    signature_bytes: int
    nonce_bytes: int
    pq_migration_enabled: bool
    pq_target_algorithm: str


@dataclass(frozen=True)
class ZKConfig:
    """Zero-knowledge proof configuration."""

    circuit_name: str
    circuit_version: str
    backend: str
    hash_algorithm: str
    ihsan_minimum_fixed: int
    snr_minimum_fixed: int
    proof_scheme: str
    security_bits: int
    recursive: bool
    batch_size: int


# ══════════════════════════════════════════════════════════════════════════════
# CONSTITUTION CLASS
# ══════════════════════════════════════════════════════════════════════════════


class Constitution:
    """
    Singleton class representing the BIZRA Constitution.

    Provides thread-safe access to constitutional invariants and configuration.
    The constitution is loaded once and cached; reloading requires explicit call
    to reload() which will verify hash integrity.

    Usage:
        constitution = Constitution.load()
        threshold = constitution.ihsan.minimum_threshold
        hash = constitution.hash
    """

    _instance: Optional[Constitution] = None
    _lock = threading.Lock()

    # Immutable sections that CANNOT be changed without re-genesis
    IMMUTABLE_PATHS: FrozenSet[str] = frozenset(
        {
            "invariants.ihsan.minimum_threshold",
            "agents.pat.can_commit",
            "agents.sat.can_propose",
        }
    )

    def __init__(
        self,
        raw_config: Dict[str, Any],
        source_path: Path,
        hash_value: str,
    ):
        """
        Initialize constitution. Use Constitution.load() instead.

        Args:
            raw_config: Parsed TOML configuration
            source_path: Path to constitution.toml
            hash_value: BLAKE3/SHA256 hash of the file
        """
        self._raw = raw_config
        self._source_path = source_path
        self._hash = hash_value
        self._loaded_at = datetime.now(timezone.utc)

        # Parse and validate sections
        self._ihsan = self._parse_ihsan()
        self._justice = self._parse_justice()
        self._gates = self._parse_gates()
        self._tiers = self._parse_tiers()
        self._agents = self._parse_agents()
        self._crypto = self._parse_crypto()
        self._zk = self._parse_zk()

        # Validate immutable constraints
        self._validate_immutable()

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        verify_hash: Optional[str] = None,
    ) -> Constitution:
        """
        Load and validate constitution.toml.

        Args:
            path: Path to constitution.toml (default: ./constitution.toml)
            verify_hash: Expected hash for integrity verification

        Returns:
            Constitution instance (singleton)

        Raises:
            ConstitutionNotFoundError: If file not found
            ConstitutionValidationError: If validation fails
            ConstitutionTamperingError: If hash verification fails
        """
        if not TOML_AVAILABLE:
            raise ConstitutionError(
                "toml library required. Install with: pip install toml"
            )

        with cls._lock:
            # Return cached instance if already loaded
            if cls._instance is not None:
                if verify_hash and cls._instance.hash != verify_hash:
                    raise ConstitutionTamperingError(
                        f"Hash mismatch: expected {verify_hash}, got {cls._instance.hash}"
                    )
                return cls._instance

            # Resolve path
            if path is None:
                path = Path("constitution.toml")
            path = Path(path)

            if not path.exists():
                raise ConstitutionNotFoundError(f"Constitution not found: {path}")

            # Parse TOML first to read meta settings
            content = path.read_bytes()
            try:
                raw_config = toml.loads(content.decode("utf-8"))
            except Exception as e:
                raise ConstitutionValidationError(f"Invalid TOML: {e}")

            # Determine hash algorithm from meta section
            meta = raw_config.get("meta", {})
            hash_algorithm = meta.get("hash_algorithm", "blake3").lower()
            canonical_format = meta.get("canonical_format", "raw").lower()

            # Compute hash using canonical format if specified
            if canonical_format == "rfc8785" or canonical_format == "jcs":
                # Use canonical JSON representation for deterministic hashing
                hash_input = _canonical_json_encode(raw_config)
            else:
                # Use raw file bytes
                hash_input = content

            # Compute hash using specified algorithm
            if hash_algorithm == "blake3":
                if not BLAKE3_AVAILABLE:
                    raise HashAlgorithmError(
                        "Constitution requires blake3 but it is not installed. "
                        "Install with: pip install blake3"
                    )
                hash_value = blake3.blake3(hash_input).hexdigest()
            elif hash_algorithm == "sha256":
                hash_value = hashlib.sha256(hash_input).hexdigest()
            elif hash_algorithm == "sha3-256":
                hash_value = hashlib.sha3_256(hash_input).hexdigest()
            else:
                raise ConstitutionValidationError(
                    f"Unsupported hash algorithm: {hash_algorithm}. "
                    f"Supported: blake3, sha256, sha3-256"
                )

            # Verify hash if provided
            if verify_hash and hash_value != verify_hash:
                raise ConstitutionTamperingError(
                    f"Hash mismatch: expected {verify_hash}, got {hash_value}"
                )

            # Create instance
            cls._instance = cls(raw_config, path, hash_value)
            return cls._instance

    @classmethod
    def reload(cls, path: Optional[Path] = None) -> Constitution:
        """Force reload of constitution (clears cache)."""
        with cls._lock:
            cls._instance = None
        return cls.load(path)

    @classmethod
    def get(cls) -> Constitution:
        """Get cached constitution instance."""
        if cls._instance is None:
            raise ConstitutionError(
                "Constitution not loaded. Call Constitution.load() first."
            )
        return cls._instance

    # ──────────────────────────────────────────────────────────────────────────
    # PROPERTIES
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def version(self) -> str:
        """Constitution version."""
        return self._raw.get("meta", {}).get("version", "unknown")

    @property
    def meta(self) -> Any:
        """Meta section as namespace object for backward compatibility."""

        class MetaNamespace:
            def __init__(self, raw: Dict[str, Any]):
                self._raw = raw

            @property
            def version(self) -> str:
                return self._raw.get("version", "unknown")

            @property
            def name(self) -> str:
                return self._raw.get("name", "bizra-constitution")

            @property
            def hash_algorithm(self) -> str:
                return self._raw.get("hash_algorithm", "blake3")

            @property
            def canonical_format(self) -> str:
                return self._raw.get("canonical_format", "raw")

            @property
            def schema_version(self) -> str:
                return self._raw.get("schema_version", "unknown")

        return MetaNamespace(self._raw.get("meta", {}))

    @property
    def hash(self) -> str:
        """BLAKE3/SHA256 hash of constitution.toml."""
        return self._hash

    @property
    def hash_short(self) -> str:
        """Shortened hash for display (first 16 chars)."""
        return self._hash[:16]

    @property
    def loaded_at(self) -> datetime:
        """When the constitution was loaded."""
        return self._loaded_at

    @property
    def ihsan(self) -> IhsanInvariants:
        """Ihsān invariants."""
        return self._ihsan

    @property
    def justice(self) -> JusticeInvariants:
        """Justice invariants."""
        return self._justice

    @property
    def gates(self) -> Dict[str, GateConfig]:
        """Verification gates."""
        return self._gates

    @property
    def tiers(self) -> Dict[str, TierConfig]:
        """Verification tiers."""
        return self._tiers

    @property
    def agents(self) -> Dict[str, AgentRole]:
        """Agent role configurations."""
        return self._agents

    @property
    def crypto(self) -> CryptoConfig:
        """Cryptographic configuration."""
        return self._crypto

    @property
    def zk(self) -> ZKConfig:
        """Zero-knowledge proof configuration."""
        return self._zk

    # ──────────────────────────────────────────────────────────────────────────
    # PARSING METHODS
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_ihsan(self) -> IhsanInvariants:
        """Parse Ihsān invariants section."""
        inv = self._raw.get("invariants", {}).get("ihsan", {})
        weights = inv.get("weights", {})
        snr = inv.get("snr", {})

        return IhsanInvariants(
            minimum_threshold=inv.get("minimum_threshold", 0.95),
            emergency_threshold=inv.get("emergency_threshold", 0.90),
            precision=inv.get("precision", 3),
            snr_contribution=weights.get("snr_contribution", 0.40),
            impact_contribution=weights.get("impact_contribution", 0.35),
            intent_contribution=weights.get("intent_contribution", 0.25),
            snr_minimum=snr.get("minimum", 0.75),
            snr_excellent=snr.get("excellent", 0.95),
            snr_critical_minimum=snr.get("critical_action_minimum", 0.90),
        )

    def _parse_justice(self) -> JusticeInvariants:
        """Parse justice invariants section."""
        inv = self._raw.get("invariants", {}).get("justice", {})

        return JusticeInvariants(
            max_gini_coefficient=inv.get("max_gini_coefficient", 0.35),
            min_allocation_floor=inv.get("min_allocation_floor", 0.05),
            max_single_entity_share=inv.get("max_single_entity_share", 0.25),
            rebalance_trigger=inv.get("rebalance_trigger", 0.30),
        )

    def _parse_gates(self) -> Dict[str, GateConfig]:
        """Parse verification gates."""
        gates_raw = self._raw.get("gates", {})
        gates = {}

        for name, config in gates_raw.items():
            if isinstance(config, dict) and (
                "order" in config or "id" in config or "tier" in config
            ):
                max_latency = config.get(
                    "max_latency_ms", config.get("timeout_ms", 100)
                )
                gates[name] = GateConfig(
                    id=config.get("id", name),
                    order=config.get("order", 0),
                    name=config.get("name", name.upper()),
                    description=config.get("description", ""),
                    tier=config.get("tier", "MEDIUM"),
                    max_latency_ms=max_latency,
                    reject_code=config.get("reject_code", 99),
                    mandatory=config.get("mandatory", config.get("required", True)),
                    timeout_ms=max_latency,
                )

        return gates

    def _parse_tiers(self) -> Dict[str, TierConfig]:
        """Parse verification tiers."""
        tiers_raw = self._raw.get("tiers", {})
        tiers = {}

        for name, config in tiers_raw.items():
            if isinstance(config, dict):
                tiers[name] = TierConfig(
                    name=config.get("name", name.upper()),
                    max_latency_ms=config.get("max_latency_ms", 100),
                    description=config.get("description", ""),
                    parallel_allowed=config.get("parallel_allowed", False),
                )

        return tiers

    def _parse_agents(self) -> Dict[str, AgentRole]:
        """Parse agent role configurations."""
        agents_raw = self._raw.get("agents", {})
        agents = {}

        for name, config in agents_raw.items():
            if isinstance(config, dict):
                agents[name] = AgentRole(
                    role=config.get("role", name.upper()),
                    description=config.get("description", ""),
                    can_propose=config.get("can_propose", False),
                    can_sign=config.get("can_sign", False),
                    can_commit=config.get("can_commit", False),
                    can_verify=config.get("can_verify", False),
                )

        return agents

    def _parse_crypto(self) -> CryptoConfig:
        """Parse cryptographic configuration."""
        crypto = self._raw.get("crypto", {})
        digest = crypto.get("digest", {})
        sig = crypto.get("signature", {})
        nonce = crypto.get("nonce", {})
        pq = crypto.get("pq_migration", {})

        return CryptoConfig(
            digest_algorithm=digest.get("algorithm", "blake3"),
            digest_domain_prefix=digest.get("domain_prefix", "bizra-pci-v1:"),
            digest_output_bytes=digest.get("output_bytes", 32),
            signature_algorithm=sig.get("algorithm", "ed25519"),
            signature_key_bytes=sig.get("key_bytes", 32),
            signature_bytes=sig.get("signature_bytes", 64),
            nonce_bytes=nonce.get("bytes", 32),
            pq_migration_enabled=pq.get("enabled", False),
            pq_target_algorithm=pq.get("target_algorithm", "dilithium5"),
        )

    def _parse_zk(self) -> ZKConfig:
        """Parse ZK circuit configuration."""
        zk = self._raw.get("zk", {})
        circuit = zk.get("circuit", {})
        thresholds = circuit.get("thresholds", {})
        proof = zk.get("proof", {})

        return ZKConfig(
            circuit_name=circuit.get("name", "ihsan-verifier"),
            circuit_version=circuit.get("version", "1.0.0"),
            backend=circuit.get("backend", "risc0"),
            hash_algorithm=circuit.get("hash_algorithm", "sha256"),
            ihsan_minimum_fixed=thresholds.get("ihsan_minimum_fixed", 950),
            snr_minimum_fixed=thresholds.get("snr_minimum_fixed", 750),
            proof_scheme=proof.get("scheme", "stark"),
            security_bits=proof.get("security_bits", 128),
            recursive=proof.get("recursive", True),
            batch_size=proof.get("batch_size", 1000),
        )

    def _validate_immutable(self):
        """Validate immutable constitutional constraints."""
        # PAT cannot commit
        if self._agents.get("pat") and self._agents["pat"].can_commit:
            raise ConstitutionValidationError(
                "IMMUTABLE VIOLATION: PAT agent cannot have commit capability"
            )

        # SAT cannot propose
        if self._agents.get("sat") and self._agents["sat"].can_propose:
            raise ConstitutionValidationError(
                "IMMUTABLE VIOLATION: SAT agent cannot have propose capability"
            )

        # Ihsān threshold must be >= 0.95
        if self._ihsan.minimum_threshold < 0.95:
            raise ConstitutionValidationError(
                f"IMMUTABLE VIOLATION: Ihsān threshold {self._ihsan.minimum_threshold} < 0.95"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def immutable_sections(self) -> FrozenSet[str]:
        """Get the set of immutable section paths from governance.immutable."""
        governance = self._raw.get("governance", {})
        immutable = governance.get("immutable", {})
        sections = immutable.get("sections", [])
        return frozenset(sections) | self.IMMUTABLE_PATHS

    def is_immutable(self, path: str) -> bool:
        """Check if a configuration path is immutable."""
        return path in self.immutable_sections

    def get_gate_chain(self) -> List[GateConfig]:
        """Get gates sorted by execution order."""
        return sorted(self._gates.values(), key=lambda g: g.order)

    def get_tier_latency(self, tier_name: str) -> int:
        """Get maximum latency for a tier."""
        tier = self._tiers.get(tier_name.lower())
        if tier:
            return tier.max_latency_ms
        return 100  # Default

    def validate_action(
        self,
        ihsan_score: float,
        snr_score: float,
        agent_role: str,
        action_type: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Pre-validate an action against constitutional constraints.

        Args:
            ihsan_score: Agent's Ihsān score
            snr_score: Signal-to-Noise ratio
            agent_role: "pat" or "sat"
            action_type: "propose", "commit", etc.

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        # Check Ihsān threshold
        if ihsan_score < self._ihsan.minimum_threshold:
            return (
                False,
                f"Ihsān {ihsan_score} below threshold {self._ihsan.minimum_threshold}",
            )

        # Check SNR threshold
        if snr_score < self._ihsan.snr_minimum:
            return False, f"SNR {snr_score} below threshold {self._ihsan.snr_minimum}"

        # Check role permissions
        agent = self._agents.get(agent_role)
        if not agent:
            return False, f"Unknown agent role: {agent_role}"

        action_map = {
            "propose": "can_propose",
            "sign": "can_sign",
            "commit": "can_commit",
            "verify": "can_verify",
        }

        required_capability = action_map.get(action_type)
        if required_capability:
            if not getattr(agent, required_capability, False):
                return False, f"Role {agent_role} cannot perform action: {action_type}"

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Export constitution as dictionary."""
        return {
            "version": self.version,
            "hash": self.hash,
            "ihsan": {
                "minimum_threshold": self._ihsan.minimum_threshold,
                "minimum_fixed": self._ihsan.minimum_fixed,
            },
            "gates": {name: gate.name for name, gate in self._gates.items()},
            "agents": {name: agent.role for name, agent in self._agents.items()},
        }

    def __repr__(self) -> str:
        return f"<Constitution v{self.version} hash={self.hash_short}>"


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def get_ihsan_threshold() -> float:
    """Get the current Ihsān threshold from constitution."""
    return Constitution.get().ihsan.minimum_threshold


def get_constitution_hash() -> str:
    """Get the constitution hash for policy binding."""
    return Constitution.get().hash


def validate_ihsan(score: float) -> bool:
    """Check if an Ihsān score meets the constitutional threshold."""
    return score >= Constitution.get().ihsan.minimum_threshold


# ══════════════════════════════════════════════════════════════════════════════
# MODULE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "Constitution",
    "ConstitutionError",
    "ConstitutionNotFoundError",
    "ConstitutionValidationError",
    "ConstitutionTamperingError",
    "ImmutableViolationError",
    "HashAlgorithmError",
    "IhsanInvariants",
    "JusticeInvariants",
    "GateConfig",
    "TierConfig",
    "AgentRole",
    "CryptoConfig",
    "ZKConfig",
    "get_ihsan_threshold",
    "get_constitution_hash",
    "validate_ihsan",
]
