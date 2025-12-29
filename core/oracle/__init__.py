"""
BIZRA Oracle Package
════════════════════════════════════════════════════════════════════════════════

External oracle interfaces for independent verification of claims.

This module addresses the Goodhart vulnerability identified in the SAPE
audit: self-referential scoring systems can be gamed. External oracles
provide independent verification that breaks the feedback loop.

Oracle Types:
- LocalOracle: File-based, for development/testing
- RemoteOracle: API-based, for production (future)
- ConsortiumOracle: Multi-party, for high-stakes (future)
- TemporalOracle: Time-locked, for future verification (future)

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from core.oracle.external_oracle import (
    ExternalOracle,
    LocalOracle,
    OracleRegistry,
    OracleAttestation,
    OracleType,
    OracleVerdict,
    IhsanOracleVerifier,
    get_oracle,
    register_oracle,
    ORACLE_IHSAN_THRESHOLD,
    ORACLE_SPOT_CHECK_RATE,
    ORACLE_DRIFT_THRESHOLD,
)

__all__ = [
    # Oracles
    "ExternalOracle",
    "LocalOracle",
    "OracleRegistry",
    # Attestations
    "OracleAttestation",
    # Enums
    "OracleType",
    "OracleVerdict",
    # Verifiers
    "IhsanOracleVerifier",
    # Functions
    "get_oracle",
    "register_oracle",
    # Constants
    "ORACLE_IHSAN_THRESHOLD",
    "ORACLE_SPOT_CHECK_RATE",
    "ORACLE_DRIFT_THRESHOLD",
]
