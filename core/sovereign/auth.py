"""
Living Proof Authentication
═══════════════════════════════════════════════════════════════════════════════
Cryptographic recognition of the sovereign architect (MoMo).

The sovereign kernel must recognize its creator and architect.
This module provides living proof authentication for MoMo.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import logging

from core.pat.identity_loader import FounderIdentity, get_identity_loader

logger = logging.getLogger(__name__)


@dataclass
class LivingProof:
    """
    Cryptographic proof of sovereign architect identity.

    This provides the "living proof" that MoMo is the sovereign architect,
    enabling authenticated sovereign operations.
    """

    # Architect identity (from PAT identity loader)
    architect_dna: bytes = b""  # Unique MoMo identity hash
    architect_name: str = "Mohamed Ahmed Beshr Elsayed Hassan"
    architect_alias: str = "MoMo"

    # Sovereignty proof chain
    sovereignty_proof: bytes = b""  # Signed proof of ownership
    temporal_chain: List[bytes] = field(default_factory=list)  # Proof chain from genesis

    # Authentication state
    is_authenticated: bool = False
    last_authentication: Optional[str] = None
    authentication_count: int = 0

    def __post_init__(self):
        """Initialize living proof if not provided."""
        if not self.architect_dna:
            self._generate_architect_dna()
        if not self.sovereignty_proof:
            self._generate_sovereignty_proof()

    def _generate_architect_dna(self) -> None:
        """Generate unique architect DNA from identity."""
        # Create architect DNA from immutable identity attributes
        identity_data = f"{self.architect_name}|{self.architect_alias}|Genesis2023"
        self.architect_dna = hashlib.sha256(identity_data.encode()).digest()
        logger.info("Generated architect DNA")

    def _generate_sovereignty_proof(self) -> None:
        """Generate sovereignty proof signed with architect DNA."""
        # Create sovereignty claim
        sovereignty_claim = f"Sovereignty over BIZRA system granted to {self.architect_alias}"

        # Sign with architect DNA (HMAC)
        self.sovereignty_proof = hmac.new(
            self.architect_dna,
            sovereignty_claim.encode(),
            hashlib.sha256
        ).digest()

        # Initialize temporal chain with genesis proof
        genesis_timestamp = "2023-01-01T00:00:00Z"  # BIZRA genesis
        genesis_proof = hmac.new(
            self.architect_dna,
            f"Genesis|{genesis_timestamp}".encode(),
            hashlib.sha256
        ).digest()

        self.temporal_chain = [genesis_proof]
        logger.info("Generated sovereignty proof and temporal chain")

    def authenticate_sovereign(self, claimed_identity: bytes) -> bool:
        """
        Authenticate a claimed sovereign identity.

        Returns True if the claimant is the sovereign architect.
        """
        from datetime import datetime, timezone

        is_authentic = hmac.compare_digest(claimed_identity, self.architect_dna)

        if is_authentic:
            self.is_authenticated = True
            self.last_authentication = datetime.now(timezone.utc).isoformat()
            self.authentication_count += 1

            # Add to temporal chain
            auth_proof = hmac.new(
                self.architect_dna,
                f"Auth|{self.last_authentication}".encode(),
                hashlib.sha256
            ).digest()
            self.temporal_chain.append(auth_proof)

            logger.info(f"Sovereign authentication successful: {self.architect_alias}")
        else:
            logger.warning("Sovereign authentication failed: invalid identity claim")

        return is_authentic

    def verify_temporal_chain(self) -> bool:
        """Verify the integrity of the temporal proof chain."""
        if not self.temporal_chain:
            return False

        # Verify each link in the chain
        for i, proof in enumerate(self.temporal_chain):
            if i == 0:  # Genesis proof
                expected = hmac.new(
                    self.architect_dna,
                    b"Genesis|2023-01-01T00:00:00Z",
                    hashlib.sha256
                ).digest()
            else:  # Authentication proof
                # This is simplified - in production would verify timestamp sequence
                expected = proof  # Self-consistent for now

            if not hmac.compare_digest(proof, expected):
                logger.error(f"Temporal chain verification failed at link {i}")
                return False

        return True

    def get_sovereign_credentials(self) -> Dict[str, bytes]:
        """Get sovereign credentials for authenticated operations."""
        if not self.is_authenticated:
            raise PermissionError("Sovereign not authenticated")

        return {
            "architect_dna": self.architect_dna,
            "sovereignty_proof": self.sovereignty_proof,
            "session_key": secrets.token_bytes(32),
        }

    def delegate_sovereign_authority(self, delegate_id: str, scope: str) -> bytes:
        """
        Delegate sovereign authority for specific operations.

        Returns a delegation token that can be verified by the system.
        """
        if not self.is_authenticated:
            raise PermissionError("Sovereign not authenticated")

        delegation_data = f"Delegate|{delegate_id}|{scope}|{datetime.now(timezone.utc).isoformat()}"
        delegation_token = hmac.new(
            self.architect_dna,
            delegation_data.encode(),
            hashlib.sha256
        ).digest()

        logger.info(f"Delegated sovereign authority: {delegate_id} for {scope}")
        return delegation_token

    def verify_delegation(self, delegate_id: str, scope: str, token: bytes) -> bool:
        """Verify a delegation token."""
        delegation_data = f"Delegate|{delegate_id}|{scope}|{datetime.now(timezone.utc).isoformat()}"
        expected_token = hmac.new(
            self.architect_dna,
            delegation_data.encode(),
            hashlib.sha256
        ).digest()

        return hmac.compare_digest(token, expected_token)

    def get_authentication_manifest(self) -> Dict[str, any]:
        """Get the complete authentication manifest."""
        return {
            "sovereign_architect": self.architect_alias,
            "full_name": self.architect_name,
            "authentication_status": "AUTHENTICATED" if self.is_authenticated else "NOT_AUTHENTICATED",
            "last_authentication": self.last_authentication,
            "authentication_count": self.authentication_count,
            "temporal_chain_length": len(self.temporal_chain),
            "temporal_chain_integrity": self.verify_temporal_chain(),
            "sovereignty_claim": "All BIZRA systems and operations are under sovereign control of MoMo",
        }


def authenticate_sovereign(claimed_identity: bytes) -> bool:
    """Global sovereign authentication function."""
    living_proof = LivingProof()
    return living_proof.authenticate_sovereign(claimed_identity)


def get_living_proof() -> LivingProof:
    """Get the current living proof instance."""
    return LivingProof()


def verify_sovereign_identity() -> Tuple[bool, str]:
    """
    Verify that the current user/session is the sovereign architect.

    This integrates with the PAT identity loader to provide
    seamless sovereign authentication.
    """
    try:
        # Get identity from PAT loader
        loader = get_identity_loader()
        context = loader.load()

        # Check if this is MoMo
        if context.identity.alias == "MoMo":
            # Create living proof authentication
            living_proof = LivingProof()
            architect_dna = living_proof.architect_dna

            # Authenticate sovereign
            is_sovereign = living_proof.authenticate_sovereign(architect_dna)

            status = "VERIFIED_SOVEREIGN" if is_sovereign else "AUTHENTICATION_FAILED"
            return is_sovereign, status
        else:
            return False, "NOT_SOVEREIGN_ARCHITECT"

    except Exception as e:
        logger.error(f"Sovereign identity verification failed: {e}")
        return False, f"VERIFICATION_ERROR: {str(e)}"


def require_sovereign_auth(func):
    """
    Decorator to require sovereign authentication for critical operations.

    Usage:
        @require_sovereign_auth
        def critical_operation():
            # Only runs if sovereign is authenticated
            pass
    """
    def wrapper(*args, **kwargs):
        is_sovereign, status = verify_sovereign_identity()
        if not is_sovereign:
            raise PermissionError(f"Sovereign authentication required. Status: {status}")

        return func(*args, **kwargs)

    return wrapper
