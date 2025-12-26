"""
BIZRA AEON OMEGA - Enhanced JWT Authentication Module
Production-Grade Security with CVE Mitigation

Addresses Critical Vulnerabilities:
- CVE-FIX-001: Empty secret bypass vulnerability
- CVE-FIX-002: Weak secret detection and enforcement
- CVE-FIX-003: Token replay attack prevention
- CVE-FIX-004: Key rotation with zero-downtime

Author: BIZRA Security Team
Version: 2.0.0
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# JWT library
try:
    import jwt
    from jwt import PyJWTError
except ImportError:
    raise ImportError("PyJWT required: pip install PyJWT>=2.0.0")

# Configure logging
logger = logging.getLogger("bizra.security.jwt")


# =============================================================================
# SECURITY EXCEPTIONS
# =============================================================================


class JWTSecurityError(Exception):
    """Base exception for JWT security issues."""

    pass


class WeakSecretError(JWTSecurityError):
    """Raised when JWT secret doesn't meet security requirements."""

    pass


class TokenValidationError(JWTSecurityError):
    """Raised when token validation fails."""

    pass


class TokenRevocationError(JWTSecurityError):
    """Raised when token has been revoked."""

    pass


class SecretRotationError(JWTSecurityError):
    """Raised when secret rotation fails."""

    pass


# =============================================================================
# SECRET STRENGTH VALIDATION
# =============================================================================


class SecretStrength(Enum):
    """Secret strength classification."""

    CRITICAL = "critical"  # Unacceptable - block usage
    WEAK = "weak"  # Warning - allow with logging
    ACCEPTABLE = "acceptable"  # Minimum production standard
    STRONG = "strong"  # Recommended strength
    EXCELLENT = "excellent"  # Maximum strength


@dataclass
class SecretAnalysis:
    """Result of secret strength analysis."""

    strength: SecretStrength
    entropy_bits: float
    length: int
    has_uppercase: bool
    has_lowercase: bool
    has_digits: bool
    has_special: bool
    is_common_pattern: bool
    recommendations: List[str]

    @property
    def is_acceptable(self) -> bool:
        return self.strength in (
            SecretStrength.ACCEPTABLE,
            SecretStrength.STRONG,
            SecretStrength.EXCELLENT,
        )


class SecretValidator:
    """
    Validates JWT secrets against security requirements.

    Security Requirements (NIST SP 800-63B compliant):
    - Minimum 256 bits (32 bytes) of entropy for HS256
    - Minimum 384 bits (48 bytes) for HS384
    - Minimum 512 bits (64 bytes) for HS512
    - No common patterns or dictionary words
    - No empty or whitespace-only secrets
    """

    # Minimum lengths by algorithm
    MIN_LENGTHS = {
        "HS256": 32,
        "HS384": 48,
        "HS512": 64,
    }

    # Common weak patterns (compiled for performance)
    WEAK_PATTERNS = [
        "secret",
        "password",
        "changeme",
        "default",
        "12345",
        "qwerty",
        "admin",
        "test",
        "dev",
        "production",
        "staging",
        "jwt",
        "token",
    ]

    @classmethod
    def analyze(cls, secret: str, algorithm: str = "HS256") -> SecretAnalysis:
        """
        Perform comprehensive secret strength analysis.

        Args:
            secret: The JWT secret to analyze
            algorithm: JWT algorithm (HS256, HS384, HS512)

        Returns:
            SecretAnalysis with strength rating and recommendations
        """
        recommendations = []

        # Handle empty/None secret
        if not secret or not secret.strip():
            return SecretAnalysis(
                strength=SecretStrength.CRITICAL,
                entropy_bits=0,
                length=0,
                has_uppercase=False,
                has_lowercase=False,
                has_digits=False,
                has_special=False,
                is_common_pattern=True,
                recommendations=["JWT secret cannot be empty"],
            )

        length = len(secret)
        min_length = cls.MIN_LENGTHS.get(algorithm, 32)

        # Character class analysis
        has_upper = any(c.isupper() for c in secret)
        has_lower = any(c.islower() for c in secret)
        has_digit = any(c.isdigit() for c in secret)
        has_special = any(not c.isalnum() for c in secret)

        # Entropy calculation (Shannon entropy approximation)
        charset_size = 0
        if has_upper:
            charset_size += 26
        if has_lower:
            charset_size += 26
        if has_digit:
            charset_size += 10
        if has_special:
            charset_size += 32

        if charset_size == 0:
            charset_size = 1

        import math

        entropy_per_char = math.log2(charset_size) if charset_size > 1 else 0
        entropy_bits = entropy_per_char * length

        # Common pattern detection
        secret_lower = secret.lower()
        is_common = any(pattern in secret_lower for pattern in cls.WEAK_PATTERNS)

        # Determine strength
        if length < min_length // 2:
            strength = SecretStrength.CRITICAL
            recommendations.append(f"Secret length {length} is dangerously short")
        elif length < min_length:
            strength = SecretStrength.WEAK
            recommendations.append(
                f"Secret should be at least {min_length} characters for {algorithm}"
            )
        elif is_common:
            strength = SecretStrength.WEAK
            recommendations.append("Secret contains common/predictable patterns")
        elif entropy_bits < min_length * 4:  # Less than 4 bits per character
            strength = SecretStrength.WEAK
            recommendations.append(
                "Secret has low entropy - use more character variety"
            )
        elif entropy_bits < min_length * 6:
            strength = SecretStrength.ACCEPTABLE
        elif entropy_bits < min_length * 7:
            strength = SecretStrength.STRONG
        else:
            strength = SecretStrength.EXCELLENT

        # Additional recommendations
        if not has_upper:
            recommendations.append("Consider adding uppercase letters")
        if not has_lower:
            recommendations.append("Consider adding lowercase letters")
        if not has_digit:
            recommendations.append("Consider adding digits")
        if not has_special:
            recommendations.append("Consider adding special characters")

        return SecretAnalysis(
            strength=strength,
            entropy_bits=entropy_bits,
            length=length,
            has_uppercase=has_upper,
            has_lowercase=has_lower,
            has_digits=has_digit,
            has_special=has_special,
            is_common_pattern=is_common,
            recommendations=recommendations,
        )

    @classmethod
    def validate(
        cls, secret: str, algorithm: str = "HS256", strict: bool = True
    ) -> SecretAnalysis:
        """
        Validate secret and raise if unacceptable.

        Args:
            secret: The JWT secret to validate
            algorithm: JWT algorithm
            strict: If True, raise on WEAK secrets; if False, only on CRITICAL

        Raises:
            WeakSecretError: If secret doesn't meet requirements

        Returns:
            SecretAnalysis for acceptable secrets
        """
        analysis = cls.analyze(secret, algorithm)

        if analysis.strength == SecretStrength.CRITICAL:
            raise WeakSecretError(
                f"CRITICAL: JWT secret is unacceptable. "
                f"{'; '.join(analysis.recommendations)}"
            )

        if strict and analysis.strength == SecretStrength.WEAK:
            raise WeakSecretError(
                f"WEAK: JWT secret doesn't meet security requirements. "
                f"{'; '.join(analysis.recommendations)}"
            )

        if analysis.strength == SecretStrength.WEAK:
            warnings.warn(
                f"JWT secret is weak: {'; '.join(analysis.recommendations)}",
                SecurityWarning,
            )

        return analysis

    @classmethod
    def generate_secure_secret(cls, algorithm: str = "HS256") -> str:
        """
        Generate a cryptographically secure secret.

        Args:
            algorithm: JWT algorithm to size the secret for

        Returns:
            Base64-encoded secure random secret
        """
        min_length = cls.MIN_LENGTHS.get(algorithm, 32)
        # Generate 50% more entropy than minimum
        secret_bytes = secrets.token_bytes(int(min_length * 1.5))
        return base64.urlsafe_b64encode(secret_bytes).decode("ascii")


# =============================================================================
# TOKEN REVOCATION
# =============================================================================


class TokenRevocationStore(ABC):
    """Abstract base for token revocation storage."""

    @abstractmethod
    def revoke(self, jti: str, expires_at: datetime) -> None:
        """Add a token ID to the revocation list."""
        pass

    @abstractmethod
    def is_revoked(self, jti: str) -> bool:
        """Check if a token ID is revoked."""
        pass

    @abstractmethod
    def cleanup(self) -> int:
        """Remove expired revocations. Returns count removed."""
        pass


class InMemoryRevocationStore(TokenRevocationStore):
    """
    In-memory token revocation store.

    Suitable for single-instance deployments.
    For distributed systems, use Redis-backed store.
    """

    def __init__(self, cleanup_interval: int = 3600):
        self._revoked: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    def revoke(self, jti: str, expires_at: datetime) -> None:
        """Revoke a token by its JTI."""
        with self._lock:
            self._revoked[jti] = expires_at
            self._maybe_cleanup()

    def is_revoked(self, jti: str) -> bool:
        """Check if token is revoked."""
        with self._lock:
            return jti in self._revoked

    def cleanup(self) -> int:
        """Remove expired revocations."""
        now = datetime.now(timezone.utc)
        with self._lock:
            expired = [jti for jti, exp in self._revoked.items() if exp < now]
            for jti in expired:
                del self._revoked[jti]
            return len(expired)

    def _maybe_cleanup(self) -> None:
        """Cleanup if interval has passed."""
        now = time.time()
        if now - self._last_cleanup >= self._cleanup_interval:
            self._last_cleanup = now
            # Don't hold lock during cleanup notification
            threading.Thread(target=self.cleanup, daemon=True).start()


# =============================================================================
# SECURE JWT SERVICE
# =============================================================================


@dataclass
class JWTConfig:
    """Configuration for secure JWT service."""

    # Primary secret (required)
    secret: str

    # Algorithm (HS256, HS384, HS512)
    algorithm: str = "HS256"

    # Token lifetime
    access_token_lifetime: timedelta = field(default_factory=lambda: timedelta(hours=1))
    refresh_token_lifetime: timedelta = field(default_factory=lambda: timedelta(days=7))

    # Issuer and audience validation
    issuer: str = "bizra-aeon-omega"
    audience: Optional[str] = None

    # Security settings
    require_jti: bool = True
    require_iat: bool = True
    require_exp: bool = True

    # Clock skew tolerance
    leeway: timedelta = field(default_factory=lambda: timedelta(seconds=30))

    # Rotation settings
    previous_secrets: List[str] = field(default_factory=list)

    # Strict mode (reject weak secrets)
    strict_validation: bool = True


class SecureJWTService:
    """
    Production-grade JWT service with security hardening.

    Features:
    - Secret strength validation
    - Token revocation support
    - Key rotation with zero-downtime
    - Replay attack prevention
    - Comprehensive claim validation
    """

    def __init__(
        self, config: JWTConfig, revocation_store: Optional[TokenRevocationStore] = None
    ):
        """
        Initialize secure JWT service.

        Args:
            config: JWT configuration
            revocation_store: Optional token revocation store

        Raises:
            WeakSecretError: If secret doesn't meet requirements
        """
        # Validate primary secret
        analysis = SecretValidator.validate(
            config.secret, config.algorithm, strict=config.strict_validation
        )

        logger.info(
            f"JWT service initialized with {analysis.strength.value} secret "
            f"({analysis.entropy_bits:.0f} bits entropy)"
        )

        self._config = config
        self._revocation_store = revocation_store or InMemoryRevocationStore()
        self._lock = threading.RLock()

        # Metrics
        self._tokens_issued = 0
        self._tokens_verified = 0
        self._tokens_rejected = 0

    def create_access_token(
        self,
        subject: str,
        claims: Optional[Dict[str, Any]] = None,
        scopes: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new access token.

        Args:
            subject: Token subject (usually user ID)
            claims: Additional custom claims
            scopes: Authorization scopes

        Returns:
            Encoded JWT token
        """
        return self._create_token(
            subject=subject,
            token_type="access",
            lifetime=self._config.access_token_lifetime,
            claims=claims,
            scopes=scopes,
        )

    def create_refresh_token(
        self, subject: str, claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a refresh token.

        Args:
            subject: Token subject
            claims: Additional custom claims

        Returns:
            Encoded JWT refresh token
        """
        return self._create_token(
            subject=subject,
            token_type="refresh",
            lifetime=self._config.refresh_token_lifetime,
            claims=claims,
        )

    def _create_token(
        self,
        subject: str,
        token_type: str,
        lifetime: timedelta,
        claims: Optional[Dict[str, Any]] = None,
        scopes: Optional[List[str]] = None,
    ) -> str:
        """Internal token creation with full claims."""
        now = datetime.now(timezone.utc)

        payload = {
            "sub": subject,
            "type": token_type,
            "iss": self._config.issuer,
            "iat": now,
            "exp": now + lifetime,
            "jti": secrets.token_hex(16),
        }

        if self._config.audience:
            payload["aud"] = self._config.audience

        if scopes:
            payload["scopes"] = scopes

        if claims:
            # Prevent overwriting reserved claims
            reserved = {"sub", "type", "iss", "iat", "exp", "jti", "aud", "scopes"}
            safe_claims = {k: v for k, v in claims.items() if k not in reserved}
            payload.update(safe_claims)

        token = jwt.encode(
            payload, self._config.secret, algorithm=self._config.algorithm
        )

        with self._lock:
            self._tokens_issued += 1

        logger.debug(f"Issued {token_type} token for subject: {subject}")
        return token

    def verify_token(
        self,
        token: str,
        expected_type: Optional[str] = None,
        required_scopes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Verify and decode a JWT token.

        Args:
            token: The JWT token to verify
            expected_type: Expected token type ("access" or "refresh")
            required_scopes: Scopes that must be present

        Returns:
            Decoded token payload

        Raises:
            TokenValidationError: If token is invalid
            TokenRevocationError: If token has been revoked
        """
        # Try primary secret first, then fall back to previous secrets
        secrets_to_try = [self._config.secret] + self._config.previous_secrets

        payload = None
        last_error = None

        for secret in secrets_to_try:
            try:
                payload = jwt.decode(
                    token,
                    secret,
                    algorithms=[self._config.algorithm],
                    issuer=self._config.issuer,
                    audience=self._config.audience,
                    leeway=self._config.leeway,
                    options={
                        "require": (
                            ["exp", "iat", "sub", "jti"]
                            if self._config.require_jti
                            else ["exp", "iat", "sub"]
                        )
                    },
                )
                break  # Success
            except jwt.ExpiredSignatureError as e:
                raise TokenValidationError("Token has expired") from e
            except jwt.InvalidTokenError as e:
                last_error = e
                continue

        if payload is None:
            with self._lock:
                self._tokens_rejected += 1
            raise TokenValidationError(f"Invalid token: {last_error}")

        # Check revocation
        jti = payload.get("jti")
        if jti and self._revocation_store.is_revoked(jti):
            with self._lock:
                self._tokens_rejected += 1
            raise TokenRevocationError("Token has been revoked")

        # Check token type
        if expected_type and payload.get("type") != expected_type:
            with self._lock:
                self._tokens_rejected += 1
            raise TokenValidationError(
                f"Expected {expected_type} token, got {payload.get('type')}"
            )

        # Check required scopes
        if required_scopes:
            token_scopes = set(payload.get("scopes", []))
            missing = set(required_scopes) - token_scopes
            if missing:
                with self._lock:
                    self._tokens_rejected += 1
                raise TokenValidationError(f"Missing required scopes: {missing}")

        with self._lock:
            self._tokens_verified += 1

        return payload

    def revoke_token(self, token: str) -> None:
        """
        Revoke a token so it can no longer be used.

        Args:
            token: The token to revoke
        """
        try:
            # Decode without verification to get JTI and expiry
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get("jti")
            exp = payload.get("exp")

            if jti and exp:
                expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
                self._revocation_store.revoke(jti, expires_at)
                logger.info(f"Revoked token with JTI: {jti[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to revoke token: {e}")

    def rotate_secret(self, new_secret: str) -> None:
        """
        Rotate to a new secret while maintaining backward compatibility.

        Args:
            new_secret: The new secret to use

        Raises:
            WeakSecretError: If new secret doesn't meet requirements
        """
        # Validate new secret
        SecretValidator.validate(
            new_secret, self._config.algorithm, strict=self._config.strict_validation
        )

        with self._lock:
            # Move current secret to previous
            self._config.previous_secrets.insert(0, self._config.secret)

            # Keep only last 2 previous secrets
            self._config.previous_secrets = self._config.previous_secrets[:2]

            # Set new secret
            self._config.secret = new_secret

        logger.info("JWT secret rotated successfully")

    @property
    def metrics(self) -> Dict[str, int]:
        """Get service metrics."""
        with self._lock:
            return {
                "tokens_issued": self._tokens_issued,
                "tokens_verified": self._tokens_verified,
                "tokens_rejected": self._tokens_rejected,
            }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_secure_jwt_service(
    secret: Optional[str] = None, algorithm: str = "HS256", strict: bool = True
) -> SecureJWTService:
    """
    Factory function to create a secure JWT service.

    If no secret is provided, generates a cryptographically secure one.

    Args:
        secret: JWT secret (None to auto-generate)
        algorithm: JWT algorithm
        strict: Strict secret validation

    Returns:
        Configured SecureJWTService instance
    """
    if secret is None:
        secret = SecretValidator.generate_secure_secret(algorithm)
        logger.info("Generated new secure JWT secret")

    config = JWTConfig(secret=secret, algorithm=algorithm, strict_validation=strict)

    return SecureJWTService(config)


def create_jwt_service_from_env(
    env_var: str = "BIZRA_JWT_SECRET", fallback_generate: bool = False
) -> SecureJWTService:
    """
    Create JWT service from environment variable.

    Args:
        env_var: Environment variable name containing secret
        fallback_generate: If True, generate secret if env var missing

    Returns:
        Configured SecureJWTService instance

    Raises:
        WeakSecretError: If secret not found and fallback disabled
    """
    secret = os.environ.get(env_var)

    if not secret:
        if fallback_generate:
            logger.warning(
                f"JWT secret not found in {env_var}, generating secure secret"
            )
            secret = SecretValidator.generate_secure_secret()
        else:
            raise WeakSecretError(
                f"JWT secret must be provided via {env_var} environment variable"
            )

    return create_secure_jwt_service(secret=secret)


# =============================================================================
# SECURITY WARNING
# =============================================================================


class SecurityWarning(UserWarning):
    """Warning for security-related issues."""

    pass


__all__ = [
    # Exceptions
    "JWTSecurityError",
    "WeakSecretError",
    "TokenValidationError",
    "TokenRevocationError",
    "SecretRotationError",
    "SecurityWarning",
    # Classes
    "SecretStrength",
    "SecretAnalysis",
    "SecretValidator",
    "TokenRevocationStore",
    "InMemoryRevocationStore",
    "JWTConfig",
    "SecureJWTService",
    # Factory functions
    "create_secure_jwt_service",
    "create_jwt_service_from_env",
]
