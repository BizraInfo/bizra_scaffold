"""
BIZRA AEON OMEGA - Attestation Rate Limiting Integration
Protects Attestation Endpoints from Spam and Abuse

Addresses: Rate Limiting Gaps - Missing protection against attestation spam

This module integrates the hardening.py RateLimiter with attestation
operations to prevent:
- Attestation flooding attacks
- Byzantine spam from malicious nodes
- Resource exhaustion from verification requests

Author: BIZRA Security Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

from core.security.hardening import (
    RateLimiter, RateLimitConfig, RateLimitResult, RateLimitAlgorithm
)
from core.memory.memory_management import BoundedList, LRUCache

logger = logging.getLogger("bizra.security.attestation_ratelimit")

T = TypeVar("T")


# =============================================================================
# ATTESTATION OPERATION TYPES
# =============================================================================

class AttestationOperation(Enum):
    """Types of attestation operations with different limits."""
    CREATE = auto()       # Creating new attestations
    VERIFY = auto()       # Verifying attestations
    BATCH_VERIFY = auto() # Batch verification (expensive)
    QUERY = auto()        # Querying attestation status
    REVOKE = auto()       # Revoking attestations
    ROTATE = auto()       # Key rotation requests


# =============================================================================
# RATE LIMIT CONFIGURATIONS
# =============================================================================

@dataclass
class AttestationRateLimitConfig:
    """
    Rate limit configuration for attestation operations.
    
    Default limits are conservative to prevent abuse while allowing
    legitimate high-throughput operations.
    """
    # Operation-specific limits (per minute)
    create_limit: int = 100         # 100 attestations/min per identity
    verify_limit: int = 1000        # 1000 verifications/min
    batch_verify_limit: int = 10    # 10 batch operations/min (expensive)
    query_limit: int = 500          # 500 queries/min
    revoke_limit: int = 50          # 50 revocations/min
    rotate_limit: int = 5           # 5 rotations/min (rare operation)
    
    # Global limits across all operations
    global_limit: int = 5000        # 5000 total ops/min
    
    # Burst allowance (multiplier)
    burst_multiplier: float = 1.5
    
    # Window size in seconds
    window_seconds: int = 60
    
    # Algorithm for rate limiting
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    
    # Node reputation adjustment
    reputation_enabled: bool = True
    high_reputation_multiplier: float = 2.0
    low_reputation_divisor: float = 4.0
    
    # Trusted nodes (bypass limits)
    trusted_nodes: Set[str] = field(default_factory=set)
    
    # Blocked nodes
    blocked_nodes: Set[str] = field(default_factory=set)


# =============================================================================
# NODE REPUTATION TRACKER
# =============================================================================

@dataclass
class NodeReputation:
    """Reputation score for a node."""
    node_id: str
    score: float = 1.0  # 0.0 = untrusted, 1.0 = neutral, 2.0 = highly trusted
    successful_attestations: int = 0
    failed_attestations: int = 0
    violations: int = 0
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ReputationTracker:
    """
    Tracks node reputation for adaptive rate limiting.
    
    High reputation nodes get higher limits.
    Low reputation nodes get stricter limits.
    """
    
    def __init__(
        self,
        decay_rate: float = 0.01,  # Reputation decay per hour
        violation_penalty: float = 0.2,
        success_bonus: float = 0.01
    ):
        self._decay_rate = decay_rate
        self._violation_penalty = violation_penalty
        self._success_bonus = success_bonus
        self._reputations: LRUCache[str, NodeReputation] = LRUCache(
            max_size=10000,
            ttl_seconds=86400  # 24 hours
        )
    
    def get_reputation(self, node_id: str) -> NodeReputation:
        """Get or create reputation for node."""
        rep = self._reputations.get(node_id)
        if rep is None:
            rep = NodeReputation(node_id=node_id)
            self._reputations.put(node_id, rep)
        
        # Apply time decay
        self._apply_decay(rep)
        
        return rep
    
    def record_success(self, node_id: str) -> None:
        """Record successful operation."""
        rep = self.get_reputation(node_id)
        rep.successful_attestations += 1
        rep.score = min(2.0, rep.score + self._success_bonus)
        rep.last_activity = datetime.now(timezone.utc)
    
    def record_failure(self, node_id: str) -> None:
        """Record failed operation."""
        rep = self.get_reputation(node_id)
        rep.failed_attestations += 1
        # Small penalty for failures
        rep.score = max(0.0, rep.score - self._success_bonus * 0.5)
    
    def record_violation(self, node_id: str) -> None:
        """Record rate limit violation."""
        rep = self.get_reputation(node_id)
        rep.violations += 1
        rep.score = max(0.0, rep.score - self._violation_penalty)
        logger.warning(f"Rate limit violation by {node_id}, new score: {rep.score:.2f}")
    
    def _apply_decay(self, rep: NodeReputation) -> None:
        """Apply time-based decay to reputation."""
        now = datetime.now(timezone.utc)
        hours_elapsed = (now - rep.last_activity).total_seconds() / 3600
        
        if hours_elapsed > 0:
            # Decay towards neutral (1.0)
            if rep.score > 1.0:
                rep.score = max(1.0, rep.score - self._decay_rate * hours_elapsed)
            elif rep.score < 1.0:
                rep.score = min(1.0, rep.score + self._decay_rate * hours_elapsed)


# =============================================================================
# ATTESTATION RATE LIMITER
# =============================================================================

class AttestationRateLimiter:
    """
    Specialized rate limiter for attestation operations.
    
    Features:
    - Per-operation rate limits
    - Per-node tracking
    - Reputation-based limit adjustment
    - Global rate limiting
    - Violation logging and alerting
    """
    
    def __init__(self, config: Optional[AttestationRateLimitConfig] = None):
        self.config = config or AttestationRateLimitConfig()
        
        # Create limiters for each operation type
        self._operation_limiters: Dict[AttestationOperation, RateLimiter] = {}
        self._global_limiter: RateLimiter
        
        self._init_limiters()
        
        # Reputation tracking
        self._reputation = ReputationTracker() if self.config.reputation_enabled else None
        
        # Violation history (bounded)
        self._violations = BoundedList[Dict[str, Any]](max_size=1000)
        
        # Metrics
        self._total_allowed = 0
        self._total_denied = 0
        self._start_time = time.time()
    
    def _init_limiters(self) -> None:
        """Initialize rate limiters for each operation type."""
        limits = {
            AttestationOperation.CREATE: self.config.create_limit,
            AttestationOperation.VERIFY: self.config.verify_limit,
            AttestationOperation.BATCH_VERIFY: self.config.batch_verify_limit,
            AttestationOperation.QUERY: self.config.query_limit,
            AttestationOperation.REVOKE: self.config.revoke_limit,
            AttestationOperation.ROTATE: self.config.rotate_limit,
        }
        
        for op, limit in limits.items():
            config = RateLimitConfig(
                requests_per_window=limit,
                window_seconds=self.config.window_seconds,
                burst_multiplier=self.config.burst_multiplier,
                algorithm=self.config.algorithm,
                by_ip=False,
                by_user=True
            )
            self._operation_limiters[op] = RateLimiter(config)
        
        # Global limiter
        global_config = RateLimitConfig(
            requests_per_window=self.config.global_limit,
            window_seconds=self.config.window_seconds,
            burst_multiplier=self.config.burst_multiplier,
            algorithm=self.config.algorithm
        )
        self._global_limiter = RateLimiter(global_config)
    
    def check_rate_limit(
        self,
        node_id: str,
        operation: AttestationOperation,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RateLimitResult:
        """
        Check if operation is allowed for the given node.
        
        Args:
            node_id: Identifier of the requesting node
            operation: Type of attestation operation
            metadata: Optional metadata for logging
        
        Returns:
            RateLimitResult indicating if operation is allowed
        """
        # Check if node is blocked
        if node_id in self.config.blocked_nodes:
            return self._denied_result("Node is blocked")
        
        # Check if node is trusted (bypass limits)
        if node_id in self.config.trusted_nodes:
            self._total_allowed += 1
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                reset_after=0.0,
                limit=999999,
                current=0
            )
        
        # Get reputation-adjusted limit
        limit_multiplier = self._get_limit_multiplier(node_id)
        
        # Check operation-specific limit
        # RateLimiter.check() expects (ip, user_id, endpoint) - map node_id as user_id
        op_limiter = self._operation_limiters[operation]
        op_result = op_limiter.check(
            user_id=node_id,
            endpoint=operation.name
        )
        
        if not op_result.allowed:
            self._record_violation(node_id, operation, "operation_limit", metadata)
            self._total_denied += 1
            return op_result
        
        # Check global limit
        global_result = self._global_limiter.check(
            user_id=node_id
        )
        
        if not global_result.allowed:
            self._record_violation(node_id, operation, "global_limit", metadata)
            self._total_denied += 1
            return global_result
        
        # Allowed
        self._total_allowed += 1
        if self._reputation:
            self._reputation.record_success(node_id)
        
        return op_result
    
    def _get_limit_multiplier(self, node_id: str) -> float:
        """Get limit multiplier based on reputation."""
        if not self._reputation:
            return 1.0
        
        rep = self._reputation.get_reputation(node_id)
        
        if rep.score >= 1.5:
            return self.config.high_reputation_multiplier
        elif rep.score <= 0.5:
            return 1.0 / self.config.low_reputation_divisor
        
        return 1.0
    
    def _denied_result(self, reason: str) -> RateLimitResult:
        """Create a denied result."""
        return RateLimitResult(
            allowed=False,
            remaining=0,
            reset_after=float(self.config.window_seconds),
            retry_after=float(self.config.window_seconds),
            limit=0,
            current=0
        )
    
    def _record_violation(
        self,
        node_id: str,
        operation: AttestationOperation,
        violation_type: str,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Record a rate limit violation."""
        violation = {
            "node_id": node_id,
            "operation": operation.name,
            "type": violation_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        
        self._violations.append(violation)
        
        if self._reputation:
            self._reputation.record_violation(node_id)
        
        logger.warning(
            f"Rate limit violation: {violation_type} for {node_id} "
            f"on {operation.name}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        elapsed = time.time() - self._start_time
        total = self._total_allowed + self._total_denied
        
        return {
            "total_allowed": self._total_allowed,
            "total_denied": self._total_denied,
            "denial_rate": self._total_denied / max(1, total),
            "requests_per_second": total / max(1, elapsed),
            "violation_count": len(self._violations),
            "uptime_seconds": elapsed
        }
    
    def get_recent_violations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent violations."""
        return self._violations.last_n(limit)
    
    def block_node(self, node_id: str) -> None:
        """Block a node from making requests."""
        self.config.blocked_nodes.add(node_id)
        logger.warning(f"Blocked node: {node_id}")
    
    def unblock_node(self, node_id: str) -> None:
        """Unblock a previously blocked node."""
        self.config.blocked_nodes.discard(node_id)
        logger.info(f"Unblocked node: {node_id}")
    
    def trust_node(self, node_id: str) -> None:
        """Add node to trusted list (bypasses limits)."""
        self.config.trusted_nodes.add(node_id)
        logger.info(f"Added trusted node: {node_id}")


# =============================================================================
# DECORATOR FOR RATE LIMITED OPERATIONS
# =============================================================================

def rate_limited(
    operation: AttestationOperation,
    limiter: AttestationRateLimiter,
    node_id_extractor: Callable[[Any], str]
):
    """
    Decorator to apply rate limiting to attestation operations.
    
    Args:
        operation: Type of attestation operation
        limiter: The rate limiter instance
        node_id_extractor: Function to extract node_id from first argument
    
    Example:
        @rate_limited(
            AttestationOperation.CREATE,
            attestation_limiter,
            lambda request: request.node_id
        )
        async def create_attestation(request: AttestationRequest) -> Attestation:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            node_id = node_id_extractor(args[0] if args else kwargs.get("request"))
            
            result = limiter.check_rate_limit(node_id, operation)
            
            if not result.allowed:
                from core.security.hardening import RateLimitExceeded
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {operation.name}",
                    retry_after=result.retry_after
                )
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            node_id = node_id_extractor(args[0] if args else kwargs.get("request"))
            
            result = limiter.check_rate_limit(node_id, operation)
            
            if not result.allowed:
                from core.security.hardening import RateLimitExceeded
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {operation.name}",
                    retry_after=result.retry_after
                )
            
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# =============================================================================
# ATTESTATION API INTEGRATION
# =============================================================================

class AttestationSecurityMiddleware:
    """
    Middleware for integrating rate limiting with attestation API.
    
    Can be used with FastAPI, Flask, or other frameworks.
    """
    
    def __init__(
        self,
        limiter: Optional[AttestationRateLimiter] = None,
        config: Optional[AttestationRateLimitConfig] = None
    ):
        self.limiter = limiter or AttestationRateLimiter(config)
        
        # Map endpoints to operations
        self._endpoint_map: Dict[str, AttestationOperation] = {
            "/attestation/create": AttestationOperation.CREATE,
            "/attestation/verify": AttestationOperation.VERIFY,
            "/attestation/batch-verify": AttestationOperation.BATCH_VERIFY,
            "/attestation/query": AttestationOperation.QUERY,
            "/attestation/revoke": AttestationOperation.REVOKE,
            "/attestation/rotate": AttestationOperation.ROTATE,
        }
    
    def get_operation(self, path: str) -> Optional[AttestationOperation]:
        """Get operation type for path."""
        for endpoint, op in self._endpoint_map.items():
            if path.startswith(endpoint):
                return op
        return None
    
    async def __call__(
        self,
        node_id: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RateLimitResult:
        """
        Check rate limit for request.
        
        Returns RateLimitResult that can be used to:
        - Block request if not allowed
        - Add rate limit headers to response
        """
        operation = self.get_operation(path)
        
        if operation is None:
            # Unknown operation - apply global limit only
            return self.limiter._global_limiter.check(node_id)
        
        return self.limiter.check_rate_limit(node_id, operation, metadata)
    
    def add_headers(self, result: RateLimitResult, headers: Dict[str, str]) -> None:
        """Add rate limit headers to response."""
        headers["X-RateLimit-Limit"] = str(result.limit)
        headers["X-RateLimit-Remaining"] = str(result.remaining)
        headers["X-RateLimit-Reset"] = str(int(result.reset_after))
        
        if result.retry_after:
            headers["Retry-After"] = str(int(result.retry_after))


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_attestation_limiter(
    create_limit: int = 100,
    verify_limit: int = 1000,
    global_limit: int = 5000,
    reputation_enabled: bool = True
) -> AttestationRateLimiter:
    """Create attestation rate limiter with custom limits."""
    config = AttestationRateLimitConfig(
        create_limit=create_limit,
        verify_limit=verify_limit,
        global_limit=global_limit,
        reputation_enabled=reputation_enabled
    )
    return AttestationRateLimiter(config)


def create_strict_limiter() -> AttestationRateLimiter:
    """Create rate limiter with strict limits for high-security environments."""
    config = AttestationRateLimitConfig(
        create_limit=50,
        verify_limit=500,
        batch_verify_limit=5,
        global_limit=2000,
        burst_multiplier=1.0,  # No burst
        reputation_enabled=True,
        low_reputation_divisor=10.0  # Very strict for low-rep nodes
    )
    return AttestationRateLimiter(config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "AttestationOperation",
    
    # Configuration
    "AttestationRateLimitConfig",
    
    # Reputation
    "NodeReputation",
    "ReputationTracker",
    
    # Rate Limiter
    "AttestationRateLimiter",
    
    # Decorator
    "rate_limited",
    
    # Middleware
    "AttestationSecurityMiddleware",
    
    # Factory
    "create_attestation_limiter",
    "create_strict_limiter",
]
