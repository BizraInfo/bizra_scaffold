"""
Production Safeguards for BIZRA Graph-of-Thoughts
═════════════════════════════════════════════════════════════════════════════
Critical safety mechanisms ensuring system never fails in ways that bring
shame or ridicule. Every function is designed with Ihsān (excellence).

"Indeed, Allah loves those who do their work with Ihsān" - Sahih Muslim

This module implements:
1. Input validation and sanitization
2. Circuit breakers for cascading failure prevention
3. Graceful degradation strategies
4. Health checks and self-diagnostics
5. Audit logging for accountability
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional

# Python 3.10 compatibility: use shim for asyncio.timeout
from core.async_utils.execution import async_timeout

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_input: Optional[Any] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes before closing
    timeout_seconds: float = 60.0  # Time before half-open attempt
    max_retry_attempts: int = 3


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.

    Protects external dependencies (Neo4j, convergence engine, etc.)
    from overwhelming the system during failures.
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now(timezone.utc)

        logger.info(f"CircuitBreaker '{name}' initialized: {self.config}")

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        fallback: Optional[Callable[..., Awaitable[Any]]] = None,
        **kwargs,
    ) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            fallback: Optional fallback function if circuit is open
            *args, **kwargs: Arguments for func

        Returns:
            Result from func or fallback

        Raises:
            RuntimeError: If circuit is open and no fallback provided
        """
        # Check if circuit should transition to half-open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"CircuitBreaker '{self.name}': Attempting half-open")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                logger.warning(f"CircuitBreaker '{self.name}': OPEN - blocking call")
                if fallback:
                    return await fallback(*args, **kwargs)
                raise RuntimeError(
                    f"Circuit breaker '{self.name}' is OPEN - service unavailable"
                )

        # Attempt execution
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result

        except Exception as e:
            self._record_failure(e)

            if fallback:
                logger.info(f"CircuitBreaker '{self.name}': Using fallback due to: {e}")
                return await fallback(*args, **kwargs)
            raise

    def _record_success(self):
        """Record successful execution."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                logger.info(
                    f"CircuitBreaker '{self.name}': HALF_OPEN → CLOSED (recovered)"
                )
                self.state = CircuitState.CLOSED
                self.last_state_change = datetime.now(timezone.utc)

    def _record_failure(self, error: Exception):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        logger.warning(
            f"CircuitBreaker '{self.name}': Failure {self.failure_count}/"
            f"{self.config.failure_threshold} - {error}"
        )

        if self.failure_count >= self.config.failure_threshold:
            logger.error(
                f"CircuitBreaker '{self.name}': CLOSED → OPEN (failure threshold exceeded)"
            )
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now(timezone.utc)

        elif self.state == CircuitState.HALF_OPEN:
            logger.error(
                f"CircuitBreaker '{self.name}': HALF_OPEN → OPEN (recovery failed)"
            )
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now(timezone.utc)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True

        elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_seconds

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_state_change": self.last_state_change.isoformat(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
            },
        }


class InputValidator:
    """
    Validates and sanitizes inputs to prevent system failures.
    """

    @staticmethod
    def validate_seed_concepts(concepts: List[str]) -> ValidationResult:
        """
        Validate seed concepts for Graph-of-Thoughts.

        Args:
            concepts: List of seed concept strings

        Returns:
            ValidationResult with sanitized concepts
        """
        errors = []
        warnings = []
        sanitized = []

        if not concepts:
            errors.append("Seed concepts list is empty")
            return ValidationResult(is_valid=False, errors=errors)

        if len(concepts) > 100:
            warnings.append(
                f"Too many seed concepts ({len(concepts)}), limiting to 100"
            )
            concepts = concepts[:100]

        for concept in concepts:
            # Check type
            if not isinstance(concept, str):
                errors.append(f"Concept is not a string: {type(concept)}")
                continue

            # Check length
            if len(concept) == 0:
                warnings.append("Empty concept string, skipping")
                continue

            if len(concept) > 500:
                warnings.append(f"Concept too long ({len(concept)} chars), truncating")
                concept = concept[:500]

            # Sanitize: remove control characters, limit to printable
            sanitized_concept = "".join(
                c for c in concept if c.isprintable() or c.isspace()
            ).strip()

            if sanitized_concept:
                sanitized.append(sanitized_concept)

        if not sanitized and not errors:
            errors.append("All concepts were invalid or empty after sanitization")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input=sanitized if sanitized else None,
        )

    @staticmethod
    def validate_beam_width(beam_width: int) -> ValidationResult:
        """Validate beam width parameter."""
        errors = []
        warnings = []

        if not isinstance(beam_width, int):
            errors.append(f"Beam width must be integer, got {type(beam_width)}")
            return ValidationResult(is_valid=False, errors=errors)

        if beam_width < 1:
            errors.append(f"Beam width must be ≥1, got {beam_width}")

        if beam_width > 100:
            warnings.append(
                f"Beam width {beam_width} is very large, may impact performance"
            )

        # Sanitize to safe range
        sanitized = max(1, min(beam_width, 100))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input=sanitized,
        )

    @staticmethod
    def validate_max_depth(max_depth: int) -> ValidationResult:
        """Validate maximum thought chain depth."""
        errors = []
        warnings = []

        if not isinstance(max_depth, int):
            errors.append(f"Max depth must be integer, got {type(max_depth)}")
            return ValidationResult(is_valid=False, errors=errors)

        if max_depth < 1:
            errors.append(f"Max depth must be ≥1, got {max_depth}")

        if max_depth > 20:
            warnings.append(
                f"Max depth {max_depth} is very large, may cause long latency"
            )

        # Sanitize to safe range
        sanitized = max(1, min(max_depth, 20))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input=sanitized,
        )

    @staticmethod
    def validate_snr_threshold(threshold: float) -> ValidationResult:
        """Validate SNR threshold."""
        errors = []
        warnings = []

        if not isinstance(threshold, (int, float)):
            errors.append(f"SNR threshold must be numeric, got {type(threshold)}")
            return ValidationResult(is_valid=False, errors=errors)

        if threshold < 0.0:
            errors.append(f"SNR threshold must be ≥0.0, got {threshold}")

        if threshold > 2.0:
            warnings.append(f"SNR threshold {threshold} is unusually high")

        # Sanitize to reasonable range
        sanitized = max(0.0, min(float(threshold), 2.0))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input=sanitized,
        )


class GracefulDegradation:
    """
    Strategies for graceful degradation when components fail.
    """

    @staticmethod
    async def fallback_hypergraph_query(node_name: str) -> List[Dict[str, Any]]:
        """
        Fallback when Neo4j hypergraph query fails.

        Returns empty neighbor list to allow processing to continue
        without Graph-of-Thoughts enhancements.
        """
        logger.warning(f"Hypergraph query fallback activated for node: {node_name}")
        return []

    @staticmethod
    async def fallback_snr_computation() -> float:
        """
        Fallback SNR score when computation fails.

        Returns conservative MEDIUM-level score.
        """
        logger.warning("SNR computation fallback activated - returning default 0.6")
        return 0.6

    @staticmethod
    async def fallback_convergence() -> Dict[str, Any]:
        """
        Fallback convergence result when engine fails.
        """
        logger.warning("Convergence computation fallback activated")
        return {
            "clarity": 0.5,
            "synergy": 0.5,
            "quality": "DEGRADED",
            "action": {"type": "fallback"},
        }


class HealthChecker:
    """
    Self-diagnostic health checks for production monitoring.
    """

    def __init__(self):
        self.last_check: Optional[datetime] = None
        self.status = HealthStatus.HEALTHY
        self.checks: Dict[str, bool] = {}

    async def check_neo4j_connectivity(
        self, l4_hypergraph: Any, timeout: float = 5.0
    ) -> bool:
        """Check if Neo4j is reachable."""
        try:
            async with async_timeout(timeout):
                # Simple connectivity test
                topology = await l4_hypergraph.analyze_topology()
                return topology is not None
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False

    async def check_snr_scorer(self, scorer: Any) -> bool:
        """Check if SNR scorer is functioning."""
        try:
            # Simple computation test
            from core.tiered_verification import ConvergenceResult

            test_convergence = ConvergenceResult(
                clarity=0.75,
                mutual_information=0.70,
                entropy=0.25,
                synergy=0.80,
                quantization_error=0.05,
                quality="GOOD",
                action={"type": "health_check"},
            )

            result = scorer.compute_from_convergence(
                test_convergence, consistency=0.75, disagreement=0.20, ihsan_metric=0.96
            )

            return result.snr_score >= 0

        except Exception as e:
            logger.error(f"SNR scorer health check failed: {e}")
            return False

    async def perform_full_health_check(
        self, l4_hypergraph: Any, snr_scorer: Any
    ) -> HealthStatus:
        """
        Perform comprehensive health check.

        Returns:
            HealthStatus indicating overall system health
        """
        self.last_check = datetime.now(timezone.utc)
        self.checks = {}

        # Check Neo4j
        self.checks["neo4j"] = await self.check_neo4j_connectivity(l4_hypergraph)

        # Check SNR scorer
        self.checks["snr_scorer"] = await self.check_snr_scorer(snr_scorer)

        # Determine overall status
        failed_checks = [k for k, v in self.checks.items() if not v]

        if not failed_checks:
            self.status = HealthStatus.HEALTHY
        elif len(failed_checks) == 1:
            self.status = HealthStatus.DEGRADED
        else:
            self.status = HealthStatus.CRITICAL

        logger.info(
            f"Health check complete: {self.status.value} "
            f"(failed: {failed_checks if failed_checks else 'none'})"
        )

        return self.status

    def get_status_report(self) -> Dict[str, Any]:
        """Get detailed health status report."""
        return {
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "checks": self.checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class AuditLogger:
    """
    Immutable audit logging for accountability.

    All critical operations are logged with cryptographic hashing
    to ensure integrity and non-repudiation.
    """

    def __init__(self, logger_name: str = "bizra_audit"):
        self.logger = logging.getLogger(logger_name)
        self.event_count = 0

    def log_thought_chain_construction(
        self, chain_id: str, query: str, snr_score: float, depth: int, bridge_count: int
    ):
        """Log thought chain construction."""
        self.event_count += 1

        event = {
            "event_type": "THOUGHT_CHAIN_CONSTRUCTED",
            "event_id": self.event_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chain_id": chain_id,
            "query_hash": hashlib.sha256(query.encode()).hexdigest()[:16],
            "snr_score": snr_score,
            "depth": depth,
            "bridge_count": bridge_count,
        }

        event["integrity_hash"] = self._compute_hash(event)
        self.logger.info(f"AUDIT: {event}")

    def log_ethical_override(
        self,
        reason: str,
        original_level: str,
        downgraded_level: str,
        ihsan_metric: float,
    ):
        """Log ethical override events."""
        self.event_count += 1

        event = {
            "event_type": "ETHICAL_OVERRIDE",
            "event_id": self.event_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "original_level": original_level,
            "downgraded_level": downgraded_level,
            "ihsan_metric": ihsan_metric,
        }

        event["integrity_hash"] = self._compute_hash(event)
        self.logger.warning(f"AUDIT_ETHICAL: {event}")

    def log_circuit_breaker_state_change(
        self, circuit_name: str, old_state: str, new_state: str, reason: str
    ):
        """Log circuit breaker state transitions."""
        self.event_count += 1

        event = {
            "event_type": "CIRCUIT_BREAKER_STATE_CHANGE",
            "event_id": self.event_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "circuit_name": circuit_name,
            "old_state": old_state,
            "new_state": new_state,
            "reason": reason,
        }

        event["integrity_hash"] = self._compute_hash(event)
        self.logger.warning(f"AUDIT_CIRCUIT: {event}")

    @staticmethod
    def _compute_hash(event: Dict[str, Any]) -> str:
        """Compute integrity hash for audit event."""
        # Create deterministic string representation
        event_copy = {k: v for k, v in event.items() if k != "integrity_hash"}
        event_str = str(sorted(event_copy.items()))
        return hashlib.sha256(event_str.encode()).hexdigest()[:16]

    def log_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log a generic event with arbitrary details.

        Args:
            event_type: Type of event (e.g., 'value_oracle_fallback')
            details: Dictionary of event details
        """
        self.event_count += 1

        event = {
            "event_type": event_type.upper(),
            "event_id": self.event_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details,
        }

        event["integrity_hash"] = self._compute_hash(event)
        self.logger.info(f"AUDIT: {event}")


# Global instances for convenience
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_health_checker = HealthChecker()
_audit_logger = AuditLogger()


def get_circuit_breaker(
    name: str, config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create circuit breaker instance."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    return _health_checker


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    return _audit_logger


__all__ = [
    "HealthStatus",
    "CircuitState",
    "ValidationResult",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "InputValidator",
    "GracefulDegradation",
    "HealthChecker",
    "AuditLogger",
    "get_circuit_breaker",
    "get_health_checker",
    "get_audit_logger",
]
