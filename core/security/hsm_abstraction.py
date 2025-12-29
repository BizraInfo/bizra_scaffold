# core/security/hsm_abstraction.py
"""
BIZRA AEON OMEGA - Security Policy Abstraction
Node0-Exclusive Security Enforcement with Ihsan Gate

This module provides:
- SecurityPolicy: Access control enforcement with Ihsan compliance
- AccessResult: Structured access decision with audit trail
- PolicyRule: Declarative access rule definitions

The security model follows Ihsan principles:
1. Fail-closed: Default deny unless explicitly permitted
2. Least privilege: Minimum access required for operation
3. Auditability: All access decisions logged with context
4. Ihsan gate: Access denied if Ihsan metric < threshold

Author: BIZRA Security Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("bizra.security.policy")


class AccessDecision(Enum):
    """Access control decision."""
    ALLOW = auto()
    DENY = auto()
    DENY_IHSAN = auto()  # Denied due to Ihsan threshold
    DENY_POLICY = auto()  # Denied by explicit policy rule


@dataclass
class AccessResult:
    """Structured result of access control check."""
    decision: AccessDecision
    resource: str
    operation: str
    principal: str
    reason: str
    ihsan_score: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def allowed(self) -> bool:
        """Whether access was allowed."""
        return self.decision == AccessDecision.ALLOW
    
    def to_audit_dict(self) -> Dict[str, Any]:
        """Convert to audit log format."""
        return {
            "decision": self.decision.name,
            "resource": self.resource,
            "operation": self.operation,
            "principal": self.principal,
            "reason": self.reason,
            "ihsan_score": self.ihsan_score,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PolicyRule:
    """Declarative access control rule."""
    name: str
    resource_pattern: str  # Regex pattern for resource matching
    operations: Set[str]   # Allowed operations: read, write, sign, etc.
    principals: Set[str]   # Allowed principals (or "*" for any)
    deny: bool = False     # If True, this is a deny rule (takes precedence)
    ihsan_threshold: float = 0.95  # Minimum Ihsan score required
    
    def matches(self, resource: str, operation: str, principal: str) -> bool:
        """Check if rule applies to this access request."""
        if not re.match(self.resource_pattern, resource):
            return False
        if operation not in self.operations and "*" not in self.operations:
            return False
        if principal not in self.principals and "*" not in self.principals:
            return False
        return True


class SecurityPolicy:
    """
    Central security policy enforcement with Ihsan compliance gate.
    
    Follows fail-closed model: access denied unless explicitly permitted.
    Ihsan gate: Even permitted access denied if Ihsan < threshold.
    """
    
    DEFAULT_IHSAN_THRESHOLD = 0.95
    
    def __init__(
        self,
        rules: Optional[List[PolicyRule]] = None,
        default_ihsan_threshold: float = 0.95,
    ):
        """
        Initialize security policy.
        
        Args:
            rules: List of policy rules (deny rules evaluated first)
            default_ihsan_threshold: Default Ihsan threshold if not in rule
        """
        self._rules: List[PolicyRule] = rules or []
        self._default_ihsan_threshold = default_ihsan_threshold
        self._audit_log: List[AccessResult] = []
        
        # Built-in deny rules for sensitive resources
        self._builtin_deny_rules = [
            PolicyRule(
                name="deny_private_keys",
                resource_pattern=r"^/keys/.*\.private$",
                operations={"*"},
                principals={"*"},
                deny=True,
            ),
            PolicyRule(
                name="deny_root_key",
                resource_pattern=r"^/keys/root$",
                operations={"*"},
                principals={"*"},
                deny=True,
            ),
            PolicyRule(
                name="deny_cross_node",
                resource_pattern=r"^/nodes/(?!self/).*",
                operations={"write", "sign", "delete"},
                principals={"*"},
                deny=True,
            ),
        ]
    
    def add_rule(self, rule: PolicyRule) -> None:
        """Add a policy rule."""
        self._rules.append(rule)
    
    def check_access(
        self,
        resource: str,
        operation: str,
        principal: str,
        ihsan_score: float = 1.0,
    ) -> AccessResult:
        """
        Check if access should be permitted.
        
        Evaluation order:
        1. Built-in deny rules (highest precedence)
        2. Custom deny rules
        3. Custom allow rules
        4. Ihsan threshold check
        5. Default deny
        
        Args:
            resource: Resource path being accessed
            operation: Operation type (read, write, sign, etc.)
            principal: Identity requesting access
            ihsan_score: Current Ihsan metric (0.0-1.0)
        
        Returns:
            AccessResult with decision and reason
        """
        # Phase 1: Check built-in deny rules
        for rule in self._builtin_deny_rules:
            if rule.matches(resource, operation, principal):
                result = AccessResult(
                    decision=AccessDecision.DENY_POLICY,
                    resource=resource,
                    operation=operation,
                    principal=principal,
                    reason=f"Denied by built-in rule: {rule.name}",
                    ihsan_score=ihsan_score,
                )
                self._log_access(result)
                return result
        
        # Phase 2: Check custom deny rules
        deny_rules = [r for r in self._rules if r.deny]
        for rule in deny_rules:
            if rule.matches(resource, operation, principal):
                result = AccessResult(
                    decision=AccessDecision.DENY_POLICY,
                    resource=resource,
                    operation=operation,
                    principal=principal,
                    reason=f"Denied by policy rule: {rule.name}",
                    ihsan_score=ihsan_score,
                )
                self._log_access(result)
                return result
        
        # Phase 3: Check allow rules
        allow_rules = [r for r in self._rules if not r.deny]
        for rule in allow_rules:
            if rule.matches(resource, operation, principal):
                # Phase 4: Ihsan threshold check
                threshold = rule.ihsan_threshold
                if ihsan_score < threshold:
                    result = AccessResult(
                        decision=AccessDecision.DENY_IHSAN,
                        resource=resource,
                        operation=operation,
                        principal=principal,
                        reason=f"Ihsan score {ihsan_score:.3f} < threshold {threshold:.3f}",
                        ihsan_score=ihsan_score,
                    )
                    self._log_access(result)
                    return result
                
                # Access permitted
                result = AccessResult(
                    decision=AccessDecision.ALLOW,
                    resource=resource,
                    operation=operation,
                    principal=principal,
                    reason=f"Permitted by rule: {rule.name}",
                    ihsan_score=ihsan_score,
                )
                self._log_access(result)
                return result
        
        # Phase 5: Default deny
        result = AccessResult(
            decision=AccessDecision.DENY,
            resource=resource,
            operation=operation,
            principal=principal,
            reason="No matching allow rule (default deny)",
            ihsan_score=ihsan_score,
        )
        self._log_access(result)
        return result
    
    def _log_access(self, result: AccessResult) -> None:
        """Log access decision for audit."""
        self._audit_log.append(result)
        log_level = logging.INFO if result.allowed else logging.WARNING
        logger.log(
            log_level,
            "Access %s: %s on %s by %s - %s",
            result.decision.name,
            result.operation,
            result.resource,
            result.principal,
            result.reason,
        )
    
    def get_audit_log(self) -> List[AccessResult]:
        """Get access audit log."""
        return self._audit_log.copy()
    
    def clear_audit_log(self) -> None:
        """Clear audit log (use with caution)."""
        self._audit_log.clear()


# Convenience factory for common policy configurations
def create_node0_policy() -> SecurityPolicy:
    """Create security policy for Node0 operations."""
    policy = SecurityPolicy(default_ihsan_threshold=0.98)
    
    # Allow public key reads for all principals
    policy.add_rule(PolicyRule(
        name="allow_public_key_read",
        resource_pattern=r"^/keys/.*\.public$",
        operations={"read"},
        principals={"*"},
        ihsan_threshold=0.90,  # Lower threshold for public reads
    ))
    
    # Allow self-node operations for authenticated principals
    policy.add_rule(PolicyRule(
        name="allow_self_node_ops",
        resource_pattern=r"^/nodes/self/.*",
        operations={"read", "write", "sign"},
        principals={"*"},
        ihsan_threshold=0.95,
    ))
    
    return policy
