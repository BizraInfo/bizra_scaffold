# tests/test_policy_deny_keys.py
"""
BIZRA AEON OMEGA - Policy Deny Keys Tests
Tests that policy enforcement correctly denies unauthorized key access.

Security Model:
- Fail-closed: Default deny unless explicitly permitted
- Built-in rules protect: private keys, root key, cross-node access
- Ihsan gate: Access denied if Ihsan metric < threshold
"""

import pytest
import sys
sys.path.insert(0, '.')

from core.security.hsm_abstraction import (
    SecurityPolicy,
    PolicyRule,
    AccessDecision,
    create_node0_policy,
)


class TestPolicyDenyKeys:
    """Test policy enforcement for key access denial."""

    def test_deny_private_key_access(self):
        """Private keys must be denied to all principals (built-in rule)."""
        policy = SecurityPolicy()
        
        # Private key access should be denied by built-in rule
        result = policy.check_access(
            resource="/keys/sat_signing.private",
            operation="read",
            principal="pat_agent_001",
        )
        assert not result.allowed
        assert result.decision == AccessDecision.DENY_POLICY
        assert "deny_private_keys" in result.reason

    def test_deny_root_key_access(self):
        """Root keys must be denied to all contexts (built-in rule)."""
        policy = SecurityPolicy()
        
        result = policy.check_access(
            resource="/keys/root",
            operation="read",
            principal="regular_agent",
        )
        assert not result.allowed
        assert result.decision == AccessDecision.DENY_POLICY
        assert "deny_root_key" in result.reason

    def test_allow_public_key_read(self):
        """Public keys should be readable by all agents with valid Ihsan."""
        policy = create_node0_policy()  # Has public key read rule
        
        result = policy.check_access(
            resource="/keys/sat_verification.public",
            operation="read",
            principal="any_agent",
            ihsan_score=0.95,
        )
        assert result.allowed
        assert result.decision == AccessDecision.ALLOW
        assert "allow_public_key_read" in result.reason

    def test_deny_cross_node_key_access(self):
        """Write/sign operations on other nodes must be denied."""
        policy = SecurityPolicy()
        
        # Cross-node write should be denied
        result = policy.check_access(
            resource="/nodes/other_node/data",
            operation="write",
            principal="node_a_agent",
        )
        assert not result.allowed
        assert result.decision == AccessDecision.DENY_POLICY
        assert "deny_cross_node" in result.reason

    def test_ihsan_gate_enforcement(self):
        """Access denied if Ihsan score below threshold."""
        policy = create_node0_policy()
        
        # Low Ihsan should be denied even for public key reads
        result_low = policy.check_access(
            resource="/keys/verification.public",
            operation="read",
            principal="low_ihsan_agent",
            ihsan_score=0.80,  # Below 0.90 threshold for public reads
        )
        assert not result_low.allowed
        assert result_low.decision == AccessDecision.DENY_IHSAN
        assert "Ihsan score" in result_low.reason
        
        # High Ihsan should be allowed
        result_high = policy.check_access(
            resource="/keys/verification.public",
            operation="read",
            principal="high_ihsan_agent",
            ihsan_score=0.95,  # Above 0.90 threshold
        )
        assert result_high.allowed
        assert result_high.decision == AccessDecision.ALLOW

    def test_default_deny(self):
        """Resources without matching rules should be denied."""
        policy = SecurityPolicy()  # No custom rules
        
        result = policy.check_access(
            resource="/some/unprotected/resource",
            operation="read",
            principal="any_agent",
        )
        assert not result.allowed
        assert result.decision == AccessDecision.DENY
        assert "default deny" in result.reason


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
