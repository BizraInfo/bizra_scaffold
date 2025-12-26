"""
BIZRA Agents Module
═══════════════════════════════════════════════════════════════════════════════
Dual-agent architecture implementation per PROTOCOL.md Section 6.

- PAT (Prover/Builder): Constructs and signs proposals, cannot commit
- SAT (Verifier/Governor): Verifies and commits, issues receipts

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from core.agents.pat import PATAgent
from core.agents.sat import SATAgent

__all__ = [
    "PATAgent",
    "SATAgent",
]
