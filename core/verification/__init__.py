"""
BIZRA Verification Module
═════════════════════════════════════════════════════════════════════════════

This module provides verification and metrics collection for BIZRA.

Exports:
    - MetricsVerifier: Main verification engine
    - MetricsReceipt: Receipt data structure
    - VerificationState: State enum (HYPOTHESIS, PENDING, VERIFIED, FAIL_CLOSED)
    - VerificationMode: Mode enum (metrics, full, perf, determinism)
    - ExecutionProfile: Profile enum (ci, dev, prod, benchmark)
"""

from .metrics_verifier import (
    MetricsVerifier,
    MetricsReceipt,
    VerificationState,
    VerificationMode,
    ExecutionProfile,
    ClaimVerification,
    LocMetrics,
    TestMetrics,
    CoverageMetrics,
    PerformanceMetrics,
    GraphMetrics,
    HealthScorecard,
)

__all__ = [
    "MetricsVerifier",
    "MetricsReceipt",
    "VerificationState",
    "VerificationMode",
    "ExecutionProfile",
    "ClaimVerification",
    "LocMetrics",
    "TestMetrics",
    "CoverageMetrics",
    "PerformanceMetrics",
    "GraphMetrics",
    "HealthScorecard",
]
