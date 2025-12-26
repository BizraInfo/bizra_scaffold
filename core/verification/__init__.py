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
    - ClaimVerification: Verification result for a single claim
    - LocMetrics: Lines of code breakdown metrics
    - TestMetrics: Test execution results
    - CoverageMetrics: Code coverage metrics
    - PerformanceMetrics: Performance benchmark results
    - GraphMetrics: Knowledge graph metrics
    - HealthScorecard: Health scorecard with Ihsān dimensions
"""

from .metrics_verifier import (
    ClaimVerification,
    CoverageMetrics,
    ExecutionProfile,
    GraphMetrics,
    HealthScorecard,
    LocMetrics,
    MetricsReceipt,
    MetricsVerifier,
    PerformanceMetrics,
    TestMetrics,
    VerificationMode,
    VerificationState,
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
