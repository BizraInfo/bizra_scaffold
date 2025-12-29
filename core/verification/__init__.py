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
    
Peak Masterpiece v4 - FATE Engine & MAPE-K Loop:
    - FATEEngine: Formal Alignment & Transcendence Engine (Z3 SMT)
    - IhsanVector: 8-dimensional Ihsān scoring
    - CausalDrag: Ω coefficient enforcement
    - MAPEKEngine: Autonomic self-healing loop
    - ConvergenceState: Quantized convergence monitoring
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

# Peak Masterpiece v4 - FATE Engine
from .fate_engine import (
    FATEEngine,
    FATEVerdict,
    FATEReceipt,
    IhsanVector,
    CausalDrag,
    ActionProposal,
    ActionRisk,
    FormalConstraint,
    ConstraintType,
    ConstraintBuilders,
    IHSAN_THRESHOLD,
    IHSAN_WEIGHTS,
    CAUSAL_DRAG_MAX,
    quick_verify,
    compute_ihsan_from_dimensions,
    Z3_AVAILABLE,
)

# Peak Masterpiece v4 - MAPE-K Engine
from .mape_k_engine import (
    MAPEKEngine,
    HealthStatus,
    HealthMetric,
    HealthSnapshot,
    AnalysisResult,
    HealingAction,
    HealingPlan,
    ExecutionResult,
    ActionType,
    ConvergenceState,
    KnowledgeBase,
    KnowledgeEntry,
    KAPPA_DEFAULT,
    LAMBDA_DEFAULT,
    CONVERGENCE_TARGET,
)

__all__ = [
    # Metrics Verifier
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
    # FATE Engine
    "FATEEngine",
    "FATEVerdict",
    "FATEReceipt",
    "IhsanVector",
    "CausalDrag",
    "ActionProposal",
    "ActionRisk",
    "FormalConstraint",
    "ConstraintType",
    "ConstraintBuilders",
    "IHSAN_THRESHOLD",
    "IHSAN_WEIGHTS",
    "CAUSAL_DRAG_MAX",
    "quick_verify",
    "compute_ihsan_from_dimensions",
    "Z3_AVAILABLE",
    # MAPE-K Engine
    "MAPEKEngine",
    "HealthStatus",
    "HealthMetric",
    "HealthSnapshot",
    "AnalysisResult",
    "HealingAction",
    "HealingPlan",
    "ExecutionResult",
    "ActionType",
    "ConvergenceState",
    "KnowledgeBase",
    "KnowledgeEntry",
    "KAPPA_DEFAULT",
    "LAMBDA_DEFAULT",
    "CONVERGENCE_TARGET",
]
