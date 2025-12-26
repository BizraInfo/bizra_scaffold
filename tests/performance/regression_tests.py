"""
BIZRA AEON OMEGA - Performance Regression Testing Framework
CI/CD Integration with Threshold-Based Quality Gates

This module provides:
- Automated performance benchmarks for CI/CD
- Threshold-based pass/fail gates
- Regression detection with statistical analysis
- GitHub Actions / Azure DevOps integration
- Performance trend tracking

Author: BIZRA Performance Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

T = TypeVar("T")


# =============================================================================
# RESULT TYPES
# =============================================================================


class TestResult(Enum):
    """Performance test result."""

    PASSED = auto()
    FAILED = auto()
    DEGRADED = auto()  # Performance degraded but within tolerance
    IMPROVED = auto()  # Performance improved
    SKIPPED = auto()
    ERROR = auto()


@dataclass
class BenchmarkMetrics:
    """Metrics from a benchmark run."""

    name: str
    iterations: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    stddev_ms: float
    p95_ms: float
    p99_ms: float
    throughput: float  # Operations per second
    memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "stddev_ms": round(self.stddev_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "throughput": round(self.throughput, 2),
            "memory_mb": round(self.memory_mb, 2) if self.memory_mb else None,
        }


@dataclass
class ThresholdConfig:
    """Performance threshold configuration."""

    max_mean_ms: float
    max_p95_ms: float
    max_p99_ms: float
    min_throughput: float
    max_memory_mb: Optional[float] = None
    degradation_tolerance: float = 0.1  # 10% degradation allowed
    improvement_threshold: float = 0.05  # 5% improvement is notable


@dataclass
class RegressionResult:
    """Result of regression analysis."""

    benchmark_name: str
    result: TestResult
    current: BenchmarkMetrics
    baseline: Optional[BenchmarkMetrics]
    threshold: ThresholdConfig
    violations: List[str]
    improvements: List[str]
    regression_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "result": self.result.name,
            "current": self.current.to_dict(),
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "violations": self.violations,
            "improvements": self.improvements,
            "regression_percent": round(self.regression_percent, 2),
        }


@dataclass
class PerformanceReport:
    """Complete performance test report."""

    timestamp: datetime
    commit_sha: Optional[str]
    branch: Optional[str]
    results: List[RegressionResult]
    passed: int = 0
    failed: int = 0
    degraded: int = 0
    improved: int = 0

    def __post_init__(self):
        for r in self.results:
            if r.result == TestResult.PASSED:
                self.passed += 1
            elif r.result == TestResult.FAILED:
                self.failed += 1
            elif r.result == TestResult.DEGRADED:
                self.degraded += 1
            elif r.result == TestResult.IMPROVED:
                self.improved += 1

    @property
    def success(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "summary": {
                "passed": self.passed,
                "failed": self.failed,
                "degraded": self.degraded,
                "improved": self.improved,
                "total": len(self.results),
            },
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Performance Regression Report",
            "",
            f"**Timestamp:** {self.timestamp.isoformat()}",
            f"**Commit:** {self.commit_sha or 'N/A'}",
            f"**Branch:** {self.branch or 'N/A'}",
            "",
            "## Summary",
            "",
            f"| Status | Count |",
            f"|--------|-------|",
            f"| ✅ Passed | {self.passed} |",
            f"| ❌ Failed | {self.failed} |",
            f"| ⚠️ Degraded | {self.degraded} |",
            f"| 🚀 Improved | {self.improved} |",
            "",
            "## Details",
            "",
        ]

        for r in self.results:
            icon = {
                TestResult.PASSED: "✅",
                TestResult.FAILED: "❌",
                TestResult.DEGRADED: "⚠️",
                TestResult.IMPROVED: "🚀",
                TestResult.SKIPPED: "⏭️",
                TestResult.ERROR: "💥",
            }.get(r.result, "❓")

            lines.append(f"### {icon} {r.benchmark_name}")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Mean | {r.current.mean_ms:.2f}ms |")
            lines.append(f"| P95 | {r.current.p95_ms:.2f}ms |")
            lines.append(f"| P99 | {r.current.p99_ms:.2f}ms |")
            lines.append(f"| Throughput | {r.current.throughput:.0f} ops/s |")
            lines.append("")

            if r.violations:
                lines.append("**Violations:**")
                for v in r.violations:
                    lines.append(f"- ❌ {v}")
                lines.append("")

            if r.improvements:
                lines.append("**Improvements:**")
                for i in r.improvements:
                    lines.append(f"- 🚀 {i}")
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


class BenchmarkRunner:
    """
    Runs performance benchmarks and collects metrics.
    """

    def __init__(self, warmup_iterations: int = 5, default_iterations: int = 100):
        self._warmup = warmup_iterations
        self._default_iterations = default_iterations

    async def run_async(
        self, name: str, func: Callable[[], Any], iterations: Optional[int] = None
    ) -> BenchmarkMetrics:
        """Run async benchmark."""
        n = iterations or self._default_iterations

        # Warmup
        for _ in range(self._warmup):
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()

        # Measure
        times_ms = []
        for _ in range(n):
            start = time.perf_counter()
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
            elapsed = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed)

        return self._calculate_metrics(name, times_ms)

    def run_sync(
        self, name: str, func: Callable[[], Any], iterations: Optional[int] = None
    ) -> BenchmarkMetrics:
        """Run synchronous benchmark."""
        n = iterations or self._default_iterations

        # Warmup
        for _ in range(self._warmup):
            func()

        # Measure
        times_ms = []
        for _ in range(n):
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed)

        return self._calculate_metrics(name, times_ms)

    def _calculate_metrics(self, name: str, times_ms: List[float]) -> BenchmarkMetrics:
        """Calculate metrics from timing data."""
        sorted_times = sorted(times_ms)
        n = len(times_ms)

        return BenchmarkMetrics(
            name=name,
            iterations=n,
            min_ms=min(times_ms),
            max_ms=max(times_ms),
            mean_ms=statistics.mean(times_ms),
            median_ms=statistics.median(times_ms),
            stddev_ms=statistics.stdev(times_ms) if n > 1 else 0.0,
            p95_ms=sorted_times[int(n * 0.95)],
            p99_ms=sorted_times[int(n * 0.99)],
            throughput=1000 / statistics.mean(times_ms),  # ops/sec
        )


# =============================================================================
# REGRESSION DETECTOR
# =============================================================================


class RegressionDetector:
    """
    Detects performance regressions against baselines and thresholds.
    """

    def __init__(self):
        self._baselines: Dict[str, BenchmarkMetrics] = {}
        self._thresholds: Dict[str, ThresholdConfig] = {}

    def set_baseline(self, name: str, metrics: BenchmarkMetrics) -> None:
        """Set baseline for comparison."""
        self._baselines[name] = metrics

    def set_threshold(self, name: str, threshold: ThresholdConfig) -> None:
        """Set threshold for benchmark."""
        self._thresholds[name] = threshold

    def load_baselines(self, path: Path) -> None:
        """Load baselines from JSON file."""
        if path.exists():
            with open(path) as f:
                data = json.load(f)

            for name, metrics in data.items():
                self._baselines[name] = BenchmarkMetrics(**metrics)

    def save_baselines(self, path: Path) -> None:
        """Save current baselines to JSON file."""
        data = {name: m.to_dict() for name, m in self._baselines.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def check(self, current: BenchmarkMetrics) -> RegressionResult:
        """Check current metrics against baseline and thresholds."""
        name = current.name
        baseline = self._baselines.get(name)
        threshold = self._thresholds.get(name, self._default_threshold())

        violations = []
        improvements = []
        regression_percent = 0.0

        # Check absolute thresholds
        if current.mean_ms > threshold.max_mean_ms:
            violations.append(
                f"Mean {current.mean_ms:.2f}ms exceeds threshold {threshold.max_mean_ms}ms"
            )

        if current.p95_ms > threshold.max_p95_ms:
            violations.append(
                f"P95 {current.p95_ms:.2f}ms exceeds threshold {threshold.max_p95_ms}ms"
            )

        if current.p99_ms > threshold.max_p99_ms:
            violations.append(
                f"P99 {current.p99_ms:.2f}ms exceeds threshold {threshold.max_p99_ms}ms"
            )

        if current.throughput < threshold.min_throughput:
            violations.append(
                f"Throughput {current.throughput:.0f} ops/s below threshold {threshold.min_throughput}"
            )

        # Check against baseline
        if baseline:
            mean_change = (current.mean_ms - baseline.mean_ms) / baseline.mean_ms
            regression_percent = mean_change * 100

            if mean_change > threshold.degradation_tolerance:
                violations.append(
                    f"Mean increased {regression_percent:.1f}% "
                    f"(tolerance: {threshold.degradation_tolerance * 100}%)"
                )
            elif mean_change < -threshold.improvement_threshold:
                improvements.append(f"Mean improved by {-regression_percent:.1f}%")

            p95_change = (current.p95_ms - baseline.p95_ms) / baseline.p95_ms
            if p95_change > threshold.degradation_tolerance:
                violations.append(f"P95 increased {p95_change * 100:.1f}%")
            elif p95_change < -threshold.improvement_threshold:
                improvements.append(f"P95 improved by {-p95_change * 100:.1f}%")

        # Determine result
        if violations:
            result = TestResult.FAILED
        elif improvements:
            result = TestResult.IMPROVED
        elif baseline and regression_percent > 0:
            result = TestResult.DEGRADED
        else:
            result = TestResult.PASSED

        return RegressionResult(
            benchmark_name=name,
            result=result,
            current=current,
            baseline=baseline,
            threshold=threshold,
            violations=violations,
            improvements=improvements,
            regression_percent=regression_percent,
        )

    def _default_threshold(self) -> ThresholdConfig:
        """Default threshold for unregistered benchmarks."""
        return ThresholdConfig(
            max_mean_ms=100.0, max_p95_ms=200.0, max_p99_ms=500.0, min_throughput=10.0
        )


# =============================================================================
# CI/CD INTEGRATION
# =============================================================================


class CIIntegration:
    """
    Integration with CI/CD systems.
    """

    @staticmethod
    def get_commit_sha() -> Optional[str]:
        """Get current commit SHA."""
        return os.environ.get("GITHUB_SHA") or os.environ.get("BUILD_SOURCEVERSION")

    @staticmethod
    def get_branch() -> Optional[str]:
        """Get current branch."""
        ref = os.environ.get("GITHUB_REF") or os.environ.get("BUILD_SOURCEBRANCH")
        if ref and ref.startswith("refs/heads/"):
            return ref[11:]
        return ref

    @staticmethod
    def is_ci() -> bool:
        """Check if running in CI environment."""
        return any(
            [
                os.environ.get("CI"),
                os.environ.get("GITHUB_ACTIONS"),
                os.environ.get("TF_BUILD"),
                os.environ.get("JENKINS_URL"),
            ]
        )

    @staticmethod
    def set_output(name: str, value: str) -> None:
        """Set GitHub Actions output."""
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"{name}={value}\n")

    @staticmethod
    def write_summary(markdown: str) -> None:
        """Write to GitHub Actions job summary."""
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write(markdown)


# =============================================================================
# PERFORMANCE TEST SUITE
# =============================================================================


class PerformanceTestSuite:
    """
    Complete performance testing framework.

    Usage:
        suite = PerformanceTestSuite()

        # Register benchmarks
        suite.add_benchmark(
            "attestation_create",
            create_attestation,
            threshold=ThresholdConfig(max_mean_ms=50, ...)
        )

        # Run tests
        report = await suite.run()

        # Check result
        if not report.success:
            sys.exit(1)
    """

    def __init__(
        self, baseline_path: Optional[Path] = None, report_path: Optional[Path] = None
    ):
        self._runner = BenchmarkRunner()
        self._detector = RegressionDetector()
        self._benchmarks: List[Tuple[str, Callable, int, ThresholdConfig]] = []
        self._baseline_path = baseline_path or Path(".performance/baselines.json")
        self._report_path = report_path or Path(".performance/report.json")

        # Load existing baselines
        self._detector.load_baselines(self._baseline_path)

    def add_benchmark(
        self,
        name: str,
        func: Callable[[], Any],
        iterations: int = 100,
        threshold: Optional[ThresholdConfig] = None,
    ) -> None:
        """Add a benchmark to the suite."""
        t = threshold or ThresholdConfig(
            max_mean_ms=100.0, max_p95_ms=200.0, max_p99_ms=500.0, min_throughput=10.0
        )
        self._benchmarks.append((name, func, iterations, t))
        self._detector.set_threshold(name, t)

    async def run(self, update_baselines: bool = False) -> PerformanceReport:
        """Run all benchmarks and generate report."""
        results = []

        for name, func, iterations, threshold in self._benchmarks:
            try:
                if asyncio.iscoroutinefunction(func):
                    metrics = await self._runner.run_async(name, func, iterations)
                else:
                    metrics = self._runner.run_sync(name, func, iterations)

                result = self._detector.check(metrics)
                results.append(result)

                if update_baselines:
                    self._detector.set_baseline(name, metrics)

            except Exception as e:
                results.append(
                    RegressionResult(
                        benchmark_name=name,
                        result=TestResult.ERROR,
                        current=BenchmarkMetrics(
                            name=name,
                            iterations=0,
                            min_ms=0,
                            max_ms=0,
                            mean_ms=0,
                            median_ms=0,
                            stddev_ms=0,
                            p95_ms=0,
                            p99_ms=0,
                            throughput=0,
                        ),
                        baseline=None,
                        threshold=threshold,
                        violations=[f"Error: {e}"],
                        improvements=[],
                    )
                )

        # Save baselines if updating
        if update_baselines:
            self._detector.save_baselines(self._baseline_path)

        report = PerformanceReport(
            timestamp=datetime.now(timezone.utc),
            commit_sha=CIIntegration.get_commit_sha(),
            branch=CIIntegration.get_branch(),
            results=results,
        )

        # Save report
        self._save_report(report)

        # CI integration
        if CIIntegration.is_ci():
            CIIntegration.set_output("passed", str(report.success).lower())
            CIIntegration.write_summary(report.to_markdown())

        return report

    def _save_report(self, report: PerformanceReport) -> None:
        """Save report to file."""
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)


# =============================================================================
# BIZRA-SPECIFIC BENCHMARKS
# =============================================================================


def create_bizra_performance_suite() -> PerformanceTestSuite:
    """
    Create performance test suite with BIZRA-specific benchmarks.
    """
    suite = PerformanceTestSuite()

    # Attestation creation benchmark
    async def attestation_create():
        from core.architecture.modular_components import CognitiveProcessor, Observation

        processor = CognitiveProcessor()
        obs = Observation(id="bench", data=b"benchmark data " * 100)
        await processor.compute_convergence(obs)

    suite.add_benchmark(
        "attestation_create",
        attestation_create,
        iterations=100,
        threshold=ThresholdConfig(
            max_mean_ms=10.0, max_p95_ms=25.0, max_p99_ms=50.0, min_throughput=100.0
        ),
    )

    # Verification benchmark
    async def verification():
        from core.architecture.modular_components import (
            ConvergenceQuality,
            ConvergenceResult,
            Observation,
            UrgencyLevel,
            VerificationCoordinator,
        )

        coord = VerificationCoordinator()
        obs = Observation(id="bench", data=b"verification test")
        conv = ConvergenceResult(
            clarity=0.9,
            mutual_information=0.8,
            entropy=0.3,
            synergy=0.85,
            quantization_error=0.01,
            quality=ConvergenceQuality.EXCELLENT,
            action={},
        )
        await coord.verify(obs, conv)

    suite.add_benchmark(
        "verification",
        verification,
        iterations=100,
        threshold=ThresholdConfig(
            max_mean_ms=5.0, max_p95_ms=15.0, max_p99_ms=30.0, min_throughput=200.0
        ),
    )

    # Value assessment benchmark
    async def value_assessment():
        from core.architecture.modular_components import (
            ConvergenceQuality,
            ConvergenceResult,
            ValueAssessor,
        )

        assessor = ValueAssessor()
        conv = ConvergenceResult(
            clarity=0.9,
            mutual_information=0.8,
            entropy=0.3,
            synergy=0.85,
            quantization_error=0.01,
            quality=ConvergenceQuality.EXCELLENT,
            action={},
        )
        await assessor.assess(conv)

    suite.add_benchmark(
        "value_assessment",
        value_assessment,
        iterations=100,
        threshold=ThresholdConfig(
            max_mean_ms=2.0, max_p95_ms=5.0, max_p99_ms=10.0, min_throughput=500.0
        ),
    )

    # JWT operations benchmark
    def jwt_operations():
        from core.security.jwt_hardened import create_secure_jwt_service

        service = create_secure_jwt_service()
        token = service.create_access_token(subject="benchmark")
        service.verify_token(token)

    suite.add_benchmark(
        "jwt_operations",
        jwt_operations,
        iterations=100,
        threshold=ThresholdConfig(
            max_mean_ms=5.0, max_p95_ms=10.0, max_p99_ms=25.0, min_throughput=200.0
        ),
    )

    # Memory management benchmark
    def memory_operations():
        from core.memory.memory_management import BoundedList

        bl = BoundedList[int](max_size=1000)
        for i in range(2000):
            bl.append(i)

    suite.add_benchmark(
        "memory_operations",
        memory_operations,
        iterations=100,
        threshold=ThresholdConfig(
            max_mean_ms=1.0, max_p95_ms=2.0, max_p99_ms=5.0, min_throughput=1000.0
        ),
    )

    return suite


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================


async def main():
    """Run performance tests from CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Performance Testing")
    parser.add_argument(
        "--update-baselines",
        action="store_true",
        help="Update baselines with current results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("performance_report.md"),
        help="Output path for markdown report",
    )
    args = parser.parse_args()

    print("Running BIZRA Performance Tests...")

    suite = create_bizra_performance_suite()
    report = await suite.run(update_baselines=args.update_baselines)

    # Write markdown report
    with open(args.output, "w") as f:
        f.write(report.to_markdown())

    print(
        f"\nResults: {report.passed} passed, {report.failed} failed, "
        f"{report.degraded} degraded, {report.improved} improved"
    )

    if not report.success:
        print("\n❌ Performance regression detected!")
        sys.exit(1)

    print("\n✅ All performance tests passed!")


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Results
    "TestResult",
    "BenchmarkMetrics",
    "ThresholdConfig",
    "RegressionResult",
    "PerformanceReport",
    # Runner
    "BenchmarkRunner",
    "RegressionDetector",
    # CI Integration
    "CIIntegration",
    # Suite
    "PerformanceTestSuite",
    "create_bizra_performance_suite",
]
