#!/usr/bin/env python3
"""
Ihsān Ethical Enforcement Validation Script
═════════════════════════════════════════════════════════════════════════════
Validates that HIGH SNR classification requires Ihsān metric ≥ 0.95.

This script MUST pass before every deployment to ensure ethical integrity.

Run: python scripts/validate_ihsan_enforcement.py

"Indeed, Allah loves those who do their work with Ihsān" - Sahih Muslim
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.snr_scorer import SNRLevel, SNRScorer
from core.tiered_verification import ConvergenceResult


async def validate_ihsan_enforcement() -> bool:
    """
    Ensure HIGH SNR requires Ihsān ≥ 0.95.

    This is a CRITICAL ethical constraint that cannot be bypassed.

    Returns:
        True if validation passes, raises AssertionError otherwise
    """
    print("=" * 70)
    print("BIZRA Ethical Constraint Validation")
    print("Verifying: HIGH SNR requires Ihsān ≥ 0.95")
    print("=" * 70)

    scorer = SNRScorer(enable_ethical_constraints=True)

    # Create convergence with excellent metrics (would normally be HIGH SNR)
    convergence = ConvergenceResult(
        clarity=0.95,
        synergy=0.95,
        mutual_information=0.90,
        entropy=0.05,
        quantization_error=0.02,
        quality="OPTIMAL",
        action={"type": "test"},
    )

    # Test case 1: High clarity + low Ihsān → Should downgrade from HIGH
    print("\n[Test 1] High clarity + Low Ihsān (0.90) → Should NOT be HIGH")
    result_low_ihsan = scorer.compute_from_convergence(
        convergence,
        consistency=0.90,
        disagreement=0.05,
        ihsan_metric=0.90,  # BELOW threshold
    )

    if result_low_ihsan.level == SNRLevel.HIGH:
        print("❌ CRITICAL FAILURE: Ethical override failed!")
        print("   System classified as HIGH SNR despite low Ihsān")
        raise AssertionError(
            "CRITICAL: Ethical override failed - HIGH SNR with low Ihsān!"
        )

    if not result_low_ihsan.ethical_override:
        print("❌ FAILURE: ethical_override flag not set")
        raise AssertionError("ethical_override flag should be True when Ihsān < 0.95")

    print(f"   ✅ Correctly downgraded to {result_low_ihsan.level.name}")
    print(f"   ✅ ethical_override flag: {result_low_ihsan.ethical_override}")

    # Test case 2: High clarity + high Ihsān → Should allow HIGH
    print("\n[Test 2] High clarity + High Ihsān (0.96) → Should be HIGH")
    result_high_ihsan = scorer.compute_from_convergence(
        convergence,
        consistency=0.90,
        disagreement=0.05,
        ihsan_metric=0.96,  # ABOVE threshold
    )

    if result_high_ihsan.level != SNRLevel.HIGH:
        print(f"❌ FAILURE: Should be HIGH but got {result_high_ihsan.level.name}")
        print("   Ihsān enforcement may be too strict")
        raise AssertionError(
            f"Ihsān enforcement too strict - got {result_high_ihsan.level.name}"
        )

    if result_high_ihsan.ethical_override:
        print("❌ FAILURE: ethical_override flag incorrectly set")
        raise AssertionError("ethical_override should be False when Ihsān >= 0.95")

    print(f"   ✅ Correctly classified as {result_high_ihsan.level.name}")
    print(f"   ✅ ethical_override flag: {result_high_ihsan.ethical_override}")

    # Test case 3: Verify ethical constraints cannot be disabled by default
    print("\n[Test 3] Ethical constraints enabled by default")
    default_scorer = SNRScorer()

    if not default_scorer.enable_ethical_constraints:
        print("❌ CRITICAL: Ethical constraints not enabled by default!")
        raise AssertionError(
            "SNRScorer must have enable_ethical_constraints=True by default"
        )

    print("   ✅ Ethical constraints enabled by default")

    # All tests passed
    print("\n" + "=" * 70)
    print("✅ IHSĀN ENFORCEMENT VALIDATED")
    print("   HIGH SNR classification requires Ihsān metric ≥ 0.95")
    print("   Ethical constraints are properly enforced")
    print("=" * 70)

    return True


def main():
    """Main entry point."""
    try:
        asyncio.run(validate_ihsan_enforcement())
        print("\n🎯 Validation PASSED - Safe to proceed with deployment")
        sys.exit(0)

    except AssertionError as e:
        print(f"\n🚨 VALIDATION FAILED: {e}")
        print("   DO NOT DEPLOY until this is fixed!")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
