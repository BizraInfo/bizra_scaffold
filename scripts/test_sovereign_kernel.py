#!/usr/bin/env python3
"""
Test script for the Sovereign Kernel.

This script demonstrates the activation and operation of BIZRA's sovereign kernel.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sovereign.kernel import initialize_sovereign_kernel, execute_sovereign_command, get_kernel_status


async def test_sovereign_kernel():
    """Test the sovereign kernel functionality."""
    print("🔥 BIZRA SOVEREIGN KERNEL TEST 🔥")
    print("=" * 60)

    try:
        # Initialize sovereignty
        print("Initializing sovereign kernel...")
        success = await initialize_sovereign_kernel()

        if not success:
            print("❌ Sovereign kernel initialization failed")
            return False

        print("✅ Sovereign kernel initialized")

        # Get initial status
        status = get_kernel_status()
        print(f"Kernel Version: {status['kernel_version']}")
        print(f"Sovereign Architect: {status['sovereign_architect']}")
        print(f"Sovereignty Status: {'ESTABLISHED' if status['sovereignty_established'] else 'FAILED'}")

        # Test sovereign commands
        print("\nTesting sovereign commands...")

        # Test boundary verification
        print("Testing boundary awareness...")
        boundary_result = await execute_sovereign_command("sovereign:boundary")
        print(f"✓ Boundary Status: {boundary_result.get('sovereignty_status', 'UNKNOWN')}")

        # Test resource orchestration
        print("Testing resource orchestration...")
        resource_result = await execute_sovereign_command("system:resources")
        resources = resource_result.get('resources_controlled', {})
        print(f"✓ Resources Controlled: {resources}")

        # Test model discovery
        print("Testing model discovery...")
        model_result = await execute_sovereign_command("model:status")
        available = model_result.get('available_models', 0)
        print(f"✓ Models Available: {available}")

        # Test sovereignty verification
        print("Testing sovereignty verification...")
        verify_result = await execute_sovereign_command("sovereign:verify")
        sovereignty_ok = verify_result.get('sovereignty_verified', False)
        print(f"✓ Sovereignty Verified: {sovereignty_ok}")

        # Get final status
        final_status = get_kernel_status()
        print("\nFinal Kernel Status:")
        print(f"  Operations Processed: {final_status['operations_processed']}")
        print(f"  Sovereignty Verifications: {final_status['sovereignty_verifications']}")
        print(f"  Autonomous Mode: {final_status['autonomous_mode']}")

        print("\n" + "=" * 60)
        print("🎯 SOVEREIGN KERNEL TEST COMPLETE")
        print("BIZRA is now operating under sovereign control")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Sovereign kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_sovereign_kernel())
    sys.exit(0 if success else 1)
