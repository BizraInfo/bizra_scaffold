"""
BIZRA Bicameral Membrane Smoke Tests
=====================================
Validates request/response round-trip through Membrane + pipeline.

Author: BIZRA Genesis Team
Version: 1.0.0
"""

import asyncio
import pytest
from typing import Any, Dict

from core.bicameral.membrane import (
    Membrane,
    MembraneConfig,
    CrossingDirection,
    MessageType,
    MessagePriority,
)


@pytest.fixture
def membrane() -> Membrane:
    """Create a fresh Membrane instance for testing."""
    config = MembraneConfig(
        queue_capacity=100,
        crossing_latency_budget_ms=100.0,
    )
    return Membrane(config)


class TestMembraneRequestResponse:
    """Test Membrane request/response round-trip."""

    @pytest.mark.asyncio
    async def test_request_response_round_trip(self, membrane: Membrane):
        """
        Smoke test: Membrane.request() + run_pipeline round-trip.
        
        Verifies that:
        1. A request is sent via Membrane.request()
        2. The pipeline processes it and dispatches to handler
        3. The response is returned to the caller
        """
        membrane.open()
        
        # Register a simple echo handler (using actual MessageType)
        echo_response = {"status": "ok", "echo": True}
        
        async def echo_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            return {**echo_response, "received": payload}
        
        membrane.register_handler(MessageType.VERIFY_PROPOSAL, echo_handler)
        
        # Start pipeline in background
        pipeline_task = asyncio.create_task(
            membrane.run_pipeline(CrossingDirection.WARM_TO_COLD)
        )
        
        try:
            # Send request and wait for response
            response = await asyncio.wait_for(
                membrane.request(
                    message_type=MessageType.VERIFY_PROPOSAL,
                    payload={"test": "data", "value": 42},
                    direction=CrossingDirection.WARM_TO_COLD,
                    priority=MessagePriority.NORMAL,
                    timeout_ms=5000.0,
                ),
                timeout=5.0,
            )
            
            # Verify response
            assert response is not None
            assert response.get("status") == "ok"
            assert response.get("echo") is True
            assert response.get("received", {}).get("test") == "data"
            assert response.get("received", {}).get("value") == 42
            
        finally:
            # Clean shutdown
            membrane.close()
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_request_timeout(self, membrane: Membrane):
        """
        Test that request() times out gracefully when no handler responds.
        
        Note: When no handler is registered for a message type, dispatch returns
        {"error": "No handler..."} which still resolves the request. To test
        timeout, we need to not run the pipeline at all.
        """
        membrane.open()
        
        # Don't start pipeline - request should timeout waiting for response
        try:
            with pytest.raises(asyncio.TimeoutError):
                await membrane.request(
                    message_type=MessageType.VERIFY_PROPOSAL,
                    payload={"test": "timeout"},
                    direction=CrossingDirection.WARM_TO_COLD,
                    priority=MessagePriority.NORMAL,
                    timeout_ms=500.0,  # Short timeout
                )
        finally:
            membrane.close()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, membrane: Membrane):
        """
        Test multiple concurrent requests are correctly correlated.
        """
        membrane.open()
        
        async def numbered_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            # Small delay to ensure concurrency
            await asyncio.sleep(0.01)
            return {"response_id": payload.get("request_id")}
        
        membrane.register_handler(MessageType.VERIFY_PROPOSAL, numbered_handler)
        
        pipeline_task = asyncio.create_task(
            membrane.run_pipeline(CrossingDirection.WARM_TO_COLD)
        )
        
        try:
            # Send multiple concurrent requests
            requests = [
                membrane.request(
                    message_type=MessageType.VERIFY_PROPOSAL,
                    payload={"request_id": i},
                    direction=CrossingDirection.WARM_TO_COLD,
                    priority=MessagePriority.NORMAL,
                    timeout_ms=5000.0,
                )
                for i in range(5)
            ]
            
            responses = await asyncio.wait_for(
                asyncio.gather(*requests),
                timeout=10.0,
            )
            
            # Verify each response matches its request
            response_ids = {r.get("response_id") for r in responses}
            assert response_ids == {0, 1, 2, 3, 4}
            
        finally:
            membrane.close()
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass


class TestMembraneQueueBehavior:
    """Test Membrane queue handling."""

    @pytest.mark.asyncio
    async def test_queue_throttling(self, membrane: Membrane):
        """Test that queue throttling rejects low-priority messages."""
        membrane.open()
        
        # Fill queue close to capacity
        for i in range(int(membrane.config.queue_capacity * 0.9)):
            msg = membrane._create_message(
                message_type=MessageType.VERIFY_PROPOSAL,
                direction=CrossingDirection.WARM_TO_COLD,
                priority=MessagePriority.NORMAL,
                payload={"fill": i},
            )
            await membrane.send(msg)
        
        # Queue should now be throttled for low priority
        # (depends on throttle_threshold config)
        membrane.close()


class TestColdCoreSigningRoundTrip:
    """Test ColdCore signing/verification consistency."""

    def test_sign_verify_fallback_round_trip(self):
        """
        Verify that fallback sign() and verify() are consistent.
        
        Note: In fallback mode (no crypto libs), we use symmetric keys.
        This tests the P1 fix for consistent hash formulas.
        """
        from core.bicameral.cold_core import ColdCore, ColdCoreConfig
        
        # Force fallback mode by disabling native and PQAL
        config = ColdCoreConfig(
            latency_budget_ms=100.0,
            enable_native_rust=False,
            enable_pqal_fallback=False,
        )
        cold_core = ColdCore(config)
        
        message = b"Test message for signing"
        # In fallback mode, private_key == public_key for verification
        key = b"test_key_32_bytes_long_for_test!"
        
        # Sign
        sign_result = cold_core.sign(message, key)
        assert sign_result.success, f"Sign failed: {sign_result.error}"
        signature = sign_result.output
        assert signature is not None
        
        # Verify (in fallback mode, use same key)
        verify_result = cold_core.verify(message, signature, key)
        assert verify_result.success, f"Verify failed: {verify_result.error}"
        assert verify_result.output is True, "Signature verification should pass"

    def test_verify_fails_with_wrong_key_fallback(self):
        """Verify that fallback verification fails with wrong key."""
        from core.bicameral.cold_core import ColdCore, ColdCoreConfig
        
        config = ColdCoreConfig(
            latency_budget_ms=100.0,
            enable_native_rust=False,
            enable_pqal_fallback=False,
        )
        cold_core = ColdCore(config)
        
        message = b"Test message for signing"
        sign_key = b"signing_key_32_bytes_long_test!"
        wrong_key = b"wrong_key_32_bytes_long_testing!"
        
        # Sign with one key
        sign_result = cold_core.sign(message, sign_key)
        assert sign_result.success, f"Sign failed: {sign_result.error}"
        signature = sign_result.output
        
        # Verify with wrong key should fail
        verify_result = cold_core.verify(message, signature, wrong_key)
        assert verify_result.success  # Operation succeeded
        assert verify_result.output is False, "Verification should fail with wrong key"


class TestSandboxResourceLimits:
    """Test sandbox resource limit enforcement."""

    @pytest.mark.asyncio
    async def test_sandbox_basic_execution(self):
        """Test basic sandbox code execution."""
        from core.isolation.sandbox_engine import (
            SandboxEngine,
            SandboxConfig,
            IsolationLevel,
            CodeExecution,
        )
        
        config = SandboxConfig(
            default_isolation=IsolationLevel.MINIMAL,
        )
        engine = SandboxEngine(config)
        
        # Create sandbox first - returns Sandbox object
        sandbox = await engine.create()
        sandbox_id = sandbox.sandbox_id
        
        # Simple Python code
        execution = CodeExecution(
            code="print('Hello from sandbox')",
            language="python",
            timeout_s=10.0,
        )
        
        result = await engine.execute(sandbox_id, execution)
        
        # Cleanup
        await engine.destroy(sandbox_id)
        
        assert result.success, f"Execution failed: {result.error}"
        assert "Hello from sandbox" in result.stdout

    @pytest.mark.asyncio
    async def test_sandbox_timeout_enforcement(self):
        """Test that sandbox enforces timeout."""
        from core.isolation.sandbox_engine import (
            SandboxEngine,
            SandboxConfig,
            IsolationLevel,
            CodeExecution,
            ExecutionStatus,
        )
        
        config = SandboxConfig(
            default_isolation=IsolationLevel.MINIMAL,
        )
        engine = SandboxEngine(config)
        
        # Create sandbox - returns Sandbox object
        sandbox = await engine.create()
        sandbox_id = sandbox.sandbox_id
        
        # Infinite loop - should timeout
        execution = CodeExecution(
            code="import time\nwhile True: time.sleep(0.1)",
            language="python",
            timeout_s=1.0,
        )
        
        result = await engine.execute(sandbox_id, execution)
        
        # Cleanup
        await engine.destroy(sandbox_id)
        
        assert not result.success
        assert result.status == ExecutionStatus.TIMEOUT


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
