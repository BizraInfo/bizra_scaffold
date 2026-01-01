#!/usr/bin/env python3
"""
Demo: Sovereign Model Hub
═══════════════════════════════════════════════════════════════════════════════
Demonstrates the unified orchestration of all AI models under sovereign control.

This demo shows how the Sovereign Model Hub:
- Discovers models across Ollama, LLM Studio, CUDA
- Provides intelligent routing based on SNR analysis
- Enables unified sovereign control over fragmented resources
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sovereign.model_hub import SovereignModelHub, ModelRequest


async def demo_sovereign_models():
    """Demonstrate sovereign model orchestration."""
    print("🤖 BIZRA SOVEREIGN MODEL HUB DEMO 🤖")
    print("=" * 60)

    try:
        # Initialize the sovereign model hub
        print("Initializing Sovereign Model Hub...")
        model_hub = SovereignModelHub()

        # Get manifest
        manifest = model_hub.get_sovereign_manifest()
        print(f"Models Discovered: {manifest['total_models']}")
        print(f"Providers: {', '.join(manifest['providers'])}")
        print(f"Capabilities: {', '.join(manifest['capabilities'])}")

        # Test different types of requests
        test_requests = [
            {
                "name": "Simple Chat",
                "task": "Hello, can you help me understand sovereign AI systems?",
                "context": {"domain": "ai_ethics", "urgency": "normal"},
                "complexity": 0.3
            },
            {
                "name": "Code Generation",
                "task": "Write a Python function to calculate Ihsan metrics",
                "context": {"domain": "programming", "language": "python"},
                "complexity": 0.6
            },
            {
                "name": "Complex Analysis",
                "task": "Analyze the architectural implications of Graph of Thoughts for sovereign AI systems",
                "context": {"domain": "ai_architecture", "depth": "deep"},
                "complexity": 0.9
            }
        ]

        print("\nTesting Intelligent Model Routing...")

        for test_case in test_requests:
            print(f"\n--- {test_case['name']} ---")
            print(f"Task: {test_case['task'][:60]}...")
            print(f"Complexity: {test_case['complexity']}")

            # Create model request
            request = ModelRequest(
                task=test_case["task"],
                context=test_case["context"],
                requirements={"complexity_boost": test_case["complexity"]}
            )

            # Execute sovereign task
            print("Routing to optimal model...")
            response = await model_hub.execute_sovereign_task(request)

            if response.success:
                print("✅ SUCCESS")
                print(f"Model Used: {response.model_used}")
                print(f"Execution Time: {response.execution_time_ms:.2f}ms")
                print(f"Response: {response.content[:100]}..." if len(response.content) > 100 else f"Response: {response.content}")
            else:
                print("❌ FAILED")
                print(f"Error: {response.content}")

        # Show routing statistics
        print("\n" + "=" * 60)
        print("ROUTING STATISTICS")
        print("=" * 60)

        routing_stats = {
            "decisions": len(model_hub.routing_engine.routing_history),
            "models_used": len(set(h["selected_model"] for h in model_hub.routing_engine.routing_history))
        }

        print(f"Routing Decisions Made: {routing_stats['decisions']}")
        print(f"Models Utilized: {routing_stats['models_used']}")

        if model_hub.routing_engine.routing_history:
            print("\nRecent Routing Decisions:")
            for i, decision in enumerate(model_hub.routing_engine.routing_history[-3:], 1):
                print(f"  {i}. {decision['selected_model']} (complexity: {decision['request_complexity']:.2f})")

        print("\n" + "=" * 60)
        print("🎯 SOVEREIGN MODEL HUB DEMO COMPLETE")
        print("12+ models now unified under sovereign control")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Sovereign Model Hub demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_model_comparison():
    """Compare model performance across different providers."""
    print("\n🔄 MODEL PERFORMANCE COMPARISON")
    print("=" * 40)

    model_hub = SovereignModelHub()

    # Simple test task
    test_task = "Explain the concept of sovereignty in AI systems in one sentence."
    request = ModelRequest(
        task=test_task,
        context={"domain": "ai_ethics", "style": "concise"},
        requirements={}
    )

    print(f"Test Task: {test_task}")

    # Try to execute on available models
    available_models = list(model_hub.models.keys())
    if not available_models:
        print("No models available for testing")
        return

    print(f"Testing {len(available_models)} available models...")

    results = []
    for model_key in available_models[:3]:  # Test first 3 models
        try:
            print(f"  Testing {model_key}...")
            # Force specific model by temporarily modifying routing
            original_models = model_hub.models.copy()
            model_hub.models = {model_key: original_models[model_key]}

            response = await model_hub.execute_sovereign_task(request)
            results.append({
                "model": model_key,
                "success": response.success,
                "time_ms": response.execution_time_ms,
                "response_length": len(response.content) if response.content else 0
            })

            # Restore models
            model_hub.models = original_models

        except Exception as e:
            results.append({
                "model": model_key,
                "success": False,
                "error": str(e)
            })

    # Display results
    print("\nResults:")
    for result in results:
        status = "✅" if result["success"] else "❌"
        if result["success"]:
            print(f"  {status} {result['model']}: {result['time_ms']:.1f}ms, {result['response_length']} chars")
        else:
            print(f"  {status} {result['model']}: Failed")


if __name__ == "__main__":
    async def main():
        success1 = await demo_sovereign_models()
        if success1:
            await demo_model_comparison()

        sys.exit(0 if success1 else 1)

    asyncio.run(main())
