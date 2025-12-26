"""
Graph-of-Thoughts Demo
═════════════════════════════════════════════════════════════════════════════
Demonstrates the enhanced cognitive processing with Graph-of-Thoughts,
SNR scoring, and interdisciplinary reasoning.

This example shows:
1. Setting up domain-aware knowledge graph
2. Seeding with interdisciplinary concepts
3. Running graph-of-thoughts reasoning
4. Analyzing SNR quality metrics
5. Discovering domain bridges
"""

import asyncio
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# BIZRA imports
from core.enhanced_cognitive_integration import (
    EnhancedCognitiveProcessor,
    process_with_graph_of_thoughts,
)
from core.layers.memory_layers_v2 import L4SemanticHyperGraphV2
from core.narrative_compiler import NarrativeCompiler
from core.observability import MetricsCollector
from core.tiered_verification import QuantizedConvergence
from core.ultimate_integration import Observation, UrgencyLevel


async def setup_demo_knowledge_graph(l4: L4SemanticHyperGraphV2):
    """
    Populate knowledge graph with interdisciplinary demo data.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Building Demo Knowledge Graph")
    print("=" * 70)

    # Math domain concepts
    await l4.create_hyperedge(
        nodes=["Optimization", "GradientDescent", "LocalMinima"],
        relation="MATHEMATICAL_TECHNIQUE",
        domain_tags=["math", "computer_science"],
    )

    await l4.create_hyperedge(
        nodes=["Entropy", "InformationTheory", "Uncertainty"],
        relation="MEASURES",
        domain_tags=["math", "physics"],
    )

    # Physics domain concepts
    await l4.create_hyperedge(
        nodes=["ThermodynamicEquilibrium", "Entropy", "FreeEnergy"],
        relation="THERMODYNAMIC_STATE",
        domain_tags=["physics"],
    )

    await l4.create_hyperedge(
        nodes=["FreeEnergy", "Optimization", "MinimumEnergyState"],
        relation="ENERGY_MINIMIZATION",
        domain_tags=["physics", "math"],
    )

    # Economics domain concepts
    await l4.create_hyperedge(
        nodes=["MarketEquilibrium", "SupplyDemand", "PriceOptimization"],
        relation="ECONOMIC_PRINCIPLE",
        domain_tags=["economics"],
    )

    await l4.create_hyperedge(
        nodes=["PriceOptimization", "Optimization", "UtilityMaximization"],
        relation="OPTIMIZATION_PROBLEM",
        domain_tags=["economics", "math"],
    )

    # Ethics domain concepts
    await l4.create_hyperedge(
        nodes=["ConsequentialEthics", "UtilityMaximization", "Harm"],
        relation="ETHICAL_FRAMEWORK",
        domain_tags=["ethics", "economics"],
    )

    await l4.create_hyperedge(
        nodes=["Ihsan", "Excellence", "Benevolence"],
        relation="VIRTUE_ETHICS",
        domain_tags=["ethics"],
    )

    # Psychology domain concepts
    await l4.create_hyperedge(
        nodes=["CognitiveLoad", "Attention", "WorkingMemory"],
        relation="COGNITIVE_MECHANISM",
        domain_tags=["psychology"],
    )

    await l4.create_hyperedge(
        nodes=["Attention", "InformationTheory", "SignalToNoise"],
        relation="INFORMATION_PROCESSING",
        domain_tags=["psychology", "math"],
    )

    # Cross-domain bridges (explicit)
    await l4.create_hyperedge(
        nodes=["FreeEnergy", "InformationTheory", "Prediction"],
        relation="FREE_ENERGY_PRINCIPLE",
        domain_tags=["physics", "math", "psychology"],
    )

    await l4.create_hyperedge(
        nodes=["MarketEquilibrium", "ThermodynamicEquilibrium", "Stability"],
        relation="EQUILIBRIUM_ANALOGY",
        domain_tags=["economics", "physics"],
    )

    print("✅ Created 12 hyperedges across 5 domains")
    print("   Domains: math, physics, economics, ethics, psychology")

    # Analyze topology
    topology = await l4.analyze_topology()
    print(f"\n📊 Knowledge Graph Statistics:")
    print(f"   Nodes: {topology.get('node_count', 0)}")
    print(f"   Edges: {topology.get('edge_count', 0)}")
    print(f"   Domain Bridges: {topology.get('domain_bridge_count', 0)}")
    print(f"   Unique Domains: {topology.get('unique_domains', 0)}")
    print(
        f"   Interdisciplinary Ratio: {topology.get('interdisciplinary_ratio', 0.0):.2%}"
    )


async def run_graph_of_thoughts_demo(l4: L4SemanticHyperGraphV2):
    """
    Run graph-of-thoughts reasoning demonstration.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Graph-of-Thoughts Reasoning")
    print("=" * 70)

    # Create observation
    observation = Observation(
        id="demo_001",
        data=b"Optimize system performance while maintaining ethical integrity",
        urgency=UrgencyLevel.NEAR_REAL_TIME,
        context={
            "query": "How can optimization principles from different domains inform ethical AI design?",
            "domains_of_interest": ["math", "ethics", "economics"],
        },
    )

    print(f"\n🔍 Query: {observation.context['query']}")

    # Process with graph-of-thoughts
    result = await process_with_graph_of_thoughts(observation, l4, enable_got=True)

    print(f"\n⚡ Processing Complete:")
    print(f"   Total Time: {result.processing_time_ms:.1f}ms")
    print(f"   Graph Construction: {result.graph_construction_time_ms:.1f}ms")
    print(f"   SNR Computation: {result.snr_computation_time_ms:.1f}ms")

    # Display SNR metrics
    print(f"\n📈 Signal Quality (SNR):")
    print(f"   Overall SNR: {result.overall_snr.snr_score:.3f}")
    print(f"   Level: {result.overall_snr.level.name}")
    print(f"   Signal Strength: {result.overall_snr.signal_strength:.3f}")
    print(f"   Noise Floor: {result.overall_snr.noise_floor:.3f}")
    print(f"   Ihsān Metric: {result.overall_snr.ihsan_metric:.3f}")

    if result.overall_snr.ethical_override:
        print(f"   ⚠️  Ethical Override: SNR downgraded due to Ihsān < 0.95")

    # Display thought chains
    print(f"\n🧠 Thought Chains: {len(result.all_chains)} chains constructed")

    if result.top_thought_chain:
        chain = result.top_thought_chain
        print(f"\n   🏆 Best Chain (ID: {chain.id}):")
        print(f"      Total SNR: {chain.total_snr:.3f}")
        print(f"      Avg SNR: {chain.avg_snr:.3f}")
        print(f"      Depth: {chain.max_depth} hops")
        print(f"      Domain Diversity: {chain.domain_diversity:.3f}")
        print(f"      Thought Sequence:")

        for i, thought in enumerate(chain.thoughts[:5], 1):  # Show first 5
            domains_str = ", ".join(thought.domains) if thought.domains else "general"
            snr_str = f"{thought.get_snr_score():.3f}" if thought.snr_metrics else "N/A"
            print(f"         {i}. {thought.content} ({thought.thought_type.name})")
            print(f"            Domains: [{domains_str}], SNR: {snr_str}")

        if len(chain.thoughts) > 5:
            print(f"         ... and {len(chain.thoughts) - 5} more thoughts")

    # Display domain bridges
    print(f"\n🌉 Domain Bridges: {len(result.domain_bridges)} discovered")

    for i, bridge in enumerate(result.domain_bridges[:5], 1):  # Show first 5
        print(f"\n   Bridge {i}: {bridge.source_domain} ↔ {bridge.target_domain}")
        print(f"      Type: {bridge.bridge_type.name}")
        print(f"      Concepts: {bridge.source_concept} → {bridge.target_concept}")
        print(f"      Strength: {bridge.strength:.2f}")
        print(f"      Novelty: {bridge.novelty:.2f}")
        print(f"      SNR: {bridge.snr_score:.3f}")

    if len(result.domain_bridges) > 5:
        print(f"\n   ... and {len(result.domain_bridges) - 5} more bridges")

    return result


async def analyze_interdisciplinary_paths(l4: L4SemanticHyperGraphV2):
    """
    Find and analyze interdisciplinary paths in knowledge graph.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Interdisciplinary Path Analysis")
    print("=" * 70)

    # Find paths between distant concepts
    paths = await l4.find_interdisciplinary_paths(
        source_node="Optimization", target_node="Ihsan", max_hops=5, min_domains=2
    )

    print(
        f"\n🔗 Found {len(paths)} interdisciplinary paths from 'Optimization' to 'Ihsan'"
    )

    for i, path in enumerate(paths[:3], 1):  # Show top 3
        print(f"\n   Path {i}:")
        print(f"      Hops: {path['hop_count']}")
        print(f"      Domains Crossed: {path['domain_diversity']}")
        print(f"      Domain Sequence: {' → '.join(path['domains_crossed'])}")
        print(f"      Node Sequence: {' → '.join(path['node_sequence'][:5])}...")
        print(f"      Path Weight: {path['path_weight']:.3f}")

        if path.get("domain_transitions"):
            print(f"      Domain Transitions:")
            for trans in path["domain_transitions"]:
                print(f"         • {trans['from_domain']} → {trans['to_domain']}")


async def display_narrative(result):
    """
    Display the compiled narrative with Graph-of-Thoughts insights.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Compiled Narrative")
    print("=" * 70)

    narrative = result.ultimate_result.explanation

    print(f"\n📝 {narrative.style.name} Report")
    print(f"   Reading Time: ~{narrative.reading_time_seconds}s")
    print(f"   Complexity: {narrative.complexity_score:.2f}")

    print(f"\n{narrative.summary}")

    if narrative.sections:
        print(f"\n   Sections:")
        for section in narrative.sections[:3]:  # Show first 3
            print(f"\n   • {section.title}")
            # Truncate content for display
            content_preview = (
                section.content[:150] + "..."
                if len(section.content) > 150
                else section.content
            )
            print(f"     {content_preview}")


async def main():
    """
    Main demo orchestration.
    """
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 14 + "BIZRA Graph-of-Thoughts Demo" + " " * 26 + "║")
    print(
        "║"
        + " " * 8
        + "Interdisciplinary Reasoning with SNR Optimization"
        + " " * 10
        + "║"
    )
    print("╚" + "=" * 68 + "╝")

    # Initialize L4 knowledge graph
    # NOTE: Update these credentials for your Neo4j instance
    neo4j_uri = "bolt://localhost:7687"
    neo4j_auth = ("neo4j", "password")

    print(f"\n🔌 Connecting to Neo4j: {neo4j_uri}")

    l4 = L4SemanticHyperGraphV2(neo4j_uri, neo4j_auth)

    try:
        await l4.initialize()
        print("✅ Connected to knowledge graph")

        # Run demo steps
        await setup_demo_knowledge_graph(l4)
        result = await run_graph_of_thoughts_demo(l4)
        await analyze_interdisciplinary_paths(l4)
        await display_narrative(result)

        print("\n" + "=" * 70)
        print("✅ Demo Complete!")
        print("=" * 70)

        print("\n🎯 Key Takeaways:")
        print("   • Graph-of-Thoughts enables transparent multi-hop reasoning")
        print("   • SNR scoring surfaces highest-quality insights")
        print("   • Domain bridges discover cross-disciplinary connections")
        print("   • Ethical constraints (Ihsān ≥ 0.95) integrated throughout")
        print(f"   • Performance: {result.processing_time_ms:.1f}ms end-to-end")

        print("\n📚 Next Steps:")
        print("   1. Integrate with your existing cognitive_sovereign.py")
        print("   2. Expand knowledge graph with domain-specific content")
        print("   3. Tune beam_width and max_depth for your use case")
        print("   4. Set up monitoring dashboards (Prometheus + Grafana)")
        print("   5. Deploy to production with CI/CD pipeline")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Tip: Make sure Neo4j is running and credentials are correct")
        print(
            f"   You can start Neo4j with: docker run -p 7687:7687 -p 7474:7474 neo4j"
        )

    finally:
        await l4.close()
        print("\n🔌 Disconnected from knowledge graph")


if __name__ == "__main__":
    asyncio.run(main())
