"""
BIZRA AEON OMEGA - Graph of Thoughts Reasoning Engine
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Pattern: Multi-Hop Interdisciplinary Reasoning with SNR-Guided Search

Implements graph-of-thoughts architecture for:
1. Thought Chain Construction - Building reasoning paths through knowledge graph
2. Interdisciplinary Traversal - Crossing domain boundaries for novel insights
3. SNR-Weighted Ranking - Prioritizing high-signal paths
4. Beam Search Exploration - Top-K expansion to manage combinatorial explosion
5. Retrograde Signaling - Propagating high-SNR discoveries back to attention layer

Addresses cognitive limitations through:
- Explicit thought chains vs. implicit black-box reasoning
- Cross-domain connection discovery
- Signal amplification for breakthrough insights
- Computational tractability via beam search pruning

Integration with BIZRA:
- L4 Semantic HyperGraph: Knowledge substrate
- MetaCognitiveOrchestrator: Strategy selection
- SNRScorer: Path quality assessment
- APEXOrchestrator: Event sourcing for thought chains
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.snr_scorer import SNRLevel, SNRMetrics, SNRScorer

logger = logging.getLogger(__name__)


class ThoughtType(Enum):
    """Types of thoughts in reasoning chain."""

    PERCEPTION = auto()  # Sensory/attention-driven concept
    MEMORY = auto()  # Retrieved from episodic/semantic memory
    INFERENCE = auto()  # Derived through logical reasoning
    ANALOGY = auto()  # Cross-domain mapping
    SYNTHESIS = auto()  # Integration of multiple thoughts
    HYPOTHESIS = auto()  # Speculative exploration
    VALIDATION = auto()  # Verification/testing


class DomainBridgeType(Enum):
    """Types of cross-domain connections."""

    ANALOGY = auto()  # Structural similarity across domains
    CAUSALITY = auto()  # Causal mechanism transferable
    EMERGENCE = auto()  # Higher-level pattern from integration
    REDUCTION = auto()  # Lower-level explanation
    ISOMORPHISM = auto()  # One-to-one structural mapping
    HOMOLOGY = auto()  # Shared evolutionary/conceptual origin


@dataclass
class Thought:
    """
    Single thought node in reasoning graph.

    Represents atomic cognitive element with:
    - Content: Entity/concept from knowledge graph
    - Context: Surrounding relational structure
    - Quality: SNR metrics for signal strength
    """

    id: str  # Unique thought identifier
    content: str  # Entity name from knowledge graph
    thought_type: ThoughtType  # Classification of thought
    snr_metrics: Optional[SNRMetrics]  # Quality assessment

    # Graph context
    source_nodes: List[str]  # HyperGraph entities this thought connects
    domains: Set[str]  # Knowledge domains (math, physics, etc.)

    # Provenance
    depth: int = 0  # Distance from root in thought chain
    parent_thoughts: List[str] = field(default_factory=list)  # Preceding thoughts
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Metadata
    activation_strength: float = 1.0  # How strongly this thought is activated
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_snr_score(self) -> float:
        """Get SNR score or default to 0.5."""
        return self.snr_metrics.snr_score if self.snr_metrics else 0.5


@dataclass
class DomainBridge:
    """
    Cross-domain connection discovered during traversal.

    Represents insight that connects concepts across disciplines.
    """

    id: str  # Unique bridge identifier
    bridge_type: DomainBridgeType  # Type of cross-domain connection
    source_domain: str  # Origin domain
    target_domain: str  # Destination domain
    source_concept: str  # Concept in source domain
    target_concept: str  # Concept in target domain

    # Connection properties
    strength: float  # How strong the connection [0,1]
    novelty: float  # How novel/surprising [0,1]
    snr_score: float  # Signal quality of this bridge

    # Supporting evidence
    connecting_path: List[str]  # Thought IDs forming bridge
    shared_properties: List[str]  # Common structural features

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThoughtChain:
    """
    Sequence of thoughts forming coherent reasoning path.

    Represents one complete line of reasoning from query to conclusion.
    """

    id: str  # Unique chain identifier
    thoughts: List[Thought]  # Ordered sequence of thoughts
    bridges: List[DomainBridge]  # Cross-domain connections discovered

    # Chain properties
    total_snr: float  # Aggregate SNR for entire chain
    avg_snr: float  # Average SNR per thought
    max_depth: int  # Longest path in chain
    domain_diversity: float  # Entropy of domain distribution

    # Quality metrics
    coherence: float = 1.0  # Logical consistency [0,1]
    novelty: float = 0.0  # Surprising/creative score [0,1]
    completeness: float = 0.0  # Coverage of query [0,1]

    # Metadata
    query: str = ""  # Original query/prompt
    conclusion: str = ""  # Final synthesis
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BeamSearchState:
    """
    State during beam search exploration.

    Maintains top-K candidate paths for expansion.
    """

    beam: List[Tuple[float, List[str]]]  # (score, path) heap
    beam_width: int  # Maximum beam size (K)
    visited: Set[str]  # Nodes already explored
    depth: int = 0  # Current search depth
    max_depth: int = 5  # Maximum depth limit


class GraphOfThoughtsEngine:
    """
    Graph-of-Thoughts reasoning engine.

    Core Capabilities:
    1. Construct thought chains through multi-hop traversal
    2. Discover cross-domain bridges (interdisciplinary insights)
    3. Rank paths by SNR-weighted quality
    4. Prune search space via beam search (top-K expansion)
    5. Generate explicit reasoning narratives

    Algorithm:
    1. Initialize beam with seed concepts from query
    2. For each depth level:
       a. Expand top-K candidates via hyperedge traversal
       b. Compute SNR for new thoughts
       c. Detect domain crossings (potential bridges)
       d. Prune to top-K by aggregate SNR
    3. Return top thought chains with discovered bridges

    Integration Points:
    - L4SemanticHyperGraphV2: Knowledge graph queries
    - SNRScorer: Quality assessment
    - MetaCognitiveOrchestrator: Strategy selection
    - APEXOrchestrator: Event emission
    """

    def __init__(
        self,
        snr_scorer: SNRScorer,
        beam_width: int = 10,
        max_depth: int = 5,
        min_snr_threshold: float = 0.3,
        novelty_bonus: float = 0.2,
    ):
        """
        Initialize Graph-of-Thoughts engine.

        Args:
            snr_scorer: SNR scorer for quality assessment
            beam_width: Maximum beam size (K in top-K)
            max_depth: Maximum thought chain depth
            min_snr_threshold: Minimum SNR to keep thought
            novelty_bonus: Bonus for cross-domain bridges [0,1]
        """
        self.snr_scorer = snr_scorer
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.min_snr_threshold = min_snr_threshold
        self.novelty_bonus = novelty_bonus

        # Caching and Performance Scaffolding
        self._chain_cache: Dict[str, List[ThoughtChain]] = {}
        self._cache_ttl = 3600  # 1 hour
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_max_entries = 1000  # P2 FIX: Prevent unbounded cache growth

        # Statistics
        self.total_thoughts_generated = 0
        self.total_bridges_discovered = 0
        self.total_chains_constructed = 0

        logger.info(
            f"GraphOfThoughtsEngine initialized: beam_width={beam_width}, "
            f"max_depth={max_depth}, min_snr={min_snr_threshold}"
        )

    def _adaptive_beam_width(self, query: str, resource_budget: float = 1.0) -> int:
        """
        Compute adaptive beam width based on query complexity and resource budget.
        
        Complexity factors:
        - Query length
        - Presence of interdisciplinary keywords
        
        Returns:
            Beam width >= 1 (P3 FIX: never 0)
        """
        base_width = self.beam_width
        
        # Complexity estimation
        complexity = 1.0
        if len(query) > 100: 
            complexity += 0.2
        
        keywords = ["bridge", "connect", "interdisciplinary", "cross-domain", "synthesis"]
        if any(kw in query.lower() for kw in keywords):
            complexity += 0.3
            
        # Resource factor (0.5 to 2.0)
        resource_factor = max(0.5, min(2.0, resource_budget))
        
        # P3 FIX: Ensure minimum beam width of 1 to prevent search termination
        return max(1, int(base_width * complexity * resource_factor))
    
    def _evict_cache_if_needed(self) -> None:
        """P2 FIX: Evict oldest cache entries if cache exceeds max size."""
        if len(self._chain_cache) <= self._cache_max_entries:
            return
        
        # Sort by timestamp and remove oldest entries
        sorted_keys = sorted(
            self._cache_timestamps.keys(),
            key=lambda k: self._cache_timestamps[k]
        )
        
        # Remove oldest 10% of entries
        num_to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:num_to_remove]:
            del self._chain_cache[key]
            del self._cache_timestamps[key]
        
        logger.info(f"Cache eviction: removed {num_to_remove} entries")

    async def reason(
        self,
        query: str,
        seed_concepts: List[str],
        hypergraph_query_fn: Callable[[str], Any],
        convergence_fn: Callable[[str, Dict], Any],
        top_k_chains: int = 5,
        resource_budget: float = 1.0,
    ) -> List[ThoughtChain]:
        """
        Execute graph-of-thoughts reasoning with adaptive beam search and caching.

        Args:
            query: Natural language query/prompt
            seed_concepts: Initial concepts to start exploration
            hypergraph_query_fn: Function to query knowledge graph
            convergence_fn: Function to compute convergence metrics
            top_k_chains: Number of top chains to return
            resource_budget: Computational budget factor [0.5, 2.0]

        Returns:
            List of top-K thought chains ranked by SNR
        """
        # P2 FIX: Cache key now includes resource_budget and adaptive beam width
        adaptive_width = self._adaptive_beam_width(query, resource_budget)
        cache_key = f"{query}:{','.join(sorted(seed_concepts))}:budget={resource_budget}:width={adaptive_width}"
        
        # 1. Check Cache
        if cache_key in self._chain_cache:
            if time.time() - self._cache_timestamps[cache_key] < self._cache_ttl:
                logger.info(f"Returning cached thought chains for query: '{query}'")
                return self._chain_cache[cache_key][:top_k_chains]

        start_time = time.perf_counter()
        
        # 2. Use pre-calculated adaptive beam width from cache key computation
        logger.info(f"Starting graph-of-thoughts reasoning. Adaptive beam_width: {adaptive_width}")

        # Initialize beam search
        beam_state = BeamSearchState(
            beam=[], beam_width=adaptive_width, visited=set(), max_depth=self.max_depth
        )

        # Seed beam with initial concepts
        for concept in seed_concepts:
            heapq.heappush(
                beam_state.beam, (-1.0, [concept])  # Negative for max-heap behavior
            )
            beam_state.visited.add(concept)

        # Beam search expansion
        all_thoughts: Dict[str, Thought] = {}
        all_bridges: List[DomainBridge] = []

        for depth in range(self.max_depth):
            beam_state.depth = depth

            logger.debug(f"Depth {depth}: Beam size = {len(beam_state.beam)}")

            # Expand current beam
            new_beam = []

            for neg_score, path in beam_state.beam:
                current_node = path[-1]

                # Query hypergraph for neighbors
                neighbors = await hypergraph_query_fn(current_node)

                for neighbor_info in neighbors:
                    neighbor_id = neighbor_info["id"]

                    # Skip if already visited
                    if neighbor_id in beam_state.visited:
                        continue

                    # Create thought for this neighbor
                    thought = await self._create_thought(
                        neighbor_id, neighbor_info, depth + 1, path, convergence_fn
                    )

                    all_thoughts[thought.id] = thought
                    self.total_thoughts_generated += 1

                    # Check SNR threshold
                    if thought.get_snr_score() < self.min_snr_threshold:
                        continue

                    # Extend path
                    new_path = path + [neighbor_id]

                    # Compute path score (aggregate SNR)
                    path_score = self._compute_path_score(new_path, all_thoughts)

                    # Check for domain bridge
                    bridge = self._detect_domain_bridge(path, neighbor_id, all_thoughts)

                    if bridge:
                        all_bridges.append(bridge)
                        self.total_bridges_discovered += 1
                        # Bonus for discovering bridge
                        path_score += self.novelty_bonus
                        logger.info(
                            f"Domain bridge discovered: {bridge.source_domain} -> "
                            f"{bridge.target_domain} (SNR: {bridge.snr_score:.3f})"
                        )

                    heapq.heappush(new_beam, (-path_score, new_path))
                    beam_state.visited.add(neighbor_id)

            if not new_beam:
                logger.warning(
                    f"No new thoughts found at depth {depth}. Stopping exploration."
                )
                break

            # Prune to beam width (keep top-K)
            beam_state.beam = heapq.nsmallest(
                beam_state.beam_width, new_beam, key=lambda x: x[0]
            )

        # Convert top beam paths to thought chains
        chains = self._construct_chains(
            beam_state.beam, all_thoughts, all_bridges, query
        )

        self.total_chains_constructed += len(chains)

        # Rank and return top-K chains
        ranked_chains = sorted(chains, key=lambda c: c.total_snr, reverse=True)

        # Update Cache (with P2 FIX: eviction policy)
        self._evict_cache_if_needed()
        self._chain_cache[cache_key] = ranked_chains
        self._cache_timestamps[cache_key] = time.time()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Graph-of-thoughts reasoning complete: {len(ranked_chains)} chains, "
            f"{len(all_thoughts)} thoughts, {len(all_bridges)} bridges in {elapsed_ms:.1f}ms"
        )

        return ranked_chains[:top_k_chains]

    async def _create_thought(
        self,
        concept_id: str,
        concept_info: Dict[str, Any],
        depth: int,
        parent_path: List[str],
        convergence_fn: Callable,
    ) -> Thought:
        """Create thought node with SNR assessment."""

        # Extract domain tags from concept info
        domains = set(concept_info.get("domains", []))

        # Compute convergence metrics
        context = {"parent_path": parent_path, "depth": depth, "domains": list(domains)}

        convergence_result = await convergence_fn(concept_id, context)

        # Compute SNR (simplified - full integration would use all components)
        snr_metrics = self.snr_scorer.compute_from_convergence(
            convergence_result,
            consistency=concept_info.get("consistency", 0.7),
            disagreement=concept_info.get("disagreement", 0.2),
            ihsan_metric=concept_info.get("ihsan", 0.95),
        )

        # Classify thought type
        thought_type = self._classify_thought_type(concept_info, parent_path, domains)

        thought = Thought(
            id=f"thought_{concept_id}_{depth}",
            content=concept_id,
            thought_type=thought_type,
            snr_metrics=snr_metrics,
            source_nodes=[concept_id],
            domains=domains,
            depth=depth,
            parent_thoughts=[f"thought_{p}_{i}" for i, p in enumerate(parent_path)],
            metadata=concept_info,
        )

        return thought

    def _classify_thought_type(
        self, concept_info: Dict, parent_path: List[str], domains: Set[str]
    ) -> ThoughtType:
        """Classify thought type based on context."""

        # Compute depth from parent_path
        depth = len(parent_path) if parent_path is not None else 0

        # Check for domain crossing
        if depth > 0:
            parent_domains = set(concept_info.get("parent_domains", []))
            if domains and parent_domains and not domains.intersection(parent_domains):
                return ThoughtType.ANALOGY

        # Default classification
        relation_type = concept_info.get("relation_type", "")

        if "memory" in relation_type.lower():
            return ThoughtType.MEMORY
        elif "infer" in relation_type.lower():
            return ThoughtType.INFERENCE
        elif depth == 0:
            return ThoughtType.PERCEPTION
        else:
            return ThoughtType.SYNTHESIS

    def _compute_path_score(
        self, path: List[str], thoughts: Dict[str, Thought]
    ) -> float:
        """Compute aggregate SNR score for path.

        Note: Seed concepts (at index 0) don't have Thought objects,
        so we start scoring from index 1 (depth 1).
        """
        if not path:
            return 0.0

        scores = []
        # Start from index 1 since seed concepts at index 0 have no Thought
        for depth, concept_id in enumerate(path):
            if depth == 0:
                # Seed concept - no thought object, use default score
                continue
            thought_id = f"thought_{concept_id}_{depth}"
            if thought_id in thoughts:
                scores.append(thoughts[thought_id].get_snr_score())

        if not scores:
            return 1.0  # Return default score if only seed concept in path

        # Aggregate: geometric mean for balanced contribution
        product = math.prod(scores)
        return product ** (1.0 / len(scores))

    def _detect_domain_bridge(
        self, parent_path: List[str], current_concept: str, thoughts: Dict[str, Thought]
    ) -> Optional[DomainBridge]:
        """Detect if current step creates cross-domain bridge.

        Args:
            parent_path: Path before adding current_concept
            current_concept: The newly added concept
            thoughts: Dictionary of all created thoughts

        Returns:
            DomainBridge if domain crossing detected, None otherwise
        """
        if not parent_path:
            return None

        # Current thought depth is len(parent_path) since it extends the path
        current_depth = len(parent_path)
        current_thought_id = f"thought_{current_concept}_{current_depth}"

        if current_thought_id not in thoughts:
            return None

        current_domains = thoughts[current_thought_id].domains

        # Parent is at the end of parent_path, at depth len(parent_path)-1
        # But seed concept at index 0 has no thought - check for this
        parent_concept = parent_path[-1]
        parent_depth = len(parent_path) - 1

        if parent_depth == 0:
            # Parent is seed concept - no thought object, can't detect bridge
            return None

        parent_thought_id = f"thought_{parent_concept}_{parent_depth}"

        if parent_thought_id not in thoughts:
            return None

        parent_domains = thoughts[parent_thought_id].domains

        # Check for domain crossing
        if not current_domains or not parent_domains:
            return None

        crossed_domains = current_domains - parent_domains

        if crossed_domains:
            # Bridge detected!
            source_domain = next(iter(parent_domains))  # Safe: checked non-empty
            target_domain = next(iter(crossed_domains))  # Safe: checked non-empty

            bridge = DomainBridge(
                id=f"bridge_{parent_concept}_{current_concept}",
                bridge_type=DomainBridgeType.ANALOGY,  # Default type
                source_domain=source_domain,
                target_domain=target_domain,
                source_concept=parent_concept,
                target_concept=current_concept,
                strength=0.8,  # Could be computed from edge weights
                novelty=0.7,  # Could be computed from graph statistics
                snr_score=thoughts[current_thought_id].get_snr_score(),
                connecting_path=[parent_thought_id, current_thought_id],
                shared_properties=[],  # Could be extracted from hyperedge properties
            )

            return bridge

        return None

    def _construct_chains(
        self,
        beam_paths: List[Tuple[float, List[str]]],
        thoughts: Dict[str, Thought],
        bridges: List[DomainBridge],
        query: str,
    ) -> List[ThoughtChain]:
        """Convert beam paths to thought chains.

        Note: Seed concepts at index 0 don't have Thought objects,
        so chain_thoughts starts from depth 1.
        """
        chains = []

        for neg_score, path in beam_paths:
            if not path:
                continue

            # Collect thoughts in order (skip seed at index 0)
            chain_thoughts = []
            for depth, concept_id in enumerate(path):
                if depth == 0:
                    # Seed concept has no Thought object
                    continue
                thought_id = f"thought_{concept_id}_{depth}"
                if thought_id in thoughts:
                    chain_thoughts.append(thoughts[thought_id])

            # Collect bridges on this path
            path_set = set(path)
            chain_bridges = [
                b
                for b in bridges
                if b.source_concept in path_set and b.target_concept in path_set
            ]

            # Compute chain metrics
            snr_scores = [t.get_snr_score() for t in chain_thoughts]
            total_snr = sum(snr_scores) if snr_scores else 0.0
            avg_snr = total_snr / len(snr_scores) if snr_scores else 0.0

            # Domain diversity (entropy of domain distribution)
            domain_counts = defaultdict(int)
            for t in chain_thoughts:
                for d in t.domains:
                    domain_counts[d] += 1

            total_domains = sum(domain_counts.values())
            if total_domains > 0:
                domain_probs = [c / total_domains for c in domain_counts.values()]
                diversity = -sum(p * math.log2(p) if p > 0 else 0 for p in domain_probs)
            else:
                diversity = 0.0

            chain = ThoughtChain(
                id=f"chain_{len(chains)}",
                thoughts=chain_thoughts,
                bridges=chain_bridges,
                total_snr=total_snr,
                avg_snr=avg_snr,
                max_depth=len(path),  # Full path length including seed
                domain_diversity=diversity,
                query=query,
                conclusion=path[-1],  # Safe: checked path non-empty above
                metadata={
                    "path": path,
                    "beam_score": -neg_score,
                    "seed_concept": path[0],  # Include seed for reference
                },
            )

            chains.append(chain)

        return chains

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_thoughts_generated": self.total_thoughts_generated,
            "total_bridges_discovered": self.total_bridges_discovered,
            "total_chains_constructed": self.total_chains_constructed,
            "avg_bridges_per_chain": (
                self.total_bridges_discovered / self.total_chains_constructed
                if self.total_chains_constructed > 0
                else 0.0
            ),
        }


# Export public API
__all__ = [
    "ThoughtType",
    "DomainBridgeType",
    "Thought",
    "DomainBridge",
    "ThoughtChain",
    "GraphOfThoughtsEngine",
]
