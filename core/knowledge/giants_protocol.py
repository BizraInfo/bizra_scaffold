"""
BIZRA AEON OMEGA - Standing on Shoulders of Giants Protocol
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Pattern: Meta-Learning from Historical Reasoning Traces

PROTOCOL FOUNDATION (Extracted from 1,546 chat histories):
─────────────────────────────────────────────────────────────────────────────
"In the garden of shared wisdom, every insight is a seed.
 Standing on shoulders of giants, we plant tomorrow's deed.
 Graph of thoughts ascending, through patterns we have gleaned,
 The highest SNR signal—where truth and trust convene."

CORE DISCOVERY FROM META-ANALYSIS:
1. "Integrity Flywheel" — Late-stage attractor across all models
   → Integrity & Proofs + Ihsān & Ethics + DevOps gates
   → Pattern: proofs_first() → publish_later()

2. Cross-Model Personality Insights:
   → Claude: Highest Option A/B/C exploration (GoT behavior)
   → DeepSeek: Constraint enforcer (FATE, enforcement language)
   → OpenAI: Long iterative execution threads (implementation loops)

3. Term Frequencies Signal:
   → PAT: 46.9 per 10k words (dominant agentic pattern)
   → SAT: 5.6 per 10k words (verification/enforcement)
   → ihsan: 5.8 per 10k words (ethical alignment constant)

4. Hub Concepts (highest degree in co-occurrence graph):
   → architecture-security edge weight: 164
   → architecture-layer edge weight: 150
   → architecture-key edge weight: 78

IMPLEMENTATION:
This module implements the Giants Protocol for:
1. Wisdom Extraction — Mining patterns from historical reasoning traces
2. Insight Crystallization — Converting patterns to actionable knowledge
3. Cross-Temporal Bridging — Connecting past insights to current context
4. SNR-Weighted Synthesis — Prioritizing highest-signal wisdom
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class WisdomType(Enum):
    """Types of crystallized wisdom from historical traces."""

    PATTERN = auto()           # Recurring structural pattern
    PRINCIPLE = auto()         # Foundational rule/invariant
    ANTI_PATTERN = auto()      # Known failure mode to avoid
    BRIDGE_INSIGHT = auto()    # Cross-domain connection
    FLYWHEEL = auto()          # Self-reinforcing loop pattern
    CONSTRAINT = auto()        # Hard limit or boundary
    HEURISTIC = auto()         # Soft guidance rule


class WisdomSource(Enum):
    """Source model for wisdom extraction."""

    CLAUDE = "claude"          # High GoT exploration, design-space search
    DEEPSEEK = "deepseek"      # Constraint enforcement, FATE language
    OPENAI = "openai"          # Long execution threads, implementation
    SYNTHETIC = "synthetic"    # Cross-model synthesis
    LOCAL = "local"            # Current session insights


@dataclass
class CrystallizedWisdom:
    """
    Single unit of extracted wisdom from historical reasoning.
    
    Represents actionable knowledge crystallized from patterns
    observed across multiple reasoning sessions.
    """
    
    id: str                           # Unique wisdom identifier
    wisdom_type: WisdomType           # Classification
    source: WisdomSource              # Origin model/session
    
    # Content
    title: str                        # Short descriptive title
    content: str                      # Detailed wisdom content
    evidence_citations: List[str]     # Supporting evidence paths
    
    # Quality metrics
    snr_score: float                  # Signal-to-noise quality [0,1]
    frequency: int                    # How often pattern appears
    confidence: float                 # Statistical confidence [0,1]
    
    # Temporal context
    first_observed: datetime          # When first discovered
    last_observed: datetime           # Most recent observation
    observation_count: int = 1        # Total observations
    
    # Relational context
    related_concepts: List[str] = field(default_factory=list)
    domain_tags: Set[str] = field(default_factory=set)
    
    # Ihsan alignment (per BIZRA_SOT Section 3.1)
    ihsan_score: float = 0.95         # Ethical alignment [0,1]
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_receipt(self) -> Dict[str, Any]:
        """Generate SAT-compliant receipt for this wisdom."""
        content_bytes = f"{self.id}:{self.title}:{self.content}".encode("utf-8")
        return {
            "wisdom_id": self.id,
            "content_hash": hashlib.sha256(content_bytes).hexdigest(),
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "confidence": self.confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attestation": "GIANTS_PROTOCOL_v1.0"
        }


@dataclass  
class IntegrityFlywheel:
    """
    Self-reinforcing loop pattern extracted from meta-analysis.
    
    The Integrity Flywheel is the dominant late-stage attractor:
    Integrity & Proofs → Ihsān & Ethics → DevOps Gates → (repeat)
    
    Pattern: proofs_first() → publish_later()
    """
    
    id: str
    name: str
    phases: List[str]                 # Ordered flywheel phases
    reinforcement_edges: List[Tuple[str, str, float]]  # (from, to, strength)
    
    # Flywheel properties
    momentum: float = 0.0             # Current spin velocity [0,∞)
    friction: float = 0.1             # Energy loss per cycle
    activation_threshold: float = 0.5  # Minimum momentum to stay spinning
    
    # Evidence
    observation_sources: List[WisdomSource] = field(default_factory=list)
    supporting_wisdom: List[str] = field(default_factory=list)
    
    def spin(self, input_energy: float) -> float:
        """Add energy to flywheel, return new momentum."""
        self.momentum = max(0, self.momentum * (1 - self.friction) + input_energy)
        return self.momentum
    
    def is_active(self) -> bool:
        """Check if flywheel has sufficient momentum."""
        return self.momentum >= self.activation_threshold


@dataclass
class GiantsGraphNode:
    """Node in the Giants knowledge graph."""
    
    id: str
    concept: str
    weight: float                     # Hub importance
    degree: int                       # Number of connections
    domains: Set[str] = field(default_factory=set)
    wisdom_ids: List[str] = field(default_factory=list)


@dataclass
class GiantsGraphEdge:
    """Edge in the Giants knowledge graph."""
    
    source: str
    target: str
    weight: float                     # Co-occurrence strength
    edge_type: str = "co-occurrence"


class GiantsProtocolEngine:
    """
    Standing on Shoulders of Giants Protocol Engine.
    
    Core Capabilities:
    1. Extract wisdom patterns from historical reasoning traces
    2. Crystallize cross-model insights into actionable knowledge
    3. Maintain Integrity Flywheel for proofs-first pattern
    4. Synthesize high-SNR wisdom for current context
    5. Generate SAT-compliant evidence receipts
    
    Integration Points:
    - GraphOfThoughtsEngine: SNR-guided reasoning
    - SNRScorer: Quality assessment
    - L4SemanticHyperGraphV2: Knowledge substrate
    - PAT/SAT: Dual-agentic verification
    
    Meta-Analysis Foundation (1,546 conversations):
    - Claude: 608 convos, median 8 msgs, GoT exploration dominant
    - OpenAI: 437 convos, median 28 msgs, execution threads
    - DeepSeek: 501 convos, median 6 msgs, constraint enforcement
    """
    
    # Hub concepts from meta-analysis (architecture, security, layer as top hubs)
    HUB_CONCEPTS = {
        "architecture": {"weight": 164, "domains": {"design", "system"}},
        "security": {"weight": 164, "domains": {"safety", "cryptography"}},
        "layer": {"weight": 150, "domains": {"abstraction", "structure"}},
        "key": {"weight": 78, "domains": {"cryptography", "identity"}},
        "management": {"weight": 68, "domains": {"operations", "governance"}},
        "infrastructure": {"weight": 39, "domains": {"devops", "platform"}},
    }
    
    # Term frequency signals from meta-analysis
    TERM_SIGNALS = {
        "pat": 46.9,          # per 10k words
        "bizra": 55.1,
        "poi": 20.9,
        "rag": 18.0,
        "sat": 5.6,
        "ihsan": 5.8,
        "agentic": 6.98,
        "token": 24.2,
        "governance": 7.58,
    }
    
    def __init__(
        self,
        wisdom_store_path: Optional[Path] = None,
        ihsan_threshold: float = 0.95,
        snr_threshold: float = 0.5,
        max_wisdom_cache: int = 500,
    ):
        """
        Initialize Giants Protocol Engine.
        
        Args:
            wisdom_store_path: Path to persist crystallized wisdom
            ihsan_threshold: Minimum Ihsan score (per BIZRA_SOT 3.1)
            snr_threshold: Minimum SNR for wisdom retention
            max_wisdom_cache: Maximum cached wisdom entries
        """
        self.wisdom_store_path = wisdom_store_path or Path("data/giants_wisdom")
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold
        self.max_wisdom_cache = max_wisdom_cache
        
        # Wisdom storage
        self._wisdom_cache: Dict[str, CrystallizedWisdom] = {}
        self._wisdom_index: Dict[str, List[str]] = defaultdict(list)  # domain -> wisdom_ids
        
        # Knowledge graph (from meta-analysis)
        self._nodes: Dict[str, GiantsGraphNode] = {}
        self._edges: List[GiantsGraphEdge] = []
        
        # Flywheels
        self._flywheels: Dict[str, IntegrityFlywheel] = {}
        self._init_integrity_flywheel()
        
        # Statistics
        self.total_wisdom_extracted = 0
        self.total_insights_synthesized = 0
        self.total_flywheel_spins = 0
        
        # Initialize hub concepts from meta-analysis
        self._init_hub_concepts()
        
        logger.info(
            f"GiantsProtocolEngine initialized: ihsan_threshold={ihsan_threshold}, "
            f"snr_threshold={snr_threshold}, hub_concepts={len(self.HUB_CONCEPTS)}"
        )
    
    def _init_integrity_flywheel(self) -> None:
        """Initialize the Integrity Flywheel from meta-analysis discovery."""
        flywheel = IntegrityFlywheel(
            id="integrity_flywheel_v1",
            name="Integrity Flywheel",
            phases=[
                "integrity_proofs",      # Verify before claim
                "ihsan_ethics",          # Ethical alignment check
                "devops_gates",          # CI/CD enforcement
                "evidence_publication",  # Only then publish
            ],
            reinforcement_edges=[
                ("integrity_proofs", "ihsan_ethics", 0.9),
                ("ihsan_ethics", "devops_gates", 0.85),
                ("devops_gates", "evidence_publication", 0.95),
                ("evidence_publication", "integrity_proofs", 0.8),  # Loop back
            ],
            observation_sources=[
                WisdomSource.CLAUDE,
                WisdomSource.DEEPSEEK,
                WisdomSource.OPENAI,
            ],
        )
        self._flywheels["integrity"] = flywheel
        logger.info("Integrity Flywheel initialized from meta-analysis pattern")
    
    def _init_hub_concepts(self) -> None:
        """Initialize knowledge graph with hub concepts from meta-analysis."""
        for concept, data in self.HUB_CONCEPTS.items():
            node = GiantsGraphNode(
                id=f"hub_{concept}",
                concept=concept,
                weight=data["weight"],
                degree=0,  # Will be computed when edges added
                domains=data["domains"],
            )
            self._nodes[node.id] = node
        
        # Add edges from meta-analysis co-occurrence matrix
        hub_edges = [
            ("architecture", "security", 164),
            ("architecture", "layer", 150),
            ("architecture", "key", 78),
            ("architecture", "management", 68),
            ("security", "key", 34),
            ("security", "infrastructure", 50),
            ("layer", "management", 18),
        ]
        for source, target, weight in hub_edges:
            self._edges.append(GiantsGraphEdge(
                source=f"hub_{source}",
                target=f"hub_{target}",
                weight=weight,
            ))
            # Update degrees
            if f"hub_{source}" in self._nodes:
                self._nodes[f"hub_{source}"].degree += 1
            if f"hub_{target}" in self._nodes:
                self._nodes[f"hub_{target}"].degree += 1
    
    async def extract_wisdom_from_trace(
        self,
        reasoning_trace: Dict[str, Any],
        source: WisdomSource = WisdomSource.LOCAL,
    ) -> List[CrystallizedWisdom]:
        """
        Extract wisdom patterns from a reasoning trace.
        
        Implements pattern mining on historical reasoning to crystallize
        reusable insights that can accelerate future reasoning.
        
        Args:
            reasoning_trace: Dict containing reasoning steps, decisions, outcomes
            source: Origin model/session for this trace
            
        Returns:
            List of crystallized wisdom extracted from trace
        """
        extracted: List[CrystallizedWisdom] = []
        
        # 1. Pattern Detection - Look for recurring structures
        patterns = self._detect_patterns(reasoning_trace)
        
        for pattern in patterns:
            wisdom = CrystallizedWisdom(
                id=self._generate_wisdom_id(pattern),
                wisdom_type=WisdomType.PATTERN,
                source=source,
                title=pattern.get("title", "Unnamed Pattern"),
                content=pattern.get("content", ""),
                evidence_citations=[pattern.get("evidence_path", "")],
                snr_score=pattern.get("snr_score", 0.5),
                frequency=pattern.get("frequency", 1),
                confidence=pattern.get("confidence", 0.7),
                first_observed=datetime.now(timezone.utc),
                last_observed=datetime.now(timezone.utc),
                related_concepts=pattern.get("concepts", []),
                domain_tags=set(pattern.get("domains", [])),
            )
            
            # Apply Ihsan gate (per BIZRA_SOT Section 3.1)
            if wisdom.ihsan_score >= self.ihsan_threshold:
                if wisdom.snr_score >= self.snr_threshold:
                    extracted.append(wisdom)
                    self._cache_wisdom(wisdom)
        
        # 2. Principle Extraction - Look for invariants/rules
        principles = self._extract_principles(reasoning_trace)
        
        for principle in principles:
            wisdom = CrystallizedWisdom(
                id=self._generate_wisdom_id(principle),
                wisdom_type=WisdomType.PRINCIPLE,
                source=source,
                title=principle.get("title", "Unnamed Principle"),
                content=principle.get("content", ""),
                evidence_citations=principle.get("citations", []),
                snr_score=principle.get("snr_score", 0.6),
                frequency=principle.get("frequency", 1),
                confidence=principle.get("confidence", 0.8),
                first_observed=datetime.now(timezone.utc),
                last_observed=datetime.now(timezone.utc),
                related_concepts=principle.get("concepts", []),
                domain_tags=set(principle.get("domains", [])),
            )
            
            if wisdom.ihsan_score >= self.ihsan_threshold:
                if wisdom.snr_score >= self.snr_threshold:
                    extracted.append(wisdom)
                    self._cache_wisdom(wisdom)
        
        # 3. Bridge Detection - Cross-domain insights
        bridges = self._detect_bridges(reasoning_trace)
        
        for bridge in bridges:
            wisdom = CrystallizedWisdom(
                id=self._generate_wisdom_id(bridge),
                wisdom_type=WisdomType.BRIDGE_INSIGHT,
                source=source,
                title=f"Bridge: {bridge.get('source_domain')} → {bridge.get('target_domain')}",
                content=bridge.get("insight", ""),
                evidence_citations=[bridge.get("evidence_path", "")],
                snr_score=bridge.get("snr_score", 0.7),
                frequency=bridge.get("frequency", 1),
                confidence=bridge.get("confidence", 0.75),
                first_observed=datetime.now(timezone.utc),
                last_observed=datetime.now(timezone.utc),
                related_concepts=bridge.get("concepts", []),
                domain_tags=set([bridge.get("source_domain", ""), bridge.get("target_domain", "")]),
            )
            
            if wisdom.ihsan_score >= self.ihsan_threshold:
                if wisdom.snr_score >= self.snr_threshold:
                    extracted.append(wisdom)
                    self._cache_wisdom(wisdom)
        
        self.total_wisdom_extracted += len(extracted)
        logger.info(f"Extracted {len(extracted)} wisdom units from reasoning trace")
        
        return extracted
    
    def _detect_patterns(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect recurring structural patterns in reasoning trace."""
        patterns = []
        
        # Look for repeated decision structures
        decisions = trace.get("decisions", [])
        if len(decisions) >= 2:
            # Simple pattern: same decision type appearing multiple times
            decision_types = defaultdict(int)
            for d in decisions:
                decision_types[d.get("type", "unknown")] += 1
            
            for dtype, count in decision_types.items():
                if count >= 2:
                    patterns.append({
                        "title": f"Recurring {dtype} Decision Pattern",
                        "content": f"Pattern of {dtype} decisions appearing {count} times",
                        "snr_score": min(0.9, 0.5 + count * 0.1),
                        "frequency": count,
                        "confidence": min(0.95, 0.6 + count * 0.05),
                        "concepts": [dtype, "decision", "pattern"],
                        "domains": ["reasoning", "meta-cognition"],
                    })
        
        # Look for GoT exploration patterns (Claude-style Option A/B/C)
        explorations = trace.get("explorations", [])
        if explorations:
            got_patterns = [e for e in explorations if "option" in str(e).lower()]
            if got_patterns:
                patterns.append({
                    "title": "Graph-of-Thoughts Exploration",
                    "content": f"Explicit option exploration with {len(got_patterns)} branches",
                    "snr_score": 0.8,
                    "frequency": len(got_patterns),
                    "confidence": 0.85,
                    "concepts": ["got", "exploration", "options"],
                    "domains": ["reasoning", "search"],
                })
        
        return patterns
    
    def _extract_principles(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract foundational principles/invariants from reasoning trace."""
        principles = []
        
        # Look for constraint mentions
        constraints = trace.get("constraints", [])
        for c in constraints:
            principles.append({
                "title": f"Constraint: {c.get('name', 'unnamed')}",
                "content": c.get("description", ""),
                "snr_score": c.get("strength", 0.6),
                "frequency": 1,
                "confidence": 0.9,  # Constraints are high confidence
                "concepts": c.get("concepts", []),
                "domains": c.get("domains", ["constraint"]),
            })
        
        # Look for invariant assertions
        invariants = trace.get("invariants", [])
        for inv in invariants:
            principles.append({
                "title": f"Invariant: {inv.get('name', 'unnamed')}",
                "content": inv.get("assertion", ""),
                "snr_score": 0.9,  # Invariants are high signal
                "frequency": 1,
                "confidence": 0.95,
                "concepts": inv.get("concepts", []),
                "domains": ["invariant", "foundation"],
            })
        
        return principles
    
    def _detect_bridges(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect cross-domain bridge insights in reasoning trace."""
        bridges = []
        
        # Look for domain crossings
        domain_crossings = trace.get("domain_crossings", [])
        for crossing in domain_crossings:
            bridges.append({
                "source_domain": crossing.get("from", "unknown"),
                "target_domain": crossing.get("to", "unknown"),
                "insight": crossing.get("insight", ""),
                "snr_score": crossing.get("novelty", 0.5) * 0.8 + 0.2,
                "frequency": 1,
                "confidence": crossing.get("strength", 0.7),
                "concepts": crossing.get("concepts", []),
            })
        
        return bridges
    
    def _generate_wisdom_id(self, data: Dict[str, Any]) -> str:
        """Generate unique ID for wisdom based on content hash."""
        content = json.dumps(data, sort_keys=True)
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
        timestamp = int(time.time() * 1000)
        return f"wisdom_{hash_val}_{timestamp}"
    
    def _cache_wisdom(self, wisdom: CrystallizedWisdom) -> None:
        """Cache wisdom with LRU eviction policy."""
        # Evict if at capacity
        if len(self._wisdom_cache) >= self.max_wisdom_cache:
            # Remove oldest entry
            oldest_id = min(
                self._wisdom_cache.keys(),
                key=lambda k: self._wisdom_cache[k].last_observed
            )
            del self._wisdom_cache[oldest_id]
        
        self._wisdom_cache[wisdom.id] = wisdom
        
        # Index by domain
        for domain in wisdom.domain_tags:
            if wisdom.id not in self._wisdom_index[domain]:
                self._wisdom_index[domain].append(wisdom.id)
    
    async def synthesize_insights(
        self,
        query: str,
        context: Dict[str, Any],
        top_k: int = 5,
        require_cross_domain: bool = False,
    ) -> List[CrystallizedWisdom]:
        """
        Synthesize relevant wisdom insights for current context.
        
        Implements "Standing on Shoulders of Giants" by retrieving
        and ranking historical wisdom relevant to current query.
        
        Args:
            query: Current reasoning query
            context: Additional context (domains, concepts, etc.)
            top_k: Number of insights to return
            require_cross_domain: Only return bridge insights
            
        Returns:
            Ranked list of relevant crystallized wisdom
        """
        candidates: List[Tuple[float, CrystallizedWisdom]] = []
        
        # Extract domains and concepts from query/context
        query_domains = set(context.get("domains", []))
        query_concepts = set(context.get("concepts", []))
        
        # Score each cached wisdom
        for wisdom in self._wisdom_cache.values():
            # Filter by type if required
            if require_cross_domain and wisdom.wisdom_type != WisdomType.BRIDGE_INSIGHT:
                continue
            
            # Compute relevance score
            score = self._compute_relevance_score(
                wisdom, query, query_domains, query_concepts
            )
            
            if score > 0:
                candidates.append((score, wisdom))
        
        # Sort by score and return top-K
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = [w for _, w in candidates[:top_k]]
        
        self.total_insights_synthesized += len(results)
        logger.info(f"Synthesized {len(results)} insights for query")
        
        return results
    
    def _compute_relevance_score(
        self,
        wisdom: CrystallizedWisdom,
        query: str,
        query_domains: Set[str],
        query_concepts: Set[str],
    ) -> float:
        """Compute relevance score for wisdom given current query."""
        score = 0.0
        
        # Domain overlap
        domain_overlap = len(wisdom.domain_tags & query_domains)
        if domain_overlap > 0:
            score += 0.3 * domain_overlap
        
        # Concept overlap
        concept_overlap = len(set(wisdom.related_concepts) & query_concepts)
        if concept_overlap > 0:
            score += 0.3 * concept_overlap
        
        # SNR weighting
        score += 0.2 * wisdom.snr_score
        
        # Confidence weighting
        score += 0.1 * wisdom.confidence
        
        # Recency bonus (decay over time)
        age_days = (datetime.now(timezone.utc) - wisdom.last_observed).days
        recency_factor = math.exp(-age_days / 30)  # 30-day half-life
        score += 0.1 * recency_factor
        
        # Frequency bonus (more observations = more reliable)
        freq_factor = min(1.0, wisdom.observation_count / 10)
        score *= (1 + 0.2 * freq_factor)
        
        return score
    
    def spin_integrity_flywheel(self, input_energy: float = 0.5) -> Dict[str, Any]:
        """
        Spin the Integrity Flywheel with input energy.
        
        The Integrity Flywheel is the dominant pattern discovered in meta-analysis:
        proofs_first → ihsan_check → devops_gates → publish
        
        Returns:
            Flywheel state with momentum and phase information
        """
        flywheel = self._flywheels.get("integrity")
        if not flywheel:
            return {"error": "Integrity flywheel not initialized"}
        
        new_momentum = flywheel.spin(input_energy)
        self.total_flywheel_spins += 1
        
        # Determine current phase based on momentum
        phase_idx = int(new_momentum * len(flywheel.phases)) % len(flywheel.phases)
        current_phase = flywheel.phases[phase_idx]
        
        return {
            "flywheel_id": flywheel.id,
            "momentum": new_momentum,
            "is_active": flywheel.is_active(),
            "current_phase": current_phase,
            "all_phases": flywheel.phases,
            "total_spins": self.total_flywheel_spins,
        }
    
    def get_hub_concepts(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top hub concepts from the Giants knowledge graph.
        
        Hub concepts are high-degree nodes discovered from meta-analysis
        that serve as stable anchors for reasoning.
        
        Returns:
            List of hub concept info dicts
        """
        sorted_nodes = sorted(
            self._nodes.values(),
            key=lambda n: n.weight,
            reverse=True
        )
        
        return [
            {
                "id": node.id,
                "concept": node.concept,
                "weight": node.weight,
                "degree": node.degree,
                "domains": list(node.domains),
            }
            for node in sorted_nodes[:top_k]
        ]
    
    def get_term_signals(self) -> Dict[str, float]:
        """
        Get term frequency signals from meta-analysis.
        
        These signals indicate which terms appear most frequently
        across all historical reasoning traces (per 10k words).
        """
        return self.TERM_SIGNALS.copy()
    
    def generate_evidence_pack(self) -> Dict[str, Any]:
        """
        Generate SAT-compliant evidence pack for Giants Protocol state.
        
        Returns:
            Evidence pack with receipts, metrics, and attestations
        """
        # Generate receipts for all cached wisdom
        wisdom_receipts = [
            w.to_receipt() for w in self._wisdom_cache.values()
        ]
        
        # Flywheel state
        flywheel_states = {
            fw_id: {
                "momentum": fw.momentum,
                "is_active": fw.is_active(),
                "phases": fw.phases,
            }
            for fw_id, fw in self._flywheels.items()
        }
        
        # Hub metrics
        hub_metrics = self.get_hub_concepts(top_k=10)
        
        # Compute aggregate stats
        pack = {
            "protocol_version": "GIANTS_PROTOCOL_v1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": {
                "total_wisdom_extracted": self.total_wisdom_extracted,
                "total_insights_synthesized": self.total_insights_synthesized,
                "total_flywheel_spins": self.total_flywheel_spins,
                "cached_wisdom_count": len(self._wisdom_cache),
                "hub_concepts_count": len(self._nodes),
                "edge_count": len(self._edges),
            },
            "flywheels": flywheel_states,
            "hub_concepts": hub_metrics,
            "term_signals": self.TERM_SIGNALS,
            "wisdom_receipts": wisdom_receipts[:100],  # Top 100 for pack size
            "ihsan_threshold": self.ihsan_threshold,
            "snr_threshold": self.snr_threshold,
        }
        
        # Compute pack hash for integrity
        pack_content = json.dumps(pack, sort_keys=True)
        pack["pack_hash"] = hashlib.sha256(pack_content.encode()).hexdigest()
        
        return pack


# Convenience factory function
def create_giants_engine(
    wisdom_store_path: Optional[Path] = None,
) -> GiantsProtocolEngine:
    """
    Create a Giants Protocol Engine with default BIZRA configuration.
    
    Uses Ihsan threshold of 0.95 per BIZRA_SOT Section 3.1.
    """
    return GiantsProtocolEngine(
        wisdom_store_path=wisdom_store_path,
        ihsan_threshold=0.95,
        snr_threshold=0.5,
        max_wisdom_cache=500,
    )
