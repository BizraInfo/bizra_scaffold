"""
BIZRA AEON OMEGA - Self-Evolving Agent Memory System
═══════════════════════════════════════════════════════════════════════════════
Composite Cognitive Architecture for Persistent Agent Memory

Addresses the structural amnesia problem in AI architectures by implementing:
1. Hierarchical Memory Tiers (Working → Episodic → Semantic → Procedural)
2. Memory Consolidation (L1 → L2 → L3 → L4 promotion paths)
3. Retrieval-Augmented Cognition (Context-aware memory retrieval)
4. Self-Evolution (Learning from experience, pattern crystallization)
5. SNR-Guided Retention (High-signal memories prioritized)

Economic Impact:
- Eliminates redundant relearning costs
- Enables cross-session knowledge transfer
- Reduces token consumption through memory compression
- Achieves continuous improvement via experience accumulation

Integration Points:
- L2WorkingMemoryV2: Short-term working memory with compression
- L3EpisodicMemoryV2: Episode storage with FAISS similarity search
- L4SemanticHyperGraphV2: Knowledge graph for semantic relations
- GraphOfThoughtsEngine: Reasoning over accumulated knowledge
- SNRScorer: Quality-based retention decisions

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from core.memory.memory_management import (
    BoundedList,
    LRUCache,
    SlidingWindowStats,
    TTLDict,
)
from core.snr_scorer import SNRLevel, SNRMetrics, SNRScorer

logger = logging.getLogger("bizra.agent_memory")


# =============================================================================
# MEMORY TIER DEFINITIONS
# =============================================================================


class MemoryTier(Enum):
    """Memory tier classification for hierarchical storage."""

    WORKING = auto()  # L1: Immediate working memory (seconds)
    EPISODIC = auto()  # L2: Episode-based memory (minutes to hours)
    SEMANTIC = auto()  # L3: Knowledge/fact memory (persistent)
    PROCEDURAL = auto()  # L4: Skill/procedure memory (persistent)
    META = auto()  # L5: Memory about memory (self-reflection)


class MemoryState(Enum):
    """State of a memory item in the consolidation pipeline."""

    ACTIVE = auto()  # Currently in use
    CONSOLIDATING = auto()  # Being processed for promotion
    CONSOLIDATED = auto()  # Successfully promoted
    DECAYING = auto()  # Being forgotten (low priority)
    ARCHIVED = auto()  # Long-term storage


class RetentionPolicy(Enum):
    """Policies for memory retention decisions."""

    ALWAYS_RETAIN = auto()  # Critical memories (high SNR, ethical)
    SNR_BASED = auto()  # Retain based on signal quality
    TIME_BASED = auto()  # Retain based on recency
    ACCESS_BASED = auto()  # Retain based on usage frequency
    HYBRID = auto()  # Combination of factors


# =============================================================================
# MEMORY ITEM DATA STRUCTURES
# =============================================================================


@dataclass
class MemorySignature:
    """Unique signature for memory deduplication and linking."""

    content_hash: str  # SHA-256 hash of content
    semantic_hash: str  # Hash based on meaning (for similar detection)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_content(cls, content: str, embedding: Optional[np.ndarray] = None) -> "MemorySignature":
        """Create signature from content."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Semantic hash from embedding if available
        if embedding is not None:
            # Locality-sensitive hashing for semantic similarity
            semantic_hash = cls._lsh_hash(embedding)
        else:
            # Fallback: use content hash
            semantic_hash = content_hash[:16]
        
        return cls(content_hash=content_hash, semantic_hash=semantic_hash)
    
    @staticmethod
    def _lsh_hash(embedding: np.ndarray, num_planes: int = 8) -> str:
        """Locality-sensitive hash for semantic similarity."""
        np.random.seed(42)  # Deterministic for consistency
        planes = np.random.randn(num_planes, embedding.shape[0])
        projections = np.dot(planes, embedding) > 0
        hash_val = sum(2**i for i, b in enumerate(projections) if b)
        return hex(hash_val)[2:].zfill(num_planes // 4)


@dataclass
class MemoryItem:
    """
    Single memory item with full provenance and quality metadata.
    
    Represents atomic unit of agent memory with:
    - Content: The actual memory (text, structured data)
    - Quality: SNR metrics for retention decisions
    - Provenance: Origin, transformations, links
    - Lifecycle: Creation, access, consolidation tracking
    """

    id: str  # Unique identifier
    content: str  # Memory content
    tier: MemoryTier  # Current storage tier
    state: MemoryState  # Lifecycle state
    
    # Quality metrics
    snr_score: float = 0.5  # Signal-to-noise ratio [0, 1]
    snr_level: SNRLevel = SNRLevel.MEDIUM  # Classification
    ihsan_score: float = 0.95  # Ethical alignment score
    
    # Embedding for semantic search
    embedding: Optional[np.ndarray] = None
    signature: Optional[MemorySignature] = None
    
    # Provenance
    source: str = "unknown"  # Where this memory came from
    session_id: Optional[str] = None  # Session that created it
    parent_ids: List[str] = field(default_factory=list)  # Parent memories
    child_ids: List[str] = field(default_factory=list)  # Derived memories
    
    # Context tags
    domains: Set[str] = field(default_factory=set)  # Knowledge domains
    tags: Set[str] = field(default_factory=set)  # User-defined tags
    
    # Lifecycle tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    consolidation_count: int = 0  # Times promoted to higher tier
    
    # Retention
    priority: float = 1.0  # Retention priority [0, 1]
    retention_policy: RetentionPolicy = RetentionPolicy.HYBRID
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update access tracking."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
        # Boost priority on access
        self.priority = min(1.0, self.priority + 0.05)

    def compute_retention_score(self) -> float:
        """
        Compute overall retention score for memory pruning.
        
        Combines:
        - SNR score (signal quality)
        - Recency (time since last access)
        - Frequency (access count)
        - Priority (explicit priority)
        - Ethical alignment (ihsan)
        """
        now = datetime.now(timezone.utc)
        
        # Recency factor: exponential decay from last access
        hours_since_access = (now - self.last_accessed).total_seconds() / 3600
        recency_score = math.exp(-hours_since_access / 24)  # 24-hour half-life
        
        # Frequency factor: log-scaled access count
        frequency_score = math.log1p(self.access_count) / 10
        
        # Combine factors
        retention = (
            0.35 * self.snr_score +
            0.20 * recency_score +
            0.15 * frequency_score +
            0.20 * self.priority +
            0.10 * self.ihsan_score
        )
        
        return min(1.0, max(0.0, retention))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "tier": self.tier.name,
            "state": self.state.name,
            "snr_score": self.snr_score,
            "snr_level": self.snr_level.name,
            "ihsan_score": self.ihsan_score,
            "source": self.source,
            "session_id": self.session_id,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "domains": list(self.domains),
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "priority": self.priority,
            "metadata": self.metadata,
        }


@dataclass
class MemoryQuery:
    """Query specification for memory retrieval."""

    query_text: Optional[str] = None  # Text to search for
    query_embedding: Optional[np.ndarray] = None  # Vector for similarity
    
    # Filters
    tiers: Optional[List[MemoryTier]] = None
    domains: Optional[Set[str]] = None
    tags: Optional[Set[str]] = None
    min_snr: float = 0.0
    min_retention: float = 0.0
    
    # Time range
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    accessed_after: Optional[datetime] = None
    
    # Limits
    max_results: int = 10
    include_archived: bool = False


@dataclass
class MemorySearchResult:
    """Result of memory search with relevance scoring."""

    item: MemoryItem
    relevance_score: float  # How relevant to query [0, 1]
    distance: float  # Embedding distance (lower = more similar)
    match_type: str  # "semantic", "keyword", "exact"


# =============================================================================
# MEMORY CONSOLIDATION ENGINE
# =============================================================================


class MemoryConsolidator:
    """
    Handles memory consolidation (promotion between tiers).
    
    Implements the cognitive psychology model of memory consolidation:
    - Working → Episodic: Immediate experiences become episodes
    - Episodic → Semantic: Repeated patterns become knowledge
    - Semantic → Procedural: Knowledge becomes skills
    
    SNR-guided consolidation:
    - HIGH SNR memories promoted faster
    - LOW SNR memories decay and are pruned
    """

    def __init__(
        self,
        snr_scorer: SNRScorer,
        promotion_threshold: float = 0.7,
        decay_threshold: float = 0.3,
        consolidation_interval_seconds: float = 60.0,
    ):
        """
        Initialize consolidator.
        
        Args:
            snr_scorer: SNR scorer for quality assessment
            promotion_threshold: Minimum retention score for promotion
            decay_threshold: Score below which memories decay
            consolidation_interval_seconds: How often to run consolidation
        """
        self.snr_scorer = snr_scorer
        self.promotion_threshold = promotion_threshold
        self.decay_threshold = decay_threshold
        self.consolidation_interval = consolidation_interval_seconds
        
        # Statistics
        self.promotions = 0
        self.decays = 0
        self.last_consolidation: Optional[datetime] = None
        
        logger.info(
            f"MemoryConsolidator initialized: "
            f"promote>{promotion_threshold:.2f}, decay<{decay_threshold:.2f}"
        )

    def should_promote(self, item: MemoryItem) -> bool:
        """Check if item should be promoted to higher tier."""
        retention_score = item.compute_retention_score()
        
        # Additional promotion criteria
        conditions = [
            retention_score >= self.promotion_threshold,
            item.access_count >= 2,  # Accessed multiple times
            item.snr_score >= 0.6,  # Reasonable signal quality
            item.ihsan_score >= 0.90,  # Ethically aligned
        ]
        
        return all(conditions)

    def should_decay(self, item: MemoryItem) -> bool:
        """Check if item should decay (be forgotten)."""
        retention_score = item.compute_retention_score()
        
        # Never decay high-SNR or frequently accessed memories
        if item.snr_level == SNRLevel.HIGH:
            return False
        if item.access_count >= 5:
            return False
        
        return retention_score < self.decay_threshold

    def get_promotion_tier(self, current_tier: MemoryTier) -> Optional[MemoryTier]:
        """Get next tier for promotion."""
        tier_order = [
            MemoryTier.WORKING,
            MemoryTier.EPISODIC,
            MemoryTier.SEMANTIC,
            MemoryTier.PROCEDURAL,
        ]
        
        try:
            current_idx = tier_order.index(current_tier)
            if current_idx < len(tier_order) - 1:
                return tier_order[current_idx + 1]
        except ValueError:
            pass
        
        return None

    async def consolidate(
        self,
        memories: List[MemoryItem],
        on_promote: Optional[Callable[[MemoryItem, MemoryTier], None]] = None,
        on_decay: Optional[Callable[[MemoryItem], None]] = None,
    ) -> Dict[str, int]:
        """
        Run consolidation on memory collection.
        
        Args:
            memories: List of memory items to consolidate
            on_promote: Callback when item is promoted
            on_decay: Callback when item decays
            
        Returns:
            Statistics dict with promotions and decays
        """
        start_time = time.perf_counter()
        promoted = 0
        decayed = 0
        
        for item in memories:
            if item.state in (MemoryState.ARCHIVED, MemoryState.CONSOLIDATING):
                continue
            
            # Check for promotion
            if self.should_promote(item):
                new_tier = self.get_promotion_tier(item.tier)
                if new_tier:
                    old_tier = item.tier
                    item.tier = new_tier
                    item.consolidation_count += 1
                    item.state = MemoryState.CONSOLIDATED
                    promoted += 1
                    self.promotions += 1
                    
                    if on_promote:
                        on_promote(item, old_tier)
                    
                    logger.debug(
                        f"Memory promoted: {item.id} "
                        f"{old_tier.name} → {new_tier.name}"
                    )
            
            # Check for decay
            elif self.should_decay(item):
                item.state = MemoryState.DECAYING
                decayed += 1
                self.decays += 1
                
                if on_decay:
                    on_decay(item)
                
                logger.debug(f"Memory decaying: {item.id}")
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.last_consolidation = datetime.now(timezone.utc)
        
        logger.info(
            f"Consolidation complete: {promoted} promoted, {decayed} decayed "
            f"in {elapsed_ms:.1f}ms"
        )
        
        return {"promoted": promoted, "decayed": decayed, "elapsed_ms": elapsed_ms}


# =============================================================================
# MEMORY RETRIEVAL ENGINE
# =============================================================================


class MemoryRetriever:
    """
    Context-aware memory retrieval with semantic search.
    
    Implements retrieval-augmented cognition:
    - Vector similarity for semantic matching
    - Keyword search for exact matching
    - Context filtering for relevance
    - SNR-weighted ranking
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize retriever.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            similarity_threshold: Minimum similarity for matches
        """
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # Search statistics
        self.total_searches = 0
        self.total_results = 0

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))

    def search(
        self,
        memories: List[MemoryItem],
        query: MemoryQuery,
    ) -> List[MemorySearchResult]:
        """
        Search memories based on query.
        
        Args:
            memories: Memory items to search
            query: Query specification
            
        Returns:
            Sorted list of search results
        """
        self.total_searches += 1
        results: List[MemorySearchResult] = []
        
        for item in memories:
            # Apply filters
            if not self._matches_filters(item, query):
                continue
            
            # Compute relevance
            relevance, distance, match_type = self._compute_relevance(item, query)
            
            if relevance >= self.similarity_threshold:
                results.append(MemorySearchResult(
                    item=item,
                    relevance_score=relevance,
                    distance=distance,
                    match_type=match_type,
                ))
        
        # Sort by relevance (highest first)
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        # Limit results
        results = results[:query.max_results]
        
        # Update accessed items
        for result in results:
            result.item.touch()
        
        self.total_results += len(results)
        
        return results

    def _matches_filters(self, item: MemoryItem, query: MemoryQuery) -> bool:
        """Check if item matches query filters."""
        # Tier filter
        if query.tiers and item.tier not in query.tiers:
            return False
        
        # Domain filter
        if query.domains and not item.domains.intersection(query.domains):
            return False
        
        # Tag filter
        if query.tags and not item.tags.intersection(query.tags):
            return False
        
        # SNR filter
        if item.snr_score < query.min_snr:
            return False
        
        # Retention filter
        if item.compute_retention_score() < query.min_retention:
            return False
        
        # Time filters
        if query.created_after and item.created_at < query.created_after:
            return False
        if query.created_before and item.created_at > query.created_before:
            return False
        if query.accessed_after and item.last_accessed < query.accessed_after:
            return False
        
        # Archive filter
        if not query.include_archived and item.state == MemoryState.ARCHIVED:
            return False
        
        return True

    def _compute_relevance(
        self,
        item: MemoryItem,
        query: MemoryQuery,
    ) -> Tuple[float, float, str]:
        """
        Compute relevance score for item.
        
        Returns: (relevance, distance, match_type)
        """
        relevance = 0.0
        distance = float("inf")
        match_type = "none"
        
        # Semantic similarity (embedding-based)
        if query.query_embedding is not None and item.embedding is not None:
            similarity = self.cosine_similarity(query.query_embedding, item.embedding)
            distance = 1.0 - similarity
            if similarity > relevance:
                relevance = similarity
                match_type = "semantic"
        
        # Keyword matching
        if query.query_text:
            text_lower = item.content.lower()
            query_lower = query.query_text.lower()
            
            # Exact match
            if query_lower in text_lower:
                keyword_score = 0.9
                if keyword_score > relevance:
                    relevance = keyword_score
                    match_type = "exact"
                    distance = 0.1
            else:
                # Partial match (word overlap)
                query_words = set(query_lower.split())
                content_words = set(text_lower.split())
                overlap = len(query_words & content_words) / max(len(query_words), 1)
                if overlap > relevance:
                    relevance = overlap
                    match_type = "keyword"
                    distance = 1.0 - overlap
        
        # Boost by SNR
        relevance *= (0.5 + 0.5 * item.snr_score)
        
        return relevance, distance, match_type


# =============================================================================
# SELF-EVOLVING AGENT MEMORY SYSTEM
# =============================================================================


class AgentMemorySystem:
    """
    Complete self-evolving agent memory system.
    
    Provides:
    - Multi-tier persistent memory storage
    - Automatic consolidation and pruning
    - Context-aware retrieval
    - Self-evolution through pattern crystallization
    - Session continuity across restarts
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Agent Memory System                          │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐         │
    │  │ Working │→ │ Episodic │→ │ Semantic │→ │ Procedural│         │
    │  │   (L1)  │  │   (L2)   │  │   (L3)   │  │   (L4)    │         │
    │  └────┬────┘  └────┬─────┘  └────┬─────┘  └─────┬─────┘         │
    │       │            │             │              │                │
    │       └────────────┴──────┬──────┴──────────────┘                │
    │                           │                                      │
    │                    ┌──────▼──────┐                               │
    │                    │ Consolidator │←── SNR Scorer               │
    │                    └──────┬──────┘                               │
    │                           │                                      │
    │                    ┌──────▼──────┐                               │
    │                    │  Retriever  │←── Embedding Model           │
    │                    └─────────────┘                               │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        snr_scorer: SNRScorer,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        max_working_memory: int = 100,
        max_episodic_memory: int = 1000,
        max_semantic_memory: int = 10000,
        max_procedural_memory: int = 5000,
        consolidation_interval_seconds: float = 60.0,
        auto_consolidate: bool = True,
    ):
        """
        Initialize agent memory system.
        
        Args:
            snr_scorer: SNR scorer for quality assessment
            embedding_fn: Function to compute embeddings
            max_working_memory: Max items in working memory
            max_episodic_memory: Max items in episodic memory
            max_semantic_memory: Max items in semantic memory
            max_procedural_memory: Max items in procedural memory
            consolidation_interval_seconds: Consolidation frequency
            auto_consolidate: Enable automatic consolidation
        """
        self.snr_scorer = snr_scorer
        self.embedding_fn = embedding_fn or self._default_embedding
        
        # Memory tier storage
        self._working: Dict[str, MemoryItem] = {}
        self._episodic: Dict[str, MemoryItem] = {}
        self._semantic: Dict[str, MemoryItem] = {}
        self._procedural: Dict[str, MemoryItem] = {}
        
        # Tier size limits
        self._limits = {
            MemoryTier.WORKING: max_working_memory,
            MemoryTier.EPISODIC: max_episodic_memory,
            MemoryTier.SEMANTIC: max_semantic_memory,
            MemoryTier.PROCEDURAL: max_procedural_memory,
        }
        
        # Component engines
        self.consolidator = MemoryConsolidator(
            snr_scorer=snr_scorer,
            consolidation_interval_seconds=consolidation_interval_seconds,
        )
        self.retriever = MemoryRetriever()
        
        # Session tracking
        self._current_session_id: Optional[str] = None
        self._session_start: Optional[datetime] = None
        
        # Auto-consolidation
        self._auto_consolidate = auto_consolidate
        self._consolidation_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = SlidingWindowStats(window_seconds=300.0)
        
        # Index for fast lookup
        self._signature_index: Dict[str, str] = {}  # content_hash → memory_id
        
        logger.info(
            f"AgentMemorySystem initialized: "
            f"working={max_working_memory}, episodic={max_episodic_memory}, "
            f"semantic={max_semantic_memory}, procedural={max_procedural_memory}"
        )

    def _default_embedding(self, text: str) -> np.ndarray:
        """Default embedding (random but deterministic for demo)."""
        # Simple hash-based embedding for testing
        # In production, use a real embedding model
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(768).astype(np.float32)

    def _get_storage(self, tier: MemoryTier) -> Dict[str, MemoryItem]:
        """Get storage dict for tier."""
        return {
            MemoryTier.WORKING: self._working,
            MemoryTier.EPISODIC: self._episodic,
            MemoryTier.SEMANTIC: self._semantic,
            MemoryTier.PROCEDURAL: self._procedural,
        }.get(tier, self._working)

    def _enforce_limits(self, tier: MemoryTier) -> int:
        """Enforce size limits on tier, pruning lowest-retention items."""
        storage = self._get_storage(tier)
        limit = self._limits.get(tier, 1000)
        
        if len(storage) <= limit:
            return 0
        
        # Sort by retention score (ascending)
        items = sorted(
            storage.values(),
            key=lambda x: x.compute_retention_score()
        )
        
        # Prune excess items
        to_remove = len(storage) - limit
        removed = 0
        
        for item in items[:to_remove]:
            if item.snr_level != SNRLevel.HIGH:  # Never prune HIGH SNR
                del storage[item.id]
                if item.signature:
                    self._signature_index.pop(item.signature.content_hash, None)
                removed += 1
        
        if removed > 0:
            logger.info(f"Pruned {removed} items from {tier.name}")
        
        return removed

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new memory session.
        
        Args:
            session_id: Optional session identifier (generated if None)
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        self._current_session_id = session_id
        self._session_start = datetime.now(timezone.utc)
        
        # Start auto-consolidation if enabled
        if self._auto_consolidate and self._consolidation_task is None:
            self._consolidation_task = asyncio.create_task(
                self._consolidation_loop()
            )
        
        logger.info(f"Memory session started: {session_id}")
        return session_id

    def end_session(self) -> Dict[str, Any]:
        """
        End current session and return session summary.
        
        Returns:
            Session statistics
        """
        if self._consolidation_task:
            self._consolidation_task.cancel()
            self._consolidation_task = None
        
        session_duration = None
        if self._session_start:
            session_duration = (
                datetime.now(timezone.utc) - self._session_start
            ).total_seconds()
        
        stats = {
            "session_id": self._current_session_id,
            "duration_seconds": session_duration,
            "memories_created": sum(
                1 for m in self._all_memories()
                if m.session_id == self._current_session_id
            ),
            "total_memories": len(list(self._all_memories())),
            "tier_counts": self.get_tier_counts(),
        }
        
        self._current_session_id = None
        self._session_start = None
        
        logger.info(f"Memory session ended: {stats}")
        return stats

    # =========================================================================
    # MEMORY OPERATIONS
    # =========================================================================

    async def remember(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.WORKING,
        domains: Optional[Set[str]] = None,
        tags: Optional[Set[str]] = None,
        source: str = "user",
        snr_score: Optional[float] = None,
        ihsan_score: float = 0.95,
        priority: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """
        Store a new memory.
        
        Args:
            content: Memory content
            tier: Initial storage tier
            domains: Knowledge domains
            tags: User-defined tags
            source: Memory source identifier
            snr_score: Pre-computed SNR (computed if None)
            ihsan_score: Ethical alignment score
            priority: Retention priority
            metadata: Additional metadata
            
        Returns:
            Created memory item
        """
        start_time = time.perf_counter()
        
        # Generate ID
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        
        # Compute embedding
        embedding = self.embedding_fn(content)
        
        # Create signature
        signature = MemorySignature.from_content(content, embedding)
        
        # Check for duplicate
        existing_id = self._signature_index.get(signature.content_hash)
        if existing_id:
            existing = self._find_by_id(existing_id)
            if existing:
                existing.touch()
                logger.debug(f"Duplicate memory detected, touching: {existing_id}")
                return existing
        
        # Compute SNR if not provided
        if snr_score is None:
            # Use default scoring (simplified)
            snr_score = 0.5 + 0.3 * priority
        
        # Classify SNR level
        snr_level = self._classify_snr(snr_score)
        
        # Create memory item
        item = MemoryItem(
            id=memory_id,
            content=content,
            tier=tier,
            state=MemoryState.ACTIVE,
            snr_score=snr_score,
            snr_level=snr_level,
            ihsan_score=ihsan_score,
            embedding=embedding,
            signature=signature,
            source=source,
            session_id=self._current_session_id,
            domains=domains or set(),
            tags=tags or set(),
            priority=priority,
            metadata=metadata or {},
        )
        
        # Store in appropriate tier
        storage = self._get_storage(tier)
        storage[memory_id] = item
        
        # Update signature index
        self._signature_index[signature.content_hash] = memory_id
        
        # Enforce limits
        self._enforce_limits(tier)
        
        # Record statistics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._stats.record(elapsed_ms)
        
        logger.debug(
            f"Memory stored: {memory_id} in {tier.name}, "
            f"SNR={snr_score:.2f}, latency={elapsed_ms:.1f}ms"
        )
        
        return item

    async def recall(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        tiers: Optional[List[MemoryTier]] = None,
        domains: Optional[Set[str]] = None,
        tags: Optional[Set[str]] = None,
        min_snr: float = 0.0,
        max_results: int = 10,
    ) -> List[MemorySearchResult]:
        """
        Recall memories matching query.
        
        Args:
            query_text: Text to search for
            query_embedding: Vector for semantic search
            tiers: Tiers to search (all if None)
            domains: Filter by domains
            tags: Filter by tags
            min_snr: Minimum SNR score
            max_results: Maximum results to return
            
        Returns:
            List of matching memories with relevance scores
        """
        # Build query
        if query_embedding is None and query_text:
            query_embedding = self.embedding_fn(query_text)
        
        query = MemoryQuery(
            query_text=query_text,
            query_embedding=query_embedding,
            tiers=tiers,
            domains=domains,
            tags=tags,
            min_snr=min_snr,
            max_results=max_results,
        )
        
        # Search all memories
        all_memories = list(self._all_memories())
        results = self.retriever.search(all_memories, query)
        
        return results

    async def forget(
        self,
        memory_id: str,
        hard_delete: bool = False,
    ) -> bool:
        """
        Forget a memory.
        
        Args:
            memory_id: Memory to forget
            hard_delete: If True, permanently delete; else archive
            
        Returns:
            True if memory was found and forgotten
        """
        item = self._find_by_id(memory_id)
        if not item:
            return False
        
        storage = self._get_storage(item.tier)
        
        if hard_delete:
            del storage[memory_id]
            if item.signature:
                self._signature_index.pop(item.signature.content_hash, None)
        else:
            item.state = MemoryState.ARCHIVED
        
        logger.info(f"Memory forgotten: {memory_id} (hard_delete={hard_delete})")
        return True

    async def consolidate(self) -> Dict[str, int]:
        """
        Manually trigger memory consolidation.
        
        Returns:
            Consolidation statistics
        """
        all_memories = list(self._all_memories())
        
        def on_promote(item: MemoryItem, old_tier: MemoryTier) -> None:
            # Move to new tier storage
            old_storage = self._get_storage(old_tier)
            new_storage = self._get_storage(item.tier)
            
            old_storage.pop(item.id, None)
            new_storage[item.id] = item

        def on_decay(item: MemoryItem) -> None:
            # Mark for archival
            item.state = MemoryState.DECAYING

        result = await self.consolidator.consolidate(
            all_memories,
            on_promote=on_promote,
            on_decay=on_decay,
        )
        
        return result

    # =========================================================================
    # PATTERN CRYSTALLIZATION (Self-Evolution)
    # =========================================================================

    async def crystallize_patterns(
        self,
        min_occurrences: int = 3,
        min_snr: float = 0.6,
    ) -> List[MemoryItem]:
        """
        Identify and crystallize recurring patterns into semantic memory.
        
        This is the core self-evolution mechanism:
        - Detect frequently accessed episodic memories
        - Extract common patterns
        - Promote to semantic memory as crystallized knowledge
        
        Args:
            min_occurrences: Minimum times pattern must appear
            min_snr: Minimum SNR for pattern
            
        Returns:
            List of newly crystallized memories
        """
        crystallized = []
        
        # Find high-frequency episodic memories
        episodic_items = [
            item for item in self._episodic.values()
            if item.access_count >= min_occurrences
            and item.snr_score >= min_snr
            and item.state == MemoryState.ACTIVE
        ]
        
        # Group by semantic similarity
        clusters = self._cluster_by_similarity(episodic_items)
        
        for cluster in clusters:
            if len(cluster) < min_occurrences:
                continue
            
            # Extract pattern from cluster
            pattern = self._extract_pattern(cluster)
            
            # Create crystallized semantic memory
            crystal = await self.remember(
                content=pattern,
                tier=MemoryTier.SEMANTIC,
                domains=self._merge_domains(cluster),
                tags={"crystallized", "pattern"},
                source="crystallization",
                snr_score=self._aggregate_snr(cluster),
                priority=1.0,
                metadata={
                    "source_memories": [m.id for m in cluster],
                    "occurrence_count": len(cluster),
                },
            )
            
            # Link parent memories
            for item in cluster:
                item.child_ids.append(crystal.id)
                crystal.parent_ids.append(item.id)
            
            crystallized.append(crystal)
            
            logger.info(
                f"Pattern crystallized: {crystal.id} from {len(cluster)} episodes"
            )
        
        return crystallized

    def _cluster_by_similarity(
        self,
        items: List[MemoryItem],
        threshold: float = 0.8,
    ) -> List[List[MemoryItem]]:
        """Cluster items by semantic similarity."""
        if not items:
            return []
        
        clusters: List[List[MemoryItem]] = []
        used = set()
        
        for item in items:
            if item.id in used:
                continue
            
            cluster = [item]
            used.add(item.id)
            
            for other in items:
                if other.id in used:
                    continue
                
                if item.embedding is not None and other.embedding is not None:
                    similarity = self.retriever.cosine_similarity(
                        item.embedding, other.embedding
                    )
                    if similarity >= threshold:
                        cluster.append(other)
                        used.add(other.id)
            
            clusters.append(cluster)
        
        return clusters

    def _extract_pattern(self, cluster: List[MemoryItem]) -> str:
        """Extract common pattern from cluster (simplified)."""
        # In production, use LLM or more sophisticated extraction
        # For now, use longest common content
        if not cluster:
            return ""
        
        contents = [item.content for item in cluster]
        
        # Simple: use first item's content with pattern annotation
        return f"[PATTERN] {contents[0]}"

    def _merge_domains(self, items: List[MemoryItem]) -> Set[str]:
        """Merge domains from multiple items."""
        domains: Set[str] = set()
        for item in items:
            domains.update(item.domains)
        return domains

    def _aggregate_snr(self, items: List[MemoryItem]) -> float:
        """Aggregate SNR scores from items."""
        if not items:
            return 0.5
        scores = [item.snr_score for item in items]
        return sum(scores) / len(scores)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _all_memories(self) -> List[MemoryItem]:
        """Get all memories across all tiers."""
        for storage in [self._working, self._episodic, self._semantic, self._procedural]:
            yield from storage.values()

    def _find_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """Find memory by ID across all tiers."""
        for storage in [self._working, self._episodic, self._semantic, self._procedural]:
            if memory_id in storage:
                return storage[memory_id]
        return None

    def _classify_snr(self, score: float) -> SNRLevel:
        """Classify SNR score into level."""
        if score >= 0.8:
            return SNRLevel.HIGH
        elif score >= 0.5:
            return SNRLevel.MEDIUM
        else:
            return SNRLevel.LOW

    async def _consolidation_loop(self) -> None:
        """Background consolidation loop."""
        while True:
            try:
                await asyncio.sleep(self.consolidator.consolidation_interval)
                await self.consolidate()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation error: {e}")

    # =========================================================================
    # STATISTICS AND MONITORING
    # =========================================================================

    def get_tier_counts(self) -> Dict[str, int]:
        """Get memory counts by tier."""
        return {
            "working": len(self._working),
            "episodic": len(self._episodic),
            "semantic": len(self._semantic),
            "procedural": len(self._procedural),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        all_memories = list(self._all_memories())
        
        snr_scores = [m.snr_score for m in all_memories]
        retention_scores = [m.compute_retention_score() for m in all_memories]
        
        return {
            "total_memories": len(all_memories),
            "tier_counts": self.get_tier_counts(),
            "state_counts": {
                state.name: sum(1 for m in all_memories if m.state == state)
                for state in MemoryState
            },
            "snr_distribution": {
                "avg": sum(snr_scores) / len(snr_scores) if snr_scores else 0,
                "min": min(snr_scores) if snr_scores else 0,
                "max": max(snr_scores) if snr_scores else 0,
            },
            "retention_distribution": {
                "avg": sum(retention_scores) / len(retention_scores) if retention_scores else 0,
                "min": min(retention_scores) if retention_scores else 0,
                "max": max(retention_scores) if retention_scores else 0,
            },
            "consolidation_stats": {
                "promotions": self.consolidator.promotions,
                "decays": self.consolidator.decays,
                "last_consolidation": (
                    self.consolidator.last_consolidation.isoformat()
                    if self.consolidator.last_consolidation
                    else None
                ),
            },
            "retrieval_stats": {
                "total_searches": self.retriever.total_searches,
                "total_results": self.retriever.total_results,
            },
            "session_id": self._current_session_id,
            "latency_stats": self._stats.stats(),
        }

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def export_to_dict(self) -> Dict[str, Any]:
        """Export all memories to dictionary for persistence."""
        return {
            "version": "1.0.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "memories": {
                "working": [m.to_dict() for m in self._working.values()],
                "episodic": [m.to_dict() for m in self._episodic.values()],
                "semantic": [m.to_dict() for m in self._semantic.values()],
                "procedural": [m.to_dict() for m in self._procedural.values()],
            },
            "statistics": self.get_statistics(),
        }

    def export_to_json(self, filepath: str) -> None:
        """Export memories to JSON file."""
        data = self.export_to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Memories exported to {filepath}")

    def import_from_dict(
        self,
        data: Dict[str, Any],
        validate_snr: bool = True,
        validate_ihsan: bool = True,
        min_snr: float = 0.3,
        min_ihsan: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Import memories from dictionary.
        
        Args:
            data: Exported memory dict (from export_to_dict())
            validate_snr: Whether to validate SNR thresholds
            validate_ihsan: Whether to validate Ihsān thresholds
            min_snr: Minimum SNR score for import (default: 0.3)
            min_ihsan: Minimum Ihsān score for import (default: 0.95)
            
        Returns:
            Import statistics dict
        """
        imported = {"working": 0, "episodic": 0, "semantic": 0, "procedural": 0}
        rejected = {"low_snr": 0, "low_ihsan": 0, "duplicate": 0, "invalid": 0}
        
        memories_data = data.get("memories", {})
        
        tier_mapping = {
            "working": (MemoryTier.WORKING, self._working),
            "episodic": (MemoryTier.EPISODIC, self._episodic),
            "semantic": (MemoryTier.SEMANTIC, self._semantic),
            "procedural": (MemoryTier.PROCEDURAL, self._procedural),
        }
        
        for tier_name, (tier_enum, storage) in tier_mapping.items():
            tier_data = memories_data.get(tier_name, [])
            
            for item_dict in tier_data:
                try:
                    # Validate thresholds
                    snr_score = item_dict.get("snr_score", 0.5)
                    ihsan_score = item_dict.get("ihsan_score", 0.95)
                    
                    if validate_snr and snr_score < min_snr:
                        rejected["low_snr"] += 1
                        continue
                        
                    if validate_ihsan and ihsan_score < min_ihsan:
                        rejected["low_ihsan"] += 1
                        continue
                    
                    # Check for duplicate
                    memory_id = item_dict.get("id", f"mem_{uuid.uuid4().hex[:12]}")
                    if memory_id in storage:
                        rejected["duplicate"] += 1
                        continue
                    
                    # Reconstruct MemoryItem
                    item = self._reconstruct_memory_item(item_dict, tier_enum)
                    
                    # Store
                    storage[item.id] = item
                    if item.signature:
                        self._signature_index[item.signature.content_hash] = item.id
                    
                    imported[tier_name] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to import memory: {e}")
                    rejected["invalid"] += 1
        
        # Enforce limits after bulk import
        for tier in [MemoryTier.WORKING, MemoryTier.EPISODIC, MemoryTier.SEMANTIC, MemoryTier.PROCEDURAL]:
            self._enforce_limits(tier)
        
        total_imported = sum(imported.values())
        total_rejected = sum(rejected.values())
        
        logger.info(
            f"Memory import complete: {total_imported} imported, "
            f"{total_rejected} rejected (snr={rejected['low_snr']}, "
            f"ihsan={rejected['low_ihsan']}, dup={rejected['duplicate']}, "
            f"invalid={rejected['invalid']})"
        )
        
        return {
            "imported": imported,
            "rejected": rejected,
            "total_imported": total_imported,
            "total_rejected": total_rejected,
        }

    def _reconstruct_memory_item(
        self,
        item_dict: Dict[str, Any],
        tier: MemoryTier,
    ) -> MemoryItem:
        """Reconstruct MemoryItem from serialized dict."""
        # Parse datetime fields
        created_at = item_dict.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        elif created_at is None:
            created_at = datetime.now(timezone.utc)
            
        last_accessed = item_dict.get("last_accessed")
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
        elif last_accessed is None:
            last_accessed = datetime.now(timezone.utc)
        
        # Parse SNR level
        snr_level_str = item_dict.get("snr_level", "MEDIUM")
        try:
            snr_level = SNRLevel[snr_level_str]
        except KeyError:
            snr_level = SNRLevel.MEDIUM
        
        # Parse state
        state_str = item_dict.get("state", "ACTIVE")
        try:
            state = MemoryState[state_str]
        except KeyError:
            state = MemoryState.ACTIVE
        
        # Create signature if content available
        content = item_dict.get("content", "")
        signature = MemorySignature.from_content(content)
        
        # Recompute embedding
        embedding = self.embedding_fn(content)
        
        return MemoryItem(
            id=item_dict.get("id", f"mem_{uuid.uuid4().hex[:12]}"),
            content=content,
            tier=tier,
            state=state,
            snr_score=item_dict.get("snr_score", 0.5),
            snr_level=snr_level,
            ihsan_score=item_dict.get("ihsan_score", 0.95),
            embedding=embedding,
            signature=signature,
            source=item_dict.get("source", "imported"),
            session_id=item_dict.get("session_id"),
            parent_ids=item_dict.get("parent_ids", []),
            child_ids=item_dict.get("child_ids", []),
            domains=set(item_dict.get("domains", [])),
            tags=set(item_dict.get("tags", [])),
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=item_dict.get("access_count", 0),
            consolidation_count=item_dict.get("consolidation_count", 0),
            priority=item_dict.get("priority", 1.0),
            metadata=item_dict.get("metadata", {}),
        )

    def import_from_json(
        self,
        filepath: str,
        validate_snr: bool = True,
        validate_ihsan: bool = True,
        min_snr: float = 0.3,
        min_ihsan: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Import memories from JSON file.
        
        Args:
            filepath: Path to JSON file (from export_to_json())
            validate_snr: Whether to validate SNR thresholds
            validate_ihsan: Whether to validate Ihsān thresholds
            min_snr: Minimum SNR score for import
            min_ihsan: Minimum Ihsān score for import
            
        Returns:
            Import statistics dict
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"Loading memories from {filepath}")
        return self.import_from_dict(
            data,
            validate_snr=validate_snr,
            validate_ihsan=validate_ihsan,
            min_snr=min_snr,
            min_ihsan=min_ihsan,
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_agent_memory(
    snr_scorer: Optional[SNRScorer] = None,
    embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
    **kwargs,
) -> AgentMemorySystem:
    """
    Factory function to create agent memory system.
    
    Args:
        snr_scorer: SNR scorer (created if None)
        embedding_fn: Embedding function (default if None)
        **kwargs: Additional arguments for AgentMemorySystem
        
    Returns:
        Configured AgentMemorySystem instance
    """
    if snr_scorer is None:
        snr_scorer = SNRScorer()
    
    return AgentMemorySystem(
        snr_scorer=snr_scorer,
        embedding_fn=embedding_fn,
        **kwargs,
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "MemoryTier",
    "MemoryState",
    "RetentionPolicy",
    # Data structures
    "MemorySignature",
    "MemoryItem",
    "MemoryQuery",
    "MemorySearchResult",
    # Engines
    "MemoryConsolidator",
    "MemoryRetriever",
    "AgentMemorySystem",
    # Factory
    "create_agent_memory",
]
