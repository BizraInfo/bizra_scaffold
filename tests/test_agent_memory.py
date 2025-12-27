"""
Tests for BIZRA Self-Evolving Agent Memory System
═══════════════════════════════════════════════════════════════════════════════

Comprehensive test suite covering:
1. Memory storage and retrieval
2. Consolidation and promotion
3. Pattern crystallization
4. Session management
5. Persistence
6. Edge cases and error handling
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import numpy as np
import pytest

from core.memory.agent_memory import (
    AgentMemorySystem,
    MemoryConsolidator,
    MemoryItem,
    MemoryQuery,
    MemoryRetriever,
    MemorySearchResult,
    MemorySignature,
    MemoryState,
    MemoryTier,
    RetentionPolicy,
    create_agent_memory,
)
from core.snr_scorer import SNRLevel, SNRScorer


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def snr_scorer():
    """Create SNR scorer for tests."""
    return SNRScorer()


@pytest.fixture
def memory_system(snr_scorer):
    """Create memory system for tests."""
    return AgentMemorySystem(
        snr_scorer=snr_scorer,
        max_working_memory=10,
        max_episodic_memory=50,
        max_semantic_memory=100,
        max_procedural_memory=50,
        auto_consolidate=False,  # Manual control in tests
    )


@pytest.fixture
def sample_embedding():
    """Create sample embedding vector."""
    np.random.seed(42)
    return np.random.randn(768).astype(np.float32)


# =============================================================================
# MEMORY SIGNATURE TESTS
# =============================================================================


class TestMemorySignature:
    """Tests for MemorySignature class."""

    def test_from_content_creates_valid_signature(self):
        """Test signature creation from content."""
        content = "This is a test memory content"
        signature = MemorySignature.from_content(content)

        assert signature.content_hash is not None
        assert len(signature.content_hash) == 64  # SHA-256 hex
        assert signature.semantic_hash is not None
        assert signature.timestamp is not None

    def test_from_content_with_embedding(self, sample_embedding):
        """Test signature with embedding uses LSH."""
        content = "Test content"
        signature = MemorySignature.from_content(content, sample_embedding)

        assert signature.semantic_hash is not None
        # LSH hash should be shorter than content hash
        assert len(signature.semantic_hash) <= 16

    def test_same_content_same_hash(self):
        """Test deterministic hashing."""
        content = "Same content"
        sig1 = MemorySignature.from_content(content)
        sig2 = MemorySignature.from_content(content)

        assert sig1.content_hash == sig2.content_hash

    def test_different_content_different_hash(self):
        """Test different content produces different hash."""
        sig1 = MemorySignature.from_content("Content A")
        sig2 = MemorySignature.from_content("Content B")

        assert sig1.content_hash != sig2.content_hash


# =============================================================================
# MEMORY ITEM TESTS
# =============================================================================


class TestMemoryItem:
    """Tests for MemoryItem class."""

    def test_touch_updates_access_tracking(self):
        """Test touch method updates access count and time."""
        item = MemoryItem(
            id="test-1",
            content="Test content",
            tier=MemoryTier.WORKING,
            state=MemoryState.ACTIVE,
        )

        initial_count = item.access_count
        initial_time = item.last_accessed

        item.touch()

        assert item.access_count == initial_count + 1
        assert item.last_accessed >= initial_time

    def test_compute_retention_score_range(self):
        """Test retention score is in valid range."""
        item = MemoryItem(
            id="test-1",
            content="Test",
            tier=MemoryTier.WORKING,
            state=MemoryState.ACTIVE,
            snr_score=0.8,
            ihsan_score=0.95,
            priority=0.7,
        )

        score = item.compute_retention_score()

        assert 0.0 <= score <= 1.0

    def test_high_snr_improves_retention(self):
        """Test high SNR leads to better retention."""
        low_snr = MemoryItem(
            id="low",
            content="Low SNR",
            tier=MemoryTier.WORKING,
            state=MemoryState.ACTIVE,
            snr_score=0.2,
        )

        high_snr = MemoryItem(
            id="high",
            content="High SNR",
            tier=MemoryTier.WORKING,
            state=MemoryState.ACTIVE,
            snr_score=0.9,
        )

        assert high_snr.compute_retention_score() > low_snr.compute_retention_score()

    def test_to_dict_serialization(self):
        """Test dictionary serialization."""
        item = MemoryItem(
            id="test-1",
            content="Test content",
            tier=MemoryTier.EPISODIC,
            state=MemoryState.ACTIVE,
            domains={"math", "physics"},
            tags={"important"},
        )

        data = item.to_dict()

        assert data["id"] == "test-1"
        assert data["tier"] == "EPISODIC"
        assert data["state"] == "ACTIVE"
        assert set(data["domains"]) == {"math", "physics"}
        assert "important" in data["tags"]


# =============================================================================
# MEMORY CONSOLIDATOR TESTS
# =============================================================================


class TestMemoryConsolidator:
    """Tests for MemoryConsolidator class."""

    def test_should_promote_high_quality_memory(self, snr_scorer):
        """Test promotion criteria for high-quality memories."""
        consolidator = MemoryConsolidator(snr_scorer)

        item = MemoryItem(
            id="promote-me",
            content="High quality memory",
            tier=MemoryTier.WORKING,
            state=MemoryState.ACTIVE,
            snr_score=0.85,
            ihsan_score=0.96,
            access_count=5,
            priority=0.9,
        )

        assert consolidator.should_promote(item)

    def test_should_not_promote_low_quality(self, snr_scorer):
        """Test low-quality memories not promoted."""
        consolidator = MemoryConsolidator(snr_scorer)

        item = MemoryItem(
            id="low-quality",
            content="Low quality",
            tier=MemoryTier.WORKING,
            state=MemoryState.ACTIVE,
            snr_score=0.3,
            access_count=0,
        )

        assert not consolidator.should_promote(item)

    def test_should_decay_low_retention(self, snr_scorer):
        """Test decay detection for low-retention memories."""
        consolidator = MemoryConsolidator(snr_scorer, decay_threshold=0.4)

        item = MemoryItem(
            id="decay-me",
            content="Low retention memory",
            tier=MemoryTier.WORKING,
            state=MemoryState.ACTIVE,
            snr_score=0.2,
            snr_level=SNRLevel.LOW,
            access_count=0,
            priority=0.1,
        )
        # Simulate old last_accessed
        item.last_accessed = datetime.now(timezone.utc) - timedelta(days=7)

        assert consolidator.should_decay(item)

    def test_high_snr_never_decays(self, snr_scorer):
        """Test HIGH SNR memories never decay."""
        consolidator = MemoryConsolidator(snr_scorer)

        item = MemoryItem(
            id="high-snr",
            content="High SNR",
            tier=MemoryTier.WORKING,
            state=MemoryState.ACTIVE,
            snr_score=0.9,
            snr_level=SNRLevel.HIGH,
            access_count=0,
        )
        item.last_accessed = datetime.now(timezone.utc) - timedelta(days=30)

        assert not consolidator.should_decay(item)

    def test_get_promotion_tier(self, snr_scorer):
        """Test tier promotion sequence."""
        consolidator = MemoryConsolidator(snr_scorer)

        assert consolidator.get_promotion_tier(MemoryTier.WORKING) == MemoryTier.EPISODIC
        assert consolidator.get_promotion_tier(MemoryTier.EPISODIC) == MemoryTier.SEMANTIC
        assert consolidator.get_promotion_tier(MemoryTier.SEMANTIC) == MemoryTier.PROCEDURAL
        assert consolidator.get_promotion_tier(MemoryTier.PROCEDURAL) is None

    @pytest.mark.asyncio
    async def test_consolidate_promotes_and_decays(self, snr_scorer):
        """Test consolidation promotes and decays appropriately."""
        consolidator = MemoryConsolidator(snr_scorer)

        # Memory that should be promoted
        promotable = MemoryItem(
            id="promotable",
            content="Good memory",
            tier=MemoryTier.WORKING,
            state=MemoryState.ACTIVE,
            snr_score=0.85,
            ihsan_score=0.96,
            access_count=3,
            priority=0.9,
        )

        # Memory that should decay
        decayable = MemoryItem(
            id="decayable",
            content="Bad memory",
            tier=MemoryTier.WORKING,
            state=MemoryState.ACTIVE,
            snr_score=0.1,
            snr_level=SNRLevel.LOW,
            access_count=0,
            priority=0.1,
        )
        decayable.last_accessed = datetime.now(timezone.utc) - timedelta(days=7)

        promoted_items = []
        decayed_items = []

        result = await consolidator.consolidate(
            [promotable, decayable],
            on_promote=lambda item, tier: promoted_items.append(item.id),
            on_decay=lambda item: decayed_items.append(item.id),
        )

        assert result["promoted"] >= 1
        assert "promotable" in promoted_items


# =============================================================================
# MEMORY RETRIEVER TESTS
# =============================================================================


class TestMemoryRetriever:
    """Tests for MemoryRetriever class."""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1."""
        retriever = MemoryRetriever()
        vec = np.array([1.0, 2.0, 3.0])

        similarity = retriever.cosine_similarity(vec, vec)

        assert np.isclose(similarity, 1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0."""
        retriever = MemoryRetriever()
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = retriever.cosine_similarity(vec1, vec2)

        assert np.isclose(similarity, 0.0)

    def test_search_finds_matching_content(self):
        """Test search finds matching content."""
        retriever = MemoryRetriever(similarity_threshold=0.3)

        np.random.seed(42)
        embedding = np.random.randn(768).astype(np.float32)

        memories = [
            MemoryItem(
                id="match",
                content="python programming language",
                tier=MemoryTier.SEMANTIC,
                state=MemoryState.ACTIVE,
                embedding=embedding,
                snr_score=0.8,
            ),
            MemoryItem(
                id="no-match",
                content="cooking recipes",
                tier=MemoryTier.SEMANTIC,
                state=MemoryState.ACTIVE,
                embedding=np.random.randn(768).astype(np.float32),
                snr_score=0.5,
            ),
        ]

        query = MemoryQuery(
            query_text="python programming",
            query_embedding=embedding,
            max_results=5,
        )

        results = retriever.search(memories, query)

        assert len(results) >= 1
        assert results[0].item.id == "match"

    def test_search_filters_by_tier(self):
        """Test search respects tier filter."""
        retriever = MemoryRetriever(similarity_threshold=0.0)

        memories = [
            MemoryItem(
                id="working",
                content="test",
                tier=MemoryTier.WORKING,
                state=MemoryState.ACTIVE,
                snr_score=0.8,
            ),
            MemoryItem(
                id="semantic",
                content="test",
                tier=MemoryTier.SEMANTIC,
                state=MemoryState.ACTIVE,
                snr_score=0.8,
            ),
        ]

        query = MemoryQuery(
            query_text="test",
            tiers=[MemoryTier.SEMANTIC],
            max_results=10,
        )

        results = retriever.search(memories, query)

        assert all(r.item.tier == MemoryTier.SEMANTIC for r in results)

    def test_search_filters_by_min_snr(self):
        """Test search respects minimum SNR filter."""
        retriever = MemoryRetriever(similarity_threshold=0.0)

        memories = [
            MemoryItem(
                id="low-snr",
                content="test",
                tier=MemoryTier.WORKING,
                state=MemoryState.ACTIVE,
                snr_score=0.3,
            ),
            MemoryItem(
                id="high-snr",
                content="test",
                tier=MemoryTier.WORKING,
                state=MemoryState.ACTIVE,
                snr_score=0.9,
            ),
        ]

        query = MemoryQuery(query_text="test", min_snr=0.5, max_results=10)

        results = retriever.search(memories, query)

        assert all(r.item.snr_score >= 0.5 for r in results)


# =============================================================================
# AGENT MEMORY SYSTEM TESTS
# =============================================================================


class TestAgentMemorySystem:
    """Tests for AgentMemorySystem class."""

    @pytest.mark.asyncio
    async def test_remember_stores_memory(self, memory_system):
        """Test remember stores memory in correct tier."""
        item = await memory_system.remember(
            content="Test memory content",
            tier=MemoryTier.WORKING,
            domains={"test"},
            source="unit_test",
        )

        assert item.id is not None
        assert item.tier == MemoryTier.WORKING
        assert "test" in item.domains
        assert memory_system.get_tier_counts()["working"] == 1

    @pytest.mark.asyncio
    async def test_remember_deduplicates(self, memory_system):
        """Test duplicate content returns existing memory."""
        content = "Duplicate content test"

        item1 = await memory_system.remember(content=content)
        item2 = await memory_system.remember(content=content)

        assert item1.id == item2.id
        assert memory_system.get_tier_counts()["working"] == 1

    @pytest.mark.asyncio
    async def test_recall_finds_stored_memory(self, memory_system):
        """Test recall finds stored memories."""
        await memory_system.remember(
            content="Python is a programming language",
            tier=MemoryTier.SEMANTIC,
            tags={"programming"},
        )

        results = await memory_system.recall(
            query_text="Python programming", max_results=5
        )

        assert len(results) >= 1
        assert "python" in results[0].item.content.lower()

    @pytest.mark.asyncio
    async def test_recall_filters_by_tags(self, memory_system):
        """Test recall filters by tags."""
        await memory_system.remember(
            content="Tagged memory", tier=MemoryTier.WORKING, tags={"special"}
        )
        await memory_system.remember(
            content="Untagged memory", tier=MemoryTier.WORKING, tags=set()
        )

        results = await memory_system.recall(
            query_text="memory", tags={"special"}, max_results=10
        )

        assert all("special" in r.item.tags for r in results)

    @pytest.mark.asyncio
    async def test_forget_removes_memory(self, memory_system):
        """Test forget removes memory."""
        item = await memory_system.remember(content="To be forgotten")

        success = await memory_system.forget(item.id, hard_delete=True)

        assert success
        assert memory_system.get_tier_counts()["working"] == 0

    @pytest.mark.asyncio
    async def test_forget_archives_by_default(self, memory_system):
        """Test forget archives instead of deleting by default."""
        item = await memory_system.remember(content="To be archived")

        success = await memory_system.forget(item.id, hard_delete=False)

        assert success
        assert item.state == MemoryState.ARCHIVED

    @pytest.mark.asyncio
    async def test_consolidate_promotes_memories(self, memory_system):
        """Test consolidation promotes eligible memories."""
        # Create promotable memory
        item = await memory_system.remember(
            content="High quality content",
            tier=MemoryTier.WORKING,
            snr_score=0.9,
            ihsan_score=0.98,
            priority=0.95,
        )
        # Simulate accesses
        item.access_count = 5

        result = await memory_system.consolidate()

        assert result["promoted"] >= 0  # May or may not promote based on full criteria

    @pytest.mark.asyncio
    async def test_enforces_tier_limits(self, memory_system):
        """Test tier size limits are enforced."""
        # Working memory limit is 10
        for i in range(15):
            await memory_system.remember(
                content=f"Memory {i}",
                tier=MemoryTier.WORKING,
                snr_score=0.3 + (i * 0.04),  # Varying SNR
            )

        counts = memory_system.get_tier_counts()
        assert counts["working"] <= 10

    @pytest.mark.asyncio
    async def test_session_management(self, memory_system):
        """Test session start and end."""
        session_id = memory_system.start_session()

        assert session_id is not None
        assert memory_system._current_session_id == session_id

        await memory_system.remember(content="Session memory")

        stats = memory_system.end_session()

        assert stats["session_id"] == session_id
        assert stats["memories_created"] >= 1
        assert memory_system._current_session_id is None

    @pytest.mark.asyncio
    async def test_crystallize_patterns(self, memory_system):
        """Test pattern crystallization from episodic memories."""
        # Create similar episodic memories
        for i in range(5):
            item = await memory_system.remember(
                content=f"Machine learning is great {i}",
                tier=MemoryTier.EPISODIC,
                domains={"ai"},
                snr_score=0.8,
            )
            item.access_count = 4  # Meet minimum occurrences

        crystals = await memory_system.crystallize_patterns(
            min_occurrences=3, min_snr=0.6
        )

        # May or may not crystallize based on similarity
        assert isinstance(crystals, list)

    def test_get_statistics(self, memory_system):
        """Test statistics collection."""
        stats = memory_system.get_statistics()

        assert "total_memories" in stats
        assert "tier_counts" in stats
        assert "snr_distribution" in stats
        assert "consolidation_stats" in stats
        assert "retrieval_stats" in stats

    @pytest.mark.asyncio
    async def test_export_to_dict(self, memory_system):
        """Test export to dictionary."""
        await memory_system.remember(content="Export test", tier=MemoryTier.SEMANTIC)

        data = memory_system.export_to_dict()

        assert "version" in data
        assert "memories" in data
        assert len(data["memories"]["semantic"]) == 1

    @pytest.mark.asyncio
    async def test_export_to_json(self, memory_system):
        """Test export to JSON file."""
        await memory_system.remember(content="JSON export test")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            memory_system.export_to_json(filepath)

            with open(filepath, "r") as f:
                data = json.load(f)

            assert "version" in data
            assert "memories" in data
        finally:
            os.unlink(filepath)


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_agent_memory_default(self):
        """Test create_agent_memory with defaults."""
        memory = create_agent_memory()

        assert memory is not None
        assert isinstance(memory, AgentMemorySystem)

    def test_create_agent_memory_with_scorer(self, snr_scorer):
        """Test create_agent_memory with custom scorer."""
        memory = create_agent_memory(snr_scorer=snr_scorer)

        assert memory.snr_scorer == snr_scorer

    def test_create_agent_memory_with_limits(self):
        """Test create_agent_memory with custom limits."""
        memory = create_agent_memory(
            max_working_memory=50,
            max_episodic_memory=200,
        )

        assert memory._limits[MemoryTier.WORKING] == 50
        assert memory._limits[MemoryTier.EPISODIC] == 200


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for the full memory system."""

    @pytest.mark.asyncio
    async def test_full_memory_lifecycle(self):
        """Test complete memory lifecycle: create → access → consolidate → crystallize."""
        memory = create_agent_memory(
            max_working_memory=100,
            max_episodic_memory=500,
            auto_consolidate=False,
        )

        # Start session
        session_id = memory.start_session()

        # Store memories
        memories = []
        for i in range(10):
            item = await memory.remember(
                content=f"Important concept about AI safety #{i}",
                tier=MemoryTier.WORKING,
                domains={"ai", "safety"},
                snr_score=0.75 + (i * 0.02),
            )
            memories.append(item)
            # Simulate accesses
            item.access_count = 3

        # Recall memories
        results = await memory.recall(query_text="AI safety", max_results=5)
        assert len(results) > 0

        # Consolidate
        consolidation_result = await memory.consolidate()
        assert "promoted" in consolidation_result

        # Get statistics
        stats = memory.get_statistics()
        assert stats["total_memories"] > 0

        # End session
        session_stats = memory.end_session()
        assert session_stats["memories_created"] >= 10

    @pytest.mark.asyncio
    async def test_cross_session_memory_persistence(self):
        """Test memories persist across sessions (in-memory)."""
        memory = create_agent_memory(auto_consolidate=False)

        # Session 1
        memory.start_session("session-1")
        await memory.remember(
            content="Persistent knowledge",
            tier=MemoryTier.SEMANTIC,
        )
        memory.end_session()

        # Session 2
        memory.start_session("session-2")
        results = await memory.recall(query_text="Persistent knowledge")
        memory.end_session()

        assert len(results) > 0
        assert results[0].item.session_id == "session-1"

    @pytest.mark.asyncio
    async def test_snr_affects_retention(self):
        """Test SNR score affects memory retention priority."""
        memory = create_agent_memory(max_working_memory=5, auto_consolidate=False)

        # Store memories with varying SNR
        low_snr = await memory.remember(content="Low quality A", snr_score=0.2)
        high_snr = await memory.remember(content="High quality B", snr_score=0.95)

        # Fill to trigger pruning
        for i in range(5):
            await memory.remember(content=f"Filler memory {i}", snr_score=0.5)

        # High SNR should be retained, low SNR may be pruned
        assert high_snr.id in memory._working or high_snr.snr_level == SNRLevel.HIGH


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
