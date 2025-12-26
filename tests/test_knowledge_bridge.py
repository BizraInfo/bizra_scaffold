r"""
Tests for BIZRA Knowledge Graph Bridge.

Comprehensive test suite covering:
- Knowledge node creation and graph operations
- Knowledge extraction from files
- Graph algorithms (BFS, path finding)
- Bridge integration with data lake
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.knowledge_bridge import (
    NodeType,
    EdgeType,
    ExtractionMethod,
    KnowledgeNode,
    KnowledgeEdge,
    ExtractionResult,
    GraphStatistics,
    KnowledgeGraph,
    KnowledgeExtractor,
    KnowledgeGraphBridge,
    create_knowledge_bridge,
)


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE NODE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeNode:
    """Tests for KnowledgeNode class."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = KnowledgeNode(
            id="node-1",
            node_type=NodeType.CONCEPT,
            label="TestConcept",
        )
        
        assert node.id == "node-1"
        assert node.node_type == NodeType.CONCEPT
        assert node.label == "TestConcept"
    
    def test_node_with_source(self):
        """Test node with source file."""
        node = KnowledgeNode(
            id="node-2",
            node_type=NodeType.ENTITY,
            label="TestEntity",
            source_file="/path/to/file.py",
            source_line=42,
        )
        
        assert node.source_file == "/path/to/file.py"
        assert node.source_line == 42
    
    def test_node_with_properties(self):
        """Test node with custom properties."""
        node = KnowledgeNode(
            id="node-3",
            node_type=NodeType.TOPIC,
            label="TestTopic",
            properties=frozenset([("key1", "value1"), ("key2", "value2")]),
        )
        
        props = dict(node.properties)
        assert props["key1"] == "value1"
        assert props["key2"] == "value2"
    
    def test_node_signal_strength(self):
        """Test node signal strength."""
        node = KnowledgeNode(
            id="node-4",
            node_type=NodeType.CONCEPT,
            label="HighSignal",
            signal_strength=0.95,
        )
        
        assert node.signal_strength == 0.95
    
    def test_node_to_dict(self):
        """Test node serialization."""
        node = KnowledgeNode(
            id="node-5",
            node_type=NodeType.FILE,
            label="test.py",
            signal_strength=0.8,
        )
        
        data = node.to_dict()
        
        assert data["id"] == "node-5"
        assert data["type"] == "FILE"
        assert data["label"] == "test.py"
        assert data["signal_strength"] == 0.8
    
    def test_node_hash(self):
        """Test node hashing for set operations."""
        node1 = KnowledgeNode(id="same-id", node_type=NodeType.CONCEPT, label="A")
        node2 = KnowledgeNode(id="same-id", node_type=NodeType.CONCEPT, label="B")
        
        assert hash(node1) == hash(node2)  # Same ID = same hash


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE EDGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeEdge:
    """Tests for KnowledgeEdge class."""
    
    def test_edge_creation(self):
        """Test basic edge creation."""
        edge = KnowledgeEdge(
            source_id="node-1",
            target_id="node-2",
            edge_type=EdgeType.CONTAINS,
        )
        
        assert edge.source_id == "node-1"
        assert edge.target_id == "node-2"
        assert edge.edge_type == EdgeType.CONTAINS
    
    def test_edge_with_weight(self):
        """Test edge with custom weight."""
        edge = KnowledgeEdge(
            source_id="node-1",
            target_id="node-2",
            edge_type=EdgeType.RELATED_TO,
            weight=0.9,
        )
        
        assert edge.weight == 0.9
    
    def test_edge_with_evidence(self):
        """Test edge with evidence."""
        edge = KnowledgeEdge(
            source_id="node-1",
            target_id="node-2",
            edge_type=EdgeType.DERIVED_FROM,
            evidence="Found in line 42 of source file",
        )
        
        assert edge.evidence == "Found in line 42 of source file"
    
    def test_edge_to_dict(self):
        """Test edge serialization."""
        edge = KnowledgeEdge(
            source_id="src",
            target_id="tgt",
            edge_type=EdgeType.PART_OF,
            weight=0.7,
        )
        
        data = edge.to_dict()
        
        assert data["source"] == "src"
        assert data["target"] == "tgt"
        assert data["type"] == "PART_OF"
        assert data["weight"] == 0.7


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph class."""
    
    @pytest.fixture
    def graph(self):
        """Create graph for testing."""
        return KnowledgeGraph()
    
    @pytest.fixture
    def populated_graph(self, graph):
        """Create graph with sample data."""
        # Add nodes
        nodes = [
            KnowledgeNode(id="n1", node_type=NodeType.CONCEPT, label="Concept1", signal_strength=0.8),
            KnowledgeNode(id="n2", node_type=NodeType.CONCEPT, label="Concept2", signal_strength=0.7),
            KnowledgeNode(id="n3", node_type=NodeType.ENTITY, label="Entity1", signal_strength=0.9),
            KnowledgeNode(id="n4", node_type=NodeType.FILE, label="file.py", signal_strength=0.6),
        ]
        for node in nodes:
            graph.add_node(node)
        
        # Add edges
        edges = [
            KnowledgeEdge(source_id="n4", target_id="n1", edge_type=EdgeType.CONTAINS),
            KnowledgeEdge(source_id="n4", target_id="n2", edge_type=EdgeType.CONTAINS),
            KnowledgeEdge(source_id="n1", target_id="n2", edge_type=EdgeType.RELATED_TO),
            KnowledgeEdge(source_id="n2", target_id="n3", edge_type=EdgeType.IMPLIES),
        ]
        for edge in edges:
            graph.add_edge(edge)
        
        return graph
    
    # Node operations
    
    def test_add_node(self, graph):
        """Test adding a node."""
        node = KnowledgeNode(id="test", node_type=NodeType.CONCEPT, label="Test")
        result = graph.add_node(node)
        
        assert result is True
        assert graph.get_node("test") == node
    
    def test_add_duplicate_node(self, graph):
        """Test adding duplicate node fails."""
        node = KnowledgeNode(id="dup", node_type=NodeType.CONCEPT, label="Test")
        graph.add_node(node)
        result = graph.add_node(node)
        
        assert result is False
    
    def test_get_node(self, populated_graph):
        """Test getting a node by ID."""
        node = populated_graph.get_node("n1")
        
        assert node is not None
        assert node.label == "Concept1"
    
    def test_get_nonexistent_node(self, graph):
        """Test getting nonexistent node returns None."""
        node = graph.get_node("nonexistent")
        assert node is None
    
    def test_remove_node(self, populated_graph):
        """Test removing a node."""
        result = populated_graph.remove_node("n3")
        
        assert result is True
        assert populated_graph.get_node("n3") is None
    
    def test_get_nodes_by_type(self, populated_graph):
        """Test getting nodes by type."""
        concepts = populated_graph.get_nodes_by_type(NodeType.CONCEPT)
        
        assert len(concepts) == 2
        assert all(n.node_type == NodeType.CONCEPT for n in concepts)
    
    # Edge operations
    
    def test_add_edge(self, graph):
        """Test adding an edge."""
        graph.add_node(KnowledgeNode(id="a", node_type=NodeType.CONCEPT, label="A"))
        graph.add_node(KnowledgeNode(id="b", node_type=NodeType.CONCEPT, label="B"))
        
        edge = KnowledgeEdge(source_id="a", target_id="b", edge_type=EdgeType.RELATED_TO)
        result = graph.add_edge(edge)
        
        assert result is True
    
    def test_add_edge_missing_nodes(self, graph):
        """Test adding edge with missing nodes fails."""
        edge = KnowledgeEdge(source_id="missing1", target_id="missing2", edge_type=EdgeType.RELATED_TO)
        result = graph.add_edge(edge)
        
        assert result is False
    
    def test_get_edge(self, populated_graph):
        """Test getting an edge."""
        edge = populated_graph.get_edge("n4", "n1", EdgeType.CONTAINS)
        
        assert edge is not None
        assert edge.edge_type == EdgeType.CONTAINS
    
    def test_remove_edge(self, populated_graph):
        """Test removing an edge."""
        result = populated_graph.remove_edge("n1", "n2", EdgeType.RELATED_TO)
        
        assert result is True
        assert populated_graph.get_edge("n1", "n2") is None
    
    def test_get_neighbors(self, populated_graph):
        """Test getting neighbor nodes."""
        neighbors = populated_graph.get_neighbors("n2")
        
        assert len(neighbors) >= 2  # n1 and n3 at minimum
    
    def test_get_outgoing_edges(self, populated_graph):
        """Test getting outgoing edges."""
        edges = populated_graph.get_outgoing_edges("n4")
        
        assert len(edges) == 2  # Contains n1 and n2
    
    # Graph algorithms
    
    def test_bfs(self, populated_graph):
        """Test breadth-first search."""
        results = populated_graph.bfs("n4", max_depth=3)
        
        assert len(results) > 0
        assert results[0][0].id == "n4"  # Start node
        assert results[0][1] == 0  # Depth 0
    
    def test_find_path(self, populated_graph):
        """Test path finding."""
        path = populated_graph.find_path("n4", "n3")
        
        assert path is not None
        assert path[0].id == "n4"
        assert path[-1].id == "n3"
    
    def test_find_path_no_connection(self, graph):
        """Test path finding with disconnected nodes."""
        graph.add_node(KnowledgeNode(id="x", node_type=NodeType.CONCEPT, label="X"))
        graph.add_node(KnowledgeNode(id="y", node_type=NodeType.CONCEPT, label="Y"))
        
        path = graph.find_path("x", "y")
        assert path is None
    
    def test_find_related(self, populated_graph):
        """Test finding related nodes."""
        related = populated_graph.find_related("n4", max_results=5)
        
        assert len(related) > 0
        assert all(score > 0 for _, score in related)
    
    # Statistics
    
    def test_get_statistics(self, populated_graph):
        """Test graph statistics."""
        stats = populated_graph.get_statistics()
        
        assert stats.node_count == 4
        assert stats.edge_count == 4
        assert stats.avg_signal_strength > 0
    
    # Serialization
    
    def test_to_dict(self, populated_graph):
        """Test graph serialization."""
        data = populated_graph.to_dict()
        
        assert "nodes" in data
        assert "edges" in data
        assert "statistics" in data
        assert len(data["nodes"]) == 4
    
    def test_save_and_load(self, populated_graph, tmp_path):
        """Test saving and loading graph."""
        save_path = tmp_path / "test_graph.json"
        populated_graph.save(save_path)
        
        loaded = KnowledgeGraph.load(save_path)
        
        assert loaded.get_statistics().node_count == 4
        assert loaded.get_node("n1") is not None


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE EXTRACTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeExtractor:
    """Tests for KnowledgeExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor for testing."""
        return KnowledgeExtractor(min_signal_strength=0.3)
    
    def test_can_extract_python(self, extractor):
        """Test Python file is extractable."""
        path = Path("test.py")
        assert extractor.can_extract(path) is True
    
    def test_can_extract_markdown(self, extractor):
        """Test Markdown file is extractable."""
        path = Path("README.md")
        assert extractor.can_extract(path) is True
    
    def test_cannot_extract_binary(self, extractor):
        """Test binary file is not extractable."""
        path = Path("image.png")
        assert extractor.can_extract(path) is False
    
    @pytest.mark.asyncio
    async def test_extract_from_python_file(self, extractor, tmp_path):
        """Test extraction from Python file."""
        # Create test file
        test_file = tmp_path / "test_module.py"
        test_file.write_text("""
class TestClass:
    '''A test class.'''
    pass

def test_function():
    '''A test function.'''
    pass

# TODO: Important note here
""")
        
        result = await extractor.extract_from_file(test_file, snr_score=0.8)
        
        assert result.source_file == str(test_file)
        assert len(result.nodes) > 0  # File node + concepts
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_extract_from_nonexistent_file(self, extractor):
        """Test extraction from nonexistent file."""
        result = await extractor.extract_from_file(Path("/nonexistent/file.py"))
        
        assert len(result.nodes) == 0
        assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_extract_concepts(self, extractor, tmp_path):
        """Test concept extraction."""
        test_file = tmp_path / "concepts.py"
        test_file.write_text("""
class DataProcessor:
    pass

class EventHandler:
    pass

def process_data():
    pass
""")
        
        result = await extractor.extract_from_file(test_file, snr_score=0.7)
        
        # Should find: file node + DataProcessor + EventHandler + process_data
        concept_nodes = [n for n in result.nodes if n.node_type == NodeType.CONCEPT]
        assert len(concept_nodes) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH BRIDGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeGraphBridge:
    """Tests for KnowledgeGraphBridge class."""
    
    @pytest.fixture
    def bridge(self):
        """Create bridge for testing."""
        return KnowledgeGraphBridge()
    
    def test_initialization(self, bridge):
        """Test bridge initialization."""
        assert bridge.graph is not None
        assert bridge.extractor is not None
        assert bridge.VERSION == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_process_file(self, bridge, tmp_path):
        """Test processing a single file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("class TestClass: pass")
        
        result = await bridge.process_file(test_file, snr_score=0.8)
        
        assert result.source_file == str(test_file)
        assert bridge._files_processed == 1
    
    @pytest.mark.asyncio
    async def test_process_directory(self, bridge, tmp_path):
        """Test processing a directory."""
        # Create test files
        (tmp_path / "file1.py").write_text("class File1Class: pass")
        (tmp_path / "file2.py").write_text("class File2Class: pass")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.py").write_text("class File3Class: pass")
        
        results = await bridge.process_directory(tmp_path, recursive=True)
        
        assert len(results) == 3
        assert bridge._files_processed == 3
    
    def test_search(self, bridge):
        """Test knowledge search."""
        # Add some nodes directly
        bridge.graph.add_node(KnowledgeNode(
            id="search-1",
            node_type=NodeType.CONCEPT,
            label="TestConcept",
            signal_strength=0.8,
        ))
        bridge.graph.add_node(KnowledgeNode(
            id="search-2",
            node_type=NodeType.CONCEPT,
            label="AnotherTest",
            signal_strength=0.7,
        ))
        
        results = bridge.search("Test")
        
        assert len(results) == 2
        assert all("test" in node.label.lower() for node, _ in results)
    
    def test_search_no_results(self, bridge):
        """Test search with no results."""
        results = bridge.search("nonexistent")
        assert len(results) == 0
    
    def test_find_related(self, bridge):
        """Test finding related nodes."""
        # Add nodes and edges
        bridge.graph.add_node(KnowledgeNode(id="rel-1", node_type=NodeType.CONCEPT, label="A"))
        bridge.graph.add_node(KnowledgeNode(id="rel-2", node_type=NodeType.CONCEPT, label="B"))
        bridge.graph.add_edge(KnowledgeEdge(
            source_id="rel-1",
            target_id="rel-2",
            edge_type=EdgeType.RELATED_TO,
        ))
        
        related = bridge.find_related("rel-1")
        
        assert len(related) >= 1
    
    def test_get_statistics(self, bridge):
        """Test getting bridge statistics."""
        stats = bridge.get_statistics()
        
        assert "version" in stats
        assert "files_processed" in stats
        assert "nodes_created" in stats
        assert "graph" in stats
    
    def test_save_and_load_graph(self, bridge, tmp_path):
        """Test saving and loading graph."""
        # Add some data
        bridge.graph.add_node(KnowledgeNode(
            id="save-1",
            node_type=NodeType.CONCEPT,
            label="SaveTest",
        ))
        
        save_path = tmp_path / "bridge_graph.json"
        bridge.save_graph(save_path)
        
        # Create new bridge and load
        new_bridge = KnowledgeGraphBridge()
        new_bridge.load_graph(save_path)
        
        assert new_bridge.graph.get_node("save-1") is not None


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFactory:
    """Tests for factory functions."""
    
    def test_create_knowledge_bridge(self):
        """Test bridge factory without watcher."""
        bridge = create_knowledge_bridge()
        
        assert isinstance(bridge, KnowledgeGraphBridge)
        assert bridge.watcher is None
    
    def test_create_knowledge_bridge_with_watcher(self):
        """Test bridge factory with mock watcher."""
        mock_watcher = MagicMock()
        mock_watcher.add_listener = MagicMock()
        
        bridge = create_knowledge_bridge(watcher=mock_watcher)
        
        assert bridge.watcher == mock_watcher


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for knowledge bridge."""
    
    @pytest.mark.asyncio
    async def test_full_extraction_pipeline(self, tmp_path):
        """Test complete extraction pipeline."""
        # Create test project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("""
class Application:
    '''Main application class.'''
    
    def run(self):
        '''Run the application.'''
        pass

class Database:
    '''Database connection handler.'''
    pass
""")
        
        (tmp_path / "README.md").write_text("""
# Test Project

This is a test project.

## Features

- Feature 1
- Feature 2
""")
        
        # Process with bridge
        bridge = create_knowledge_bridge()
        results = await bridge.process_directory(tmp_path, recursive=True)
        
        assert len(results) == 2  # main.py and README.md
        
        # Search for concepts
        app_results = bridge.search("Application")
        assert len(app_results) >= 1
        
        # Get statistics
        stats = bridge.get_statistics()
        assert stats["files_processed"] == 2
        assert stats["nodes_created"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
