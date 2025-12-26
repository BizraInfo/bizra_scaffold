r"""
BIZRA AEON OMEGA - Knowledge Graph Bridge
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Pattern: Semantic Knowledge Integration & Discovery

The Knowledge Graph Bridge connects the Data Lake Watcher to the Graph of
Thoughts system, enabling:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                    KNOWLEDGE GRAPH BRIDGE                                │
  │                                                                          │
  │  ┌──────────────────┐                    ┌──────────────────────────┐   │
  │  │   DATA LAKE      │     ─────────►     │    GRAPH OF THOUGHTS     │   │
  │  │   WATCHER        │                    │                          │   │
  │  │                  │                    │  ┌────┐   ┌────┐         │   │
  │  │  523K Files      │                    │  │Node│───│Node│         │   │
  │  │  - Manifests     │    Knowledge       │  └─┬──┘   └──┬─┘         │   │
  │  │  - Checksums     │    Extraction      │    │    ╲    │           │   │
  │  │  - SNR Scores    │    ─────────►      │  ┌─┴──┐  ┌┴───┐          │   │
  │  │                  │                    │  │Node│──│Node│          │   │
  │  └──────────────────┘                    │  └────┘  └────┘          │   │
  │                                          └──────────────────────────┘   │
  │                                                                          │
  │  Features:                                                               │
  │  ═════════════════════════════════════════════════════════════════════  │
  │  • Semantic Fingerprinting: Extract concepts from file content          │
  │  • Entity Recognition: Identify key entities in knowledge files         │
  │  • Relationship Discovery: Build knowledge graph edges                  │
  │  • Cross-Domain Insights: Connect disparate knowledge domains           │
  │  • SNR-Weighted Ranking: Prioritize high-signal knowledge              │
  └─────────────────────────────────────────────────────────────────────────┘

BIZRA SOT Compliance:
  - Section 3 (Invariants): Knowledge extraction respects IM ≥ 0.95
  - Section 7 (Evidence Policy): All extractions are evidence-backed
  - Section 10 (Knowledge Management): Proper ontology adherence

Author: BIZRA Genesis Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, payload: Dict[str, Any], indent: int = 2) -> None:
    """Atomically write JSON data to disk with a write-then-rename pattern."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    tmp_path = Path(tmp_file.name)
    try:
        json.dump(payload, tmp_file, indent=indent, ensure_ascii=False)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_file.close()

        try:
            os.chmod(tmp_path, 0o644)
        except OSError:
            pass

        tmp_path.replace(path)
    except Exception:
        try:
            tmp_file.close()
        except Exception:
            pass
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""

    CONCEPT = auto()  # Abstract concept
    ENTITY = auto()  # Named entity
    FILE = auto()  # Source file reference
    TOPIC = auto()  # Topic cluster
    DOMAIN = auto()  # Knowledge domain
    RELATIONSHIP = auto()  # Reified relationship


class EdgeType(Enum):
    """Types of edges in the knowledge graph."""

    CONTAINS = auto()  # File contains concept
    RELATED_TO = auto()  # Semantic relationship
    DERIVED_FROM = auto()  # Derivation relationship
    SIMILAR_TO = auto()  # Similarity relationship
    PART_OF = auto()  # Composition relationship
    IMPLIES = auto()  # Logical implication


class ExtractionMethod(Enum):
    """Methods for knowledge extraction."""

    KEYWORD = auto()  # Keyword extraction
    PATTERN = auto()  # Pattern matching
    SEMANTIC = auto()  # Semantic analysis
    STRUCTURAL = auto()  # Structure-based


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class KnowledgeNode:
    """
    Immutable node in the knowledge graph.

    Represents a unit of knowledge extracted from the data lake.
    """

    id: str
    node_type: NodeType
    label: str

    # Source provenance
    source_file: Optional[str] = None
    source_line: Optional[int] = None

    # Metadata
    properties: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)

    # SNR weighting
    signal_strength: float = 0.5

    # Temporal
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node."""
        return {
            "id": self.id,
            "type": self.node_type.name,
            "label": self.label,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "properties": dict(self.properties),
            "signal_strength": self.signal_strength,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class KnowledgeEdge:
    """
    Immutable edge in the knowledge graph.

    Represents a relationship between knowledge nodes.
    """

    source_id: str
    target_id: str
    edge_type: EdgeType

    # Weight (0.0-1.0)
    weight: float = 0.5

    # Metadata
    properties: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)

    # Evidence
    evidence: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.edge_type))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge."""
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.name,
            "weight": self.weight,
            "properties": dict(self.properties),
            "evidence": self.evidence,
        }


@dataclass
class ExtractionResult:
    """Result of knowledge extraction from a file."""

    source_file: str
    nodes: List[KnowledgeNode]
    edges: List[KnowledgeEdge]

    # Metrics
    extraction_time_ms: float = 0.0
    method: ExtractionMethod = ExtractionMethod.KEYWORD

    # Quality
    confidence: float = 0.5
    snr_score: float = 0.5


@dataclass
class GraphStatistics:
    """Statistics about the knowledge graph."""

    node_count: int
    edge_count: int
    node_type_distribution: Dict[str, int]
    edge_type_distribution: Dict[str, int]
    avg_signal_strength: float
    density: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize statistics."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "node_types": self.node_type_distribution,
            "edge_types": self.edge_type_distribution,
            "avg_signal_strength": self.avg_signal_strength,
            "density": self.density,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════


class KnowledgeGraph:
    """
    In-memory knowledge graph for semantic knowledge representation.

    Provides:
    - Node/edge storage with efficient lookup
    - Graph traversal algorithms
    - Semantic similarity computation
    - SNR-weighted ranking
    """

    def __init__(self):
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._edges: Dict[Tuple[str, str, EdgeType], KnowledgeEdge] = {}

        # Indexes
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._nodes_by_type: Dict[NodeType, Set[str]] = defaultdict(set)
        self._nodes_by_source: Dict[str, Set[str]] = defaultdict(set)

    # ═══════════════════════════════════════════════════════════════════════════
    # NODE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def add_node(self, node: KnowledgeNode) -> bool:
        """
        Add a node to the graph.

        Returns True if node was added, False if it already exists.
        """
        if node.id in self._nodes:
            return False

        self._nodes[node.id] = node
        self._nodes_by_type[node.node_type].add(node.id)

        if node.source_file:
            self._nodes_by_source[node.source_file].add(node.id)

        return True

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all connected edges."""
        if node_id not in self._nodes:
            return False

        node = self._nodes[node_id]

        # Remove connected edges
        for target in list(self._adjacency[node_id]):
            self.remove_edge(node_id, target)
        for source in list(self._reverse_adjacency[node_id]):
            self.remove_edge(source, node_id)

        # Remove from indexes
        self._nodes_by_type[node.node_type].discard(node_id)
        if node.source_file:
            self._nodes_by_source[node.source_file].discard(node_id)

        del self._nodes[node_id]
        return True

    def get_nodes_by_type(self, node_type: NodeType) -> List[KnowledgeNode]:
        """Get all nodes of a specific type."""
        return [self._nodes[nid] for nid in self._nodes_by_type[node_type]]

    def get_nodes_by_source(self, source_file: str) -> List[KnowledgeNode]:
        """Get all nodes extracted from a specific file."""
        return [self._nodes[nid] for nid in self._nodes_by_source[source_file]]

    # ═══════════════════════════════════════════════════════════════════════════
    # EDGE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def add_edge(self, edge: KnowledgeEdge) -> bool:
        """
        Add an edge to the graph.

        Returns True if edge was added, False if it already exists.
        """
        key = (edge.source_id, edge.target_id, edge.edge_type)
        if key in self._edges:
            return False

        # Ensure nodes exist
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            return False

        self._edges[key] = edge
        self._adjacency[edge.source_id].add(edge.target_id)
        self._reverse_adjacency[edge.target_id].add(edge.source_id)

        return True

    def get_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: Optional[EdgeType] = None,
    ) -> Optional[KnowledgeEdge]:
        """Get an edge by source, target, and optionally type."""
        if edge_type:
            return self._edges.get((source_id, target_id, edge_type))

        # Find any edge between source and target
        for et in EdgeType:
            edge = self._edges.get((source_id, target_id, et))
            if edge:
                return edge
        return None

    def remove_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: Optional[EdgeType] = None,
    ) -> bool:
        """Remove an edge from the graph."""
        if edge_type:
            keys_to_remove = [(source_id, target_id, edge_type)]
        else:
            keys_to_remove = [
                (source_id, target_id, et)
                for et in EdgeType
                if (source_id, target_id, et) in self._edges
            ]

        removed = False
        for key in keys_to_remove:
            if key in self._edges:
                del self._edges[key]
                removed = True

        if removed:
            has_remaining = any(
                (source_id, target_id, et) in self._edges for et in EdgeType
            )
            if not has_remaining:
                self._adjacency[source_id].discard(target_id)
                self._reverse_adjacency[target_id].discard(source_id)

        return removed

    def get_neighbors(self, node_id: str) -> List[KnowledgeNode]:
        """Get all nodes connected to the given node."""
        neighbor_ids = self._adjacency[node_id] | self._reverse_adjacency[node_id]
        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    def get_outgoing_edges(self, node_id: str) -> List[KnowledgeEdge]:
        """Get all outgoing edges from a node."""
        return [edge for edge in self._edges.values() if edge.source_id == node_id]

    def get_incoming_edges(self, node_id: str) -> List[KnowledgeEdge]:
        """Get all incoming edges to a node."""
        return [edge for edge in self._edges.values() if edge.target_id == node_id]

    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH ALGORITHMS
    # ═══════════════════════════════════════════════════════════════════════════

    def bfs(
        self,
        start_id: str,
        max_depth: int = 10,
    ) -> List[Tuple[KnowledgeNode, int]]:
        """
        Breadth-first search from a starting node.

        Returns list of (node, depth) tuples.
        """
        if start_id not in self._nodes:
            return []

        visited = {start_id}
        queue = [(start_id, 0)]
        result = [(self._nodes[start_id], 0)]

        while queue:
            current_id, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            for neighbor_id in self._adjacency[current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
                    result.append((self._nodes[neighbor_id], depth + 1))

        return result

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 10,
    ) -> Optional[List[KnowledgeNode]]:
        """
        Find shortest path between two nodes.

        Returns list of nodes in path, or None if no path exists.
        """
        if start_id not in self._nodes or end_id not in self._nodes:
            return None

        if start_id == end_id:
            return [self._nodes[start_id]]

        visited = {start_id}
        queue = [(start_id, [start_id])]

        while queue:
            current_id, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            for neighbor_id in self._adjacency[current_id]:
                if neighbor_id == end_id:
                    return [self._nodes[nid] for nid in path + [neighbor_id]]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def find_related(
        self,
        node_id: str,
        max_results: int = 10,
    ) -> List[Tuple[KnowledgeNode, float]]:
        """
        Find nodes most related to the given node.

        Uses SNR-weighted edge traversal.
        Returns list of (node, relevance_score) tuples.
        """
        if node_id not in self._nodes:
            return []

        # Collect all reachable nodes with weighted scores
        scores: Dict[str, float] = defaultdict(float)

        for node, depth in self.bfs(node_id, max_depth=3):
            if node.id == node_id:
                continue

            # Score based on depth and signal strength
            depth_factor = 1.0 / (depth + 1)
            scores[node.id] += depth_factor * node.signal_strength

        # Sort by score and return top results
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self._nodes[nid], score) for nid, score in sorted_nodes[:max_results]]

    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════

    def get_statistics(self) -> GraphStatistics:
        """Compute graph statistics."""
        node_count = len(self._nodes)
        edge_count = len(self._edges)

        # Type distributions
        node_type_dist = {
            nt.name: len(nids) for nt, nids in self._nodes_by_type.items()
        }

        edge_type_dist: Dict[str, int] = defaultdict(int)
        for edge in self._edges.values():
            edge_type_dist[edge.edge_type.name] += 1

        # Average signal strength
        avg_signal = 0.0
        if node_count > 0:
            avg_signal = (
                sum(n.signal_strength for n in self._nodes.values()) / node_count
            )

        # Graph density
        max_edges = node_count * (node_count - 1)
        density = edge_count / max_edges if max_edges > 0 else 0.0

        return GraphStatistics(
            node_count=node_count,
            edge_count=edge_count,
            node_type_distribution=node_type_dist,
            edge_type_distribution=dict(edge_type_dist),
            avg_signal_strength=avg_signal,
            density=density,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SERIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
            "statistics": self.get_statistics().to_dict(),
        }

    def save(self, path: Path) -> None:
        """Save graph to JSON file."""
        _atomic_write_json(path, self.to_dict(), indent=2)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        """Load graph from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = cls()

        # Load nodes
        for node_data in data.get("nodes", []):
            node = KnowledgeNode(
                id=node_data["id"],
                node_type=NodeType[node_data["type"]],
                label=node_data["label"],
                source_file=node_data.get("source_file"),
                source_line=node_data.get("source_line"),
                properties=frozenset(node_data.get("properties", {}).items()),
                signal_strength=node_data.get("signal_strength", 0.5),
                created_at=node_data.get(
                    "created_at",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            graph.add_node(node)

        # Load edges
        for edge_data in data.get("edges", []):
            edge = KnowledgeEdge(
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                edge_type=EdgeType[edge_data["type"]],
                weight=edge_data.get("weight", 0.5),
                properties=frozenset(edge_data.get("properties", {}).items()),
                evidence=edge_data.get("evidence"),
            )
            graph.add_edge(edge)

        return graph


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════


class KnowledgeExtractor:
    """
    Extracts knowledge nodes and edges from file content.

    Supports multiple extraction methods:
    - Keyword extraction
    - Pattern matching
    - Structural analysis
    """

    # Common knowledge patterns
    CONCEPT_PATTERNS = [
        r"\b(?:class|interface|protocol)\s+(\w+)",  # Class definitions
        r"\bdef\s+(\w+)\s*\(",  # Function definitions
        r"\b(?:const|var|let)\s+(\w+)\s*=",  # Variable definitions
        r"#\s*(?:TODO|FIXME|NOTE|IMPORTANT):\s*(.+)",  # Important comments
        r"\bSection\s+(\d+(?:\.\d+)*)",  # Section references
    ]

    # Relationship patterns
    RELATIONSHIP_PATTERNS = [
        (r"(\w+)\s+(?:extends|inherits from)\s+(\w+)", EdgeType.DERIVED_FROM),
        (r"(\w+)\s+(?:implements|conforms to)\s+(\w+)", EdgeType.PART_OF),
        (r"(\w+)\s+(?:uses|calls|imports)\s+(\w+)", EdgeType.RELATED_TO),
        (r"(\w+)\s+(?:contains|includes)\s+(\w+)", EdgeType.CONTAINS),
    ]

    # Knowledge-bearing file extensions
    KNOWLEDGE_EXTENSIONS = {
        ".md",
        ".txt",
        ".rst",
        ".py",
        ".js",
        ".ts",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".html",
        ".xml",
        ".csv",
    }

    def __init__(self, min_signal_strength: float = 0.3):
        """
        Initialize extractor.

        Args:
            min_signal_strength: Minimum signal strength for extraction
        """
        self.min_signal_strength = min_signal_strength
        self._compiled_concept_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.CONCEPT_PATTERNS
        ]
        self._compiled_relationship_patterns = [
            (re.compile(p, re.IGNORECASE), et) for p, et in self.RELATIONSHIP_PATTERNS
        ]

    def can_extract(self, file_path: Path) -> bool:
        """Check if file is suitable for knowledge extraction."""
        return file_path.suffix.lower() in self.KNOWLEDGE_EXTENSIONS

    async def extract_from_file(
        self,
        file_path: Path,
        snr_score: float = 0.5,
    ) -> ExtractionResult:
        """
        Extract knowledge from a single file.

        Args:
            file_path: Path to the file
            snr_score: Pre-computed SNR score for the file

        Returns:
            ExtractionResult containing nodes and edges
        """
        start_time = time.time()

        try:
            content = await asyncio.to_thread(
                file_path.read_text,
                encoding="utf-8",
                errors="ignore",
            )
        except (FileNotFoundError, IsADirectoryError):
            return ExtractionResult(
                source_file=str(file_path),
                nodes=[],
                edges=[],
                confidence=0.0,
            )
        except OSError as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return ExtractionResult(
                source_file=str(file_path),
                nodes=[],
                edges=[],
                confidence=0.0,
            )

        # Create file node
        file_node = self._create_file_node(file_path, snr_score)
        nodes = [file_node]
        edges = []

        # Extract concepts
        concept_nodes = self._extract_concepts(content, file_path, snr_score)
        nodes.extend(concept_nodes)

        # Create edges from file to concepts
        for concept in concept_nodes:
            edges.append(
                KnowledgeEdge(
                    source_id=file_node.id,
                    target_id=concept.id,
                    edge_type=EdgeType.CONTAINS,
                    weight=concept.signal_strength,
                    evidence=f"Extracted from {file_path.name}",
                )
            )

        # Extract relationships between concepts
        relationship_edges = self._extract_relationships(content, concept_nodes)
        edges.extend(relationship_edges)

        extraction_time = (time.time() - start_time) * 1000

        return ExtractionResult(
            source_file=str(file_path),
            nodes=nodes,
            edges=edges,
            extraction_time_ms=extraction_time,
            method=ExtractionMethod.PATTERN,
            confidence=min(0.9, snr_score + 0.2),
            snr_score=snr_score,
        )

    def _create_file_node(
        self,
        file_path: Path,
        snr_score: float,
    ) -> KnowledgeNode:
        """Create a node representing a file."""
        file_id = hashlib.sha256(str(file_path).encode()).hexdigest()[:16]

        return KnowledgeNode(
            id=f"file:{file_id}",
            node_type=NodeType.FILE,
            label=file_path.name,
            source_file=str(file_path),
            signal_strength=snr_score,
            properties=frozenset(
                [
                    ("extension", file_path.suffix),
                    ("path", str(file_path)),
                ]
            ),
        )

    def _extract_concepts(
        self,
        content: str,
        file_path: Path,
        base_snr: float,
    ) -> List[KnowledgeNode]:
        """Extract concept nodes from content."""
        concepts = []
        seen_labels = set()

        for pattern in self._compiled_concept_patterns:
            for match in pattern.finditer(content):
                label = match.group(1).strip()

                # Skip if already seen or too short
                if label in seen_labels or len(label) < 3:
                    continue

                seen_labels.add(label)

                # Compute signal strength based on pattern type
                signal = self._compute_concept_signal(label, match, base_snr)

                if signal >= self.min_signal_strength:
                    concept_id = hashlib.sha256(
                        f"{file_path}:{label}".encode()
                    ).hexdigest()[:16]

                    concepts.append(
                        KnowledgeNode(
                            id=f"concept:{concept_id}",
                            node_type=NodeType.CONCEPT,
                            label=label,
                            source_file=str(file_path),
                            source_line=content[: match.start()].count("\n") + 1,
                            signal_strength=signal,
                        )
                    )

        return concepts

    def _compute_concept_signal(
        self,
        label: str,
        match: re.Match,
        base_snr: float,
    ) -> float:
        """Compute signal strength for a concept."""
        signal = base_snr

        # Boost for meaningful patterns
        if label[0].isupper():
            signal += 0.1  # Capitalized = likely important

        if len(label) > 10:
            signal += 0.05  # Longer = more specific

        # Boost for important keywords
        important_keywords = {"ihsan", "snr", "apex", "verification", "ethics"}
        if label.lower() in important_keywords:
            signal += 0.2

        return min(signal, 1.0)

    def _extract_relationships(
        self,
        content: str,
        nodes: List[KnowledgeNode],
    ) -> List[KnowledgeEdge]:
        """Extract relationship edges from content."""
        edges = []

        # Build label -> strongest node mapping to avoid collisions
        label_to_node: Dict[str, KnowledgeNode] = {}
        for node in nodes:
            key = node.label.lower()
            existing = label_to_node.get(key)
            if not existing or node.signal_strength > existing.signal_strength:
                label_to_node[key] = node

        for pattern, edge_type in self._compiled_relationship_patterns:
            for match in pattern.finditer(content):
                source_label = match.group(1).lower()
                target_label = match.group(2).lower()

                source_node = label_to_node.get(source_label)
                target_node = label_to_node.get(target_label)

                if source_node and target_node:
                    edges.append(
                        KnowledgeEdge(
                            source_id=source_node.id,
                            target_id=target_node.id,
                            edge_type=edge_type,
                            weight=(
                                source_node.signal_strength
                                + target_node.signal_strength
                            )
                            / 2,
                            evidence=match.group(0),
                        )
                    )

        return edges


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════


class KnowledgeGraphBridge:
    """
    Bridges the Data Lake Watcher with the Knowledge Graph.

    Responsibilities:
    - Monitor data lake for changes
    - Extract knowledge from new/modified files
    - Update knowledge graph
    - Provide semantic search capabilities
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        watcher: Optional[Any] = None,
        graph: Optional[KnowledgeGraph] = None,
        extractor: Optional[KnowledgeExtractor] = None,
    ):
        """
        Initialize the bridge.

        Args:
            watcher: DataLakeWatcher instance
            graph: KnowledgeGraph instance (created if None)
            extractor: KnowledgeExtractor instance (created if None)
        """
        self.watcher = watcher
        self.graph = graph or KnowledgeGraph()
        self.extractor = extractor or KnowledgeExtractor()

        # Statistics
        self._files_processed = 0
        self._nodes_created = 0
        self._edges_created = 0
        self._started_at = datetime.now(timezone.utc)

        # Register change listener if watcher available
        if self.watcher:
            if hasattr(self.watcher, "add_change_listener"):
                self.watcher.add_change_listener(self._on_file_change_sync)
            elif hasattr(self.watcher, "add_listener"):
                self.watcher.add_listener(self._on_file_change)
            else:
                logger.warning("Watcher does not support change listener registration.")

    def _on_file_change_sync(self, change: Any) -> None:
        """Bridge sync watcher callbacks into the async handler."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._on_file_change(change))
        else:
            loop.create_task(self._on_file_change(change))

    async def _on_file_change(self, changes: Any) -> None:
        """Handle file changes from the watcher."""
        if not isinstance(changes, list):
            changes = [changes]

        for change in changes:
            file_path: Optional[Path] = None
            if isinstance(change, (str, Path)):
                file_path = Path(change)
            elif hasattr(change, "path"):
                file_path = Path(change.path)
            elif hasattr(change, "asset") and hasattr(change.asset, "path"):
                file_path = Path(change.asset.path)

            if not file_path or not self.extractor.can_extract(file_path):
                continue

            # Get SNR score from watcher or change asset when available
            snr = 0.5
            if (
                hasattr(change, "asset")
                and getattr(change.asset, "snr_score", None) is not None
            ):
                snr = change.asset.snr_score
            elif self.watcher and hasattr(self.watcher, "get_snr_score"):
                get_score = self.watcher.get_snr_score
                try:
                    score_result = get_score(file_path)
                    snr = (
                        await score_result
                        if asyncio.iscoroutine(score_result)
                        else score_result
                    )
                except Exception as e:
                    logger.warning(f"Failed to get SNR score for {file_path}: {e}")

            result = await self.extractor.extract_from_file(file_path, snr)
            self._apply_extraction(result)

    def _apply_extraction(self, result: ExtractionResult) -> None:
        """Apply extraction result to the graph."""
        for node in result.nodes:
            if self.graph.add_node(node):
                self._nodes_created += 1

        for edge in result.edges:
            if self.graph.add_edge(edge):
                self._edges_created += 1

        self._files_processed += 1

    async def process_file(
        self, file_path: Path, snr_score: float = 0.5
    ) -> ExtractionResult:
        """
        Process a single file.

        Args:
            file_path: Path to the file
            snr_score: SNR score for the file

        Returns:
            ExtractionResult
        """
        result = await self.extractor.extract_from_file(file_path, snr_score)
        self._apply_extraction(result)
        return result

    async def process_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> List[ExtractionResult]:
        """
        Process all knowledge files in a directory.

        Args:
            directory: Directory to process
            recursive: Whether to process subdirectories

        Returns:
            List of ExtractionResults
        """
        results = []

        if not directory.exists():
            return results

        pattern = "**/*" if recursive else "*"

        for file_path in directory.glob(pattern):
            if file_path.is_file() and self.extractor.can_extract(file_path):
                result = await self.process_file(file_path)
                results.append(result)

        return results

    def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Tuple[KnowledgeNode, float]]:
        """
        Search for nodes matching a query.

        Uses label matching and SNR weighting.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of (node, relevance_score) tuples
        """
        query_lower = query.lower()
        results = []

        for node in self.graph._nodes.values():
            # Simple label matching
            if query_lower in node.label.lower():
                score = node.signal_strength

                # Exact match bonus
                if query_lower == node.label.lower():
                    score *= 2

                results.append((node, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def find_related(
        self,
        node_id: str,
        max_results: int = 10,
    ) -> List[Tuple[KnowledgeNode, float]]:
        """Find nodes related to a given node."""
        return self.graph.find_related(node_id, max_results)

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        uptime = (datetime.now(timezone.utc) - self._started_at).total_seconds()
        graph_stats = self.graph.get_statistics()

        return {
            "version": self.VERSION,
            "uptime_seconds": uptime,
            "files_processed": self._files_processed,
            "nodes_created": self._nodes_created,
            "edges_created": self._edges_created,
            "graph": graph_stats.to_dict(),
        }

    def save_graph(self, path: Path) -> None:
        """Save the knowledge graph to file."""
        self.graph.save(path)

    def load_graph(self, path: Path) -> None:
        """Load the knowledge graph from file."""
        self.graph = KnowledgeGraph.load(path)
        stats = self.graph.get_statistics()
        self._nodes_created = stats.node_count
        self._edges_created = stats.edge_count
        self._files_processed = len(self.graph.get_nodes_by_type(NodeType.FILE))


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def create_knowledge_bridge(
    watcher: Optional[Any] = None,
) -> KnowledgeGraphBridge:
    """
    Factory function to create a Knowledge Graph Bridge.

    Args:
        watcher: Optional DataLakeWatcher instance

    Returns:
        Configured KnowledgeGraphBridge
    """
    return KnowledgeGraphBridge(watcher=watcher)


async def run_knowledge_bridge_cli():
    """CLI entry point for the Knowledge Graph Bridge."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="BIZRA Knowledge Graph Bridge")
    parser.add_argument("--process", type=str, help="Process directory")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--save", type=str, help="Save graph to file")
    parser.add_argument("--load", type=str, help="Load graph from file")

    args = parser.parse_args()

    bridge = create_knowledge_bridge()

    print("=" * 70)
    print(" BIZRA KNOWLEDGE GRAPH BRIDGE")
    print("=" * 70)
    print(f" Version: {KnowledgeGraphBridge.VERSION}")
    print(f" Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    print()

    if args.load:
        try:
            bridge.load_graph(Path(args.load))
            print(f"Loaded graph from {args.load}")
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"Failed to load graph from {args.load}: {e}", file=sys.stderr)
            return

    if args.process:
        results = await bridge.process_directory(Path(args.process))
        print(f"Processed {len(results)} files")
        total_nodes = sum(len(r.nodes) for r in results)
        total_edges = sum(len(r.edges) for r in results)
        print(f"  Nodes extracted: {total_nodes}")
        print(f"  Edges extracted: {total_edges}")

    if args.search:
        results = bridge.search(args.search)
        print(f"Search results for '{args.search}':")
        for node, score in results:
            print(f"  [{score:.2f}] {node.node_type.name}: {node.label}")

    if args.save:
        try:
            bridge.save_graph(Path(args.save))
            print(f"Saved graph to {args.save}")
        except OSError as e:
            print(f"Failed to save graph to {args.save}: {e}", file=sys.stderr)
            return

    if args.stats or not any([args.process, args.search, args.save]):
        stats = bridge.get_statistics()
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(run_knowledge_bridge_cli())
