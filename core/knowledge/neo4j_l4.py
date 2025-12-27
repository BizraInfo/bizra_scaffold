"""
BIZRA Neo4j L4 Semantic HyperGraph Adapter
═══════════════════════════════════════════════════════════════════════════════
Production-ready Neo4j integration for L4 semantic memory layer.

This adapter provides:
1. Connection pooling with circuit breaker pattern
2. Hyperedge implementation using intermediate nodes
3. Rich-club and small-world topology analysis
4. Fallback to NetworkX when Neo4j unavailable
5. Async-first design with sync wrappers

The L4 layer stores semantic relationships as a knowledge graph,
enabling complex queries about entity relationships and concepts.

Author: BIZRA Cognitive Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

# Optional Neo4j import
try:
    from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
    from neo4j.exceptions import (
        AuthError,
        ServiceUnavailable,
        SessionExpired,
        TransientError,
    )
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncDriver = Any
    AsyncSession = Any

# NetworkX for fallback
import networkx as nx

logger = logging.getLogger("bizra.neo4j_l4")


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject calls
    HALF_OPEN = auto()   # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for Neo4j connections.
    
    Prevents cascading failures when Neo4j is unavailable.
    """
    
    failure_threshold: int = 5
    reset_timeout: float = 30.0
    half_open_max_calls: int = 3
    
    # State
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0
    
    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self._close()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self._open()
        elif (
            self.state == CircuitState.CLOSED
            and self.failure_count >= self.failure_threshold
        ):
            self._open()
    
    def can_execute(self) -> bool:
        """Check if a call can be made."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if reset timeout has passed
            if time.time() - self.last_failure_time >= self.reset_timeout:
                self._half_open()
                return True
            return False
        
        # HALF_OPEN
        if self.half_open_calls < self.half_open_max_calls:
            self.half_open_calls += 1
            return True
        return False
    
    def _open(self) -> None:
        """Open the circuit."""
        self.state = CircuitState.OPEN
        logger.warning("Circuit breaker OPENED - Neo4j calls blocked")
    
    def _close(self) -> None:
        """Close the circuit."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        logger.info("Circuit breaker CLOSED - Neo4j calls resumed")
    
    def _half_open(self) -> None:
        """Transition to half-open."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        logger.info("Circuit breaker HALF_OPEN - testing Neo4j connection")


# =============================================================================
# NEO4J CONNECTION MANAGER
# =============================================================================


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""
    
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""  # Must be provided via env or explicit config
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: float = 30.0
    
    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Load configuration from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


class Neo4jConnectionManager:
    """
    Manages Neo4j driver lifecycle with connection pooling.
    """
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        self.config = config or Neo4jConfig.from_env()
        self._driver: Optional[AsyncDriver] = None
        self.circuit_breaker = CircuitBreaker()
        self._connected = False
    
    async def connect(self) -> bool:
        """
        Establish connection to Neo4j.
        
        Returns True if connected, False otherwise.
        """
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not installed, using fallback")
            return False
        
        if not self.config.password:
            logger.warning("Neo4j password not configured, using fallback")
            return False
        
        try:
            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_acquisition_timeout,
            )
            
            # Verify connection
            async with self._driver.session(database=self.config.database) as session:
                await session.run("RETURN 1")
            
            self._connected = True
            logger.info(f"Connected to Neo4j at {self.config.uri}")
            return True
            
        except (AuthError, ServiceUnavailable) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._driver = None
            return False
    
    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._connected = False
            logger.info("Disconnected from Neo4j")
    
    @asynccontextmanager
    async def session(self):
        """Get a Neo4j session with circuit breaker protection."""
        if not self._connected or not self._driver:
            raise RuntimeError("Not connected to Neo4j")
        
        if not self.circuit_breaker.can_execute():
            raise RuntimeError("Circuit breaker is open")
        
        session = None
        try:
            session = self._driver.session(database=self.config.database)
            yield session
            self.circuit_breaker.record_success()
        except (ServiceUnavailable, SessionExpired, TransientError) as e:
            self.circuit_breaker.record_failure()
            raise
        finally:
            if session:
                await session.close()
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self._driver is not None


# =============================================================================
# NEO4J L4 HYPERGRAPH IMPLEMENTATION
# =============================================================================


class Neo4jL4HyperGraph:
    """
    Neo4j implementation of L4 Semantic HyperGraph.
    
    Hyperedges are modeled as intermediate nodes connecting entities.
    This preserves Neo4j's property graph model while enabling
    multi-way relationships.
    """
    
    def __init__(self, connection_manager: Optional[Neo4jConnectionManager] = None):
        self.connection_manager = connection_manager or Neo4jConnectionManager()
        
        # Fallback NetworkX graph
        self._fallback_graph = nx.DiGraph()
        self._using_fallback = True
    
    async def initialize(self) -> bool:
        """
        Initialize connection to Neo4j.
        
        Returns True if Neo4j connected, False if using fallback.
        """
        if await self.connection_manager.connect():
            # Create indexes
            await self._create_indexes()
            self._using_fallback = False
            return True
        
        logger.info("Using NetworkX fallback for L4 HyperGraph")
        self._using_fallback = True
        return False
    
    async def _create_indexes(self) -> None:
        """Create Neo4j indexes for optimal query performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX IF NOT EXISTS FOR (h:HyperEdge) ON (h.id)",
            "CREATE INDEX IF NOT EXISTS FOR (h:HyperEdge) ON (h.relation)",
        ]
        
        try:
            async with self.connection_manager.session() as session:
                for index_query in indexes:
                    await session.run(index_query)
            logger.info("Neo4j indexes created")
        except Exception as e:
            logger.warning(f"Index creation failed: {e}")
    
    async def create_hyperedge(
        self,
        nodes: List[str],
        relation: str,
        weights: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a multi-way semantic relationship.
        
        In Neo4j, this creates:
        - Entity nodes for each participant
        - A HyperEdge node representing the relationship
        - PARTICIPATES edges from entities to hyperedge
        - IMPLICIT edges between co-participating entities
        
        Returns: The hyperedge ID
        """
        edge_id = f"edge_{hashlib.sha256(f'{nodes}_{relation}_{datetime.now(timezone.utc)}'.encode()).hexdigest()[:16]}"
        weights = weights or [1.0] * len(nodes)
        metadata = metadata or {}
        timestamp = datetime.now(timezone.utc).isoformat()
        
        if self._using_fallback:
            return await self._create_hyperedge_fallback(
                edge_id, nodes, relation, weights, metadata, timestamp
            )
        
        try:
            async with self.connection_manager.session() as session:
                # Create hyperedge node
                await session.run(
                    """
                    CREATE (h:HyperEdge {
                        id: $edge_id,
                        relation: $relation,
                        timestamp: $timestamp,
                        participant_count: $count
                    })
                    """,
                    edge_id=edge_id,
                    relation=relation,
                    timestamp=timestamp,
                    count=len(nodes),
                )
                
                # Create entities and relationships
                for i, node_id in enumerate(nodes):
                    await session.run(
                        """
                        MERGE (e:Entity {id: $node_id})
                        WITH e
                        MATCH (h:HyperEdge {id: $edge_id})
                        CREATE (e)-[:PARTICIPATES {weight: $weight, order: $order}]->(h)
                        """,
                        node_id=node_id,
                        edge_id=edge_id,
                        weight=weights[i],
                        order=i,
                    )
                
                # Create implicit edges between co-participants
                await session.run(
                    """
                    MATCH (h:HyperEdge {id: $edge_id})<-[:PARTICIPATES]-(e1:Entity)
                    MATCH (h)<-[:PARTICIPATES]-(e2:Entity)
                    WHERE e1.id < e2.id
                    MERGE (e1)-[:IMPLICIT {via: $edge_id, weight: 0.5}]-(e2)
                    """,
                    edge_id=edge_id,
                )
                
            logger.debug(f"Created hyperedge {edge_id} with {len(nodes)} participants")
            return edge_id
            
        except Exception as e:
            logger.error(f"Neo4j hyperedge creation failed: {e}")
            # Fallback
            self._using_fallback = True
            return await self._create_hyperedge_fallback(
                edge_id, nodes, relation, weights, metadata, timestamp
            )
    
    async def _create_hyperedge_fallback(
        self,
        edge_id: str,
        nodes: List[str],
        relation: str,
        weights: List[float],
        metadata: Dict[str, Any],
        timestamp: str,
    ) -> str:
        """Fallback implementation using NetworkX."""
        self._fallback_graph.add_node(
            edge_id,
            type="HyperEdge",
            relation=relation,
            timestamp=timestamp,
            **metadata,
        )
        
        for i, node_id in enumerate(nodes):
            self._fallback_graph.add_node(node_id, type="Entity")
            self._fallback_graph.add_edge(
                node_id, edge_id, weight=weights[i], type="PARTICIPATES"
            )
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self._fallback_graph.add_edge(
                    nodes[i], nodes[j], weight=0.5, type="IMPLICIT"
                )
        
        return edge_id
    
    async def get_entity_relations(
        self,
        entity_id: str,
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Get all relations for an entity up to max_depth hops.
        """
        if self._using_fallback:
            return await self._get_entity_relations_fallback(entity_id, max_depth)
        
        try:
            async with self.connection_manager.session() as session:
                result = await session.run(
                    """
                    MATCH path = (e:Entity {id: $entity_id})-[*1..$depth]-(related)
                    RETURN path
                    LIMIT 100
                    """,
                    entity_id=entity_id,
                    depth=max_depth * 2,  # Account for hyperedge hops
                )
                
                relations = []
                async for record in result:
                    path = record["path"]
                    relations.append({
                        "nodes": [n["id"] for n in path.nodes if "id" in n],
                        "length": len(path),
                    })
                
                return relations
                
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return await self._get_entity_relations_fallback(entity_id, max_depth)
    
    async def _get_entity_relations_fallback(
        self,
        entity_id: str,
        max_depth: int,
    ) -> List[Dict[str, Any]]:
        """Fallback for entity relations using NetworkX."""
        if entity_id not in self._fallback_graph:
            return []
        
        relations = []
        for target in nx.single_source_shortest_path_length(
            self._fallback_graph, entity_id, cutoff=max_depth * 2
        ):
            if target != entity_id:
                try:
                    path = nx.shortest_path(self._fallback_graph, entity_id, target)
                    relations.append({
                        "nodes": path,
                        "length": len(path),
                    })
                except nx.NetworkXNoPath:
                    pass
        
        return relations[:100]
    
    async def analyze_topology(self) -> Dict[str, float]:
        """
        Analyze HyperGraph topology metrics.
        
        Returns:
        - clustering_coefficient: Local clustering
        - rich_club_coefficient: Hub connectivity
        - node_count: Total nodes
        - edge_count: Total edges
        """
        if self._using_fallback:
            return await self._analyze_topology_fallback()
        
        try:
            async with self.connection_manager.session() as session:
                # Get counts
                result = await session.run(
                    """
                    MATCH (n)
                    WITH count(n) as nodeCount
                    MATCH ()-[r]->()
                    RETURN nodeCount, count(r) as edgeCount
                    """
                )
                record = await result.single()
                node_count = record["nodeCount"] if record else 0
                edge_count = record["edgeCount"] if record else 0
                
                # Get degree distribution for rich-club
                result = await session.run(
                    """
                    MATCH (n:Entity)
                    RETURN n.id as id, size((n)--()) as degree
                    ORDER BY degree DESC
                    LIMIT 20
                    """
                )
                
                degrees = []
                async for record in result:
                    degrees.append(record["degree"])
                
                rich_club = self._calculate_rich_club_from_degrees(degrees)
                
                return {
                    "clustering_coefficient": 0.0,  # Complex to compute in Cypher
                    "rich_club_coefficient": rich_club,
                    "node_count": float(node_count),
                    "edge_count": float(edge_count),
                }
                
        except Exception as e:
            logger.error(f"Topology analysis failed: {e}")
            return await self._analyze_topology_fallback()
    
    async def _analyze_topology_fallback(self) -> Dict[str, float]:
        """Fallback topology analysis using NetworkX."""
        if len(self._fallback_graph) < 2:
            return {
                "clustering_coefficient": 0.0,
                "rich_club_coefficient": 0.0,
                "node_count": float(len(self._fallback_graph)),
                "edge_count": float(self._fallback_graph.number_of_edges()),
            }
        
        undir = self._fallback_graph.to_undirected()
        
        return {
            "clustering_coefficient": nx.average_clustering(undir),
            "rich_club_coefficient": self._calculate_rich_club(undir),
            "node_count": float(len(self._fallback_graph)),
            "edge_count": float(self._fallback_graph.number_of_edges()),
        }
    
    def _calculate_rich_club_from_degrees(
        self,
        degrees: List[int],
        k: int = 5,
    ) -> float:
        """Calculate rich-club coefficient from degree list."""
        if len(degrees) < 2:
            return 0.0
        
        rich_count = sum(1 for d in degrees if d >= k)
        if rich_count < 2:
            return 0.0
        
        # Estimate based on degree distribution
        return min(1.0, rich_count / (len(degrees) * 0.2))
    
    def _calculate_rich_club(self, graph, k: int = 5) -> float:
        """Calculate rich-club coefficient from NetworkX graph."""
        degrees = [d for _, d in graph.degree()]
        if not degrees:
            return 0.0
        
        rich_nodes = [n for n, d in graph.degree() if d >= k]
        if len(rich_nodes) < 2:
            return 0.0
        
        subgraph = graph.subgraph(rich_nodes)
        possible_edges = len(rich_nodes) * (len(rich_nodes) - 1) / 2
        return subgraph.number_of_edges() / possible_edges if possible_edges > 0 else 0.0
    
    async def search_semantic(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for entities matching a semantic query.
        
        In production, this would use vector similarity search.
        Current implementation uses simple substring matching.
        """
        if self._using_fallback:
            # Simple search in fallback
            results = []
            query_lower = query.lower()
            for node in self._fallback_graph.nodes():
                if query_lower in str(node).lower():
                    results.append({
                        "id": node,
                        "type": self._fallback_graph.nodes[node].get("type", "Unknown"),
                        "score": 1.0,
                    })
            return results[:limit]
        
        try:
            async with self.connection_manager.session() as session:
                result = await session.run(
                    """
                    MATCH (n)
                    WHERE n.id CONTAINS $query OR n.relation CONTAINS $query
                    RETURN n.id as id, labels(n)[0] as type
                    LIMIT $limit
                    """,
                    query=query,
                    limit=limit,
                )
                
                results = []
                async for record in result:
                    results.append({
                        "id": record["id"],
                        "type": record["type"],
                        "score": 1.0,
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def close(self) -> None:
        """Close Neo4j connection."""
        await self.connection_manager.close()
    
    @property
    def is_using_neo4j(self) -> bool:
        """Check if using real Neo4j or fallback."""
        return not self._using_fallback


# =============================================================================
# MAIN - FOR TESTING
# =============================================================================


async def main():
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("BIZRA Neo4j L4 HyperGraph - Test Run")
    print("=" * 80)
    
    # Create graph (will use fallback if Neo4j not available)
    graph = Neo4jL4HyperGraph()
    connected = await graph.initialize()
    
    print(f"\nNeo4j connected: {connected}")
    print(f"Using fallback: {graph._using_fallback}")
    
    # Create some hyperedges
    print("\n📊 Creating hyperedges...")
    
    await graph.create_hyperedge(
        nodes=["User:Alice", "Resource:Document1", "Action:Read"],
        relation="ACCESS",
        weights=[1.0, 0.8, 1.0],
    )
    
    await graph.create_hyperedge(
        nodes=["User:Alice", "User:Bob", "Project:BIZRA"],
        relation="COLLABORATES",
        weights=[1.0, 1.0, 1.0],
    )
    
    await graph.create_hyperedge(
        nodes=["Concept:Ihsan", "Concept:Ethics", "Concept:Excellence"],
        relation="RELATED_TO",
        weights=[0.9, 0.8, 0.7],
    )
    
    print("   Created 3 hyperedges")
    
    # Query relations
    print("\n🔍 Querying relations for User:Alice...")
    relations = await graph.get_entity_relations("User:Alice")
    print(f"   Found {len(relations)} relation paths")
    
    # Analyze topology
    print("\n📈 Analyzing topology...")
    topology = await graph.analyze_topology()
    for key, value in topology.items():
        print(f"   {key}: {value:.4f}")
    
    # Search
    print("\n🔎 Semantic search for 'User'...")
    results = await graph.search_semantic("User")
    for r in results:
        print(f"   {r['id']} ({r['type']})")
    
    await graph.close()
    print("\n✅ Test complete")


if __name__ == "__main__":
    asyncio.run(main())
