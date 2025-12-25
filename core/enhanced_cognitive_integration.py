"""
BIZRA Enhanced Cognitive Integration
═════════════════════════════════════════════════════════════════════════════
Integrates Graph-of-Thoughts, SNR Scoring, and Domain-Aware Knowledge Graph
with existing BIZRA cognitive architecture.

This module provides the glue layer connecting:
- Graph-of-Thoughts reasoning engine
- SNR-based insight quality scoring
- Domain-aware knowledge graph (L4)
- Existing cognitive layers (L1-L3, L5-L7)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

# BIZRA core imports
from core.graph_of_thoughts import (
    GraphOfThoughtsEngine,
    ThoughtChain,
    DomainBridge,
    Thought,
    ThoughtType
)
from core.snr_scorer import (
    SNRScorer,
    SNRMetrics,
    SNRLevel,
    SNRThresholds
)
from core.layers.memory_layers_v2 import L4SemanticHyperGraphV2
from core.tiered_verification import QuantizedConvergence, ConvergenceResult
from core.narrative_compiler import CognitiveSynthesis, NarrativeCompiler, NarrativeStyle
from core.ultimate_integration import UltimateResult, Observation
from core.observability import MeterProvider
from core.production_safeguards import (
    InputValidator,
    CircuitBreaker,
    CircuitBreakerConfig,
    GracefulDegradation,
    get_circuit_breaker,
    get_health_checker,
    get_audit_logger
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedCognitiveResult:
    """
    Result from enhanced cognitive processing with Graph-of-Thoughts.
    """
    # Original cognitive output
    ultimate_result: UltimateResult
    
    # Graph-of-Thoughts enhancements
    top_thought_chain: Optional[ThoughtChain]
    all_chains: List[ThoughtChain]
    domain_bridges: List[DomainBridge]
    
    # SNR quality metrics
    overall_snr: SNRMetrics
    component_snr: Dict[str, SNRMetrics]
    
    # Performance metrics
    processing_time_ms: float
    graph_construction_time_ms: float
    snr_computation_time_ms: float


class EnhancedCognitiveProcessor:
    """
    Enhanced cognitive processor integrating Graph-of-Thoughts reasoning.
    
    Workflow:
    1. Receive observation
    2. Extract seed concepts (attention-driven)
    3. Construct thought chains via graph-of-thoughts
    4. Compute SNR for all insights
    5. Rank and select best chain
    6. Synthesize with existing cognitive pipeline
    7. Generate enhanced narrative
    """
    
    def __init__(
        self,
        l4_hypergraph: L4SemanticHyperGraphV2,
        quantized_convergence: QuantizedConvergence,
        narrative_compiler: NarrativeCompiler,
        metrics_collector: Optional[MeterProvider] = None,
        enable_graph_of_thoughts: bool = True,
        beam_width: int = 10,
        max_thought_depth: int = 5
    ):
        """
        Initialize enhanced cognitive processor.
        
        Args:
            l4_hypergraph: Domain-aware knowledge graph
            quantized_convergence: Convergence computation engine
            narrative_compiler: Narrative generation
            metrics_collector: Observability metrics
            enable_graph_of_thoughts: Toggle graph-of-thoughts (for A/B testing)
            beam_width: Beam search width
            max_thought_depth: Maximum thought chain depth
        """
        self.l4 = l4_hypergraph
        self.convergence = quantized_convergence
        self.narrative_compiler = narrative_compiler
        self.metrics = metrics_collector or MeterProvider("enhanced_cognitive")
        
        # Initialize production safeguards
        self.validator = InputValidator()
        self.neo4j_circuit = get_circuit_breaker(
            'neo4j_hypergraph',
            CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30.0)
        )
        self.convergence_circuit = get_circuit_breaker(
            'convergence_engine',
            CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60.0)
        )
        self.health_checker = get_health_checker()
        self.audit_logger = get_audit_logger()
        
        # Initialize SNR scorer
        self.snr_scorer = SNRScorer(
            thresholds=SNRThresholds(
                high_threshold=0.80,
                medium_threshold=0.50,
                min_ihsan_for_high=0.95
            ),
            enable_ethical_constraints=True
        )
        
        # Initialize Graph-of-Thoughts engine
        self.enable_got = enable_graph_of_thoughts
        if self.enable_got:
            self.got_engine = GraphOfThoughtsEngine(
                snr_scorer=self.snr_scorer,
                beam_width=beam_width,
                max_depth=max_thought_depth,
                min_snr_threshold=0.3,
                novelty_bonus=0.2
            )
        
        logger.info(
            f"EnhancedCognitiveProcessor initialized: "
            f"GoT={'enabled' if self.enable_got else 'disabled'}, "
            f"beam_width={beam_width}, max_depth={max_thought_depth}"
        )
    
    async def process(self, observation: Observation) -> EnhancedCognitiveResult:
        """
        Process observation with enhanced Graph-of-Thoughts reasoning.
        
        Args:
            observation: Input observation
            
        Returns:
            EnhancedCognitiveResult with thought chains and SNR metrics
        """
        import time
        start_time = time.perf_counter()
        
        # Step 1: Extract seed concepts from observation
        seed_concepts = await self._extract_seed_concepts(observation)
        
        # Validate seed concepts
        validation = self.validator.validate_seed_concepts(seed_concepts)
        if not validation.is_valid:
            logger.error(f"Invalid seed concepts: {validation.errors}")
            seed_concepts = []
        else:
            seed_concepts = validation.sanitized_input
            if validation.warnings:
                for warning in validation.warnings:
                    logger.warning(f"Seed concept warning: {warning}")
        
        logger.debug(f"Extracted {len(seed_concepts)} seed concepts: {seed_concepts}")
        
        # Step 2: Construct thought chains (if enabled)
        thought_chains = []
        domain_bridges = []
        graph_time_ms = 0.0
        
        if self.enable_got and seed_concepts:
            graph_start = time.perf_counter()
            
            thought_chains = await self.got_engine.reason(
                query=observation.context.get('query', 'Process observation'),
                seed_concepts=seed_concepts,
                hypergraph_query_fn=self._query_hypergraph,
                convergence_fn=self._compute_convergence_for_concept,
                top_k_chains=5
            )
            
            # Extract all domain bridges
            for chain in thought_chains:
                domain_bridges.extend(chain.bridges)
            
            graph_time_ms = (time.perf_counter() - graph_start) * 1000
            
            logger.info(
                f"Graph-of-Thoughts: {len(thought_chains)} chains, "
                f"{len(domain_bridges)} bridges in {graph_time_ms:.1f}ms"
            )
            
            # Audit log thought chain construction
            if thought_chains:
                best_chain = thought_chains[0]
                self.audit_logger.log_thought_chain_construction(
                    chain_id=best_chain.id,
                    query=observation.context.get('query', 'unknown'),
                    snr_score=best_chain.total_snr,
                    depth=best_chain.max_depth,
                    bridge_count=len(domain_bridges)
                )
        
        # Step 3: Compute SNR metrics (with circuit breaker)
        snr_start = time.perf_counter()
        
        convergence_result = await self.convergence_circuit.call(
            self._safe_compute_convergence,
            observation,
            fallback=self._fallback_convergence
        )
        
        # Fetch disagreement from value oracle
        disagreement = await self._get_disagreement_from_oracle(observation)
        
        # Fetch ihsan_metric from ethics engine
        ihsan_metric = await self._get_ihsan_from_ethics(observation)
        
        # Validate retrieved values - fail loudly if production sources return None
        if disagreement is None:
            logger.warning(
                "PRODUCTION WARNING: disagreement is None from value oracle, "
                "using fallback 0.3 (pessimistic). This should not occur in production."
            )
            self.audit_logger.log_event(
                'value_oracle_fallback',
                {'reason': 'disagreement_none', 'fallback_value': 0.3}
            )
            disagreement = 0.3  # Pessimistic fallback
        
        if ihsan_metric is None:
            logger.error(
                "CRITICAL: ihsan_metric is None from ethics engine. "
                "Using fallback 0.0 to force LOW SNR. Fix ethics engine integration!"
            )
            self.audit_logger.log_event(
                'ethics_engine_fallback',
                {'reason': 'ihsan_metric_none', 'fallback_value': 0.0}
            )
            ihsan_metric = 0.0  # Force LOW SNR when ethics unavailable
        
        overall_snr = self.snr_scorer.compute_from_convergence(
            convergence_result,
            consistency=self._compute_interdisciplinary_consistency(thought_chains),
            disagreement=disagreement,
            ihsan_metric=ihsan_metric
        )
        
        # Audit log if ethical override occurred
        if overall_snr.ethical_override:
            self.audit_logger.log_ethical_override(
                reason="Ihsān metric below threshold for HIGH SNR",
                original_level="HIGH",
                downgraded_level=overall_snr.level.name,
                ihsan_metric=overall_snr.ihsan_metric
            )
        
        # Compute distinct SNR for each component using component-specific signals
        perception_consistency = self._compute_perception_consistency(observation)
        perception_snr = self.snr_scorer.compute_from_convergence(
            convergence_result,
            consistency=perception_consistency,
            disagreement=disagreement * 0.8,  # Perception typically has less disagreement
            ihsan_metric=ihsan_metric
        )
        
        component_snr = {
            'convergence': overall_snr,
            'perception': perception_snr,
        }
        
        snr_time_ms = (time.perf_counter() - snr_start) * 1000
        
        # Step 4: Create cognitive synthesis with enhancements
        synthesis = await self._create_enhanced_synthesis(
            observation,
            convergence_result,
            thought_chains,
            domain_bridges,
            overall_snr,
            component_snr
        )
        
        # Step 5: Generate narrative
        narrative = self.narrative_compiler.compile(synthesis, style=NarrativeStyle.TECHNICAL)
        
        # Step 6: Create ultimate result
        ultimate_result = UltimateResult(
            action=convergence_result.action,
            confidence=0.85,  # Simplified
            verification=None,  # Would integrate with verification engine
            value=None,  # Would integrate with value oracle
            ethics=None,  # Would integrate with ethics engine
            health=None,  # Would integrate with health monitor
            explanation=narrative,
            snr_metrics={
                'overall': overall_snr.to_dict(),
                'components': {k: v.to_dict() for k, v in component_snr.items()}
            },
            thought_chains=[
                {
                    'id': c.id,
                    'thoughts': [t.id for t in c.thoughts],
                    'total_snr': c.total_snr,
                    'avg_snr': c.avg_snr,
                    'domain_diversity': c.domain_diversity
                }
                for c in thought_chains
            ],
            domain_bridges=[
                {
                    'id': b.id,
                    'source_domain': b.source_domain,
                    'target_domain': b.target_domain,
                    'snr_score': b.snr_score,
                    'novelty': b.novelty
                }
                for b in domain_bridges
            ]
        )
        
        total_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Record metrics
        self.metrics.record_snr_metrics(
            overall_snr.snr_score,
            overall_snr.level.name,
            'overall'
        )
        
        if thought_chains:
            best_chain = thought_chains[0]
            self.metrics.record_thought_graph_metrics(
                chain_depth=best_chain.max_depth,
                domain_diversity=best_chain.domain_diversity,
                bridge_count=len(domain_bridges)
            )
        
        result = EnhancedCognitiveResult(
            ultimate_result=ultimate_result,
            top_thought_chain=thought_chains[0] if thought_chains else None,
            all_chains=thought_chains,
            domain_bridges=domain_bridges,
            overall_snr=overall_snr,
            component_snr=component_snr,
            processing_time_ms=total_time_ms,
            graph_construction_time_ms=graph_time_ms,
            snr_computation_time_ms=snr_time_ms
        )
        
        logger.info(
            f"Enhanced processing complete: "
            f"SNR={overall_snr.snr_score:.3f} ({overall_snr.level.name}), "
            f"{len(thought_chains)} chains, {len(domain_bridges)} bridges, "
            f"{total_time_ms:.1f}ms total"
        )
        
        return result
    
    async def _extract_seed_concepts(self, observation: Observation) -> List[str]:
        """
        Extract seed concepts from observation for graph-of-thoughts exploration.
        
        In full integration, this would use L1 (attention) to identify
        salient entities. For now, we extract from context.
        """
        # Simple extraction from context
        context = observation.context
        
        seeds = []
        
        # Extract from query if present
        if 'query' in context:
            # In real implementation, would use NER/entity extraction
            query = context['query'].lower()
            
            # Simple keyword extraction
            keywords = [
                'task', 'strategy', 'result', 'observation',
                'action', 'decision', 'analysis', 'synthesis'
            ]
            
            for keyword in keywords:
                if keyword in query:
                    seeds.append(keyword.capitalize())
        
        # Fallback to generic seeds
        if not seeds:
            seeds = ['Observation', 'Task', 'Analysis']
        
        # Limit to reasonable number
        return seeds[:5]
    
    async def _query_hypergraph(self, node_name: str) -> List[Dict[str, Any]]:
        """
        Query hypergraph for neighbors of given node (with circuit breaker).
        
        Wrapper for L4SemanticHyperGraphV2.get_neighbors_with_domains()
        """
        return await self.neo4j_circuit.call(
            self._safe_query_hypergraph,
            node_name,
            fallback=GracefulDegradation.fallback_hypergraph_query
        )
    
    async def _safe_query_hypergraph(self, node_name: str) -> List[Dict[str, Any]]:
        """Safe hypergraph query with timeout."""
        try:
            async with asyncio.timeout(5.0):  # 5 second timeout
                neighbors = await self.l4.get_neighbors_with_domains(
                    node_name,
                    max_neighbors=20
                )
                return neighbors
        except asyncio.TimeoutError:
            logger.warning(f"Hypergraph query timeout for {node_name}")
            raise
        except Exception as e:
            logger.warning(f"Hypergraph query failed for {node_name}: {e}")
            raise
    
    async def _safe_compute_convergence(self, observation: Observation) -> ConvergenceResult:
        """Safe convergence computation with timeout."""
        try:
            async with asyncio.timeout(10.0):  # 10 second timeout
                return self.convergence.compute(observation)
        except asyncio.TimeoutError:
            logger.error("Convergence computation timeout")
            raise
        except Exception as e:
            logger.error(f"Convergence computation failed: {e}")
            raise
    
    async def _fallback_convergence(self, observation: Observation) -> ConvergenceResult:
        """Fallback convergence result when computation fails."""
        logger.warning("Using fallback convergence result")
        return ConvergenceResult(
            clarity=0.50,
            mutual_information=0.45,
            entropy=0.40,
            synergy=0.55,
            quantization_error=0.15,
            quality="DEGRADED",
            action={'type': 'fallback', 'reason': 'convergence_failure'}
        )
    
    async def _compute_convergence_for_concept(
        self,
        concept: str,
        context: Dict[str, Any]
    ) -> ConvergenceResult:
        """
        Compute convergence metrics for a concept.
        
        In full integration, would use actual convergence computation.
        For now, returns simplified result.
        """
        # Simplified convergence result
        # In real implementation, would call QuantizedConvergence.compute()
        return ConvergenceResult(
            clarity=0.75,
            mutual_information=0.65,
            entropy=0.25,
            synergy=0.80,
            quantization_error=0.05,
            quality="GOOD",
            action={'type': 'concept_activation', 'concept': concept}
        )
    
    def _compute_interdisciplinary_consistency(
        self,
        thought_chains: List[ThoughtChain]
    ) -> float:
        """
        Compute interdisciplinary consistency across thought chains.
        
        Higher consistency when chains agree on domain connections.
        """
        if not thought_chains:
            return 0.7  # Default
        
        # Average domain diversity across chains
        avg_diversity = sum(c.domain_diversity for c in thought_chains) / len(thought_chains)
        
        # Normalize to [0, 1]
        # High diversity = good interdisciplinary exploration
        consistency = min(1.0, avg_diversity / 2.0)
        
        return consistency
    
    async def _get_disagreement_from_oracle(self, observation: Observation) -> Optional[float]:
        """
        Fetch disagreement score from the PluralisticValueOracle.
        
        The disagreement score measures how much the ensemble of value oracles
        disagree on the assessment, indicating uncertainty in value estimation.
        
        Returns:
            Disagreement score [0, 1] or None if oracle unavailable
        """
        try:
            # Import here to avoid circular dependency
            from core.value_oracle import PluralisticValueOracle, Convergence
            
            # Get or create value oracle instance
            if not hasattr(self, '_value_oracle'):
                self._value_oracle = PluralisticValueOracle()
            
            # Create convergence object from observation for oracle evaluation
            import uuid
            convergence = Convergence(
                id=str(uuid.uuid4()),
                clarity_score=observation.context.get('clarity', 0.7),
                mutual_information=observation.context.get('mutual_info', 0.6),
                entropy=observation.context.get('entropy', 0.3),
                synergy=observation.context.get('synergy', 0.7),
                quantization_error=observation.context.get('quant_error', 0.1),
            )
            
            assessment = await self._value_oracle.compute_value(convergence)
            
            logger.debug(f"Value oracle disagreement: {assessment.disagreement_score:.4f}")
            return assessment.disagreement_score
            
        except ImportError as e:
            logger.error(f"Failed to import value oracle: {e}")
            return None
        except Exception as e:
            logger.warning(f"Value oracle evaluation failed: {e}")
            return None
    
    async def _get_ihsan_from_ethics(self, observation: Observation) -> Optional[float]:
        """
        Fetch Ihsān metric from the ConsequentialEthicsEngine.
        
        The Ihsān score is derived from the IhsanEvaluator's assessment of
        the action's alignment with Islamic ethical principles (ikhlas, karama,
        adl, kamal, istidama).
        
        Returns:
            Ihsān metric [0, 1] or None if ethics engine unavailable
        """
        try:
            # Import here to avoid circular dependency
            from core.consequential_ethics import (
                ConsequentialEthicsEngine,
                Action,
                Context as EthicsContext,
                EthicalFramework
            )
            
            # Get or create ethics engine instance
            if not hasattr(self, '_ethics_engine'):
                self._ethics_engine = ConsequentialEthicsEngine()
            
            # Create action/context from observation for ethics evaluation
            action = Action(
                id=observation.context.get('action_id', 'cognitive_process'),
                description=observation.context.get('query', 'Cognitive processing'),
                intended_outcome=observation.context.get('intent', 'Knowledge synthesis'),
            )
            
            context = EthicsContext(
                stakeholders=observation.context.get('stakeholders', ['user', 'system']),
                affected_parties=observation.context.get('affected', ['user']),
                domain=observation.context.get('domain', 'cognitive'),
            )
            
            verdict = await self._ethics_engine.evaluate(action, context)
            
            # Extract Ihsān score from the Ihsān framework evaluation
            ihsan_score = None
            for evaluation in verdict.evaluations:
                if evaluation.framework == EthicalFramework.IHSAN:
                    # Convert from [-1, 1] to [0, 1]
                    ihsan_score = (evaluation.score + 1) / 2
                    break
            
            if ihsan_score is None:
                logger.warning("IhsanEvaluator not found in verdict, using overall score")
                # Fallback to overall score (weighted across all frameworks)
                ihsan_score = (verdict.overall_score + 1) / 2
            
            logger.debug(f"Ethics engine Ihsān metric: {ihsan_score:.4f}")
            return ihsan_score
            
        except ImportError as e:
            logger.error(f"Failed to import ethics engine: {e}")
            return None
        except Exception as e:
            logger.warning(f"Ethics engine evaluation failed: {e}")
            return None
    
    def _compute_perception_consistency(self, observation: Observation) -> float:
        """
        Compute perception-specific consistency for component SNR.
        
        Perception consistency is derived from the coherence of sensory-like
        signals in the observation context.
        
        Returns:
            Perception consistency [0, 1]
        """
        context = observation.context
        
        # Perception signals: how coherent are the input signals?
        signal_count = 0
        signal_strength_sum = 0.0
        
        # Check for various perception-related context fields
        perception_fields = ['visual', 'textual', 'semantic', 'structural']
        for field in perception_fields:
            if field in context:
                signal_count += 1
                # Assume context values are normalized [0, 1] or can be coerced
                value = context[field]
                if isinstance(value, (int, float)):
                    signal_strength_sum += min(1.0, max(0.0, float(value)))
                else:
                    signal_strength_sum += 0.5  # Default for non-numeric
        
        if signal_count == 0:
            # No explicit perception signals, use moderate default
            return 0.65
        
        # Average signal strength as consistency measure
        return signal_strength_sum / signal_count
    
    async def _create_enhanced_synthesis(
        self,
        observation: Observation,
        convergence: ConvergenceResult,
        thought_chains: List[ThoughtChain],
        bridges: List[DomainBridge],
        overall_snr: SNRMetrics,
        component_snr: Dict[str, SNRMetrics]
    ) -> CognitiveSynthesis:
        """
        Create CognitiveSynthesis with Graph-of-Thoughts enhancements.
        """
        synthesis = CognitiveSynthesis(
            action=convergence.action,
            confidence=0.85,
            verification_tier='INCREMENTAL',
            value_score=0.78,
            ethical_verdict={'permitted': True, 'severity': 'ACCEPTABLE'},
            health_status='HEALTHY',
            ihsan_scores={'im': 0.96, 'adl': 0.94, 'amanah': 0.97},
            interdisciplinary_consistency=self._compute_interdisciplinary_consistency(thought_chains),
            quantization_error=convergence.quantization_error,
            
            # Enhanced fields
            snr_scores={k: v.snr_score for k, v in component_snr.items()},
            thought_graph_metrics={
                'chain_count': len(thought_chains),
                'chain_depth': thought_chains[0].max_depth if thought_chains else 0,
                'domain_diversity': thought_chains[0].domain_diversity if thought_chains else 0.0,
                'avg_snr': thought_chains[0].avg_snr if thought_chains else 0.0,
                'bridge_count': len(bridges)
            },
            domain_bridges=[
                {
                    'id': b.id,
                    'source_domain': b.source_domain,
                    'target_domain': b.target_domain,
                    'source_concept': b.source_concept,
                    'target_concept': b.target_concept,
                    'snr_score': b.snr_score,
                    'novelty': b.novelty,
                    'strength': b.strength
                }
                for b in bridges
            ]
        )
        
        return synthesis


# Convenience function for quick integration
async def process_with_graph_of_thoughts(
    observation: Observation,
    l4_hypergraph: L4SemanticHyperGraphV2,
    enable_got: bool = True
) -> EnhancedCognitiveResult:
    """
    Quick helper to process observation with Graph-of-Thoughts.
    
    Example:
        l4 = L4SemanticHyperGraphV2(neo4j_uri, neo4j_auth)
        await l4.initialize()
        
        obs = Observation(
            id='test_001',
            data=b'test observation',
            context={'query': 'Analyze task complexity'}
        )
        
        result = await process_with_graph_of_thoughts(obs, l4)
        print(f"SNR: {result.overall_snr.snr_score:.3f}")
        print(f"Bridges: {len(result.domain_bridges)}")
    """
    # Create minimal dependencies
    convergence = QuantizedConvergence()
    narrative = NarrativeCompiler()
    metrics = MeterProvider("quick_process")
    
    processor = EnhancedCognitiveProcessor(
        l4_hypergraph=l4_hypergraph,
        quantized_convergence=convergence,
        narrative_compiler=narrative,
        metrics_collector=metrics,
        enable_graph_of_thoughts=enable_got
    )
    
    return await processor.process(observation)


__all__ = [
    'EnhancedCognitiveProcessor',
    'EnhancedCognitiveResult',
    'process_with_graph_of_thoughts'
]
