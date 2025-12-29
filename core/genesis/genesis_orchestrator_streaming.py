"""
BIZRA Genesis Orchestrator (Streaming Extension)
════════════════════════════════════════════════════════════════════════════════
Enhances the base Orchestrator with Event Streaming capabilities.

This module provides real-time observability into the Genesis Orchestrator's
reasoning process by yielding events at each processing phase. The terminal
(or any listener) can visualize thoughts *as they are born*, rather than
waiting for the final result.

"The stream of consciousness, made visible.
 Every thought, every gate, every crystallization—observed in real-time."

ARCHITECTURE:
────────────────────────────────────────────────────────────────────────────────

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    STREAMING ORCHESTRATOR                                │
    │                                                                          │
    │   process_streaming(problem) ──▶ AsyncIterator[GenesisEvent]            │
    │                                                                          │
    │   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
    │   │   LENS     │  │  WISDOM    │  │  THOUGHT   │  │   ATTEST   │       │
    │   │  ANALYSIS  │──▶│  SEEDING  │──▶│ EXPANSION │──▶│  BINDING  │       │
    │   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘       │
    │         │               │               │               │              │
    │         ▼               ▼               ▼               ▼              │
    │   ┌─────────────────────────────────────────────────────────────┐      │
    │   │                    EVENT STREAM                              │      │
    │   │  [START] [LENS] [WISDOM] [THOUGHT] [SNR] [CRYSTAL] [ATTEST] │      │
    │   └─────────────────────────────────────────────────────────────┘      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Author: BIZRA Genesis Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from .genesis_events import (
    GenesisEvent,
    GenesisEventType,
    GenesisEventBus,
    GenesisEventListener,
    create_event,
    system_start,
    lens_activated,
    wisdom_seed_loaded,
    thought_created,
    thought_pruned,
    snr_computed,
    crystal_added,
    attestation_complete,
)
from .genesis_orchestrator import (
    GenesisOrchestrator,
    GenesisResult,
    DomainLens,
    ThoughtNode,
    WisdomEntry,
    SNR_THRESHOLD_HIGH,
    SNR_THRESHOLD_MEDIUM,
    SNR_THRESHOLD_IHSAN,
)


class StreamingGenesisOrchestrator(GenesisOrchestrator):
    """
    Genesis Orchestrator with real-time event streaming.
    
    Extends the base orchestrator to yield events at each processing
    phase, enabling real-time visualization in the Genesis Terminal.
    
    Key Features:
    - AsyncIterator-based streaming (process_streaming)
    - Event bus for multi-listener support
    - Correlation IDs for tracing related events
    - Configurable event granularity
    
    Usage:
        orchestrator = StreamingGenesisOrchestrator()
        
        async for event in orchestrator.process_streaming("Design X"):
            print(f"[{event.progress:.0%}] {event.phase}: {event.type.name}")
    """
    
    def __init__(
        self,
        beam_width: int = 8,
        max_depth: int = 5,
        fail_closed: bool = True,
        event_bus: Optional[GenesisEventBus] = None,
        emit_delay: float = 0.05,  # Delay between events for UX pacing
    ):
        """
        Initialize the streaming orchestrator.
        
        Args:
            beam_width: Number of thought paths to maintain
            max_depth: Maximum depth of thought expansion
            fail_closed: Whether to reject MEDIUM SNR in gating
            event_bus: Optional shared event bus
            emit_delay: Delay between events (for visualization pacing)
        """
        super().__init__(
            beam_width=beam_width,
            max_depth=max_depth,
            fail_closed=fail_closed,
        )
        self._event_bus = event_bus or GenesisEventBus(keep_history=True)
        self._emit_delay = emit_delay
        self._listeners: List[GenesisEventListener] = []
        self._last_result: Optional[GenesisResult] = None
    
    @property
    def last_result(self) -> Optional[GenesisResult]:
        """Get the last processing result."""
        return self._last_result
    
    @property
    def event_bus(self) -> GenesisEventBus:
        """Get the event bus."""
        return self._event_bus
    
    def add_listener(self, listener: GenesisEventListener) -> None:
        """Add an event listener."""
        self._listeners.append(listener)
        self._event_bus.add_listener(listener)
    
    def remove_listener(self, listener: GenesisEventListener) -> None:
        """Remove an event listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)
        self._event_bus.remove_listener(listener)
    
    async def _emit(self, event: GenesisEvent) -> None:
        """Emit an event to the bus and apply pacing delay."""
        await self._event_bus.emit(event)
        if self._emit_delay > 0:
            await asyncio.sleep(self._emit_delay)
    
    async def process_streaming(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[GenesisEvent]:
        """
        Process a problem with real-time event streaming.
        
        Yields events at each processing phase:
        1. SYSTEM_START (0%)
        2. LENS_* events (5-20%)
        3. WISDOM_* events (20-30%)
        4. THOUGHT_* events (30-70%)
        5. SNR_* events (70-85%)
        6. CRYSTAL_* events (85-95%)
        7. ATTEST_* events (95-100%)
        
        Args:
            problem: The problem to process
            context: Optional additional context
            
        Yields:
            GenesisEvent for each processing step
        """
        context = context or {}
        correlation_id = str(uuid.uuid4())[:8]
        
        self._total_operations += 1
        
        # ═════════════════════════════════════════════════════════════════════
        # Phase 1: System Start (0-5%)
        # ═════════════════════════════════════════════════════════════════════
        
        event = system_start(problem, correlation_id)
        await self._emit(event)
        yield event
        
        # ═════════════════════════════════════════════════════════════════════
        # Phase 2: Interdisciplinary Lens Analysis (5-20%)
        # ═════════════════════════════════════════════════════════════════════
        
        yield create_event(
            GenesisEventType.LENS_ANALYSIS_START,
            "Lens Analysis",
            0.05,
            correlation_id=correlation_id,
        )
        
        # Analyze through each lens
        lenses = list(DomainLens)
        for i, lens in enumerate(lenses):
            progress = 0.05 + (i / len(lenses)) * 0.12
            
            # Call the actual lens analyzer
            insight = await self.lens_system._lens_analyzers[lens](problem, context)
            
            event = lens_activated(
                lens_name=lens.value.upper(),
                confidence=insight.confidence,
                progress=progress,
                correlation_id=correlation_id,
            )
            await self._emit(event)
            yield event
        
        # Synthesize
        lens_insights = await self.lens_system.analyze(problem, context)
        synthesized = await self.lens_system.synthesize(lens_insights, problem)
        
        yield create_event(
            GenesisEventType.LENS_SYNTHESIS_COMPLETE,
            "Synthesis",
            0.20,
            correlation_id=correlation_id,
            snr=synthesized.snr_score,
            ihsan=synthesized.ihsan_score,
        )
        
        # ═════════════════════════════════════════════════════════════════════
        # Phase 3: Wisdom Seeding (Giants Protocol) (20-30%)
        # ═════════════════════════════════════════════════════════════════════
        
        yield create_event(
            GenesisEventType.WISDOM_SEEDING_START,
            "Giants Protocol",
            0.22,
            correlation_id=correlation_id,
        )
        
        wisdom_seeds = self._seed_from_giants(problem, synthesized)
        
        for i, seed in enumerate(wisdom_seeds[:5]):  # Limit events for UX
            progress = 0.22 + (i / max(len(wisdom_seeds), 1)) * 0.06
            event = wisdom_seed_loaded(
                wisdom_id=seed.id,
                title=seed.title,
                snr=seed.snr_score,
                progress=progress,
                correlation_id=correlation_id,
            )
            await self._emit(event)
            yield event
        
        yield create_event(
            GenesisEventType.WISDOM_SEEDING_COMPLETE,
            "Giants Protocol",
            0.30,
            correlation_id=correlation_id,
            seeds_count=len(wisdom_seeds),
        )
        
        # ═════════════════════════════════════════════════════════════════════
        # Phase 4: Graph of Thoughts Expansion (30-70%)
        # ═════════════════════════════════════════════════════════════════════
        
        yield create_event(
            GenesisEventType.THOUGHT_EXPANSION_START,
            "Expansion",
            0.32,
            correlation_id=correlation_id,
            beam_width=self.beam_width,
            max_depth=self.max_depth,
        )
        
        # Expand thoughts with streaming
        all_thoughts: List[ThoughtNode] = []
        
        # Create root node
        root_id = f"thought_{hashlib.sha256(synthesized.content.encode()).hexdigest()[:12]}"
        root_node = ThoughtNode(
            id=root_id,
            content=synthesized.content[:100] + "...",
            depth=0,
            snr_score=synthesized.snr_score,
            ihsan_score=synthesized.ihsan_score,
            confidence=0.85,
            wisdom_seeds=[w.id for w in wisdom_seeds],
        )
        
        self._thought_graph[root_id] = root_node
        self._root_nodes.append(root_id)
        all_thoughts.append(root_node)
        
        # Emit root thought
        event = thought_created(
            content=root_node.content[:80],
            snr=root_node.snr_score,
            ihsan=root_node.ihsan_score,
            depth=0,
            node_id=root_id,
            progress=0.35,
            correlation_id=correlation_id,
        )
        await self._emit(event)
        yield event
        
        # Beam search expansion
        beam = [root_node]
        
        for depth in range(1, self.max_depth + 1):
            depth_progress_base = 0.35 + (depth / self.max_depth) * 0.30
            
            yield create_event(
                GenesisEventType.THOUGHT_BEAM_COMPLETE,
                f"Depth {depth}",
                depth_progress_base,
                correlation_id=correlation_id,
                depth=depth,
                beam_size=len(beam),
            )
            
            next_beam = []
            
            for parent in beam:
                # Generate children (simplified for streaming demo)
                children = await self._generate_child_thoughts_streaming(
                    parent, wisdom_seeds, correlation_id, depth, depth_progress_base
                )
                
                for child in children:
                    child.depth = depth
                    child.parent_id = parent.id
                    parent.children_ids.append(child.id)
                    self._thought_graph[child.id] = child
                    all_thoughts.append(child)
                    
                    # Emit thought created
                    event = thought_created(
                        content=child.content[:80],
                        snr=child.snr_score,
                        ihsan=child.ihsan_score,
                        depth=depth,
                        node_id=child.id,
                        progress=depth_progress_base + 0.02,
                        correlation_id=correlation_id,
                    )
                    await self._emit(event)
                    yield event
                    
                    # Emit SNR score
                    passed = child.passes_snr_gate()
                    yield snr_computed(
                        score=child.snr_score,
                        ihsan=child.ihsan_score,
                        passed=passed,
                        progress=depth_progress_base + 0.03,
                        correlation_id=correlation_id,
                    )
                    
                    if passed:
                        next_beam.append(child)
                    else:
                        # Emit pruned event
                        yield thought_pruned(
                            node_id=child.id,
                            snr=child.snr_score,
                            reason="Below SNR threshold",
                            progress=depth_progress_base + 0.04,
                            correlation_id=correlation_id,
                        )
            
            # Prune to beam width
            next_beam.sort(key=lambda n: n.snr_score, reverse=True)
            beam = next_beam[:self.beam_width]
            
            if not beam:
                break
        
        yield create_event(
            GenesisEventType.THOUGHT_EXPANSION_COMPLETE,
            "Expansion",
            0.70,
            correlation_id=correlation_id,
            total_thoughts=len(all_thoughts),
        )
        
        # ═════════════════════════════════════════════════════════════════════
        # Phase 5: SNR Gating (70-85%)
        # ═════════════════════════════════════════════════════════════════════
        
        yield create_event(
            GenesisEventType.SNR_GATE_START,
            "SNR Gating",
            0.72,
            correlation_id=correlation_id,
        )
        
        # Gate all thoughts
        best_thoughts = self.snr_engine.rank_and_select(
            [(node, node.snr_score, node.ihsan_score) for node in all_thoughts],
            top_k=self.beam_width,
        )
        
        yield create_event(
            GenesisEventType.SNR_GATE_COMPLETE,
            "SNR Gating",
            0.85,
            correlation_id=correlation_id,
            passed_count=len(best_thoughts),
            rejected_count=len(all_thoughts) - len(best_thoughts),
        )
        
        # ═════════════════════════════════════════════════════════════════════
        # Phase 6: Crystallization (85-95%)
        # ═════════════════════════════════════════════════════════════════════
        
        yield create_event(
            GenesisEventType.CRYSTAL_START,
            "Crystallization",
            0.87,
            correlation_id=correlation_id,
        )
        
        crystallized = []
        for i, thought in enumerate(best_thoughts):
            if thought.snr_score >= SNR_THRESHOLD_HIGH:
                wisdom_id = f"wisdom_{thought.id}"
                
                # Create wisdom entry
                wisdom_entry = WisdomEntry(
                    id=wisdom_id,
                    title=f"Insight: {problem[:30]}...",
                    content=thought.content,
                    source="genesis_orchestrator",
                    snr_score=thought.snr_score,
                    ihsan_score=thought.ihsan_score,
                    observation_count=1,
                    first_observed=thought.created_at,
                    last_observed=thought.created_at,
                    related_concepts=list(thought.domain_tags),
                )
                
                self.wisdom_repo.add(wisdom_entry)
                
                crystallized.append({
                    "id": thought.id,
                    "wisdom_id": wisdom_id,
                    "snr": thought.snr_score,
                })
                
                progress = 0.87 + (i / max(len(best_thoughts), 1)) * 0.06
                event = crystal_added(
                    wisdom_id=wisdom_id,
                    title=wisdom_entry.title,
                    snr=thought.snr_score,
                    progress=progress,
                    correlation_id=correlation_id,
                )
                await self._emit(event)
                yield event
        
        # Save wisdom
        self.wisdom_repo.save()
        
        yield create_event(
            GenesisEventType.CRYSTAL_COMPLETE,
            "Crystallization",
            0.95,
            correlation_id=correlation_id,
            crystallized_count=len(crystallized),
        )
        
        # ═════════════════════════════════════════════════════════════════════
        # Phase 7: Genesis Attestation (95-100%)
        # ═════════════════════════════════════════════════════════════════════
        
        yield create_event(
            GenesisEventType.ATTEST_START,
            "Binding",
            0.96,
            correlation_id=correlation_id,
        )
        
        # Bind to genesis
        attestation = await self._bind_to_genesis(crystallized)
        
        yield create_event(
            GenesisEventType.ATTEST_HASH_COMPUTED,
            "Binding",
            0.98,
            correlation_id=correlation_id,
            hash=attestation["attestation_hash"][:16] + "...",
        )
        
        yield create_event(
            GenesisEventType.ATTEST_BOUND_TO_NODE,
            "Binding",
            0.99,
            correlation_id=correlation_id,
            node_id=attestation.get("node_id"),
        )
        
        # Final event
        event = attestation_complete(
            attestation_hash=attestation["attestation_hash"],
            node_id=attestation.get("node_id"),
            correlation_id=correlation_id,
        )
        await self._emit(event)
        yield event
        
        # Update stats
        self._high_snr_outputs += len(crystallized)
        
        # Store result for retrieval
        if best_thoughts:
            avg_snr = sum(t.snr_score for t in best_thoughts) / len(best_thoughts)
            avg_ihsan = sum(t.ihsan_score for t in best_thoughts) / len(best_thoughts)
        else:
            avg_snr = 0.7
            avg_ihsan = 0.95
        
        self._last_result = GenesisResult(
            synthesis=f"Synthesized {len(best_thoughts)} insights across {len([l for l in DomainLens])} domains",
            confidence=avg_snr,
            snr_score=avg_snr,
            ihsan_score=avg_ihsan,
            attestation_hash=attestation["attestation_hash"],
            thoughts=best_thoughts,
            lenses_applied=[l.value for l in DomainLens],
            wisdom_seeds_used=len(wisdom_seeds),
            processing_time_ms=0.0,  # Could add timing
        )
    
    async def _generate_child_thoughts_streaming(
        self,
        parent: ThoughtNode,
        wisdom_seeds: List[WisdomEntry],
        correlation_id: str,
        depth: int,
        progress_base: float,
    ) -> List[ThoughtNode]:
        """
        Generate child thoughts with streaming-friendly output.
        
        Generates 2-3 child thoughts per parent with varied SNR scores.
        """
        children = []
        directions = ["refine", "contrast", "synthesize"]
        
        for direction in directions:
            # Generate varied content
            child_content = f"[{direction.upper()}] Extending {parent.content[:50]}..."
            
            # Compute child metrics with realistic variance
            base_snr = parent.snr_score
            variance = random.uniform(-0.15, 0.10)  # Slight bias toward decay
            child_snr = max(0.3, min(0.98, base_snr + variance))
            
            # Ihsan tends to be stable
            child_ihsan = parent.ihsan_score + random.uniform(-0.02, 0.02)
            child_ihsan = max(0.90, min(0.99, child_ihsan))
            
            child_id = f"thought_{hashlib.sha256(child_content.encode()).hexdigest()[:12]}"
            
            child = ThoughtNode(
                id=child_id,
                content=child_content,
                depth=depth,
                snr_score=child_snr,
                ihsan_score=child_ihsan,
                confidence=parent.confidence * 0.95,
                wisdom_seeds=parent.wisdom_seeds,
            )
            
            children.append(child)
        
        return children


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


async def run_streaming_demo(problem: str) -> None:
    """
    Demonstration of streaming orchestrator.
    
    Args:
        problem: The problem to process
    """
    orchestrator = StreamingGenesisOrchestrator(
        beam_width=4,
        max_depth=3,
        fail_closed=False,
        emit_delay=0.1,
    )
    
    print("\n" + "═" * 70)
    print("GENESIS ORCHESTRATOR - STREAMING DEMO")
    print("═" * 70)
    print(f"\nProblem: {problem[:60]}...")
    print("-" * 70)
    
    async for event in orchestrator.process_streaming(problem):
        # Color based on event type
        type_name = event.type.name
        progress_bar = "█" * int(event.progress * 20) + "░" * (20 - int(event.progress * 20))
        
        # Highlight important events
        if "COMPLETE" in type_name or "ATTEST" in type_name:
            print(f"\n  ✓ [{progress_bar}] {event.phase}: {type_name}")
        elif "CREATED" in type_name:
            snr = event.data.get("snr", 0)
            indicator = "🟢" if snr >= 0.8 else "🟡" if snr >= 0.5 else "🔴"
            print(f"    {indicator} SNR: {snr:.2f} - {event.data.get('content', '')[:50]}")
        elif "PRUNED" in type_name:
            print(f"    ✗ Pruned: {event.data.get('node_id', '')[:16]}")
        elif "LENS" in type_name and "ACTIVATED" in type_name:
            print(f"    ◈ Lens: {event.data.get('lens', '')} (conf: {event.data.get('confidence', 0):.2f})")
        else:
            print(f"  [{progress_bar}] {event.phase}: {type_name}")
    
    print("\n" + "═" * 70)
    print("STREAMING COMPLETE")
    print("═" * 70)


if __name__ == "__main__":
    import asyncio
    
    asyncio.run(run_streaming_demo(
        "Design a self-correcting governance mechanism for the Genesis Node "
        "that maintains Ihsān compliance under adversarial conditions"
    ))
