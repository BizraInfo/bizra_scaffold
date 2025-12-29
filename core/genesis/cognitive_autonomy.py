"""
Autonomous Cognitive Core — Meta-Cognitive Self-Improvement Engine.

This is the meta-layer that watches the Genesis Orchestrator think,
learns from its reasoning chains, and crystallizes wisdom autonomously.

Architecture:
────────────────────────────────────────────────────────────────────
                    ┌─────────────────────────────┐
                    │   META-COGNITIVE MONITOR    │
                    │  (Watches the system think) │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌───────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  REASONING    │        │    INTEGRITY    │        │    WISDOM       │
│  LOOP         │◄──────►│    FLYWHEEL     │◄──────►│    CRYSTAL-     │
│  CONTROLLER   │        │                 │        │    LIZER        │
└───────────────┘        └─────────────────┘        └─────────────────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   STREAMING GENESIS         │
                    │   ORCHESTRATOR              │
                    └─────────────────────────────┘

The Cognitive Autonomy Engine embodies:
1. Interdisciplinary Thinking — Cross-domain pattern recognition
2. Graph of Thoughts — Beam search with meta-level analysis
3. SNR Highest Score — Quality gating at meta level
4. Giants Protocol — Learning from accumulated wisdom

Author: BIZRA Genesis System
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from core.genesis.genesis_events import (
    GenesisEvent,
    GenesisEventBus,
    GenesisEventListener,
    GenesisEventType,
    create_event,
)
from core.genesis.genesis_orchestrator import (
    GenesisOrchestrator,
    GenesisResult,
    InterdisciplinaryLensSystem,
    ThoughtNode,
    WisdomEntry,
    WisdomRepository,
)
from core.genesis.genesis_orchestrator_streaming import StreamingGenesisOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class CognitiveState(Enum):
    """State of the cognitive autonomy engine."""
    
    DORMANT = auto()        # Not yet started
    AWAKENING = auto()      # Initializing
    REASONING = auto()      # Active reasoning cycle
    REFLECTING = auto()     # Meta-cognitive analysis
    CRYSTALLIZING = auto()  # Wisdom extraction
    INTEGRATING = auto()    # Flywheel improvement
    IDLE = auto()           # Waiting for next task
    SUSPENDED = auto()      # Paused by operator
    TERMINATED = auto()     # Shutdown complete


class ReasoningQuality(Enum):
    """Quality assessment of a reasoning cycle."""
    
    EXCEPTIONAL = "exceptional"   # SNR ≥ 0.95, all lenses converged
    HIGH = "high"                 # SNR ≥ 0.85, strong convergence
    SATISFACTORY = "satisfactory" # SNR ≥ 0.70, acceptable
    MARGINAL = "marginal"         # SNR ≥ 0.50, needs improvement
    FAILED = "failed"             # SNR < 0.50 or error


class LearningSignal(Enum):
    """Type of learning signal from meta-cognition."""
    
    REINFORCE = auto()      # Successful pattern, strengthen
    PRUNE = auto()          # Failed pattern, weaken
    EXPLORE = auto()        # Novel pattern, investigate
    CONSOLIDATE = auto()    # Repeated success, crystallize


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class ReasoningCycleMetrics:
    """Metrics from a single reasoning cycle."""
    
    cycle_id: str
    problem_hash: str
    duration_ms: float
    thought_nodes_created: int
    thought_nodes_pruned: int
    lenses_activated: Set[str]
    final_snr: float
    ihsan_score: float
    wisdom_seeds_used: int
    wisdom_crystallized: int
    quality: ReasoningQuality
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "problem_hash": self.problem_hash,
            "duration_ms": self.duration_ms,
            "thought_nodes_created": self.thought_nodes_created,
            "thought_nodes_pruned": self.thought_nodes_pruned,
            "lenses_activated": list(self.lenses_activated),
            "final_snr": self.final_snr,
            "ihsan_score": self.ihsan_score,
            "wisdom_seeds_used": self.wisdom_seeds_used,
            "wisdom_crystallized": self.wisdom_crystallized,
            "quality": self.quality.value,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningCycleMetrics:
        """Deserialize from dictionary."""
        return cls(
            cycle_id=data["cycle_id"],
            problem_hash=data["problem_hash"],
            duration_ms=data["duration_ms"],
            thought_nodes_created=data["thought_nodes_created"],
            thought_nodes_pruned=data["thought_nodes_pruned"],
            lenses_activated=set(data["lenses_activated"]),
            final_snr=data["final_snr"],
            ihsan_score=data["ihsan_score"],
            wisdom_seeds_used=data["wisdom_seeds_used"],
            wisdom_crystallized=data["wisdom_crystallized"],
            quality=ReasoningQuality(data["quality"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class MetaCognitiveInsight:
    """Insight from meta-cognitive analysis."""
    
    insight_id: str
    source_cycles: List[str]  # Cycle IDs that contributed
    pattern_type: str
    description: str
    confidence: float  # 0.0 to 1.0
    learning_signal: LearningSignal
    recommended_action: str
    snr_impact: float  # Expected SNR improvement
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "insight_id": self.insight_id,
            "source_cycles": self.source_cycles,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "confidence": self.confidence,
            "learning_signal": self.learning_signal.name,
            "recommended_action": self.recommended_action,
            "snr_impact": self.snr_impact,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class IntegrityFlywheelState:
    """State of the integrity flywheel."""
    
    momentum: float = 0.0           # Accumulated positive outcomes
    resistance: float = 0.0         # Accumulated negative outcomes
    angular_velocity: float = 0.0   # Current improvement rate
    total_cycles: int = 0
    exceptional_cycles: int = 0
    failed_cycles: int = 0
    last_crystallization: Optional[datetime] = None
    wisdom_entries_added: int = 0
    
    @property
    def efficiency(self) -> float:
        """Calculate flywheel efficiency (0.0 to 1.0)."""
        if self.total_cycles == 0:
            return 0.0
        return self.exceptional_cycles / self.total_cycles
    
    @property
    def health(self) -> str:
        """Flywheel health status."""
        ratio = self.momentum / max(self.resistance, 0.01)
        if ratio > 10:
            return "THRIVING"
        elif ratio > 5:
            return "HEALTHY"
        elif ratio > 2:
            return "STABLE"
        elif ratio > 1:
            return "STRAINED"
        else:
            return "DEGRADED"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "momentum": self.momentum,
            "resistance": self.resistance,
            "angular_velocity": self.angular_velocity,
            "total_cycles": self.total_cycles,
            "exceptional_cycles": self.exceptional_cycles,
            "failed_cycles": self.failed_cycles,
            "last_crystallization": (
                self.last_crystallization.isoformat()
                if self.last_crystallization else None
            ),
            "wisdom_entries_added": self.wisdom_entries_added,
            "efficiency": self.efficiency,
            "health": self.health,
        }


# =============================================================================
# PROTOCOLS
# =============================================================================


class MetaCognitiveObserver(Protocol):
    """Protocol for observing meta-cognitive events."""
    
    async def on_state_change(
        self,
        old_state: CognitiveState,
        new_state: CognitiveState,
    ) -> None:
        """Called when cognitive state changes."""
        ...
    
    async def on_cycle_complete(
        self,
        metrics: ReasoningCycleMetrics,
    ) -> None:
        """Called when a reasoning cycle completes."""
        ...
    
    async def on_insight_discovered(
        self,
        insight: MetaCognitiveInsight,
    ) -> None:
        """Called when meta-cognition discovers an insight."""
        ...
    
    async def on_flywheel_update(
        self,
        state: IntegrityFlywheelState,
    ) -> None:
        """Called when the integrity flywheel updates."""
        ...


# =============================================================================
# META-COGNITIVE MONITOR
# =============================================================================


class MetaCognitiveMonitor:
    """
    The system that watches the system think.
    
    Analyzes reasoning patterns across cycles to discover
    meta-level insights that can improve future reasoning.
    """
    
    def __init__(
        self,
        history_window: int = 100,
        insight_threshold: float = 0.70,
    ):
        """
        Initialize the meta-cognitive monitor.
        
        Args:
            history_window: Number of cycles to analyze
            insight_threshold: Confidence threshold for insights
        """
        self._history_window = history_window
        self._insight_threshold = insight_threshold
        self._cycle_history: List[ReasoningCycleMetrics] = []
        self._insights: List[MetaCognitiveInsight] = []
        self._pattern_cache: Dict[str, int] = {}  # Pattern -> occurrence count
    
    def record_cycle(self, metrics: ReasoningCycleMetrics) -> None:
        """Record a completed reasoning cycle for analysis."""
        self._cycle_history.append(metrics)
        
        # Trim to window size
        if len(self._cycle_history) > self._history_window:
            self._cycle_history = self._cycle_history[-self._history_window:]
        
        # Update pattern cache
        self._update_patterns(metrics)
    
    def _update_patterns(self, metrics: ReasoningCycleMetrics) -> None:
        """Update pattern recognition cache."""
        # Lens activation pattern
        lens_pattern = "_".join(sorted(metrics.lenses_activated))
        self._pattern_cache[f"lens:{lens_pattern}"] = (
            self._pattern_cache.get(f"lens:{lens_pattern}", 0) + 1
        )
        
        # Quality pattern
        self._pattern_cache[f"quality:{metrics.quality.value}"] = (
            self._pattern_cache.get(f"quality:{metrics.quality.value}", 0) + 1
        )
        
        # SNR band pattern
        snr_band = self._snr_to_band(metrics.final_snr)
        self._pattern_cache[f"snr:{snr_band}"] = (
            self._pattern_cache.get(f"snr:{snr_band}", 0) + 1
        )
    
    def _snr_to_band(self, snr: float) -> str:
        """Convert SNR to band label."""
        if snr >= 0.95:
            return "exceptional"
        elif snr >= 0.85:
            return "high"
        elif snr >= 0.70:
            return "good"
        elif snr >= 0.50:
            return "medium"
        else:
            return "low"
    
    async def analyze(self) -> List[MetaCognitiveInsight]:
        """
        Analyze cycle history to discover insights.
        
        Returns:
            List of discovered insights
        """
        insights: List[MetaCognitiveInsight] = []
        
        if len(self._cycle_history) < 5:
            return insights
        
        # Pattern analysis
        insights.extend(await self._analyze_lens_patterns())
        insights.extend(await self._analyze_quality_trends())
        insights.extend(await self._analyze_snr_correlations())
        insights.extend(await self._analyze_efficiency_patterns())
        
        # Filter by confidence threshold
        insights = [i for i in insights if i.confidence >= self._insight_threshold]
        
        # Store discoveries
        self._insights.extend(insights)
        
        return insights
    
    async def _analyze_lens_patterns(self) -> List[MetaCognitiveInsight]:
        """Analyze which lens combinations lead to success."""
        insights: List[MetaCognitiveInsight] = []
        
        # Group by lens pattern
        pattern_outcomes: Dict[str, List[ReasoningQuality]] = {}
        for cycle in self._cycle_history:
            pattern = frozenset(cycle.lenses_activated)
            key = "_".join(sorted(pattern))
            if key not in pattern_outcomes:
                pattern_outcomes[key] = []
            pattern_outcomes[key].append(cycle.quality)
        
        # Find winning patterns
        for pattern, outcomes in pattern_outcomes.items():
            if len(outcomes) < 3:
                continue
            
            exceptional_rate = sum(
                1 for o in outcomes
                if o in (ReasoningQuality.EXCEPTIONAL, ReasoningQuality.HIGH)
            ) / len(outcomes)
            
            if exceptional_rate >= 0.8:
                insights.append(MetaCognitiveInsight(
                    insight_id=f"lens_pattern_{uuid.uuid4().hex[:8]}",
                    source_cycles=[c.cycle_id for c in self._cycle_history[-10:]],
                    pattern_type="lens_synergy",
                    description=f"Lens combination [{pattern}] yields {exceptional_rate:.0%} success rate",
                    confidence=min(exceptional_rate, 0.95),
                    learning_signal=LearningSignal.REINFORCE,
                    recommended_action=f"Prioritize lens activation: {pattern}",
                    snr_impact=0.05,
                ))
        
        return insights
    
    async def _analyze_quality_trends(self) -> List[MetaCognitiveInsight]:
        """Analyze quality trends over time."""
        insights: List[MetaCognitiveInsight] = []
        
        if len(self._cycle_history) < 10:
            return insights
        
        # Compare recent vs older
        recent = self._cycle_history[-10:]
        older = self._cycle_history[-20:-10] if len(self._cycle_history) >= 20 else []
        
        if not older:
            return insights
        
        recent_snr = sum(c.final_snr for c in recent) / len(recent)
        older_snr = sum(c.final_snr for c in older) / len(older)
        
        delta = recent_snr - older_snr
        
        if delta > 0.05:
            insights.append(MetaCognitiveInsight(
                insight_id=f"trend_improving_{uuid.uuid4().hex[:8]}",
                source_cycles=[c.cycle_id for c in recent],
                pattern_type="quality_trend",
                description=f"SNR improving: {older_snr:.2f} → {recent_snr:.2f} (+{delta:.2f})",
                confidence=0.85,
                learning_signal=LearningSignal.CONSOLIDATE,
                recommended_action="Continue current reasoning strategies",
                snr_impact=delta,
            ))
        elif delta < -0.05:
            insights.append(MetaCognitiveInsight(
                insight_id=f"trend_degrading_{uuid.uuid4().hex[:8]}",
                source_cycles=[c.cycle_id for c in recent],
                pattern_type="quality_trend",
                description=f"SNR degrading: {older_snr:.2f} → {recent_snr:.2f} ({delta:.2f})",
                confidence=0.85,
                learning_signal=LearningSignal.EXPLORE,
                recommended_action="Investigate reasoning strategy changes",
                snr_impact=abs(delta),
            ))
        
        return insights
    
    async def _analyze_snr_correlations(self) -> List[MetaCognitiveInsight]:
        """Analyze correlations between factors and SNR."""
        insights: List[MetaCognitiveInsight] = []
        
        # Analyze thought efficiency
        high_snr_cycles = [c for c in self._cycle_history if c.final_snr >= 0.85]
        low_snr_cycles = [c for c in self._cycle_history if c.final_snr < 0.70]
        
        if len(high_snr_cycles) >= 3 and len(low_snr_cycles) >= 3:
            high_prune_rate = sum(
                c.thought_nodes_pruned / max(c.thought_nodes_created, 1)
                for c in high_snr_cycles
            ) / len(high_snr_cycles)
            
            low_prune_rate = sum(
                c.thought_nodes_pruned / max(c.thought_nodes_created, 1)
                for c in low_snr_cycles
            ) / len(low_snr_cycles)
            
            if high_prune_rate > low_prune_rate + 0.1:
                insights.append(MetaCognitiveInsight(
                    insight_id=f"prune_correlation_{uuid.uuid4().hex[:8]}",
                    source_cycles=[c.cycle_id for c in high_snr_cycles[-5:]],
                    pattern_type="efficiency_correlation",
                    description=f"Higher pruning rate ({high_prune_rate:.0%}) correlates with better SNR",
                    confidence=0.80,
                    learning_signal=LearningSignal.REINFORCE,
                    recommended_action="Increase beam search pruning aggressiveness",
                    snr_impact=0.08,
                ))
        
        return insights
    
    async def _analyze_efficiency_patterns(self) -> List[MetaCognitiveInsight]:
        """Analyze reasoning efficiency patterns."""
        insights: List[MetaCognitiveInsight] = []
        
        # Find cycles that achieved high SNR with minimal thoughts
        efficient_cycles = [
            c for c in self._cycle_history
            if c.final_snr >= 0.85 and c.thought_nodes_created <= 10
        ]
        
        if len(efficient_cycles) >= 3:
            avg_wisdom_seeds = sum(
                c.wisdom_seeds_used for c in efficient_cycles
            ) / len(efficient_cycles)
            
            if avg_wisdom_seeds >= 2:
                insights.append(MetaCognitiveInsight(
                    insight_id=f"wisdom_efficiency_{uuid.uuid4().hex[:8]}",
                    source_cycles=[c.cycle_id for c in efficient_cycles[-5:]],
                    pattern_type="wisdom_leverage",
                    description=f"Giants Protocol enabled efficient reasoning (avg {avg_wisdom_seeds:.1f} seeds)",
                    confidence=0.85,
                    learning_signal=LearningSignal.CONSOLIDATE,
                    recommended_action="Increase wisdom repository access priority",
                    snr_impact=0.10,
                ))
        
        return insights
    
    def get_recent_insights(self, limit: int = 10) -> List[MetaCognitiveInsight]:
        """Get recent insights."""
        return self._insights[-limit:]
    
    def get_pattern_frequencies(self) -> Dict[str, int]:
        """Get pattern frequency cache."""
        return dict(self._pattern_cache)


# =============================================================================
# WISDOM CRYSTALLIZATION PIPELINE
# =============================================================================


class WisdomCrystallizationPipeline:
    """
    Automatic extraction and storage of insights from successful reasoning.
    
    The pipeline watches for high-SNR reasoning chains and extracts
    reusable wisdom that can accelerate future reasoning.
    """
    
    def __init__(
        self,
        wisdom_repo: WisdomRepository,
        crystallization_threshold: float = 0.88,
        ihsan_requirement: float = 0.95,
    ):
        """
        Initialize the crystallization pipeline.
        
        Args:
            wisdom_repo: Repository for storing crystallized wisdom
            crystallization_threshold: Minimum SNR for crystallization
            ihsan_requirement: Minimum Ihsān score for crystallization
        """
        self._wisdom_repo = wisdom_repo
        self._crystallization_threshold = crystallization_threshold
        self._ihsan_requirement = ihsan_requirement
        self._pending_crystals: List[Dict[str, Any]] = []
        self._crystallization_count = 0
    
    async def process_result(
        self,
        result: GenesisResult,
        metrics: ReasoningCycleMetrics,
    ) -> List[WisdomEntry]:
        """
        Process a reasoning result for crystallization.
        
        Args:
            result: The reasoning result
            metrics: Cycle metrics
            
        Returns:
            List of crystallized wisdom entries
        """
        crystals: List[WisdomEntry] = []
        
        # Check crystallization eligibility
        if not self._is_eligible(metrics):
            return crystals
        
        # Extract crystallizable insights
        insights = await self._extract_insights(result, metrics)
        
        # Crystallize each insight
        for insight in insights:
            crystal = await self._crystallize(insight, result, metrics)
            if crystal:
                crystals.append(crystal)
                self._crystallization_count += 1
        
        return crystals
    
    def _is_eligible(self, metrics: ReasoningCycleMetrics) -> bool:
        """Check if cycle is eligible for crystallization."""
        return (
            metrics.final_snr >= self._crystallization_threshold
            and metrics.ihsan_score >= self._ihsan_requirement
            and metrics.quality in (
                ReasoningQuality.EXCEPTIONAL,
                ReasoningQuality.HIGH,
            )
        )
    
    async def _extract_insights(
        self,
        result: GenesisResult,
        metrics: ReasoningCycleMetrics,
    ) -> List[Dict[str, Any]]:
        """Extract crystallizable insights from result."""
        insights: List[Dict[str, Any]] = []
        
        # Extract from synthesis
        if result.synthesis:
            # Check for novel patterns
            synthesis_hash = hashlib.sha256(
                result.synthesis.encode()
            ).hexdigest()[:16]
            
            insights.append({
                "type": "synthesis_pattern",
                "content": result.synthesis[:500],  # Truncate for storage
                "hash": synthesis_hash,
                "snr": metrics.final_snr,
                "lenses": list(metrics.lenses_activated),
            })
        
        # Extract from attestation (high-trust insights)
        if result.attestation_hash:
            insights.append({
                "type": "attested_conclusion",
                "content": f"Attested reasoning with hash {result.attestation_hash[:16]}...",
                "hash": result.attestation_hash[:16],
                "snr": metrics.final_snr,
                "attested": True,
            })
        
        return insights
    
    async def _crystallize(
        self,
        insight: Dict[str, Any],
        result: GenesisResult,
        metrics: ReasoningCycleMetrics,
    ) -> Optional[WisdomEntry]:
        """Crystallize an insight into wisdom."""
        # Check for duplicates
        existing = self._wisdom_repo.search(insight["hash"], top_k=1)
        if existing and existing[0].snr_score >= insight["snr"]:
            return None  # Already have equal or better wisdom
        
        now = datetime.now(timezone.utc)
        
        # Create wisdom entry (matching WisdomEntry dataclass signature)
        entry = WisdomEntry(
            id=f"crystal_{uuid.uuid4().hex[:12]}",
            title=f"{insight['type'].replace('_', ' ').title()}",
            content=insight["content"],
            source=f"Autonomous crystallization from cycle {metrics.cycle_id}",
            snr_score=insight["snr"],
            ihsan_score=metrics.ihsan_score,
            observation_count=1,
            first_observed=now,
            last_observed=now,
            related_concepts=[insight["type"], "auto_crystallized"],
        )
        
        # Store in repository
        self._wisdom_repo.add(entry)
        
        logger.info(
            f"Crystallized wisdom: {entry.id} (SNR: {entry.snr_score:.2f})"
        )
        
        return entry
    
    @property
    def crystallization_count(self) -> int:
        """Total crystals created."""
        return self._crystallization_count


# =============================================================================
# INTEGRITY FLYWHEEL
# =============================================================================


class IntegrityFlywheel:
    """
    Self-reinforcing improvement loop.
    
    Positive outcomes add momentum, negative outcomes add resistance.
    High momentum enables more aggressive exploration.
    Low momentum triggers conservative consolidation.
    """
    
    # Momentum multipliers
    EXCEPTIONAL_MOMENTUM = 2.0
    HIGH_MOMENTUM = 1.5
    SATISFACTORY_MOMENTUM = 0.5
    MARGINAL_RESISTANCE = 0.5
    FAILED_RESISTANCE = 2.0
    
    # Decay rate (per cycle)
    MOMENTUM_DECAY = 0.02
    RESISTANCE_DECAY = 0.01
    
    def __init__(self, initial_momentum: float = 1.0):
        """Initialize the flywheel."""
        self._state = IntegrityFlywheelState(
            momentum=initial_momentum,
            resistance=0.0,
            angular_velocity=0.0,
        )
    
    def update(self, metrics: ReasoningCycleMetrics) -> IntegrityFlywheelState:
        """
        Update flywheel based on cycle outcome.
        
        Args:
            metrics: Completed cycle metrics
            
        Returns:
            Updated flywheel state
        """
        self._state.total_cycles += 1
        
        # Apply decay
        self._state.momentum *= (1 - self.MOMENTUM_DECAY)
        self._state.resistance *= (1 - self.RESISTANCE_DECAY)
        
        # Add momentum or resistance based on quality
        if metrics.quality == ReasoningQuality.EXCEPTIONAL:
            self._state.momentum += self.EXCEPTIONAL_MOMENTUM
            self._state.exceptional_cycles += 1
        elif metrics.quality == ReasoningQuality.HIGH:
            self._state.momentum += self.HIGH_MOMENTUM
        elif metrics.quality == ReasoningQuality.SATISFACTORY:
            self._state.momentum += self.SATISFACTORY_MOMENTUM
        elif metrics.quality == ReasoningQuality.MARGINAL:
            self._state.resistance += self.MARGINAL_RESISTANCE
        elif metrics.quality == ReasoningQuality.FAILED:
            self._state.resistance += self.FAILED_RESISTANCE
            self._state.failed_cycles += 1
        
        # Calculate angular velocity (improvement rate)
        net_force = self._state.momentum - self._state.resistance
        self._state.angular_velocity = net_force / max(self._state.total_cycles, 1)
        
        return self._state
    
    def record_crystallization(self) -> None:
        """Record a wisdom crystallization event."""
        self._state.wisdom_entries_added += 1
        self._state.last_crystallization = datetime.now(timezone.utc)
        # Crystallization adds bonus momentum
        self._state.momentum += 0.5
    
    @property
    def state(self) -> IntegrityFlywheelState:
        """Current flywheel state."""
        return self._state
    
    @property
    def should_explore(self) -> bool:
        """Whether flywheel momentum supports exploration."""
        return self._state.angular_velocity > 0.1
    
    @property
    def should_consolidate(self) -> bool:
        """Whether flywheel suggests consolidation."""
        return self._state.angular_velocity < -0.05


# =============================================================================
# REASONING LOOP CONTROLLER
# =============================================================================


class ReasoningLoopController:
    """
    Manages continuous reasoning cycles.
    
    The controller orchestrates:
    1. Problem intake
    2. Reasoning execution
    3. Metrics collection
    4. Meta-cognitive analysis
    5. Wisdom crystallization
    6. Flywheel updates
    """
    
    def __init__(
        self,
        orchestrator: StreamingGenesisOrchestrator,
        wisdom_repo: WisdomRepository,
        max_concurrent_cycles: int = 1,
    ):
        """
        Initialize the reasoning loop controller.
        
        Args:
            orchestrator: The streaming genesis orchestrator
            wisdom_repo: Wisdom repository for storage
            max_concurrent_cycles: Max parallel reasoning (default 1)
        """
        self._orchestrator = orchestrator
        self._wisdom_repo = wisdom_repo
        self._max_concurrent = max_concurrent_cycles
        
        # Components
        self._meta_monitor = MetaCognitiveMonitor()
        self._crystallizer = WisdomCrystallizationPipeline(wisdom_repo)
        self._flywheel = IntegrityFlywheel()
        
        # State
        self._current_state = CognitiveState.DORMANT
        self._active_cycles: Dict[str, asyncio.Task] = {}
        self._observers: List[MetaCognitiveObserver] = []
        self._cycle_count = 0
    
    async def _transition_state(self, new_state: CognitiveState) -> None:
        """Transition to a new cognitive state."""
        old_state = self._current_state
        self._current_state = new_state
        
        for observer in self._observers:
            try:
                await observer.on_state_change(old_state, new_state)
            except Exception as e:
                logger.warning(f"Observer error on state change: {e}")
    
    def add_observer(self, observer: MetaCognitiveObserver) -> None:
        """Add a meta-cognitive observer."""
        self._observers.append(observer)
    
    def remove_observer(self, observer: MetaCognitiveObserver) -> None:
        """Remove a meta-cognitive observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    async def reason(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GenesisResult, ReasoningCycleMetrics]:
        """
        Execute a complete reasoning cycle with meta-cognition.
        
        Args:
            problem: The problem to reason about
            context: Optional additional context
            
        Returns:
            Tuple of (result, metrics)
        """
        cycle_id = f"cycle_{uuid.uuid4().hex[:12]}"
        problem_hash = hashlib.sha256(problem.encode()).hexdigest()[:16]
        
        await self._transition_state(CognitiveState.REASONING)
        
        start_time = time.perf_counter()
        
        # Tracking
        thoughts_created = 0
        thoughts_pruned = 0
        lenses_activated: Set[str] = set()
        wisdom_seeds = 0
        
        # Collect events during streaming
        events: List[GenesisEvent] = []
        
        try:
            async for event in self._orchestrator.process_streaming(problem, context):
                events.append(event)
                
                # Track metrics from events
                if event.type == GenesisEventType.THOUGHT_NODE_CREATED:
                    thoughts_created += 1
                elif event.type == GenesisEventType.THOUGHT_NODE_PRUNED:
                    thoughts_pruned += 1
                elif event.type == GenesisEventType.LENS_ACTIVATED:
                    lenses_activated.add(event.data.get("lens", "unknown"))
                elif event.type == GenesisEventType.WISDOM_SEED_LOADED:
                    wisdom_seeds += 1
            
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Get the final result
            result = self._orchestrator.last_result
            if not result:
                # Fallback if no result stored
                result = GenesisResult(
                    synthesis="Reasoning completed",
                    confidence=0.7,
                    snr_score=0.7,
                    ihsan_score=0.95,
                    attestation_hash=None,
                    thoughts=[],
                    lenses_applied=list(lenses_activated),
                )
            
            # Determine quality
            quality = self._assess_quality(result)
            
            # Build metrics
            metrics = ReasoningCycleMetrics(
                cycle_id=cycle_id,
                problem_hash=problem_hash,
                duration_ms=duration_ms,
                thought_nodes_created=thoughts_created,
                thought_nodes_pruned=thoughts_pruned,
                lenses_activated=lenses_activated,
                final_snr=result.snr_score,
                ihsan_score=result.ihsan_score,
                wisdom_seeds_used=wisdom_seeds,
                wisdom_crystallized=0,  # Updated below
                quality=quality,
            )
            
            self._cycle_count += 1
            
        except Exception as e:
            logger.error(f"Reasoning cycle failed: {e}")
            end_time = time.perf_counter()
            
            result = GenesisResult(
                synthesis=f"Error: {str(e)}",
                confidence=0.0,
                snr_score=0.0,
                ihsan_score=0.0,
                attestation_hash=None,
                thoughts=[],
                lenses_applied=[],
            )
            
            metrics = ReasoningCycleMetrics(
                cycle_id=cycle_id,
                problem_hash=problem_hash,
                duration_ms=(end_time - start_time) * 1000,
                thought_nodes_created=thoughts_created,
                thought_nodes_pruned=thoughts_pruned,
                lenses_activated=lenses_activated,
                final_snr=0.0,
                ihsan_score=0.0,
                wisdom_seeds_used=wisdom_seeds,
                wisdom_crystallized=0,
                quality=ReasoningQuality.FAILED,
            )
        
        # Meta-cognitive reflection
        await self._transition_state(CognitiveState.REFLECTING)
        self._meta_monitor.record_cycle(metrics)
        insights = await self._meta_monitor.analyze()
        
        for insight in insights:
            for observer in self._observers:
                try:
                    await observer.on_insight_discovered(insight)
                except Exception as e:
                    logger.warning(f"Observer error on insight: {e}")
        
        # Crystallization
        await self._transition_state(CognitiveState.CRYSTALLIZING)
        crystals = await self._crystallizer.process_result(result, metrics)
        
        # Update metrics with crystallization count
        metrics = ReasoningCycleMetrics(
            cycle_id=metrics.cycle_id,
            problem_hash=metrics.problem_hash,
            duration_ms=metrics.duration_ms,
            thought_nodes_created=metrics.thought_nodes_created,
            thought_nodes_pruned=metrics.thought_nodes_pruned,
            lenses_activated=metrics.lenses_activated,
            final_snr=metrics.final_snr,
            ihsan_score=metrics.ihsan_score,
            wisdom_seeds_used=metrics.wisdom_seeds_used,
            wisdom_crystallized=len(crystals),
            quality=metrics.quality,
            timestamp=metrics.timestamp,
        )
        
        # Flywheel integration
        await self._transition_state(CognitiveState.INTEGRATING)
        flywheel_state = self._flywheel.update(metrics)
        
        for crystal in crystals:
            self._flywheel.record_crystallization()
        
        # Notify observers
        for observer in self._observers:
            try:
                await observer.on_cycle_complete(metrics)
                await observer.on_flywheel_update(flywheel_state)
            except Exception as e:
                logger.warning(f"Observer error: {e}")
        
        await self._transition_state(CognitiveState.IDLE)
        
        return result, metrics
    
    def _assess_quality(self, result: GenesisResult) -> ReasoningQuality:
        """Assess the quality of a reasoning result."""
        snr = result.snr_score
        ihsan = result.ihsan_score
        
        if snr >= 0.95 and ihsan >= 0.98:
            return ReasoningQuality.EXCEPTIONAL
        elif snr >= 0.85 and ihsan >= 0.95:
            return ReasoningQuality.HIGH
        elif snr >= 0.70 and ihsan >= 0.90:
            return ReasoningQuality.SATISFACTORY
        elif snr >= 0.50:
            return ReasoningQuality.MARGINAL
        else:
            return ReasoningQuality.FAILED
    
    @property
    def state(self) -> CognitiveState:
        """Current cognitive state."""
        return self._current_state
    
    @property
    def flywheel(self) -> IntegrityFlywheel:
        """Access to the integrity flywheel."""
        return self._flywheel
    
    @property
    def meta_monitor(self) -> MetaCognitiveMonitor:
        """Access to the meta-cognitive monitor."""
        return self._meta_monitor
    
    @property
    def crystallizer(self) -> WisdomCrystallizationPipeline:
        """Access to the crystallization pipeline."""
        return self._crystallizer
    
    @property
    def cycle_count(self) -> int:
        """Total reasoning cycles completed."""
        return self._cycle_count


# =============================================================================
# AUTONOMOUS COGNITIVE ENGINE
# =============================================================================


class AutonomousCognitiveEngine:
    """
    The complete autonomous cognitive system.
    
    This is the peak masterpiece that unifies:
    - Streaming Genesis Orchestrator (brain)
    - Meta-Cognitive Monitor (self-awareness)
    - Wisdom Crystallization (learning)
    - Integrity Flywheel (improvement)
    - Event System (observability)
    
    The engine can run autonomously, continuously improving
    its reasoning capabilities through self-reflection.
    """
    
    def __init__(
        self,
        beam_width: int = 5,
        max_depth: int = 4,
        fail_closed: bool = True,
        wisdom_path: Optional[Path] = None,
    ):
        """
        Initialize the autonomous cognitive engine.
        
        Args:
            beam_width: Beam search width
            max_depth: Maximum thought depth
            fail_closed: Whether to fail on low SNR
            wisdom_path: Path for wisdom storage
        """
        # Core orchestrator
        self._orchestrator = StreamingGenesisOrchestrator(
            beam_width=beam_width,
            max_depth=max_depth,
            fail_closed=fail_closed,
        )
        
        # Wisdom repository
        self._wisdom_path = wisdom_path or Path("data/wisdom")
        self._wisdom_repo = WisdomRepository(storage_path=self._wisdom_path)
        self._orchestrator.wisdom_repo = self._wisdom_repo
        
        # Reasoning loop controller
        self._controller = ReasoningLoopController(
            self._orchestrator,
            self._wisdom_repo,
        )
        
        # Event bus
        self._event_bus = GenesisEventBus(keep_history=True, max_history=1000)
        
        # State
        self._started = False
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the cognitive engine."""
        if self._started:
            return
        
        logger.info("Autonomous Cognitive Engine starting...")
        
        # Load existing wisdom
        self._wisdom_repo.load()
        
        await self._controller._transition_state(CognitiveState.AWAKENING)
        
        self._started = True
        
        await self._controller._transition_state(CognitiveState.IDLE)
        
        logger.info(
            f"Cognitive Engine ready. "
            f"Wisdom entries: {len(self._wisdom_repo._wisdom)}"
        )
    
    async def stop(self) -> None:
        """Stop the cognitive engine."""
        if not self._started:
            return
        
        logger.info("Cognitive Engine shutting down...")
        
        await self._controller._transition_state(CognitiveState.SUSPENDED)
        
        # Save wisdom
        self._wisdom_repo.save()
        
        await self._controller._transition_state(CognitiveState.TERMINATED)
        
        self._started = False
        self._shutdown_event.set()
        
        logger.info("Cognitive Engine terminated.")
    
    async def reason(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GenesisResult, ReasoningCycleMetrics]:
        """
        Perform autonomous reasoning on a problem.
        
        Args:
            problem: The problem to reason about
            context: Optional additional context
            
        Returns:
            Tuple of (result, metrics)
        """
        if not self._started:
            await self.start()
        
        return await self._controller.reason(problem, context)
    
    async def reason_streaming(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[GenesisEvent]:
        """
        Stream reasoning events.
        
        Args:
            problem: The problem to reason about
            context: Optional additional context
            
        Yields:
            Genesis events as they occur
        """
        if not self._started:
            await self.start()
        
        async for event in self._orchestrator.process_streaming(problem, context):
            await self._event_bus.emit(event)
            yield event
    
    def add_observer(self, observer: MetaCognitiveObserver) -> None:
        """Add a meta-cognitive observer."""
        self._controller.add_observer(observer)
    
    def add_event_listener(
        self,
        listener: GenesisEventListener,
        event_types: Optional[List[GenesisEventType]] = None,
    ) -> None:
        """Add an event listener."""
        self._event_bus.add_listener(listener, event_types)
    
    @property
    def controller(self) -> ReasoningLoopController:
        """Access the reasoning loop controller."""
        return self._controller
    
    @property
    def flywheel_state(self) -> IntegrityFlywheelState:
        """Current flywheel state."""
        return self._controller.flywheel.state
    
    @property
    def recent_insights(self) -> List[MetaCognitiveInsight]:
        """Recent meta-cognitive insights."""
        return self._controller.meta_monitor.get_recent_insights()
    
    @property
    def wisdom_count(self) -> int:
        """Number of wisdom entries."""
        return len(self._wisdom_repo._wisdom)
    
    @property
    def cycle_count(self) -> int:
        """Total reasoning cycles."""
        return self._controller.cycle_count
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        return {
            "state": self._controller.state.name,
            "started": self._started,
            "cycles_completed": self._controller.cycle_count,
            "wisdom_entries": self.wisdom_count,
            "crystallizations": self._controller.crystallizer.crystallization_count,
            "flywheel": self._controller.flywheel.state.to_dict(),
            "recent_insights": [
                i.to_dict() for i in self.recent_insights[-5:]
            ],
            "pattern_frequencies": self._controller.meta_monitor.get_pattern_frequencies(),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def create_cognitive_engine(
    wisdom_path: Optional[Path] = None,
    **kwargs,
) -> AutonomousCognitiveEngine:
    """
    Create and start an autonomous cognitive engine.
    
    Args:
        wisdom_path: Path for wisdom storage
        **kwargs: Additional arguments for engine
        
    Returns:
        Started cognitive engine
    """
    engine = AutonomousCognitiveEngine(wisdom_path=wisdom_path, **kwargs)
    await engine.start()
    return engine


# =============================================================================
# DEMO
# =============================================================================


async def demo():
    """Demonstrate the autonomous cognitive engine."""
    print("=" * 70)
    print("AUTONOMOUS COGNITIVE ENGINE - DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Create engine
    engine = await create_cognitive_engine(
        beam_width=3,
        max_depth=3,
        fail_closed=False,  # Allow demo to run
    )
    
    problems = [
        "Design a secure distributed consensus protocol for IoT devices",
        "Optimize energy efficiency in neural network inference",
        "Create a privacy-preserving data sharing mechanism",
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{'─' * 70}")
        print(f"CYCLE {i}: {problem[:50]}...")
        print("─" * 70)
        
        result, metrics = await engine.reason(problem)
        
        print(f"\n  Quality: {metrics.quality.value}")
        print(f"  SNR: {metrics.final_snr:.2f}")
        print(f"  Ihsān: {metrics.ihsan_score:.2f}")
        print(f"  Thoughts: {metrics.thought_nodes_created} created, {metrics.thought_nodes_pruned} pruned")
        print(f"  Wisdom Seeds: {metrics.wisdom_seeds_used}")
        print(f"  Crystallized: {metrics.wisdom_crystallized}")
        print(f"  Duration: {metrics.duration_ms:.1f}ms")
    
    # Show final status
    print("\n" + "=" * 70)
    print("ENGINE STATUS")
    print("=" * 70)
    status = engine.get_status()
    print(f"  State: {status['state']}")
    print(f"  Cycles: {status['cycles_completed']}")
    print(f"  Wisdom: {status['wisdom_entries']}")
    print(f"  Crystallizations: {status['crystallizations']}")
    print(f"\n  Flywheel:")
    fw = status['flywheel']
    print(f"    Health: {fw['health']}")
    print(f"    Momentum: {fw['momentum']:.2f}")
    print(f"    Efficiency: {fw['efficiency']:.0%}")
    
    if status['recent_insights']:
        print(f"\n  Recent Insights:")
        for insight in status['recent_insights'][:3]:
            print(f"    - {insight['pattern_type']}: {insight['description'][:50]}...")
    
    await engine.stop()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo())
