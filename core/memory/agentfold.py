"""
BIZRA AgentFold Engine - Proactive Context Folding
═══════════════════════════════════════════════════════════════════════════════
Implements the Think-Fold-Act cycle from the Cognitive Permanence Roadmap.

The agent's operational cycle is: PERCEIVE → REASON → FOLD → ACT → UPDATE

At every time step t, the agent produces:
    (t_h, f_t, e_t, a_t)
Where:
    t_h = Thinking (Chain-of-Thought reasoning)
    f_t = Folding Directive (how to mutate context window)
    e_t = Explanation (natural language justification)
    a_t = Action (external tool execution)

This turns the agent into an active curator of its own mind.

Author: BIZRA Cognitive Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("bizra.agentfold")


# =============================================================================
# FOLDING TYPES AND DIRECTIVES
# =============================================================================


class FoldType(Enum):
    """Types of folding operations."""
    
    GRANULAR = auto()  # L1 → L2: Compress latest interaction
    DEEP = auto()       # L2 → L3: Fuse sequence into episode
    PRUNE = auto()      # Remove failed branch
    CHECKPOINT = auto() # Fibonacci-scheduled consolidation
    NONE = auto()       # No folding needed


@dataclass
class FoldingDirective:
    """
    Explicit instruction on how to mutate the context window.
    
    This is the f_t component of the (t_h, f_t, e_t, a_t) quadruplet.
    """
    
    fold_type: FoldType
    target_steps: List[int]  # Steps to fold (e.g., [5, 6, 7, 8, 9, 10])
    compression_ratio: float  # Target compression (0.0-1.0)
    preserve_keys: List[str]  # Keys that must survive folding
    justification: str  # Why this fold is needed
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entropy_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_type": self.fold_type.name,
            "target_steps": self.target_steps,
            "compression_ratio": self.compression_ratio,
            "preserve_keys": self.preserve_keys,
            "justification": self.justification,
            "timestamp": self.timestamp.isoformat(),
            "entropy_score": self.entropy_score,
        }


@dataclass
class CognitiveStep:
    """A single step in the agent's cognitive trace."""
    
    step_id: int
    thinking: str  # t_h: Chain of thought
    action: str    # a_t: Tool/action taken
    observation: str  # Raw observation from environment
    summary: Optional[str] = None  # Compressed summary (after folding)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = 0
    success: bool = True
    
    @property
    def is_folded(self) -> bool:
        return self.summary is not None
    
    def get_content(self) -> str:
        """Return summary if folded, else full observation."""
        return self.summary if self.is_folded else self.observation


@dataclass
class AgentFoldOutput:
    """
    The complete output of a Think-Fold-Act cycle.
    
    Corresponds to (t_h, f_t, e_t, a_t) quadruplet.
    """
    
    thinking: str           # t_h
    folding: FoldingDirective  # f_t
    explanation: str        # e_t
    action: str             # a_t
    
    # Step tracking
    step_id: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# FIBONACCI SCHEDULER
# =============================================================================


class FibonacciScheduler:
    """
    Non-linear scheduling for Deep Consolidation.
    
    Forces consolidation at Fibonacci steps: 13, 21, 34, 55, 89, 144...
    This spacing aligns with increasing complexity of long-horizon tasks.
    """
    
    # Pre-computed Fibonacci sequence (useful range for agent tasks)
    SEQUENCE = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
    
    # Start consolidation at step 13 (first "significant" checkpoint)
    MIN_STEP = 13
    
    def __init__(self, start_from: int = MIN_STEP):
        self.current_index = self._find_start_index(start_from)
        self.consolidation_history: List[int] = []
    
    def _find_start_index(self, start_from: int) -> int:
        """Find the index in SEQUENCE >= start_from."""
        for i, val in enumerate(self.SEQUENCE):
            if val >= start_from:
                return i
        return len(self.SEQUENCE) - 1
    
    def is_consolidation_step(self, step: int) -> bool:
        """Check if this step triggers Fibonacci consolidation."""
        return step in self.SEQUENCE and step >= self.MIN_STEP
    
    def get_next_consolidation_step(self, current_step: int) -> int:
        """Get the next scheduled consolidation step."""
        for val in self.SEQUENCE:
            if val > current_step:
                return val
        # Beyond our sequence, extrapolate
        return self._extrapolate_fibonacci(current_step)
    
    def _extrapolate_fibonacci(self, beyond: int) -> int:
        """Generate next Fibonacci number beyond our sequence."""
        a, b = self.SEQUENCE[-2], self.SEQUENCE[-1]
        while b <= beyond:
            a, b = b, a + b
        return b
    
    def record_consolidation(self, step: int) -> None:
        """Record that consolidation happened at this step."""
        self.consolidation_history.append(step)


# =============================================================================
# ENTROPY CALCULATOR (Memory Poisoning Defense)
# =============================================================================


class EntropyValidator:
    """
    Shannon Entropy validation for memory quality.
    
    Filters out low-quality, repetitive, or empty hallucinations.
    Minimum entropy threshold: 3.5 bits.
    """
    
    MIN_ENTROPY = 3.5
    
    @staticmethod
    def calculate_shannon_entropy(text: str) -> float:
        """
        Calculate Shannon entropy of text in bits.
        
        Higher entropy = more information content.
        Lower entropy = repetitive/low-quality content.
        """
        if not text:
            return 0.0
        
        # Character frequency distribution
        freq: Dict[str, int] = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        
        length = len(text)
        entropy = 0.0
        
        for count in freq.values():
            if count > 0:
                probability = count / length
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    @classmethod
    def validate(cls, content: str) -> Tuple[bool, float]:
        """
        Validate content meets minimum entropy threshold.
        
        Returns: (is_valid, entropy_score)
        """
        entropy = cls.calculate_shannon_entropy(content)
        return (entropy >= cls.MIN_ENTROPY, entropy)


# =============================================================================
# FOLDING STRATEGY ENGINE
# =============================================================================


class FoldingStrategyEngine:
    """
    Manages folding strategies based on configuration.
    
    Loaded from config/expertise.yaml folding_strategy section.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/expertise.yaml")
        self.config = self._load_config()
        self.fibonacci = FibonacciScheduler()
        self.entropy = EntropyValidator()
        
        # Tracking
        self.steps_since_last_fold = 0
        self.current_token_count = 0
        self.retry_count = 0
    
    def _load_config(self) -> Dict[str, Any]:
        """Load folding strategy from expertise.yaml."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
                return full_config.get("folding_strategy", {})
        except FileNotFoundError:
            logger.warning("Expertise config not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default folding configuration."""
        return {
            "granular": {
                "triggers": [
                    {"condition": "tool_success AND steps_since_last_fold >= 3"},
                    {"condition": "token_count > 3000"},
                ],
                "compression_target": 0.618,
                "min_entropy": 3.5,
            },
            "deep": {
                "triggers": [
                    {"condition": "subtask_complete"},
                    {"condition": "tool_failure AND retry_count >= 2"},
                    {"condition": "fibonacci_step_reached"},
                ],
                "fibonacci_schedule": {"enabled": True},
            },
            "safety": {
                "max_compression_ratio": 0.90,
                "preserve_safety_warnings": True,
                "preserve_error_lessons": True,
            },
        }
    
    def evaluate_fold_need(
        self,
        current_step: int,
        context: Dict[str, Any],
    ) -> Optional[FoldingDirective]:
        """
        Evaluate if folding is needed based on current context.
        
        Args:
            current_step: Current step number
            context: Dictionary with keys like 'tool_success', 'token_count', etc.
        
        Returns:
            FoldingDirective if folding needed, None otherwise
        """
        # Check Fibonacci schedule first (highest priority)
        if self.fibonacci.is_consolidation_step(current_step):
            return self._create_deep_fold(
                current_step,
                justification=f"Fibonacci checkpoint at step {current_step}",
            )
        
        # Check for failure recovery
        if context.get("tool_failure") and self.retry_count >= 2:
            return self._create_prune_fold(
                current_step,
                justification="Pruning failed branch after 2+ retries",
            )
        
        # Check for subtask completion
        if context.get("subtask_complete"):
            return self._create_deep_fold(
                current_step,
                justification="Subtask completed, consolidating progress",
            )
        
        # Check for granular condensation
        if context.get("tool_success") and self.steps_since_last_fold >= 3:
            return self._create_granular_fold(
                current_step,
                justification="Compressing after 3+ successful steps",
            )
        
        # Check token overflow
        token_count = context.get("token_count", 0)
        if token_count > 3000:
            return self._create_granular_fold(
                current_step,
                justification=f"Token count ({token_count}) exceeds threshold",
            )
        
        return None
    
    def _create_granular_fold(
        self, current_step: int, justification: str
    ) -> FoldingDirective:
        """Create a granular (L1→L2) folding directive."""
        return FoldingDirective(
            fold_type=FoldType.GRANULAR,
            target_steps=[current_step],
            compression_ratio=self.config.get("granular", {}).get(
                "compression_target", 0.618
            ),
            preserve_keys=["action", "result", "success"],
            justification=justification,
        )
    
    def _create_deep_fold(
        self, current_step: int, justification: str
    ) -> FoldingDirective:
        """Create a deep (L2→L3) folding directive."""
        # Deep fold targets the last N steps since consolidation
        last_consolidation = (
            self.fibonacci.consolidation_history[-1]
            if self.fibonacci.consolidation_history
            else 0
        )
        target_range = list(range(last_consolidation + 1, current_step + 1))
        
        return FoldingDirective(
            fold_type=FoldType.DEEP,
            target_steps=target_range,
            compression_ratio=0.80,  # More aggressive for deep folds
            preserve_keys=["episode_summary", "lessons_learned", "key_decisions"],
            justification=justification,
        )
    
    def _create_prune_fold(
        self, current_step: int, justification: str
    ) -> FoldingDirective:
        """Create a prune (failed branch) folding directive."""
        return FoldingDirective(
            fold_type=FoldType.PRUNE,
            target_steps=[current_step - 1, current_step],  # Prune last 2 steps
            compression_ratio=0.95,  # Heavy compression for failures
            preserve_keys=["error_type", "lesson_learned"],
            justification=justification,
        )
    
    def record_step(self, success: bool) -> None:
        """Record a step for tracking."""
        self.steps_since_last_fold += 1
        if not success:
            self.retry_count += 1
        else:
            self.retry_count = 0
    
    def record_fold(self, step: int, fold_type: FoldType) -> None:
        """Record that folding happened."""
        self.steps_since_last_fold = 0
        if fold_type in (FoldType.DEEP, FoldType.CHECKPOINT):
            self.fibonacci.record_consolidation(step)


# =============================================================================
# AGENTFOLD ORCHESTRATOR
# =============================================================================


class AgentFoldOrchestrator:
    """
    Main orchestrator for the Think-Fold-Act cycle.
    
    Coordinates between:
    - L1 (Immediate perception)
    - L2 (Working memory / granular summaries)
    - L3 (Episodic memory / deep consolidation)
    - AgentFold strategy engine
    """
    
    def __init__(
        self,
        strategy_engine: Optional[FoldingStrategyEngine] = None,
        compressor: Optional[Callable[[str], str]] = None,
    ):
        self.strategy = strategy_engine or FoldingStrategyEngine()
        self.compressor = compressor or self._default_compressor
        
        # Cognitive trace
        self.trace: List[CognitiveStep] = []
        self.current_step = 0
        
        # Layer stores
        self.l2_summaries: List[str] = []
        self.l3_episodes: List[Dict[str, Any]] = []
        
        # Entropy validator
        self.entropy = EntropyValidator()
    
    def _default_compressor(self, content: str) -> str:
        """Default compression: truncate to 100 chars with hash suffix."""
        if len(content) <= 100:
            return content
        
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"{content[:90]}... [hash:{content_hash}]"
    
    def process_step(
        self,
        thinking: str,
        action: str,
        observation: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentFoldOutput:
        """
        Process a complete Think-Fold-Act cycle step.
        
        Args:
            thinking: The agent's Chain-of-Thought reasoning
            action: The tool/action executed
            observation: Raw observation from environment
            context: Additional context (tool_success, token_count, etc.)
        
        Returns:
            AgentFoldOutput with the (t_h, f_t, e_t, a_t) quadruplet
        """
        self.current_step += 1
        context = context or {}
        
        # Create cognitive step
        step = CognitiveStep(
            step_id=self.current_step,
            thinking=thinking,
            action=action,
            observation=observation,
            token_count=len(observation.split()),
            success=context.get("tool_success", True),
        )
        
        # Validate entropy before storing
        is_valid, entropy = self.entropy.validate(observation)
        if not is_valid:
            logger.warning(
                f"Step {self.current_step}: Low entropy ({entropy:.2f}), flagging for review"
            )
        
        # Add to trace
        self.trace.append(step)
        
        # Evaluate if folding is needed
        context["token_count"] = sum(s.token_count for s in self.trace if not s.is_folded)
        folding_directive = self.strategy.evaluate_fold_need(self.current_step, context)
        
        # Execute folding if needed
        explanation = "No folding required"
        if folding_directive:
            explanation = self._execute_fold(folding_directive)
            folding_directive.entropy_score = entropy
        else:
            folding_directive = FoldingDirective(
                fold_type=FoldType.NONE,
                target_steps=[],
                compression_ratio=0.0,
                preserve_keys=[],
                justification="No folding needed",
            )
        
        # Record step for tracking
        self.strategy.record_step(step.success)
        
        return AgentFoldOutput(
            thinking=thinking,
            folding=folding_directive,
            explanation=explanation,
            action=action,
            step_id=self.current_step,
        )
    
    def _execute_fold(self, directive: FoldingDirective) -> str:
        """Execute a folding directive."""
        if directive.fold_type == FoldType.GRANULAR:
            return self._granular_fold(directive)
        elif directive.fold_type == FoldType.DEEP:
            return self._deep_fold(directive)
        elif directive.fold_type == FoldType.PRUNE:
            return self._prune_fold(directive)
        else:
            return "Unknown fold type"
    
    def _granular_fold(self, directive: FoldingDirective) -> str:
        """Execute granular condensation (L1 → L2)."""
        for step_id in directive.target_steps:
            step = self._find_step(step_id)
            if step and not step.is_folded:
                step.summary = self.compressor(step.observation)
                self.l2_summaries.append(step.summary)
        
        self.strategy.record_fold(self.current_step, FoldType.GRANULAR)
        return f"Compressed {len(directive.target_steps)} steps to L2"
    
    def _deep_fold(self, directive: FoldingDirective) -> str:
        """Execute deep consolidation (L2 → L3)."""
        # Gather all L2 summaries in range
        summaries = []
        for step_id in directive.target_steps:
            step = self._find_step(step_id)
            if step:
                summaries.append(step.get_content())
        
        # Create episode
        episode = {
            "episode_id": f"EP_{self.current_step}",
            "step_range": f"{min(directive.target_steps)}-{max(directive.target_steps)}",
            "summary": self.compressor(" | ".join(summaries)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.l3_episodes.append(episode)
        
        # Clear folded steps from active trace
        for step_id in directive.target_steps:
            step = self._find_step(step_id)
            if step:
                step.summary = f"[Consolidated to {episode['episode_id']}]"
        
        self.strategy.record_fold(self.current_step, FoldType.DEEP)
        return f"Consolidated {len(directive.target_steps)} steps into {episode['episode_id']}"
    
    def _prune_fold(self, directive: FoldingDirective) -> str:
        """Execute prune (failed branch removal)."""
        lessons = []
        for step_id in directive.target_steps:
            step = self._find_step(step_id)
            if step and not step.success:
                # Extract lesson from failure
                lesson = f"Step {step_id} failed: {step.action[:50]}"
                lessons.append(lesson)
                step.summary = f"[PRUNED: {lesson}]"
        
        self.strategy.record_fold(self.current_step, FoldType.PRUNE)
        return f"Pruned {len(directive.target_steps)} failed steps, extracted {len(lessons)} lessons"
    
    def _find_step(self, step_id: int) -> Optional[CognitiveStep]:
        """Find a step by ID."""
        for step in self.trace:
            if step.step_id == step_id:
                return step
        return None
    
    def get_active_context(self) -> List[str]:
        """Get the current active context (unfolded or summarized)."""
        return [step.get_content() for step in self.trace]
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about current context state."""
        total_steps = len(self.trace)
        folded_steps = sum(1 for s in self.trace if s.is_folded)
        total_tokens = sum(s.token_count for s in self.trace)
        active_tokens = sum(s.token_count for s in self.trace if not s.is_folded)
        
        return {
            "total_steps": total_steps,
            "folded_steps": folded_steps,
            "active_steps": total_steps - folded_steps,
            "total_tokens": total_tokens,
            "active_tokens": active_tokens,
            "compression_ratio": 1 - (active_tokens / max(1, total_tokens)),
            "l2_summaries": len(self.l2_summaries),
            "l3_episodes": len(self.l3_episodes),
            "next_fibonacci_checkpoint": self.strategy.fibonacci.get_next_consolidation_step(
                self.current_step
            ),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_agentfold(config_path: Optional[Path] = None) -> AgentFoldOrchestrator:
    """Create an AgentFold orchestrator with default configuration."""
    strategy = FoldingStrategyEngine(config_path)
    return AgentFoldOrchestrator(strategy_engine=strategy)


# =============================================================================
# MAIN - FOR TESTING
# =============================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("BIZRA AgentFold Engine - Test Run")
    print("=" * 80)
    
    # Create orchestrator
    orchestrator = create_agentfold()
    
    # Simulate some steps
    for i in range(15):
        output = orchestrator.process_step(
            thinking=f"Thinking about step {i+1}...",
            action=f"tool_call_{i+1}",
            observation=f"Observation from tool call {i+1}. " * 10,
            context={"tool_success": i != 5, "subtask_complete": i == 10},
        )
        
        if output.folding.fold_type != FoldType.NONE:
            print(f"\n[Step {output.step_id}] FOLD: {output.folding.fold_type.name}")
            print(f"  Justification: {output.folding.justification}")
            print(f"  Result: {output.explanation}")
    
    print("\n" + "=" * 80)
    print("Context Statistics:")
    stats = orchestrator.get_context_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
