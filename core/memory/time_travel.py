"""
BIZRA Time Travel Engine - L3 Checkpoint Recovery
═══════════════════════════════════════════════════════════════════════════════
Enables rollback to L3 episodic memory checkpoints when tasks fail.

The Time Travel Engine provides resilience through checkpoint-based recovery:

1. Checkpoint Creation: At Deep Consolidation points (Fibonacci steps),
   the system saves a complete state snapshot.

2. Failure Detection: When a task fails catastrophically, the engine
   identifies the last stable checkpoint.

3. State Restoration: The cognitive workspace is restored to the checkpoint
   state, allowing the agent to attempt a different strategy.

4. Strategy Pivoting: Lessons from the failed branch are preserved in the
   checkpoint metadata to prevent repeating the same mistake.

This mechanism prevents the agent from restarting from scratch on failures.

Author: BIZRA Cognitive Team
Version: 1.0.0
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("bizra.timetravel")


# =============================================================================
# CHECKPOINT TYPES
# =============================================================================


class CheckpointType(Enum):
    """Types of checkpoints."""
    
    FIBONACCI = auto()      # Scheduled at Fibonacci steps
    SUBTASK = auto()        # End of subtask
    MANUAL = auto()         # Explicitly requested
    PRE_RISK = auto()       # Before high-risk operation
    RECOVERY = auto()       # Created during recovery


class RecoveryReason(Enum):
    """Reasons for triggering recovery."""
    
    TASK_FAILURE = auto()
    RETRY_EXHAUSTED = auto()
    HALLUCINATION_DETECTED = auto()
    INFINITE_LOOP = auto()
    CONTEXT_OVERFLOW = auto()
    USER_REQUESTED = auto()


@dataclass
class CognitiveState:
    """
    Complete cognitive state at a point in time.
    
    This is what gets saved at checkpoints and restored during recovery.
    """
    
    # Memory layer states
    l1_buffer: List[Dict[str, Any]]
    l2_summaries: List[str]
    l3_episodes: List[Dict[str, Any]]
    
    # AgentFold state
    current_step: int
    steps_since_fold: int
    active_context: List[str]
    
    # Task state
    goal: str
    completed_subtasks: List[str]
    pending_subtasks: List[str]
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "l1_buffer": self.l1_buffer,
            "l2_summaries": self.l2_summaries,
            "l3_episodes": self.l3_episodes,
            "current_step": self.current_step,
            "steps_since_fold": self.steps_since_fold,
            "active_context": self.active_context,
            "goal": self.goal,
            "completed_subtasks": self.completed_subtasks,
            "pending_subtasks": self.pending_subtasks,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveState":
        return cls(
            l1_buffer=data.get("l1_buffer", []),
            l2_summaries=data.get("l2_summaries", []),
            l3_episodes=data.get("l3_episodes", []),
            current_step=data.get("current_step", 0),
            steps_since_fold=data.get("steps_since_fold", 0),
            active_context=data.get("active_context", []),
            goal=data.get("goal", ""),
            completed_subtasks=data.get("completed_subtasks", []),
            pending_subtasks=data.get("pending_subtasks", []),
            timestamp=datetime.fromisoformat(data["timestamp"]) 
                if "timestamp" in data else datetime.now(timezone.utc),
        )
    
    def compute_hash(self) -> str:
        """Compute hash of state for integrity verification."""
        serialized = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass
class Checkpoint:
    """
    A saved checkpoint of cognitive state.
    """
    
    checkpoint_id: str
    checkpoint_type: CheckpointType
    state: CognitiveState
    state_hash: str
    
    # Context
    step_number: int
    episode_id: Optional[str] = None
    
    # Lessons from this branch (populated if created during recovery)
    lessons_learned: List[str] = field(default_factory=list)
    failed_strategies: List[str] = field(default_factory=list)
    
    # Chain linking
    previous_checkpoint_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_type": self.checkpoint_type.name,
            "state": self.state.to_dict(),
            "state_hash": self.state_hash,
            "step_number": self.step_number,
            "episode_id": self.episode_id,
            "lessons_learned": self.lessons_learned,
            "failed_strategies": self.failed_strategies,
            "previous_checkpoint_id": self.previous_checkpoint_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    
    success: bool
    restored_checkpoint_id: str
    restored_step: int
    lessons_from_failure: List[str]
    new_checkpoint_id: Optional[str] = None
    message: str = ""


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================


class CheckpointManager:
    """
    Manages checkpoint creation, storage, and retrieval.
    """
    
    # Maximum checkpoints to retain (prevent unbounded growth)
    MAX_CHECKPOINTS = 50
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/checkpoints")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory checkpoint index
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.checkpoint_order: List[str] = []
        
        # Load existing checkpoints
        self._load_checkpoints()
    
    def _load_checkpoints(self) -> None:
        """Load checkpoints from storage."""
        for cp_file in self.storage_path.glob("*.json"):
            try:
                with open(cp_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                checkpoint = Checkpoint(
                    checkpoint_id=data["checkpoint_id"],
                    checkpoint_type=CheckpointType[data["checkpoint_type"]],
                    state=CognitiveState.from_dict(data["state"]),
                    state_hash=data["state_hash"],
                    step_number=data["step_number"],
                    episode_id=data.get("episode_id"),
                    lessons_learned=data.get("lessons_learned", []),
                    failed_strategies=data.get("failed_strategies", []),
                    previous_checkpoint_id=data.get("previous_checkpoint_id"),
                    created_at=datetime.fromisoformat(data["created_at"]),
                )
                
                self.checkpoints[checkpoint.checkpoint_id] = checkpoint
                self.checkpoint_order.append(checkpoint.checkpoint_id)
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint {cp_file}: {e}")
        
        # Sort by step number
        self.checkpoint_order.sort(
            key=lambda cid: self.checkpoints[cid].step_number
        )
        
        logger.info(f"Loaded {len(self.checkpoints)} checkpoints")
    
    def create_checkpoint(
        self,
        state: CognitiveState,
        checkpoint_type: CheckpointType,
        step_number: int,
        episode_id: Optional[str] = None,
    ) -> Checkpoint:
        """Create a new checkpoint."""
        # Generate ID
        checkpoint_id = f"cp_{step_number}_{hashlib.sha256(str(datetime.now(timezone.utc)).encode()).hexdigest()[:8]}"
        
        # Get previous checkpoint
        previous_id = self.checkpoint_order[-1] if self.checkpoint_order else None
        
        # Create checkpoint
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            state=copy.deepcopy(state),  # Deep copy to prevent mutation
            state_hash=state.compute_hash(),
            step_number=step_number,
            episode_id=episode_id,
            previous_checkpoint_id=previous_id,
        )
        
        # Store
        self.checkpoints[checkpoint_id] = checkpoint
        self.checkpoint_order.append(checkpoint_id)
        
        # Persist
        self._save_checkpoint(checkpoint)
        
        # Prune old checkpoints if needed
        self._prune_old_checkpoints()
        
        logger.info(
            f"Created checkpoint {checkpoint_id} at step {step_number} "
            f"(type: {checkpoint_type.name})"
        )
        
        return checkpoint
    
    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to storage."""
        cp_file = self.storage_path / f"{checkpoint.checkpoint_id}.json"
        with open(cp_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)
    
    def _prune_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond MAX_CHECKPOINTS."""
        while len(self.checkpoint_order) > self.MAX_CHECKPOINTS:
            oldest_id = self.checkpoint_order.pop(0)
            
            # Remove from memory
            del self.checkpoints[oldest_id]
            
            # Remove from storage
            cp_file = self.storage_path / f"{oldest_id}.json"
            if cp_file.exists():
                cp_file.unlink()
            
            logger.debug(f"Pruned old checkpoint: {oldest_id}")
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a specific checkpoint."""
        return self.checkpoints.get(checkpoint_id)
    
    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the most recent checkpoint."""
        if not self.checkpoint_order:
            return None
        return self.checkpoints.get(self.checkpoint_order[-1])
    
    def get_checkpoint_before_step(self, step: int) -> Optional[Checkpoint]:
        """Get the most recent checkpoint before a given step."""
        for checkpoint_id in reversed(self.checkpoint_order):
            checkpoint = self.checkpoints.get(checkpoint_id)
            if checkpoint and checkpoint.step_number < step:
                return checkpoint
        return None
    
    def get_stable_checkpoint(self) -> Optional[Checkpoint]:
        """
        Get the most recent 'stable' checkpoint.
        
        Stable = FIBONACCI or SUBTASK type (not recovery or pre-risk)
        """
        stable_types = {CheckpointType.FIBONACCI, CheckpointType.SUBTASK}
        
        for checkpoint_id in reversed(self.checkpoint_order):
            checkpoint = self.checkpoints.get(checkpoint_id)
            if checkpoint and checkpoint.checkpoint_type in stable_types:
                return checkpoint
        
        return None
    
    def add_lessons_to_checkpoint(
        self,
        checkpoint_id: str,
        lessons: List[str],
        failed_strategies: Optional[List[str]] = None,
    ) -> None:
        """Add lessons learned to a checkpoint (after recovery)."""
        checkpoint = self.checkpoints.get(checkpoint_id)
        if checkpoint:
            checkpoint.lessons_learned.extend(lessons)
            if failed_strategies:
                checkpoint.failed_strategies.extend(failed_strategies)
            self._save_checkpoint(checkpoint)


# =============================================================================
# TIME TRAVEL ENGINE
# =============================================================================


class TimeTravelEngine:
    """
    The Time Travel Engine for cognitive state recovery.
    
    Enables the agent to:
    1. Save checkpoints at key moments
    2. Roll back to previous states on failure
    3. Learn from failures to avoid repeating mistakes
    """
    
    # Fibonacci sequence for automatic checkpoints
    FIBONACCI_STEPS = [13, 21, 34, 55, 89, 144, 233, 377]
    
    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        
        # Recovery statistics
        self.recovery_count = 0
        self.successful_recoveries = 0
        self.lessons_accumulated: List[str] = []
    
    def should_checkpoint(
        self,
        step: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[CheckpointType]]:
        """
        Determine if a checkpoint should be created at this step.
        
        Returns: (should_checkpoint, checkpoint_type)
        """
        context = context or {}
        
        # Fibonacci checkpoint
        if step in self.FIBONACCI_STEPS:
            return (True, CheckpointType.FIBONACCI)
        
        # Subtask completion
        if context.get("subtask_complete"):
            return (True, CheckpointType.SUBTASK)
        
        # Before high-risk operation
        if context.get("high_risk_operation"):
            return (True, CheckpointType.PRE_RISK)
        
        return (False, None)
    
    def create_checkpoint(
        self,
        state: CognitiveState,
        step: int,
        checkpoint_type: Optional[CheckpointType] = None,
        episode_id: Optional[str] = None,
    ) -> Checkpoint:
        """Create a checkpoint of the current state."""
        if checkpoint_type is None:
            checkpoint_type = CheckpointType.MANUAL
        
        return self.checkpoint_manager.create_checkpoint(
            state=state,
            checkpoint_type=checkpoint_type,
            step_number=step,
            episode_id=episode_id,
        )
    
    def recover(
        self,
        reason: RecoveryReason,
        current_step: int,
        failed_strategy: Optional[str] = None,
        lesson: Optional[str] = None,
    ) -> RecoveryResult:
        """
        Recover to the last stable checkpoint.
        
        Args:
            reason: Why recovery is needed
            current_step: Current step number
            failed_strategy: Description of what failed
            lesson: Lesson learned from failure
            
        Returns:
            RecoveryResult with restored state info
        """
        self.recovery_count += 1
        
        logger.warning(
            f"Recovery triggered at step {current_step}: {reason.name}"
        )
        
        # Find the best checkpoint to restore
        checkpoint = self._select_recovery_checkpoint(current_step, reason)
        
        if checkpoint is None:
            return RecoveryResult(
                success=False,
                restored_checkpoint_id="",
                restored_step=0,
                lessons_from_failure=[],
                message="No suitable checkpoint found for recovery",
            )
        
        # Collect lessons
        lessons = []
        if lesson:
            lessons.append(lesson)
        lessons.append(f"Failed at step {current_step}: {reason.name}")
        
        failed_strategies = []
        if failed_strategy:
            failed_strategies.append(failed_strategy)
        
        # Add lessons to the checkpoint
        self.checkpoint_manager.add_lessons_to_checkpoint(
            checkpoint.checkpoint_id,
            lessons,
            failed_strategies,
        )
        
        # Create recovery checkpoint at restored state
        recovery_checkpoint = self.checkpoint_manager.create_checkpoint(
            state=checkpoint.state,
            checkpoint_type=CheckpointType.RECOVERY,
            step_number=checkpoint.step_number,
            episode_id=checkpoint.episode_id,
        )
        
        # Update statistics
        self.successful_recoveries += 1
        self.lessons_accumulated.extend(lessons)
        
        logger.info(
            f"Recovery successful: restored to checkpoint {checkpoint.checkpoint_id} "
            f"at step {checkpoint.step_number}"
        )
        
        return RecoveryResult(
            success=True,
            restored_checkpoint_id=checkpoint.checkpoint_id,
            restored_step=checkpoint.step_number,
            lessons_from_failure=checkpoint.lessons_learned + lessons,
            new_checkpoint_id=recovery_checkpoint.checkpoint_id,
            message=f"Restored to step {checkpoint.step_number}, "
                    f"skipping {current_step - checkpoint.step_number} steps",
        )
    
    def _select_recovery_checkpoint(
        self,
        current_step: int,
        reason: RecoveryReason,
    ) -> Optional[Checkpoint]:
        """Select the best checkpoint for recovery."""
        # For most reasons, use the last stable checkpoint
        if reason in {
            RecoveryReason.TASK_FAILURE,
            RecoveryReason.RETRY_EXHAUSTED,
            RecoveryReason.USER_REQUESTED,
        }:
            return self.checkpoint_manager.get_stable_checkpoint()
        
        # For hallucination or loop, go back further
        if reason in {
            RecoveryReason.HALLUCINATION_DETECTED,
            RecoveryReason.INFINITE_LOOP,
        }:
            # Get checkpoint at least 5 steps back
            return self.checkpoint_manager.get_checkpoint_before_step(
                current_step - 5
            )
        
        # For context overflow, get earliest reasonable checkpoint
        if reason == RecoveryReason.CONTEXT_OVERFLOW:
            # Go back to a Fibonacci checkpoint
            for cp_id in reversed(self.checkpoint_manager.checkpoint_order):
                cp = self.checkpoint_manager.checkpoints.get(cp_id)
                if cp and cp.checkpoint_type == CheckpointType.FIBONACCI:
                    return cp
        
        return self.checkpoint_manager.get_stable_checkpoint()
    
    def get_lessons_for_goal(self, goal: str) -> List[str]:
        """
        Get accumulated lessons relevant to a goal.
        
        Checks all checkpoints for lessons that might apply.
        """
        all_lessons = []
        
        for checkpoint in self.checkpoint_manager.checkpoints.values():
            # Check if goal matches checkpoint's state
            if goal.lower() in checkpoint.state.goal.lower():
                all_lessons.extend(checkpoint.lessons_learned)
        
        # Also include recently accumulated lessons
        all_lessons.extend(self.lessons_accumulated[-10:])
        
        return list(set(all_lessons))  # Deduplicate
    
    def get_failed_strategies_for_goal(self, goal: str) -> List[str]:
        """Get strategies that have failed for similar goals."""
        failed = []
        
        for checkpoint in self.checkpoint_manager.checkpoints.values():
            if goal.lower() in checkpoint.state.goal.lower():
                failed.extend(checkpoint.failed_strategies)
        
        return list(set(failed))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get time travel statistics."""
        return {
            "total_checkpoints": len(self.checkpoint_manager.checkpoints),
            "recovery_count": self.recovery_count,
            "successful_recoveries": self.successful_recoveries,
            "recovery_success_rate": (
                self.successful_recoveries / self.recovery_count
                if self.recovery_count > 0 else 1.0
            ),
            "lessons_accumulated": len(self.lessons_accumulated),
        }


# =============================================================================
# INTEGRATION WITH AGENTFOLD
# =============================================================================


class TimeTravelAgentFoldIntegration:
    """
    Integrates Time Travel with AgentFold orchestrator.
    
    Provides automatic checkpoint creation at Fibonacci steps
    and recovery on folding failures.
    """
    
    def __init__(
        self,
        time_travel: Optional[TimeTravelEngine] = None,
    ):
        self.time_travel = time_travel or TimeTravelEngine()
    
    def on_step_complete(
        self,
        step: int,
        state: CognitiveState,
        context: Dict[str, Any],
    ) -> Optional[Checkpoint]:
        """
        Called after each AgentFold step.
        
        Creates checkpoint if appropriate.
        """
        should_cp, cp_type = self.time_travel.should_checkpoint(step, context)
        
        if should_cp and cp_type:
            return self.time_travel.create_checkpoint(
                state=state,
                step=step,
                checkpoint_type=cp_type,
            )
        
        return None
    
    def on_fold_failure(
        self,
        step: int,
        error: str,
    ) -> RecoveryResult:
        """
        Called when a folding operation fails.
        
        Triggers recovery to last stable checkpoint.
        """
        return self.time_travel.recover(
            reason=RecoveryReason.TASK_FAILURE,
            current_step=step,
            failed_strategy=f"Folding at step {step}",
            lesson=f"Folding failed: {error}",
        )
    
    def on_context_overflow(
        self,
        step: int,
        token_count: int,
    ) -> RecoveryResult:
        """
        Called when context window overflows.
        
        Triggers recovery with aggressive consolidation.
        """
        return self.time_travel.recover(
            reason=RecoveryReason.CONTEXT_OVERFLOW,
            current_step=step,
            lesson=f"Context overflow at {token_count} tokens",
        )


# =============================================================================
# MAIN - FOR TESTING
# =============================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("BIZRA Time Travel Engine - Test Run")
    print("=" * 80)
    
    # Create engine
    engine = TimeTravelEngine()
    
    # Simulate creating checkpoints
    for step in [13, 21, 25, 34]:
        state = CognitiveState(
            l1_buffer=[{"item": f"perception_{step}"}],
            l2_summaries=[f"summary_{step}"],
            l3_episodes=[{"id": f"episode_{step}"}],
            current_step=step,
            steps_since_fold=3,
            active_context=[f"context_{step}"],
            goal="Complete the complex task",
            completed_subtasks=[f"subtask_{i}" for i in range(step // 10)],
            pending_subtasks=["final_subtask"],
        )
        
        should_cp, cp_type = engine.should_checkpoint(step)
        if should_cp:
            cp = engine.create_checkpoint(state, step, cp_type)
            print(f"  Created checkpoint at step {step}: {cp.checkpoint_id}")
    
    # Simulate failure and recovery
    print("\n" + "-" * 40)
    print("Simulating failure at step 40...")
    
    result = engine.recover(
        reason=RecoveryReason.TASK_FAILURE,
        current_step=40,
        failed_strategy="Attempted aggressive parsing",
        lesson="Parsing failed on malformed input",
    )
    
    print(f"  Recovery success: {result.success}")
    print(f"  Restored to step: {result.restored_step}")
    print(f"  Lessons: {result.lessons_from_failure}")
    
    print("\n" + "=" * 80)
    print("Statistics:")
    for key, value in engine.get_stats().items():
        print(f"  {key}: {value}")
