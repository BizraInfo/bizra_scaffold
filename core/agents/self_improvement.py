"""
BIZRA Self-Improvement Agent - Autonomous Knowledge Evolution
═══════════════════════════════════════════════════════════════════════════════
Diffs pre/post codebase states and updates expertise.yaml automatically.

The Self-Improvement Agent embodies BIZRA's Ihsān principle by continuously
refining the system's knowledge based on actual execution experience.

Key Capabilities:

1. Codebase Diffing: Compares pre/post task states to identify changes
2. Pattern Extraction: Identifies successful patterns from diffs
3. Expertise Update: Writes new knowledge to expertise.yaml
4. Schema Evolution: Proposes new schemas based on observed patterns
5. Invariant Strengthening: Tightens invariants based on failures

This agent runs after every successful task completion and after recoveries,
ensuring the system continuously learns from experience.

Author: BIZRA Cognitive Team
Version: 1.0.0
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

logger = logging.getLogger("bizra.self_improvement")


# =============================================================================
# IMPROVEMENT TYPES
# =============================================================================


class ImprovementType(Enum):
    """Types of self-improvement actions."""
    
    NEW_PATTERN = auto()        # Discovered a new successful pattern
    INVARIANT_UPDATE = auto()   # Tightening or loosening invariants
    SCHEMA_ADDITION = auto()    # Adding new schemas
    STRATEGY_EVOLVED = auto()   # Evolved existing strategy
    FAILURE_LEARNED = auto()    # Learned from failure


class PatternCategory(Enum):
    """Categories of discovered patterns."""
    
    CODE_STRUCTURE = auto()     # Code organization patterns
    ERROR_HANDLING = auto()     # Error handling patterns
    NAMING = auto()             # Naming conventions
    ARCHITECTURE = auto()       # Architecture patterns
    WORKFLOW = auto()           # Workflow patterns
    TESTING = auto()            # Testing patterns


@dataclass
class CodeChange:
    """A single code change extracted from diff."""
    
    file_path: str
    change_type: str  # "add", "modify", "delete"
    lines_added: int
    lines_removed: int
    content_sample: str  # First 200 chars
    patterns_detected: List[str] = field(default_factory=list)


@dataclass
class DiscoveredPattern:
    """A pattern discovered from code changes."""
    
    pattern_id: str
    category: PatternCategory
    description: str
    example_code: str
    confidence: float  # 0.0 to 1.0
    occurrences: int
    source_files: List[str] = field(default_factory=list)


@dataclass
class ExpertiseUpdate:
    """An update to be applied to expertise.yaml."""
    
    update_type: ImprovementType
    section: str  # Section in expertise.yaml
    key: str
    value: Any
    justification: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# CODEBASE DIFFER
# =============================================================================


class CodebaseDiffer:
    """
    Diffs two codebase states to identify changes.
    """
    
    # File patterns to analyze
    ANALYZABLE_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".toml", ".md"}
    
    # Patterns to ignore
    IGNORE_PATTERNS = {
        "__pycache__",
        ".git",
        "node_modules",
        ".env",
        "venv",
        "*.pyc",
        "evidence/packs",
    }
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
    
    def snapshot_state(self) -> Dict[str, str]:
        """
        Create a snapshot of the current codebase state.
        
        Returns: dict mapping file paths to content hashes
        """
        snapshot = {}
        
        for file_path in self._iter_files():
            try:
                content = file_path.read_text(encoding="utf-8")
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                rel_path = str(file_path.relative_to(self.workspace_path))
                snapshot[rel_path] = content_hash
            except Exception as e:
                logger.debug(f"Could not snapshot {file_path}: {e}")
        
        return snapshot
    
    def diff_snapshots(
        self,
        before: Dict[str, str],
        after: Dict[str, str],
    ) -> List[CodeChange]:
        """
        Compare two snapshots and return changes.
        """
        changes = []
        
        all_files = set(before.keys()) | set(after.keys())
        
        for file_path in all_files:
            if file_path in before and file_path not in after:
                changes.append(self._create_change(file_path, "delete", before))
            elif file_path not in before and file_path in after:
                changes.append(self._create_change(file_path, "add", after))
            elif before.get(file_path) != after.get(file_path):
                changes.append(self._create_change(file_path, "modify", after))
        
        return changes
    
    def _create_change(
        self,
        file_path: str,
        change_type: str,
        snapshot: Dict[str, str],
    ) -> CodeChange:
        """Create a CodeChange object for a file."""
        full_path = self.workspace_path / file_path
        
        content_sample = ""
        lines_added = 0
        lines_removed = 0
        
        try:
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
                content_sample = content[:200]
                lines_added = len(content.splitlines())
        except Exception:
            pass
        
        return CodeChange(
            file_path=file_path,
            change_type=change_type,
            lines_added=lines_added,
            lines_removed=lines_removed,
            content_sample=content_sample,
        )
    
    def _iter_files(self):
        """Iterate over analyzable files."""
        for root, dirs, files in os.walk(self.workspace_path):
            # Filter directories
            dirs[:] = [
                d for d in dirs
                if not any(p in d for p in self.IGNORE_PATTERNS)
            ]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in self.ANALYZABLE_EXTENSIONS:
                    yield file_path


# =============================================================================
# PATTERN ANALYZER
# =============================================================================


class PatternAnalyzer:
    """
    Analyzes code changes to discover patterns.
    """
    
    # Pattern detectors
    PATTERNS = {
        PatternCategory.ERROR_HANDLING: [
            r"try:\s+.*\s+except",
            r"raise\s+\w+Error",
            r"logging\.(error|warning|exception)",
        ],
        PatternCategory.CODE_STRUCTURE: [
            r"class\s+\w+:",
            r"def\s+__init__",
            r"@dataclass",
            r"@property",
        ],
        PatternCategory.NAMING: [
            r"_[a-z]+_[a-z]+",  # snake_case
            r"[A-Z][a-z]+[A-Z]",  # PascalCase
        ],
        PatternCategory.TESTING: [
            r"def\s+test_",
            r"assert\s+",
            r"pytest\.",
        ],
        PatternCategory.ARCHITECTURE: [
            r"from\s+\.\.\s+import",
            r"__all__\s*=",
            r"Protocol\)",
        ],
    }
    
    def analyze_changes(
        self,
        changes: List[CodeChange],
        workspace_path: Path,
    ) -> List[DiscoveredPattern]:
        """
        Analyze code changes and discover patterns.
        """
        patterns_found: Dict[str, DiscoveredPattern] = {}
        
        for change in changes:
            if change.change_type == "delete":
                continue
            
            file_path = workspace_path / change.file_path
            if not file_path.exists():
                continue
            
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            
            # Check each pattern category
            for category, regexes in self.PATTERNS.items():
                for regex in regexes:
                    matches = list(re.finditer(regex, content))
                    if matches:
                        pattern_id = f"{category.name}_{hashlib.sha256(regex.encode()).hexdigest()[:8]}"
                        
                        if pattern_id not in patterns_found:
                            # Extract example
                            match = matches[0]
                            start = max(0, match.start() - 50)
                            end = min(len(content), match.end() + 50)
                            example = content[start:end]
                            
                            patterns_found[pattern_id] = DiscoveredPattern(
                                pattern_id=pattern_id,
                                category=category,
                                description=f"Pattern: {regex}",
                                example_code=example,
                                confidence=min(len(matches) / 5.0, 1.0),
                                occurrences=len(matches),
                                source_files=[change.file_path],
                            )
                        else:
                            patterns_found[pattern_id].occurrences += len(matches)
                            patterns_found[pattern_id].source_files.append(change.file_path)
        
        return list(patterns_found.values())


# =============================================================================
# EXPERTISE UPDATER
# =============================================================================


class ExpertiseUpdater:
    """
    Updates expertise.yaml based on discovered patterns and learnings.
    """
    
    def __init__(self, expertise_path: Path):
        self.expertise_path = expertise_path
        self.pending_updates: List[ExpertiseUpdate] = []
    
    def load_expertise(self) -> Dict[str, Any]:
        """Load current expertise.yaml."""
        if self.expertise_path.exists():
            with open(self.expertise_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def save_expertise(self, data: Dict[str, Any]) -> None:
        """Save expertise.yaml atomically."""
        # Write to temp file first
        temp_path = self.expertise_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        # Atomic rename
        temp_path.replace(self.expertise_path)
    
    def propose_update(
        self,
        update_type: ImprovementType,
        section: str,
        key: str,
        value: Any,
        justification: str,
    ) -> ExpertiseUpdate:
        """Propose an update to expertise."""
        update = ExpertiseUpdate(
            update_type=update_type,
            section=section,
            key=key,
            value=value,
            justification=justification,
        )
        self.pending_updates.append(update)
        return update
    
    def apply_pending_updates(self) -> int:
        """Apply all pending updates to expertise.yaml."""
        if not self.pending_updates:
            return 0
        
        expertise = self.load_expertise()
        applied = 0
        
        for update in self.pending_updates:
            # Ensure section exists
            if update.section not in expertise:
                expertise[update.section] = {}
            
            section = expertise[update.section]
            
            # Handle nested keys
            if "." in update.key:
                parts = update.key.split(".")
                for part in parts[:-1]:
                    if part not in section:
                        section[part] = {}
                    section = section[part]
                section[parts[-1]] = update.value
            else:
                section[update.key] = update.value
            
            applied += 1
            
            logger.info(
                f"Applied update: {update.update_type.name} "
                f"to {update.section}.{update.key}"
            )
        
        # Add timestamp
        if "self_improvement" not in expertise:
            expertise["self_improvement"] = {}
        expertise["self_improvement"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        expertise["self_improvement"]["updates_applied"] = applied
        
        self.save_expertise(expertise)
        self.pending_updates.clear()
        
        return applied


# =============================================================================
# SELF-IMPROVEMENT AGENT
# =============================================================================


class SelfImprovementAgent:
    """
    The Self-Improvement Agent that learns from execution experience.
    """
    
    def __init__(
        self,
        workspace_path: Path,
        expertise_path: Optional[Path] = None,
    ):
        self.workspace_path = workspace_path
        self.expertise_path = expertise_path or (workspace_path / "config" / "expertise.yaml")
        
        self.differ = CodebaseDiffer(workspace_path)
        self.pattern_analyzer = PatternAnalyzer()
        self.expertise_updater = ExpertiseUpdater(self.expertise_path)
        
        # State
        self.pre_snapshot: Optional[Dict[str, str]] = None
        self.improvement_history: List[Dict[str, Any]] = []
    
    def begin_task(self) -> None:
        """
        Call at the start of a task to capture pre-state.
        """
        self.pre_snapshot = self.differ.snapshot_state()
        logger.info(f"Captured pre-task snapshot: {len(self.pre_snapshot)} files")
    
    def complete_task(
        self,
        task_name: str,
        success: bool,
        lessons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Call at the end of a task to analyze changes and update expertise.
        
        Returns: Summary of improvements made
        """
        if self.pre_snapshot is None:
            logger.warning("No pre-snapshot available, skipping improvement")
            return {"improvements": 0, "patterns": 0}
        
        # Capture post-state
        post_snapshot = self.differ.snapshot_state()
        
        # Diff
        changes = self.differ.diff_snapshots(self.pre_snapshot, post_snapshot)
        logger.info(f"Detected {len(changes)} file changes")
        
        # Analyze patterns
        patterns = self.pattern_analyzer.analyze_changes(changes, self.workspace_path)
        logger.info(f"Discovered {len(patterns)} patterns")
        
        # Generate updates based on patterns
        for pattern in patterns:
            if pattern.confidence >= 0.5:  # Only high-confidence patterns
                self._create_pattern_update(pattern, task_name)
        
        # Record lessons learned
        if lessons:
            for lesson in lessons:
                self.expertise_updater.propose_update(
                    update_type=ImprovementType.FAILURE_LEARNED if not success else ImprovementType.NEW_PATTERN,
                    section="lessons_learned",
                    key=f"lesson_{hashlib.sha256(lesson.encode()).hexdigest()[:8]}",
                    value={
                        "lesson": lesson,
                        "task": task_name,
                        "success": success,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    justification=f"Learned from {'failed' if not success else 'successful'} task: {task_name}",
                )
        
        # Apply all updates
        applied = self.expertise_updater.apply_pending_updates()
        
        # Record in history
        summary = {
            "task": task_name,
            "success": success,
            "changes": len(changes),
            "patterns": len(patterns),
            "improvements": applied,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.improvement_history.append(summary)
        
        # Clear pre-snapshot
        self.pre_snapshot = None
        
        return summary
    
    def _create_pattern_update(
        self,
        pattern: DiscoveredPattern,
        task_name: str,
    ) -> None:
        """Create an expertise update from a discovered pattern."""
        self.expertise_updater.propose_update(
            update_type=ImprovementType.NEW_PATTERN,
            section="discovered_patterns",
            key=pattern.pattern_id,
            value={
                "category": pattern.category.name,
                "description": pattern.description,
                "confidence": pattern.confidence,
                "occurrences": pattern.occurrences,
                "source_files": pattern.source_files[:5],  # Limit to 5
                "discovered_during": task_name,
            },
            justification=f"Discovered pattern with {pattern.confidence:.2f} confidence",
        )
    
    def learn_from_failure(
        self,
        task_name: str,
        error: str,
        failed_strategy: str,
    ) -> Dict[str, Any]:
        """
        Learn from a task failure.
        """
        # Update invariants based on failure
        self.expertise_updater.propose_update(
            update_type=ImprovementType.FAILURE_LEARNED,
            section="failure_patterns",
            key=f"failure_{hashlib.sha256(error.encode()).hexdigest()[:8]}",
            value={
                "error": error[:500],  # Truncate long errors
                "strategy": failed_strategy,
                "task": task_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            justification=f"Recording failure pattern to avoid repetition",
        )
        
        # Apply
        applied = self.expertise_updater.apply_pending_updates()
        
        return {
            "task": task_name,
            "failure_recorded": True,
            "improvements": applied,
        }
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        """Get statistics on self-improvement."""
        total_improvements = sum(h["improvements"] for h in self.improvement_history)
        total_patterns = sum(h["patterns"] for h in self.improvement_history)
        
        return {
            "total_tasks_analyzed": len(self.improvement_history),
            "total_improvements": total_improvements,
            "total_patterns_discovered": total_patterns,
            "success_rate": (
                sum(1 for h in self.improvement_history if h.get("success", True))
                / len(self.improvement_history)
                if self.improvement_history else 1.0
            ),
        }


# =============================================================================
# MAIN - FOR TESTING
# =============================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("BIZRA Self-Improvement Agent - Test Run")
    print("=" * 80)
    
    # Get workspace path
    workspace_path = Path(__file__).parent.parent.parent
    
    # Create agent
    agent = SelfImprovementAgent(workspace_path)
    
    # Simulate task execution
    print("\n📸 Capturing pre-task snapshot...")
    agent.begin_task()
    
    # Simulate some code changes (in reality, task would make changes)
    print("   (In real use, the task would make code changes here)")
    
    # Complete task
    print("\n📊 Analyzing changes and updating expertise...")
    summary = agent.complete_task(
        task_name="test_self_improvement",
        success=True,
        lessons=["Self-improvement agents should run after every task"],
    )
    
    print(f"\n✅ Task completed:")
    print(f"   Changes detected: {summary['changes']}")
    print(f"   Patterns discovered: {summary['patterns']}")
    print(f"   Improvements applied: {summary['improvements']}")
    
    # Simulate learning from failure
    print("\n🔄 Simulating failure learning...")
    failure_result = agent.learn_from_failure(
        task_name="test_failed_task",
        error="AssertionError: Expected 5 but got 3",
        failed_strategy="Used greedy algorithm without backtracking",
    )
    print(f"   Failure recorded: {failure_result['failure_recorded']}")
    
    # Show stats
    print("\n" + "=" * 80)
    print("Self-Improvement Statistics:")
    for key, value in agent.get_improvement_stats().items():
        print(f"   {key}: {value}")
