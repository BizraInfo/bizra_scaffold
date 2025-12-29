"""
BIZRA AATC Reflector Agent - Autonomous API & Tool Creation
═══════════════════════════════════════════════════════════════════════════════
Analyzes successful L3 execution traces and crystallizes them into L5 tools.

The Reflector Agent automates the "crystallization" of fluid intelligence into
durable code. When the agent successfully completes a novel, complex task using
LLM reasoning, the Reflector:

1. Trace Analysis: Analyzes L3 execution logs to identify the sequence of
   perceptions and actions that led to success.

2. Abstraction: Identifies variable parameters vs. static logic.

3. Synthesis: Generates a deterministic script that encapsulates the logic.

4. Fingerprinting: Indexes the tool with a "Contextual Fingerprint" for matching.

This is the key mechanism for building "muscle memory" - the agent only has to
"think" about a problem once. Once solved, the solution becomes a reflex.

Author: BIZRA Cognitive Team
Version: 1.0.0
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger("bizra.aatc.reflector")


# =============================================================================
# TRACE TYPES
# =============================================================================


class ActionType(Enum):
    """Types of actions in an execution trace."""
    
    TOOL_CALL = auto()
    DECISION = auto()
    OBSERVATION = auto()
    TRANSFORMATION = auto()
    VALIDATION = auto()


@dataclass
class TraceStep:
    """A single step in an execution trace."""
    
    step_id: int
    action_type: ActionType
    action: str  # Tool name or decision description
    inputs: Dict[str, Any]  # Parameters/context provided
    outputs: Dict[str, Any]  # Results obtained
    success: bool
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Optional: UI/environment state for fingerprinting
    context_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action_type": self.action_type.name,
            "action": self.action,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "context_state": self.context_state,
        }


@dataclass
class ExecutionTrace:
    """Complete execution trace from L3 episodic memory."""
    
    trace_id: str
    goal: str  # What the trace was trying to achieve
    steps: List[TraceStep]
    overall_success: bool
    total_duration_ms: float
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of steps."""
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.success) / len(self.steps)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "overall_success": self.overall_success,
            "total_duration_ms": self.total_duration_ms,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "retry_count": self.retry_count,
        }


# =============================================================================
# TOOL SYNTHESIS
# =============================================================================


@dataclass
class SynthesizedTool:
    """A deterministic tool synthesized from execution traces."""
    
    tool_id: str
    name: str
    description: str
    
    # The synthesized code
    code: str
    function_name: str
    
    # Parameters extracted from trace analysis
    parameters: List[Dict[str, Any]]  # [{name, type, required, default}]
    return_type: str
    
    # Fingerprinting for matching
    context_fingerprint: str  # Hash of required UI/environment state
    trigger_patterns: List[str]  # Regex patterns that trigger this tool
    
    # Provenance
    source_traces: List[str]  # trace_ids used to synthesize
    success_rate: float
    invocation_count: int = 0
    last_used: Optional[datetime] = None
    
    # SAT binding
    sat_receipt_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "function_name": self.function_name,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "context_fingerprint": self.context_fingerprint,
            "trigger_patterns": self.trigger_patterns,
            "source_traces": self.source_traces,
            "success_rate": self.success_rate,
            "invocation_count": self.invocation_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "sat_receipt_id": self.sat_receipt_id,
        }


# =============================================================================
# PATTERN ANALYZER
# =============================================================================


class PatternAnalyzer:
    """
    Analyzes execution traces to extract reusable patterns.
    
    Identifies:
    - Variable parameters (values that change between traces)
    - Static logic (sequence that stays constant)
    - Required context (environment state needed for execution)
    """
    
    def __init__(self, min_pattern_occurrences: int = 2):
        self.min_occurrences = min_pattern_occurrences
    
    def analyze(self, traces: List[ExecutionTrace]) -> Dict[str, Any]:
        """
        Analyze multiple traces to extract patterns.
        
        Returns analysis with:
        - common_sequence: Steps that appear in all traces
        - variable_inputs: Parameters that vary between traces
        - static_inputs: Parameters that stay constant
        - required_context: Context state needed
        """
        if not traces:
            return {"error": "No traces provided"}
        
        if len(traces) < self.min_occurrences:
            logger.warning(
                f"Only {len(traces)} traces, need {self.min_occurrences} for pattern"
            )
        
        # Find common action sequence
        common_sequence = self._extract_common_sequence(traces)
        
        # Identify variable vs static inputs
        variable_inputs, static_inputs = self._classify_inputs(traces)
        
        # Extract required context
        required_context = self._extract_required_context(traces)
        
        return {
            "common_sequence": common_sequence,
            "variable_inputs": variable_inputs,
            "static_inputs": static_inputs,
            "required_context": required_context,
            "trace_count": len(traces),
            "avg_success_rate": sum(t.success_rate for t in traces) / len(traces),
        }
    
    def _extract_common_sequence(
        self, traces: List[ExecutionTrace]
    ) -> List[Dict[str, Any]]:
        """Extract the action sequence common to all traces."""
        if not traces:
            return []
        
        # Start with first trace's actions
        common = [(s.action_type.name, s.action) for s in traces[0].steps]
        
        # Intersect with other traces
        for trace in traces[1:]:
            trace_actions = [(s.action_type.name, s.action) for s in trace.steps]
            # Find longest common subsequence
            common = self._lcs(common, trace_actions)
        
        return [{"action_type": a[0], "action": a[1]} for a in common]
    
    def _lcs(
        self, seq1: List[Tuple[str, str]], seq2: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Find longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Backtrack to find the sequence
        result = []
        i, j = m, n
        while i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                result.append(seq1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        
        return result[::-1]
    
    def _classify_inputs(
        self, traces: List[ExecutionTrace]
    ) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
        """Classify inputs as variable (changing) or static (constant)."""
        all_inputs: Dict[str, List[Any]] = {}
        
        # Collect all input values for each key
        for trace in traces:
            for step in trace.steps:
                for key, value in step.inputs.items():
                    if key not in all_inputs:
                        all_inputs[key] = []
                    all_inputs[key].append(value)
        
        variable_inputs = {}
        static_inputs = {}
        
        for key, values in all_inputs.items():
            unique_values = set(str(v) for v in values)
            if len(unique_values) > 1:
                variable_inputs[key] = values
            else:
                static_inputs[key] = values[0] if values else None
        
        return variable_inputs, static_inputs
    
    def _extract_required_context(
        self, traces: List[ExecutionTrace]
    ) -> Dict[str, Any]:
        """Extract context state required for this pattern."""
        context_keys: Dict[str, int] = {}
        
        for trace in traces:
            for step in trace.steps:
                if step.context_state:
                    for key in step.context_state.keys():
                        context_keys[key] = context_keys.get(key, 0) + 1
        
        # Keys present in all traces are required
        total = len(traces)
        required = {k: v for k, v in context_keys.items() if v >= total * 0.8}
        
        return required


# =============================================================================
# CODE SYNTHESIZER
# =============================================================================


class CodeSynthesizer:
    """
    Synthesizes Python code from analyzed patterns.
    
    Generates a deterministic function that replicates the successful
    execution trace without requiring LLM reasoning.
    """
    
    TEMPLATE = '''
def {function_name}({parameters}) -> {return_type}:
    """
    {docstring}
    
    Auto-generated by AATC Reflector from execution traces.
    Source traces: {source_traces}
    Success rate: {success_rate:.2%}
    """
{body}
'''
    
    def synthesize(
        self,
        name: str,
        analysis: Dict[str, Any],
        traces: List[ExecutionTrace],
    ) -> str:
        """
        Synthesize Python code from pattern analysis.
        
        Args:
            name: Function name
            analysis: Output from PatternAnalyzer.analyze()
            traces: Original traces for reference
            
        Returns:
            Python code as string
        """
        # Build parameter list from variable inputs
        params = self._build_parameters(analysis.get("variable_inputs", {}))
        param_str = ", ".join(params) if params else ""
        
        # Build function body from common sequence
        body = self._build_body(analysis.get("common_sequence", []), traces)
        
        # Build docstring
        docstring = self._build_docstring(name, analysis)
        
        # Format template
        code = self.TEMPLATE.format(
            function_name=self._sanitize_name(name),
            parameters=param_str,
            return_type="Dict[str, Any]",
            docstring=docstring,
            source_traces=", ".join(t.trace_id for t in traces[:3]),
            success_rate=analysis.get("avg_success_rate", 0.0),
            body=body,
        )
        
        return code.strip()
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize function name to valid Python identifier."""
        # Replace spaces and special chars with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())
        # Ensure doesn't start with number
        if sanitized and sanitized[0].isdigit():
            sanitized = "fn_" + sanitized
        return sanitized or "synthesized_tool"
    
    def _build_parameters(self, variable_inputs: Dict[str, List[Any]]) -> List[str]:
        """Build parameter list from variable inputs."""
        params = []
        for key, values in variable_inputs.items():
            # Infer type from values
            if all(isinstance(v, str) for v in values):
                type_hint = "str"
            elif all(isinstance(v, (int, float)) for v in values):
                type_hint = "float"
            elif all(isinstance(v, bool) for v in values):
                type_hint = "bool"
            else:
                type_hint = "Any"
            
            params.append(f"{key}: {type_hint}")
        
        return params
    
    def _build_body(
        self, common_sequence: List[Dict[str, Any]], traces: List[ExecutionTrace]
    ) -> str:
        """Build function body from common action sequence."""
        if not common_sequence:
            return "    return {'status': 'no_action', 'result': None}"
        
        lines = ["    results = []"]
        
        for i, action in enumerate(common_sequence):
            action_type = action.get("action_type", "UNKNOWN")
            action_name = action.get("action", "unknown")
            
            if action_type == "TOOL_CALL":
                lines.append(f"    # Step {i+1}: {action_name}")
                lines.append(f"    result_{i} = execute_tool('{action_name}')")
                lines.append(f"    results.append(result_{i})")
            elif action_type == "TRANSFORMATION":
                lines.append(f"    # Step {i+1}: Transform - {action_name}")
                lines.append(f"    # TODO: Implement transformation logic")
            elif action_type == "VALIDATION":
                lines.append(f"    # Step {i+1}: Validate - {action_name}")
                lines.append(f"    if not validate_{i}():")
                lines.append(f"        raise ValueError('Validation failed at step {i+1}')")
        
        lines.append("")
        lines.append("    return {'status': 'success', 'results': results}")
        
        return "\n".join(lines)
    
    def _build_docstring(self, name: str, analysis: Dict[str, Any]) -> str:
        """Build function docstring."""
        lines = [f"Crystallized tool: {name}"]
        
        if analysis.get("variable_inputs"):
            lines.append("")
            lines.append("Args:")
            for key in analysis["variable_inputs"].keys():
                lines.append(f"    {key}: Variable parameter from traces")
        
        if analysis.get("static_inputs"):
            lines.append("")
            lines.append("Static configuration:")
            for key, value in analysis["static_inputs"].items():
                lines.append(f"    {key}: {value}")
        
        return "\n    ".join(lines)


# =============================================================================
# CONTEXTUAL FINGERPRINTER
# =============================================================================


class ContextualFingerprinter:
    """
    Creates contextual fingerprints for tool matching.
    
    A fingerprint is a hash of the UI/environment state required for
    a tool to be applicable. When the agent encounters a matching
    fingerprint, it can bypass LLM reasoning and use the crystallized tool.
    """
    
    # Keys to include in fingerprint (prioritized)
    FINGERPRINT_KEYS = [
        "page_title",
        "url_pattern",
        "dom_structure_hash",
        "window_title",
        "active_application",
        "file_extension",
        "api_endpoint",
    ]
    
    def create_fingerprint(self, context_state: Dict[str, Any]) -> str:
        """
        Create a fingerprint from context state.
        
        Returns SHA-256 hash of normalized context.
        """
        # Extract relevant keys
        fingerprint_data = {}
        for key in self.FINGERPRINT_KEYS:
            if key in context_state:
                fingerprint_data[key] = self._normalize_value(context_state[key])
        
        # Sort and serialize
        serialized = json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":"))
        
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def _normalize_value(self, value: Any) -> Any:
        """Normalize value for consistent fingerprinting."""
        if isinstance(value, str):
            # Normalize URLs by removing query params
            if value.startswith(("http://", "https://")):
                return value.split("?")[0]
            return value.lower().strip()
        return value
    
    def extract_trigger_patterns(
        self, traces: List[ExecutionTrace]
    ) -> List[str]:
        """Extract regex patterns that trigger this tool."""
        patterns = []
        
        for trace in traces:
            # Extract from goal
            goal_words = trace.goal.lower().split()
            if len(goal_words) >= 2:
                pattern = r"\b" + r"\s+".join(goal_words[:3]) + r"\b"
                patterns.append(pattern)
            
            # Extract from first action
            if trace.steps:
                first_action = trace.steps[0].action.lower()
                patterns.append(r"\b" + re.escape(first_action) + r"\b")
        
        return list(set(patterns))[:5]  # Limit to 5 patterns
    
    def match_fingerprint(
        self, current_context: Dict[str, Any], tool_fingerprint: str
    ) -> Tuple[bool, float]:
        """
        Check if current context matches a tool's fingerprint.
        
        Returns: (is_match, similarity_score)
        """
        current_fp = self.create_fingerprint(current_context)
        
        if current_fp == tool_fingerprint:
            return (True, 1.0)
        
        # Partial matching for flexibility
        # Compare individual keys
        current_data = {k: current_context.get(k) for k in self.FINGERPRINT_KEYS}
        similarity = self._compute_similarity(current_data, tool_fingerprint)
        
        return (similarity >= 0.8, similarity)
    
    def _compute_similarity(
        self, current_data: Dict[str, Any], stored_fingerprint: str
    ) -> float:
        """Compute similarity between current context and stored fingerprint."""
        # Simplified: just check if fingerprints match
        # In production, would decode and compare fields
        current_fp = hashlib.sha256(
            json.dumps(current_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Count matching characters (rough similarity)
        matches = sum(1 for a, b in zip(current_fp, stored_fingerprint) if a == b)
        return matches / len(stored_fingerprint)


# =============================================================================
# REFLECTOR AGENT
# =============================================================================


class ReflectorAgent:
    """
    The AATC Reflector Agent.
    
    Analyzes successful execution traces from L3 and crystallizes them
    into L5 deterministic tools.
    
    Workflow:
    1. Receive completed trace from L3 episodic memory
    2. Check if trace meets crystallization criteria (success rate >= 0.95)
    3. Analyze pattern with PatternAnalyzer
    4. Synthesize code with CodeSynthesizer
    5. Create fingerprint with ContextualFingerprinter
    6. Store in L5 and update expertise.yaml
    """
    
    # Minimum success rate for crystallization
    CRYSTALLIZATION_THRESHOLD = 0.95
    
    # Minimum traces needed to crystallize
    MIN_TRACES = 2
    
    def __init__(
        self,
        expertise_path: Optional[Path] = None,
        tools_dir: Optional[Path] = None,
    ):
        self.expertise_path = expertise_path or Path("config/expertise.yaml")
        self.tools_dir = tools_dir or Path("data/crystallized_tools")
        
        # Components
        self.analyzer = PatternAnalyzer()
        self.synthesizer = CodeSynthesizer()
        self.fingerprinter = ContextualFingerprinter()
        
        # Trace buffer (accumulate similar traces before crystallizing)
        self.trace_buffer: Dict[str, List[ExecutionTrace]] = {}
        
        # Ensure directories exist
        self.tools_dir.mkdir(parents=True, exist_ok=True)
    
    def receive_trace(self, trace: ExecutionTrace) -> Optional[SynthesizedTool]:
        """
        Receive a completed trace and potentially crystallize it.
        
        Returns SynthesizedTool if crystallization happened, None otherwise.
        """
        if not trace.overall_success:
            logger.debug(f"Trace {trace.trace_id} failed, skipping")
            return None
        
        if trace.success_rate < self.CRYSTALLIZATION_THRESHOLD:
            logger.debug(
                f"Trace {trace.trace_id} success rate {trace.success_rate:.2%} "
                f"below threshold {self.CRYSTALLIZATION_THRESHOLD:.2%}"
            )
            return None
        
        # Normalize goal for grouping
        goal_key = self._normalize_goal(trace.goal)
        
        # Add to buffer
        if goal_key not in self.trace_buffer:
            self.trace_buffer[goal_key] = []
        self.trace_buffer[goal_key].append(trace)
        
        # Check if we have enough traces to crystallize
        if len(self.trace_buffer[goal_key]) >= self.MIN_TRACES:
            return self._crystallize(goal_key, self.trace_buffer[goal_key])
        
        logger.info(
            f"Buffered trace for '{goal_key}' "
            f"({len(self.trace_buffer[goal_key])}/{self.MIN_TRACES})"
        )
        return None
    
    def _normalize_goal(self, goal: str) -> str:
        """Normalize goal string for grouping similar traces."""
        # Lowercase and remove special characters
        normalized = re.sub(r"[^a-z0-9\s]", "", goal.lower())
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        return normalized[:50]  # Limit length
    
    def _crystallize(
        self, goal_key: str, traces: List[ExecutionTrace]
    ) -> SynthesizedTool:
        """Crystallize traces into a deterministic tool."""
        logger.info(f"Crystallizing {len(traces)} traces for '{goal_key}'")
        
        # Analyze pattern
        analysis = self.analyzer.analyze(traces)
        
        # Synthesize code
        tool_name = self._generate_tool_name(goal_key)
        code = self.synthesizer.synthesize(tool_name, analysis, traces)
        
        # Create fingerprint
        context_state = self._merge_context_states(traces)
        fingerprint = self.fingerprinter.create_fingerprint(context_state)
        trigger_patterns = self.fingerprinter.extract_trigger_patterns(traces)
        
        # Build parameters list
        parameters = []
        for key, values in analysis.get("variable_inputs", {}).items():
            # Infer type
            sample = values[0] if values else None
            param_type = type(sample).__name__ if sample else "Any"
            parameters.append({
                "name": key,
                "type": param_type,
                "required": True,
                "default": None,
            })
        
        # Create tool
        tool_id = f"tool_{hashlib.sha256(goal_key.encode()).hexdigest()[:12]}"
        tool = SynthesizedTool(
            tool_id=tool_id,
            name=tool_name,
            description=f"Crystallized tool for: {goal_key}",
            code=code,
            function_name=self.synthesizer._sanitize_name(tool_name),
            parameters=parameters,
            return_type="Dict[str, Any]",
            context_fingerprint=fingerprint,
            trigger_patterns=trigger_patterns,
            source_traces=[t.trace_id for t in traces],
            success_rate=analysis.get("avg_success_rate", 0.0),
        )
        
        # Persist
        self._save_tool(tool)
        self._update_expertise(tool)
        
        # Clear buffer
        del self.trace_buffer[goal_key]
        
        logger.info(f"Crystallized tool: {tool.name} ({tool.tool_id})")
        return tool
    
    def _generate_tool_name(self, goal_key: str) -> str:
        """Generate a descriptive tool name from goal."""
        words = goal_key.split()[:4]
        return "_".join(words)
    
    def _merge_context_states(
        self, traces: List[ExecutionTrace]
    ) -> Dict[str, Any]:
        """Merge context states from all traces."""
        merged = {}
        for trace in traces:
            for step in trace.steps:
                if step.context_state:
                    merged.update(step.context_state)
        return merged
    
    def _save_tool(self, tool: SynthesizedTool) -> None:
        """Save tool code to filesystem."""
        tool_file = self.tools_dir / f"{tool.tool_id}.py"
        tool_file.write_text(tool.code, encoding="utf-8")
        
        # Save metadata
        meta_file = self.tools_dir / f"{tool.tool_id}.meta.json"
        meta_file.write_text(
            json.dumps(tool.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        
        logger.debug(f"Saved tool to {tool_file}")
    
    def _update_expertise(self, tool: SynthesizedTool) -> None:
        """Update expertise.yaml with new tool."""
        try:
            with open(self.expertise_path, "r", encoding="utf-8") as f:
                expertise = yaml.safe_load(f) or {}
        except FileNotFoundError:
            expertise = {}
        
        # Initialize index if needed
        if "crystallized_tools" not in expertise:
            expertise["crystallized_tools"] = {"index": {}}
        
        # Add tool to index
        expertise["crystallized_tools"]["index"][tool.name] = {
            "hash": tool.tool_id,
            "fingerprint": tool.context_fingerprint,
            "success_rate": tool.success_rate,
            "invocation_count": 0,
            "last_used": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Update version
        expertise["last_updated"] = datetime.now(timezone.utc).isoformat()
        expertise["updated_by"] = "AATC-Reflector"
        
        with open(self.expertise_path, "w", encoding="utf-8") as f:
            yaml.dump(expertise, f, default_flow_style=False, sort_keys=False)
        
        logger.debug(f"Updated expertise.yaml with tool {tool.name}")
    
    def get_matching_tool(
        self, goal: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[SynthesizedTool]:
        """
        Find a matching crystallized tool for the given goal/context.
        
        This is the Deterministic Bridge: before invoking LLM,
        check if we have a pre-crystallized tool.
        """
        # Check trigger patterns
        for goal_key, traces in self.trace_buffer.items():
            # Already in buffer, not yet crystallized
            pass
        
        # Check saved tools
        for meta_file in self.tools_dir.glob("*.meta.json"):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            # Check trigger patterns
            for pattern in meta.get("trigger_patterns", []):
                if re.search(pattern, goal.lower()):
                    # Check fingerprint if context provided
                    if context:
                        is_match, _ = self.fingerprinter.match_fingerprint(
                            context, meta.get("context_fingerprint", "")
                        )
                        if not is_match:
                            continue
                    
                    # Load and return tool
                    tool_file = self.tools_dir / f"{meta['tool_id']}.py"
                    code = tool_file.read_text(encoding="utf-8")
                    
                    return SynthesizedTool(
                        tool_id=meta["tool_id"],
                        name=meta["name"],
                        description=meta["description"],
                        code=code,
                        function_name=meta["function_name"],
                        parameters=meta["parameters"],
                        return_type=meta["return_type"],
                        context_fingerprint=meta["context_fingerprint"],
                        trigger_patterns=meta["trigger_patterns"],
                        source_traces=meta["source_traces"],
                        success_rate=meta["success_rate"],
                        invocation_count=meta.get("invocation_count", 0),
                    )
        
        return None


# =============================================================================
# DETERMINISTIC BRIDGE
# =============================================================================


class DeterministicBridge:
    """
    The Deterministic Bridge between goals and L5 tools.
    
    Before invoking expensive LLM reasoning, the bridge checks if
    a crystallized tool exists that can handle the current goal.
    
    This reduces:
    - Token costs to zero for known patterns
    - Latency from seconds to milliseconds
    - Reliability issues (deterministic vs. stochastic)
    """
    
    def __init__(self, reflector: ReflectorAgent):
        self.reflector = reflector
        self.hit_count = 0
        self.miss_count = 0
    
    def try_deterministic(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Try to execute goal using crystallized tool.
        
        Returns result dict if tool found and executed, None if miss.
        """
        tool = self.reflector.get_matching_tool(goal, context)
        
        if tool is None:
            self.miss_count += 1
            logger.debug(f"Deterministic miss for: {goal[:50]}")
            return None
        
        self.hit_count += 1
        logger.info(f"Deterministic hit! Using tool: {tool.name}")
        
        # Execute the crystallized tool
        try:
            result = self._execute_tool(tool, context or {})
            return {
                "tool_id": tool.tool_id,
                "tool_name": tool.name,
                "result": result,
                "deterministic": True,
            }
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            self.miss_count += 1
            return None
    
    def _execute_tool(
        self, tool: SynthesizedTool, context: Dict[str, Any]
    ) -> Any:
        """Execute a crystallized tool."""
        # In production, would compile and execute the code
        # For now, return a success indicator
        return {
            "status": "executed",
            "tool": tool.name,
            "message": f"Executed crystallized tool {tool.tool_id}",
        }
    
    @property
    def hit_rate(self) -> float:
        """Calculate deterministic hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
            "tools_available": len(list(self.reflector.tools_dir.glob("*.py"))),
        }


# =============================================================================
# MAIN - FOR TESTING
# =============================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("BIZRA AATC Reflector Agent - Test Run")
    print("=" * 80)
    
    # Create reflector
    reflector = ReflectorAgent()
    bridge = DeterministicBridge(reflector)
    
    # Simulate some traces
    for i in range(3):
        trace = ExecutionTrace(
            trace_id=f"trace_{i}",
            goal="login to corporate portal",
            steps=[
                TraceStep(
                    step_id=1,
                    action_type=ActionType.TOOL_CALL,
                    action="navigate_to_url",
                    inputs={"url": "https://portal.example.com"},
                    outputs={"status": "loaded"},
                    success=True,
                    duration_ms=500,
                ),
                TraceStep(
                    step_id=2,
                    action_type=ActionType.TOOL_CALL,
                    action="fill_form",
                    inputs={"username": f"user_{i}", "password": "***"},
                    outputs={"status": "filled"},
                    success=True,
                    duration_ms=200,
                ),
                TraceStep(
                    step_id=3,
                    action_type=ActionType.TOOL_CALL,
                    action="click_button",
                    inputs={"selector": "#login-btn"},
                    outputs={"status": "clicked", "response": "success"},
                    success=True,
                    duration_ms=300,
                ),
            ],
            overall_success=True,
            total_duration_ms=1000,
        )
        
        result = reflector.receive_trace(trace)
        if result:
            print(f"\n✅ Crystallized: {result.name}")
            print(f"   Code preview:\n{result.code[:200]}...")
    
    # Test deterministic bridge
    print("\n" + "=" * 80)
    print("Testing Deterministic Bridge:")
    result = bridge.try_deterministic("login to corporate portal")
    if result:
        print(f"  Hit! Tool: {result['tool_name']}")
    else:
        print("  Miss - would use LLM")
    
    print(f"\nBridge stats: {bridge.get_stats()}")
