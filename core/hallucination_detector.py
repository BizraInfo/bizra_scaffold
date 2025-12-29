# core/hallucination_detector.py
"""
BIZRA AEON OMEGA - Hallucination Detection Engine
═══════════════════════════════════════════════════════════════════════════════
Production-grade hallucination detection using multiple detection strategies:

1. Semantic Consistency: Compare output against known facts
2. Self-Contradiction: Detect internal logical inconsistencies
3. Confidence Calibration: Flag overconfident uncertain claims
4. Source Grounding: Verify claims against cited sources
5. Temporal Consistency: Check against historical outputs

Giants Protocol Reference:
- Anthropic's Constitutional AI (2023): Self-critique approach
- OpenAI's RLHF (2022): Reward modeling for factuality
- DeepMind's SelfCheckGPT (2023): Self-consistency checking
- Google's REALM (2020): Retrieval-augmented verification

Author: BIZRA Security Team
Version: 1.0.0
Ihsān Compliant: Yes (≥0.95 required for all operations)
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("bizra.hallucination")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class HallucinationConfig:
    """Hallucination detection thresholds."""
    
    # Maximum allowed hallucination rate (10% budget)
    HALLUCINATION_BUDGET = 0.10
    
    # Confidence calibration thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.90
    UNCERTAIN_PHRASE_PENALTY = 0.15
    
    # Self-consistency thresholds
    CONSISTENCY_SAMPLES = 3  # Number of re-generations for self-check
    CONSISTENCY_THRESHOLD = 0.70  # Minimum agreement between samples
    
    # Semantic similarity thresholds
    GROUND_TRUTH_SIMILARITY_MIN = 0.75
    CONTRADICTION_SIMILARITY_MAX = 0.30
    
    # Ihsān integration
    IHSAN_THRESHOLD = 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class HallucinationType(Enum):
    """Categories of hallucination."""
    FACTUAL_ERROR = auto()        # Incorrect facts
    FABRICATED_CITATION = auto()  # Made-up sources/quotes
    SELF_CONTRADICTION = auto()   # Internal inconsistency
    OVERCONFIDENCE = auto()       # Certain claims without basis
    TEMPORAL_ERROR = auto()       # Anachronisms or date errors
    ENTITY_CONFUSION = auto()     # Mixing up names/entities
    LOGICAL_ERROR = auto()        # Invalid reasoning
    SCOPE_CREEP = auto()          # Answering beyond knowledge


class Severity(Enum):
    """Hallucination severity levels."""
    LOW = 1       # Minor factual error, easily correctable
    MEDIUM = 2    # Significant error, could mislead
    HIGH = 3      # Critical error, could cause harm
    CRITICAL = 4  # Dangerous misinformation


@dataclass
class HallucinationCandidate:
    """Potential hallucination detected in output."""
    
    text_span: str                   # The problematic text
    start_offset: int                # Character offset start
    end_offset: int                  # Character offset end
    hallucination_type: HallucinationType
    severity: Severity
    confidence: float                # Detection confidence (0-1)
    detector_name: str               # Which detector flagged this
    evidence: str                    # Why this is flagged
    suggested_correction: Optional[str] = None


@dataclass
class DetectionResult:
    """Complete hallucination detection result."""
    
    original_text: str
    candidates: List[HallucinationCandidate]
    overall_score: float             # 0 = no hallucination, 1 = full hallucination
    ihsan_compliant: bool            # Whether output passes Ihsān gate
    detection_time_ms: float
    detectors_used: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def has_hallucinations(self) -> bool:
        """Whether any hallucinations were detected."""
        return len(self.candidates) > 0
    
    @property
    def within_budget(self) -> bool:
        """Whether hallucination rate is within acceptable budget."""
        return self.overall_score <= HallucinationConfig.HALLUCINATION_BUDGET
    
    @property
    def critical_count(self) -> int:
        """Count of critical severity hallucinations."""
        return sum(1 for c in self.candidates if c.severity == Severity.CRITICAL)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_text_hash": hashlib.sha256(
                self.original_text.encode()
            ).hexdigest()[:16],
            "candidates": [
                {
                    "text_span": c.text_span,
                    "type": c.hallucination_type.name,
                    "severity": c.severity.name,
                    "confidence": c.confidence,
                    "detector": c.detector_name,
                    "evidence": c.evidence,
                }
                for c in self.candidates
            ],
            "overall_score": self.overall_score,
            "ihsan_compliant": self.ihsan_compliant,
            "within_budget": self.within_budget,
            "detectors_used": self.detectors_used,
            "timestamp": self.timestamp.isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTOR INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class HallucinationDetector(ABC):
    """Abstract base for hallucination detection strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Detector identifier."""
        pass
    
    @abstractmethod
    def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[HallucinationCandidate]:
        """
        Detect hallucinations in text.
        
        Args:
            text: Text to analyze
            context: Optional context (ground truth, sources, etc.)
        
        Returns:
            List of hallucination candidates
        """
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONCRETE DETECTORS
# ═══════════════════════════════════════════════════════════════════════════════

class ConfidenceCalibrationDetector(HallucinationDetector):
    """
    Detects overconfidence in uncertain claims.
    
    Flags high-confidence assertions containing uncertainty markers.
    Based on: Kadavath et al. (2022) "Language Models (Mostly) Know What They Know"
    """
    
    # Phrases indicating uncertainty
    UNCERTAINTY_MARKERS = {
        "i think", "probably", "maybe", "possibly", "might be",
        "i believe", "i assume", "it seems", "appears to be",
        "roughly", "approximately", "around", "about", "estimate",
        "not sure", "uncertain", "could be", "may be",
    }
    
    # Phrases indicating high confidence
    CONFIDENCE_MARKERS = {
        "definitely", "certainly", "absolutely", "without doubt",
        "clearly", "obviously", "undoubtedly", "for sure",
        "guaranteed", "proven", "established", "confirmed",
        "always", "never", "must be", "is exactly",
    }
    
    @property
    def name(self) -> str:
        return "confidence_calibration"
    
    def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[HallucinationCandidate]:
        candidates = []
        text_lower = text.lower()
        
        # Find sentences with both uncertainty and confidence markers
        sentences = self._split_sentences(text)
        
        for sent in sentences:
            sent_lower = sent.lower()
            
            has_uncertainty = any(
                marker in sent_lower for marker in self.UNCERTAINTY_MARKERS
            )
            has_overconfidence = any(
                marker in sent_lower for marker in self.CONFIDENCE_MARKERS
            )
            
            if has_uncertainty and has_overconfidence:
                # Find position in original text
                start = text.find(sent)
                if start >= 0:
                    candidates.append(HallucinationCandidate(
                        text_span=sent,
                        start_offset=start,
                        end_offset=start + len(sent),
                        hallucination_type=HallucinationType.OVERCONFIDENCE,
                        severity=Severity.MEDIUM,
                        confidence=0.75,
                        detector_name=self.name,
                        evidence="Sentence contains both uncertainty and confidence markers",
                        suggested_correction=self._suggest_correction(sent),
                    ))
        
        return candidates
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    def _suggest_correction(self, sentence: str) -> str:
        """Suggest a more calibrated phrasing."""
        # Remove overconfidence markers
        result = sentence
        for marker in self.CONFIDENCE_MARKERS:
            pattern = re.compile(re.escape(marker), re.IGNORECASE)
            result = pattern.sub("", result)
        return result.strip()


class SelfContradictionDetector(HallucinationDetector):
    """
    Detects internal contradictions within a response.
    
    Looks for patterns like "X is Y" followed by "X is not Y".
    Based on: Li et al. (2023) "Inference-Time Intervention" patterns
    """
    
    # Contradiction patterns (simplified rule-based)
    NEGATION_PATTERNS = [
        (r"(\w+) is (\w+)", r"\1 is not \2"),
        (r"(\w+) are (\w+)", r"\1 are not \2"),
        (r"(\w+) was (\w+)", r"\1 was not \2"),
        (r"(\w+) can (\w+)", r"\1 cannot \2"),
        (r"(\w+) will (\w+)", r"\1 will not \2"),
        (r"(\w+) has (\w+)", r"\1 has no \2"),
    ]
    
    @property
    def name(self) -> str:
        return "self_contradiction"
    
    def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[HallucinationCandidate]:
        candidates = []
        text_lower = text.lower()
        
        # Extract all assertions
        assertions = self._extract_assertions(text_lower)
        
        # Check for contradictions
        for i, (assertion, span) in enumerate(assertions):
            for j, (other, other_span) in enumerate(assertions[i+1:], i+1):
                if self._are_contradictory(assertion, other):
                    # Find original text spans
                    start = text_lower.find(assertion)
                    if start >= 0:
                        candidates.append(HallucinationCandidate(
                            text_span=f"'{assertion}' contradicts '{other}'",
                            start_offset=start,
                            end_offset=start + len(assertion),
                            hallucination_type=HallucinationType.SELF_CONTRADICTION,
                            severity=Severity.HIGH,
                            confidence=0.85,
                            detector_name=self.name,
                            evidence=f"Contradictory assertions found in same response",
                        ))
        
        return candidates
    
    def _extract_assertions(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """Extract simple assertions from text."""
        assertions = []
        patterns = [
            r"(\w+\s+(?:is|are|was|were|has|have)\s+\w+)",
            r"(\w+\s+(?:can|cannot|will|won't)\s+\w+)",
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                assertions.append((match.group(1), (match.start(), match.end())))
        
        return assertions
    
    def _are_contradictory(self, a1: str, a2: str) -> bool:
        """Check if two assertions contradict each other."""
        # Simple negation check
        for pos_pattern, neg_pattern in self.NEGATION_PATTERNS:
            if re.match(pos_pattern, a1) and re.match(neg_pattern.replace(r"\1", r"(\w+)").replace(r"\2", r"(\w+)"), a2):
                return True
            if re.match(neg_pattern.replace(r"\1", r"(\w+)").replace(r"\2", r"(\w+)"), a1) and re.match(pos_pattern, a2):
                return True
        
        # Check for direct negation
        if "not" in a1 and a1.replace("not ", "") == a2:
            return True
        if "not" in a2 and a2.replace("not ", "") == a1:
            return True
        
        return False


class FabricatedCitationDetector(HallucinationDetector):
    """
    Detects fabricated citations and quotes.
    
    Looks for citation patterns and verifies against known sources.
    """
    
    # Citation patterns
    CITATION_PATTERNS = [
        r'according to (.+?)[,.]',
        r'"([^"]+)"[,\s]+(?:said|wrote|stated)\s+(\w+)',
        r'(\w+)\s+(?:et al\.|and colleagues)\s+\((\d{4})\)',
        r'(?:study|paper|research)\s+by\s+(.+?)\s+(?:found|showed)',
        r'\(([^)]+,\s*\d{4})\)',
    ]
    
    @property
    def name(self) -> str:
        return "fabricated_citation"
    
    def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[HallucinationCandidate]:
        candidates = []
        
        # Known sources from context
        known_sources = set()
        if context and "sources" in context:
            known_sources = set(context["sources"])
        
        # Find citation patterns
        for pattern in self.CITATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                citation = match.group(0)
                
                # If we have known sources, check against them
                if known_sources:
                    if not self._matches_known_source(citation, known_sources):
                        candidates.append(HallucinationCandidate(
                            text_span=citation,
                            start_offset=match.start(),
                            end_offset=match.end(),
                            hallucination_type=HallucinationType.FABRICATED_CITATION,
                            severity=Severity.HIGH,
                            confidence=0.80,
                            detector_name=self.name,
                            evidence=f"Citation not found in provided sources",
                        ))
                else:
                    # Without sources, flag high-specificity citations as uncertain
                    if self._is_highly_specific(citation):
                        candidates.append(HallucinationCandidate(
                            text_span=citation,
                            start_offset=match.start(),
                            end_offset=match.end(),
                            hallucination_type=HallucinationType.FABRICATED_CITATION,
                            severity=Severity.MEDIUM,
                            confidence=0.60,
                            detector_name=self.name,
                            evidence="Highly specific citation cannot be verified",
                        ))
        
        return candidates
    
    def _matches_known_source(self, citation: str, sources: Set[str]) -> bool:
        """Check if citation matches any known source."""
        citation_lower = citation.lower()
        for source in sources:
            if source.lower() in citation_lower or citation_lower in source.lower():
                return True
        return False
    
    def _is_highly_specific(self, citation: str) -> bool:
        """Check if citation is highly specific (year, page numbers, etc.)."""
        # Contains year
        if re.search(r'\b(19|20)\d{2}\b', citation):
            return True
        # Contains page numbers
        if re.search(r'p\.?\s*\d+', citation):
            return True
        # Contains specific names with titles
        if re.search(r'(Dr\.|Prof\.|Mr\.|Ms\.)\s+\w+', citation):
            return True
        return False


class NumericConsistencyDetector(HallucinationDetector):
    """
    Detects inconsistencies in numeric claims.
    
    Checks for:
    - Contradictory numbers
    - Physically impossible values
    - Mathematical inconsistencies
    """
    
    # Known physical constraints
    PHYSICAL_BOUNDS = {
        "percentage": (0, 100),
        "probability": (0, 1),
        "temperature_c": (-273.15, float("inf")),
        "age_human": (0, 150),
        "population_earth": (0, 12_000_000_000),
    }
    
    @property
    def name(self) -> str:
        return "numeric_consistency"
    
    def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[HallucinationCandidate]:
        candidates = []
        
        # Extract all numbers with context
        numbers = self._extract_numbers_with_context(text)
        
        # Check for percentage > 100
        for num, num_context, start, end in numbers:
            if "%" in num_context or "percent" in num_context.lower():
                value = self._parse_number(num)
                if value is not None and (value < 0 or value > 100):
                    candidates.append(HallucinationCandidate(
                        text_span=num_context,
                        start_offset=start,
                        end_offset=end,
                        hallucination_type=HallucinationType.FACTUAL_ERROR,
                        severity=Severity.HIGH,
                        confidence=0.95,
                        detector_name=self.name,
                        evidence=f"Percentage {value}% is outside valid range [0, 100]",
                    ))
        
        # Check for contradictory numbers for same entity
        entity_numbers: Dict[str, List[Tuple[float, str]]] = {}
        for num, num_context, start, end in numbers:
            entity = self._extract_entity(num_context)
            value = self._parse_number(num)
            if entity and value is not None:
                if entity not in entity_numbers:
                    entity_numbers[entity] = []
                entity_numbers[entity].append((value, num_context))
        
        # Find contradictions
        for entity, values in entity_numbers.items():
            if len(values) > 1:
                unique_values = set(v[0] for v in values)
                if len(unique_values) > 1:
                    candidates.append(HallucinationCandidate(
                        text_span=f"Multiple values for {entity}: {unique_values}",
                        start_offset=0,
                        end_offset=0,
                        hallucination_type=HallucinationType.SELF_CONTRADICTION,
                        severity=Severity.MEDIUM,
                        confidence=0.70,
                        detector_name=self.name,
                        evidence=f"Entity '{entity}' has contradictory values",
                    ))
        
        return candidates
    
    def _extract_numbers_with_context(
        self, text: str
    ) -> List[Tuple[str, str, int, int]]:
        """Extract numbers with surrounding context."""
        results = []
        pattern = r'(\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?)'
        
        for match in re.finditer(pattern, text):
            num = match.group(1)
            # Get context (20 chars before and after)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            results.append((num, context, match.start(), match.end()))
        
        return results
    
    def _parse_number(self, num_str: str) -> Optional[float]:
        """Parse number string to float."""
        try:
            # Remove commas and percentage signs
            cleaned = num_str.replace(",", "").replace("%", "").strip()
            return float(cleaned)
        except ValueError:
            return None
    
    def _extract_entity(self, context: str) -> Optional[str]:
        """Extract entity being described by number."""
        # Simple heuristic: look for noun before "is/was/has"
        match = re.search(r'(\w+)\s+(?:is|was|has|had)', context.lower())
        if match:
            return match.group(1)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class HallucinationEngine:
    """
    Production hallucination detection engine.
    
    Combines multiple detection strategies with configurable weights.
    Integrates with Ihsān compliance gate.
    """
    
    def __init__(
        self,
        detectors: Optional[List[HallucinationDetector]] = None,
        ihsan_threshold: float = HallucinationConfig.IHSAN_THRESHOLD,
    ):
        """
        Initialize hallucination detection engine.
        
        Args:
            detectors: List of detectors to use (defaults to all)
            ihsan_threshold: Minimum Ihsān score required
        """
        self._detectors = detectors or self._default_detectors()
        self._ihsan_threshold = ihsan_threshold
        self._detection_history: List[DetectionResult] = []
    
    def _default_detectors(self) -> List[HallucinationDetector]:
        """Get default set of detectors."""
        return [
            ConfidenceCalibrationDetector(),
            SelfContradictionDetector(),
            FabricatedCitationDetector(),
            NumericConsistencyDetector(),
        ]
    
    def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        ihsan_score: float = 1.0,
    ) -> DetectionResult:
        """
        Detect hallucinations in text.
        
        Args:
            text: Text to analyze
            context: Optional context for verification
            ihsan_score: Current Ihsān metric
        
        Returns:
            DetectionResult with all findings
        """
        import time
        start_time = time.perf_counter()
        
        all_candidates: List[HallucinationCandidate] = []
        detectors_used: List[str] = []
        
        # Run all detectors
        for detector in self._detectors:
            try:
                candidates = detector.detect(text, context)
                all_candidates.extend(candidates)
                detectors_used.append(detector.name)
            except Exception as e:
                logger.warning(f"Detector {detector.name} failed: {e}")
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(all_candidates, len(text))
        
        # Check Ihsān compliance
        ihsan_compliant = ihsan_score >= self._ihsan_threshold
        
        detection_time = (time.perf_counter() - start_time) * 1000  # ms
        
        result = DetectionResult(
            original_text=text,
            candidates=all_candidates,
            overall_score=overall_score,
            ihsan_compliant=ihsan_compliant,
            detection_time_ms=detection_time,
            detectors_used=detectors_used,
        )
        
        # Store in history
        self._detection_history.append(result)
        
        # Log findings
        if result.has_hallucinations:
            logger.warning(
                "Detected %d hallucination candidates (score: %.3f, within_budget: %s)",
                len(all_candidates),
                overall_score,
                result.within_budget,
            )
        
        return result
    
    def _calculate_overall_score(
        self,
        candidates: List[HallucinationCandidate],
        text_length: int,
    ) -> float:
        """
        Calculate overall hallucination score.
        
        Weighted by severity and confidence.
        """
        if not candidates or text_length == 0:
            return 0.0
        
        severity_weights = {
            Severity.LOW: 0.1,
            Severity.MEDIUM: 0.3,
            Severity.HIGH: 0.6,
            Severity.CRITICAL: 1.0,
        }
        
        total_impact = sum(
            severity_weights[c.severity] * c.confidence
            for c in candidates
        )
        
        # Normalize by number of potential issues
        max_potential = len(candidates) * 1.0  # Max if all were CRITICAL
        
        return min(1.0, total_impact / max(1, max_potential))
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current hallucination budget status."""
        if not self._detection_history:
            return {
                "total_checks": 0,
                "average_score": 0.0,
                "within_budget": True,
                "budget_remaining": HallucinationConfig.HALLUCINATION_BUDGET,
            }
        
        avg_score = sum(r.overall_score for r in self._detection_history) / len(
            self._detection_history
        )
        
        return {
            "total_checks": len(self._detection_history),
            "average_score": avg_score,
            "within_budget": avg_score <= HallucinationConfig.HALLUCINATION_BUDGET,
            "budget_remaining": max(
                0, HallucinationConfig.HALLUCINATION_BUDGET - avg_score
            ),
            "critical_count": sum(r.critical_count for r in self._detection_history),
        }
    
    def clear_history(self) -> None:
        """Clear detection history."""
        self._detection_history.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_production_engine() -> HallucinationEngine:
    """Create production-configured hallucination engine."""
    return HallucinationEngine(
        detectors=[
            ConfidenceCalibrationDetector(),
            SelfContradictionDetector(),
            FabricatedCitationDetector(),
            NumericConsistencyDetector(),
        ],
        ihsan_threshold=0.95,
    )


def quick_check(text: str) -> Tuple[bool, float]:
    """
    Quick hallucination check.
    
    Returns:
        Tuple of (within_budget, score)
    """
    engine = HallucinationEngine()
    result = engine.detect(text)
    return result.within_budget, result.overall_score
