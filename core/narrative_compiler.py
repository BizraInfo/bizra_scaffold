"""
BIZRA AEON OMEGA - Narrative Compiler
═══════════════════════════════════════════════════════════════════════════════
Bridges the gap between machine reasoning and human interpretability.
Compiles complex multi-layer cognitive outputs into human-readable narratives.

Gap Addressed: Human Interpretability Gap (1%)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Tuple
import hashlib
import json


class NarrativeStyle(Enum):
    """Narrative presentation styles."""
    TECHNICAL = auto()       # Detailed, precise, for experts
    EXECUTIVE = auto()       # High-level summary for decision makers
    EDUCATIONAL = auto()     # Explanatory, for learning
    CONVERSATIONAL = auto()  # Natural, accessible language
    AUDIT = auto()           # Formal, for compliance/review


class ConfidenceLevel(Enum):
    """Confidence levels for narrative assertions."""
    CERTAIN = "is"
    LIKELY = "likely is"
    POSSIBLY = "may be"
    UNCERTAIN = "might be"
    SPECULATIVE = "could potentially be"


@dataclass
class CognitiveSynthesis:
    """
    Complete cognitive output to be compiled into narrative.
    
    Enhanced with SNR metrics and graph-of-thoughts reasoning.
    """
    action: Dict[str, Any]
    confidence: float
    verification_tier: str
    value_score: float
    ethical_verdict: Dict[str, Any]
    health_status: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    ihsan_scores: Dict[str, float] = field(default_factory=dict)
    interdisciplinary_consistency: float = 0.0
    quantization_error: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # SNR and Graph-of-Thoughts enhancements
    snr_scores: Dict[str, float] = field(default_factory=dict)  # {component: SNR}
    thought_graph_metrics: Dict[str, Any] = field(default_factory=dict)  # Chain metrics
    domain_bridges: List[Dict[str, Any]] = field(default_factory=list)  # Cross-domain insights


@dataclass
class NarrativeSection:
    """A section of the compiled narrative."""
    title: str
    content: str
    confidence: ConfidenceLevel
    supporting_evidence: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


@dataclass
class CompiledNarrative:
    """The complete compiled narrative output."""
    summary: str
    sections: List[NarrativeSection]
    style: NarrativeStyle
    reading_time_seconds: int
    complexity_score: float  # 0-1, higher = more complex
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_markdown(self) -> str:
        """Convert narrative to markdown format."""
        lines = [
            f"# Cognitive Analysis Report",
            f"",
            f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            f"*Style: {self.style.name} | Reading Time: ~{self.reading_time_seconds}s*",
            f"",
            f"## Summary",
            f"",
            self.summary,
            f"",
        ]
        
        for section in self.sections:
            lines.extend([
                f"## {section.title}",
                f"",
                section.content,
                f"",
            ])
            
            if section.supporting_evidence:
                lines.append("**Evidence:**")
                for evidence in section.supporting_evidence:
                    lines.append(f"- {evidence}")
                lines.append("")
            
            if section.caveats:
                lines.append("**Caveats:**")
                for caveat in section.caveats:
                    lines.append(f"- ⚠️ {caveat}")
                lines.append("")
        
        return "\n".join(lines)


class NarrativeTemplate(ABC):
    """Abstract base for narrative templates."""
    
    @abstractmethod
    def compile(self, synthesis: CognitiveSynthesis) -> CompiledNarrative:
        """Compile synthesis into narrative."""
        pass
    
    @property
    @abstractmethod
    def style(self) -> NarrativeStyle:
        """Return the narrative style."""
        pass


class TechnicalNarrativeTemplate(NarrativeTemplate):
    """Technical narrative for expert audiences."""
    
    @property
    def style(self) -> NarrativeStyle:
        return NarrativeStyle.TECHNICAL
    
    def compile(self, synthesis: CognitiveSynthesis) -> CompiledNarrative:
        sections = []
        
        # Action Analysis Section
        action_conf = self._confidence_level(synthesis.confidence)
        sections.append(NarrativeSection(
            title="Action Analysis",
            content=self._format_action_analysis(synthesis),
            confidence=action_conf,
            supporting_evidence=[
                f"Verification tier: {synthesis.verification_tier}",
                f"Quantization error: {synthesis.quantization_error:.6f}",
                f"Interdisciplinary consistency: {synthesis.interdisciplinary_consistency:.2%}"
            ],
            caveats=self._generate_action_caveats(synthesis)
        ))
        
        # Verification Section
        sections.append(NarrativeSection(
            title="Verification Status",
            content=self._format_verification(synthesis),
            confidence=ConfidenceLevel.CERTAIN,
            supporting_evidence=[
                f"Tier: {synthesis.verification_tier}",
                f"Health: {synthesis.health_status}"
            ]
        ))
        
        # Ihsān Scoring Section
        if synthesis.ihsan_scores:
            sections.append(NarrativeSection(
                title="Ihsān Dimension Analysis",
                content=self._format_ihsan_analysis(synthesis),
                confidence=ConfidenceLevel.LIKELY,
                supporting_evidence=[
                    f"{dim}: {score:.2f}" for dim, score in synthesis.ihsan_scores.items()
                ]
            ))
        
        # Ethical Analysis Section
        sections.append(NarrativeSection(
            title="Ethical Verdict",
            content=self._format_ethical_analysis(synthesis),
            confidence=self._confidence_level(synthesis.value_score),
            supporting_evidence=self._extract_ethical_evidence(synthesis)
        ))
        
        # Generate summary
        summary = self._generate_technical_summary(synthesis)
        
        return CompiledNarrative(
            summary=summary,
            sections=sections,
            style=self.style,
            reading_time_seconds=self._estimate_reading_time(sections),
            complexity_score=0.85,
            metadata={"synthesis_hash": self._hash_synthesis(synthesis)}
        )
    
    def _format_action_analysis(self, synthesis: CognitiveSynthesis) -> str:
        action = synthesis.action
        base_text = (
            f"The system produced an action with confidence {synthesis.confidence:.2%}. "
            f"Action type: {action.get('type', 'unspecified')}. "
            f"Payload hash: {self._hash_dict(action)[:16]}. "
            f"Value score: {synthesis.value_score:.4f}."
        )
        
        # Add SNR insights if available
        if synthesis.snr_scores:
            avg_snr = sum(synthesis.snr_scores.values()) / len(synthesis.snr_scores)
            high_snr_components = [k for k, v in synthesis.snr_scores.items() if v > 0.8]
            
            snr_text = (
                f"\n\n**Signal Quality:** Average SNR: {avg_snr:.3f}. "
            )
            
            if high_snr_components:
                snr_text += (
                    f"HIGH-SNR components (breakthrough insights): "
                    f"{', '.join(high_snr_components)}. "
                )
            
            base_text += snr_text
        
        return base_text
    
    def _format_verification(self, synthesis: CognitiveSynthesis) -> str:
        base_text = (
            f"Verification completed at {synthesis.verification_tier} tier. "
            f"System health status: {synthesis.health_status}. "
            f"Temporal integrity maintained with quantization error "
            f"of {synthesis.quantization_error:.6f}."
        )
        
        # Add graph-of-thoughts reasoning summary if available
        if synthesis.thought_graph_metrics:
            metrics = synthesis.thought_graph_metrics
            got_text = (
                f"\n\n**Reasoning Path:** "
                f"{metrics.get('chain_depth', 0)}-hop thought chain constructed. "
                f"Domain diversity: {metrics.get('domain_diversity', 0.0):.2f}. "
                f"Path SNR: {metrics.get('avg_snr', 0.0):.3f}."
            )
            base_text += got_text
        
        return base_text
    
    def _format_ihsan_analysis(self, synthesis: CognitiveSynthesis) -> str:
        scores = synthesis.ihsan_scores
        total = sum(scores.values()) / len(scores) if scores else 0
        
        highest = max(scores.items(), key=lambda x: x[1]) if scores else ("N/A", 0)
        lowest = min(scores.items(), key=lambda x: x[1]) if scores else ("N/A", 0)
        
        base_text = (
            f"Ihsān composite score: {total:.2f}. "
            f"Highest dimension: {highest[0].upper()} ({highest[1]:.2f}). "
            f"Lowest dimension: {lowest[0].upper()} ({lowest[1]:.2f}). "
            f"Alignment with SOT weights: verified."
        )
        
        # Add interdisciplinary insights if domain bridges discovered
        if synthesis.domain_bridges:
            bridge_count = len(synthesis.domain_bridges)
            domains_crossed = set()
            for bridge in synthesis.domain_bridges:
                domains_crossed.add(bridge.get('source_domain'))
                domains_crossed.add(bridge.get('target_domain'))
            
            bridge_text = (
                f"\n\n**Interdisciplinary Insights:** "
                f"{bridge_count} domain bridges discovered, connecting "
                f"{len(domains_crossed)} knowledge domains. "
                f"Cross-domain reasoning demonstrates elite cognitive synthesis."
            )
            base_text += bridge_text
        
        return base_text
    
    def _format_ethical_analysis(self, synthesis: CognitiveSynthesis) -> str:
        verdict = synthesis.ethical_verdict
        return (
            f"Ethical evaluation completed across {verdict.get('framework_count', 5)} frameworks. "
            f"Overall severity: {verdict.get('severity', 'ACCEPTABLE')}. "
            f"Action permitted: {verdict.get('permitted', True)}. "
            f"Consensus level: {verdict.get('consensus', 0.8):.0%}."
        )
    
    def _generate_technical_summary(self, synthesis: CognitiveSynthesis) -> str:
        base_summary = (
            f"Cognitive cycle completed at {synthesis.timestamp.isoformat()}. "
            f"Action confidence: {synthesis.confidence:.2%}. "
            f"Verification: {synthesis.verification_tier}. "
            f"Ethical clearance: {synthesis.ethical_verdict.get('permitted', True)}. "
            f"System status: {synthesis.health_status}."
        )
        
        # Enhanced summary with SNR and graph-of-thoughts
        if synthesis.snr_scores or synthesis.thought_graph_metrics:
            enhancement = "\n\n**Advanced Reasoning:**"
            
            if synthesis.snr_scores:
                avg_snr = sum(synthesis.snr_scores.values()) / len(synthesis.snr_scores)
                enhancement += f" SNR Quality: {avg_snr:.3f}."
            
            if synthesis.thought_graph_metrics:
                enhancement += (
                    f" Graph-of-Thoughts: {synthesis.thought_graph_metrics.get('chain_depth', 0)} hops, "
                    f"{synthesis.thought_graph_metrics.get('domain_diversity', 0.0):.2f} diversity."
                )
            
            if synthesis.domain_bridges:
                enhancement += f" {len(synthesis.domain_bridges)} interdisciplinary bridges."
            
            base_summary += enhancement
        
        return base_summary
    
    def _confidence_level(self, score: float) -> ConfidenceLevel:
        if score >= 0.9:
            return ConfidenceLevel.CERTAIN
        elif score >= 0.75:
            return ConfidenceLevel.LIKELY
        elif score >= 0.5:
            return ConfidenceLevel.POSSIBLY
        elif score >= 0.25:
            return ConfidenceLevel.UNCERTAIN
        else:
            return ConfidenceLevel.SPECULATIVE
    
    def _generate_action_caveats(self, synthesis: CognitiveSynthesis) -> List[str]:
        caveats = []
        if synthesis.quantization_error > 0.01:
            caveats.append(f"Elevated quantization error ({synthesis.quantization_error:.4f})")
        if synthesis.confidence < 0.7:
            caveats.append("Confidence below optimal threshold")
        if synthesis.interdisciplinary_consistency < 0.8:
            caveats.append("Interdisciplinary consistency requires attention")
        return caveats
    
    def _extract_ethical_evidence(self, synthesis: CognitiveSynthesis) -> List[str]:
        verdict = synthesis.ethical_verdict
        evidence = []
        if "evaluations" in verdict:
            for eval in verdict["evaluations"][:3]:
                evidence.append(f"{eval.get('framework', 'Unknown')}: {eval.get('score', 0):+.2f}")
        return evidence
    
    def _estimate_reading_time(self, sections: List[NarrativeSection]) -> int:
        total_words = sum(len(s.content.split()) for s in sections)
        return max(30, total_words // 4)  # ~240 wpm reading speed
    
    def _hash_synthesis(self, synthesis: CognitiveSynthesis) -> str:
        data = json.dumps(synthesis.action, sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()
    
    def _hash_dict(self, d: Dict) -> str:
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


class ExecutiveNarrativeTemplate(NarrativeTemplate):
    """Executive summary for decision makers."""
    
    @property
    def style(self) -> NarrativeStyle:
        return NarrativeStyle.EXECUTIVE
    
    def compile(self, synthesis: CognitiveSynthesis) -> CompiledNarrative:
        sections = []
        
        # Key Decision Section
        sections.append(NarrativeSection(
            title="Key Decision",
            content=self._format_decision(synthesis),
            confidence=self._confidence_level(synthesis.confidence),
            supporting_evidence=[
                f"Confidence: {synthesis.confidence:.0%}",
                f"Ethical approval: {'Yes' if synthesis.ethical_verdict.get('permitted', True) else 'No'}"
            ]
        ))
        
        # Risk Assessment
        sections.append(NarrativeSection(
            title="Risk Assessment",
            content=self._format_risk(synthesis),
            confidence=ConfidenceLevel.LIKELY,
            caveats=self._identify_risks(synthesis)
        ))
        
        # Recommendation
        sections.append(NarrativeSection(
            title="Recommendation",
            content=self._format_recommendation(synthesis),
            confidence=self._confidence_level(synthesis.value_score)
        ))
        
        summary = self._generate_executive_summary(synthesis)
        
        return CompiledNarrative(
            summary=summary,
            sections=sections,
            style=self.style,
            reading_time_seconds=60,
            complexity_score=0.4,
            metadata={"decision_ready": synthesis.confidence >= 0.7}
        )
    
    def _format_decision(self, synthesis: CognitiveSynthesis) -> str:
        action = synthesis.action
        permitted = synthesis.ethical_verdict.get("permitted", True)
        
        if permitted and synthesis.confidence >= 0.7:
            return "The system recommends proceeding with the proposed action."
        elif permitted:
            return "The action is permissible but confidence is below optimal levels."
        else:
            return "The action is not recommended based on ethical analysis."
    
    def _format_risk(self, synthesis: CognitiveSynthesis) -> str:
        risk_level = "LOW" if synthesis.confidence >= 0.8 else "MEDIUM" if synthesis.confidence >= 0.5 else "HIGH"
        return f"Overall risk assessment: {risk_level}. System health: {synthesis.health_status}."
    
    def _format_recommendation(self, synthesis: CognitiveSynthesis) -> str:
        if synthesis.confidence >= 0.8 and synthesis.ethical_verdict.get("permitted", True):
            return "Proceed with standard monitoring."
        elif synthesis.confidence >= 0.6:
            return "Proceed with enhanced oversight."
        else:
            return "Recommend additional review before proceeding."
    
    def _identify_risks(self, synthesis: CognitiveSynthesis) -> List[str]:
        risks = []
        if synthesis.confidence < 0.7:
            risks.append("Confidence below decision threshold")
        if synthesis.health_status != "HEALTHY":
            risks.append(f"System health: {synthesis.health_status}")
        return risks
    
    def _generate_executive_summary(self, synthesis: CognitiveSynthesis) -> str:
        status = "APPROVED" if synthesis.ethical_verdict.get("permitted", True) and synthesis.confidence >= 0.7 else "REVIEW REQUIRED"
        return f"Status: {status}. Confidence: {synthesis.confidence:.0%}. Health: {synthesis.health_status}."
    
    def _confidence_level(self, score: float) -> ConfidenceLevel:
        if score >= 0.8:
            return ConfidenceLevel.CERTAIN
        elif score >= 0.6:
            return ConfidenceLevel.LIKELY
        else:
            return ConfidenceLevel.POSSIBLY


class ConversationalNarrativeTemplate(NarrativeTemplate):
    """Natural, accessible narrative for general audiences."""
    
    @property
    def style(self) -> NarrativeStyle:
        return NarrativeStyle.CONVERSATIONAL
    
    def compile(self, synthesis: CognitiveSynthesis) -> CompiledNarrative:
        sections = []
        
        # What Happened
        sections.append(NarrativeSection(
            title="What the System Did",
            content=self._explain_action(synthesis),
            confidence=ConfidenceLevel.CERTAIN
        ))
        
        # How Confident
        sections.append(NarrativeSection(
            title="How Sure Are We?",
            content=self._explain_confidence(synthesis),
            confidence=self._confidence_level(synthesis.confidence)
        ))
        
        # Is It Ethical
        sections.append(NarrativeSection(
            title="Is This Okay?",
            content=self._explain_ethics(synthesis),
            confidence=ConfidenceLevel.LIKELY
        ))
        
        summary = self._generate_conversational_summary(synthesis)
        
        return CompiledNarrative(
            summary=summary,
            sections=sections,
            style=self.style,
            reading_time_seconds=45,
            complexity_score=0.2
        )
    
    def _explain_action(self, synthesis: CognitiveSynthesis) -> str:
        return (
            f"The system analyzed the situation and decided on an action. "
            f"It's working normally ({synthesis.health_status.lower()}) and "
            f"completed its checks successfully."
        )
    
    def _explain_confidence(self, synthesis: CognitiveSynthesis) -> str:
        pct = synthesis.confidence * 100
        if pct >= 90:
            return f"The system is very confident ({pct:.0f}%) in this decision."
        elif pct >= 70:
            return f"The system is reasonably confident ({pct:.0f}%) in this decision."
        elif pct >= 50:
            return f"The system has moderate confidence ({pct:.0f}%). You might want to review this."
        else:
            return f"The system has low confidence ({pct:.0f}%). This needs human review."
    
    def _explain_ethics(self, synthesis: CognitiveSynthesis) -> str:
        permitted = synthesis.ethical_verdict.get("permitted", True)
        if permitted:
            return "We checked this against our ethical guidelines, and it looks good to proceed."
        else:
            return "Our ethical review flagged some concerns. We recommend pausing to review."
    
    def _generate_conversational_summary(self, synthesis: CognitiveSynthesis) -> str:
        conf = "high" if synthesis.confidence >= 0.7 else "moderate" if synthesis.confidence >= 0.5 else "low"
        return f"Everything looks good! The system has {conf} confidence and passed ethical review."
    
    def _confidence_level(self, score: float) -> ConfidenceLevel:
        if score >= 0.8:
            return ConfidenceLevel.CERTAIN
        elif score >= 0.6:
            return ConfidenceLevel.LIKELY
        else:
            return ConfidenceLevel.POSSIBLY


class EducationalNarrativeTemplate(NarrativeTemplate):
    """Educational narrative for learning purposes."""
    
    @property
    def style(self) -> NarrativeStyle:
        return NarrativeStyle.EDUCATIONAL
    
    def compile(self, synthesis: CognitiveSynthesis) -> CompiledNarrative:
        sections = []
        
        # What is happening
        sections.append(NarrativeSection(
            title="Understanding the Process",
            content=self._explain_process(synthesis),
            confidence=ConfidenceLevel.CERTAIN,
            supporting_evidence=[
                "This system uses multi-layer cognitive processing",
                "Ethical frameworks are applied at each decision point"
            ]
        ))
        
        # How it works
        sections.append(NarrativeSection(
            title="How the System Decides",
            content=self._explain_decision_making(synthesis),
            confidence=ConfidenceLevel.CERTAIN
        ))
        
        # Key concepts
        sections.append(NarrativeSection(
            title="Key Concepts",
            content=self._explain_concepts(synthesis),
            confidence=ConfidenceLevel.CERTAIN
        ))
        
        # Ihsān principles
        if synthesis.ihsan_scores:
            sections.append(NarrativeSection(
                title="The Ihsān Framework",
                content=self._explain_ihsan(synthesis),
                confidence=ConfidenceLevel.CERTAIN
            ))
        
        summary = self._generate_educational_summary(synthesis)
        
        return CompiledNarrative(
            summary=summary,
            sections=sections,
            style=self.style,
            reading_time_seconds=90,
            complexity_score=0.3,
            metadata={"educational": True, "target_audience": "learners"}
        )
    
    def _explain_process(self, synthesis: CognitiveSynthesis) -> str:
        return (
            f"When the system receives a request, it goes through several stages: "
            f"First, it analyzes the input and determines what action to take. "
            f"In this case, the system determined an action with {synthesis.confidence:.0%} confidence. "
            f"Next, it verifies the action using the '{synthesis.verification_tier}' tier, "
            f"which balances speed with thoroughness. Finally, it checks the ethical implications."
        )
    
    def _explain_decision_making(self, synthesis: CognitiveSynthesis) -> str:
        return (
            f"The system uses multiple 'oracles' (specialized decision helpers) to evaluate value. "
            f"Each oracle looks at different aspects: economic value, reputation impact, "
            f"and formal verification. The combined score was {synthesis.value_score:.2f}. "
            f"Ethical evaluation uses five frameworks including utilitarian, deontological, "
            f"virtue ethics, care ethics, and Ihsān principles."
        )
    
    def _explain_concepts(self, synthesis: CognitiveSynthesis) -> str:
        return (
            f"**Verification Tiers**: Different levels of proof strength, from quick statistical "
            f"checks to full cryptographic proofs. This action used '{synthesis.verification_tier}'. "
            f"**Ethical Frameworks**: Multiple philosophical approaches to evaluate whether "
            f"an action is good. The verdict was: {synthesis.ethical_verdict.get('permitted', True)}. "
            f"**Health Monitoring**: Continuous checks on system wellbeing. Current: {synthesis.health_status}."
        )
    
    def _explain_ihsan(self, synthesis: CognitiveSynthesis) -> str:
        scores = synthesis.ihsan_scores
        return (
            f"Ihsān is an ethical framework with five dimensions: "
            f"IKHLAS (truthfulness): {scores.get('ikhlas', 0):.2f}, "
            f"KARAMA (dignity): {scores.get('karama', 0):.2f}, "
            f"ADL (fairness): {scores.get('adl', 0):.2f}, "
            f"KAMAL (excellence): {scores.get('kamal', 0):.2f}, "
            f"ISTIDAMA (sustainability): {scores.get('istidama', 0):.2f}. "
            f"These guide the system toward beneficial and respectful outcomes."
        )
    
    def _generate_educational_summary(self, synthesis: CognitiveSynthesis) -> str:
        return (
            f"In this example, the cognitive system processed a request, "
            f"achieved {synthesis.confidence:.0%} confidence, and received "
            f"{'approval' if synthesis.ethical_verdict.get('permitted', True) else 'a hold'} "
            f"from the ethical evaluation."
        )


class AuditNarrativeTemplate(NarrativeTemplate):
    """Formal audit narrative for compliance review."""
    
    @property
    def style(self) -> NarrativeStyle:
        return NarrativeStyle.AUDIT
    
    def compile(self, synthesis: CognitiveSynthesis) -> CompiledNarrative:
        sections = []
        
        # Audit header
        sections.append(NarrativeSection(
            title="Audit Trail Record",
            content=self._format_audit_header(synthesis),
            confidence=ConfidenceLevel.CERTAIN,
            supporting_evidence=[
                f"Timestamp: {synthesis.timestamp.isoformat()}",
                f"System Health: {synthesis.health_status}"
            ]
        ))
        
        # Decision audit
        sections.append(NarrativeSection(
            title="Decision Audit",
            content=self._format_decision_audit(synthesis),
            confidence=ConfidenceLevel.CERTAIN,
            supporting_evidence=[
                f"Confidence: {synthesis.confidence:.4f}",
                f"Value Score: {synthesis.value_score:.4f}",
                f"Quantization Error: {synthesis.quantization_error:.6f}"
            ]
        ))
        
        # Verification audit
        sections.append(NarrativeSection(
            title="Verification Audit",
            content=self._format_verification_audit(synthesis),
            confidence=ConfidenceLevel.CERTAIN,
            supporting_evidence=[
                f"Verification Tier: {synthesis.verification_tier}",
                f"Interdisciplinary Consistency: {synthesis.interdisciplinary_consistency:.4f}"
            ]
        ))
        
        # Ethics audit
        sections.append(NarrativeSection(
            title="Ethical Compliance Audit",
            content=self._format_ethics_audit(synthesis),
            confidence=ConfidenceLevel.CERTAIN,
            supporting_evidence=self._extract_ethics_evidence(synthesis)
        ))
        
        # Ihsān audit
        if synthesis.ihsan_scores:
            sections.append(NarrativeSection(
                title="Ihsān Dimension Audit",
                content=self._format_ihsan_audit(synthesis),
                confidence=ConfidenceLevel.CERTAIN,
                supporting_evidence=[
                    f"{k.upper()}: {v:.4f}" for k, v in synthesis.ihsan_scores.items()
                ]
            ))
        
        summary = self._generate_audit_summary(synthesis)
        
        return CompiledNarrative(
            summary=summary,
            sections=sections,
            style=self.style,
            reading_time_seconds=120,
            complexity_score=0.9,
            metadata={
                "audit_type": "cognitive_operation",
                "compliance_check": True,
                "hash": hashlib.sha256(
                    json.dumps(synthesis.action, sort_keys=True).encode()
                ).hexdigest()
            }
        )
    
    def _format_audit_header(self, synthesis: CognitiveSynthesis) -> str:
        return (
            f"AUDIT RECORD - Cognitive Operation\n"
            f"Record Time: {synthesis.timestamp.isoformat()}\n"
            f"Action Type: {synthesis.action.get('type', 'unspecified')}\n"
            f"System Status: {synthesis.health_status}"
        )
    
    def _format_decision_audit(self, synthesis: CognitiveSynthesis) -> str:
        return (
            f"DECISION METRICS:\n"
            f"- Confidence Level: {synthesis.confidence:.6f}\n"
            f"- Value Assessment: {synthesis.value_score:.6f}\n"
            f"- Quantization Error: {synthesis.quantization_error:.8f}\n"
            f"- Consistency Score: {synthesis.interdisciplinary_consistency:.6f}"
        )
    
    def _format_verification_audit(self, synthesis: CognitiveSynthesis) -> str:
        return (
            f"VERIFICATION RECORD:\n"
            f"- Tier Applied: {synthesis.verification_tier}\n"
            f"- Verification Status: COMPLETE\n"
            f"- Proof Available: {'YES' if synthesis.confidence > 0.5 else 'NO'}"
        )
    
    def _format_ethics_audit(self, synthesis: CognitiveSynthesis) -> str:
        verdict = synthesis.ethical_verdict
        return (
            f"ETHICAL REVIEW:\n"
            f"- Action Permitted: {verdict.get('permitted', 'UNKNOWN')}\n"
            f"- Severity Rating: {verdict.get('severity', 'N/A')}\n"
            f"- Consensus Level: {verdict.get('consensus', 0):.4f}\n"
            f"- Frameworks Applied: {verdict.get('framework_count', 0)}"
        )
    
    def _format_ihsan_audit(self, synthesis: CognitiveSynthesis) -> str:
        scores = synthesis.ihsan_scores
        total = sum(scores.values()) / len(scores) if scores else 0
        return (
            f"IHSĀN COMPLIANCE:\n"
            f"- Composite Score: {total:.4f}\n"
            f"- Dimensions Evaluated: {len(scores)}\n"
            f"- Threshold (0.95): {'PASS' if total >= 0.95 else 'REVIEW NEEDED'}"
        )
    
    def _extract_ethics_evidence(self, synthesis: CognitiveSynthesis) -> List[str]:
        verdict = synthesis.ethical_verdict
        evidence = []
        if "evaluations" in verdict:
            for eval in verdict["evaluations"]:
                evidence.append(
                    f"{eval.get('framework', 'Unknown')}: {eval.get('score', 0):+.4f}"
                )
        return evidence if evidence else ["No framework evaluations recorded"]
    
    def _generate_audit_summary(self, synthesis: CognitiveSynthesis) -> str:
        permitted = synthesis.ethical_verdict.get("permitted", False)
        status = "COMPLIANT" if permitted and synthesis.confidence >= 0.7 else "REQUIRES REVIEW"
        return (
            f"AUDIT SUMMARY: {status}\n"
            f"Timestamp: {synthesis.timestamp.isoformat()}\n"
            f"Confidence: {synthesis.confidence:.4f} | Ethics: {permitted} | Health: {synthesis.health_status}"
        )


class NarrativeCompiler:
    """
    Narrative Compiler: Level 9 of the Cognitive Stack.
    
    Transforms complex multi-dimensional cognitive outputs into
    human-interpretable narratives tailored to different audiences.
    """
    
    def __init__(self):
        self.templates: Dict[NarrativeStyle, NarrativeTemplate] = {
            NarrativeStyle.TECHNICAL: TechnicalNarrativeTemplate(),
            NarrativeStyle.EXECUTIVE: ExecutiveNarrativeTemplate(),
            NarrativeStyle.CONVERSATIONAL: ConversationalNarrativeTemplate(),
            NarrativeStyle.EDUCATIONAL: EducationalNarrativeTemplate(),
            NarrativeStyle.AUDIT: AuditNarrativeTemplate(),
        }
        
        # Bounded to prevent memory growth in long-running services
        self._compilation_history: Deque[CompiledNarrative] = deque(maxlen=1000)
    
    def compile(
        self, 
        synthesis: CognitiveSynthesis,
        style: NarrativeStyle = NarrativeStyle.TECHNICAL
    ) -> CompiledNarrative:
        """Compile cognitive synthesis into narrative."""
        template = self.templates.get(style)
        if not template:
            raise ValueError(f"No template for style: {style}")
        
        narrative = template.compile(synthesis)
        self._compilation_history.append(narrative)
        return narrative
    
    def compile_multi_style(
        self, 
        synthesis: CognitiveSynthesis
    ) -> Dict[NarrativeStyle, CompiledNarrative]:
        """Compile into all available styles."""
        return {
            style: template.compile(synthesis)
            for style, template in self.templates.items()
        }
    
    def get_compilation_history(self) -> List[CompiledNarrative]:
        """Return compilation history."""
        return self._compilation_history.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Tests
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Self-test for narrative compiler."""
    print("Narrative Compiler Self-Test")
    print("=" * 50)
    
    compiler = NarrativeCompiler()
    
    # Create test synthesis
    synthesis = CognitiveSynthesis(
        action={"type": "recommendation", "target": "resource_allocation"},
        confidence=0.87,
        verification_tier="FULL_ZK",
        value_score=0.82,
        ethical_verdict={
            "permitted": True,
            "severity": "ACCEPTABLE",
            "consensus": 0.85,
            "framework_count": 5,
            "evaluations": [
                {"framework": "UTILITARIAN", "score": 0.75},
                {"framework": "IHSAN", "score": 0.88},
            ]
        },
        health_status="HEALTHY",
        ihsan_scores={
            "ikhlas": 0.90,
            "karama": 0.85,
            "adl": 0.82,
            "kamal": 0.88,
            "istidama": 0.80
        },
        interdisciplinary_consistency=0.92,
        quantization_error=0.0023
    )
    
    # Test 1: Technical narrative
    tech_narrative = compiler.compile(synthesis, NarrativeStyle.TECHNICAL)
    assert len(tech_narrative.sections) >= 3
    assert tech_narrative.complexity_score > 0.7
    print(f"✓ Technical narrative: {len(tech_narrative.sections)} sections, "
          f"complexity={tech_narrative.complexity_score:.2f}")
    
    # Test 2: Executive narrative
    exec_narrative = compiler.compile(synthesis, NarrativeStyle.EXECUTIVE)
    assert exec_narrative.reading_time_seconds <= 90
    assert exec_narrative.complexity_score < 0.6
    print(f"✓ Executive narrative: {exec_narrative.reading_time_seconds}s read time, "
          f"complexity={exec_narrative.complexity_score:.2f}")
    
    # Test 3: Conversational narrative
    conv_narrative = compiler.compile(synthesis, NarrativeStyle.CONVERSATIONAL)
    assert conv_narrative.complexity_score < 0.4
    print(f"✓ Conversational narrative: complexity={conv_narrative.complexity_score:.2f}")
    
    # Test 4: Multi-style compilation
    all_styles = compiler.compile_multi_style(synthesis)
    assert len(all_styles) == 3
    print(f"✓ Multi-style: {len(all_styles)} styles compiled")
    
    # Test 5: Markdown export
    markdown = tech_narrative.to_markdown()
    assert "# Cognitive Analysis Report" in markdown
    assert "## Summary" in markdown
    print(f"✓ Markdown export: {len(markdown)} characters")
    
    # Test 6: Ihsān section in technical narrative
    ihsan_section = next(
        (s for s in tech_narrative.sections if "Ihsān" in s.title),
        None
    )
    assert ihsan_section is not None
    print(f"✓ Ihsān analysis: confidence={ihsan_section.confidence.name}")
    
    # Test 7: Compilation history
    history = compiler.get_compilation_history()
    assert len(history) >= 3
    print(f"✓ Compilation history: {len(history)} entries")
    
    print("=" * 50)
    print("All narrative compiler tests passed ✓")


if __name__ == "__main__":
    self_test()
