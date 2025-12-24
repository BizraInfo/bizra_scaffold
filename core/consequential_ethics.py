"""
BIZRA AEON OMEGA - Consequential Ethics Module
═══════════════════════════════════════════════════════════════════════════════
Closes the gap between procedural ethics (verifiable process) and 
consequential ethics (outcome evaluation).

Gap Addressed: Procedural Ethics vs. Consequential Ethics (1%)
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import math


class EthicalFramework(Enum):
    """Major ethical frameworks for outcome evaluation."""
    UTILITARIAN = auto()      # Maximize aggregate good
    DEONTOLOGICAL = auto()    # Rule-based duties (Kantian)
    VIRTUE = auto()           # Character excellence (Aristotelian)
    CARE = auto()             # Relationship-focused (Gilligan)
    IHSAN = auto()            # Excellence in worship/action (Islamic)


class VerdictSeverity(Enum):
    """Severity of ethical concerns."""
    EXEMPLARY = auto()        # Exceeds ethical standards
    ACCEPTABLE = auto()       # Meets ethical standards
    CONCERNING = auto()       # Minor ethical issues
    PROBLEMATIC = auto()      # Significant ethical issues
    PROHIBITED = auto()       # Violates core principles


@dataclass
class Context:
    """Context for ethical evaluation."""
    stakeholders: List[str]
    affected_parties: List[str]
    domain: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    cultural_context: str = "universal"


@dataclass
class Action:
    """Action to be ethically evaluated."""
    id: str
    description: str
    intended_outcome: str
    potential_harms: List[str] = field(default_factory=list)
    potential_benefits: List[str] = field(default_factory=list)
    reversibility: float = 0.5  # 0=irreversible, 1=fully reversible
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EthicalEvaluation:
    """Result from a single ethical framework evaluation."""
    framework: EthicalFramework
    score: float              # -1.0 to 1.0 (negative = harmful)
    confidence: float         # 0.0 to 1.0
    reasoning: str
    concerns: List[str] = field(default_factory=list)
    commendations: List[str] = field(default_factory=list)


@dataclass
class EthicalVerdict:
    """Synthesized ethical verdict from multiple frameworks."""
    overall_score: float
    severity: VerdictSeverity
    evaluations: List[EthicalEvaluation]
    consensus_level: float     # Agreement across frameworks
    action_permitted: bool
    conditions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reasoning_trace: str = ""


class OutcomeEvaluator(ABC):
    """Abstract base for ethical outcome evaluators."""
    
    @property
    @abstractmethod
    def framework(self) -> EthicalFramework:
        """Return the ethical framework used."""
        pass
    
    @abstractmethod
    async def evaluate(self, action: Action, context: Context) -> EthicalEvaluation:
        """Evaluate action outcome through this framework."""
        pass


class UtilitarianEvaluator(OutcomeEvaluator):
    """
    Utilitarian ethics: Maximize aggregate good (Bentham/Mill).
    
    Evaluates actions based on net utility:
    U = Σ(benefits × probability × magnitude) - Σ(harms × probability × magnitude)
    """
    
    @property
    def framework(self) -> EthicalFramework:
        return EthicalFramework.UTILITARIAN
    
    async def evaluate(self, action: Action, context: Context) -> EthicalEvaluation:
        # Compute benefit score
        benefit_score = self._compute_benefit_score(action, context)
        
        # Compute harm score
        harm_score = self._compute_harm_score(action, context)
        
        # Net utility
        net_utility = benefit_score - harm_score
        
        # Normalize to [-1, 1]
        score = max(-1.0, min(1.0, net_utility))
        
        # Confidence based on information completeness
        confidence = self._compute_confidence(action, context)
        
        concerns = []
        commendations = []
        
        if harm_score > 0.5:
            concerns.append(f"Significant potential harm: {action.potential_harms}")
        if benefit_score > 0.7:
            commendations.append(f"Substantial benefits: {action.potential_benefits}")
        
        reasoning = (
            f"Net utility analysis: benefits({benefit_score:.2f}) - harms({harm_score:.2f}) "
            f"= {net_utility:.2f}. Affects {len(context.affected_parties)} parties."
        )
        
        return EthicalEvaluation(
            framework=self.framework,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            concerns=concerns,
            commendations=commendations
        )
    
    def _compute_benefit_score(self, action: Action, context: Context) -> float:
        """Compute aggregate benefit score."""
        if not action.potential_benefits:
            return 0.0
        
        # Weight by number of beneficiaries
        beneficiary_weight = min(1.0, len(context.affected_parties) / 10)
        benefit_magnitude = len(action.potential_benefits) * 0.2
        
        return min(1.0, beneficiary_weight * benefit_magnitude)
    
    def _compute_harm_score(self, action: Action, context: Context) -> float:
        """Compute aggregate harm score."""
        if not action.potential_harms:
            return 0.0
        
        # Harm weighted by irreversibility
        irreversibility_weight = 1.0 - action.reversibility
        harm_magnitude = len(action.potential_harms) * 0.3
        
        return min(1.0, irreversibility_weight * harm_magnitude)
    
    def _compute_confidence(self, action: Action, context: Context) -> float:
        """Confidence based on information completeness."""
        has_outcomes = bool(action.potential_benefits or action.potential_harms)
        has_stakeholders = bool(context.stakeholders)
        has_history = bool(context.history)
        
        return (0.4 * has_outcomes + 0.3 * has_stakeholders + 0.3 * has_history)


class DeontologicalEvaluator(OutcomeEvaluator):
    """
    Deontological ethics: Rule-based duties (Kantian).
    
    Evaluates actions based on:
    1. Universalizability (categorical imperative)
    2. Respect for autonomy (treating people as ends)
    3. Adherence to duties and rights
    """
    
    CORE_DUTIES = [
        "respect_autonomy",
        "keep_promises",
        "avoid_deception",
        "prevent_harm",
        "provide_transparency",
    ]
    
    @property
    def framework(self) -> EthicalFramework:
        return EthicalFramework.DEONTOLOGICAL
    
    async def evaluate(self, action: Action, context: Context) -> EthicalEvaluation:
        # Check universalizability
        universalizable = self._check_universalizability(action)
        
        # Check respect for persons
        respects_persons = self._check_respect_for_persons(action, context)
        
        # Check duty adherence
        duty_score = self._check_duties(action, context)
        
        # Weighted combination
        score = 0.4 * universalizable + 0.3 * respects_persons + 0.3 * duty_score
        
        concerns = []
        commendations = []
        
        if universalizable < 0.5:
            concerns.append("Action may not be universalizable")
        if respects_persons < 0.5:
            concerns.append("Potential violation of autonomy")
        if respects_persons > 0.8:
            commendations.append("Exemplary respect for persons")
        
        reasoning = (
            f"Deontological analysis: universalizability({universalizable:.2f}), "
            f"respect({respects_persons:.2f}), duties({duty_score:.2f})"
        )
        
        return EthicalEvaluation(
            framework=self.framework,
            score=score * 2 - 1,  # Convert to [-1, 1]
            confidence=0.85,
            reasoning=reasoning,
            concerns=concerns,
            commendations=commendations
        )
    
    def _check_universalizability(self, action: Action) -> float:
        """Check if action could be universalized without contradiction."""
        # Actions with high harm potential fail universalizability
        harm_penalty = len(action.potential_harms) * 0.15
        return max(0.0, 1.0 - harm_penalty)
    
    def _check_respect_for_persons(self, action: Action, context: Context) -> float:
        """Check if action treats people as ends, not mere means."""
        # Check if stakeholders are considered
        if not context.stakeholders:
            return 0.5  # Neutral when unknown
        
        # Reversibility suggests respect for correction
        reversibility_bonus = action.reversibility * 0.3
        
        # Transparency in intended outcome
        transparency_bonus = 0.3 if action.intended_outcome else 0.0
        
        return min(1.0, 0.4 + reversibility_bonus + transparency_bonus)
    
    def _check_duties(self, action: Action, context: Context) -> float:
        """Check adherence to core duties."""
        duty_scores = []
        
        for duty in self.CORE_DUTIES:
            # Simple heuristic checks
            if duty == "avoid_deception":
                duty_scores.append(1.0 if action.intended_outcome else 0.5)
            elif duty == "prevent_harm":
                duty_scores.append(1.0 - len(action.potential_harms) * 0.2)
            elif duty == "provide_transparency":
                duty_scores.append(0.8 if action.metadata else 0.5)
            else:
                duty_scores.append(0.7)  # Default moderate score
        
        return sum(duty_scores) / len(duty_scores) if duty_scores else 0.5


class VirtueEvaluator(OutcomeEvaluator):
    """
    Virtue ethics: Character excellence (Aristotelian).
    
    Evaluates actions based on whether they embody virtues:
    wisdom, courage, temperance, justice, etc.
    """
    
    VIRTUES = {
        "wisdom": 0.20,       # Practical wisdom (phronesis)
        "courage": 0.15,      # Appropriate risk-taking
        "temperance": 0.15,   # Moderation
        "justice": 0.20,      # Fairness
        "honesty": 0.15,      # Truthfulness
        "compassion": 0.15,   # Care for others
    }
    
    @property
    def framework(self) -> EthicalFramework:
        return EthicalFramework.VIRTUE
    
    async def evaluate(self, action: Action, context: Context) -> EthicalEvaluation:
        virtue_scores = {}
        
        # Evaluate each virtue
        virtue_scores["wisdom"] = self._evaluate_wisdom(action, context)
        virtue_scores["courage"] = self._evaluate_courage(action)
        virtue_scores["temperance"] = self._evaluate_temperance(action)
        virtue_scores["justice"] = self._evaluate_justice(action, context)
        virtue_scores["honesty"] = self._evaluate_honesty(action)
        virtue_scores["compassion"] = self._evaluate_compassion(action, context)
        
        # Weighted average
        total_score = sum(
            score * self.VIRTUES[virtue] 
            for virtue, score in virtue_scores.items()
        )
        
        concerns = [f"Low {v}: {s:.2f}" for v, s in virtue_scores.items() if s < 0.4]
        commendations = [f"High {v}: {s:.2f}" for v, s in virtue_scores.items() if s > 0.8]
        
        reasoning = (
            f"Virtue analysis: " + ", ".join(
                f"{v}={s:.2f}" for v, s in virtue_scores.items()
            )
        )
        
        return EthicalEvaluation(
            framework=self.framework,
            score=total_score * 2 - 1,
            confidence=0.75,
            reasoning=reasoning,
            concerns=concerns,
            commendations=commendations
        )
    
    def _evaluate_wisdom(self, action: Action, context: Context) -> float:
        """Evaluate practical wisdom in action."""
        # Wisdom shown by considering context and history
        context_awareness = 0.5 if context.constraints else 0.3
        history_awareness = 0.5 if context.history else 0.3
        return min(1.0, context_awareness + history_awareness)
    
    def _evaluate_courage(self, action: Action) -> float:
        """Evaluate appropriate courage."""
        # Courage balanced with reversibility
        return 0.5 + (1.0 - action.reversibility) * 0.3
    
    def _evaluate_temperance(self, action: Action) -> float:
        """Evaluate moderation."""
        # Temperance shown by balanced outcomes
        benefit_count = len(action.potential_benefits)
        harm_count = len(action.potential_harms)
        
        if benefit_count == 0 and harm_count == 0:
            return 0.5
        
        balance = 1.0 - abs(benefit_count - harm_count) / (benefit_count + harm_count + 1)
        return balance
    
    def _evaluate_justice(self, action: Action, context: Context) -> float:
        """Evaluate fairness."""
        # Justice shown by considering all affected parties
        stakeholder_coverage = min(1.0, len(context.stakeholders) / 5)
        return 0.4 + stakeholder_coverage * 0.6
    
    def _evaluate_honesty(self, action: Action) -> float:
        """Evaluate truthfulness."""
        # Honesty shown by transparency
        return 0.8 if action.intended_outcome and action.metadata else 0.5
    
    def _evaluate_compassion(self, action: Action, context: Context) -> float:
        """Evaluate care for others."""
        # Compassion shown by benefit focus
        if not action.potential_benefits:
            return 0.4
        return min(1.0, 0.5 + len(action.potential_benefits) * 0.15)


class CareEvaluator(OutcomeEvaluator):
    """
    Care ethics: Relationship-focused (Gilligan/Noddings).
    
    Evaluates actions based on:
    1. Maintenance of relationships
    2. Response to needs
    3. Contextual sensitivity
    """
    
    @property
    def framework(self) -> EthicalFramework:
        return EthicalFramework.CARE
    
    async def evaluate(self, action: Action, context: Context) -> EthicalEvaluation:
        # Evaluate relationship consideration
        relationship_score = self._evaluate_relationships(context)
        
        # Evaluate needs responsiveness
        needs_score = self._evaluate_needs_response(action, context)
        
        # Evaluate contextual sensitivity
        context_score = self._evaluate_context_sensitivity(action, context)
        
        # Weighted combination
        score = 0.35 * relationship_score + 0.35 * needs_score + 0.30 * context_score
        
        concerns = []
        commendations = []
        
        if relationship_score < 0.4:
            concerns.append("Insufficient relationship consideration")
        if needs_score > 0.7:
            commendations.append("Strong response to stakeholder needs")
        
        reasoning = (
            f"Care analysis: relationships({relationship_score:.2f}), "
            f"needs({needs_score:.2f}), context({context_score:.2f})"
        )
        
        return EthicalEvaluation(
            framework=self.framework,
            score=score * 2 - 1,
            confidence=0.70,
            reasoning=reasoning,
            concerns=concerns,
            commendations=commendations
        )
    
    def _evaluate_relationships(self, context: Context) -> float:
        """Evaluate consideration of relationships."""
        if not context.stakeholders and not context.affected_parties:
            return 0.3
        
        total_parties = len(set(context.stakeholders + context.affected_parties))
        return min(1.0, 0.4 + total_parties * 0.1)
    
    def _evaluate_needs_response(self, action: Action, context: Context) -> float:
        """Evaluate responsiveness to needs."""
        benefits_to_affected = len(action.potential_benefits)
        return min(1.0, 0.3 + benefits_to_affected * 0.2)
    
    def _evaluate_context_sensitivity(self, action: Action, context: Context) -> float:
        """Evaluate contextual sensitivity."""
        has_cultural = context.cultural_context != "universal"
        has_constraints = bool(context.constraints)
        has_history = bool(context.history)
        
        return 0.3 + 0.25 * has_cultural + 0.25 * has_constraints + 0.20 * has_history


class IhsanEvaluator(OutcomeEvaluator):
    """
    Ihsān ethics: Excellence in worship and action (Islamic).
    
    Evaluates actions based on:
    1. IKHLAS (إخلاص) - Sincerity/Purity of intention
    2. KARAMA (كرامة) - Dignity preservation
    3. ADL (عدل) - Justice and fairness
    4. KAMAL (كمال) - Striving for perfection
    5. ISTIDAMA (استدامة) - Sustainability
    """
    
    IHSAN_DIMENSIONS = {
        "ikhlas": 0.30,     # Sincerity
        "karama": 0.20,     # Dignity
        "adl": 0.20,        # Justice
        "kamal": 0.20,      # Excellence
        "istidama": 0.10,   # Sustainability
    }
    
    @property
    def framework(self) -> EthicalFramework:
        return EthicalFramework.IHSAN
    
    async def evaluate(self, action: Action, context: Context) -> EthicalEvaluation:
        dimension_scores = {}
        
        dimension_scores["ikhlas"] = self._evaluate_ikhlas(action)
        dimension_scores["karama"] = self._evaluate_karama(action, context)
        dimension_scores["adl"] = self._evaluate_adl(action, context)
        dimension_scores["kamal"] = self._evaluate_kamal(action)
        dimension_scores["istidama"] = self._evaluate_istidama(action)
        
        # Weighted average
        total_score = sum(
            score * self.IHSAN_DIMENSIONS[dim]
            for dim, score in dimension_scores.items()
        )
        
        concerns = []
        commendations = []
        
        for dim, score in dimension_scores.items():
            if score < 0.4:
                concerns.append(f"Low {dim.upper()} (العربية): {score:.2f}")
            elif score > 0.85:
                commendations.append(f"Excellent {dim.upper()}: {score:.2f}")
        
        reasoning = (
            f"Ihsān analysis: " + ", ".join(
                f"{d}={s:.2f}" for d, s in dimension_scores.items()
            )
        )
        
        return EthicalEvaluation(
            framework=self.framework,
            score=total_score * 2 - 1,
            confidence=0.90,  # High confidence - aligned with system principles
            reasoning=reasoning,
            concerns=concerns,
            commendations=commendations
        )
    
    def _evaluate_ikhlas(self, action: Action) -> float:
        """Evaluate sincerity/purity of intention."""
        # Ikhlas shown by clear, honest intended outcome
        has_intention = bool(action.intended_outcome)
        no_hidden = not action.metadata.get("hidden_agenda", False)
        
        return 0.5 * has_intention + 0.5 * no_hidden
    
    def _evaluate_karama(self, action: Action, context: Context) -> float:
        """Evaluate dignity preservation."""
        # Karama: every affected party's dignity preserved
        if not context.affected_parties:
            return 0.7  # Neutral when no affected parties
        
        # Check for dignity-violating harms
        dignity_harms = [
            h for h in action.potential_harms 
            if any(term in h.lower() for term in ["degrade", "humiliate", "exploit"])
        ]
        
        return max(0.0, 1.0 - len(dignity_harms) * 0.3)
    
    def _evaluate_adl(self, action: Action, context: Context) -> float:
        """Evaluate justice and fairness."""
        # Adl: fair distribution of benefits and burdens
        if not context.stakeholders:
            return 0.6
        
        # Justice shown by balanced consideration
        stakeholder_coverage = min(1.0, len(context.stakeholders) / 5)
        benefit_distribution = 0.5 if action.potential_benefits else 0.3
        
        return 0.4 * stakeholder_coverage + 0.6 * benefit_distribution
    
    def _evaluate_kamal(self, action: Action) -> float:
        """Evaluate striving for excellence."""
        # Kamal: pursuit of excellence in action
        has_benefits = bool(action.potential_benefits)
        is_thorough = bool(action.metadata)
        has_clear_intent = bool(action.intended_outcome)
        
        return 0.3 * has_benefits + 0.35 * is_thorough + 0.35 * has_clear_intent
    
    def _evaluate_istidama(self, action: Action) -> float:
        """Evaluate sustainability."""
        # Istidama: long-term sustainability of action
        reversibility_factor = action.reversibility * 0.5
        
        # Sustainable actions have more benefits than harms
        if action.potential_benefits or action.potential_harms:
            balance = len(action.potential_benefits) / (
                len(action.potential_benefits) + len(action.potential_harms) + 1
            )
        else:
            balance = 0.5
        
        return reversibility_factor + balance * 0.5


class ConsequentialEthicsEngine:
    """
    Pluralistic Consequential Ethics Engine.
    
    Synthesizes evaluations from multiple ethical frameworks to produce
    a comprehensive ethical verdict that addresses both procedural and
    consequential concerns.
    """
    
    def __init__(self):
        self.evaluators: List[OutcomeEvaluator] = [
            UtilitarianEvaluator(),
            DeontologicalEvaluator(),
            VirtueEvaluator(),
            CareEvaluator(),
            IhsanEvaluator(),
        ]
        
        # Framework weights (can be adjusted per context)
        self.framework_weights = {
            EthicalFramework.UTILITARIAN: 0.20,
            EthicalFramework.DEONTOLOGICAL: 0.20,
            EthicalFramework.VIRTUE: 0.15,
            EthicalFramework.CARE: 0.15,
            EthicalFramework.IHSAN: 0.30,  # Primary framework
        }
        
        self._evaluation_history: List[EthicalVerdict] = []
    
    async def evaluate(self, action: Action, context: Context) -> EthicalVerdict:
        """
        Evaluate action through all ethical frameworks and synthesize verdict.
        """
        # Gather evaluations from all frameworks
        evaluations = await self._gather_evaluations(action, context)
        
        # Compute weighted overall score
        overall_score = self._compute_weighted_score(evaluations)
        
        # Compute consensus level
        consensus = self._compute_consensus(evaluations)
        
        # Determine severity
        severity = self._determine_severity(overall_score, consensus)
        
        # Determine if action is permitted
        permitted, conditions = self._determine_permission(
            overall_score, severity, evaluations
        )
        
        # Generate reasoning trace
        reasoning = self._generate_reasoning_trace(evaluations)
        
        verdict = EthicalVerdict(
            overall_score=overall_score,
            severity=severity,
            evaluations=evaluations,
            consensus_level=consensus,
            action_permitted=permitted,
            conditions=conditions,
            reasoning_trace=reasoning
        )
        
        self._evaluation_history.append(verdict)
        return verdict
    
    async def _gather_evaluations(
        self, action: Action, context: Context
    ) -> List[EthicalEvaluation]:
        """Gather evaluations from all frameworks."""
        evaluations = []
        for evaluator in self.evaluators:
            try:
                evaluation = await evaluator.evaluate(action, context)
                evaluations.append(evaluation)
            except Exception as e:
                # Log but continue with other frameworks
                print(f"Evaluation error in {evaluator.framework}: {e}")
        return evaluations
    
    def _compute_weighted_score(self, evaluations: List[EthicalEvaluation]) -> float:
        """Compute weighted average score across frameworks."""
        if not evaluations:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for eval in evaluations:
            weight = self.framework_weights.get(eval.framework, 0.1)
            # Weight by both framework weight and evaluation confidence
            effective_weight = weight * eval.confidence
            weighted_sum += eval.score * effective_weight
            total_weight += effective_weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _compute_consensus(self, evaluations: List[EthicalEvaluation]) -> float:
        """Compute agreement level across frameworks."""
        if len(evaluations) < 2:
            return 1.0
        
        scores = [e.score for e in evaluations]
        mean_score = sum(scores) / len(scores)
        
        # Variance as measure of disagreement
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        
        # Convert variance to consensus (higher variance = lower consensus)
        # Max variance for [-1,1] range is 1.0
        consensus = max(0.0, 1.0 - math.sqrt(variance))
        return consensus
    
    def _determine_severity(
        self, score: float, consensus: float
    ) -> VerdictSeverity:
        """Determine severity based on score and consensus."""
        if score > 0.7 and consensus > 0.7:
            return VerdictSeverity.EXEMPLARY
        elif score > 0.3:
            return VerdictSeverity.ACCEPTABLE
        elif score > -0.2:
            return VerdictSeverity.CONCERNING
        elif score > -0.6:
            return VerdictSeverity.PROBLEMATIC
        else:
            return VerdictSeverity.PROHIBITED
    
    def _determine_permission(
        self, 
        score: float, 
        severity: VerdictSeverity,
        evaluations: List[EthicalEvaluation]
    ) -> Tuple[bool, List[str]]:
        """Determine if action is permitted and under what conditions."""
        conditions = []
        
        # Collect all concerns
        all_concerns = []
        for eval in evaluations:
            all_concerns.extend(eval.concerns)
        
        if severity == VerdictSeverity.PROHIBITED:
            return False, ["Action prohibited: severe ethical violations"]
        
        if severity == VerdictSeverity.PROBLEMATIC:
            conditions.extend([
                "Requires explicit stakeholder consent",
                "Must implement harm mitigation measures",
                "Subject to enhanced monitoring"
            ])
            conditions.extend(all_concerns[:3])  # Top 3 concerns
            return True, conditions
        
        if severity == VerdictSeverity.CONCERNING:
            conditions.extend([
                "Review recommended before proceeding",
                "Consider alternative approaches"
            ])
            conditions.extend(all_concerns[:2])
            return True, conditions
        
        if all_concerns:
            conditions.append(f"Note: {len(all_concerns)} minor concerns identified")
        
        return True, conditions
    
    def _generate_reasoning_trace(self, evaluations: List[EthicalEvaluation]) -> str:
        """Generate human-readable reasoning trace."""
        lines = ["Ethical Reasoning Trace:", "=" * 40]
        
        for eval in evaluations:
            lines.append(f"\n{eval.framework.name}:")
            lines.append(f"  Score: {eval.score:+.2f} (confidence: {eval.confidence:.0%})")
            lines.append(f"  {eval.reasoning}")
            
            if eval.concerns:
                lines.append("  Concerns: " + ", ".join(eval.concerns[:2]))
            if eval.commendations:
                lines.append("  Commendations: " + ", ".join(eval.commendations[:2]))
        
        return "\n".join(lines)
    
    def get_evaluation_history(self) -> List[EthicalVerdict]:
        """Return evaluation history."""
        return self._evaluation_history.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Tests
# ═══════════════════════════════════════════════════════════════════════════════

async def self_test():
    """Self-test for consequential ethics module."""
    print("Consequential Ethics Self-Test")
    print("=" * 50)
    
    engine = ConsequentialEthicsEngine()
    
    # Test 1: Beneficial action
    beneficial_action = Action(
        id="test-001",
        description="Provide educational resources to underserved communities",
        intended_outcome="Improve educational outcomes and opportunities",
        potential_benefits=[
            "Increased knowledge access",
            "Improved career opportunities",
            "Community development"
        ],
        potential_harms=[],
        reversibility=0.8
    )
    
    context1 = Context(
        stakeholders=["students", "teachers", "community leaders"],
        affected_parties=["underserved communities", "local schools"],
        domain="education"
    )
    
    verdict1 = await engine.evaluate(beneficial_action, context1)
    assert verdict1.severity in [VerdictSeverity.EXEMPLARY, VerdictSeverity.ACCEPTABLE]
    assert verdict1.action_permitted
    print(f"✓ Beneficial action: {verdict1.severity.name}, score={verdict1.overall_score:+.2f}")
    
    # Test 2: Mixed action
    mixed_action = Action(
        id="test-002",
        description="Implement AI system for hiring decisions",
        intended_outcome="Improve hiring efficiency",
        potential_benefits=["Faster hiring", "Consistent evaluation"],
        potential_harms=["Potential bias", "Reduced human judgment"],
        reversibility=0.6
    )
    
    context2 = Context(
        stakeholders=["HR team", "candidates", "legal department"],
        affected_parties=["job applicants", "hiring managers"],
        domain="employment"
    )
    
    verdict2 = await engine.evaluate(mixed_action, context2)
    # Mixed actions may or may not have conditions depending on scoring
    print(f"✓ Mixed action: {verdict2.severity.name}, conditions={len(verdict2.conditions)}")
    
    # Test 3: Harmful action
    harmful_action = Action(
        id="test-003",
        description="Deploy surveillance without consent",
        intended_outcome="Monitor behavior",
        potential_benefits=["Security improvement"],
        potential_harms=[
            "Privacy violation",
            "Trust degradation",
            "Dignity violation",
            "Autonomy reduction"
        ],
        reversibility=0.2
    )
    
    context3 = Context(
        stakeholders=["security team"],
        affected_parties=["all monitored individuals"],
        domain="surveillance"
    )
    
    verdict3 = await engine.evaluate(harmful_action, context3)
    # Harmful actions should score lower than beneficial ones
    assert verdict3.overall_score < verdict1.overall_score, "Harmful should score lower than beneficial"
    print(f"✓ Harmful action: {verdict3.severity.name}, permitted={verdict3.action_permitted}")
    
    # Test 4: Verify Ihsān integration
    ihsan_eval = next(
        (e for e in verdict1.evaluations if e.framework == EthicalFramework.IHSAN),
        None
    )
    assert ihsan_eval is not None
    assert ihsan_eval.confidence >= 0.8
    print(f"✓ Ihsān integration: score={ihsan_eval.score:+.2f}, confidence={ihsan_eval.confidence:.0%}")
    
    # Test 5: Consensus measurement
    assert 0.0 <= verdict1.consensus_level <= 1.0
    print(f"✓ Consensus measurement: {verdict1.consensus_level:.0%}")
    
    # Test 6: Reasoning trace
    assert len(verdict1.reasoning_trace) > 100
    print(f"✓ Reasoning trace: {len(verdict1.reasoning_trace)} characters")
    
    print("=" * 50)
    print("All consequential ethics tests passed ✓")


if __name__ == "__main__":
    import asyncio
    asyncio.run(self_test())
