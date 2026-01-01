"""
Dual Agent Integration System
═══════════════════════════════════════════════════════════════════════════════
Seamless integration of PAT (Personal Agent) and SAT (System Agent) systems.

This module provides the sovereign interface where PAT and SAT work together:
- PAT interprets MoMo's intent and personal context
- SAT executes operations with system-level verification
- Seamless handover enables autonomous dual-agent operation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import json

from core.agents.pat import PATAgent, PATConfig, create_pat_agent
from core.agents.sat import SATAgent, SATConfig, create_sat_agent, VerificationResult
from core.pci.envelope import PCIEnvelope
from core.sovereign.auth import verify_sovereign_identity, require_sovereign_auth

logger = logging.getLogger(__name__)


@dataclass
class DualAgentSession:
    """A session between PAT and SAT agents."""
    session_id: str
    sovereign_user: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operations: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "active"  # "active", "completed", "failed"

    def add_operation(self, operation: Dict[str, Any]) -> None:
        """Add an operation to the session."""
        self.operations.append(operation)

    def complete_session(self) -> None:
        """Mark session as completed."""
        self.status = "completed"

    def fail_session(self, reason: str) -> None:
        """Mark session as failed."""
        self.status = "failed"
        self.add_operation({"type": "failure", "reason": reason, "timestamp": datetime.now(timezone.utc).isoformat()})


@dataclass
class SovereignIntent:
    """A sovereign command or intent from MoMo."""
    intent: str
    context: Dict[str, Any]
    urgency: str = "normal"  # "low", "normal", "high", "critical"
    sovereign_verified: bool = False

    def verify_sovereignty(self) -> bool:
        """Verify this intent comes from the sovereign."""
        is_sovereign, status = verify_sovereign_identity()
        self.sovereign_verified = is_sovereign
        return is_sovereign


class SovereignDualAgents:
    """
    Unified PAT + SAT system under sovereign control.

    This provides the seamless integration where:
    1. PAT interprets MoMo's personal intent and context
    2. SAT executes with system-level verification and commitment
    3. Seamless handover enables autonomous operation
    """

    def __init__(self):
        # Initialize agents with sovereign configurations
        self.pat_agent = self._initialize_pat_agent()
        self.sat_agent = self._initialize_sat_agent()

        # Session management
        self.active_sessions: Dict[str, DualAgentSession] = {}
        self.session_counter = 0

        # Sovereign state
        self.sovereign_mode = False
        self.last_sovereign_verification = None

        logger.info("Sovereign Dual Agents initialized")

    def _initialize_pat_agent(self) -> PATAgent:
        """Initialize PAT agent with sovereign configuration."""
        pat_config = PATConfig(
            agent_id="sovereign-pat",
            ihsan_threshold=0.95,  # Sovereign standard
            auto_sign=True,
            validate_ihsan=True,
        )

        # In production, this would use proper cryptographic keys
        return create_pat_agent(
            agent_id=pat_config.agent_id,
            ihsan_threshold=pat_config.ihsan_threshold,
        )

    def _initialize_sat_agent(self) -> SATAgent:
        """Initialize SAT agent with sovereign configuration."""
        sat_config = SATConfig(
            agent_id="sovereign-sat",
            ihsan_threshold=0.95,  # Sovereign standard
            snr_threshold=0.80,    # High quality threshold
        )

        # In production, this would use proper cryptographic keys
        return create_sat_agent(
            agent_id=sat_config.agent_id,
            ihsan_threshold=sat_config.ihsan_threshold,
            snr_threshold=sat_config.snr_threshold,
        )

    @require_sovereign_auth
    async def process_sovereign_intent(self, intent: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a sovereign intent through the dual agent system.

        This is the main entry point for autonomous BIZRA operation.
        """
        context = context or {}

        # Verify sovereignty
        sovereign_intent = SovereignIntent(
            intent=intent,
            context=context,
            urgency=context.get("urgency", "normal")
        )

        if not sovereign_intent.verify_sovereignty():
            return {
                "status": "rejected",
                "reason": "Sovereign authentication failed",
                "sovereign_required": True
            }

        # Create session
        session_id = f"sovereign_session_{self.session_counter}"
        self.session_counter += 1

        session = DualAgentSession(
            session_id=session_id,
            sovereign_user="MoMo"
        )
        self.active_sessions[session_id] = session

        try:
            # Phase 1: PAT interpretation
            pat_result = await self._pat_interpret_intent(sovereign_intent, session)

            if not pat_result["success"]:
                session.fail_session(pat_result.get("reason", "PAT interpretation failed"))
                return {
                    "status": "failed",
                    "phase": "pat_interpretation",
                    "reason": pat_result.get("reason"),
                    "session_id": session_id
                }

            # Phase 2: SAT verification and execution
            sat_result = await self._sat_execute_proposal(pat_result["proposal"], session)

            if not sat_result["success"]:
                session.fail_session(sat_result.get("reason", "SAT execution failed"))
                return {
                    "status": "failed",
                    "phase": "sat_execution",
                    "reason": sat_result.get("reason"),
                    "session_id": session_id
                }

            # Success
            session.complete_session()
            return {
                "status": "completed",
                "session_id": session_id,
                "pat_result": pat_result,
                "sat_result": sat_result,
                "sovereign_execution": "SUCCESS"
            }

        except Exception as e:
            session.fail_session(f"Exception: {str(e)}")
            logger.error(f"Sovereign intent processing failed: {e}")
            return {
                "status": "error",
                "session_id": session_id,
                "error": str(e)
            }

    async def _pat_interpret_intent(self, sovereign_intent: SovereignIntent, session: DualAgentSession) -> Dict[str, Any]:
        """PAT phase: Interpret sovereign intent into a formal proposal."""
        try:
            # PAT analyzes the intent and creates a proposal
            action_type = self._classify_intent(sovereign_intent.intent)
            data_payload = self._extract_payload_data(sovereign_intent.intent, sovereign_intent.context)

            # Calculate Ihsan for this proposal
            ihsan_score = self._calculate_proposal_ihsan(action_type, data_payload)

            # Create PCI proposal
            proposal_result = self.pat_agent.create_proposal(
                action=action_type,
                data=data_payload,
                policy_hash="sovereign_policy_v1",  # Would be actual policy hash
                ihsan_score=ihsan_score,
                snr_score=0.85,  # Conservative estimate
                urgency=sovereign_intent.urgency,
                extra_metadata={
                    "sovereign_intent": sovereign_intent.intent,
                    "session_id": session.session_id,
                    "phase": "pat_interpretation"
                }
            )

            if proposal_result.success:
                session.add_operation({
                    "phase": "pat_interpretation",
                    "action": action_type,
                    "ihsan_score": ihsan_score,
                    "envelope_id": proposal_result.envelope.envelope_id if proposal_result.envelope else None,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                return {
                    "success": True,
                    "proposal": proposal_result,
                    "action_type": action_type,
                    "ihsan_score": ihsan_score
                }
            else:
                reject_code = proposal_result.rejection.code if proposal_result.rejection else 'Unknown'
                return {
                    "success": False,
                    "reason": f"PAT proposal rejected: {reject_code}"
                }

        except Exception as e:
            logger.error(f"PAT interpretation failed: {e}")
            return {
                "success": False,
                "reason": f"PAT interpretation error: {str(e)}"
            }

    async def _sat_execute_proposal(self, proposal_result, session: DualAgentSession) -> Dict[str, Any]:
        """SAT phase: Verify and execute the PAT proposal."""
        try:
            envelope = proposal_result.envelope

            # SAT verifies the envelope
            verification_result = self.sat_agent.verify(envelope)

            if not verification_result.success:
                rejection_reason = verification_result.rejection.reject_code if verification_result.rejection else "Unknown SAT rejection"
                return {
                    "success": False,
                    "reason": f"SAT verification failed: {rejection_reason}"
                }

            # SAT commits the verified envelope
            session.add_operation({
                "phase": "sat_execution",
                "envelope_id": envelope.envelope_id,
                "verification_tier": verification_result.report.tier_reached if verification_result.report else "unknown",
                "commit_receipt": verification_result.receipt.receipt_id if verification_result.receipt else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return {
                "success": True,
                "verification": verification_result,
                "commit_receipt": verification_result.receipt.receipt_id if verification_result.receipt else None
            }

        except Exception as e:
            logger.error(f"SAT execution failed: {e}")
            return {
                "success": False,
                "reason": f"SAT execution error: {str(e)}"
            }

    def _classify_intent(self, intent: str) -> str:
        """Classify the sovereign intent into action types."""
        intent_lower = intent.lower()

        # Simple intent classification - would be more sophisticated in production
        if any(word in intent_lower for word in ["model", "llm", "ollama", "cuda"]):
            return "model_orchestration"
        elif any(word in intent_lower for word in ["data", "knowledge", "pipeline"]):
            return "data_processing"
        elif any(word in intent_lower for word in ["system", "resource", "hardware"]):
            return "system_management"
        elif any(word in intent_lower for word in ["network", "federation", "consensus"]):
            return "federation_control"
        else:
            return "general_sovereign_operation"

    def _extract_payload_data(self, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured payload data from intent and context."""
        # This would use NLP to extract structured data from natural language
        # Simplified for now
        return {
            "raw_intent": intent,
            "context": context,
            "sovereign_command": True,
            "extracted_parameters": {},  # Would be populated by NLP
        }

    def _calculate_proposal_ihsan(self, action_type: str, data: Dict[str, Any]) -> float:
        """Calculate Ihsan score for a proposal."""
        # Simplified Ihsan calculation - would use full kernel in production
        base_score = 0.85  # Conservative base

        # Adjust based on action type
        action_multipliers = {
            "model_orchestration": 1.0,
            "data_processing": 0.95,
            "system_management": 0.98,
            "federation_control": 0.90,
            "general_sovereign_operation": 0.88
        }

        multiplier = action_multipliers.get(action_type, 0.85)
        return min(1.0, base_score * multiplier)

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a dual agent session."""
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "status": session.status,
            "sovereign_user": session.sovereign_user,
            "start_time": session.start_time.isoformat(),
            "operation_count": len(session.operations),
            "last_operation": session.operations[-1] if session.operations else None
        }

    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active dual agent sessions."""
        return [
            self.get_session_status(session_id)
            for session_id in self.active_sessions.keys()
            if self.active_sessions[session_id].status == "active"
        ]

    def get_sovereign_manifest(self) -> Dict[str, Any]:
        """Get complete dual agent system manifest."""
        return {
            "system_type": "Sovereign Dual Agent System",
            "pat_agent": {
                "id": self.pat_agent.agent_id,
                "status": "active",
                "proposals_created": self.pat_agent.stats()["proposals_created"]
            },
            "sat_agent": {
                "id": self.sat_agent.agent_id,
                "status": "active",
                "envelopes_verified": self.sat_agent.stats()["envelopes_verified"]
            },
            "active_sessions": len(self.list_active_sessions()),
            "sovereign_mode": self.sovereign_mode,
            "last_verification": self.last_sovereign_verification,
            "integration_status": "PAT_SAT_HARMONY_ACTIVE"
        }


# Global dual agents instance
_dual_agents: Optional[SovereignDualAgents] = None


def get_sovereign_dual_agents() -> SovereignDualAgents:
    """Get the global sovereign dual agents instance."""
    global _dual_agents
    if _dual_agents is None:
        _dual_agents = SovereignDualAgents()
    return _dual_agents


async def execute_sovereign_intent(intent: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a sovereign intent through the dual agent system."""
    dual_agents = get_sovereign_dual_agents()
    return await dual_agents.process_sovereign_intent(intent, context)
