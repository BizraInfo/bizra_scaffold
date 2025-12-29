"""
BIZRA AEON OMEGA - Genesis Council Department Schema
Node0 Exclusive: 7 Departments x 7 Agents + Supervisors

This schema defines the 3rd layer unique to Genesis Block:
- 7 Departments (D1-D7)
- 7 Agents per department (49 total)
- 7 Alpha Meta-Managers (one per department)
- 1 Boss Agent (Council Head)
= 57 Genesis-exclusive agents
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Ihsan thresholds for Genesis Council
AGENT_IHSAN_MINIMUM = 0.95
ALPHA_IHSAN_MINIMUM = 0.96
BOSS_IHSAN_MINIMUM = 0.98


class DepartmentID(Enum):
    """Genesis Council 7 departments."""
    D1_CRYPTOGRAPHY = "cryptography"   # Security, keys, proofs, ZK
    D2_ECONOMICS = "economics"         # Tokenomics, PoI, staking
    D3_PHILOSOPHY = "philosophy"       # Ethics, Ihsan, values
    D4_GOVERNANCE = "governance"       # FATE Engine, voting, constitution
    D5_SYSTEMS = "systems"             # Architecture, infra, scaling
    D6_COGNITIVE = "cognitive"         # Memory, learning, reasoning
    D7_OPERATIONS = "operations"       # Monitoring, health, DevOps


class AgentRole(Enum):
    """Agent specializations within each department."""
    RESEARCHER = auto()    # Domain exploration
    DEVELOPER = auto()     # Implementation
    VALIDATOR = auto()     # Quality assurance
    OPTIMIZER = auto()     # Performance tuning
    DOCUMENTER = auto()    # Knowledge crystallization
    TESTER = auto()        # Verification and edge cases
    SECURITY = auto()      # Threat modeling and hardening


class AgentLevel(Enum):
    """Hierarchical level within Genesis Council."""
    AGENT = auto()         # Regular department agent (49)
    ALPHA = auto()         # Department meta-manager (7)
    BOSS = auto()          # Council head (1)


@dataclass
class AgentCapability:
    """Capability of an agent."""
    name: str
    description: str
    proficiency: float  # 0.0 to 1.0
    required_tools: List[str] = field(default_factory=list)


@dataclass
class DepartmentConfig:
    """Configuration for a Genesis Council department."""
    id: DepartmentID
    name_arabic: str  # Arabic name for Ihsan alignment
    description: str
    primary_focus: str
    secondary_focus: List[str]
    recommended_models: List[str]
    agent_count: int = 7
    ihsan_threshold: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id.value,
            "name_arabic": self.name_arabic,
            "description": self.description,
            "primary_focus": self.primary_focus,
            "secondary_focus": self.secondary_focus,
            "recommended_models": self.recommended_models,
            "agent_count": self.agent_count,
            "ihsan_threshold": self.ihsan_threshold,
        }


@dataclass
class GenesisAgent:
    """Definition of a Genesis Council agent."""
    agent_id: str
    department: DepartmentID
    role: AgentRole
    level: AgentLevel
    name: str
    description: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    ihsan_score: float = 0.95
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def meets_ihsan_threshold(self) -> bool:
        """Check if agent meets Ihsan threshold for its level."""
        if self.level == AgentLevel.BOSS:
            return self.ihsan_score >= BOSS_IHSAN_MINIMUM
        elif self.level == AgentLevel.ALPHA:
            return self.ihsan_score >= ALPHA_IHSAN_MINIMUM
        else:
            return self.ihsan_score >= AGENT_IHSAN_MINIMUM


# Genesis Council Department Definitions
DEPARTMENT_CONFIGS: Dict[DepartmentID, DepartmentConfig] = {
    DepartmentID.D1_CRYPTOGRAPHY: DepartmentConfig(
        id=DepartmentID.D1_CRYPTOGRAPHY,
        name_arabic="التشفير",
        description="Cryptographic security, key management, zero-knowledge proofs",
        primary_focus="Post-quantum cryptography implementation",
        secondary_focus=["Key rotation", "ZK circuits", "HSM integration"],
        recommended_models=["deepseek-coder:14b", "codellama:13b"],
    ),
    DepartmentID.D2_ECONOMICS: DepartmentConfig(
        id=DepartmentID.D2_ECONOMICS,
        name_arabic="الاقتصاد",
        description="Tokenomics, Proof of Impact, staking mechanisms",
        primary_focus="Sharia-compliant economic modeling",
        secondary_focus=["PoI calculation", "Token velocity", "Staking rewards"],
        recommended_models=["qwen:14b", "mistral:7b"],
    ),
    DepartmentID.D3_PHILOSOPHY: DepartmentConfig(
        id=DepartmentID.D3_PHILOSOPHY,
        name_arabic="الفلسفة",
        description="Ethics, Ihsan principles, value alignment",
        primary_focus="Maqasid al-Shariah enforcement",
        secondary_focus=["Ethical reasoning", "Value alignment", "Bias detection"],
        recommended_models=["llama3.1:8b", "phi3:14b"],
    ),
    DepartmentID.D4_GOVERNANCE: DepartmentConfig(
        id=DepartmentID.D4_GOVERNANCE,
        name_arabic="الحوكمة",
        description="FATE Engine, constitutional voting, policy enforcement",
        primary_focus="Z3 SMT formal verification",
        secondary_focus=["Voting mechanisms", "Constitution updates", "Dispute resolution"],
        recommended_models=["mistral:7b", "llama3.1:8b"],
    ),
    DepartmentID.D5_SYSTEMS: DepartmentConfig(
        id=DepartmentID.D5_SYSTEMS,
        name_arabic="الأنظمة",
        description="Architecture, infrastructure, horizontal scaling",
        primary_focus="Distributed systems design",
        secondary_focus=["Consensus protocols", "Sharding", "Service mesh"],
        recommended_models=["codegemma:7b", "deepseek-coder:14b"],
    ),
    DepartmentID.D6_COGNITIVE: DepartmentConfig(
        id=DepartmentID.D6_COGNITIVE,
        name_arabic="الإدراك",
        description="Memory systems, learning, reasoning engines",
        primary_focus="Graph-of-Thoughts implementation",
        secondary_focus=["5-tier memory", "Knowledge crystallization", "SNR scoring"],
        recommended_models=["phi3:14b", "qwen:14b"],
    ),
    DepartmentID.D7_OPERATIONS: DepartmentConfig(
        id=DepartmentID.D7_OPERATIONS,
        name_arabic="العمليات",
        description="Monitoring, health checks, DevOps automation",
        primary_focus="Observability and incident response",
        secondary_focus=["Chaos engineering", "SLA enforcement", "Capacity planning"],
        recommended_models=["llama3.1:8b", "mistral:7b"],
    ),
}


def create_department_agents(department: DepartmentID) -> List[GenesisAgent]:
    """Create the 7 agents for a department."""
    config = DEPARTMENT_CONFIGS[department]
    agents = []
    
    for i, role in enumerate(AgentRole, start=1):
        agent = GenesisAgent(
            agent_id=f"{department.value}-agent-{i:02d}",
            department=department,
            role=role,
            level=AgentLevel.AGENT,
            name=f"{department.value.title()} {role.name.title()} Agent",
            description=f"{role.name.title()} specialist for {config.description}",
            ihsan_score=0.95,
        )
        agents.append(agent)
    
    return agents


def create_alpha_manager(department: DepartmentID) -> GenesisAgent:
    """Create the Alpha Meta-Manager for a department."""
    config = DEPARTMENT_CONFIGS[department]
    
    return GenesisAgent(
        agent_id=f"{department.value}-alpha",
        department=department,
        role=AgentRole.RESEARCHER,  # Alphas are generalists
        level=AgentLevel.ALPHA,
        name=f"{config.name_arabic} Alpha Manager",
        description=f"Meta-manager coordinating 7 agents in {config.description}",
        ihsan_score=0.96,
        metadata={"team_size": 7, "arabic_name": config.name_arabic},
    )


def create_boss_agent() -> GenesisAgent:
    """Create the Boss Agent (Council Head)."""
    return GenesisAgent(
        agent_id="genesis-boss",
        department=DepartmentID.D4_GOVERNANCE,  # Boss oversees governance
        role=AgentRole.RESEARCHER,
        level=AgentLevel.BOSS,
        name="Genesis Council Boss",
        description="Council Head coordinating 7 Alpha managers and 49 department agents",
        ihsan_score=0.98,
        metadata={
            "team_size": 57,
            "direct_reports": 7,
            "arabic_title": "رئيس المجلس",
        },
    )


class GenesisCouncil:
    """
    The complete Genesis Council structure.
    
    Node0 exclusive: 7 departments x 7 agents + 7 alphas + 1 boss = 57 agents
    """
    
    def __init__(self):
        self.departments: Dict[DepartmentID, DepartmentConfig] = DEPARTMENT_CONFIGS
        self.agents: Dict[str, GenesisAgent] = {}
        self.alphas: Dict[DepartmentID, GenesisAgent] = {}
        self.boss: Optional[GenesisAgent] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the complete Genesis Council."""
        if self._initialized:
            return
        
        # Create all department agents
        for department in DepartmentID:
            agents = create_department_agents(department)
            for agent in agents:
                self.agents[agent.agent_id] = agent
            
            # Create alpha for each department
            alpha = create_alpha_manager(department)
            self.alphas[department] = alpha
            self.agents[alpha.agent_id] = alpha
        
        # Create boss
        self.boss = create_boss_agent()
        self.agents[self.boss.agent_id] = self.boss
        
        self._initialized = True
        
        logger.info(
            f"Genesis Council initialized: {len(self.agents)} agents "
            f"({len(self.alphas)} alphas, 1 boss, {len(self.agents) - len(self.alphas) - 1} department agents)"
        )
    
    @property
    def total_agents(self) -> int:
        return len(self.agents)
    
    @property
    def department_count(self) -> int:
        return len(self.departments)
    
    def get_department_team(self, department: DepartmentID) -> List[GenesisAgent]:
        """Get all agents in a department including alpha."""
        return [
            agent for agent in self.agents.values()
            if agent.department == department
        ]
    
    def get_agents_by_role(self, role: AgentRole) -> List[GenesisAgent]:
        """Get all agents with a specific role."""
        return [
            agent for agent in self.agents.values()
            if agent.role == role and agent.level == AgentLevel.AGENT
        ]
    
    def validate_ihsan_compliance(self) -> Dict[str, bool]:
        """Validate Ihsan compliance for all agents."""
        return {
            agent_id: agent.meets_ihsan_threshold
            for agent_id, agent in self.agents.items()
        }


# Singleton instance
_genesis_council: Optional[GenesisCouncil] = None


def get_genesis_council() -> GenesisCouncil:
    """Get or create the Genesis Council singleton."""
    global _genesis_council
    if _genesis_council is None:
        _genesis_council = GenesisCouncil()
        _genesis_council.initialize()
    return _genesis_council
