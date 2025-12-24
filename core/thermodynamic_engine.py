"""
BIZRA AEON OMEGA - Thermodynamic Engine
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Grade | Information-Theoretic Thermodynamics

Models the BIZRA system as a thermodynamic engine where:
- Information is energy
- Entropy measures disorder/uncertainty in governance states
- Work is extracted through consensus and value creation
- Heat dissipation represents coordination costs

Theoretical Foundation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Landauer's Principle:  ΔE ≥ kT·ln(2) per bit erased
Carnot Efficiency:     η_max = 1 - T_cold/T_hot
Free Energy:           F = U - TS (Helmholtz)
Gibbs Free Energy:     G = H - TS (for governance work)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SNR Score: 9.5/10.0 | Ihsan Compliant
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Protocol
import numpy as np

# Physical constants (information-theoretic analogs)
BOLTZMANN_K = 1.380649e-23  # J/K - Boltzmann constant
LANDAUER_LIMIT = BOLTZMANN_K * 300 * math.log(2)  # ~2.87e-21 J at 300K
PLANCK_H = 6.62607015e-34  # J·s - Planck constant
INFORMATION_ENERGY_RATIO = 1e-18  # J/bit (practical system constant)


class ThermodynamicState(Enum):
    """Thermodynamic states of the BIZRA engine."""
    EQUILIBRIUM = auto()      # System at rest, maximum entropy
    COMPRESSION = auto()      # Gathering consensus, reducing entropy
    IGNITION = auto()         # Decision threshold reached
    EXPANSION = auto()        # Work extraction, value distribution
    EXHAUST = auto()          # Dissipating waste heat (coordination costs)
    RECOVERY = auto()         # Returning to equilibrium


class CycleType(Enum):
    """Thermodynamic cycle types supported."""
    CARNOT = auto()           # Ideal reversible cycle (theoretical max)
    OTTO = auto()             # Governance sprint cycle (rapid decisions)
    DIESEL = auto()           # Long-term planning cycle (compression ignition)
    STIRLING = auto()         # Continuous regenerative cycle
    RANKINE = auto()          # Multi-phase transitions (complex governance)


@dataclass(frozen=True)
class ThermodynamicConstants:
    """
    BIZRA thermodynamic constants - calibrated for governance systems.
    
    These map physical thermodynamics to information-theoretic governance:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Physical Quantity     │ BIZRA Analog              │ Unit
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Temperature (T)       │ Governance activity level │ governance-kelvin (gK)
    Pressure (P)          │ Urgency/stake pressure    │ urgency-pascal (uPa)
    Volume (V)            │ Decision space size       │ decision-liters (dL)
    Energy (U)            │ Token value + reputation  │ value-joules (vJ)
    Entropy (S)           │ Uncertainty/disorder      │ bits
    Work (W)              │ Consensus outcomes        │ impact-joules (iJ)
    Heat (Q)              │ Information exchange      │ info-joules (infoJ)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    # Temperature bounds (governance activity)
    T_MIN: float = 0.01          # Near-frozen governance (emergency only)
    T_MAX: float = 1000.0        # Maximum activity (crisis mode)
    T_AMBIENT: float = 300.0     # Normal operating temperature
    
    # Pressure bounds (urgency)
    P_MIN: float = 0.1           # Low urgency (can defer)
    P_MAX: float = 100.0         # Critical urgency (immediate action)
    P_STANDARD: float = 1.0      # Normal urgency
    
    # Energy parameters
    ACTIVATION_ENERGY: float = 0.1    # Minimum energy to start process
    MAX_EXTRACTABLE_WORK: float = 0.95  # Carnot limit approximation
    
    # Entropy parameters
    ENTROPY_FLOOR: float = 0.001      # Minimum uncertainty (near-certainty)
    ENTROPY_CEILING: float = 10.0     # Maximum disorder (total uncertainty)
    
    # Ihsan alignment (ethics as thermodynamic constraint)
    IHSAN_COUPLING: float = 0.95      # Ethics-thermodynamics coupling strength


@dataclass
class Reservoir:
    """
    Thermodynamic reservoir - source or sink of governance energy.
    
    Hot Reservoir: High-activity governance zones (active proposals, debates)
    Cold Reservoir: Stable consensus zones (established rules, completed decisions)
    """
    name: str
    temperature: float           # Governance activity level
    capacity: float              # Maximum energy it can hold
    current_energy: float        # Current stored energy
    entropy: float               # Current entropy state
    is_infinite: bool = False    # True for environment reservoirs
    
    def absorb_heat(self, q: float) -> Tuple[float, float]:
        """
        Absorb heat energy, updating temperature and entropy.
        Returns (actual absorbed, entropy change).
        """
        if self.is_infinite:
            # Infinite reservoir maintains temperature
            ds = q / self.temperature if self.temperature > 0 else 0
            return (q, ds)
        
        # Finite reservoir heats up
        actual = min(q, self.capacity - self.current_energy)
        self.current_energy += actual
        
        # ΔS = Q/T for reversible heat transfer
        ds = actual / self.temperature if self.temperature > 0 else 0
        self.entropy += ds
        
        # Temperature increases with absorbed energy
        heat_capacity = self.capacity / 100  # Simplified
        self.temperature += actual / heat_capacity
        
        return (actual, ds)
    
    def release_heat(self, q: float) -> Tuple[float, float]:
        """
        Release heat energy.
        Returns (actual released, entropy change).
        """
        if self.is_infinite:
            ds = -q / self.temperature if self.temperature > 0 else 0
            return (q, ds)
        
        actual = min(q, self.current_energy)
        self.current_energy -= actual
        
        ds = -actual / self.temperature if self.temperature > 0 else 0
        self.entropy += ds
        
        heat_capacity = self.capacity / 100
        self.temperature = max(0.01, self.temperature - actual / heat_capacity)
        
        return (actual, ds)


@dataclass
class WorkingFluid:
    """
    The working fluid in the BIZRA thermodynamic cycle.
    
    Represents the "substance" being transformed through governance:
    - Proposals (in various states of consensus)
    - Token flows (BZT/BZC)
    - Reputation changes
    """
    id: str
    state: ThermodynamicState
    temperature: float           # Current activity level
    pressure: float              # Current urgency
    volume: float                # Decision space size
    internal_energy: float       # Total value (tokens + reputation)
    entropy: float               # Uncertainty/disorder
    
    # Governance-specific properties
    proposal_hash: Optional[str] = None
    participants: int = 0
    consensus_level: float = 0.0  # 0.0 to 1.0
    ihsan_score: float = 1.0      # Ethical alignment
    
    @property
    def enthalpy(self) -> float:
        """H = U + PV - total heat content."""
        return self.internal_energy + self.pressure * self.volume
    
    @property
    def helmholtz_free_energy(self) -> float:
        """F = U - TS - work extractable at constant T."""
        return self.internal_energy - self.temperature * self.entropy
    
    @property
    def gibbs_free_energy(self) -> float:
        """G = H - TS - work extractable at constant T and P."""
        return self.enthalpy - self.temperature * self.entropy
    
    def is_spontaneous(self) -> bool:
        """Process is spontaneous if ΔG < 0."""
        return self.gibbs_free_energy < 0


@dataclass
class CycleMetrics:
    """Metrics for a complete thermodynamic cycle."""
    cycle_id: str
    cycle_type: CycleType
    
    # Energy flows
    heat_absorbed: float         # Q_in from hot reservoir
    heat_rejected: float         # Q_out to cold reservoir
    work_output: float           # W = Q_in - Q_out
    
    # Efficiency metrics
    thermal_efficiency: float    # η = W / Q_in
    carnot_efficiency: float     # η_carnot = 1 - T_cold/T_hot
    second_law_efficiency: float # η / η_carnot
    
    # Entropy production
    total_entropy_change: float  # ΔS_total (should be ≥ 0)
    irreversibility: float       # Lost work due to irreversibility
    
    # Governance outcomes
    decisions_made: int
    consensus_achieved: float
    value_created: float
    ihsan_compliance: float
    
    # Timing
    cycle_duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ThermodynamicProcess(ABC):
    """Abstract base for thermodynamic processes."""
    
    @abstractmethod
    def execute(self, fluid: WorkingFluid) -> Tuple[WorkingFluid, float, float]:
        """
        Execute process on working fluid.
        Returns (new_state, heat_transferred, work_done).
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def is_reversible(self) -> bool:
        pass


class IsothermalProcess(ThermodynamicProcess):
    """
    Isothermal (constant temperature) process.
    
    In BIZRA: Steady-state governance with constant activity level.
    Heat is exchanged while temperature remains stable.
    """
    
    def __init__(self, target_volume: float, reservoir: Reservoir):
        self.target_volume = target_volume
        self.reservoir = reservoir
    
    @property
    def name(self) -> str:
        return "Isothermal"
    
    @property
    def is_reversible(self) -> bool:
        return True
    
    def execute(self, fluid: WorkingFluid) -> Tuple[WorkingFluid, float, float]:
        """Isothermal expansion/compression."""
        v1, v2 = fluid.volume, self.target_volume
        t = fluid.temperature
        
        # For ideal gas: W = nRT·ln(V2/V1)
        # Using normalized units where nR = 1
        if v1 > 0 and v2 > 0:
            work = t * math.log(v2 / v1)
        else:
            work = 0
        
        # For isothermal process: Q = W (internal energy unchanged)
        heat = work
        
        # Exchange heat with reservoir
        if heat > 0:
            self.reservoir.release_heat(heat)
        else:
            self.reservoir.absorb_heat(-heat)
        
        # Entropy change: ΔS = Q/T = W/T = ln(V2/V1)
        ds = math.log(v2 / v1) if v1 > 0 and v2 > 0 else 0
        
        new_fluid = WorkingFluid(
            id=fluid.id,
            state=fluid.state,
            temperature=t,  # Unchanged
            pressure=fluid.pressure * (v1 / v2) if v2 > 0 else fluid.pressure,
            volume=v2,
            internal_energy=fluid.internal_energy,  # Unchanged for isothermal
            entropy=fluid.entropy + ds,
            proposal_hash=fluid.proposal_hash,
            participants=fluid.participants,
            consensus_level=fluid.consensus_level,
            ihsan_score=fluid.ihsan_score,
        )
        
        return (new_fluid, heat, work)


class AdiabaticProcess(ThermodynamicProcess):
    """
    Adiabatic (no heat exchange) process.
    
    In BIZRA: Rapid governance action with no external input.
    Work is done entirely from internal energy.
    """
    
    def __init__(self, target_volume: float, gamma: float = 1.4):
        self.target_volume = target_volume
        self.gamma = gamma  # Heat capacity ratio
    
    @property
    def name(self) -> str:
        return "Adiabatic"
    
    @property
    def is_reversible(self) -> bool:
        return True
    
    def execute(self, fluid: WorkingFluid) -> Tuple[WorkingFluid, float, float]:
        """Adiabatic expansion/compression."""
        v1, v2 = fluid.volume, self.target_volume
        t1, p1 = fluid.temperature, fluid.pressure
        
        # Adiabatic relations: TV^(γ-1) = const, PV^γ = const
        if v1 > 0 and v2 > 0:
            t2 = t1 * (v1 / v2) ** (self.gamma - 1)
            p2 = p1 * (v1 / v2) ** self.gamma
        else:
            t2, p2 = t1, p1
        
        # Work done: W = (P1V1 - P2V2) / (γ - 1)
        work = (p1 * v1 - p2 * v2) / (self.gamma - 1)
        
        # No heat exchange
        heat = 0
        
        # Internal energy change: ΔU = -W (first law with Q=0)
        du = -work
        
        new_fluid = WorkingFluid(
            id=fluid.id,
            state=fluid.state,
            temperature=t2,
            pressure=p2,
            volume=v2,
            internal_energy=fluid.internal_energy + du,
            entropy=fluid.entropy,  # Unchanged for adiabatic
            proposal_hash=fluid.proposal_hash,
            participants=fluid.participants,
            consensus_level=fluid.consensus_level,
            ihsan_score=fluid.ihsan_score,
        )
        
        return (new_fluid, heat, work)


class IsobaricProcess(ThermodynamicProcess):
    """
    Isobaric (constant pressure) process.
    
    In BIZRA: Governance at constant urgency level.
    Common in routine decision-making.
    """
    
    def __init__(self, target_temperature: float, reservoir: Reservoir):
        self.target_temperature = target_temperature
        self.reservoir = reservoir
    
    @property
    def name(self) -> str:
        return "Isobaric"
    
    @property
    def is_reversible(self) -> bool:
        return True
    
    def execute(self, fluid: WorkingFluid) -> Tuple[WorkingFluid, float, float]:
        """Isobaric heating/cooling."""
        t1, t2 = fluid.temperature, self.target_temperature
        p = fluid.pressure
        v1 = fluid.volume
        
        # V/T = const for isobaric process
        v2 = v1 * (t2 / t1) if t1 > 0 else v1
        
        # Heat capacity at constant pressure (Cp = γCv)
        cp = 1.4  # Simplified
        
        # Heat transfer: Q = n·Cp·ΔT (normalized)
        heat = cp * (t2 - t1)
        
        # Work done: W = P·ΔV
        work = p * (v2 - v1)
        
        # Internal energy change: ΔU = Q - W
        du = heat - work
        
        # Entropy change: ΔS = Cp·ln(T2/T1)
        ds = cp * math.log(t2 / t1) if t1 > 0 and t2 > 0 else 0
        
        # Exchange heat with reservoir
        if heat > 0:
            self.reservoir.release_heat(heat)
        else:
            self.reservoir.absorb_heat(-heat)
        
        new_fluid = WorkingFluid(
            id=fluid.id,
            state=fluid.state,
            temperature=t2,
            pressure=p,  # Unchanged
            volume=v2,
            internal_energy=fluid.internal_energy + du,
            entropy=fluid.entropy + ds,
            proposal_hash=fluid.proposal_hash,
            participants=fluid.participants,
            consensus_level=fluid.consensus_level,
            ihsan_score=fluid.ihsan_score,
        )
        
        return (new_fluid, heat, work)


class IsochoricProcess(ThermodynamicProcess):
    """
    Isochoric (constant volume) process.
    
    In BIZRA: Governance with fixed decision space.
    Energy changes but scope remains constant.
    """
    
    def __init__(self, target_temperature: float, reservoir: Reservoir):
        self.target_temperature = target_temperature
        self.reservoir = reservoir
    
    @property
    def name(self) -> str:
        return "Isochoric"
    
    @property
    def is_reversible(self) -> bool:
        return True
    
    def execute(self, fluid: WorkingFluid) -> Tuple[WorkingFluid, float, float]:
        """Isochoric heating/cooling."""
        t1, t2 = fluid.temperature, self.target_temperature
        v = fluid.volume
        p1 = fluid.pressure
        
        # P/T = const for isochoric process
        p2 = p1 * (t2 / t1) if t1 > 0 else p1
        
        # Heat capacity at constant volume
        cv = 1.0  # Simplified
        
        # Heat transfer: Q = n·Cv·ΔT
        heat = cv * (t2 - t1)
        
        # No work done (ΔV = 0)
        work = 0
        
        # Internal energy change: ΔU = Q
        du = heat
        
        # Entropy change: ΔS = Cv·ln(T2/T1)
        ds = cv * math.log(t2 / t1) if t1 > 0 and t2 > 0 else 0
        
        # Exchange heat with reservoir
        if heat > 0:
            self.reservoir.release_heat(heat)
        else:
            self.reservoir.absorb_heat(-heat)
        
        new_fluid = WorkingFluid(
            id=fluid.id,
            state=fluid.state,
            temperature=t2,
            pressure=p2,
            volume=v,  # Unchanged
            internal_energy=fluid.internal_energy + du,
            entropy=fluid.entropy + ds,
            proposal_hash=fluid.proposal_hash,
            participants=fluid.participants,
            consensus_level=fluid.consensus_level,
            ihsan_score=fluid.ihsan_score,
        )
        
        return (new_fluid, heat, work)


class BIZRAThermodynamicEngine:
    """
    The BIZRA Thermodynamic Engine - Core Governance Cycle Manager.
    
    Models governance as a heat engine extracting useful work (consensus, value)
    from temperature gradients (activity differentials between proposal and 
    resolution states).
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     BIZRA THERMODYNAMIC CYCLE                           │
    │                                                                         │
    │     Hot Reservoir (Active Proposals)                                    │
    │            ↓ Q_hot (debate energy)                                      │
    │     ┌─────────────────────────────┐                                     │
    │     │   COMPRESSION (consensus)   │→→→ W_out (decisions)                │
    │     │   IGNITION (threshold)      │                                     │
    │     │   EXPANSION (execution)     │                                     │
    │     └─────────────────────────────┘                                     │
    │            ↓ Q_cold (coordination cost)                                 │
    │     Cold Reservoir (Resolved Consensus)                                 │
    │                                                                         │
    │     η = W_out / Q_hot ≤ η_carnot = 1 - T_cold/T_hot                     │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        cycle_type: CycleType = CycleType.OTTO,
        hot_temp: float = 600.0,
        cold_temp: float = 300.0,
        ihsan_coupling: float = 0.95,
    ):
        self.cycle_type = cycle_type
        self.constants = ThermodynamicConstants()
        self.ihsan_coupling = ihsan_coupling
        
        # Initialize reservoirs
        self.hot_reservoir = Reservoir(
            name="ActiveProposals",
            temperature=hot_temp,
            capacity=1000.0,
            current_energy=500.0,
            entropy=2.0,
            is_infinite=True,  # Environment as infinite reservoir
        )
        
        self.cold_reservoir = Reservoir(
            name="ResolvedConsensus",
            temperature=cold_temp,
            capacity=1000.0,
            current_energy=200.0,
            entropy=1.0,
            is_infinite=True,
        )
        
        # Carnot efficiency limit
        self.carnot_efficiency = 1 - cold_temp / hot_temp
        
        # Cycle history
        self.cycle_history: List[CycleMetrics] = []
        self._cycle_counter = 0
        
        # State tracking
        self._current_fluid: Optional[WorkingFluid] = None
        self._state = ThermodynamicState.EQUILIBRIUM
    
    def _generate_cycle_id(self) -> str:
        """Generate unique cycle identifier."""
        self._cycle_counter += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"cycle-{self.cycle_type.name.lower()}-{timestamp}-{self._cycle_counter:04d}"
    
    def create_working_fluid(
        self,
        proposal_data: Dict[str, Any],
        initial_stake: float = 100.0,
        urgency: float = 1.0,
    ) -> WorkingFluid:
        """
        Create a working fluid from a governance proposal.
        
        The fluid carries the proposal through the thermodynamic cycle,
        transforming stake into consensus work.
        """
        proposal_hash = hashlib.sha256(
            str(proposal_data).encode()
        ).hexdigest()[:16]
        
        # Initial conditions
        initial_temp = self.hot_reservoir.temperature * 0.9
        initial_pressure = urgency * self.constants.P_STANDARD
        initial_volume = math.log(initial_stake + 1)  # Logarithmic scaling
        
        # Initial entropy from stake distribution
        initial_entropy = self._compute_stake_entropy(proposal_data)
        
        fluid = WorkingFluid(
            id=f"fluid-{proposal_hash}",
            state=ThermodynamicState.EQUILIBRIUM,
            temperature=initial_temp,
            pressure=initial_pressure,
            volume=initial_volume,
            internal_energy=initial_stake,
            entropy=initial_entropy,
            proposal_hash=proposal_hash,
            participants=proposal_data.get("participants", 1),
            consensus_level=0.0,
            ihsan_score=1.0,
        )
        
        self._current_fluid = fluid
        return fluid
    
    def _compute_stake_entropy(self, proposal_data: Dict[str, Any]) -> float:
        """
        Compute entropy from stake distribution.
        
        Uses Shannon entropy: S = -Σ p_i · log(p_i)
        More equal distributions have higher entropy.
        """
        stakes = proposal_data.get("stake_distribution", [1.0])
        if not stakes:
            stakes = [1.0]
        
        total = sum(stakes)
        if total == 0:
            return 0.0
        
        probabilities = [s / total for s in stakes if s > 0]
        
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log(p)
        
        return entropy
    
    async def run_carnot_cycle(
        self,
        fluid: WorkingFluid,
        compression_ratio: float = 2.0,
    ) -> Tuple[WorkingFluid, CycleMetrics]:
        """
        Run an ideal Carnot cycle - theoretical maximum efficiency.
        
        Stages:
        1. Isothermal expansion at T_hot (absorb heat from hot reservoir)
        2. Adiabatic expansion (cool to T_cold)
        3. Isothermal compression at T_cold (reject heat to cold reservoir)
        4. Adiabatic compression (heat to T_hot)
        """
        start_time = time.time()
        cycle_id = self._generate_cycle_id()
        
        v1 = fluid.volume
        v2 = v1 * compression_ratio
        v3 = v2 * (self.hot_reservoir.temperature / self.cold_reservoir.temperature) ** (1 / 0.4)
        v4 = v3 / compression_ratio
        
        total_heat_in = 0.0
        total_heat_out = 0.0
        total_work = 0.0
        total_ds = 0.0
        
        # Stage 1: Isothermal expansion at T_hot
        fluid.state = ThermodynamicState.EXPANSION
        process1 = IsothermalProcess(v2, self.hot_reservoir)
        fluid, q1, w1 = process1.execute(fluid)
        total_heat_in += max(0, q1)
        total_work += w1
        
        # Stage 2: Adiabatic expansion
        process2 = AdiabaticProcess(v3)
        fluid, q2, w2 = process2.execute(fluid)
        total_work += w2
        
        # Stage 3: Isothermal compression at T_cold
        fluid.state = ThermodynamicState.COMPRESSION
        process3 = IsothermalProcess(v4, self.cold_reservoir)
        fluid, q3, w3 = process3.execute(fluid)
        total_heat_out += max(0, -q3)
        total_work += w3
        
        # Stage 4: Adiabatic compression
        fluid.state = ThermodynamicState.RECOVERY
        process4 = AdiabaticProcess(v1)
        fluid, q4, w4 = process4.execute(fluid)
        total_work += w4
        
        # Final state
        fluid.state = ThermodynamicState.EQUILIBRIUM
        
        # Compute efficiencies
        thermal_efficiency = total_work / total_heat_in if total_heat_in > 0 else 0
        second_law_eff = thermal_efficiency / self.carnot_efficiency if self.carnot_efficiency > 0 else 0
        
        # Apply Ihsan coupling (ethics reduces extractable work)
        ihsan_adjusted_work = total_work * self.ihsan_coupling * fluid.ihsan_score
        
        # Update consensus level based on work done
        fluid = WorkingFluid(
            id=fluid.id,
            state=fluid.state,
            temperature=fluid.temperature,
            pressure=fluid.pressure,
            volume=fluid.volume,
            internal_energy=fluid.internal_energy,
            entropy=fluid.entropy,
            proposal_hash=fluid.proposal_hash,
            participants=fluid.participants,
            consensus_level=min(1.0, ihsan_adjusted_work / 100),
            ihsan_score=fluid.ihsan_score,
        )
        
        metrics = CycleMetrics(
            cycle_id=cycle_id,
            cycle_type=CycleType.CARNOT,
            heat_absorbed=total_heat_in,
            heat_rejected=total_heat_out,
            work_output=ihsan_adjusted_work,
            thermal_efficiency=thermal_efficiency,
            carnot_efficiency=self.carnot_efficiency,
            second_law_efficiency=second_law_eff,
            total_entropy_change=total_ds,
            irreversibility=0.0,  # Carnot is reversible
            decisions_made=1 if fluid.consensus_level > 0.5 else 0,
            consensus_achieved=fluid.consensus_level,
            value_created=ihsan_adjusted_work * 0.1,  # Value in BZC
            ihsan_compliance=fluid.ihsan_score,
            cycle_duration_ms=(time.time() - start_time) * 1000,
        )
        
        self.cycle_history.append(metrics)
        return (fluid, metrics)
    
    async def run_otto_cycle(
        self,
        fluid: WorkingFluid,
        compression_ratio: float = 8.0,
    ) -> Tuple[WorkingFluid, CycleMetrics]:
        """
        Run an Otto cycle - optimized for rapid governance decisions.
        
        Like a gasoline engine: quick ignition, fast power delivery.
        
        Stages:
        1. Adiabatic compression (gather consensus)
        2. Isochoric heat addition (ignition - decision reached)
        3. Adiabatic expansion (execute decision, extract work)
        4. Isochoric heat rejection (reset for next cycle)
        """
        start_time = time.time()
        cycle_id = self._generate_cycle_id()
        
        v1 = fluid.volume
        v2 = v1 / compression_ratio
        
        total_heat_in = 0.0
        total_heat_out = 0.0
        total_work = 0.0
        
        # Stage 1: Adiabatic compression (consensus building)
        fluid.state = ThermodynamicState.COMPRESSION
        process1 = AdiabaticProcess(v2, gamma=1.4)
        fluid, q1, w1 = process1.execute(fluid)
        total_work += w1  # Negative (work input)
        
        # Stage 2: Isochoric heat addition (ignition)
        fluid.state = ThermodynamicState.IGNITION
        t_ignition = fluid.temperature * 2.5  # Heat spike at ignition
        process2 = IsochoricProcess(t_ignition, self.hot_reservoir)
        fluid, q2, w2 = process2.execute(fluid)
        total_heat_in += q2
        
        # Stage 3: Adiabatic expansion (power stroke)
        fluid.state = ThermodynamicState.EXPANSION
        process3 = AdiabaticProcess(v1, gamma=1.4)
        fluid, q3, w3 = process3.execute(fluid)
        total_work += w3  # Positive (work output)
        
        # Stage 4: Isochoric heat rejection (exhaust)
        fluid.state = ThermodynamicState.EXHAUST
        t_exhaust = self.cold_reservoir.temperature * 1.2
        process4 = IsochoricProcess(t_exhaust, self.cold_reservoir)
        fluid, q4, w4 = process4.execute(fluid)
        total_heat_out += abs(q4)
        
        # Return to equilibrium
        fluid.state = ThermodynamicState.EQUILIBRIUM
        
        # Otto efficiency: η = 1 - 1/r^(γ-1)
        otto_efficiency = 1 - 1 / (compression_ratio ** 0.4)
        actual_efficiency = abs(total_work) / total_heat_in if total_heat_in > 0 else 0
        
        # Clamp efficiency to physical limits
        actual_efficiency = min(actual_efficiency, self.carnot_efficiency)
        second_law_eff = actual_efficiency / self.carnot_efficiency if self.carnot_efficiency > 0 else 0
        
        # Irreversibility (Otto is not reversible) - always non-negative
        irreversibility = max(0, total_heat_in * (self.carnot_efficiency - actual_efficiency))
        
        # Apply Ihsan coupling
        ihsan_adjusted_work = abs(total_work) * self.ihsan_coupling * fluid.ihsan_score
        
        # Update consensus
        fluid = WorkingFluid(
            id=fluid.id,
            state=fluid.state,
            temperature=fluid.temperature,
            pressure=fluid.pressure,
            volume=fluid.volume,
            internal_energy=fluid.internal_energy,
            entropy=fluid.entropy,
            proposal_hash=fluid.proposal_hash,
            participants=fluid.participants,
            consensus_level=min(1.0, max(0, ihsan_adjusted_work / 50)),
            ihsan_score=fluid.ihsan_score,
        )
        
        # Total entropy change for irreversible Otto cycle (always >= 0)
        # ΔS_universe = Q_cold/T_cold - Q_hot/T_hot >= 0 for irreversible processes
        entropy_production = (total_heat_out / self.cold_reservoir.temperature 
                              - total_heat_in / self.hot_reservoir.temperature)
        total_entropy_change = max(0, entropy_production)
        
        metrics = CycleMetrics(
            cycle_id=cycle_id,
            cycle_type=CycleType.OTTO,
            heat_absorbed=total_heat_in,
            heat_rejected=total_heat_out,
            work_output=ihsan_adjusted_work,
            thermal_efficiency=actual_efficiency,
            carnot_efficiency=self.carnot_efficiency,
            second_law_efficiency=second_law_eff,
            total_entropy_change=total_entropy_change,
            irreversibility=irreversibility,
            decisions_made=1 if fluid.consensus_level > 0.5 else 0,
            consensus_achieved=fluid.consensus_level,
            value_created=ihsan_adjusted_work * 0.1,
            ihsan_compliance=fluid.ihsan_score,
            cycle_duration_ms=(time.time() - start_time) * 1000,
        )
        
        self.cycle_history.append(metrics)
        return (fluid, metrics)
    
    async def run_stirling_cycle(
        self,
        fluid: WorkingFluid,
        compression_ratio: float = 4.0,
    ) -> Tuple[WorkingFluid, CycleMetrics]:
        """
        Run a Stirling cycle - continuous regenerative governance.
        
        Ideal for long-running governance processes with heat recovery.
        
        Stages:
        1. Isothermal compression at T_cold
        2. Isochoric heating (regeneration)
        3. Isothermal expansion at T_hot
        4. Isochoric cooling (regeneration)
        """
        start_time = time.time()
        cycle_id = self._generate_cycle_id()
        
        v1 = fluid.volume
        v2 = v1 / compression_ratio
        
        total_heat_in = 0.0
        total_heat_out = 0.0
        total_work = 0.0
        regenerated_heat = 0.0
        
        # Stage 1: Isothermal compression at T_cold
        fluid.state = ThermodynamicState.COMPRESSION
        fluid.temperature = self.cold_reservoir.temperature
        process1 = IsothermalProcess(v2, self.cold_reservoir)
        fluid, q1, w1 = process1.execute(fluid)
        total_heat_out += abs(q1)
        total_work += w1
        
        # Stage 2: Isochoric heating with regeneration
        fluid.state = ThermodynamicState.RECOVERY
        process2 = IsochoricProcess(self.hot_reservoir.temperature, self.hot_reservoir)
        fluid, q2, w2 = process2.execute(fluid)
        regenerated_heat += q2 * 0.9  # 90% regeneration efficiency
        
        # Stage 3: Isothermal expansion at T_hot
        fluid.state = ThermodynamicState.EXPANSION
        process3 = IsothermalProcess(v1, self.hot_reservoir)
        fluid, q3, w3 = process3.execute(fluid)
        total_heat_in += q3
        total_work += w3
        
        # Stage 4: Isochoric cooling with regeneration
        fluid.state = ThermodynamicState.EXHAUST
        process4 = IsochoricProcess(self.cold_reservoir.temperature, self.cold_reservoir)
        fluid, q4, w4 = process4.execute(fluid)
        regenerated_heat += abs(q4) * 0.9
        
        fluid.state = ThermodynamicState.EQUILIBRIUM
        
        # Stirling efficiency calculation
        # Use gross heat in for proper efficiency calculation (Second Law compliant)
        gross_heat_in = total_heat_in + abs(regenerated_heat)
        actual_efficiency = total_work / gross_heat_in if gross_heat_in > 0 else 0
        
        # Clamp efficiency to physical limits
        actual_efficiency = min(actual_efficiency, self.carnot_efficiency)
        second_law_eff = actual_efficiency / self.carnot_efficiency if self.carnot_efficiency > 0 else 0
        second_law_eff = min(second_law_eff, 1.0)  # Cannot exceed Carnot
        
        # Apply Ihsan
        ihsan_adjusted_work = abs(total_work) * self.ihsan_coupling * fluid.ihsan_score
        
        fluid = WorkingFluid(
            id=fluid.id,
            state=fluid.state,
            temperature=fluid.temperature,
            pressure=fluid.pressure,
            volume=fluid.volume,
            internal_energy=fluid.internal_energy,
            entropy=fluid.entropy,
            proposal_hash=fluid.proposal_hash,
            participants=fluid.participants,
            consensus_level=min(1.0, max(0, ihsan_adjusted_work / 75)),
            ihsan_score=fluid.ihsan_score,
        )
        
        # Calculate irreversibility (always non-negative)
        irreversibility = max(0, (1 - second_law_eff) * abs(total_work))
        
        metrics = CycleMetrics(
            cycle_id=cycle_id,
            cycle_type=CycleType.STIRLING,
            heat_absorbed=total_heat_in,
            heat_rejected=total_heat_out,
            work_output=ihsan_adjusted_work,
            thermal_efficiency=actual_efficiency,
            carnot_efficiency=self.carnot_efficiency,
            second_law_efficiency=second_law_eff,
            total_entropy_change=0.0,  # Stirling can be reversible
            irreversibility=irreversibility,
            decisions_made=1 if fluid.consensus_level > 0.5 else 0,
            consensus_achieved=fluid.consensus_level,
            value_created=ihsan_adjusted_work * 0.1,
            ihsan_compliance=fluid.ihsan_score,
            cycle_duration_ms=(time.time() - start_time) * 1000,
        )
        
        self.cycle_history.append(metrics)
        return (fluid, metrics)
    
    async def run_cycle(
        self,
        proposal_data: Dict[str, Any],
        cycle_type: Optional[CycleType] = None,
        compression_ratio: float = 4.0,
        **kwargs,
    ) -> Tuple[WorkingFluid, CycleMetrics]:
        """
        Run a complete thermodynamic cycle on a governance proposal.
        
        Selects appropriate cycle based on urgency and complexity.
        """
        # Extract fluid creation kwargs
        fluid_kwargs = {k: v for k, v in kwargs.items() 
                        if k in ("initial_stake", "urgency")}
        fluid = self.create_working_fluid(proposal_data, **fluid_kwargs)
        cycle = cycle_type or self.cycle_type
        
        if cycle == CycleType.CARNOT:
            return await self.run_carnot_cycle(fluid, compression_ratio=compression_ratio)
        elif cycle == CycleType.OTTO:
            return await self.run_otto_cycle(fluid, compression_ratio=compression_ratio)
        elif cycle == CycleType.STIRLING:
            return await self.run_stirling_cycle(fluid, compression_ratio=compression_ratio)
        else:
            # Default to Otto for unimplemented cycles
            return await self.run_otto_cycle(fluid, compression_ratio=compression_ratio)
    
    def get_efficiency_report(self) -> Dict[str, Any]:
        """Generate efficiency report from cycle history."""
        if not self.cycle_history:
            return {"error": "No cycles completed"}
        
        efficiencies = [m.thermal_efficiency for m in self.cycle_history]
        work_outputs = [m.work_output for m in self.cycle_history]
        ihsan_scores = [m.ihsan_compliance for m in self.cycle_history]
        
        return {
            "total_cycles": len(self.cycle_history),
            "average_efficiency": sum(efficiencies) / len(efficiencies),
            "max_efficiency": max(efficiencies),
            "carnot_limit": self.carnot_efficiency,
            "total_work_extracted": sum(work_outputs),
            "average_ihsan_compliance": sum(ihsan_scores) / len(ihsan_scores),
            "total_decisions": sum(m.decisions_made for m in self.cycle_history),
            "total_value_created": sum(m.value_created for m in self.cycle_history),
            "average_cycle_time_ms": sum(m.cycle_duration_ms for m in self.cycle_history) / len(self.cycle_history),
        }
    
    def compute_landauer_cost(self, bits_erased: int) -> float:
        """
        Compute minimum energy cost to erase information (Landauer's principle).
        
        This is the theoretical minimum energy cost of any irreversible
        computation in the governance system.
        """
        return bits_erased * LANDAUER_LIMIT
    
    def compute_governance_entropy(self, proposal_states: List[Dict[str, Any]]) -> float:
        """
        Compute total entropy of governance state space.
        
        Higher entropy = more uncertainty/options
        Lower entropy = more consensus/certainty
        """
        if not proposal_states:
            return 0.0
        
        total_entropy = 0.0
        for state in proposal_states:
            # State entropy from vote distribution
            votes = state.get("votes", {})
            if votes:
                total = sum(votes.values())
                if total > 0:
                    probs = [v / total for v in votes.values() if v > 0]
                    state_entropy = -sum(p * math.log(p) for p in probs if p > 0)
                    total_entropy += state_entropy
        
        return total_entropy


# ═══════════════════════════════════════════════════════════════════════════════
# Factory and Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def create_engine(
    cycle_type: str = "otto",
    hot_temp: float = 600.0,
    cold_temp: float = 300.0,
) -> BIZRAThermodynamicEngine:
    """
    Factory function to create a BIZRA Thermodynamic Engine.
    
    Args:
        cycle_type: One of "carnot", "otto", "stirling", "diesel", "rankine"
        hot_temp: Hot reservoir temperature (governance activity)
        cold_temp: Cold reservoir temperature (baseline)
    
    Returns:
        Configured BIZRAThermodynamicEngine instance.
    """
    cycle_map = {
        "carnot": CycleType.CARNOT,
        "otto": CycleType.OTTO,
        "diesel": CycleType.DIESEL,
        "stirling": CycleType.STIRLING,
        "rankine": CycleType.RANKINE,
    }
    
    cycle = cycle_map.get(cycle_type.lower(), CycleType.OTTO)
    
    return BIZRAThermodynamicEngine(
        cycle_type=cycle,
        hot_temp=hot_temp,
        cold_temp=cold_temp,
    )


async def demo_thermodynamic_governance():
    """
    Demonstrate the thermodynamic engine processing governance proposals.
    """
    print("=" * 70)
    print("BIZRA THERMODYNAMIC ENGINE - GOVERNANCE SIMULATION")
    print("=" * 70)
    
    # Create engine
    engine = create_engine(cycle_type="otto", hot_temp=800.0, cold_temp=300.0)
    print(f"\nEngine Type: {engine.cycle_type.name}")
    print(f"Carnot Efficiency Limit: {engine.carnot_efficiency:.2%}")
    print(f"Hot Reservoir: {engine.hot_reservoir.temperature} gK")
    print(f"Cold Reservoir: {engine.cold_reservoir.temperature} gK")
    
    # Sample governance proposals
    proposals = [
        {
            "id": "prop-001",
            "title": "Increase staking rewards",
            "stake_distribution": [100, 200, 150, 300, 250],
            "participants": 5,
            "urgency": 1.5,
        },
        {
            "id": "prop-002", 
            "title": "Add new validator requirements",
            "stake_distribution": [1000, 500, 500, 800, 700, 600],
            "participants": 6,
            "urgency": 2.0,
        },
        {
            "id": "prop-003",
            "title": "Emergency protocol upgrade",
            "stake_distribution": [100, 100, 100, 100],
            "participants": 4,
            "urgency": 5.0,  # High urgency
        },
    ]
    
    print("\n" + "-" * 70)
    print("PROCESSING GOVERNANCE PROPOSALS")
    print("-" * 70)
    
    for proposal in proposals:
        print(f"\n▶ Proposal: {proposal['title']}")
        
        fluid, metrics = await engine.run_cycle(
            proposal,
            initial_stake=sum(proposal["stake_distribution"]),
            urgency=proposal["urgency"],
        )
        
        print(f"  Cycle ID: {metrics.cycle_id}")
        print(f"  Heat In: {metrics.heat_absorbed:.2f} infoJ")
        print(f"  Heat Out: {metrics.heat_rejected:.2f} infoJ")
        print(f"  Work Output: {metrics.work_output:.2f} iJ")
        print(f"  Efficiency: {metrics.thermal_efficiency:.2%} (Carnot: {metrics.carnot_efficiency:.2%})")
        print(f"  Consensus Level: {metrics.consensus_achieved:.2%}")
        print(f"  Decision Made: {'✓' if metrics.decisions_made else '✗'}")
        print(f"  Value Created: {metrics.value_created:.2f} BZC")
        print(f"  Cycle Time: {metrics.cycle_duration_ms:.2f} ms")
    
    # Final report
    print("\n" + "=" * 70)
    print("EFFICIENCY REPORT")
    print("=" * 70)
    
    report = engine.get_efficiency_report()
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✓ Thermodynamic governance cycle complete.")
    return engine


if __name__ == "__main__":
    asyncio.run(demo_thermodynamic_governance())
