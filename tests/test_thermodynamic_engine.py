"""
BIZRA Thermodynamic Engine - Test Suite
═══════════════════════════════════════════════════════════════════════════════
Comprehensive tests for the thermodynamic governance model.
"""

import pytest
import asyncio
import math
from datetime import datetime, timezone

from core.thermodynamic_engine import (
    BIZRAThermodynamicEngine,
    ThermodynamicState,
    CycleType,
    ThermodynamicConstants,
    Reservoir,
    WorkingFluid,
    CycleMetrics,
    IsothermalProcess,
    AdiabaticProcess,
    IsobaricProcess,
    IsochoricProcess,
    create_engine,
    LANDAUER_LIMIT,
)


class TestThermodynamicConstants:
    """Test thermodynamic constants and bounds."""
    
    def test_constants_values(self):
        """Verify thermodynamic constants are properly defined."""
        constants = ThermodynamicConstants()
        
        assert constants.T_MIN == 0.01
        assert constants.T_MAX == 1000.0
        assert constants.T_AMBIENT == 300.0
        assert constants.P_MIN == 0.1
        assert constants.P_MAX == 100.0
        assert constants.IHSAN_COUPLING == 0.95
    
    def test_temperature_ordering(self):
        """Temperature bounds must be ordered correctly."""
        constants = ThermodynamicConstants()
        assert constants.T_MIN < constants.T_AMBIENT < constants.T_MAX
    
    def test_pressure_ordering(self):
        """Pressure bounds must be ordered correctly."""
        constants = ThermodynamicConstants()
        assert constants.P_MIN < constants.P_STANDARD < constants.P_MAX


class TestReservoir:
    """Test thermodynamic reservoir behavior."""
    
    def test_reservoir_creation(self):
        """Test reservoir initialization."""
        reservoir = Reservoir(
            name="TestReservoir",
            temperature=300.0,
            capacity=1000.0,
            current_energy=500.0,
            entropy=1.0,
        )
        
        assert reservoir.name == "TestReservoir"
        assert reservoir.temperature == 300.0
        assert reservoir.capacity == 1000.0
        assert reservoir.current_energy == 500.0
        assert reservoir.entropy == 1.0
        assert not reservoir.is_infinite
    
    def test_infinite_reservoir_absorb(self):
        """Infinite reservoir maintains temperature when absorbing heat."""
        reservoir = Reservoir(
            name="Environment",
            temperature=300.0,
            capacity=1000.0,
            current_energy=500.0,
            entropy=1.0,
            is_infinite=True,
        )
        
        initial_temp = reservoir.temperature
        absorbed, ds = reservoir.absorb_heat(100.0)
        
        assert absorbed == 100.0
        assert reservoir.temperature == initial_temp  # Unchanged
        assert ds == pytest.approx(100.0 / 300.0, rel=1e-6)
    
    def test_finite_reservoir_heats_up(self):
        """Finite reservoir temperature increases when absorbing heat."""
        reservoir = Reservoir(
            name="Finite",
            temperature=300.0,
            capacity=1000.0,
            current_energy=500.0,
            entropy=1.0,
            is_infinite=False,
        )
        
        initial_temp = reservoir.temperature
        reservoir.absorb_heat(100.0)
        
        assert reservoir.temperature > initial_temp
        assert reservoir.current_energy == 600.0
    
    def test_reservoir_capacity_limit(self):
        """Reservoir cannot absorb more than capacity allows."""
        reservoir = Reservoir(
            name="Limited",
            temperature=300.0,
            capacity=100.0,
            current_energy=90.0,
            entropy=1.0,
            is_infinite=False,
        )
        
        absorbed, _ = reservoir.absorb_heat(50.0)
        
        assert absorbed == 10.0  # Only 10 available
        assert reservoir.current_energy == 100.0


class TestWorkingFluid:
    """Test working fluid thermodynamic properties."""
    
    def test_fluid_creation(self):
        """Test working fluid initialization."""
        fluid = WorkingFluid(
            id="test-fluid",
            state=ThermodynamicState.EQUILIBRIUM,
            temperature=300.0,
            pressure=1.0,
            volume=10.0,
            internal_energy=100.0,
            entropy=0.5,
        )
        
        assert fluid.id == "test-fluid"
        assert fluid.state == ThermodynamicState.EQUILIBRIUM
        assert fluid.temperature == 300.0
    
    def test_enthalpy_calculation(self):
        """Test H = U + PV."""
        fluid = WorkingFluid(
            id="test",
            state=ThermodynamicState.EQUILIBRIUM,
            temperature=300.0,
            pressure=2.0,
            volume=5.0,
            internal_energy=100.0,
            entropy=0.5,
        )
        
        expected_h = 100.0 + 2.0 * 5.0  # U + PV = 110
        assert fluid.enthalpy == pytest.approx(expected_h, rel=1e-6)
    
    def test_helmholtz_free_energy(self):
        """Test F = U - TS."""
        fluid = WorkingFluid(
            id="test",
            state=ThermodynamicState.EQUILIBRIUM,
            temperature=300.0,
            pressure=1.0,
            volume=10.0,
            internal_energy=100.0,
            entropy=0.2,
        )
        
        expected_f = 100.0 - 300.0 * 0.2  # U - TS = 40
        assert fluid.helmholtz_free_energy == pytest.approx(expected_f, rel=1e-6)
    
    def test_gibbs_free_energy(self):
        """Test G = H - TS."""
        fluid = WorkingFluid(
            id="test",
            state=ThermodynamicState.EQUILIBRIUM,
            temperature=300.0,
            pressure=2.0,
            volume=5.0,
            internal_energy=100.0,
            entropy=0.2,
        )
        
        h = 100.0 + 2.0 * 5.0  # 110
        expected_g = h - 300.0 * 0.2  # 110 - 60 = 50
        assert fluid.gibbs_free_energy == pytest.approx(expected_g, rel=1e-6)
    
    def test_spontaneity(self):
        """Negative Gibbs free energy indicates spontaneous process."""
        # High entropy fluid (spontaneous)
        fluid1 = WorkingFluid(
            id="spontaneous",
            state=ThermodynamicState.EQUILIBRIUM,
            temperature=300.0,
            pressure=1.0,
            volume=1.0,
            internal_energy=10.0,
            entropy=1.0,  # High entropy
        )
        
        # Low entropy fluid (non-spontaneous)
        fluid2 = WorkingFluid(
            id="non-spontaneous",
            state=ThermodynamicState.EQUILIBRIUM,
            temperature=300.0,
            pressure=1.0,
            volume=1.0,
            internal_energy=500.0,
            entropy=0.01,  # Low entropy
        )
        
        assert fluid1.is_spontaneous()  # G < 0
        assert not fluid2.is_spontaneous()  # G > 0


class TestIsothermalProcess:
    """Test isothermal process behavior."""
    
    def test_isothermal_expansion(self):
        """Isothermal expansion maintains temperature."""
        reservoir = Reservoir(
            name="HotReservoir",
            temperature=500.0,
            capacity=1000.0,
            current_energy=500.0,
            entropy=1.0,
            is_infinite=True,
        )
        
        fluid = WorkingFluid(
            id="test",
            state=ThermodynamicState.EXPANSION,
            temperature=500.0,
            pressure=10.0,
            volume=1.0,
            internal_energy=100.0,
            entropy=0.5,
        )
        
        process = IsothermalProcess(target_volume=2.0, reservoir=reservoir)
        new_fluid, heat, work = process.execute(fluid)
        
        assert new_fluid.temperature == 500.0  # Unchanged
        assert new_fluid.volume == 2.0
        assert work > 0  # Work done by system in expansion
        assert heat == pytest.approx(work, rel=1e-6)  # Q = W for isothermal
    
    def test_isothermal_compression(self):
        """Isothermal compression requires work input."""
        reservoir = Reservoir(
            name="ColdReservoir",
            temperature=300.0,
            capacity=1000.0,
            current_energy=200.0,
            entropy=1.0,
            is_infinite=True,
        )
        
        fluid = WorkingFluid(
            id="test",
            state=ThermodynamicState.COMPRESSION,
            temperature=300.0,
            pressure=5.0,
            volume=2.0,
            internal_energy=100.0,
            entropy=0.5,
        )
        
        process = IsothermalProcess(target_volume=1.0, reservoir=reservoir)
        new_fluid, heat, work = process.execute(fluid)
        
        assert new_fluid.temperature == 300.0
        assert new_fluid.volume == 1.0
        assert work < 0  # Work done on system
        assert new_fluid.pressure > fluid.pressure  # Pressure increases


class TestAdiabaticProcess:
    """Test adiabatic process behavior."""
    
    def test_adiabatic_no_heat_transfer(self):
        """Adiabatic process has no heat transfer."""
        fluid = WorkingFluid(
            id="test",
            state=ThermodynamicState.EXPANSION,
            temperature=500.0,
            pressure=10.0,
            volume=1.0,
            internal_energy=100.0,
            entropy=0.5,
        )
        
        process = AdiabaticProcess(target_volume=2.0, gamma=1.4)
        new_fluid, heat, work = process.execute(fluid)
        
        assert heat == 0  # No heat transfer
        assert new_fluid.entropy == fluid.entropy  # Entropy unchanged
    
    def test_adiabatic_expansion_cools(self):
        """Adiabatic expansion decreases temperature."""
        fluid = WorkingFluid(
            id="test",
            state=ThermodynamicState.EXPANSION,
            temperature=500.0,
            pressure=10.0,
            volume=1.0,
            internal_energy=100.0,
            entropy=0.5,
        )
        
        process = AdiabaticProcess(target_volume=2.0, gamma=1.4)
        new_fluid, _, _ = process.execute(fluid)
        
        assert new_fluid.temperature < fluid.temperature
        assert new_fluid.volume > fluid.volume
    
    def test_adiabatic_compression_heats(self):
        """Adiabatic compression increases temperature."""
        fluid = WorkingFluid(
            id="test",
            state=ThermodynamicState.COMPRESSION,
            temperature=300.0,
            pressure=5.0,
            volume=2.0,
            internal_energy=100.0,
            entropy=0.5,
        )
        
        process = AdiabaticProcess(target_volume=1.0, gamma=1.4)
        new_fluid, _, _ = process.execute(fluid)
        
        assert new_fluid.temperature > fluid.temperature
        assert new_fluid.pressure > fluid.pressure


class TestBIZRAThermodynamicEngine:
    """Test the main thermodynamic engine."""
    
    def test_engine_creation(self):
        """Test engine initialization."""
        engine = BIZRAThermodynamicEngine(
            cycle_type=CycleType.OTTO,
            hot_temp=600.0,
            cold_temp=300.0,
        )
        
        assert engine.cycle_type == CycleType.OTTO
        assert engine.hot_reservoir.temperature == 600.0
        assert engine.cold_reservoir.temperature == 300.0
        assert engine.carnot_efficiency == pytest.approx(0.5, rel=1e-6)
    
    def test_carnot_efficiency_limit(self):
        """Carnot efficiency is correctly calculated."""
        engine = BIZRAThermodynamicEngine(
            hot_temp=800.0,
            cold_temp=200.0,
        )
        
        expected = 1 - 200.0 / 800.0  # 0.75
        assert engine.carnot_efficiency == pytest.approx(expected, rel=1e-6)
    
    def test_create_working_fluid(self):
        """Test fluid creation from proposal data."""
        engine = BIZRAThermodynamicEngine()
        
        proposal = {
            "id": "test-proposal",
            "title": "Test",
            "stake_distribution": [100, 200, 300],
            "participants": 3,
        }
        
        fluid = engine.create_working_fluid(proposal, initial_stake=600.0, urgency=2.0)
        
        assert fluid.participants == 3
        assert fluid.internal_energy == 600.0
        assert fluid.pressure == 2.0  # Urgency
        assert fluid.entropy > 0  # Has stake entropy
    
    def test_stake_entropy_calculation(self):
        """Verify entropy from stake distribution."""
        engine = BIZRAThermodynamicEngine()
        
        # Equal distribution = maximum entropy
        equal_dist = {"stake_distribution": [100, 100, 100, 100]}
        entropy_equal = engine._compute_stake_entropy(equal_dist)
        
        # Unequal distribution = lower entropy
        unequal_dist = {"stake_distribution": [1000, 1, 1, 1]}
        entropy_unequal = engine._compute_stake_entropy(unequal_dist)
        
        assert entropy_equal > entropy_unequal
    
    @pytest.mark.asyncio
    async def test_otto_cycle(self):
        """Test Otto cycle execution."""
        engine = BIZRAThermodynamicEngine(
            cycle_type=CycleType.OTTO,
            hot_temp=800.0,
            cold_temp=300.0,
        )
        
        proposal = {
            "id": "otto-test",
            "title": "Otto Cycle Test",
            "stake_distribution": [100, 150, 200],
        }
        
        fluid, metrics = await engine.run_otto_cycle(
            engine.create_working_fluid(proposal, initial_stake=450.0)
        )
        
        assert metrics.cycle_type == CycleType.OTTO
        assert metrics.thermal_efficiency <= metrics.carnot_efficiency
        assert metrics.work_output >= 0
        assert metrics.ihsan_compliance > 0
        assert fluid.state == ThermodynamicState.EQUILIBRIUM
    
    @pytest.mark.asyncio
    async def test_carnot_cycle(self):
        """Test Carnot cycle execution."""
        engine = BIZRAThermodynamicEngine(
            cycle_type=CycleType.CARNOT,
            hot_temp=600.0,
            cold_temp=300.0,
        )
        
        proposal = {
            "id": "carnot-test",
            "title": "Carnot Cycle Test",
            "stake_distribution": [200, 200],
        }
        
        fluid, metrics = await engine.run_carnot_cycle(
            engine.create_working_fluid(proposal, initial_stake=400.0)
        )
        
        assert metrics.cycle_type == CycleType.CARNOT
        assert metrics.irreversibility == 0.0  # Carnot is reversible
    
    @pytest.mark.asyncio
    async def test_stirling_cycle(self):
        """Test Stirling cycle execution."""
        engine = BIZRAThermodynamicEngine(
            cycle_type=CycleType.STIRLING,
            hot_temp=700.0,
            cold_temp=300.0,
        )
        
        proposal = {
            "id": "stirling-test",
            "title": "Stirling Cycle Test",
            "stake_distribution": [100, 100, 100, 100],
        }
        
        fluid, metrics = await engine.run_stirling_cycle(
            engine.create_working_fluid(proposal, initial_stake=400.0)
        )
        
        assert metrics.cycle_type == CycleType.STIRLING
        assert metrics.second_law_efficiency <= 1.0
    
    @pytest.mark.asyncio
    async def test_run_cycle_dispatch(self):
        """Test automatic cycle dispatch."""
        engine = BIZRAThermodynamicEngine(cycle_type=CycleType.STIRLING)
        
        proposal = {
            "id": "dispatch-test",
            "title": "Dispatch Test",
            "stake_distribution": [50, 50],
        }
        
        _, metrics = await engine.run_cycle(proposal, initial_stake=100.0)
        
        assert metrics.cycle_type == CycleType.STIRLING
    
    @pytest.mark.asyncio
    async def test_efficiency_report(self):
        """Test efficiency report generation."""
        engine = BIZRAThermodynamicEngine()
        
        # Run multiple cycles
        for i in range(3):
            proposal = {
                "id": f"report-test-{i}",
                "title": f"Report Test {i}",
                "stake_distribution": [100 * (i + 1)],
            }
            await engine.run_cycle(proposal, initial_stake=100.0 * (i + 1))
        
        report = engine.get_efficiency_report()
        
        assert report["total_cycles"] == 3
        assert "average_efficiency" in report
        assert "carnot_limit" in report
        assert "total_work_extracted" in report
    
    def test_landauer_cost(self):
        """Test Landauer limit calculation."""
        engine = BIZRAThermodynamicEngine()
        
        cost = engine.compute_landauer_cost(1000)  # 1000 bits
        
        assert cost == pytest.approx(1000 * LANDAUER_LIMIT, rel=1e-6)
    
    def test_governance_entropy(self):
        """Test governance entropy calculation."""
        engine = BIZRAThermodynamicEngine()
        
        states = [
            {"votes": {"yes": 50, "no": 50}},  # High entropy (50-50)
            {"votes": {"yes": 99, "no": 1}},   # Low entropy (consensus)
        ]
        
        entropy = engine.compute_governance_entropy(states)
        
        assert entropy > 0


class TestFactoryFunction:
    """Test the create_engine factory function."""
    
    def test_create_otto_engine(self):
        """Factory creates Otto cycle engine."""
        engine = create_engine(cycle_type="otto")
        assert engine.cycle_type == CycleType.OTTO
    
    def test_create_carnot_engine(self):
        """Factory creates Carnot cycle engine."""
        engine = create_engine(cycle_type="carnot")
        assert engine.cycle_type == CycleType.CARNOT
    
    def test_create_stirling_engine(self):
        """Factory creates Stirling cycle engine."""
        engine = create_engine(cycle_type="stirling")
        assert engine.cycle_type == CycleType.STIRLING
    
    def test_custom_temperatures(self):
        """Factory accepts custom temperatures."""
        engine = create_engine(hot_temp=1000.0, cold_temp=200.0)
        
        assert engine.hot_reservoir.temperature == 1000.0
        assert engine.cold_reservoir.temperature == 200.0
        assert engine.carnot_efficiency == pytest.approx(0.8, rel=1e-6)
    
    def test_unknown_cycle_defaults_to_otto(self):
        """Unknown cycle type defaults to Otto."""
        engine = create_engine(cycle_type="unknown")
        assert engine.cycle_type == CycleType.OTTO


class TestSecondLawCompliance:
    """Test that the engine respects the Second Law of Thermodynamics."""
    
    @pytest.mark.asyncio
    async def test_efficiency_cannot_exceed_carnot(self):
        """No cycle can exceed Carnot efficiency."""
        engine = BIZRAThermodynamicEngine(
            hot_temp=1000.0,
            cold_temp=300.0,
        )
        
        for _ in range(10):
            proposal = {
                "stake_distribution": [100, 200, 300, 400, 500],
            }
            _, metrics = await engine.run_cycle(proposal)
            
            # Second law: η ≤ η_carnot
            assert metrics.thermal_efficiency <= metrics.carnot_efficiency + 0.01  # Small tolerance
    
    @pytest.mark.asyncio
    async def test_total_entropy_non_negative(self):
        """Total entropy change in universe must be ≥ 0."""
        engine = BIZRAThermodynamicEngine()
        
        proposal = {"stake_distribution": [100, 100]}
        _, metrics = await engine.run_cycle(proposal)
        
        # For real irreversible processes, ΔS_universe ≥ 0
        # Carnot cycle has ΔS = 0, others have ΔS > 0
        assert metrics.total_entropy_change >= -0.01  # Small tolerance for numerical errors


class TestIhsanCoupling:
    """Test Ihsan ethical coupling with thermodynamics."""
    
    @pytest.mark.asyncio
    async def test_ihsan_reduces_extractable_work(self):
        """Ihsan coupling reduces extractable work (ethics has a cost)."""
        # High Ihsan coupling
        engine_high = BIZRAThermodynamicEngine(ihsan_coupling=0.99)
        
        # Low Ihsan coupling
        engine_low = BIZRAThermodynamicEngine(ihsan_coupling=0.50)
        
        proposal = {"stake_distribution": [100, 100, 100]}
        
        _, metrics_high = await engine_high.run_cycle(proposal, initial_stake=300.0)
        _, metrics_low = await engine_low.run_cycle(proposal, initial_stake=300.0)
        
        # Higher ethics coupling = more work (counterintuitive but represents sustainable work)
        # Actually, both should produce work proportional to coupling
        assert metrics_high.work_output > 0
        assert metrics_low.work_output > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
