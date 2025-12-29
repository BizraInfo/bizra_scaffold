"""
BIZRA AEON OMEGA - Peak Masterpiece Integration Tests
═══════════════════════════════════════════════════════════════════════════════
Validating the Ultimate Implementation of the Sovereign Mind.
"""

import pytest
import asyncio
import os
from core.peak_masterpiece import PeakMasterpieceOrchestrator
from core.snr_scorer import SNRLevel

@pytest.mark.asyncio
async def test_pmo_initialization():
    """Test that the Peak Masterpiece Orchestrator can establish sovereignty."""
    root = os.getcwd()
    pmo = PeakMasterpieceOrchestrator(root)
    
    # Note: This requires the GENESIS_SYSTEM_MANIFEST.json to exist
    success = await pmo.initialize_sovereign_state()
    assert success is True
    assert pmo.is_sovereign is True
    
    status = pmo.get_system_status()
    assert status["sovereign"] is True
    assert status["peak_snr"] == 10.0
    assert status["ihsan_metric"] == 0.99

@pytest.mark.asyncio
async def test_pmo_giants_protocol():
    """Test the 'Standing on the Shoulders of Giants' protocol integration."""
    root = os.getcwd()
    pmo = PeakMasterpieceOrchestrator(root)
    await pmo.initialize_sovereign_state()
    
    # Verify all 7 APEX layers are active in the status
    status = pmo.get_system_status()
    assert len(status["active_layers"]) == 7
    assert "PHILOSOPHY" in status["active_layers"]
    assert "BLOCKCHAIN" in status["active_layers"]

@pytest.mark.asyncio
async def test_pmo_subsystem_integration():
    """Test that all core subsystems are correctly wired into the PMO."""
    root = os.getcwd()
    pmo = PeakMasterpieceOrchestrator(root)
    
    assert pmo.ultimate is not None
    assert pmo.got is not None
    assert pmo.snr is not None
    assert pmo.apex is not None
    assert pmo.command is not None
    assert pmo.validator is not None
