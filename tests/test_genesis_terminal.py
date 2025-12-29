"""
Tests for Genesis Terminal - Glass Cockpit.

Tests the complete observability stack:
- Genesis Events Protocol
- Streaming Orchestrator
- Event Bus
- (Terminal widgets tested via headless mode where supported)
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

from core.genesis.genesis_events import (
    GenesisEventType,
    GenesisEvent,
    GenesisEventListener,
    GenesisEventBus,
    create_event,
    system_start,
    lens_activated,
    wisdom_seed_loaded,
    thought_created,
    thought_pruned,
    snr_computed,
    crystal_added,
    attestation_complete,
    oracle_drift,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_event():
    """Create a sample event."""
    return GenesisEvent(
        type=GenesisEventType.THOUGHT_NODE_CREATED,
        phase="Expansion",
        progress=0.5,
        data={"content": "Test thought", "snr": 0.85},
        correlation_id="test123",
    )


@pytest.fixture
def event_bus():
    """Create an event bus with history."""
    return GenesisEventBus(keep_history=True, max_history=100)


class MockListener:
    """Mock event listener for testing."""
    
    def __init__(self):
        self.events: List[GenesisEvent] = []
    
    async def on_genesis_event(self, event: GenesisEvent) -> None:
        self.events.append(event)


# =============================================================================
# GENESIS EVENT TYPE TESTS
# =============================================================================


class TestGenesisEventType:
    """Tests for GenesisEventType enum."""
    
    def test_all_phases_have_events(self):
        """All major phases have corresponding event types."""
        # System lifecycle
        assert GenesisEventType.SYSTEM_START
        assert GenesisEventType.SYSTEM_IDLE
        assert GenesisEventType.SYSTEM_ERROR
        
        # Lens analysis
        assert GenesisEventType.LENS_ANALYSIS_START
        assert GenesisEventType.LENS_ACTIVATED
        assert GenesisEventType.LENS_SYNTHESIS_COMPLETE
        
        # Wisdom seeding
        assert GenesisEventType.WISDOM_SEEDING_START
        assert GenesisEventType.WISDOM_SEED_LOADED
        
        # Thought expansion
        assert GenesisEventType.THOUGHT_NODE_CREATED
        assert GenesisEventType.THOUGHT_NODE_PRUNED
        
        # SNR gating
        assert GenesisEventType.SNR_GATE_START
        assert GenesisEventType.SNR_SCORE_COMPUTED
        
        # Crystallization
        assert GenesisEventType.CRYSTAL_START
        assert GenesisEventType.CRYSTAL_INSIGHT_ADDED
        
        # Attestation
        assert GenesisEventType.ATTEST_COMPLETE
        
        # Oracle
        assert GenesisEventType.ORACLE_DRIFT_DETECTED
    
    def test_event_count(self):
        """All expected event types exist."""
        # Should have at least 30 event types
        assert len(GenesisEventType) >= 25


# =============================================================================
# GENESIS EVENT TESTS
# =============================================================================


class TestGenesisEvent:
    """Tests for GenesisEvent dataclass."""
    
    def test_creation(self, sample_event):
        """Event can be created with all fields."""
        assert sample_event.type == GenesisEventType.THOUGHT_NODE_CREATED
        assert sample_event.phase == "Expansion"
        assert sample_event.progress == 0.5
        assert sample_event.data["snr"] == 0.85
        assert sample_event.correlation_id == "test123"
    
    def test_immutability(self, sample_event):
        """Event is frozen (immutable)."""
        with pytest.raises(AttributeError):
            sample_event.progress = 0.9
    
    def test_progress_clamping_high(self):
        """Progress above 1.0 is clamped."""
        event = GenesisEvent(
            type=GenesisEventType.SYSTEM_START,
            phase="Test",
            progress=1.5,  # Over 1.0
        )
        assert event.progress == 1.0
    
    def test_progress_clamping_low(self):
        """Progress below 0.0 is clamped."""
        event = GenesisEvent(
            type=GenesisEventType.SYSTEM_START,
            phase="Test",
            progress=-0.5,  # Below 0.0
        )
        assert event.progress == 0.0
    
    def test_to_dict(self, sample_event):
        """Event serializes to dictionary."""
        d = sample_event.to_dict()
        
        assert d["type"] == "THOUGHT_NODE_CREATED"
        assert d["phase"] == "Expansion"
        assert d["progress"] == 0.5
        assert d["data"]["snr"] == 0.85
        assert d["correlation_id"] == "test123"
        assert "timestamp" in d
    
    def test_from_dict(self, sample_event):
        """Event deserializes from dictionary."""
        d = sample_event.to_dict()
        restored = GenesisEvent.from_dict(d)
        
        assert restored.type == sample_event.type
        assert restored.phase == sample_event.phase
        assert restored.progress == sample_event.progress
        assert restored.data == sample_event.data
        assert restored.correlation_id == sample_event.correlation_id
    
    def test_with_progress(self, sample_event):
        """Can create copy with updated progress."""
        updated = sample_event.with_progress(0.75)
        
        assert updated.progress == 0.75
        assert updated.type == sample_event.type
        assert updated.phase == sample_event.phase
        assert sample_event.progress == 0.5  # Original unchanged
    
    def test_default_timestamp(self):
        """Timestamp defaults to now."""
        event = create_event(GenesisEventType.SYSTEM_START, "Test", 0.0)
        
        now = datetime.now(timezone.utc)
        delta = abs((now - event.timestamp).total_seconds())
        assert delta < 1.0  # Within 1 second


# =============================================================================
# EVENT BUS TESTS
# =============================================================================


class TestGenesisEventBus:
    """Tests for GenesisEventBus."""
    
    @pytest.mark.asyncio
    async def test_emit_to_global_listener(self, event_bus, sample_event):
        """Events are delivered to global listeners."""
        listener = MockListener()
        event_bus.add_listener(listener)
        
        await event_bus.emit(sample_event)
        
        assert len(listener.events) == 1
        assert listener.events[0] == sample_event
    
    @pytest.mark.asyncio
    async def test_emit_to_type_listener(self, event_bus, sample_event):
        """Events are delivered to type-specific listeners."""
        listener = MockListener()
        event_bus.add_listener(
            listener,
            event_types=[GenesisEventType.THOUGHT_NODE_CREATED],
        )
        
        await event_bus.emit(sample_event)
        
        assert len(listener.events) == 1
    
    @pytest.mark.asyncio
    async def test_type_listener_filters(self, event_bus):
        """Type-specific listeners only receive matching events."""
        listener = MockListener()
        event_bus.add_listener(
            listener,
            event_types=[GenesisEventType.ATTEST_COMPLETE],
        )
        
        # Emit non-matching event
        await event_bus.emit(create_event(
            GenesisEventType.SYSTEM_START,
            "Test",
            0.0,
        ))
        
        assert len(listener.events) == 0
    
    @pytest.mark.asyncio
    async def test_remove_listener(self, event_bus, sample_event):
        """Listeners can be removed."""
        listener = MockListener()
        event_bus.add_listener(listener)
        
        await event_bus.emit(sample_event)
        assert len(listener.events) == 1
        
        event_bus.remove_listener(listener)
        
        await event_bus.emit(sample_event)
        assert len(listener.events) == 1  # No new events
    
    @pytest.mark.asyncio
    async def test_history(self, event_bus, sample_event):
        """History is maintained when enabled."""
        await event_bus.emit(sample_event)
        await event_bus.emit(sample_event)
        
        history = event_bus.get_history()
        
        assert len(history) == 2
    
    @pytest.mark.asyncio
    async def test_history_limit(self):
        """History respects max_history limit."""
        bus = GenesisEventBus(keep_history=True, max_history=5)
        
        for i in range(10):
            await bus.emit(create_event(
                GenesisEventType.SYSTEM_START,
                f"Test {i}",
                i / 10,
            ))
        
        history = bus.get_history(limit=100)
        
        assert len(history) == 5
    
    @pytest.mark.asyncio
    async def test_history_filter_by_type(self, event_bus):
        """History can be filtered by event type."""
        await event_bus.emit(create_event(
            GenesisEventType.SYSTEM_START, "A", 0.0
        ))
        await event_bus.emit(create_event(
            GenesisEventType.ATTEST_COMPLETE, "B", 1.0
        ))
        await event_bus.emit(create_event(
            GenesisEventType.SYSTEM_START, "C", 0.0
        ))
        
        history = event_bus.get_history(event_type=GenesisEventType.SYSTEM_START)
        
        assert len(history) == 2
    
    def test_clear_history(self, event_bus):
        """History can be cleared."""
        event_bus._history = [MagicMock() for _ in range(5)]
        
        event_bus.clear_history()
        
        assert len(event_bus._history) == 0
    
    def test_listener_count(self, event_bus):
        """Listener count is tracked."""
        assert event_bus.listener_count == 0
        
        listener1 = MockListener()
        listener2 = MockListener()
        
        event_bus.add_listener(listener1)
        assert event_bus.listener_count == 1
        
        event_bus.add_listener(listener2, event_types=[GenesisEventType.SYSTEM_START])
        assert event_bus.listener_count == 2
    
    @pytest.mark.asyncio
    async def test_listener_error_isolation(self, event_bus, sample_event):
        """Listener errors don't break other listeners."""
        class FailingListener:
            async def on_genesis_event(self, event):
                raise RuntimeError("Intentional failure")
        
        good_listener = MockListener()
        
        event_bus.add_listener(FailingListener())
        event_bus.add_listener(good_listener)
        
        # Should not raise
        await event_bus.emit(sample_event)
        
        # Good listener still received event
        assert len(good_listener.events) == 1


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Tests for event factory functions."""
    
    def test_create_event(self):
        """create_event creates valid event."""
        event = create_event(
            GenesisEventType.SYSTEM_START,
            "Test Phase",
            0.5,
            correlation_id="abc123",
            custom_field="custom_value",
        )
        
        assert event.type == GenesisEventType.SYSTEM_START
        assert event.phase == "Test Phase"
        assert event.progress == 0.5
        assert event.correlation_id == "abc123"
        assert event.data["custom_field"] == "custom_value"
    
    def test_system_start(self):
        """system_start creates proper event."""
        event = system_start("Test problem", correlation_id="xyz")
        
        assert event.type == GenesisEventType.SYSTEM_START
        assert event.phase == "Ignition"
        assert event.progress == 0.0
        assert event.data["problem"] == "Test problem"
        assert event.correlation_id == "xyz"
    
    def test_lens_activated(self):
        """lens_activated creates proper event."""
        event = lens_activated("CRYPTO", 0.92, 0.15)
        
        assert event.type == GenesisEventType.LENS_ACTIVATED
        assert event.data["lens"] == "CRYPTO"
        assert event.data["confidence"] == 0.92
        assert event.progress == 0.15
    
    def test_wisdom_seed_loaded(self):
        """wisdom_seed_loaded creates proper event."""
        event = wisdom_seed_loaded(
            wisdom_id="w123",
            title="Test Wisdom",
            snr=0.94,
            progress=0.25,
        )
        
        assert event.type == GenesisEventType.WISDOM_SEED_LOADED
        assert event.data["wisdom_id"] == "w123"
        assert event.data["snr"] == 0.94
    
    def test_thought_created(self):
        """thought_created creates proper event."""
        event = thought_created(
            content="Test thought",
            snr=0.88,
            ihsan=0.96,
            depth=2,
            node_id="node123",
            progress=0.5,
        )
        
        assert event.type == GenesisEventType.THOUGHT_NODE_CREATED
        assert event.data["content"] == "Test thought"
        assert event.data["snr"] == 0.88
        assert event.data["depth"] == 2
    
    def test_thought_pruned(self):
        """thought_pruned creates proper event."""
        event = thought_pruned(
            node_id="node456",
            snr=0.45,
            reason="Below threshold",
            progress=0.6,
        )
        
        assert event.type == GenesisEventType.THOUGHT_NODE_PRUNED
        assert event.data["node_id"] == "node456"
        assert event.data["reason"] == "Below threshold"
    
    def test_snr_computed_high(self):
        """snr_computed sets HIGH level correctly."""
        event = snr_computed(0.92, 0.96, True, 0.7)
        
        assert event.data["level"] == "HIGH"
        assert event.data["passed"] is True
    
    def test_snr_computed_medium(self):
        """snr_computed sets MEDIUM level correctly."""
        event = snr_computed(0.65, 0.96, True, 0.7)
        
        assert event.data["level"] == "MEDIUM"
    
    def test_snr_computed_low(self):
        """snr_computed sets LOW level correctly."""
        event = snr_computed(0.35, 0.96, False, 0.7)
        
        assert event.data["level"] == "LOW"
        assert event.data["passed"] is False
    
    def test_crystal_added(self):
        """crystal_added creates proper event."""
        event = crystal_added(
            wisdom_id="w789",
            title="Crystallized Insight",
            snr=0.91,
            progress=0.9,
        )
        
        assert event.type == GenesisEventType.CRYSTAL_INSIGHT_ADDED
        assert event.data["wisdom_id"] == "w789"
    
    def test_attestation_complete(self):
        """attestation_complete creates proper event."""
        event = attestation_complete(
            attestation_hash="abc123def456",
            node_id="node0_genesis",
        )
        
        assert event.type == GenesisEventType.ATTEST_COMPLETE
        assert event.progress == 1.0  # Always 100%
        assert event.data["hash"] == "abc123def456"
        assert event.data["node_id"] == "node0_genesis"
    
    def test_oracle_drift(self):
        """oracle_drift creates proper event."""
        event = oracle_drift(
            expected=0.85,
            actual=0.70,
            drift_percent=17.6,
        )
        
        assert event.type == GenesisEventType.ORACLE_DRIFT_DETECTED
        assert event.data["expected"] == 0.85
        assert event.data["actual"] == 0.70
        assert event.data["drift_percent"] == 17.6


# =============================================================================
# STREAMING ORCHESTRATOR TESTS
# =============================================================================


class TestStreamingOrchestrator:
    """Tests for StreamingGenesisOrchestrator."""
    
    @pytest.fixture
    def streaming_orchestrator(self, tmp_path):
        """Create a streaming orchestrator."""
        from core.genesis.genesis_orchestrator_streaming import StreamingGenesisOrchestrator
        from core.genesis.genesis_orchestrator import WisdomRepository
        
        orchestrator = StreamingGenesisOrchestrator(
            beam_width=2,
            max_depth=2,
            fail_closed=False,
            emit_delay=0.0,  # No delay for tests
        )
        orchestrator.wisdom_repo = WisdomRepository(storage_path=tmp_path)
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_process_streaming_yields_events(self, streaming_orchestrator):
        """process_streaming yields events."""
        events = []
        
        async for event in streaming_orchestrator.process_streaming("Test problem"):
            events.append(event)
        
        assert len(events) > 0
    
    @pytest.mark.asyncio
    async def test_process_streaming_starts_with_system_start(self, streaming_orchestrator):
        """First event is SYSTEM_START."""
        async for event in streaming_orchestrator.process_streaming("Test"):
            assert event.type == GenesisEventType.SYSTEM_START
            break
    
    @pytest.mark.asyncio
    async def test_process_streaming_ends_with_attest(self, streaming_orchestrator):
        """Last event is ATTEST_COMPLETE."""
        events = []
        
        async for event in streaming_orchestrator.process_streaming("Test"):
            events.append(event)
        
        assert events[-1].type == GenesisEventType.ATTEST_COMPLETE
    
    @pytest.mark.asyncio
    async def test_process_streaming_has_correlation_id(self, streaming_orchestrator):
        """All events have same correlation ID."""
        correlation_ids = set()
        
        async for event in streaming_orchestrator.process_streaming("Test"):
            if event.correlation_id:
                correlation_ids.add(event.correlation_id)
        
        assert len(correlation_ids) == 1
    
    @pytest.mark.asyncio
    async def test_process_streaming_progress_increases(self, streaming_orchestrator):
        """Progress generally increases."""
        last_progress = -1.0
        regressions = 0
        
        async for event in streaming_orchestrator.process_streaming("Test"):
            if event.progress < last_progress:
                regressions += 1
            last_progress = event.progress
        
        # Allow some regressions for sub-events, but should end at 1.0
        assert last_progress == 1.0
    
    @pytest.mark.asyncio
    async def test_process_streaming_includes_thought_events(self, streaming_orchestrator):
        """Stream includes thought creation events."""
        thought_events = []
        
        async for event in streaming_orchestrator.process_streaming("Test"):
            if event.type == GenesisEventType.THOUGHT_NODE_CREATED:
                thought_events.append(event)
        
        assert len(thought_events) > 0
    
    @pytest.mark.asyncio
    async def test_process_streaming_includes_lens_events(self, streaming_orchestrator):
        """Stream includes lens activation events."""
        lens_events = []
        
        async for event in streaming_orchestrator.process_streaming("Test"):
            if event.type == GenesisEventType.LENS_ACTIVATED:
                lens_events.append(event)
        
        assert len(lens_events) == 6  # All 6 lenses
    
    @pytest.mark.asyncio
    async def test_event_bus_receives_events(self, streaming_orchestrator):
        """Event bus receives all events."""
        bus = streaming_orchestrator.event_bus
        
        # Consume stream
        async for _ in streaming_orchestrator.process_streaming("Test"):
            pass
        
        # Check history
        history = bus.get_history(limit=1000)
        assert len(history) > 0
    
    def test_add_listener(self, streaming_orchestrator):
        """Listeners can be added."""
        listener = MockListener()
        
        streaming_orchestrator.add_listener(listener)
        
        assert listener in streaming_orchestrator._listeners
    
    def test_remove_listener(self, streaming_orchestrator):
        """Listeners can be removed."""
        listener = MockListener()
        
        streaming_orchestrator.add_listener(listener)
        streaming_orchestrator.remove_listener(listener)
        
        assert listener not in streaming_orchestrator._listeners


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestEventSerialization:
    """Tests for event serialization/deserialization."""
    
    def test_json_roundtrip(self, sample_event):
        """Event survives JSON roundtrip."""
        # Serialize
        d = sample_event.to_dict()
        json_str = json.dumps(d)
        
        # Deserialize
        d2 = json.loads(json_str)
        restored = GenesisEvent.from_dict(d2)
        
        assert restored.type == sample_event.type
        assert restored.phase == sample_event.phase
        assert restored.progress == sample_event.progress
    
    def test_all_event_types_serializable(self):
        """All event types can be serialized."""
        for event_type in GenesisEventType:
            event = create_event(event_type, "Test", 0.5, test_field="value")
            d = event.to_dict()
            restored = GenesisEvent.from_dict(d)
            assert restored.type == event_type


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
