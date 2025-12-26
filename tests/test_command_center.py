r"""
Tests for BIZRA Node Zero Command Center.

Comprehensive test suite covering:
- Command creation and execution
- Ihsan compliance enforcement
- Event bus publish/subscribe
- Health aggregation
- Subsystem routing
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.command_center import (
    CommandId,
    Command,
    CommandResult,
    CommandType,
    SubsystemType,
    HealthLevel,
    Priority,
    SubsystemStatus,
    SystemHealthReport,
    Event,
    EventBus,
    IhsanEnforcer,
    CommandRouter,
    HealthAggregator,
    NodeZeroCommandCenter,
    DataLakeSubsystem,
    create_command_center,
)


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND ID TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCommandId:
    """Tests for CommandId class."""
    
    def test_command_id_creation(self):
        """Test that CommandId creates unique identifiers."""
        id1 = CommandId()
        id2 = CommandId()
        
        assert id1.value != id2.value
        assert len(id1.value) == 36  # UUID format
    
    def test_command_id_str(self):
        """Test CommandId string representation."""
        cmd_id = CommandId()
        assert len(str(cmd_id)) == 8  # Truncated
    
    def test_command_id_custom_value(self):
        """Test CommandId with custom value."""
        cmd_id = CommandId(value="custom-id-12345")
        assert cmd_id.value == "custom-id-12345"


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCommand:
    """Tests for Command class."""
    
    def test_command_creation(self):
        """Test basic command creation."""
        cmd = Command(
            id=CommandId(),
            command_type=CommandType.QUERY,
            target_subsystem=SubsystemType.DATA_LAKE,
            payload={"action": "status"},
        )
        
        assert cmd.command_type == CommandType.QUERY
        assert cmd.target_subsystem == SubsystemType.DATA_LAKE
        assert cmd.payload == {"action": "status"}
        assert cmd.priority == Priority.NORMAL
    
    def test_command_with_priority(self):
        """Test command with custom priority."""
        cmd = Command(
            id=CommandId(),
            command_type=CommandType.MUTATION,
            target_subsystem=SubsystemType.VERIFICATION,
            payload={},
            priority=Priority.CRITICAL,
        )
        
        assert cmd.priority == Priority.CRITICAL
    
    def test_command_ihsan_settings(self):
        """Test command Ihsan compliance settings."""
        cmd = Command(
            id=CommandId(),
            command_type=CommandType.VERIFY,
            target_subsystem=SubsystemType.ETHICS,
            payload={},
            requires_ihsan_check=True,
            min_ihsan_threshold=0.98,
        )
        
        assert cmd.requires_ihsan_check is True
        assert cmd.min_ihsan_threshold == 0.98
    
    def test_command_to_dict(self):
        """Test command serialization."""
        cmd = Command(
            id=CommandId(value="test-id-123"),
            command_type=CommandType.SCAN,
            target_subsystem=SubsystemType.DATA_LAKE,
            payload={"path": "/test"},
            actor="test_user",
        )
        
        data = cmd.to_dict()
        
        # Note: to_dict uses str(id) which truncates to 8 chars
        assert data["id"] == "test-id-"  # Truncated by __str__
        assert data["type"] == "SCAN"
        assert data["target"] == "DATA_LAKE"
        assert data["payload"]["path"] == "/test"
        assert data["actor"] == "test_user"
    
    def test_command_expiry(self):
        """Test command with expiry."""
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        cmd = Command(
            id=CommandId(),
            command_type=CommandType.QUERY,
            target_subsystem=SubsystemType.APEX,
            payload={},
            expires_at=future,
        )
        
        assert cmd.expires_at == future


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND RESULT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCommandResult:
    """Tests for CommandResult class."""
    
    def test_success_result(self):
        """Test successful command result."""
        result = CommandResult(
            command_id=CommandId(value="cmd-123"),
            success=True,
            data={"count": 100},
        )
        
        assert result.success is True
        assert result.data["count"] == 100
        assert result.error is None
    
    def test_error_result(self):
        """Test error command result."""
        result = CommandResult(
            command_id=CommandId(value="cmd-456"),
            success=False,
            error="Subsystem unavailable",
        )
        
        assert result.success is False
        assert result.error == "Subsystem unavailable"
    
    def test_result_metrics(self):
        """Test result with metrics."""
        result = CommandResult(
            command_id=CommandId(),
            success=True,
            execution_time_ms=15.5,
            ihsan_score=0.97,
            snr_score=0.85,
        )
        
        assert result.execution_time_ms == 15.5
        assert result.ihsan_score == 0.97
        assert result.snr_score == 0.85
    
    def test_result_to_dict(self):
        """Test result serialization."""
        result = CommandResult(
            command_id=CommandId(value="result-test"),
            success=True,
            data={"status": "ok"},
            executed_by=SubsystemType.DATA_LAKE,
        )
        
        data = result.to_dict()
        
        # Note: to_dict uses str(command_id) which truncates to 8 chars
        assert data["command_id"] == "result-t"  # Truncated by __str__
        assert data["success"] is True
        assert data["executed_by"] == "DATA_LAKE"


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT BUS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEventBus:
    """Tests for EventBus class."""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing."""
        return EventBus(buffer_size=100)
    
    @pytest.mark.asyncio
    async def test_publish_event(self, event_bus):
        """Test publishing an event."""
        event = Event(
            id="evt-1",
            event_type="test_event",
            source=SubsystemType.DATA_LAKE,
            payload={"data": "test"},
        )
        
        await event_bus.publish(event)
        
        assert event_bus.event_count == 1
    
    @pytest.mark.asyncio
    async def test_subscribe_and_receive(self, event_bus):
        """Test subscribing and receiving events."""
        received = []
        
        async def handler(event: Event):
            received.append(event)
        
        event_bus.subscribe("test_event", handler)
        
        event = Event(
            id="evt-2",
            event_type="test_event",
            source=SubsystemType.APEX,
            payload={},
        )
        
        await event_bus.publish(event)
        
        assert len(received) == 1
        assert received[0].id == "evt-2"
    
    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, event_bus):
        """Test wildcard event subscription."""
        received = []
        
        async def handler(event: Event):
            received.append(event)
        
        event_bus.subscribe("*", handler)
        
        await event_bus.publish(Event(
            id="evt-3",
            event_type="type_a",
            source=SubsystemType.APEX,
            payload={},
        ))
        
        await event_bus.publish(Event(
            id="evt-4",
            event_type="type_b",
            source=SubsystemType.APEX,
            payload={},
        ))
        
        assert len(received) == 2
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        received = []
        
        async def handler(event: Event):
            received.append(event)
        
        event_bus.subscribe("test_event", handler)
        event_bus.unsubscribe("test_event", handler)
        
        await event_bus.publish(Event(
            id="evt-5",
            event_type="test_event",
            source=SubsystemType.APEX,
            payload={},
        ))
        
        assert len(received) == 0
    
    def test_get_recent_events(self, event_bus):
        """Test getting recent events."""
        events = event_bus.get_recent_events(10)
        assert events == []


# ═══════════════════════════════════════════════════════════════════════════════
# IHSAN ENFORCER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIhsanEnforcer:
    """Tests for IhsanEnforcer class."""
    
    @pytest.fixture
    def enforcer(self):
        """Create enforcer for testing."""
        return IhsanEnforcer(threshold=0.95)
    
    @pytest.mark.asyncio
    async def test_compliant_command(self, enforcer):
        """Test command that passes compliance."""
        cmd = Command(
            id=CommandId(),
            command_type=CommandType.QUERY,
            target_subsystem=SubsystemType.DATA_LAKE,
            payload={"action": "status"},
            signal_strength=0.9,
            priority=Priority.HIGH,
            correlation_id="corr-123",
        )
        
        compliant, score, reason = await enforcer.check_compliance(cmd)
        
        assert compliant is True
        assert score >= 0.95
    
    @pytest.mark.asyncio
    async def test_non_compliant_command(self, enforcer):
        """Test command that fails compliance."""
        cmd = Command(
            id=CommandId(),
            command_type=CommandType.QUERY,
            target_subsystem=SubsystemType.DATA_LAKE,
            payload={},
            signal_strength=0.1,  # Very low signal
            priority=Priority.DEFERRED,
            min_ihsan_threshold=0.99,  # Very high threshold
        )
        
        compliant, score, reason = await enforcer.check_compliance(cmd)
        
        assert compliant is False
        assert "below threshold" in reason
    
    @pytest.mark.asyncio
    async def test_skip_check_when_not_required(self, enforcer):
        """Test that check is skipped when not required."""
        cmd = Command(
            id=CommandId(),
            command_type=CommandType.QUERY,
            target_subsystem=SubsystemType.DATA_LAKE,
            payload={},
            requires_ihsan_check=False,
        )
        
        compliant, score, reason = await enforcer.check_compliance(cmd)
        
        assert compliant is True
        assert score == 1.0
        assert reason == "Check not required"
    
    def test_compliance_rate(self, enforcer):
        """Test compliance rate calculation."""
        assert enforcer.compliance_rate == 1.0  # No checks yet
    
    def test_get_violations(self, enforcer):
        """Test getting violations list."""
        violations = enforcer.get_violations()
        assert violations == []


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND ROUTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCommandRouter:
    """Tests for CommandRouter class."""
    
    @pytest.fixture
    def router(self):
        """Create router for testing."""
        return CommandRouter()
    
    def test_register_subsystem(self, router):
        """Test registering a subsystem."""
        mock_subsystem = MagicMock()
        mock_subsystem.subsystem_type = SubsystemType.DATA_LAKE
        
        router.register_subsystem(mock_subsystem)
        
        assert SubsystemType.DATA_LAKE in router.get_registered_subsystems()
    
    def test_unregister_subsystem(self, router):
        """Test unregistering a subsystem."""
        mock_subsystem = MagicMock()
        mock_subsystem.subsystem_type = SubsystemType.DATA_LAKE
        
        router.register_subsystem(mock_subsystem)
        router.unregister_subsystem(SubsystemType.DATA_LAKE)
        
        assert SubsystemType.DATA_LAKE not in router.get_registered_subsystems()
    
    @pytest.mark.asyncio
    async def test_route_to_registered_subsystem(self, router):
        """Test routing to registered subsystem."""
        mock_subsystem = MagicMock()
        mock_subsystem.subsystem_type = SubsystemType.DATA_LAKE
        mock_subsystem.execute = AsyncMock(return_value=CommandResult(
            command_id=CommandId(),
            success=True,
            data={"routed": True},
        ))
        
        router.register_subsystem(mock_subsystem)
        
        cmd = Command(
            id=CommandId(),
            command_type=CommandType.QUERY,
            target_subsystem=SubsystemType.DATA_LAKE,
            payload={},
        )
        
        result = await router.route(cmd)
        
        assert result.success is True
        mock_subsystem.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_route_to_unregistered_subsystem(self, router):
        """Test routing to unregistered subsystem fails."""
        cmd = Command(
            id=CommandId(),
            command_type=CommandType.QUERY,
            target_subsystem=SubsystemType.DATA_LAKE,
            payload={},
        )
        
        result = await router.route(cmd)
        
        assert result.success is False
        assert "not registered" in result.error
    
    def test_router_stats(self, router):
        """Test router statistics."""
        stats = router.stats
        
        assert stats["routed"] == 0
        assert stats["failed"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH AGGREGATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthAggregator:
    """Tests for HealthAggregator class."""
    
    @pytest.fixture
    def aggregator(self):
        """Create aggregator for testing."""
        return HealthAggregator()
    
    def test_update_status(self, aggregator):
        """Test updating subsystem status."""
        status = SubsystemStatus(
            subsystem=SubsystemType.DATA_LAKE,
            health=HealthLevel.HEALTHY,
            uptime_seconds=3600,
            last_heartbeat=datetime.now(timezone.utc),
        )
        
        aggregator.update_status(status)
        
        report = aggregator.get_report()
        assert SubsystemType.DATA_LAKE in report.subsystems
    
    def test_health_degradation_alert(self, aggregator):
        """Test alert on health degradation."""
        # Initial healthy status
        aggregator.update_status(SubsystemStatus(
            subsystem=SubsystemType.DATA_LAKE,
            health=HealthLevel.HEALTHY,
            uptime_seconds=100,
            last_heartbeat=datetime.now(timezone.utc),
        ))
        
        # Degraded status
        aggregator.update_status(SubsystemStatus(
            subsystem=SubsystemType.DATA_LAKE,
            health=HealthLevel.CRITICAL,
            uptime_seconds=110,
            last_heartbeat=datetime.now(timezone.utc),
        ))
        
        report = aggregator.get_report()
        assert len(report.alerts) > 0
    
    def test_overall_health_calculation(self, aggregator):
        """Test overall health calculation."""
        aggregator.update_status(SubsystemStatus(
            subsystem=SubsystemType.DATA_LAKE,
            health=HealthLevel.OPTIMAL,
            uptime_seconds=100,
            last_heartbeat=datetime.now(timezone.utc),
        ))
        
        aggregator.update_status(SubsystemStatus(
            subsystem=SubsystemType.KNOWLEDGE_GRAPH,
            health=HealthLevel.OPTIMAL,
            uptime_seconds=100,
            last_heartbeat=datetime.now(timezone.utc),
        ))
        
        report = aggregator.get_report()
        assert report.overall_health == HealthLevel.OPTIMAL
    
    def test_empty_report(self, aggregator):
        """Test report with no subsystems."""
        report = aggregator.get_report()
        assert report.overall_health == HealthLevel.OFFLINE


# ═══════════════════════════════════════════════════════════════════════════════
# NODE ZERO COMMAND CENTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestNodeZeroCommandCenter:
    """Tests for NodeZeroCommandCenter class."""
    
    @pytest.fixture
    def command_center(self):
        """Create command center for testing."""
        return NodeZeroCommandCenter(ihsan_threshold=0.95)
    
    def test_initialization(self, command_center):
        """Test command center initialization."""
        assert command_center.VERSION == "1.0.0"
        assert command_center.event_bus is not None
        assert command_center.ihsan_enforcer is not None
        assert command_center.router is not None
    
    def test_register_subsystem(self, command_center):
        """Test subsystem registration."""
        mock_subsystem = MagicMock()
        mock_subsystem.subsystem_type = SubsystemType.DATA_LAKE
        
        command_center.register_subsystem(mock_subsystem)
        
        assert SubsystemType.DATA_LAKE in command_center.list_subsystems()
    
    @pytest.mark.asyncio
    async def test_execute_command(self, command_center):
        """Test command execution."""
        mock_subsystem = MagicMock()
        mock_subsystem.subsystem_type = SubsystemType.DATA_LAKE
        mock_subsystem.execute = AsyncMock(return_value=CommandResult(
            command_id=CommandId(),
            success=True,
        ))
        
        command_center.register_subsystem(mock_subsystem)
        
        cmd = command_center.create_command(
            command_type=CommandType.QUERY,
            target=SubsystemType.DATA_LAKE,
            payload={"action": "status"},
            signal_strength=1.0,  # Max signal strength to pass Ihsan check
        )
        # Add correlation_id to boost Ihsan score
        cmd.correlation_id = "test-correlation"
        
        result = await command_center.execute(cmd)
        
        assert result.success is True
        assert result.ihsan_score > 0
    
    @pytest.mark.asyncio
    async def test_execute_batch(self, command_center):
        """Test batch command execution."""
        mock_subsystem = MagicMock()
        mock_subsystem.subsystem_type = SubsystemType.DATA_LAKE
        mock_subsystem.execute = AsyncMock(return_value=CommandResult(
            command_id=CommandId(),
            success=True,
        ))
        
        command_center.register_subsystem(mock_subsystem)
        
        commands = []
        for i in range(3):
            cmd = command_center.create_command(
                command_type=CommandType.QUERY,
                target=SubsystemType.DATA_LAKE,
                payload={"id": i},
                signal_strength=1.0,  # Max signal strength
            )
            cmd.correlation_id = f"batch-{i}"  # Boost Ihsan score
            commands.append(cmd)
        
        results = await command_center.execute_batch(commands)
        
        assert len(results) == 3
        assert all(r.success for r in results)
    
    def test_get_statistics(self, command_center):
        """Test getting statistics."""
        stats = command_center.get_statistics()
        
        assert stats["version"] == "1.0.0"
        assert stats["total_commands"] == 0
        assert stats["registered_subsystems"] == 0
    
    def test_get_health_report(self, command_center):
        """Test getting health report."""
        report = command_center.get_health_report()
        
        assert isinstance(report, SystemHealthReport)
    
    def test_create_command(self, command_center):
        """Test command factory method."""
        cmd = command_center.create_command(
            command_type=CommandType.SCAN,
            target=SubsystemType.DATA_LAKE,
            payload={"path": "/test"},
            priority=Priority.HIGH,
        )
        
        assert cmd.command_type == CommandType.SCAN
        assert cmd.target_subsystem == SubsystemType.DATA_LAKE
        assert cmd.priority == Priority.HIGH
    
    def test_get_audit_log(self, command_center):
        """Test getting audit log."""
        log = command_center.get_audit_log()
        assert log == []


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LAKE SUBSYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataLakeSubsystem:
    """Tests for DataLakeSubsystem adapter."""
    
    def test_subsystem_type(self):
        """Test subsystem type property."""
        subsystem = DataLakeSubsystem()
        assert subsystem.subsystem_type == SubsystemType.DATA_LAKE
    
    @pytest.mark.asyncio
    async def test_health_check_no_watcher(self):
        """Test health check without watcher."""
        subsystem = DataLakeSubsystem()
        status = await subsystem.health_check()
        
        assert status.subsystem == SubsystemType.DATA_LAKE
        assert status.health == HealthLevel.DEGRADED
    
    @pytest.mark.asyncio
    async def test_execute_without_watcher(self):
        """Test execute without watcher fails."""
        subsystem = DataLakeSubsystem()
        
        cmd = Command(
            id=CommandId(),
            command_type=CommandType.SCAN,
            target_subsystem=SubsystemType.DATA_LAKE,
            payload={"action": "scan"},
        )
        
        result = await subsystem.execute(cmd)
        
        assert result.success is False
        assert "not initialized" in result.error


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFactory:
    """Tests for factory functions."""
    
    def test_create_command_center(self):
        """Test command center factory."""
        center = create_command_center(with_data_lake=False)
        
        assert isinstance(center, NodeZeroCommandCenter)
        assert center.list_subsystems() == []


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for command center."""
    
    @pytest.mark.asyncio
    async def test_full_command_flow(self):
        """Test complete command flow through system."""
        center = NodeZeroCommandCenter()
        
        # Create mock subsystem
        mock_subsystem = MagicMock()
        mock_subsystem.subsystem_type = SubsystemType.DATA_LAKE
        mock_subsystem.execute = AsyncMock(return_value=CommandResult(
            command_id=CommandId(),
            success=True,
            data={"processed": True},
        ))
        mock_subsystem.health_check = AsyncMock(return_value=SubsystemStatus(
            subsystem=SubsystemType.DATA_LAKE,
            health=HealthLevel.HEALTHY,
            uptime_seconds=100,
            last_heartbeat=datetime.now(timezone.utc),
        ))
        
        center.register_subsystem(mock_subsystem)
        
        # Execute command with high signal and correlation for Ihsan compliance
        cmd = center.create_command(
            command_type=CommandType.QUERY,
            target=SubsystemType.DATA_LAKE,
            payload={"action": "test"},
            signal_strength=1.0,  # Max signal
        )
        cmd.correlation_id = "flow-test"  # Boost Ihsan score
        
        result = await center.execute(cmd)
        
        # Verify
        assert result.success is True
        assert center.event_bus.event_count > 0  # Event emitted
        
        stats = center.get_statistics()
        assert stats["total_commands"] == 1
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with NodeZeroCommandCenter() as center:
            assert center is not None
            stats = center.get_statistics()
            assert "version" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
