"""
BIZRA AEON OMEGA - Membrane (Typed Message Passing)
=====================================================
Serialization Boundary Between Hemispheres

The Membrane is the "corpus callosum" of the Bicameral Engine - the typed
message passing interface between Cold Core (Rust) and Warm Surface (Python).

Characteristics:
    - Strict type enforcement
    - Serialization/deserialization boundary
    - Audit logging of all crossings
    - Latency monitoring
    - Backpressure handling

Protocol:
    1. Warm → Cold: proposal → verification → result
    2. Cold → Warm: command → execution → evidence

Author: BIZRA Genesis Team (Peak Masterpiece v5)
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

logger = logging.getLogger("bizra.bicameral.membrane")

# ============================================================================
# CONSTANTS
# ============================================================================

# Maximum message size (1 MB)
MAX_MESSAGE_SIZE_BYTES = 1_000_000

# Queue capacity
MESSAGE_QUEUE_CAPACITY = 1000

# Crossing latency budget (should be minimal)
CROSSING_LATENCY_BUDGET_MS = 5.0

# ============================================================================
# ENUMERATIONS
# ============================================================================


class MessageType(Enum):
    """Types of messages that can cross the Membrane."""
    
    # Verification requests (Warm → Cold)
    VERIFY_PROPOSAL = auto()
    CHECK_INVARIANT = auto()
    SIGN_DATA = auto()
    HASH_DATA = auto()
    
    # Verification responses (Cold → Warm)
    VERIFY_RESULT = auto()
    INVARIANT_RESULT = auto()
    SIGNATURE_RESULT = auto()
    HASH_RESULT = auto()
    
    # Command requests (Cold → Warm)
    EXECUTE_TASK = auto()
    SPAWN_AGENT = auto()
    INVOKE_TOOL = auto()
    
    # Command responses (Warm → Cold)
    TASK_RESULT = auto()
    AGENT_RESULT = auto()
    TOOL_RESULT = auto()
    
    # System messages
    HEARTBEAT = auto()
    ERROR = auto()
    ACK = auto()


class MessagePriority(Enum):
    """Priority levels for messages."""
    
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class CrossingDirection(Enum):
    """Direction of message crossing."""
    
    WARM_TO_COLD = auto()  # Request verification/crypto
    COLD_TO_WARM = auto()  # Request execution/action


class MembraneState(Enum):
    """State of the Membrane."""
    
    OPEN = auto()        # Accepting messages
    THROTTLED = auto()   # Backpressure applied
    CLOSED = auto()      # Not accepting messages


# ============================================================================
# DATA CLASSES
# ============================================================================

T = TypeVar("T")


@dataclass
class MembraneConfig:
    """Configuration for Membrane."""
    
    max_message_size: int = MAX_MESSAGE_SIZE_BYTES
    queue_capacity: int = MESSAGE_QUEUE_CAPACITY
    crossing_latency_budget_ms: float = CROSSING_LATENCY_BUDGET_MS
    enable_compression: bool = False
    enable_encryption: bool = False
    enable_audit_logging: bool = True
    throttle_threshold: float = 0.8  # Start throttling at 80% capacity


@dataclass
class Message:
    """A message crossing the Membrane."""
    
    message_id: str
    message_type: MessageType
    direction: CrossingDirection
    priority: MessagePriority
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None  # For request-response pairing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_ms: Optional[float] = None  # Time-to-live
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.name,
            "direction": self.direction.name,
            "priority": self.priority.name,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "ttl_ms": self.ttl_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType[data["message_type"]],
            direction=CrossingDirection[data["direction"]],
            priority=MessagePriority[data["priority"]],
            payload=data["payload"],
            correlation_id=data.get("correlation_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            ttl_ms=data.get("ttl_ms"),
        )
    
    def size_bytes(self) -> int:
        """Calculate approximate message size in bytes."""
        return len(json.dumps(self.to_dict()).encode())


@dataclass
class CrossingReceipt:
    """Receipt for a message crossing the Membrane."""
    
    receipt_id: str
    message_id: str
    direction: CrossingDirection
    crossing_latency_ms: float
    within_budget: bool
    source_hash: str
    destination_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "receipt_id": self.receipt_id,
            "message_id": self.message_id,
            "direction": self.direction.name,
            "crossing_latency_ms": self.crossing_latency_ms,
            "within_budget": self.within_budget,
            "source_hash": self.source_hash,
            "destination_hash": self.destination_hash,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PendingRequest:
    """A pending request awaiting response."""
    
    message: Message
    sent_at: datetime
    future: asyncio.Future
    timeout_ms: float = 30000.0  # 30 second default timeout


# ============================================================================
# MEMBRANE IMPLEMENTATION
# ============================================================================


class Membrane:
    """
    The Membrane - Typed Message Passing Interface.
    
    Handles all communication between Cold Core and Warm Surface:
    - Type enforcement
    - Serialization boundary
    - Audit logging
    - Backpressure handling
    
    Usage:
        membrane = Membrane()
        
        # Send message from Warm to Cold
        receipt = await membrane.send(message, CrossingDirection.WARM_TO_COLD)
        
        # Request-response pattern
        response = await membrane.request(
            MessageType.VERIFY_PROPOSAL,
            {"proposal": proposal_data},
            CrossingDirection.WARM_TO_COLD,
        )
    """
    
    def __init__(self, config: Optional[MembraneConfig] = None):
        """Initialize Membrane."""
        self.config = config or MembraneConfig()
        
        # Message queues (one per direction)
        self._warm_to_cold_queue: asyncio.Queue[Message] = asyncio.Queue(
            maxsize=self.config.queue_capacity
        )
        self._cold_to_warm_queue: asyncio.Queue[Message] = asyncio.Queue(
            maxsize=self.config.queue_capacity
        )
        
        # Pending requests (for request-response pattern)
        self._pending_requests: Dict[str, PendingRequest] = {}
        
        # Message handlers
        self._handlers: Dict[MessageType, Callable] = {}
        
        # State
        self._state = MembraneState.OPEN
        
        # Audit log
        self._crossing_log: List[CrossingReceipt] = []
        
        # Statistics
        self._crossings: int = 0
        self._within_budget: int = 0
        self._bytes_transferred: int = 0
        
        # Message counter
        self._message_counter: int = 0
        self._receipt_counter: int = 0
        
        logger.info("Membrane initialized")
    
    # ========================================================================
    # SEND/RECEIVE
    # ========================================================================
    
    async def send(
        self,
        message: Message,
        timeout: Optional[float] = None,
    ) -> CrossingReceipt:
        """
        Send a message across the Membrane.
        
        Args:
            message: Message to send
            timeout: Optional timeout in seconds
            
        Returns:
            CrossingReceipt confirming the crossing
            
        Raises:
            MembraneClosedException: If membrane is closed
            MessageTooLargeException: If message exceeds size limit
            QueueFullException: If queue is full
        """
        start = time.perf_counter()
        
        # Check state
        if self._state == MembraneState.CLOSED:
            raise MembraneClosedException("Membrane is closed")
        
        # Check message size
        msg_size = message.size_bytes()
        if msg_size > self.config.max_message_size:
            raise MessageTooLargeException(
                f"Message size {msg_size} exceeds limit {self.config.max_message_size}"
            )
        
        # Compute source hash
        source_hash = self._hash(json.dumps(message.to_dict()))
        
        # Get appropriate queue
        if message.direction == CrossingDirection.WARM_TO_COLD:
            q = self._warm_to_cold_queue
        else:
            q = self._cold_to_warm_queue
        
        # Check for throttling
        queue_usage = q.qsize() / self.config.queue_capacity
        if queue_usage >= self.config.throttle_threshold:
            self._state = MembraneState.THROTTLED
            if message.priority == MessagePriority.LOW:
                raise QueueThrottledException("Queue is throttled, low priority rejected")
        elif self._state == MembraneState.THROTTLED:
            self._state = MembraneState.OPEN
        
        # Enqueue message
        try:
            if timeout is not None:
                await asyncio.wait_for(q.put(message), timeout=timeout)
            else:
                await q.put(message)
        except asyncio.QueueFull:
            raise QueueFullException("Message queue is full")
        except asyncio.TimeoutError:
            # P3 FIX: Normalize TimeoutError to QueueFullException for consistent API
            raise QueueFullException(f"Message queue put timed out after {timeout}s")
        
        # Compute destination hash (after any serialization)
        destination_hash = source_hash  # Same for now (no transformation)
        
        # Record crossing
        crossing_latency = (time.perf_counter() - start) * 1000
        within_budget = crossing_latency <= self.config.crossing_latency_budget_ms
        
        receipt = self._create_receipt(
            message.message_id,
            message.direction,
            crossing_latency,
            within_budget,
            source_hash,
            destination_hash,
        )
        
        self._crossings += 1
        if within_budget:
            self._within_budget += 1
        self._bytes_transferred += msg_size
        
        return receipt
    
    async def receive(
        self,
        direction: CrossingDirection,
        timeout: Optional[float] = None,
    ) -> Optional[Message]:
        """
        Receive a message from the Membrane.
        
        Args:
            direction: Direction to receive from
            timeout: Optional timeout in seconds
            
        Returns:
            Message if available, None on timeout
        """
        # Get appropriate queue
        if direction == CrossingDirection.WARM_TO_COLD:
            q = self._warm_to_cold_queue
        else:
            q = self._cold_to_warm_queue
        
        try:
            if timeout is not None:
                message = await asyncio.wait_for(q.get(), timeout=timeout)
            else:
                message = await q.get()
            
            # Check TTL
            if message.ttl_ms is not None:
                elapsed = (datetime.now(timezone.utc) - message.timestamp).total_seconds() * 1000
                if elapsed > message.ttl_ms:
                    logger.warning(f"Message {message.message_id} expired (TTL)")
                    return None
            
            return message
        except asyncio.TimeoutError:
            return None
    
    # ========================================================================
    # REQUEST-RESPONSE PATTERN
    # ========================================================================
    
    async def request(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        direction: CrossingDirection,
        priority: MessagePriority = MessagePriority.NORMAL,
        timeout_ms: float = 30000.0,
    ) -> Dict[str, Any]:
        """
        Send a request and wait for response.
        
        Args:
            message_type: Type of request
            payload: Request payload
            direction: Direction to send
            priority: Message priority
            timeout_ms: Timeout in milliseconds
            
        Returns:
            Response payload
            
        Raises:
            TimeoutError: If no response within timeout
            MembraneException: On other errors
        """
        # Create message
        message = self._create_message(
            message_type=message_type,
            direction=direction,
            priority=priority,
            payload=payload,
        )
        
        # Create pending request with future
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Dict[str, Any]] = loop.create_future()
        
        pending = PendingRequest(
            message=message,
            sent_at=datetime.now(timezone.utc),
            future=future,
            timeout_ms=timeout_ms,
        )
        
        self._pending_requests[message.message_id] = pending
        
        try:
            # Send message
            await self.send(message)
            
            # Wait for response
            response = await asyncio.wait_for(
                future,
                timeout=timeout_ms / 1000.0,
            )
            
            return response
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request {message.message_id} timed out")
        finally:
            self._pending_requests.pop(message.message_id, None)
    
    async def respond(
        self,
        correlation_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Send a response to a pending request.
        
        Args:
            correlation_id: Original request message ID
            payload: Response payload
        """
        if correlation_id in self._pending_requests:
            pending = self._pending_requests[correlation_id]
            if not pending.future.done():
                pending.future.set_result(payload)
    
    # ========================================================================
    # MESSAGE HANDLERS
    # ========================================================================
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable,
    ) -> None:
        """
        Register a handler for a message type.
        
        Args:
            message_type: Message type to handle
            handler: Handler function (async or sync)
        """
        self._handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type.name}")
    
    async def dispatch(self, message: Message) -> Optional[Dict[str, Any]]:
        """
        Dispatch a message to its handler.
        
        Args:
            message: Message to dispatch
            
        Returns:
            Handler result if any
        """
        handler = self._handlers.get(message.message_type)
        if handler is None:
            logger.warning(f"No handler for message type: {message.message_type.name}")
            return None
        
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(message.payload)
            else:
                result = handler(message.payload)
            
            return result
        except Exception as e:
            logger.error(f"Handler error for {message.message_type.name}: {e}")
            return {"error": str(e)}
    
    # ========================================================================
    # PIPELINE
    # ========================================================================
    
    async def run_pipeline(self, direction: CrossingDirection) -> None:
        """
        Run the message processing pipeline.
        
        This continuously receives messages and dispatches them to handlers.
        
        Args:
            direction: Direction to process
        """
        logger.info(f"Starting pipeline for {direction.name}")
        
        while self._state != MembraneState.CLOSED:
            try:
                message = await self.receive(direction, timeout=1.0)
                if message is None:
                    continue
                
                result = await self.dispatch(message)
                
                # If this is a request, send response
                if message.correlation_id:
                    await self.respond(message.correlation_id, result or {})
                elif message.message_id in self._pending_requests:
                    await self.respond(message.message_id, result or {})
                    
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error
    
    # ========================================================================
    # LIFECYCLE
    # ========================================================================
    
    def open(self) -> None:
        """Open the Membrane for message passing."""
        self._state = MembraneState.OPEN
        logger.info("Membrane opened")
    
    def close(self) -> None:
        """Close the Membrane."""
        self._state = MembraneState.CLOSED
        
        # Cancel pending requests
        for pending in self._pending_requests.values():
            if not pending.future.done():
                pending.future.cancel()
        
        logger.info("Membrane closed")
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _create_message(
        self,
        message_type: MessageType,
        direction: CrossingDirection,
        priority: MessagePriority,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        ttl_ms: Optional[float] = None,
    ) -> Message:
        """Create a new message."""
        self._message_counter += 1
        message_id = f"MSG-{int(time.time()*1000000)}-{self._message_counter:06d}"
        
        return Message(
            message_id=message_id,
            message_type=message_type,
            direction=direction,
            priority=priority,
            payload=payload,
            correlation_id=correlation_id,
            ttl_ms=ttl_ms,
        )
    
    def _create_receipt(
        self,
        message_id: str,
        direction: CrossingDirection,
        crossing_latency_ms: float,
        within_budget: bool,
        source_hash: str,
        destination_hash: str,
    ) -> CrossingReceipt:
        """Create a crossing receipt."""
        self._receipt_counter += 1
        receipt_id = f"RCP-{int(time.time()*1000000)}-{self._receipt_counter:06d}"
        
        receipt = CrossingReceipt(
            receipt_id=receipt_id,
            message_id=message_id,
            direction=direction,
            crossing_latency_ms=crossing_latency_ms,
            within_budget=within_budget,
            source_hash=source_hash,
            destination_hash=destination_hash,
        )
        
        if self.config.enable_audit_logging:
            self._crossing_log.append(receipt)
            # Keep last 10000 receipts
            if len(self._crossing_log) > 10000:
                self._crossing_log = self._crossing_log[-10000:]
        
        return receipt
    
    def _hash(self, data: str) -> str:
        """Compute hash of data."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Membrane statistics."""
        return {
            "state": self._state.name,
            "crossings": self._crossings,
            "within_budget": self._within_budget,
            "budget_compliance_rate": self._within_budget / max(self._crossings, 1),
            "bytes_transferred": self._bytes_transferred,
            "pending_requests": len(self._pending_requests),
            "warm_to_cold_queue_size": self._warm_to_cold_queue.qsize(),
            "cold_to_warm_queue_size": self._cold_to_warm_queue.qsize(),
            "registered_handlers": len(self._handlers),
        }
    
    def get_crossing_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent crossing log entries."""
        return [r.to_dict() for r in self._crossing_log[-limit:]]


# ============================================================================
# EXCEPTIONS
# ============================================================================


class MembraneException(Exception):
    """Base exception for Membrane errors."""
    pass


class MembraneClosedException(MembraneException):
    """Raised when Membrane is closed."""
    pass


class MessageTooLargeException(MembraneException):
    """Raised when message exceeds size limit."""
    pass


class QueueFullException(MembraneException):
    """Raised when message queue is full."""
    pass


class QueueThrottledException(MembraneException):
    """Raised when queue is throttled."""
    pass


# ============================================================================
# SELF-TEST
# ============================================================================


async def _self_test() -> None:
    """Run Membrane self-tests."""
    print("=" * 70)
    print("BIZRA AEON OMEGA - Membrane Self-Test")
    print("=" * 70)
    
    membrane = Membrane()
    
    # Test 1: Create message
    print("\n[Test 1] Create Message")
    message = membrane._create_message(
        message_type=MessageType.VERIFY_PROPOSAL,
        direction=CrossingDirection.WARM_TO_COLD,
        priority=MessagePriority.NORMAL,
        payload={"proposal": "test"},
    )
    assert message.message_id.startswith("MSG-")
    print(f"  ✓ Created message: {message.message_id}")
    print(f"  ✓ Size: {message.size_bytes()} bytes")
    
    # Test 2: Send message
    print("\n[Test 2] Send Message")
    receipt = await membrane.send(message)
    assert receipt.within_budget
    print(f"  ✓ Message sent: {receipt.message_id}")
    print(f"  ✓ Crossing latency: {receipt.crossing_latency_ms:.3f}ms")
    
    # Test 3: Receive message
    print("\n[Test 3] Receive Message")
    received = await membrane.receive(CrossingDirection.WARM_TO_COLD, timeout=1.0)
    assert received is not None
    assert received.message_id == message.message_id
    print(f"  ✓ Received message: {received.message_id}")
    
    # Test 4: Register handler
    print("\n[Test 4] Register Handler")
    handler_called = []
    def test_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        handler_called.append(payload)
        return {"handled": True}
    
    membrane.register_handler(MessageType.VERIFY_PROPOSAL, test_handler)
    print(f"  ✓ Registered handler for VERIFY_PROPOSAL")
    
    # Test 5: Dispatch message
    print("\n[Test 5] Dispatch Message")
    result = await membrane.dispatch(message)
    assert result == {"handled": True}
    assert len(handler_called) == 1
    print(f"  ✓ Handler called with payload: {handler_called[0]}")
    
    # Test 6: Statistics
    print("\n[Test 6] Statistics")
    stats = membrane.get_statistics()
    print(f"  ✓ Crossings: {stats['crossings']}")
    print(f"  ✓ Budget compliance: {stats['budget_compliance_rate']*100:.1f}%")
    print(f"  ✓ Bytes transferred: {stats['bytes_transferred']}")
    
    # Test 7: Close membrane
    print("\n[Test 7] Close Membrane")
    membrane.close()
    assert membrane._state == MembraneState.CLOSED
    print(f"  ✓ Membrane closed")
    
    print("\n" + "=" * 70)
    print("✅ ALL MEMBRANE SELF-TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(_self_test())
