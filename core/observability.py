"""
BIZRA Observability Telemetry System
=====================================
Production-Grade OpenTelemetry-Compatible Distributed Tracing

This module implements a comprehensive observability stack for BIZRA,
providing full visibility into the DDAGI runtime with:

- Distributed Tracing (W3C Trace Context compatible)
- Metrics Collection (counters, gauges, histograms)
- Structured Logging (with trace correlation)
- Ihsan Compliance Monitoring
- Performance Profiling

OpenTelemetry Semantic Conventions:
    https://opentelemetry.io/docs/concepts/semantic-conventions/

Author: BIZRA Genesis Team
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger("bizra.telemetry")

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

T = TypeVar("T")
MetricValue = Union[int, float]
Attributes = Dict[str, Union[str, int, float, bool]]

# ============================================================================
# ENUMERATIONS
# ============================================================================


class SpanKind(Enum):
    """OpenTelemetry span kinds."""

    INTERNAL = auto()  # Internal operation
    SERVER = auto()  # Server-side handling
    CLIENT = auto()  # Client-side request
    PRODUCER = auto()  # Message producer
    CONSUMER = auto()  # Message consumer


class SpanStatus(Enum):
    """Span completion status."""

    UNSET = auto()
    OK = auto()
    ERROR = auto()


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = auto()  # Monotonically increasing
    UP_DOWN_COUNTER = auto()  # Can increase/decrease
    GAUGE = auto()  # Point-in-time value
    HISTOGRAM = auto()  # Distribution of values


class LogLevel(Enum):
    """Structured log levels."""

    TRACE = 0
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    FATAL = 50


# ============================================================================
# TRACE CONTEXT (W3C COMPATIBLE)
# ============================================================================


@dataclass(frozen=True)
class TraceContext:
    """
    W3C Trace Context for distributed tracing.

    Format: {version}-{trace_id}-{span_id}-{trace_flags}
    Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
    """

    trace_id: str  # 32 hex chars (128 bits)
    span_id: str  # 16 hex chars (64 bits)
    parent_span_id: Optional[str]
    trace_flags: int  # 1 byte flags (sampled, etc.)

    @staticmethod
    def new_root() -> TraceContext:
        """Create a new root trace context."""
        return TraceContext(
            trace_id=secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            parent_span_id=None,
            trace_flags=1,  # Sampled
        )

    def new_child(self) -> TraceContext:
        """Create a child context."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=secrets.token_hex(8),
            parent_span_id=self.span_id,
            trace_flags=self.trace_flags,
        )

    def to_traceparent(self) -> str:
        """Export as W3C traceparent header."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    @staticmethod
    def from_traceparent(header: str) -> Optional[TraceContext]:
        """Parse W3C traceparent header."""
        try:
            parts = header.split("-")
            if len(parts) != 4 or parts[0] != "00":
                return None

            return TraceContext(
                trace_id=parts[1],
                span_id=parts[2],
                parent_span_id=None,  # Unknown in header
                trace_flags=int(parts[3], 16),
            )
        except (ValueError, IndexError):
            return None

    @property
    def is_sampled(self) -> bool:
        return bool(self.trace_flags & 0x01)


# ============================================================================
# SPANS
# ============================================================================


@dataclass
class SpanEvent:
    """Event recorded during span execution."""

    name: str
    timestamp: datetime
    attributes: Attributes


@dataclass
class SpanLink:
    """Link to another span (causal relationship)."""

    context: TraceContext
    attributes: Attributes


@dataclass
class Span:
    """
    A single unit of work in a distributed trace.

    Spans form a tree structure representing the
    execution path through the system.
    """

    name: str
    context: TraceContext
    kind: SpanKind
    start_time: datetime
    end_time: Optional[datetime]
    status: SpanStatus
    status_message: str
    attributes: Dict[str, Any]
    events: List[SpanEvent]
    links: List[SpanLink]

    # Ihsan-specific
    ihsan_score: float
    apex_layer: str

    @staticmethod
    def create(
        name: str,
        context: TraceContext,
        kind: SpanKind = SpanKind.INTERNAL,
        ihsan_score: float = 0.95,
        apex_layer: str = "UNKNOWN",
        attributes: Optional[Attributes] = None,
    ) -> Span:
        return Span(
            name=name,
            context=context,
            kind=kind,
            start_time=datetime.now(timezone.utc),
            end_time=None,
            status=SpanStatus.UNSET,
            status_message="",
            attributes=dict(attributes or {}),
            events=[],
            links=[],
            ihsan_score=ihsan_score,
            apex_layer=apex_layer,
        )

    def add_event(
        self,
        name: str,
        attributes: Optional[Attributes] = None,
    ) -> None:
        """Add an event to this span."""
        self.events.append(
            SpanEvent(
                name=name,
                timestamp=datetime.now(timezone.utc),
                attributes=dict(attributes or {}),
            )
        )

    def add_link(
        self,
        context: TraceContext,
        attributes: Optional[Attributes] = None,
    ) -> None:
        """Add a link to another span."""
        self.links.append(
            SpanLink(
                context=context,
                attributes=dict(attributes or {}),
            )
        )

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_status(
        self,
        status: SpanStatus,
        message: str = "",
    ) -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.now(timezone.utc)
        if self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Export span as dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "kind": self.kind.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.name,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
            "ihsan_score": self.ihsan_score,
            "apex_layer": self.apex_layer,
        }


# ============================================================================
# TRACER
# ============================================================================


class Tracer:
    """
    Distributed tracer for creating and managing spans.

    Provides context propagation and span lifecycle management.
    """

    def __init__(
        self,
        service_name: str = "bizra",
        exporter: Optional[SpanExporter] = None,
    ):
        self.service_name = service_name
        self._exporter = exporter or InMemorySpanExporter()

        # Context storage (thread-local for sync, task-local for async)
        self._context_var: Optional[TraceContext] = None
        self._active_span: Optional[Span] = None
        self._span_stack: List[Span] = []

        # Metrics
        self._spans_created = 0
        self._spans_exported = 0

    def get_current_context(self) -> Optional[TraceContext]:
        """Get current trace context."""
        return self._context_var

    def get_active_span(self) -> Optional[Span]:
        """Get currently active span."""
        return self._active_span

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[TraceContext] = None,
        ihsan_score: float = 0.95,
        apex_layer: str = "UNKNOWN",
        attributes: Optional[Attributes] = None,
    ) -> Span:
        """
        Start a new span.

        If no parent is provided, uses current context.
        If no context exists, creates a new root span.
        """
        # Determine parent context
        if parent is None:
            parent = self._context_var

        # Create context
        if parent is None:
            context = TraceContext.new_root()
        else:
            context = parent.new_child()

        # Create span
        span = Span.create(
            name=name,
            context=context,
            kind=kind,
            ihsan_score=ihsan_score,
            apex_layer=apex_layer,
            attributes=attributes,
        )

        # Add service name
        span.set_attribute("service.name", self.service_name)

        self._spans_created += 1

        return span

    def end_span(self, span: Span) -> None:
        """End a span and export it."""
        span.end()
        self._exporter.export(span)
        self._spans_exported += 1

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        ihsan_score: float = 0.95,
        apex_layer: str = "UNKNOWN",
        attributes: Optional[Attributes] = None,
    ) -> Generator[Span, None, None]:
        """
        Context manager for span lifecycle.

        Usage:
            with tracer.span("operation") as span:
                span.set_attribute("key", "value")
                # ... do work ...
        """
        span = self.start_span(
            name=name,
            kind=kind,
            ihsan_score=ihsan_score,
            apex_layer=apex_layer,
            attributes=attributes,
        )

        # Push to stack
        prev_span = self._active_span
        prev_context = self._context_var
        self._active_span = span
        self._context_var = span.context

        try:
            yield span
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event(
                "exception",
                {
                    "exception.type": type(e).__name__,
                    "exception.message": str(e),
                },
            )
            raise
        finally:
            self.end_span(span)
            self._active_span = prev_span
            self._context_var = prev_context

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        ihsan_score: float = 0.95,
        apex_layer: str = "UNKNOWN",
        attributes: Optional[Attributes] = None,
    ):
        """Async context manager for spans."""
        span = self.start_span(
            name=name,
            kind=kind,
            ihsan_score=ihsan_score,
            apex_layer=apex_layer,
            attributes=attributes,
        )

        prev_span = self._active_span
        prev_context = self._context_var
        self._active_span = span
        self._context_var = span.context

        try:
            yield span
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event(
                "exception",
                {
                    "exception.type": type(e).__name__,
                    "exception.message": str(e),
                },
            )
            raise
        finally:
            self.end_span(span)
            self._active_span = prev_span
            self._context_var = prev_context

    def trace(
        self,
        name: Optional[str] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        ihsan_score: float = 0.95,
        apex_layer: str = "UNKNOWN",
    ):
        """
        Decorator for tracing functions.

        Usage:
            @tracer.trace("my_operation")
            async def my_function():
                ...
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            span_name = name or func.__name__

            if asyncio.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                    async with self.async_span(
                        span_name,
                        kind=kind,
                        ihsan_score=ihsan_score,
                        apex_layer=apex_layer,
                    ):
                        return await func(*args, **kwargs)

                return async_wrapper  # type: ignore
            else:

                @wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                    with self.span(
                        span_name,
                        kind=kind,
                        ihsan_score=ihsan_score,
                        apex_layer=apex_layer,
                    ):
                        return func(*args, **kwargs)

                return sync_wrapper  # type: ignore

        return decorator

    def get_metrics(self) -> Dict[str, Any]:
        """Get tracer metrics."""
        return {
            "service_name": self.service_name,
            "spans_created": self._spans_created,
            "spans_exported": self._spans_exported,
            "exporter_metrics": self._exporter.get_metrics(),
        }


# ============================================================================
# SPAN EXPORTERS
# ============================================================================


class SpanExporter(ABC):
    """Abstract span exporter."""

    @abstractmethod
    def export(self, span: Span) -> None:
        """Export a completed span."""
        ...

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get exporter metrics."""
        ...


class InMemorySpanExporter(SpanExporter):
    """
    In-memory span storage for testing and development.

    Keeps bounded number of spans in memory.
    """

    def __init__(self, max_spans: int = 10000):
        self._spans: Deque[Span] = deque(maxlen=max_spans)
        self._by_trace: Dict[str, List[Span]] = defaultdict(list)
        self._total_exported = 0
        self._lock = threading.Lock()

    def export(self, span: Span) -> None:
        with self._lock:
            self._spans.append(span)
            self._by_trace[span.context.trace_id].append(span)
            self._total_exported += 1

    def get_spans(self, limit: int = 100) -> List[Span]:
        """Get recent spans."""
        return list(self._spans)[-limit:]

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return self._by_trace.get(trace_id, [])

    def clear(self) -> None:
        """Clear all spans."""
        with self._lock:
            self._spans.clear()
            self._by_trace.clear()

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "type": "in_memory",
            "total_exported": self._total_exported,
            "current_spans": len(self._spans),
            "trace_count": len(self._by_trace),
        }


class ConsoleSpanExporter(SpanExporter):
    """Export spans to console (for development)."""

    def __init__(self, pretty: bool = True):
        self._pretty = pretty
        self._total_exported = 0

    def export(self, span: Span) -> None:
        self._total_exported += 1

        if self._pretty:
            indent = "  " * (len(span.context.trace_id) % 4)
            status_icon = "✓" if span.status == SpanStatus.OK else "✗"
            print(
                f"{indent}{status_icon} {span.name} "
                f"[{span.apex_layer}] "
                f"{span.duration_ms:.2f}ms "
                f"(ihsan={span.ihsan_score:.2f})"
            )
        else:
            print(json.dumps(span.to_dict()))

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "type": "console",
            "total_exported": self._total_exported,
        }


class CompositeSpanExporter(SpanExporter):
    """Export spans to multiple exporters."""

    def __init__(self, exporters: List[SpanExporter]):
        self._exporters = exporters

    def export(self, span: Span) -> None:
        for exporter in self._exporters:
            try:
                exporter.export(span)
            except Exception as e:
                logger.error(f"Exporter error: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "type": "composite",
            "exporters": [e.get_metrics() for e in self._exporters],
        }


# ============================================================================
# METRICS
# ============================================================================


@dataclass
class MetricPoint:
    """A single metric measurement."""

    timestamp: datetime
    value: MetricValue
    attributes: Attributes


class Metric:
    """
    Base metric class.

    Supports:
    - Labels (dimensions)
    - Aggregation
    - Export
    """

    def __init__(
        self,
        name: str,
        description: str,
        unit: str,
        metric_type: MetricType,
    ):
        self.name = name
        self.description = description
        self.unit = unit
        self.metric_type = metric_type
        self._points: Deque[MetricPoint] = deque(maxlen=10000)
        self._lock = threading.Lock()

    def _record(
        self,
        value: MetricValue,
        attributes: Optional[Attributes] = None,
    ) -> None:
        with self._lock:
            self._points.append(
                MetricPoint(
                    timestamp=datetime.now(timezone.utc),
                    value=value,
                    attributes=dict(attributes or {}),
                )
            )

    def get_points(
        self,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[MetricPoint]:
        """Get recorded points."""
        points = list(self._points)
        if since:
            points = [p for p in points if p.timestamp >= since]
        return points[-limit:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "type": self.metric_type.name,
            "point_count": len(self._points),
        }


class Counter(Metric):
    """
    Monotonically increasing counter.

    Use for: request counts, error counts, bytes processed
    """

    def __init__(self, name: str, description: str = "", unit: str = "1"):
        super().__init__(name, description, unit, MetricType.COUNTER)
        self._value = 0.0

    def add(
        self,
        value: MetricValue = 1,
        attributes: Optional[Attributes] = None,
    ) -> None:
        """Add to counter (must be positive)."""
        if value < 0:
            raise ValueError("Counter can only increase")

        with self._lock:
            self._value += value

        self._record(self._value, attributes)

    @property
    def value(self) -> float:
        return self._value


class Gauge(Metric):
    """
    Point-in-time value.

    Use for: temperature, memory usage, queue length
    """

    def __init__(self, name: str, description: str = "", unit: str = "1"):
        super().__init__(name, description, unit, MetricType.GAUGE)
        self._value = 0.0

    def set(
        self,
        value: MetricValue,
        attributes: Optional[Attributes] = None,
    ) -> None:
        """Set gauge value."""
        with self._lock:
            self._value = float(value)

        self._record(self._value, attributes)

    @property
    def value(self) -> float:
        return self._value


class Histogram(Metric):
    """
    Distribution of values with configurable buckets.

    Use for: latencies, sizes, scores
    """

    DEFAULT_BUCKETS = [
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        float("inf"),
    ]

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "1",
        buckets: Optional[List[float]] = None,
    ):
        super().__init__(name, description, unit, MetricType.HISTOGRAM)
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._bucket_counts = [0] * len(self._buckets)
        self._sum = 0.0
        self._count = 0
        self._min = float("inf")
        self._max = float("-inf")

    def record(
        self,
        value: MetricValue,
        attributes: Optional[Attributes] = None,
    ) -> None:
        """Record a value."""
        with self._lock:
            self._sum += value
            self._count += 1
            self._min = min(self._min, value)
            self._max = max(self._max, value)

            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    self._bucket_counts[i] += 1
                    break

        self._record(value, attributes)

    def get_statistics(self) -> Dict[str, float]:
        """Get histogram statistics."""
        return {
            "count": self._count,
            "sum": self._sum,
            "min": self._min if self._count > 0 else 0,
            "max": self._max if self._count > 0 else 0,
            "mean": self._sum / self._count if self._count > 0 else 0,
        }

    def get_percentile(self, p: float) -> float:
        """Estimate percentile from buckets."""
        if self._count == 0:
            return 0.0

        target = self._count * (p / 100.0)
        cumulative = 0

        for i, count in enumerate(self._bucket_counts):
            cumulative += count
            if cumulative >= target:
                return self._buckets[i]

        return self._buckets[-1]


class MeterProvider:
    """
    Central registry for metrics.

    Provides access to counters, gauges, and histograms.
    """

    def __init__(self, service_name: str = "bizra"):
        self.service_name = service_name
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def get_counter(
        self,
        name: str,
        description: str = "",
        unit: str = "1",
    ) -> Counter:
        """Get or create a counter."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description, unit)
            return self._counters[name]

    def get_gauge(
        self,
        name: str,
        description: str = "",
        unit: str = "1",
    ) -> Gauge:
        """Get or create a gauge."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description, unit)
            return self._gauges[name]

    def get_histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "1",
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """Get or create a histogram."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, unit, buckets)
            return self._histograms[name]

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metric data."""
        return {
            "service_name": self.service_name,
            "counters": {
                name: {"value": c.value, **c.to_dict()}
                for name, c in self._counters.items()
            },
            "gauges": {
                name: {"value": g.value, **g.to_dict()}
                for name, g in self._gauges.items()
            },
            "histograms": {
                name: {**h.to_dict(), "statistics": h.get_statistics()}
                for name, h in self._histograms.items()
            },
        }

    def record_snr_metrics(
        self, snr_score: float, snr_level: str, component: str = "overall"
    ) -> None:
        """
        Record SNR (Signal-to-Noise Ratio) metrics for insight quality tracking.

        Args:
            snr_score: SNR value (0.0-1.0+)
            snr_level: Classification (HIGH/MEDIUM/LOW)
            component: Component name (e.g., 'convergence', 'value_oracle')
        """
        # Record SNR score as gauge
        gauge_name = f"bizra_snr_score_{component}"
        gauge = self.get_gauge(
            gauge_name, description=f"SNR quality score for {component}", unit="ratio"
        )
        gauge.set(snr_score)

        # Record SNR level distribution as counter
        counter_name = f"bizra_snr_level_{snr_level.lower()}_{component}"
        counter = self.get_counter(
            counter_name,
            description=f"Count of {snr_level} SNR insights in {component}",
        )
        counter.add(1)

        # Record in histogram for distribution analysis
        hist_name = f"bizra_snr_distribution_{component}"
        histogram = self.get_histogram(
            hist_name,
            description=f"Distribution of SNR scores in {component}",
            unit="ratio",
            buckets=[0.0, 0.3, 0.5, 0.8, 1.0, 2.0],
        )
        histogram.record(snr_score)

    def record_thought_graph_metrics(
        self, chain_depth: int, domain_diversity: float, bridge_count: int
    ) -> None:
        """
        Record graph-of-thoughts reasoning metrics.

        Args:
            chain_depth: Thought chain length (hops)
            domain_diversity: Entropy of domain distribution
            bridge_count: Number of cross-domain bridges discovered
        """
        # Thought chain depth histogram
        hist = self.get_histogram(
            "bizra_thought_chain_depth",
            description="Depth of reasoning chains (hops)",
            unit="hops",
            buckets=[1, 2, 3, 5, 10, 20],
        )
        hist.record(chain_depth)

        # Domain diversity gauge
        gauge = self.get_gauge(
            "bizra_domain_diversity",
            description="Interdisciplinary diversity score",
            unit="entropy",
        )
        gauge.set(domain_diversity)

        # Bridge discovery counter - add all at once for efficiency
        counter = self.get_counter(
            "bizra_domain_bridges_discovered",
            description="Total cross-domain bridges discovered",
        )
        counter.add(bridge_count)

        # Average bridges per chain
        if chain_depth > 0:
            avg_gauge = self.get_gauge(
                "bizra_avg_bridges_per_hop",
                description="Average domain bridges per reasoning hop",
                unit="ratio",
            )
            avg_gauge.set(bridge_count / chain_depth)


# ============================================================================
# STRUCTURED LOGGING
# ============================================================================


@dataclass
class LogRecord:
    """Structured log record with trace correlation."""

    timestamp: datetime
    level: LogLevel
    message: str
    attributes: Attributes
    trace_id: Optional[str]
    span_id: Optional[str]
    service_name: str
    ihsan_score: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.name,
            "message": self.message,
            "attributes": self.attributes,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "service_name": self.service_name,
            "ihsan_score": self.ihsan_score,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class StructuredLogger:
    """
    Structured logger with trace correlation.

    Integrates with distributed tracing for
    log-to-trace correlation.
    """

    def __init__(
        self,
        service_name: str = "bizra",
        tracer: Optional[Tracer] = None,
        min_level: LogLevel = LogLevel.INFO,
    ):
        self.service_name = service_name
        self._tracer = tracer
        self._min_level = min_level
        self._records: Deque[LogRecord] = deque(maxlen=10000)
        self._handlers: List[Callable[[LogRecord], None]] = []

    def _log(
        self,
        level: LogLevel,
        message: str,
        ihsan_score: Optional[float] = None,
        **attributes: Any,
    ) -> None:
        """Internal log method."""
        if level.value < self._min_level.value:
            return

        # Get trace context
        trace_id = None
        span_id = None

        if self._tracer:
            ctx = self._tracer.get_current_context()
            if ctx:
                trace_id = ctx.trace_id
                span_id = ctx.span_id

        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=level,
            message=message,
            attributes=dict(attributes),
            trace_id=trace_id,
            span_id=span_id,
            service_name=self.service_name,
            ihsan_score=ihsan_score,
        )

        self._records.append(record)

        for handler in self._handlers:
            try:
                handler(record)
            except Exception as e:
                print(f"Log handler error: {e}", file=sys.stderr)

    def trace(
        self,
        message: str,
        ihsan_score: Optional[float] = None,
        **attributes: Any,
    ) -> None:
        self._log(LogLevel.TRACE, message, ihsan_score, **attributes)

    def debug(
        self,
        message: str,
        ihsan_score: Optional[float] = None,
        **attributes: Any,
    ) -> None:
        self._log(LogLevel.DEBUG, message, ihsan_score, **attributes)

    def info(
        self,
        message: str,
        ihsan_score: Optional[float] = None,
        **attributes: Any,
    ) -> None:
        self._log(LogLevel.INFO, message, ihsan_score, **attributes)

    def warn(
        self,
        message: str,
        ihsan_score: Optional[float] = None,
        **attributes: Any,
    ) -> None:
        self._log(LogLevel.WARN, message, ihsan_score, **attributes)

    def error(
        self,
        message: str,
        ihsan_score: Optional[float] = None,
        **attributes: Any,
    ) -> None:
        self._log(LogLevel.ERROR, message, ihsan_score, **attributes)

    def fatal(
        self,
        message: str,
        ihsan_score: Optional[float] = None,
        **attributes: Any,
    ) -> None:
        self._log(LogLevel.FATAL, message, ihsan_score, **attributes)

    def add_handler(self, handler: Callable[[LogRecord], None]) -> None:
        """Add a log handler."""
        self._handlers.append(handler)

    def get_records(
        self,
        level: Optional[LogLevel] = None,
        limit: int = 100,
    ) -> List[LogRecord]:
        """Get log records, optionally filtered by level."""
        records = list(self._records)
        if level:
            records = [r for r in records if r.level.value >= level.value]
        return records[-limit:]


# ============================================================================
# IHSAN COMPLIANCE MONITOR
# ============================================================================


class IhsanComplianceMonitor:
    """
    Specialized monitor for Ihsan compliance tracking.

    Provides:
    - Real-time compliance metrics
    - Threshold alerts
    - Trend analysis
    - Circuit breaker integration
    """

    def __init__(
        self,
        meter: MeterProvider,
        alert_threshold: float = 0.95,
        critical_threshold: float = 0.90,
    ):
        self._meter = meter
        self._alert_threshold = alert_threshold
        self._critical_threshold = critical_threshold

        # Metrics
        self._score_histogram = meter.get_histogram(
            "ihsan.score",
            "Distribution of Ihsan scores",
            unit="score",
            buckets=[0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0],
        )

        self._compliance_counter = meter.get_counter(
            "ihsan.compliance.total",
            "Total compliance checks",
        )

        self._violation_counter = meter.get_counter(
            "ihsan.violations.total",
            "Total Ihsan violations",
        )

        self._current_score = meter.get_gauge(
            "ihsan.current_score",
            "Current Ihsan score",
        )

        # Tracking
        self._scores: Deque[Tuple[datetime, float, str]] = deque(maxlen=10000)
        self._violations: Deque[Tuple[datetime, float, str]] = deque(
            maxlen=1000
        )  # Bounded to prevent memory growth

    def record_score(
        self,
        score: float,
        layer: str,
        operation: Optional[str] = None,
    ) -> bool:
        """
        Record an Ihsan score.

        Returns True if compliant, False if violation.
        """
        now = datetime.now(timezone.utc)
        self._scores.append((now, score, layer))
        self._score_histogram.record(score, {"layer": layer})
        self._current_score.set(score)
        self._compliance_counter.add(1, {"layer": layer})

        if score < self._alert_threshold:
            self._violation_counter.add(1, {"layer": layer, "severity": "warning"})

            if score < self._critical_threshold:
                self._violations.append((now, score, f"{layer}:{operation}"))
                return False

        return True

    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        scores = [s for _, s, _ in self._scores]

        if not scores:
            return {
                "compliant": True,
                "samples": 0,
                "mean_score": 0.0,
                "violations": 0,
            }

        return {
            "compliant": len(self._violations) == 0,
            "samples": len(scores),
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "violations": len(self._violations),
            "violation_rate": len(self._violations) / len(scores),
            "recent_violations": [
                {"time": t.isoformat(), "score": s, "location": l}
                for t, s, l in self._violations[-10:]
            ],
        }

    def get_layer_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get compliance breakdown by layer."""
        by_layer: Dict[str, List[float]] = defaultdict(list)

        for _, score, layer in self._scores:
            by_layer[layer].append(score)

        return {
            layer: {
                "count": len(scores),
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "compliance_rate": sum(1 for s in scores if s >= self._alert_threshold)
                / len(scores),
            }
            for layer, scores in by_layer.items()
        }


# ============================================================================
# TELEMETRY PROVIDER (UNIFIED ACCESS)
# ============================================================================


class TelemetryProvider:
    """
    Unified telemetry provider for BIZRA.

    Provides single access point to:
    - Tracer (distributed tracing)
    - Meter (metrics)
    - Logger (structured logging)
    - Ihsan Monitor (compliance)
    """

    _instance: Optional[TelemetryProvider] = None

    def __init__(
        self,
        service_name: str = "bizra",
        enable_console: bool = False,
    ):
        self.service_name = service_name

        # Create exporter
        exporters: List[SpanExporter] = [InMemorySpanExporter()]
        if enable_console:
            exporters.append(ConsoleSpanExporter())

        exporter = (
            CompositeSpanExporter(exporters) if len(exporters) > 1 else exporters[0]
        )

        # Initialize components
        self.tracer = Tracer(service_name, exporter)
        self.meter = MeterProvider(service_name)
        self.logger = StructuredLogger(service_name, self.tracer)
        self.ihsan_monitor = IhsanComplianceMonitor(self.meter)

        # Standard metrics
        self._operation_counter = self.meter.get_counter(
            "operations.total",
            "Total operations executed",
        )

        self._operation_latency = self.meter.get_histogram(
            "operations.latency_ms",
            "Operation latency",
            unit="ms",
        )

        self._error_counter = self.meter.get_counter(
            "errors.total",
            "Total errors",
        )

    @classmethod
    def get_instance(cls) -> TelemetryProvider:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = TelemetryProvider()
        return cls._instance

    @classmethod
    def initialize(
        cls,
        service_name: str = "bizra",
        enable_console: bool = False,
    ) -> TelemetryProvider:
        """Initialize the singleton."""
        cls._instance = TelemetryProvider(service_name, enable_console)
        return cls._instance

    def record_operation(
        self,
        name: str,
        layer: str,
        duration_ms: float,
        ihsan_score: float,
        success: bool = True,
    ) -> None:
        """Record a completed operation."""
        self._operation_counter.add(
            1,
            {
                "operation": name,
                "layer": layer,
                "success": str(success),
            },
        )

        self._operation_latency.record(
            duration_ms,
            {
                "operation": name,
                "layer": layer,
            },
        )

        self.ihsan_monitor.record_score(ihsan_score, layer, name)

        if not success:
            self._error_counter.add(1, {"operation": name, "layer": layer})

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            "service_name": self.service_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tracing": self.tracer.get_metrics(),
            "metrics": self.meter.get_all_metrics(),
            "ihsan_compliance": self.ihsan_monitor.get_compliance_report(),
            "ihsan_by_layer": self.ihsan_monitor.get_layer_breakdown(),
        }


# ============================================================================
# DEMO
# ============================================================================


async def demo_telemetry():
    """Demonstrate telemetry capabilities."""
    print("=" * 70)
    print("BIZRA OBSERVABILITY TELEMETRY DEMO")
    print("=" * 70)

    # Initialize telemetry
    telemetry = TelemetryProvider.initialize("bizra-demo", enable_console=True)

    # 1. Distributed Tracing
    print("\n1. Distributed Tracing")
    print("-" * 40)

    async with telemetry.tracer.async_span(
        "parent_operation",
        ihsan_score=0.99,
        apex_layer="COGNITIVE",
    ) as parent:
        parent.set_attribute("custom.key", "value")

        async with telemetry.tracer.async_span(
            "child_operation",
            ihsan_score=0.98,
            apex_layer="EXECUTION",
        ) as child:
            child.add_event("processing_started")
            await asyncio.sleep(0.01)
            child.add_event("processing_completed")

    # 2. Metrics
    print("\n2. Metrics Collection")
    print("-" * 40)

    counter = telemetry.meter.get_counter("demo.requests", "Demo requests")
    histogram = telemetry.meter.get_histogram("demo.latency", "Demo latency", "ms")

    for i in range(100):
        counter.add(1, {"endpoint": f"/api/{i % 5}"})
        histogram.record(i * 0.5, {"endpoint": f"/api/{i % 5}"})

    stats = histogram.get_statistics()
    print(f"  Latency stats: mean={stats['mean']:.2f}ms, max={stats['max']:.2f}ms")

    # 3. Structured Logging
    print("\n3. Structured Logging")
    print("-" * 40)

    telemetry.logger.info(
        "Operation completed",
        ihsan_score=0.97,
        operation="demo",
        duration_ms=42.5,
    )

    telemetry.logger.warn(
        "Low Ihsan score detected",
        ihsan_score=0.94,
        layer="GOVERNANCE",
    )

    records = telemetry.logger.get_records(limit=5)
    print(f"  Recent logs: {len(records)}")

    # 4. Ihsan Compliance
    print("\n4. Ihsan Compliance Monitoring")
    print("-" * 40)

    for layer in ["BLOCKCHAIN", "EXECUTION", "GOVERNANCE"]:
        for _ in range(10):
            import random

            score = random.uniform(0.93, 1.0)
            telemetry.ihsan_monitor.record_score(score, layer)

    report = telemetry.ihsan_monitor.get_compliance_report()
    print(f"  Mean Ihsan: {report['mean_score']:.4f}")
    print(f"  Violations: {report['violations']}")
    print(f"  Compliant: {report['compliant']}")

    # 5. Dashboard Data
    print("\n5. Dashboard Data")
    print("-" * 40)

    dashboard = telemetry.get_dashboard_data()
    print(f"  Spans created: {dashboard['tracing']['spans_created']}")
    print(f"  Counter count: {len(dashboard['metrics']['counters'])}")
    print(f"  Ihsan compliance: {dashboard['ihsan_compliance']['compliant']}")

    print("\n" + "=" * 70)
    print("TELEMETRY DEMO COMPLETE")
    print("=" * 70)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Trace Context
    "TraceContext",
    # Spans
    "Span",
    "SpanEvent",
    "SpanLink",
    "SpanKind",
    "SpanStatus",
    # Tracer
    "Tracer",
    # Exporters
    "SpanExporter",
    "InMemorySpanExporter",
    "ConsoleSpanExporter",
    "CompositeSpanExporter",
    # Metrics
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricType",
    "MetricPoint",
    "MeterProvider",
    # Logging
    "LogRecord",
    "LogLevel",
    "StructuredLogger",
    # Ihsan
    "IhsanComplianceMonitor",
    # Provider
    "TelemetryProvider",
]


if __name__ == "__main__":
    asyncio.run(demo_telemetry())
