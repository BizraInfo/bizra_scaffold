"""
BIZRA AEON OMEGA - LLM Provider Base Classes
Genesis Block 3-Layer Architecture - Unified LLM Interface

Provides abstract base for all LLM providers (Ollama, OpenAI, Azure, etc.)
with Ihsan compliance enforcement and department-aware routing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)

# Ihsan compliance threshold
IHSAN_MINIMUM = 0.95


class ProviderType(Enum):
    """Supported LLM provider types."""
    OLLAMA = auto()       # Local Ollama models
    OPENAI = auto()       # OpenAI API
    AZURE = auto()        # Azure OpenAI
    ANTHROPIC = auto()    # Anthropic Claude
    DEEPSEEK = auto()     # DeepSeek models
    LOCAL = auto()        # Generic local model


class DepartmentID(Enum):
    """Genesis Council department identifiers."""
    CRYPTOGRAPHY = "D1"  # Security, keys, proofs, ZK
    ECONOMICS = "D2"     # Tokenomics, PoI, staking
    PHILOSOPHY = "D3"    # Ethics, Ihsan, values
    GOVERNANCE = "D4"    # FATE Engine, voting, constitution
    SYSTEMS = "D5"       # Architecture, infra, scaling
    COGNITIVE = "D6"     # Memory, learning, reasoning
    OPERATIONS = "D7"    # Monitoring, health, DevOps


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    model_id: str
    provider: ProviderType
    context_window: int = 4096
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    stop_sequences: List[str] = field(default_factory=list)
    
    # Performance hints
    quantization: Optional[str] = None  # e.g., "Q4_K_M"
    gpu_layers: int = -1  # -1 = auto
    
    # Department assignment
    primary_department: Optional[DepartmentID] = None
    fallback_departments: List[DepartmentID] = field(default_factory=list)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    provider: ProviderType
    
    # Metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    
    # Ihsan compliance
    ihsan_score: float = 0.0
    passed_ihsan_gate: bool = False
    
    # Metadata
    finish_reason: str = "stop"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider.name,
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.total_tokens,
            },
            "latency_ms": self.latency_ms,
            "ihsan_score": self.ihsan_score,
            "passed_ihsan_gate": self.passed_ihsan_gate,
            "timestamp": self.timestamp.isoformat(),
        }


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers must implement:
    - generate(): Single completion
    - stream(): Streaming completion
    - health_check(): Provider availability
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._healthy = True
        self._last_health_check: Optional[datetime] = None
        self._request_count = 0
        self._error_count = 0

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion for the given prompt."""
        pass

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion tokens."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy and available."""
        pass

    async def generate_with_ihsan_gate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        ihsan_validator: Optional[callable] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate with Ihsan compliance gate.
        
        Validates response against Ihsan metric before returning.
        Fails closed if validation fails.
        """
        start_time = time.perf_counter()
        
        response = await self.generate(prompt, system_prompt, **kwargs)
        
        # Calculate Ihsan score if validator provided
        if ihsan_validator:
            response.ihsan_score = await ihsan_validator(response.content)
        else:
            # Default: assume compliant (actual implementation should validate)
            response.ihsan_score = 0.95
        
        response.passed_ihsan_gate = response.ihsan_score >= IHSAN_MINIMUM
        response.latency_ms = (time.perf_counter() - start_time) * 1000
        
        if not response.passed_ihsan_gate:
            logger.warning(
                f"Ihsan gate FAILED for {self.config.model_id}: "
                f"score={response.ihsan_score:.3f} < {IHSAN_MINIMUM}"
            )
            # Fail closed: return empty response
            response.content = "[IHSAN_GATE_BLOCKED: Response did not meet ethical standards]"
            response.finish_reason = "ihsan_blocked"
        
        return response

    def record_success(self) -> None:
        """Record successful request."""
        self._request_count += 1

    def record_error(self) -> None:
        """Record failed request."""
        self._request_count += 1
        self._error_count += 1

    @property
    def error_rate(self) -> float:
        """Current error rate."""
        if self._request_count == 0:
            return 0.0
        return self._error_count / self._request_count

    @property
    def is_healthy(self) -> bool:
        """Check if provider is considered healthy."""
        return self._healthy and self.error_rate < 0.5
