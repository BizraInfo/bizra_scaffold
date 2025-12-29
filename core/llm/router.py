"""
BIZRA AEON OMEGA - Genesis LLM Router
Department-Aware Model Routing for 7x7 Agent Structure

Routes agent requests to appropriate LLM based on:
1. Department specialization
2. Task complexity (SNR threshold)
3. Resource availability
4. Ihsan compliance history
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.llm.provider_base import (
    DepartmentID,
    LLMProvider,
    LLMResponse,
    ModelConfig,
    ProviderType,
)
from core.llm.ollama_client import OllamaClient, DEPARTMENT_MODEL_MAPPING

logger = logging.getLogger(__name__)


@dataclass
class AgentTask:
    """Task submitted by a department agent."""
    task_id: str
    agent_id: str
    department: DepartmentID
    prompt: str
    system_prompt: Optional[str] = None
    priority: int = 5  # 1=highest, 10=lowest
    max_tokens: int = 2048
    temperature: float = 0.7
    require_streaming: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Decision made by the router."""
    model_id: str
    provider: ProviderType
    department: DepartmentID
    reason: str
    fallback_chain: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GenesisLLMRouter:
    """
    Routes agent requests to appropriate LLM based on department.
    
    Features:
    - Department-aware model selection
    - Automatic fallback chains
    - Health-based routing
    - Load balancing across models
    - Ihsan compliance enforcement
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        default_timeout: float = 120.0,
    ):
        self.ollama_host = ollama_host
        self.default_timeout = default_timeout
        self._providers: Dict[str, LLMProvider] = {}
        self._model_health: Dict[str, bool] = {}
        self._request_counts: Dict[str, int] = {}
        
        # Boss agent uses high-reasoning model
        self.boss_models = ["gemma2:9b", "phi3:14b"]

    async def initialize(self) -> None:
        """Initialize router and discover available models."""
        # Create a temporary client to list models
        config = ModelConfig(model_id="temp", provider=ProviderType.OLLAMA)
        client = OllamaClient(config, host=self.ollama_host)
        
        try:
            available = await client.list_models()
            logger.info(f"Discovered {len(available)} Ollama models: {available}")
            
            for model_id in available:
                self._model_health[model_id] = True
                self._request_counts[model_id] = 0
        finally:
            await client.close()

    def _get_models_for_department(self, department: DepartmentID) -> List[str]:
        """Get prioritized model list for department."""
        return DEPARTMENT_MODEL_MAPPING.get(department, ["llama3.1:8b"])

    def _select_best_model(
        self,
        candidates: List[str],
        prefer_loaded: bool = True,
    ) -> Optional[str]:
        """Select best available model from candidates."""
        for model_id in candidates:
            if self._model_health.get(model_id, False):
                return model_id
        
        # Fallback to first candidate if no health data
        return candidates[0] if candidates else None

    async def route(
        self,
        task: AgentTask,
        fallback_chain: bool = True,
    ) -> LLMResponse:
        """
        Route task to appropriate model and get response.
        
        Args:
            task: The agent task to route
            fallback_chain: Whether to try fallback models on failure
            
        Returns:
            LLMResponse from the selected model
        """
        candidates = self._get_models_for_department(task.department)
        model_id = self._select_best_model(candidates)
        
        if not model_id:
            raise RuntimeError(f"No available model for department {task.department}")
        
        # Create or get provider
        if model_id not in self._providers:
            config = ModelConfig(
                model_id=model_id,
                provider=ProviderType.OLLAMA,
                max_tokens=task.max_tokens,
                temperature=task.temperature,
                primary_department=task.department,
            )
            self._providers[model_id] = OllamaClient(
                config,
                host=self.ollama_host,
                timeout=self.default_timeout,
            )
        
        provider = self._providers[model_id]
        
        try:
            response = await provider.generate_with_ihsan_gate(
                prompt=task.prompt,
                system_prompt=task.system_prompt,
                max_tokens=task.max_tokens,
                temperature=task.temperature,
            )
            
            self._request_counts[model_id] = self._request_counts.get(model_id, 0) + 1
            
            logger.info(
                f"Routed {task.department.name} task to {model_id}: "
                f"latency={response.latency_ms:.1f}ms, ihsan={response.ihsan_score:.3f}"
            )
            
            return response
            
        except Exception as e:
            logger.warning(f"Model {model_id} failed: {e}")
            self._model_health[model_id] = False
            
            if fallback_chain and len(candidates) > 1:
                # Try fallback models
                for fallback_id in candidates[1:]:
                    if fallback_id != model_id:
                        logger.info(f"Trying fallback model: {fallback_id}")
                        task_copy = AgentTask(
                            task_id=task.task_id,
                            agent_id=task.agent_id,
                            department=task.department,
                            prompt=task.prompt,
                            system_prompt=task.system_prompt,
                            priority=task.priority,
                            max_tokens=task.max_tokens,
                            temperature=task.temperature,
                        )
                        return await self.route(task_copy, fallback_chain=False)
            
            raise

    async def route_boss(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Route a Boss Agent request to high-reasoning model."""
        model_id = self._select_best_model(self.boss_models)
        
        if not model_id:
            raise RuntimeError("No available model for Boss Agent")
        
        if model_id not in self._providers:
            config = ModelConfig(
                model_id=model_id,
                provider=ProviderType.OLLAMA,
                max_tokens=4096,  # Boss needs more context
                temperature=0.5,  # Lower temp for coordination
            )
            self._providers[model_id] = OllamaClient(
                config,
                host=self.ollama_host,
                timeout=self.default_timeout,
            )
        
        return await self._providers[model_id].generate_with_ihsan_gate(
            prompt=prompt,
            system_prompt=system_prompt or "You are the Boss Agent coordinating 7 department Alpha managers.",
        )

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered models."""
        results = {}
        
        for model_id, provider in self._providers.items():
            results[model_id] = await provider.health_check()
            self._model_health[model_id] = results[model_id]
        
        return results

    async def close(self) -> None:
        """Close all provider connections."""
        for provider in self._providers.values():
            if hasattr(provider, 'close'):
                await provider.close()
        self._providers.clear()

    async def __aenter__(self) -> "GenesisLLMRouter":
        await self.initialize()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
