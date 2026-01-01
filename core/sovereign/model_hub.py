"""
Sovereign Model Hub
═══════════════════════════════════════════════════════════════════════════════
Unified orchestration of all AI models under sovereign control.

This hub provides the foundation for Phase Beta by unifying:
- 12+ local models (Ollama, LLM Studio, CUDA)
- Intelligent routing based on SNR and task requirements
- Sovereign authentication and resource allocation
- Real-time performance monitoring and optimization
"""

from __future__ import annotations

import asyncio
import logging
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers."""
    OLLAMA = "ollama"
    LLM_STUDIO = "llm_studio"
    CUDA = "cuda"
    API = "api"
    LOCAL = "local"


class ModelCapability(Enum):
    """Model capabilities for intelligent routing."""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    GPU_ACCELERATED = "gpu_accelerated"
    PARALLEL_PROCESSING = "parallel_processing"


@dataclass
class ModelEndpoint:
    """Represents a single model endpoint."""
    provider: ModelProvider
    model_name: str
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    local_path: Optional[str] = None
    capabilities: List[ModelCapability] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def is_available(self) -> bool:
        """Check if model endpoint is available."""
        # Implementation would ping the endpoint
        return True  # Placeholder

    def get_sovereign_hash(self) -> str:
        """Get sovereign verification hash for this model."""
        data = f"{self.provider.value}:{self.model_name}:{self.endpoint_url or self.local_path}"
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ModelRequest:
    """A request to execute on a model."""
    task: str
    context: Dict[str, Any]
    requirements: Dict[str, Any] = field(default_factory=dict)
    sovereignty_verified: bool = False

    def get_complexity_score(self) -> float:
        """Estimate task complexity for model selection."""
        # Simple complexity estimation based on task length and context
        complexity = len(self.task) / 1000.0  # Length factor
        complexity += len(str(self.context)) / 5000.0  # Context factor
        complexity += self.requirements.get("complexity_boost", 0.0)
        return min(1.0, complexity)


@dataclass
class ModelResponse:
    """Response from model execution."""
    success: bool
    content: Any
    model_used: str
    execution_time_ms: float
    sovereignty_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SNRRoutingEngine:
    """
    Intelligent model routing based on Signal-to-Noise Ratio analysis.
    """

    def __init__(self):
        self.routing_history: List[Dict[str, Any]] = []
        self.performance_stats: Dict[str, Dict[str, float]] = {}

    def select_optimal_model(self, request: ModelRequest, available_models: List[ModelEndpoint]) -> Optional[ModelEndpoint]:
        """
        Select the optimal model for a request based on SNR analysis.
        """
        if not available_models:
            return None

        # Calculate scores for each model
        model_scores = []
        for model in available_models:
            score = self._calculate_model_score(model, request)
            model_scores.append((model, score))

        # Select highest scoring model
        best_model, best_score = max(model_scores, key=lambda x: x[1])

        # Record routing decision
        self.routing_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_complexity": request.get_complexity_score(),
            "selected_model": best_model.model_name,
            "score": best_score,
            "available_models": len(available_models)
        })

        logger.info(f"Selected model {best_model.model_name} with score {best_score:.3f}")
        return best_model

    def _calculate_model_score(self, model: ModelEndpoint, request: ModelRequest) -> float:
        """Calculate suitability score for model on request."""
        score = 0.0

        # Capability matching (0.4 weight)
        required_caps = self._extract_required_capabilities(request)
        capability_score = self._calculate_capability_match(model, required_caps)
        score += 0.4 * capability_score

        # Performance history (0.3 weight)
        performance_score = self._get_performance_score(model)
        score += 0.3 * performance_score

        # Complexity suitability (0.2 weight)
        complexity_score = self._calculate_complexity_suitability(model, request.get_complexity_score())
        score += 0.2 * complexity_score

        # Current load (0.1 weight)
        load_score = self._calculate_load_score(model)
        score += 0.1 * load_score

        return score

    def _extract_required_capabilities(self, request: ModelRequest) -> List[ModelCapability]:
        """Extract required capabilities from request."""
        caps = []
        task_lower = request.task.lower()

        if any(word in task_lower for word in ["generate", "write", "create"]):
            caps.append(ModelCapability.TEXT_GENERATION)
        if any(word in task_lower for word in ["chat", "converse", "talk"]):
            caps.append(ModelCapability.CHAT)
        if any(word in task_lower for word in ["code", "programming", "function"]):
            caps.append(ModelCapability.CODE_GENERATION)
        if any(word in task_lower for word in ["analyze", "understand", "reason"]):
            caps.append(ModelCapability.ANALYSIS)

        # GPU acceleration for complex tasks
        if request.get_complexity_score() > 0.7:
            caps.append(ModelCapability.GPU_ACCELERATED)

        return caps

    def _calculate_capability_match(self, model: ModelEndpoint, required_caps: List[ModelCapability]) -> float:
        """Calculate how well model capabilities match requirements."""
        if not required_caps:
            return 1.0

        matches = sum(1 for cap in required_caps if cap in model.capabilities)
        return matches / len(required_caps)

    def _get_performance_score(self, model: ModelEndpoint) -> float:
        """Get historical performance score for model."""
        model_key = f"{model.provider.value}:{model.model_name}"
        stats = self.performance_stats.get(model_key, {})

        # Combine latency, quality, and reliability scores
        latency_score = 1.0 - min(1.0, stats.get("avg_latency_ms", 1000) / 5000)  # Better with lower latency
        quality_score = stats.get("avg_quality", 0.8)
        reliability_score = stats.get("success_rate", 0.95)

        return (latency_score + quality_score + reliability_score) / 3.0

    def _calculate_complexity_suitability(self, model: ModelEndpoint, complexity: float) -> float:
        """Calculate how suitable model is for task complexity."""
        # Larger models typically better for complex tasks
        model_size = self._estimate_model_size(model)
        size_factor = min(1.0, model_size / 10e9)  # Assume 10B params is "large"

        # Complex tasks need capable models
        if complexity > 0.7:
            return size_factor
        elif complexity > 0.3:
            return 0.8  # Medium suitability
        else:
            return 1.0 - (size_factor * 0.3)  # Simple tasks can use smaller models

    def _calculate_load_score(self, model: ModelEndpoint) -> float:
        """Calculate load-based availability score."""
        # This would check current usage/load
        # For now, assume all models are available
        return 1.0

    def _estimate_model_size(self, model: ModelEndpoint) -> float:
        """Estimate model parameter count."""
        # Simple estimation based on model name
        name_lower = model.model_name.lower()

        if "70b" in name_lower or "70-b" in name_lower:
            return 70e9
        elif "30b" in name_lower or "30-b" in name_lower:
            return 30e9
        elif "13b" in name_lower or "13-b" in name_lower:
            return 13e9
        elif "7b" in name_lower or "7-b" in name_lower:
            return 7e9
        elif "3b" in name_lower or "3-b" in name_lower:
            return 3e9
        else:
            return 1e9  # Default small model

    def record_performance(self, model: ModelEndpoint, execution_time_ms: float, success: bool, quality_score: float):
        """Record model performance for future routing decisions."""
        model_key = f"{model.provider.value}:{model.model_name}"

        if model_key not in self.performance_stats:
            self.performance_stats[model_key] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_latency_ms": 0,
                "total_quality": 0,
                "avg_latency_ms": 0,
                "avg_quality": 0,
                "success_rate": 0
            }

        stats = self.performance_stats[model_key]
        stats["total_executions"] += 1
        stats["total_latency_ms"] += execution_time_ms
        stats["total_quality"] += quality_score

        if success:
            stats["successful_executions"] += 1

        # Update averages
        stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_executions"]
        stats["avg_quality"] = stats["total_quality"] / stats["total_executions"]
        stats["success_rate"] = stats["successful_executions"] / stats["total_executions"]


class SovereignModelHub:
    """
    Sovereign Model Hub - Unified control over all AI models.

    This is the foundation of Phase Beta, providing:
    - Discovery and registration of all available models
    - Intelligent routing based on SNR analysis
    - Sovereign authentication and resource allocation
    - Real-time performance monitoring
    """

    def __init__(self):
        self.models: Dict[str, ModelEndpoint] = {}
        self.routing_engine = SNRRoutingEngine()
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.operation_counter = 0

        # Auto-discover models on initialization
        self._discover_models()

    def _discover_models(self) -> None:
        """Discover all available models across providers."""
        logger.info("Discovering AI models across all providers...")

        # Ollama models
        self._discover_ollama_models()

        # LLM Studio models
        self._discover_llm_studio_models()

        # CUDA models
        self._discover_cuda_models()

        # API-based models
        self._discover_api_models()

        logger.info(f"Discovered {len(self.models)} models across {len(set(m.provider for m in self.models.values()))} providers")

    def _discover_ollama_models(self) -> None:
        """Discover models available through Ollama."""
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 1:
                            model_name = parts[0]
                            endpoint = ModelEndpoint(
                                provider=ModelProvider.OLLAMA,
                                model_name=model_name,
                                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT]
                            )
                            self.models[f"ollama:{model_name}"] = endpoint
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Ollama not available for model discovery")

    def _discover_llm_studio_models(self) -> None:
        """Discover models available through LLM Studio."""
        try:
            import requests
            response = requests.get("http://localhost:1234/v1/models", timeout=5)

            if response.status_code == 200:
                models_data = response.json()
                for model_info in models_data.get("data", []):
                    model_name = model_info.get("id", "unknown")
                    endpoint = ModelEndpoint(
                        provider=ModelProvider.LLM_STUDIO,
                        model_name=model_name,
                        endpoint_url="http://localhost:1234",
                        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT, ModelCapability.COMPLETION]
                    )
                    self.models[f"llm_studio:{model_name}"] = endpoint
        except Exception:
            logger.warning("LLM Studio not available for model discovery")

    def _discover_cuda_models(self) -> None:
        """Discover CUDA-accelerated models."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    device_name = torch.cuda.get_device_name(i)
                    model_name = f"cuda_{i}_{device_name.replace(' ', '_')}"
                    endpoint = ModelEndpoint(
                        provider=ModelProvider.CUDA,
                        model_name=model_name,
                        capabilities=[ModelCapability.GPU_ACCELERATED, ModelCapability.PARALLEL_PROCESSING]
                    )
                    self.models[f"cuda:{model_name}"] = endpoint
        except ImportError:
            logger.warning("PyTorch not available for CUDA model discovery")

    def _discover_api_models(self) -> None:
        """Discover API-based models (OpenAI, Anthropic, etc.)."""
        # This would check for API keys and configured endpoints
        # For now, placeholder for future API model integration
        pass

    async def execute_sovereign_task(self, request: ModelRequest) -> ModelResponse:
        """
        Execute a task using the optimal available model under sovereign control.
        """
        start_time = time.perf_counter()

        # Select optimal model
        available_models = [m for m in self.models.values() if m.is_available()]
        if not available_models:
            return ModelResponse(
                success=False,
                content="No models available",
                model_used="none",
                execution_time_ms=0,
                sovereignty_hash=""
            )

        optimal_model = self.routing_engine.select_optimal_model(request, available_models)
        if not optimal_model:
            return ModelResponse(
                success=False,
                content="No suitable model found",
                model_used="none",
                execution_time_ms=0,
                sovereignty_hash=""
            )

        # Execute on selected model
        operation_id = f"sovereign_op_{self.operation_counter}"
        self.operation_counter += 1

        self.active_operations[operation_id] = {
            "model": optimal_model.model_name,
            "task": request.task[:100] + "..." if len(request.task) > 100 else request.task,
            "start_time": datetime.now(timezone.utc).isoformat()
        }

        try:
            # Route to appropriate execution method
            result = await self._execute_on_model(optimal_model, request)

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Record performance
            quality_score = 0.8  # Would be calculated based on result quality
            self.routing_engine.record_performance(optimal_model, execution_time_ms, result["success"], quality_score)

            # Clean up
            del self.active_operations[operation_id]

            return ModelResponse(
                success=result["success"],
                content=result["content"],
                model_used=optimal_model.model_name,
                execution_time_ms=execution_time_ms,
                sovereignty_hash=optimal_model.get_sovereign_hash(),
                metadata=result.get("metadata", {})
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Model execution failed: {e}")

            # Clean up
            del self.active_operations[operation_id]

            return ModelResponse(
                success=False,
                content=str(e),
                model_used=optimal_model.model_name,
                execution_time_ms=execution_time_ms,
                sovereignty_hash=optimal_model.get_sovereign_hash()
            )

    async def _execute_on_model(self, model: ModelEndpoint, request: ModelRequest) -> Dict[str, Any]:
        """Execute request on specific model."""
        if model.provider == ModelProvider.OLLAMA:
            return await self._execute_ollama(model, request)
        elif model.provider == ModelProvider.LLM_STUDIO:
            return await self._execute_llm_studio(model, request)
        elif model.provider == ModelProvider.CUDA:
            return await self._execute_cuda(model, request)
        else:
            return {"success": False, "content": f"Unsupported provider: {model.provider}", "metadata": {}}

    async def _execute_ollama(self, model: ModelEndpoint, request: ModelRequest) -> Dict[str, Any]:
        """Execute on Ollama model."""
        try:
            import subprocess

            # Prepare prompt
            prompt = self._prepare_prompt(request)

            # Execute via ollama CLI
            result = subprocess.run(
                ["ollama", "run", model.model_name],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "content": result.stdout.strip(),
                    "metadata": {"provider": "ollama", "exit_code": result.returncode}
                }
            else:
                return {
                    "success": False,
                    "content": result.stderr.strip(),
                    "metadata": {"provider": "ollama", "exit_code": result.returncode}
                }

        except Exception as e:
            return {"success": False, "content": str(e), "metadata": {"provider": "ollama"}}

    async def _execute_llm_studio(self, model: ModelEndpoint, request: ModelRequest) -> Dict[str, Any]:
        """Execute on LLM Studio model."""
        try:
            import requests

            prompt = self._prepare_prompt(request)

            payload = {
                "model": model.model_name,
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7
            }

            response = requests.post(
                f"{model.endpoint_url}/v1/completions",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("text", "")
                return {
                    "success": True,
                    "content": content,
                    "metadata": {"provider": "llm_studio", "status_code": response.status_code}
                }
            else:
                return {
                    "success": False,
                    "content": f"API error: {response.status_code}",
                    "metadata": {"provider": "llm_studio", "status_code": response.status_code}
                }

        except Exception as e:
            return {"success": False, "content": str(e), "metadata": {"provider": "llm_studio"}}

    async def _execute_cuda(self, model: ModelEndpoint, request: ModelRequest) -> Dict[str, Any]:
        """Execute on CUDA-accelerated model."""
        # This would integrate with PyTorch CUDA models
        # For now, return placeholder
        return {
            "success": True,
            "content": f"CUDA execution placeholder for {model.model_name}",
            "metadata": {"provider": "cuda", "status": "placeholder"}
        }

    def _prepare_prompt(self, request: ModelRequest) -> str:
        """Prepare prompt for model execution."""
        context_str = json.dumps(request.context, indent=2) if request.context else ""
        return f"Task: {request.task}\n\nContext:\n{context_str}\n\nPlease provide a helpful response."

    def get_sovereign_manifest(self) -> Dict[str, Any]:
        """Get complete model hub manifest."""
        return {
            "sovereignty_status": "ACTIVE",
            "total_models": len(self.models),
            "providers": list(set(m.provider.value for m in self.models.values())),
            "capabilities": list(set(cap.value for m in self.models.values() for cap in m.capabilities)),
            "active_operations": len(self.active_operations),
            "routing_decisions": len(self.routing_engine.routing_history),
            "models": [
                {
                    "name": model.model_name,
                    "provider": model.provider.value,
                    "capabilities": [cap.value for cap in model.capabilities],
                    "sovereign_hash": model.get_sovereign_hash()
                }
                for model in self.models.values()
            ]
        }

    def register_model(self, endpoint: ModelEndpoint) -> None:
        """Register a new model endpoint."""
        key = f"{endpoint.provider.value}:{endpoint.model_name}"
        self.models[key] = endpoint
        logger.info(f"Registered model: {key}")

    def unregister_model(self, provider: ModelProvider, model_name: str) -> None:
        """Unregister a model endpoint."""
        key = f"{provider.value}:{model_name}"
        if key in self.models:
            del self.models[key]
            logger.info(f"Unregistered model: {key}")
