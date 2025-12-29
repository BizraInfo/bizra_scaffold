"""
BIZRA AEON OMEGA - Ollama LLM Client
Genesis Block 3-Layer Architecture - Local Model Integration

Provides async client for Ollama API with:
- Health monitoring
- Streaming support
- Department-aware model selection
- Ihsan compliance enforcement
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from core.llm.provider_base import (
    LLMProvider,
    LLMResponse,
    ModelConfig,
    ProviderType,
    DepartmentID,
)

logger = logging.getLogger(__name__)

# Default Ollama endpoint
OLLAMA_DEFAULT_HOST = "http://localhost:11434"


class OllamaClient(LLMProvider):
    """
    Async Ollama API client for local LLM inference.
    
    Supports:
    - Multiple models (DeepSeek, Qwen, Llama, Mistral, etc.)
    - Streaming responses
    - Health checks
    - Model hot-swapping
    """

    def __init__(
        self,
        config: ModelConfig,
        host: str = OLLAMA_DEFAULT_HOST,
        timeout: float = 120.0,
    ):
        super().__init__(config)
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for OllamaClient")
        
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.host}/api/tags") as resp:
                if resp.status != 200:
                    self._healthy = False
                    return False
                
                data = await resp.json()
                models = [m["name"] for m in data.get("models", [])]
                
                # Check if our model is available
                model_available = any(
                    self.config.model_id in m for m in models
                )
                
                self._healthy = model_available
                return model_available
                
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            self._healthy = False
            return False

    async def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.host}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [m["name"] for m in data.get("models", [])]
                return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion using Ollama."""
        start_time = time.perf_counter()
        
        payload = {
            "model": self.config.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.host}/api/generate",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    self.record_error()
                    raise RuntimeError(f"Ollama error: {error_text}")
                
                data = await resp.json()
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                self.record_success()
                
                return LLMResponse(
                    content=data.get("response", ""),
                    model=self.config.model_id,
                    provider=ProviderType.OLLAMA,
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                    total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                    latency_ms=latency_ms,
                    finish_reason="stop",
                    metadata={
                        "total_duration": data.get("total_duration"),
                        "load_duration": data.get("load_duration"),
                        "context": data.get("context"),
                    },
                )
                
        except aiohttp.ClientError as e:
            self.record_error()
            raise RuntimeError(f"Ollama connection error: {e}")

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion tokens from Ollama."""
        payload = {
            "model": self.config.model_id,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.host}/api/generate",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    self.record_error()
                    raise RuntimeError(f"Ollama stream error: {error_text}")
                
                async for line in resp.content:
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                self.record_success()
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except aiohttp.ClientError as e:
            self.record_error()
            raise RuntimeError(f"Ollama stream connection error: {e}")

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()


# Department-to-model mapping for Genesis Council
DEPARTMENT_MODEL_MAPPING: Dict[DepartmentID, List[str]] = {
    DepartmentID.CRYPTOGRAPHY: ["deepseek-coder:14b", "codellama:13b"],
    DepartmentID.ECONOMICS: ["qwen:14b", "mistral:7b"],
    DepartmentID.PHILOSOPHY: ["llama3.1:8b", "phi3:14b"],
    DepartmentID.GOVERNANCE: ["mistral:7b", "llama3.1:8b"],
    DepartmentID.SYSTEMS: ["codegemma:7b", "deepseek-coder:14b"],
    DepartmentID.COGNITIVE: ["phi3:14b", "qwen:14b"],
    DepartmentID.OPERATIONS: ["llama3.1:8b", "mistral:7b"],
}
