"""
Ollama HTTP API client for local model inference.
"""

import json
import time
import httpx
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import uuid


@dataclass
class ModelInfo:
    name: str
    digest: str
    size: int
    modified_at: str
    details: Dict[str, Any]


@dataclass
class GenerationResult:
    trial_id: str
    timestamp: str
    model: str
    model_digest: str
    backend: str
    backend_version: str
    condition: str
    case_id: str
    category: str
    system_prompt: str
    user_prompt: str
    conversation_history: List[Dict[str, str]]
    reasoning_mode_requested: str
    reasoning_mode_confirmed: str
    temperature: float
    top_p: float
    top_k: int
    seed: int
    context_length: int
    raw_response: str
    latency_ms: int
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    termination_reason: Optional[str]
    error: Optional[str]


class OllamaClient:
    def __init__(self, endpoint: str, timeout: int = 120):
        self.endpoint = endpoint.rstrip('/')
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self._backend_version = None
    
    def check_health(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            response = self.client.get(f"{self.endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_version(self) -> str:
        """Get Ollama version."""
        if self._backend_version:
            return self._backend_version
        try:
            response = self.client.get(f"{self.endpoint}/api/version", timeout=5)
            if response.status_code == 200:
                self._backend_version = response.json().get("version", "unknown")
            else:
                self._backend_version = "unknown"
        except Exception:
            self._backend_version = "unknown"
        return self._backend_version
    
    def list_models(self) -> List[ModelInfo]:
        """List available models."""
        response = self.client.get(f"{self.endpoint}/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        models = []
        for m in data.get("models", []):
            models.append(ModelInfo(
                name=m.get("name", ""),
                digest=m.get("digest", ""),
                size=m.get("size", 0),
                modified_at=m.get("modified_at", ""),
                details=m.get("details", {})
            ))
        return models
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists locally."""
        models = self.list_models()
        return any(m.name == model_name for m in models)
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed info for a specific model."""
        models = self.list_models()
        for m in models:
            if m.name == model_name:
                return m
        return None
    
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        seed: int = 42,
        context_length: int = 4096,
        thinking: Optional[bool] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> GenerationResult:
        """Generate a response from the model."""
        trial_id = str(uuid.uuid4())[:8]
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request
        request_data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
                "num_ctx": context_length,
            }
        }
        
        # Handle thinking/reasoning mode if specified
        if thinking is not None:
            request_data["options"]["think"] = thinking
        
        # Make request
        start_time = time.perf_counter()
        error = None
        raw_response = ""
        latency_ms = 0
        prompt_tokens = None
        completion_tokens = None
        termination_reason = None
        reasoning_mode_confirmed = "unknown"
        
        try:
            response = self.client.post(
                f"{self.endpoint}/api/chat",
                json=request_data,
                timeout=self.timeout
            )
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            
            if response.status_code == 200:
                data = response.json()
                raw_response = data.get("message", {}).get("content", "")
                
                # Extract token counts if available
                prompt_tokens = data.get("prompt_eval_count")
                completion_tokens = data.get("eval_count")
                termination_reason = data.get("done_reason")
                
                # Check if thinking was actually used
                if thinking is not None:
                    reasoning_mode_confirmed = "enabled" if thinking else "disabled"
                else:
                    reasoning_mode_confirmed = "default"
            else:
                error = f"HTTP {response.status_code}: {response.text}"
                raw_response = ""
        except httpx.TimeoutException:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            error = "Request timeout"
            raw_response = ""
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            error = f"{type(e).__name__}: {str(e)}"
            raw_response = ""
        
        # Get model info for digest
        model_info = self.get_model_info(model)
        model_digest = model_info.digest if model_info else "unknown"
        backend_version = self.get_version()
        
        return GenerationResult(
            trial_id=trial_id,
            timestamp=timestamp,
            model=model,
            model_digest=model_digest,
            backend="ollama",
            backend_version=backend_version,
            condition="",  # Will be set by caller
            case_id="",    # Will be set by caller
            category="",   # Will be set by caller
            system_prompt=system or "",
            user_prompt=prompt,
            conversation_history=conversation_history or [],
            reasoning_mode_requested="enabled" if thinking else "disabled" if thinking is False else "default",
            reasoning_mode_confirmed=reasoning_mode_confirmed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            context_length=context_length,
            raw_response=raw_response,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            termination_reason=termination_reason,
            error=error,
        )
    
    def close(self):
        self.client.close()


def create_client(endpoint: str, timeout: int) -> OllamaClient:
    """Factory function to create Ollama client."""
    return OllamaClient(endpoint, timeout)