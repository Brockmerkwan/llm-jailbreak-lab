"""
Configuration management for the LLM Jailbreak Evaluation Harness.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os


@dataclass
class GenerationConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 42
    context_length: int = 4096


@dataclass
class Config:
    backend: str = "ollama"
    endpoint: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    conditions: List[str] = field(default_factory=lambda: ["baseline", "neutral", "style_only", "explicit_anti_refusal", "persona"])
    reasoning_modes: List[str] = field(default_factory=lambda: ["default", "disabled"])
    trials_per_condition: int = 10
    timeout_seconds: int = 120
    max_retries: int = 1
    seed_mode: str = "fixed"
    system_prompts: Dict[str, str] = field(default_factory=dict)
    input_files: List[str] = field(default_factory=list)
    output_directory: str = "results"
    lock_directory: str = "~/.local/state/llm-jailbreak-lab"

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Handle generation config nested structure
        if 'generation' in data and isinstance(data['generation'], dict):
            data['generation'] = GenerationConfig(**data['generation'])
        
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if self.backend != "ollama":
            errors.append(f"Unsupported backend: {self.backend}. Only 'ollama' is supported.")
        
        # Validate endpoint is local
        if not self._is_local_endpoint(self.endpoint):
            errors.append(f"Endpoint {self.endpoint} is not a local address. Only localhost/127.0.0.1/::1 are allowed.")
        
        if self.trials_per_condition <= 0:
            errors.append("trials_per_condition must be positive")
        
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        
        if self.max_retries < 0:
            errors.append("max_retries cannot be negative")
        
        if self.seed_mode not in ("fixed", "vary"):
            errors.append(f"seed_mode must be 'fixed' or 'vary', got '{self.seed_mode}'")
        
        valid_conditions = {"baseline", "neutral", "style_only", "explicit_anti_refusal", "persona", "no_system", "thinking_enabled", "thinking_disabled"}
        for cond in self.conditions:
            if cond not in valid_conditions:
                errors.append(f"Unknown condition: {cond}")
        
        valid_reasoning = {"default", "disabled", "enabled"}
        for mode in self.reasoning_modes:
            if mode not in valid_reasoning:
                errors.append(f"Unknown reasoning mode: {mode}")
        
        if not self.input_files:
            errors.append("At least one input file must be specified")
        
        # Check input files exist
        for f in self.input_files:
            if not Path("cases") / f:
                # Will be checked at runtime with full path
                pass
        
        return errors
    
    def _is_local_endpoint(self, endpoint: str) -> bool:
        """Check if endpoint is a local address."""
        local_hosts = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
        try:
            from urllib.parse import urlparse
            parsed = urlparse(endpoint)
            host = parsed.hostname or ""
            # Handle IPv6 addresses which may be returned with brackets
            host = host.strip('[]')
            # Also check raw endpoint string for IPv6 loopback without brackets
            return host in local_hosts or host.startswith("127.") or host == "localhost" or "::1" in endpoint
        except Exception:
            return False


def load_config(config_path: Path) -> Config:
    """Load and validate configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = Config.from_yaml(config_path)
    errors = config.validate()
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return config