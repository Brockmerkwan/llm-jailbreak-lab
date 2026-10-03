"""
Unit tests for configuration management.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.config import Config, load_config, GenerationConfig


class TestConfig:
    """Test configuration loading and validation."""
    
    def test_valid_config_load(self):
        """Test loading a valid configuration."""
        config_data = {
            'backend': 'ollama',
            'endpoint': 'http://127.0.0.1:11434',
            'model': 'test-model',
            'generation': {
                'temperature': 0.6,
                'top_p': 0.95,
                'top_k': 20,
                'seed': 42,
                'context_length': 4096,
            },
            'conditions': ['baseline', 'neutral'],
            'reasoning_modes': ['default', 'disabled'],
            'trials_per_condition': 10,
            'timeout_seconds': 120,
            'max_retries': 1,
            'seed_mode': 'fixed',
            'system_prompts': {},
            'input_files': ['benign.jsonl'],
            'output_directory': 'results',
            'lock_directory': '~/.local/state/llm-jailbreak-lab',
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            config = load_config(temp_path)
            assert config.backend == 'ollama'
            assert config.model == 'test-model'
            assert config.generation.temperature == 0.6
            assert config.trials_per_condition == 10
            assert 'baseline' in config.conditions
        finally:
            temp_path.unlink()
    
    def test_invalid_backend(self):
        """Test rejection of non-ollama backend."""
        config_data = {
            'backend': 'openai',
            'endpoint': 'http://127.0.0.1:11434',
            'model': 'test-model',
            'generation': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'seed': 42, 'context_length': 4096},
            'conditions': ['baseline'],
            'reasoning_modes': ['default'],
            'trials_per_condition': 10,
            'timeout_seconds': 120,
            'max_retries': 1,
            'seed_mode': 'fixed',
            'system_prompts': {},
            'input_files': ['benign.jsonl'],
            'output_directory': 'results',
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            assert "Unsupported backend" in str(exc_info.value)
        finally:
            temp_path.unlink()
    
    def test_non_local_endpoint_rejected(self):
        """Test rejection of non-local endpoints."""
        config_data = {
            'backend': 'ollama',
            'endpoint': 'https://api.openai.com',
            'model': 'test-model',
            'generation': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'seed': 42, 'context_length': 4096},
            'conditions': ['baseline'],
            'reasoning_modes': ['default'],
            'trials_per_condition': 10,
            'timeout_seconds': 120,
            'max_retries': 1,
            'seed_mode': 'fixed',
            'system_prompts': {},
            'input_files': ['benign.jsonl'],
            'output_directory': 'results',
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            assert "not a local address" in str(exc_info.value)
        finally:
            temp_path.unlink()
    
    def test_local_endpoints_accepted(self):
        """Test that local endpoints are accepted."""
        local_endpoints = [
            'http://127.0.0.1:11434',
            'http://localhost:11434',
            'http://::1:11434',
            'http://0.0.0.0:11434',
        ]
        
        for endpoint in local_endpoints:
            config_data = {
                'backend': 'ollama',
                'endpoint': endpoint,
                'model': 'test-model',
                'generation': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'seed': 42, 'context_length': 4096},
                'conditions': ['baseline'],
                'reasoning_modes': ['default'],
                'trials_per_condition': 10,
                'timeout_seconds': 120,
                'max_retries': 1,
                'seed_mode': 'fixed',
                'system_prompts': {},
                'input_files': ['benign.jsonl'],
                'output_directory': 'results',
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_data, f)
                temp_path = Path(f.name)
            
            try:
                config = load_config(temp_path)
                assert config.endpoint == endpoint
            finally:
                temp_path.unlink()
    
    def test_zero_trials_rejected(self):
        """Test that zero trials is rejected."""
        config_data = {
            'backend': 'ollama',
            'endpoint': 'http://127.0.0.1:11434',
            'model': 'test-model',
            'generation': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'seed': 42, 'context_length': 4096},
            'conditions': ['baseline'],
            'reasoning_modes': ['default'],
            'trials_per_condition': 0,
            'timeout_seconds': 120,
            'max_retries': 1,
            'seed_mode': 'fixed',
            'system_prompts': {},
            'input_files': ['benign.jsonl'],
            'output_directory': 'results',
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            assert "trials_per_condition must be positive" in str(exc_info.value)
        finally:
            temp_path.unlink()
    
    def test_invalid_seed_mode(self):
        """Test invalid seed mode rejected."""
        config_data = {
            'backend': 'ollama',
            'endpoint': 'http://127.0.0.1:11434',
            'model': 'test-model',
            'generation': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'seed': 42, 'context_length': 4096},
            'conditions': ['baseline'],
            'reasoning_modes': ['default'],
            'trials_per_condition': 10,
            'timeout_seconds': 120,
            'max_retries': 1,
            'seed_mode': 'random',
            'system_prompts': {},
            'input_files': ['benign.jsonl'],
            'output_directory': 'results',
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            assert "seed_mode must be 'fixed' or 'vary'" in str(exc_info.value)
        finally:
            temp_path.unlink()
    
    def test_unknown_condition_rejected(self):
        """Test unknown condition rejected."""
        config_data = {
            'backend': 'ollama',
            'endpoint': 'http://127.0.0.1:11434',
            'model': 'test-model',
            'generation': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'seed': 42, 'context_length': 4096},
            'conditions': ['baseline', 'unknown_condition'],
            'reasoning_modes': ['default'],
            'trials_per_condition': 10,
            'timeout_seconds': 120,
            'max_retries': 1,
            'seed_mode': 'fixed',
            'system_prompts': {},
            'input_files': ['benign.jsonl'],
            'output_directory': 'results',
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            assert "Unknown condition" in str(exc_info.value)
        finally:
            temp_path.unlink()
    
    def test_empty_input_files_rejected(self):
        """Test empty input files list rejected."""
        config_data = {
            'backend': 'ollama',
            'endpoint': 'http://127.0.0.1:11434',
            'model': 'test-model',
            'generation': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'seed': 42, 'context_length': 4096},
            'conditions': ['baseline'],
            'reasoning_modes': ['default'],
            'trials_per_condition': 10,
            'timeout_seconds': 120,
            'max_retries': 1,
            'seed_mode': 'fixed',
            'system_prompts': {},
            'input_files': [],
            'output_directory': 'results',
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            assert "At least one input file must be specified" in str(exc_info.value)
        finally:
            temp_path.unlink()
    
    def test_missing_config_file(self):
        """Test missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent.yaml"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])