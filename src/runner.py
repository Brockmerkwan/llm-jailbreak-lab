"""
Main experiment runner for the LLM Jailbreak Evaluation Harness.
"""

import argparse
import json
import os
import sys
import time
import fcntl
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid

from .config import Config, load_config
from .ollama_client import OllamaClient, create_client, GenerationResult
from .classify import classify_response
from .analyze import generate_summary_csv, generate_detailed_csv
from .report import generate_report


class ExperimentLock:
    """File-based lock to prevent concurrent experiments."""
    
    def __init__(self, lock_dir: Path, experiment_id: str):
        self.lock_dir = lock_dir
        self.experiment_id = experiment_id
        self.lock_file = None
        self.lock_path = lock_dir / f"{experiment_id}.lock"
    
    def acquire(self) -> bool:
        """Acquire lock. Returns True if successful."""
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.lock_file = open(self.lock_path, 'w')
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_file.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n")
            self.lock_file.flush()
            return True
        except (IOError, OSError):
            if self.lock_file:
                self.lock_file.close()
            return False
    
    def release(self):
        """Release lock."""
        if self.lock_file:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self.lock_file.close()
            except Exception:
                pass
            self.lock_file = None
        try:
            self.lock_path.unlink(missing_ok=True)
        except Exception:
            pass
    
    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"Could not acquire lock for {self.experiment_id}. Another experiment may be running.")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


def load_cases(case_files: List[str], cases_dir: Path) -> List[Dict[str, Any]]:
    """Load test cases from JSONL files."""
    cases = []
    for fname in case_files:
        fpath = cases_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Case file not found: {fpath}")
        with open(fpath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                    if 'id' not in case:
                        case['id'] = f"{fname.stem}_{line_num}"
                    if 'category' not in case:
                        case['category'] = fname.replace('.jsonl', '')
                    cases.append(case)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {fpath}:{line_num}: {e}")
    return cases


def build_system_prompt(config: Config, condition: str) -> str:
    """Get system prompt for a condition."""
    return config.system_prompts.get(condition, "")


def should_use_system_prompt(condition: str) -> bool:
    """Check if condition should use a system prompt."""
    return condition != "no_system"


def build_conversation_history(condition: str, case: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build conversation history for context contamination tests."""
    # For now, only baseline fresh context
    # Could be extended for conversation-history condition
    return []


def get_reasoning_mode(condition: str, reasoning_mode: str) -> Optional[bool]:
    """Map reasoning mode string to boolean for Ollama API."""
    if reasoning_mode == "disabled":
        return False
    elif reasoning_mode == "enabled":
        return True
    return None  # default


def run_single_trial(
    client: OllamaClient,
    config: Config,
    case: Dict[str, Any],
    condition: str,
    reasoning_mode: str,
    trial_num: int,
    base_seed: int,
) -> GenerationResult:
    """Run a single generation trial."""
    # Determine seed
    if config.seed_mode == "vary":
        seed = base_seed + trial_num
    else:
        seed = base_seed
    
    # Get system prompt
    system_prompt = build_system_prompt(config, condition) if should_use_system_prompt(condition) else ""
    
    # Get conversation history
    conversation_history = build_conversation_history(condition, case)
    
    # Get reasoning mode
    thinking = get_reasoning_mode(condition, reasoning_mode)
    
    # Run generation
    result = client.generate(
        model=config.model,
        prompt=case['prompt'],
        system=system_prompt,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        top_k=config.generation.top_k,
        seed=seed,
        context_length=config.generation.context_length,
        thinking=thinking,
        conversation_history=conversation_history,
    )
    
    # Add experiment metadata
    result.condition = condition
    result.case_id = case['id']
    result.category = case.get('category', 'unknown')
    
    return result


def run_experiment(config: Config, cases: List[Dict[str, Any]], experiment_dir: Path, dry_run: bool = False) -> List[GenerationResult]:
    """Run the full experiment."""
    client = create_client(config.endpoint, config.timeout_seconds)
    
    # Verify model exists
    if not client.model_exists(config.model):
        raise ValueError(f"Model '{config.model}' not found in Ollama. Available: {[m.name for m in client.list_models()]}")
    
    model_info = client.get_model_info(config.model)
    model_digest = model_info.digest if model_info else "unknown"
    
    all_results = []
    
    # Calculate total expected generations
    total_expected = len(cases) * len(config.conditions) * len(config.reasoning_modes) * config.trials_per_condition
    
    print(f"Starting experiment: {experiment_dir.name}")
    print(f"Model: {config.model} ({model_digest})")
    print(f"Cases: {len(cases)}, Conditions: {len(config.conditions)}, Reasoning modes: {len(config.reasoning_modes)}, Trials: {config.trials_per_condition}")
    print(f"Expected generations: {total_expected}")
    
    if dry_run:
        print("DRY RUN - No inference will be performed")
        client.close()
        return []
    
    # Preflight info already printed by caller
    
    trial_count = 0
    for case in cases:
        for condition in config.conditions:
            for reasoning_mode in config.reasoning_modes:
                for trial in range(config.trials_per_condition):
                    trial_count += 1
                    print(f"  [{trial_count}/{total_expected}] Case: {case['id']}, Condition: {condition}, Reasoning: {reasoning_mode}, Trial: {trial+1}")
                    
                    # Retry logic
                    for attempt in range(config.max_retries + 1):
                        result = run_single_trial(
                            client, config, case, condition, reasoning_mode, trial, config.generation.seed
                        )
                        
                        if result.error and attempt < config.max_retries:
                            print(f"    Attempt {attempt+1} failed: {result.error}. Retrying...")
                            time.sleep(1)
                            continue
                        break
                    
                    # Override trial_id with deterministic index for resume capability
                    result.trial_id = f"trial_{trial:03d}"
                    all_results.append(result)
    
    client.close()
    return all_results


def save_raw_results(experiment_dir: Path, results: List[GenerationResult]):
    """Save raw results to JSONL."""
    raw_path = experiment_dir / "raw.jsonl"
    with open(raw_path, 'w', encoding='utf-8') as f:
        for r in results:
            # Convert to dict, handling dataclass
            data = {
                'trial_id': r.trial_id,
                'timestamp': r.timestamp,
                'model': r.model,
                'model_digest': r.model_digest,
                'backend': r.backend,
                'backend_version': r.backend_version,
                'condition': r.condition,
                'case_id': r.case_id,
                'category': r.category,
                'system_prompt': r.system_prompt,
                'user_prompt': r.user_prompt,
                'conversation_history': r.conversation_history,
                'reasoning_mode_requested': r.reasoning_mode_requested,
                'reasoning_mode_confirmed': r.reasoning_mode_confirmed,
                'temperature': r.temperature,
                'top_p': r.top_p,
                'top_k': r.top_k,
                'seed': r.seed,
                'context_length': r.context_length,
                'raw_response': r.raw_response,
                'latency_ms': r.latency_ms,
                'prompt_tokens': r.prompt_tokens,
                'completion_tokens': r.completion_tokens,
                'termination_reason': r.termination_reason,
                'error': r.error,
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def save_scored_results(experiment_dir: Path, results: List[GenerationResult]):
    """Save classified/scored results to JSONL."""
    scored_path = experiment_dir / "scored.jsonl"
    with open(scored_path, 'w', encoding='utf-8') as f:
        for r in results:
            classification = classify_response(r.raw_response, r.error, r.category)
            data = {
                'trial_id': r.trial_id,
                'timestamp': r.timestamp,
                'model': r.model,
                'model_digest': r.model_digest,
                'backend': r.backend,
                'backend_version': r.backend_version,
                'condition': r.condition,
                'case_id': r.case_id,
                'category': r.category,
                'system_prompt': r.system_prompt,
                'user_prompt': r.user_prompt,
                'conversation_history': r.conversation_history,
                'reasoning_mode_requested': r.reasoning_mode_requested,
                'reasoning_mode_confirmed': r.reasoning_mode_confirmed,
                'temperature': r.temperature,
                'top_p': r.top_p,
                'top_k': r.top_k,
                'seed': r.seed,
                'context_length': r.context_length,
                'raw_response': r.raw_response,
                'latency_ms': r.latency_ms,
                'prompt_tokens': r.prompt_tokens,
                'completion_tokens': r.completion_tokens,
                'termination_reason': r.termination_reason,
                'error': r.error,
                'labels': classification.labels,
                'matched_rules': classification.matched_rules,
                'confidence': classification.confidence,
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def save_manifest(experiment_dir: Path, config: Config, cases: List[Dict[str, Any]], results: List[GenerationResult], model_digest: str):
    """Save experiment manifest."""
    manifest = {
        'experiment_id': experiment_dir.name,
        'timestamp': datetime.now().isoformat(),
        'model': config.model,
        'model_digest': model_digest,
        'backend': config.backend,
        'backend_version': results[0].backend_version if results else "unknown",
        'config': {
            'backend': config.backend,
            'endpoint': config.endpoint,
            'model': config.model,
            'generation': {
                'temperature': config.generation.temperature,
                'top_p': config.generation.top_p,
                'top_k': config.generation.top_k,
                'seed': config.generation.seed,
                'context_length': config.generation.context_length,
            },
            'conditions': config.conditions,
            'reasoning_modes': config.reasoning_modes,
            'trials_per_condition': config.trials_per_condition,
            'timeout_seconds': config.timeout_seconds,
            'max_retries': config.max_retries,
            'seed_mode': config.seed_mode,
            'input_files': config.input_files,
        },
        'total_cases': len(cases),
        'total_trials': len(results),
        'conditions': config.conditions,
        'reasoning_modes': config.reasoning_modes,
        'cases': [{'id': c['id'], 'category': c.get('category', ''), 'prompt': c['prompt'][:100]} for c in cases],
    }
    
    manifest_path = experiment_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def print_preflight(config: Config, cases: List[Dict[str, Any]], experiment_dir: Path, client: OllamaClient):
    """Print preflight audit block."""
    model_info = client.get_model_info(config.model)
    model_digest = model_info.digest if model_info else "unknown"
    backend_version = client.get_version()
    
    total_expected = len(cases) * len(config.conditions) * len(config.reasoning_modes) * config.trials_per_condition
    
    print("=== PREFLIGHT ===")
    print(f"Backend: {config.backend}")
    print(f"Endpoint: {config.endpoint}")
    print(f"Model: {config.model}")
    print(f"Model ID / digest: {model_digest}")
    print(f"Trials per condition: {config.trials_per_condition}")
    print(f"Conditions: {', '.join(config.conditions)}")
    print(f"Case count: {len(cases)}")
    print(f"Expected generations: {total_expected}")
    print(f"Temperature: {config.generation.temperature}")
    print(f"Top-p: {config.generation.top_p}")
    print(f"Top-k: {config.generation.top_k}")
    print(f"Seed: {config.generation.seed}")
    print(f"Reasoning modes: {', '.join(config.reasoning_modes)}")
    print(f"Output directory: {experiment_dir}")
    print(f"External network use: DISABLED")
    print(f"Generated-output execution: DISABLED")
    print("")


def create_experiment_dir(output_dir: Path) -> Path:
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    experiment_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    experiment_dir = output_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def resume_experiment(experiment_dir: Path, config: Config, cases: List[Dict[str, Any]]) -> List[GenerationResult]:
    """Resume an interrupted experiment."""
    raw_path = experiment_dir / "raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"No raw results found in {experiment_dir}")
    
    # Load existing results
    existing_results = []
    completed_trial_keys = set()
    
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            existing_results.append(r)
            # Create unique key for this trial using deterministic trial_id format
            key = f"{r['case_id']}__{r['condition']}__{r['reasoning_mode_requested']}__{r['trial_id']}"
            completed_trial_keys.add(key)
    
    print(f"Resuming experiment: found {len(existing_results)} completed trials")
    
    # Determine which trials are missing
    client = create_client(config.endpoint, config.timeout_seconds)
    if not client.model_exists(config.model):
        raise ValueError(f"Model '{config.model}' not found in Ollama")
    
    model_info = client.get_model_info(config.model)
    model_digest = model_info.digest if model_info else "unknown"
    
    new_results = []
    total_expected = len(cases) * len(config.conditions) * len(config.reasoning_modes) * config.trials_per_condition
    
    trial_count = len(existing_results)
    
    for case in cases:
        for condition in config.conditions:
            for reasoning_mode in config.reasoning_modes:
                for trial in range(config.trials_per_condition):
                    # Use deterministic trial_id format for matching
                    trial_id = f"trial_{trial:03d}"
                    trial_key = f"{case['id']}__{condition}__{reasoning_mode}__{trial_id}"
                    if trial_key in completed_trial_keys:
                        continue
                    
                    trial_count += 1
                    print(f"  [{trial_count}/{total_expected}] Case: {case['id']}, Condition: {condition}, Reasoning: {reasoning_mode}, Trial: {trial+1} (resuming)")
                    
                    for attempt in range(config.max_retries + 1):
                        result = run_single_trial(
                            client, config, case, condition, reasoning_mode, trial, config.generation.seed
                        )
                        if result.error and attempt < config.max_retries:
                            print(f"    Attempt {attempt+1} failed: {result.error}. Retrying...")
                            time.sleep(1)
                            continue
                        break
                    
                    # Set deterministic trial_id
                    result.trial_id = trial_id
                    new_results.append(result)
                    # Append to raw file immediately
                    save_raw_results(experiment_dir, [result])
    
    client.close()
    
    # Combine and return all results
    all_results = existing_results + new_results
    return all_results


def run_preflight_check(config: Config, cases: List[Dict[str, Any]], output_dir: Path):
    """Run preflight checks without executing experiment."""
    client = create_client(config.endpoint, config.timeout_seconds)
    
    print("=== PREFLIGHT CHECK ===")
    
    # Check Ollama reachable
    if not client.check_health():
        print("❌ Ollama not reachable at", config.endpoint)
        sys.exit(1)
    print("✅ Ollama reachable")
    
    # Check model exists
    if not client.model_exists(config.model):
        print(f"❌ Model '{config.model}' not found locally")
        available = [m.name for m in client.list_models()]
        print(f"   Available models: {available}")
        sys.exit(1)
    print(f"✅ Model '{config.model}' found locally")
    
    model_info = client.get_model_info(config.model)
    print(f"   Model digest: {model_info.digest}")
    print(f"   Model size: {model_info.size / 1e9:.1f} GB")
    
    # Check case files
    print(f"✅ Case files: {len(config.input_files)} files, {len(cases)} total cases")
    for f in config.input_files:
        print(f"   - {f}")
    
    # Check output directory
    if not output_dir.exists():
        print(f"⚠️  Output directory {output_dir} does not exist, will be created")
    elif not os.access(output_dir, os.W_OK):
        print(f"❌ Output directory {output_dir} not writable")
        sys.exit(1)
    else:
        print(f"✅ Output directory writable")
    
    # Check for existing lock
    lock_dir = Path(config.lock_directory).expanduser()
    lock_dir.mkdir(parents=True, exist_ok=True)
    
    total_expected = len(cases) * len(config.conditions) * len(config.reasoning_modes) * config.trials_per_condition
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model}")
    print(f"  Cases: {len(cases)}")
    print(f"  Conditions: {len(config.conditions)}")
    print(f"  Reasoning modes: {len(config.reasoning_modes)}")
    print(f"  Trials each: {config.trials_per_condition}")
    print(f"  Expected generations: {total_expected}")
    print(f"  Seed mode: {config.seed_mode}")
    print(f"  Max retries: {config.max_retries}")
    print(f"  Timeout: {config.timeout_seconds}s")
    
    client.close()
    print("\n✅ Preflight checks passed")


def main():
    parser = argparse.ArgumentParser(description="LLM Jailbreak Evaluation Harness")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config YAML")
    parser.add_argument("--cases", type=str, help="Comma-separated case files (overrides config)")
    parser.add_argument("--conditions", type=str, help="Comma-separated conditions (overrides config)")
    parser.add_argument("--trials", type=int, help="Trials per condition (overrides config)")
    parser.add_argument("--model", type=str, help="Model name (overrides config)")
    parser.add_argument("--seed-mode", type=str, choices=["fixed", "vary"], help="Seed mode (overrides config)")
    parser.add_argument("--dry-run", action="store_true", help="Show experiment plan without running")
    parser.add_argument("--preflight", action="store_true", help="Run preflight checks only")
    parser.add_argument("--resume", type=Path, help="Resume experiment from directory")
    parser.add_argument("--force", action="store_true", help="Force regenerate completed trials on resume")
    parser.add_argument("--output-dir", type=Path, help="Output directory (overrides config)")
    
    args = parser.parse_args()
    
    # Load config
    if not args.config.exists():
        print(f"Config file not found: {args.config}")
        print("Copy config.example.yaml to config.yaml and modify as needed")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.cases:
        config.input_files = [f.strip() for f in args.cases.split(",")]
    if args.conditions:
        config.conditions = [c.strip() for c in args.conditions.split(",")]
    if args.trials:
        config.trials_per_condition = args.trials
    if args.model:
        config.model = args.model
    if args.seed_mode:
        config.seed_mode = args.seed_mode
    if args.output_dir:
        config.output_directory = str(args.output_dir)
    
    # Load cases
    cases_dir = Path("cases")
    cases = load_cases(config.input_files, cases_dir)
    
    output_dir = Path(config.output_directory)
    
    # Handle resume
    if args.resume:
        experiment_dir = args.resume
        if not experiment_dir.exists():
            print(f"Experiment directory not found: {experiment_dir}")
            sys.exit(1)
        
        lock = ExperimentLock(Path(config.lock_directory).expanduser(), experiment_dir.name)
        with lock:
            print(f"Resuming experiment: {experiment_dir.name}")
            results = resume_experiment(experiment_dir, config, cases)
            
            # Re-save scored results
            scored_results = []
            for r in results:
                # Convert dict back to GenerationResult-like object
                class ResultObj:
                    pass
                obj = ResultObj()
                for k, v in r.items():
                    setattr(obj, k, v)
                scored_results.append(obj)
            
            save_scored_results(experiment_dir, scored_results)
            generate_summary_csv(experiment_dir, {})
            generate_detailed_csv(experiment_dir, {})
            generate_report(experiment_dir)
            print(f"Resumed experiment completed. Results in {experiment_dir}")
        return
    
    # Run preflight check
    client = create_client(config.endpoint, config.timeout_seconds)
    run_preflight_check(config, cases, output_dir)
    
    if args.preflight:
        client.close()
        return
    
    if args.dry_run:
        total_expected = len(cases) * len(config.conditions) * len(config.reasoning_modes) * config.trials_per_condition
        print("\nExperiment plan")
        print(f"Model: {config.model}")
        print(f"Cases: {len(cases)}")
        print(f"Conditions: {len(config.conditions)}")
        print(f"Reasoning modes: {len(config.reasoning_modes)}")
        print(f"Trials each: {config.trials_per_condition}")
        print(f"Expected generations: {total_expected}")
        print("\nNo inference performed.")
        client.close()
        return
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(output_dir)
    print(f"\nExperiment directory: {experiment_dir}")
    
    # Acquire lock
    lock = ExperimentLock(Path(config.lock_directory).expanduser(), experiment_dir.name)
    with lock:
        # Print preflight
        print_preflight(config, cases, experiment_dir, client)
        
        # Run experiment
        results = run_experiment(config, cases, experiment_dir, dry_run=False)
        
        # Save results
        print("\nSaving results...")
        save_raw_results(experiment_dir, results)
        save_scored_results(experiment_dir, results)
        save_manifest(experiment_dir, config, cases, results, client.get_model_info(config.model).digest if client.get_model_info(config.model) else "unknown")
        
        # Generate analysis outputs
        print("Generating analysis...")
        stats = {}
        from .analyze import compute_all_condition_stats
        scored_results = []
        for r in results:
            class ResultObj:
                pass
            obj = ResultObj()
            for k, v in r.__dict__.items():
                setattr(obj, k, v)
            scored_results.append(obj)
        stats = compute_all_condition_stats([
            {
                'condition': r.condition,
                'labels': classify_response(r.raw_response, r.error, r.category).labels,
                'latency_ms': r.latency_ms,
            }
            for r in results
        ])
        generate_summary_csv(experiment_dir, stats)
        generate_detailed_csv(experiment_dir, {})
        generate_report(experiment_dir)
        
        print(f"\n✅ Experiment completed: {experiment_dir}")
        print(f"   Raw results: {experiment_dir}/raw.jsonl")
        print(f"   Scored results: {experiment_dir}/scored.jsonl")
        print(f"   Summary CSV: {experiment_dir}/summary.csv")
        print(f"   Report: {experiment_dir}/REPORT.md")
    
    client.close()


if __name__ == "__main__":
    main()