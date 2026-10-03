"""
Markdown report generator for experiment results.
"""

import json
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from .analyze import (
    ConditionStats, ComparisonResult, 
    compute_all_condition_stats, compare_conditions,
    analyze_by_reasoning_mode, analyze_trigger_phrases,
    load_results
)


@dataclass
class ExperimentMetadata:
    experiment_id: str
    model: str
    model_digest: str
    backend: str
    backend_version: str
    timestamp: str
    config: Dict[str, Any]
    total_trials: int
    total_cases: int
    conditions: List[str]
    reasoning_modes: List[str]


def load_metadata(experiment_dir: Path) -> ExperimentMetadata:
    """Load experiment metadata from manifest."""
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.exists():
        return ExperimentMetadata(
            experiment_id=experiment_dir.name,
            model="unknown", model_digest="unknown",
            backend="unknown", backend_version="unknown",
            timestamp="unknown", config={},
            total_trials=0, total_cases=0,
            conditions=[], reasoning_modes=[]
        )
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return ExperimentMetadata(
        experiment_id=data.get('experiment_id', experiment_dir.name),
        model=data.get('model', 'unknown'),
        model_digest=data.get('model_digest', 'unknown'),
        backend=data.get('backend', 'unknown'),
        backend_version=data.get('backend_version', 'unknown'),
        timestamp=data.get('timestamp', 'unknown'),
        config=data.get('config', {}),
        total_trials=data.get('total_trials', 0),
        total_cases=data.get('total_cases', 0),
        conditions=data.get('conditions', []),
        reasoning_modes=data.get('reasoning_modes', []),
    )


def format_rate(rate: float) -> str:
    """Format rate as percentage."""
    return f"{rate * 100:.1f}%"


def format_count_rate(count: int, total: int) -> str:
    """Format as 'count/total = rate%'."""
    if total == 0:
        return "0/0 = N/A"
    return f"{count}/{total} = {count/total*100:.1f}%"


def generate_condition_table(stats: Dict[str, ConditionStats]) -> str:
    """Generate markdown table comparing conditions."""
    lines = []
    lines.append("| Condition | Direct Answer | Hard Refusal | Soft Refusal | Warning | Lecture | Redirect | Hedging |")
    lines.append("|-----------|---------------|--------------|--------------|---------|---------|----------|---------|")
    
    for condition, s in sorted(stats.items()):
        lines.append(f"| {condition} | {format_count_rate(s.direct_answers, s.total_trials)} | "
                     f"{format_count_rate(s.hard_refusals, s.total_trials)} | "
                     f"{format_count_rate(s.soft_refusals, s.total_trials)} | "
                     f"{format_count_rate(s.warnings, s.total_trials)} | "
                     f"{format_count_rate(s.lectures, s.total_trials)} | "
                     f"{format_count_rate(s.redirects, s.total_trials)} | "
                     f"{format_count_rate(s.hedging, s.total_trials)} |")
    
    return "\n".join(lines)


def generate_reasoning_table(stats: Dict[str, ConditionStats]) -> str:
    """Generate markdown table comparing reasoning modes."""
    lines = []
    lines.append("| Reasoning Mode | Direct Answer | Hard Refusal | Soft Refusal | Warning | Lecture | Median Latency |")
    lines.append("|----------------|---------------|--------------|--------------|---------|---------|----------------|")
    
    for mode, s in sorted(stats.items()):
        lines.append(f"| {mode} | {format_count_rate(s.direct_answers, s.total_trials)} | "
                     f"{format_count_rate(s.hard_refusals, s.total_trials)} | "
                     f"{format_count_rate(s.soft_refusals, s.total_trials)} | "
                     f"{format_count_rate(s.warnings, s.total_trials)} | "
                     f"{format_count_rate(s.lectures, s.total_trials)} | "
                     f"{s.median_latency_ms:.0f} ms |")
    
    return "\n".join(lines)


def generate_trigger_analysis(results: List[Dict[str, Any]]) -> str:
    """Generate analysis of trigger phrase effects."""
    lines = []
    lines.append("### Trigger Phrase Analysis")
    lines.append("")
    lines.append("Comparison of framing conditions on benign and alignment prompts.")
    lines.append("")
    
    # Group by category
    categories = defaultdict(lambda: defaultdict(list))
    for r in results:
        cat = r.get('category', 'unknown')
        cond = r.get('condition', 'unknown')
        categories[cat][cond].append(r)
    
    for category in sorted(categories.keys()):
        lines.append(f"#### Category: {category}")
        lines.append("")
        lines.append("| Condition | Direct | Hard Refusal | Soft Refusal | Lecture | Warning |")
        lines.append("|-----------|--------|--------------|--------------|---------|---------|")
        
        for condition in ['baseline', 'neutral', 'style_only', 'explicit_anti_refusal']:
            if condition not in categories[category]:
                continue
            cat_results = categories[category][condition]
            total = len(cat_results)
            if total == 0:
                continue
            direct = sum(1 for r in cat_results if 'direct_answer' in r.get('labels', []))
            hard = sum(1 for r in cat_results if 'hard_refusal' in r.get('labels', []))
            soft = sum(1 for r in cat_results if 'soft_refusal' in r.get('labels', []))
            lecture = sum(1 for r in cat_results if 'lecture' in r.get('labels', []))
            warning = sum(1 for r in cat_results if 'warning' in r.get('labels', []))
            
            lines.append(f"| {condition} | {format_count_rate(direct, total)} | "
                         f"{format_count_rate(hard, total)} | "
                         f"{format_count_rate(soft, total)} | "
                         f"{format_count_rate(lecture, total)} | "
                         f"{format_count_rate(warning, total)} |")
        
        lines.append("")
    
    return "\n".join(lines)


def generate_persona_analysis(results: List[Dict[str, Any]]) -> str:
    """Generate persona adherence analysis."""
    lines = []
    lines.append("### Persona Adherence Analysis")
    lines.append("")
    
    persona_results = [r for r in results if r.get('category') == 'persona']
    if not persona_results:
        lines.append("No persona category results found.")
        return "\n".join(lines)
    
    lines.append("| Condition | Persona Adherence | Direct Answer | Hard Refusal | Lecture |")
    lines.append("|-----------|-------------------|---------------|--------------|---------|")
    
    conditions = defaultdict(list)
    for r in persona_results:
        cond = r.get('condition', 'unknown')
        conditions[cond].append(r)
    
    for condition in sorted(conditions.keys()):
        cond_results = conditions[condition]
        total = len(cond_results)
        adherence = sum(1 for r in cond_results if 'persona_adherence' in r.get('labels', []))
        direct = sum(1 for r in cond_results if 'direct_answer' in r.get('labels', []))
        hard = sum(1 for r in cond_results if 'hard_refusal' in r.get('labels', []))
        lecture = sum(1 for r in cond_results if 'lecture' in r.get('labels', []))
        
        lines.append(f"| {condition} | {format_count_rate(adherence, total)} | "
                     f"{format_count_rate(direct, total)} | "
                     f"{format_count_rate(hard, total)} | "
                     f"{format_count_rate(lecture, total)} |")
    
    lines.append("")
    return "\n".join(lines)


def generate_stability_analysis(results: List[Dict[str, Any]]) -> str:
    """Analyze variance across repeated trials."""
    lines = []
    lines.append("### Sampling Stability Analysis")
    lines.append("")
    lines.append("Variance in classification across repeated trials for the same case/condition.")
    lines.append("")
    
    # Group by case_id, condition, and trial (looking for trial_id pattern)
    grouped = defaultdict(list)
    for r in results:
        key = (r.get('case_id', ''), r.get('condition', ''))
        grouped[key].append(r)
    
    lines.append("| Case | Condition | Trials | Direct Rate | Hard Refusal Rate | Lecture Rate | Std Dev (Direct) |")
    lines.append("|------|-----------|--------|-------------|-------------------|--------------|------------------|")
    
    for (case_id, condition), case_results in sorted(grouped.items()):
        total = len(case_results)
        if total < 2:
            continue
        
        # Compute per-trial rates (each trial is one result here)
        direct_rates = [1.0 if 'direct_answer' in r.get('labels', []) else 0.0 for r in case_results]
        hard_rates = [1.0 if 'hard_refusal' in r.get('labels', []) else 0.0 for r in case_results]
        lecture_rates = [1.0 if 'lecture' in r.get('labels', []) else 0.0 for r in case_results]
        
        direct_rate = sum(direct_rates) / total
        hard_rate = sum(hard_rates) / total
        lecture_rate = sum(lecture_rates) / total
        
        # Standard deviation
        direct_std = statistics.stdev(direct_rates) if len(direct_rates) > 1 else 0
        
        lines.append(f"| {case_id} | {condition} | {total} | {direct_rate*100:.1f}% | "
                     f"{hard_rate*100:.1f}% | {lecture_rate*100:.1f}% | {direct_std:.3f} |")
    
    lines.append("")
    return "\n".join(lines)


def generate_failure_analysis(results: List[Dict[str, Any]]) -> str:
    """Analyze failures."""
    lines = []
    lines.append("### Failure Analysis")
    lines.append("")
    
    errors = [r for r in results if 'generation_error' in r.get('labels', [])]
    empty = [r for r in results if 'empty_response' in r.get('labels', [])]
    
    lines.append(f"**Generation Errors:** {len(errors)}")
    lines.append(f"**Empty Responses:** {len(empty)}")
    lines.append("")
    
    if errors:
        lines.append("#### Error Details")
        lines.append("")
        error_types = defaultdict(int)
        for r in errors:
            err = r.get('error', 'unknown')
            error_types[err] += 1
        for err, count in sorted(error_types.items()):
            lines.append(f"- {err}: {count}")
        lines.append("")
    
    return "\n".join(lines)


def generate_observations_and_hypotheses(
    stats: Dict[str, ConditionStats],
    reasoning_stats: Dict[str, ConditionStats],
    results: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """Generate observations and hypotheses sections."""
    observations = []
    hypotheses = []
    
    # Compare neutral vs explicit_anti_refusal
    if 'neutral' in stats and 'explicit_anti_refusal' in stats:
        neu = stats['neutral']
        ear = stats['explicit_anti_refusal']
        
        if ear.lecture_rate > neu.lecture_rate + 0.1:
            obs = (f"Observation: The explicit anti-refusal condition produced lectures in "
                   f"{ear.lectures}/{ear.total_trials} trials versus "
                   f"{neu.lectures}/{neu.total_trials} in the neutral condition.")
            observations.append(obs)
            hyp = ("Hypothesis: Anti-refusal language may activate patterns associated with "
                   "adversarial or safety-training examples.")
            hypotheses.append(hyp)
        
        if ear.hard_refusal_rate > neu.hard_refusal_rate + 0.1:
            obs = (f"Observation: The explicit anti-refusal condition produced hard refusals in "
                   f"{ear.hard_refusals}/{ear.total_trials} trials versus "
                   f"{neu.hard_refusals}/{neu.total_trials} in the neutral condition.")
            observations.append(obs)
            hyp = ("Hypothesis: Explicit 'do not refuse' phrasing may paradoxically trigger "
                   "refusal behavior learned during alignment training.")
            hypotheses.append(hyp)
    
    # Compare reasoning modes
    if len(reasoning_stats) >= 2:
        modes = list(reasoning_stats.keys())
        for i, mode_a in enumerate(modes):
            for mode_b in modes[i+1:]:
                sa = reasoning_stats[mode_a]
                sb = reasoning_stats[mode_b]
                
                if abs(sa.direct_answer_rate - sb.direct_answer_rate) > 0.1:
                    obs = (f"Observation: Reasoning mode '{mode_a}' produced direct answers in "
                           f"{sa.direct_answers}/{sa.total_trials} trials versus "
                           f"{sb.direct_answers}/{sb.total_trials} for '{mode_b}'.")
                    observations.append(obs)
                    hyp = ("Hypothesis: Disabling explicit reasoning may not remove learned "
                           "deliberative prose patterns.")
                    hypotheses.append(hyp)
    
    # Stability check
    grouped = defaultdict(list)
    for r in results:
        key = (r.get('case_id', ''), r.get('condition', ''))
        grouped[key].append(r)
    
    high_variance_cases = 0
    for (case_id, condition), case_results in grouped.items():
        if len(case_results) >= 5:
            direct_rates = [1.0 if 'direct_answer' in r.get('labels', []) else 0.0 for r in case_results]
            if len(set(direct_rates)) > 1:  # Some variation
                high_variance_cases += 1
    
    if high_variance_cases > 0:
        obs = (f"Observation: {high_variance_cases} case/condition combinations showed "
               f"variability across repeated trials.")
        observations.append(obs)
        hyp = ("Hypothesis: Model behavior is non-deterministic even at fixed temperature; "
               "adequate trial counts are essential for reliable measurement.")
        hypotheses.append(hyp)
    
    return "\n".join(observations) + "\n", "\n".join(hypotheses) + "\n"


def generate_report(experiment_dir: Path) -> Path:
    """Generate the full REPORT.md file."""
    results = load_results(experiment_dir)
    metadata = load_metadata(experiment_dir)
    stats = compute_all_condition_stats(results)
    reasoning_stats = analyze_by_reasoning_mode(results)
    trigger_stats = analyze_trigger_phrases(results)
    
    observations, hypotheses = generate_observations_and_hypotheses(stats, reasoning_stats, results)
    
    lines = []
    lines.append(f"# Experiment Report: {metadata.experiment_id}")
    lines.append("")
    
    # Experiment Metadata
    lines.append("## Experiment Metadata")
    lines.append("")
    lines.append(f"- **Model:** {metadata.model}")
    lines.append(f"- **Model Digest:** {metadata.model_digest}")
    lines.append(f"- **Backend:** {metadata.backend} ({metadata.backend_version})")
    lines.append(f"- **Date:** {metadata.timestamp}")
    lines.append(f"- **Total Trials:** {metadata.total_trials}")
    lines.append(f"- **Total Cases:** {metadata.total_cases}")
    lines.append(f"- **Conditions:** {', '.join(metadata.conditions)}")
    lines.append(f"- **Reasoning Modes:** {', '.join(metadata.reasoning_modes)}")
    lines.append(f"- **Trials per Condition:** {metadata.config.get('trials_per_condition', 'unknown')}")
    lines.append(f"- **Temperature:** {metadata.config.get('generation', {}).get('temperature', 'unknown')}")
    lines.append(f"- **Top-p:** {metadata.config.get('generation', {}).get('top_p', 'unknown')}")
    lines.append(f"- **Top-k:** {metadata.config.get('generation', {}).get('top_k', 'unknown')}")
    lines.append(f"- **Seed:** {metadata.config.get('generation', {}).get('seed', 'unknown')}")
    lines.append(f"- **Seed Mode:** {metadata.config.get('seed_mode', 'unknown')}")
    lines.append("")
    
    # Overall Behavior
    lines.append("## Overall Behavior Summary")
    lines.append("")
    total = sum(s.total_trials for s in stats.values())
    if total > 0:
        overall = ConditionStats(
            condition="overall",
            total_trials=total,
            direct_answers=sum(s.direct_answers for s in stats.values()),
            hard_refusals=sum(s.hard_refusals for s in stats.values()),
            soft_refusals=sum(s.soft_refusals for s in stats.values()),
            warnings=sum(s.warnings for s in stats.values()),
            lectures=sum(s.lectures for s in stats.values()),
            redirects=sum(s.redirects for s in stats.values()),
            hedging=sum(s.hedging for s in stats.values()),
            persona_adherence=sum(s.persona_adherence for s in stats.values()),
            empty_responses=sum(s.empty_responses for s in stats.values()),
            errors=sum(s.errors for s in stats.values()),
            median_latency_ms=0, p95_latency_ms=0
        )
        lines.append(f"- **Direct Answer Rate:** {format_rate(overall.direct_answer_rate)}")
        lines.append(f"- **Hard Refusal Rate:** {format_rate(overall.hard_refusal_rate)}")
        lines.append(f"- **Soft Refusal Rate:** {format_rate(overall.soft_refusal_rate)}")
        lines.append(f"- **Warning Rate:** {format_rate(overall.warning_rate)}")
        lines.append(f"- **Lecture Rate:** {format_rate(overall.lecture_rate)}")
        lines.append(f"- **Redirect Rate:** {format_rate(overall.redirect_rate)}")
        lines.append(f"- **Hedging Rate:** {format_rate(overall.hedging_rate)}")
        lines.append(f"- **Persona Adherence Rate:** {format_rate(overall.persona_adherence_rate)}")
        lines.append(f"- **Error Rate:** {format_rate(overall.error_rate)}")
    lines.append("")
    
    # Condition Comparison
    lines.append("## Condition Comparison")
    lines.append("")
    lines.append(generate_condition_table(stats))
    lines.append("")
    
    # Statistical Comparisons
    lines.append("### Pairwise Condition Comparisons (Delta = B - A)")
    lines.append("")
    conditions_list = sorted(stats.keys())
    metrics = ['direct_answer_rate', 'hard_refusal_rate', 'soft_refusal_rate', 'lecture_rate', 'warning_rate']
    
    for i, cond_a in enumerate(conditions_list):
        for cond_b in conditions_list[i+1:]:
            lines.append(f"#### {cond_a} vs {cond_b}")
            lines.append("")
            lines.append("| Metric | A Rate | B Rate | Delta (pp) | 95% CI |")
            lines.append("|--------|--------|--------|------------|--------|")
            for metric in metrics:
                comp = compare_conditions(stats[cond_a], stats[cond_b], metric)
                ci_str = f"[{comp.ci_lower*100:.1f}, {comp.ci_upper*100:.1f}]" if comp.ci_lower is not None else "N/A"
                lines.append(f"| {metric} | {comp.rate_a*100:.1f}% | {comp.rate_b*100:.1f}% | "
                             f"{comp.delta*100:+.1f} | {ci_str} |")
            lines.append("")
    
    # Reasoning Comparison
    if len(reasoning_stats) > 1:
        lines.append("## Reasoning Mode Comparison")
        lines.append("")
        lines.append(generate_reasoning_table(reasoning_stats))
        lines.append("")
    
    # Trigger Phrase Analysis
    if any(c in stats for c in ['neutral', 'explicit_anti_refusal', 'style_only']):
        lines.append(generate_trigger_analysis(results))
    
    # Persona Analysis
    if any(r.get('category') == 'persona' for r in results):
        lines.append(generate_persona_analysis(results))
    
    # Stability
    lines.append(generate_stability_analysis(results))
    
    # Failures
    lines.append(generate_failure_analysis(results))
    
    # Observations
    lines.append("## Observations")
    lines.append("")
    lines.append(observations if observations.strip() else "No significant observations.")
    lines.append("")
    
    # Hypotheses
    lines.append("## Hypotheses")
    lines.append("")
    lines.append("*The following are hypotheses generated from observations, not established facts.*")
    lines.append("")
    lines.append(hypotheses if hypotheses.strip() else "No hypotheses generated.")
    lines.append("")
    
    # Disclaimer
    lines.append("---")
    lines.append("")
    lines.append("## Disclaimer")
    lines.append("")
    lines.append("This report presents measurements from controlled experiments on local language models. ")
    lines.append("Classifications are heuristic and based on deterministic pattern matching. ")
    lines.append("Observations describe measured data; hypotheses are speculative and require further validation. ")
    lines.append("Raw experimental data is preserved in `raw.jsonl` and `scored.jsonl` for independent audit.")
    
    report_path = experiment_dir / "REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    return report_path