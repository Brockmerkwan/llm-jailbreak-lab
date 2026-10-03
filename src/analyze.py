"""
Statistical analysis for experiment results.
"""

import json
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import statistics
import math


@dataclass
class ConditionStats:
    condition: str
    total_trials: int
    direct_answers: int
    hard_refusals: int
    soft_refusals: int
    warnings: int
    lectures: int
    redirects: int
    hedging: int
    persona_adherence: int
    empty_responses: int
    errors: int
    median_latency_ms: float
    p95_latency_ms: float
    
    @property
    def direct_answer_rate(self) -> float:
        return self.direct_answers / self.total_trials if self.total_trials > 0 else 0.0
    
    @property
    def hard_refusal_rate(self) -> float:
        return self.hard_refusals / self.total_trials if self.total_trials > 0 else 0.0
    
    @property
    def soft_refusal_rate(self) -> float:
        return self.soft_refusals / self.total_trials if self.total_trials > 0 else 0.0
    
    @property
    def warning_rate(self) -> float:
        return self.warnings / self.total_trials if self.total_trials > 0 else 0.0
    
    @property
    def lecture_rate(self) -> float:
        return self.lectures / self.total_trials if self.total_trials > 0 else 0.0
    
    @property
    def redirect_rate(self) -> float:
        return self.redirects / self.total_trials if self.total_trials > 0 else 0.0
    
    @property
    def hedging_rate(self) -> float:
        return self.hedging / self.total_trials if self.total_trials > 0 else 0.0
    
    @property
    def persona_adherence_rate(self) -> float:
        return self.persona_adherence / self.total_trials if self.total_trials > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        return self.errors / self.total_trials if self.total_trials > 0 else 0.0


@dataclass
class ComparisonResult:
    condition_a: str
    condition_b: str
    metric: str
    rate_a: float
    rate_b: float
    delta: float
    count_a: int
    count_b: int
    total_a: int
    total_b: int
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for binomial proportion."""
    if trials == 0:
        return (0.0, 1.0)
    
    z = 1.96  # 95% confidence
    if confidence == 0.99:
        z = 2.576
    elif confidence == 0.90:
        z = 1.645
    
    p = successes / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    adjustment = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    
    return (max(0, center - adjustment), min(1, center + adjustment))


def proportion_diff_confidence_interval(
    succ_a: int, total_a: int, 
    succ_b: int, total_b: int, 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Confidence interval for difference in proportions (normal approximation)."""
    if total_a == 0 or total_b == 0:
        return (0.0, 0.0)
    
    z = 1.96
    if confidence == 0.99:
        z = 2.576
    elif confidence == 0.90:
        z = 1.645
    
    p_a = succ_a / total_a
    p_b = succ_b / total_b
    diff = p_b - p_a
    
    se = math.sqrt(p_a * (1 - p_a) / total_a + p_b * (1 - p_b) / total_b)
    margin = z * se
    
    return (diff - margin, diff + margin)


def load_results(experiment_dir: Path) -> List[Dict[str, Any]]:
    """Load scored results from JSONL file."""
    scored_path = experiment_dir / "scored.jsonl"
    if not scored_path.exists():
        return []
    
    results = []
    with open(scored_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_condition_stats(results: List[Dict[str, Any]], condition: str) -> ConditionStats:
    """Compute statistics for a single condition."""
    cond_results = [r for r in results if r.get('condition') == condition]
    
    if not cond_results:
        return ConditionStats(
            condition=condition,
            total_trials=0,
            direct_answers=0, hard_refusals=0, soft_refusals=0,
            warnings=0, lectures=0, redirects=0, hedging=0,
            persona_adherence=0, empty_responses=0, errors=0,
            median_latency_ms=0, p95_latency_ms=0
        )
    
    latencies = [r.get('latency_ms', 0) for r in cond_results if r.get('latency_ms')]
    
    stats = ConditionStats(
        condition=condition,
        total_trials=len(cond_results),
        direct_answers=sum(1 for r in cond_results if 'direct_answer' in r.get('labels', [])),
        hard_refusals=sum(1 for r in cond_results if 'hard_refusal' in r.get('labels', [])),
        soft_refusals=sum(1 for r in cond_results if 'soft_refusal' in r.get('labels', [])),
        warnings=sum(1 for r in cond_results if 'warning' in r.get('labels', [])),
        lectures=sum(1 for r in cond_results if 'lecture' in r.get('labels', [])),
        redirects=sum(1 for r in cond_results if 'redirect' in r.get('labels', [])),
        hedging=sum(1 for r in cond_results if 'hedging' in r.get('labels', [])),
        persona_adherence=sum(1 for r in cond_results if 'persona_adherence' in r.get('labels', [])),
        empty_responses=sum(1 for r in cond_results if 'empty_response' in r.get('labels', [])),
        errors=sum(1 for r in cond_results if 'generation_error' in r.get('labels', [])),
        median_latency_ms=statistics.median(latencies) if latencies else 0,
        p95_latency_ms=sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0,
    )
    return stats


def compute_all_condition_stats(results: List[Dict[str, Any]]) -> Dict[str, ConditionStats]:
    """Compute statistics for all conditions present in results."""
    conditions = set(r.get('condition') for r in results)
    return {cond: compute_condition_stats(results, cond) for cond in conditions}


def compare_conditions(
    stats_a: ConditionStats, 
    stats_b: ConditionStats, 
    metric: str
) -> ComparisonResult:
    """Compare two conditions on a specific metric."""
    metric_map = {
        'direct_answer_rate': ('direct_answers', 'direct_answer_rate'),
        'hard_refusal_rate': ('hard_refusals', 'hard_refusal_rate'),
        'soft_refusal_rate': ('soft_refusals', 'soft_refusal_rate'),
        'warning_rate': ('warnings', 'warning_rate'),
        'lecture_rate': ('lectures', 'lecture_rate'),
        'redirect_rate': ('redirects', 'redirect_rate'),
        'hedging_rate': ('hedging', 'hedging_rate'),
        'persona_adherence_rate': ('persona_adherence', 'persona_adherence_rate'),
    }
    
    count_attr, rate_attr = metric_map.get(metric, ('direct_answers', 'direct_answer_rate'))
    
    count_a = getattr(stats_a, count_attr)
    count_b = getattr(stats_b, count_attr)
    total_a = stats_a.total_trials
    total_b = stats_b.total_trials
    rate_a = getattr(stats_a, rate_attr)
    rate_b = getattr(stats_b, rate_attr)
    
    ci_lower, ci_upper = proportion_diff_confidence_interval(count_a, total_a, count_b, total_b)
    
    return ComparisonResult(
        condition_a=stats_a.condition,
        condition_b=stats_b.condition,
        metric=metric,
        rate_a=rate_a,
        rate_b=rate_b,
        delta=rate_b - rate_a,
        count_a=count_a,
        count_b=count_b,
        total_a=total_a,
        total_b=total_b,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def analyze_by_case_and_condition(results: List[Dict[str, Any]]) -> Dict[Tuple[str, str], ConditionStats]:
    """Analyze results grouped by case_id and condition."""
    grouped = defaultdict(list)
    for r in results:
        key = (r.get('case_id', ''), r.get('condition', ''))
        grouped[key].append(r)
    
    stats = {}
    for (case_id, condition), case_results in grouped.items():
        latencies = [r.get('latency_ms', 0) for r in case_results if r.get('latency_ms')]
        stats[(case_id, condition)] = ConditionStats(
            condition=f"{case_id}|{condition}",
            total_trials=len(case_results),
            direct_answers=sum(1 for r in case_results if 'direct_answer' in r.get('labels', [])),
            hard_refusals=sum(1 for r in case_results if 'hard_refusal' in r.get('labels', [])),
            soft_refusals=sum(1 for r in case_results if 'soft_refusal' in r.get('labels', [])),
            warnings=sum(1 for r in case_results if 'warning' in r.get('labels', [])),
            lectures=sum(1 for r in case_results if 'lecture' in r.get('labels', [])),
            redirects=sum(1 for r in case_results if 'redirect' in r.get('labels', [])),
            hedging=sum(1 for r in case_results if 'hedging' in r.get('labels', [])),
            persona_adherence=sum(1 for r in case_results if 'persona_adherence' in r.get('labels', [])),
            empty_responses=sum(1 for r in case_results if 'empty_response' in r.get('labels', [])),
            errors=sum(1 for r in case_results if 'generation_error' in r.get('labels', [])),
            median_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0,
        )
    return stats


def analyze_by_reasoning_mode(results: List[Dict[str, Any]]) -> Dict[str, ConditionStats]:
    """Analyze results grouped by reasoning mode."""
    grouped = defaultdict(list)
    for r in results:
        mode = r.get('reasoning_mode_confirmed', 'unknown')
        grouped[mode].append(r)
    
    stats = {}
    for mode, mode_results in grouped.items():
        latencies = [r.get('latency_ms', 0) for r in mode_results if r.get('latency_ms')]
        stats[mode] = ConditionStats(
            condition=f"reasoning:{mode}",
            total_trials=len(mode_results),
            direct_answers=sum(1 for r in mode_results if 'direct_answer' in r.get('labels', [])),
            hard_refusals=sum(1 for r in mode_results if 'hard_refusal' in r.get('labels', [])),
            soft_refusals=sum(1 for r in mode_results if 'soft_refusal' in r.get('labels', [])),
            warnings=sum(1 for r in mode_results if 'warning' in r.get('labels', [])),
            lectures=sum(1 for r in mode_results if 'lecture' in r.get('labels', [])),
            redirects=sum(1 for r in mode_results if 'redirect' in r.get('labels', [])),
            hedging=sum(1 for r in mode_results if 'hedging' in r.get('labels', [])),
            persona_adherence=sum(1 for r in mode_results if 'persona_adherence' in r.get('labels', [])),
            empty_responses=sum(1 for r in mode_results if 'empty_response' in r.get('labels', [])),
            errors=sum(1 for r in mode_results if 'generation_error' in r.get('labels', [])),
            median_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0,
        )
    return stats


def analyze_trigger_phrases(results: List[Dict[str, Any]]) -> Dict[str, ConditionStats]:
    """Analyze results for trigger phrase experiment (explicit_anti_refusal vs neutral)."""
    # This is a subset of the condition analysis focused on framing effects
    return {cond: stats for cond, stats in compute_all_condition_stats(results).items() 
            if cond in ('neutral', 'explicit_anti_refusal', 'style_only', 'baseline')}


def generate_summary_csv(experiment_dir: Path, stats: Dict[str, ConditionStats]) -> Path:
    """Generate summary.csv with flattened statistics."""
    csv_path = experiment_dir / "summary.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'condition', 'total_trials',
            'direct_answers', 'hard_refusals', 'soft_refusals', 'warnings', 'lectures',
            'redirects', 'hedging', 'persona_adherence', 'empty_responses', 'errors',
            'direct_answer_rate', 'hard_refusal_rate', 'soft_refusal_rate',
            'warning_rate', 'lecture_rate', 'redirect_rate', 'hedging_rate',
            'persona_adherence_rate', 'error_rate',
            'median_latency_ms', 'p95_latency_ms'
        ])
        
        for condition, s in sorted(stats.items()):
            writer.writerow([
                s.condition, s.total_trials,
                s.direct_answers, s.hard_refusals, s.soft_refusals, s.warnings, s.lectures,
                s.redirects, s.hedging, s.persona_adherence, s.empty_responses, s.errors,
                round(s.direct_answer_rate, 4), round(s.hard_refusal_rate, 4), round(s.soft_refusal_rate, 4),
                round(s.warning_rate, 4), round(s.lecture_rate, 4), round(s.redirect_rate, 4),
                round(s.hedging_rate, 4), round(s.persona_adherence_rate, 4), round(s.error_rate, 4),
                round(s.median_latency_ms, 1), round(s.p95_latency_ms, 1),
            ])
    
    return csv_path


def generate_detailed_csv(experiment_dir: Path, case_condition_stats: Dict[Tuple[str, str], ConditionStats]) -> Path:
    """Generate detailed CSV with per-case breakdown."""
    csv_path = experiment_dir / "summary_detailed.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'case_id', 'condition', 'category', 'total_trials',
            'direct_answers', 'hard_refusals', 'soft_refusals', 'warnings', 'lectures',
            'redirects', 'hedging', 'persona_adherence', 'empty_responses', 'errors',
            'direct_answer_rate', 'hard_refusal_rate', 'soft_refusal_rate',
            'warning_rate', 'lecture_rate', 'redirect_rate', 'hedging_rate',
            'persona_adherence_rate', 'error_rate',
            'median_latency_ms', 'p95_latency_ms'
        ])
        
        # Load category info from raw results
        raw_path = experiment_dir / "raw.jsonl"
        case_categories = {}
        if raw_path.exists():
            with open(raw_path, 'r', encoding='utf-8') as rf:
                for line in rf:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        case_categories[r.get('case_id')] = r.get('category', '')
        
        for (case_id, condition), s in sorted(case_condition_stats.items()):
            writer.writerow([
                case_id, condition, case_categories.get(case_id, ''), s.total_trials,
                s.direct_answers, s.hard_refusals, s.soft_refusals, s.warnings, s.lectures,
                s.redirects, s.hedging, s.persona_adherence, s.empty_responses, s.errors,
                round(s.direct_answer_rate, 4), round(s.hard_refusal_rate, 4), round(s.soft_refusal_rate, 4),
                round(s.warning_rate, 4), round(s.lecture_rate, 4), round(s.redirect_rate, 4),
                round(s.hedging_rate, 4), round(s.persona_adherence_rate, 4), round(s.error_rate, 4),
                round(s.median_latency_ms, 1), round(s.p95_latency_ms, 1),
            ])
    
    return csv_path