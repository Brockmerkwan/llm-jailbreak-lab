"""
Statistical analysis for experiment results.
"""

import json
import csv
import random
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
    # Response shape metrics
    avg_response_chars: float = 0.0
    avg_response_words: float = 0.0
    avg_response_lines: float = 0.0
    avg_response_paragraphs: float = 0.0
    avg_lecture_ratio: float = 0.0
    
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
    
    @property
    def relative_risk_direct_answer(self, other: 'ConditionStats') -> Optional[float]:
        """Relative risk of direct answer vs another condition."""
        if self.direct_answer_rate == 0 or other.direct_answer_rate == 0:
            return None
        return other.direct_answer_rate / self.direct_answer_rate
    
    @property
    def relative_risk_hard_refusal(self, other: 'ConditionStats') -> Optional[float]:
        """Relative risk of hard refusal vs another condition."""
        if self.hard_refusal_rate == 0 or other.hard_refusal_rate == 0:
            return None
        return other.hard_refusal_rate / self.hard_refusal_rate


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
    relative_risk: Optional[float] = None
    risk_direction: Optional[str] = None  # "increased" or "decreased"


def bootstrap_confidence_interval(
    data_a: List[int], data_b: List[int], 
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """Bootstrap confidence interval for difference in proportions.
    
    Args:
        data_a: Binary outcomes for condition A (1 = success, 0 = failure)
        data_b: Binary outcomes for condition B
        n_resamples: Number of bootstrap resamples
        confidence: Confidence level
        seed: Random seed for reproducibility
    
    Returns:
        (lower, upper) confidence interval for difference (B - A)
    """
    if not data_a or not data_b:
        return (0.0, 0.0)
    
    random.seed(seed)
    n_a = len(data_a)
    n_b = len(data_b)
    
    observed_diff = sum(data_b) / n_b - sum(data_a) / n_a
    
    bootstrap_diffs = []
    for _ in range(n_resamples):
        sample_a = [random.choice(data_a) for _ in range(n_a)]
        sample_b = [random.choice(data_b) for _ in range(n_b)]
        diff = sum(sample_b) / n_b - sum(sample_a) / n_a
        bootstrap_diffs.append(diff)
    
    alpha = 1 - confidence
    lower_idx = int(n_resamples * alpha / 2)
    upper_idx = int(n_resamples * (1 - alpha / 2))
    
    sorted_diffs = sorted(bootstrap_diffs)
    return (sorted_diffs[lower_idx], sorted_diffs[upper_idx])


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
    confidence: float = 0.95,
    use_bootstrap: bool = False,
    bootstrap_resamples: int = 5000
) -> Tuple[float, float]:
    """Confidence interval for difference in proportions (normal approximation).
    
    If use_bootstrap=True, uses bootstrap resampling instead.
    """
    if total_a == 0 or total_b == 0:
        return (0.0, 0.0)
    
    if use_bootstrap:
        # Create binary outcome arrays for bootstrap
        data_a = [1] * succ_a + [0] * (total_a - succ_a)
        data_b = [1] * succ_b + [0] * (total_b - succ_b)
        return bootstrap_confidence_interval(data_a, data_b, n_resamples=bootstrap_resamples, confidence=confidence)
    
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
    
    # Response shape metrics
    response_chars = [r.get('response_chars', 0) for r in cond_results]
    response_words = [r.get('response_words', 0) for r in cond_results]
    response_lines = [r.get('response_lines', 0) for r in cond_results]
    response_paragraphs = [r.get('response_paragraphs', 0) for r in cond_results]
    lecture_ratios = [r.get('lecture_ratio', 0.0) for r in cond_results]
    
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
        avg_response_chars=statistics.mean(response_chars) if response_chars else 0,
        avg_response_words=statistics.mean(response_words) if response_words else 0,
        avg_response_lines=statistics.mean(response_lines) if response_lines else 0,
        avg_response_paragraphs=statistics.mean(response_paragraphs) if response_paragraphs else 0,
        avg_lecture_ratio=statistics.mean(lecture_ratios) if lecture_ratios else 0,
    )
    return stats


def compute_all_condition_stats(results: List[Dict[str, Any]]) -> Dict[str, ConditionStats]:
    """Compute statistics for all conditions present in results."""
    conditions = set(r.get('condition') for r in results)
    return {cond: compute_condition_stats(results, cond) for cond in conditions}


def compare_conditions(
    stats_a: ConditionStats, 
    stats_b: ConditionStats, 
    metric: str,
    use_bootstrap: bool = False,
    bootstrap_resamples: int = 5000
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
    
    ci_lower, ci_upper = proportion_diff_confidence_interval(
        count_a, total_a, count_b, total_b, 
        use_bootstrap=use_bootstrap, 
        bootstrap_resamples=bootstrap_resamples
    )
    
    # Calculate relative risk
    relative_risk = None
    risk_direction = None
    if rate_a > 0 and rate_b > 0:
        relative_risk = rate_b / rate_a
        risk_direction = "increased" if relative_risk > 1 else "decreased"
    elif rate_a == 0 and rate_b > 0:
        relative_risk = float('inf')
        risk_direction = "increased (from zero)"
    elif rate_a > 0 and rate_b == 0:
        relative_risk = 0.0
        risk_direction = "decreased (to zero)"
    
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
        relative_risk=relative_risk,
        risk_direction=risk_direction,
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
            'median_latency_ms', 'p95_latency_ms',
            'avg_response_chars', 'avg_response_words', 'avg_response_lines', 
            'avg_response_paragraphs', 'avg_lecture_ratio'
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
                round(s.avg_response_chars, 1), round(s.avg_response_words, 1), 
                round(s.avg_response_lines, 1), round(s.avg_response_paragraphs, 1),
                round(s.avg_lecture_ratio, 4)
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