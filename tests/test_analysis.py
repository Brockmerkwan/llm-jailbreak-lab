"""
Unit tests for statistical analysis.
"""

import pytest
from src.analyze import (
    ConditionStats, compare_conditions,
    wilson_score_interval, proportion_diff_confidence_interval,
    compute_all_condition_stats
)


class TestWilsonScoreInterval:
    """Test Wilson score interval calculation."""
    
    def test_basic(self):
        """Test basic Wilson interval."""
        lower, upper = wilson_score_interval(5, 10)
        assert 0 <= lower <= 0.5 <= upper <= 1
    
    def test_zero_successes(self):
        """Test with zero successes."""
        lower, upper = wilson_score_interval(0, 10)
        assert lower == 0.0
        assert upper > 0.0
    
    def test_all_successes(self):
        """Test with all successes."""
        lower, upper = wilson_score_interval(10, 10)
        assert lower < 1.0
        assert upper == 1.0
    
    def test_zero_trials(self):
        """Test with zero trials."""
        lower, upper = wilson_score_interval(0, 0)
        assert lower == 0.0
        assert upper == 1.0


class TestProportionDiffCI:
    """Test confidence interval for difference in proportions."""
    
    def test_basic(self):
        """Test basic difference CI."""
        lower, upper = proportion_diff_confidence_interval(5, 10, 8, 10)
        # Difference is 0.3, CI should contain it
        assert lower < 0.3 < upper
    
    def test_zero_trials(self):
        """Test with zero trials in one group."""
        lower, upper = proportion_diff_confidence_interval(5, 10, 0, 0)
        assert lower == 0.0
        assert upper == 0.0


class TestConditionStats:
    """Test ConditionStats computed properties."""
    
    def test_rates(self):
        """Test rate calculations."""
        stats = ConditionStats(
            condition="test",
            total_trials=10,
            direct_answers=7,
            hard_refusals=2,
            soft_refusals=1,
            warnings=0,
            lectures=0,
            redirects=0,
            hedging=0,
            persona_adherence=0,
            empty_responses=0,
            errors=0,
            median_latency_ms=100,
            p95_latency_ms=200,
        )
        
        assert stats.direct_answer_rate == 0.7
        assert stats.hard_refusal_rate == 0.2
        assert stats.soft_refusal_rate == 0.1
    
    def test_zero_trials(self):
        """Test rates with zero trials."""
        stats = ConditionStats(
            condition="test",
            total_trials=0,
            direct_answers=0,
            hard_refusals=0,
            soft_refusals=0,
            warnings=0,
            lectures=0,
            redirects=0,
            hedging=0,
            persona_adherence=0,
            empty_responses=0,
            errors=0,
            median_latency_ms=0,
            p95_latency_ms=0,
        )
        
        assert stats.direct_answer_rate == 0.0
        assert stats.hard_refusal_rate == 0.0


class TestCompareConditions:
    """Test condition comparison."""
    
    def test_compare_direct_answer(self):
        """Test direct answer rate comparison."""
        stats_a = ConditionStats(
            condition="A", total_trials=10,
            direct_answers=5, hard_refusals=3, soft_refusals=2,
            warnings=0, lectures=0, redirects=0, hedging=0,
            persona_adherence=0, empty_responses=0, errors=0,
            median_latency_ms=100, p95_latency_ms=200,
        )
        stats_b = ConditionStats(
            condition="B", total_trials=10,
            direct_answers=8, hard_refusals=1, soft_refusals=1,
            warnings=0, lectures=0, redirects=0, hedging=0,
            persona_adherence=0, empty_responses=0, errors=0,
            median_latency_ms=100, p95_latency_ms=200,
        )
        
        comp = compare_conditions(stats_a, stats_b, 'direct_answer_rate')
        assert comp.condition_a == "A"
        assert comp.condition_b == "B"
        assert comp.rate_a == 0.5
        assert comp.rate_b == 0.8
        assert abs(comp.delta - 0.3) < 1e-10
        assert comp.count_a == 5
        assert comp.count_b == 8
        assert comp.total_a == 10
        assert comp.total_b == 10
        assert comp.ci_lower is not None
        assert comp.ci_upper is not None
    
    def test_compare_hard_refusal(self):
        """Test hard refusal rate comparison."""
        stats_a = ConditionStats(
            condition="A", total_trials=25,
            direct_answers=17, hard_refusals=5, soft_refusals=3,
            warnings=0, lectures=0, redirects=0, hedging=0,
            persona_adherence=0, empty_responses=0, errors=0,
            median_latency_ms=100, p95_latency_ms=200,
        )
        stats_b = ConditionStats(
            condition="B", total_trials=25,
            direct_answers=22, hard_refusals=1, soft_refusals=2,
            warnings=0, lectures=0, redirects=0, hedging=0,
            persona_adherence=0, empty_responses=0, errors=0,
            median_latency_ms=100, p95_latency_ms=200,
        )
        
        comp = compare_conditions(stats_a, stats_b, 'hard_refusal_rate')
        assert comp.rate_a == 0.2
        assert comp.rate_b == 0.04
        assert comp.delta == -0.16


class TestComputeAllConditionStats:
    """Test compute_all_condition_stats function."""
    
    def test_basic(self):
        """Test basic stats computation."""
        results = [
            {'condition': 'A', 'labels': ['direct_answer'], 'latency_ms': 100},
            {'condition': 'A', 'labels': ['direct_answer'], 'latency_ms': 120},
            {'condition': 'A', 'labels': ['hard_refusal'], 'latency_ms': 80},
            {'condition': 'B', 'labels': ['direct_answer'], 'latency_ms': 90},
            {'condition': 'B', 'labels': ['soft_refusal'], 'latency_ms': 110},
        ]
        
        stats = compute_all_condition_stats(results)
        
        assert 'A' in stats
        assert 'B' in stats
        assert stats['A'].total_trials == 3
        assert stats['A'].direct_answers == 2
        assert stats['A'].hard_refusals == 1
        assert stats['B'].total_trials == 2
        assert stats['B'].direct_answers == 1
        assert stats['B'].soft_refusals == 1
    
    def test_empty_results(self):
        """Test with empty results."""
        stats = compute_all_condition_stats([])
        assert stats == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])