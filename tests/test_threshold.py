import numpy as np

from src.threshold import BusinessAssumptions, profit_curve


def make_synthetic(n=10_000, seed=42):
    """Well-separated synthetic scores: negatives around 0.8, positives around 0.2."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)  # 0 = negative, 1 = positive
    p_negative = np.where(
        y_true == 0,
        rng.beta(8, 2, n),  # negatives get high P(negative)
        rng.beta(2, 8, n),  # positives get low P(negative)
    )
    return y_true, p_negative


def test_breakeven_precision_matches_presentation():
    a = BusinessAssumptions()
    assert abs(a.breakeven_precision - 1 / 9) < 1e-9


def test_optimal_threshold_is_below_half():
    """Core teacher-feedback claim: cheap contact ($1) vs $9 value per save
    means the profit-optimal threshold is lower than the default 0.5."""
    y_true, p_negative = make_synthetic()
    result = profit_curve(y_true, p_negative)
    assert result["optimal_threshold"] < 0.5
    assert result["optimal_profit"] >= result["profit_at_05"]


def test_profit_positive_at_optimum():
    y_true, p_negative = make_synthetic()
    result = profit_curve(y_true, p_negative)
    assert result["optimal_profit"] > 0
