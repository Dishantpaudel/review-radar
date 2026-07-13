"""Business-optimal decision threshold.

Teacher feedback: don't pick the threshold that maximizes F1 —
pick the one that maximizes money. Contacting a flagged customer
costs C = $1; saving one is worth r * ARPU * M = $9. Because the
value/cost ratio is 9:1, it pays to contact far more people than
an F1-optimal threshold would suggest.

Profit per month at a given threshold:
    profit = saved_customers * (r * ARPU * M) - contacted * C
where saved_customers = true_negatives_caught * p_at_risk * r.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BusinessAssumptions:
    """Numbers from the project's business case (per month)."""

    reviews_per_month: int = 300_000
    p_at_risk: float = 0.30       # share of angry reviewers who would actually leave
    win_back_rate: float = 0.15   # share of contacted at-risk customers we keep
    arpu: float = 10.0            # revenue per user per month, $
    retained_months: int = 6
    contact_cost: float = 1.0     # cost of one outreach, $

    @property
    def value_per_save(self) -> float:
        return self.arpu * self.retained_months  # $60

    @property
    def breakeven_precision(self) -> float:
        """Minimum precision where outreach still profits: C / (r * ARPU * M)."""
        return self.contact_cost / (self.win_back_rate * self.value_per_save)


def profit_curve(
    y_true: np.ndarray,
    p_negative: np.ndarray,
    assumptions: BusinessAssumptions | None = None,
    thresholds: np.ndarray | None = None,
) -> dict:
    """Sweep thresholds and compute expected monthly profit at each.

    y_true: 1 = positive, 0 = negative. Test-set rates are scaled up
    to the monthly review volume from the business assumptions.
    """
    a = assumptions or BusinessAssumptions()
    thresholds = thresholds if thresholds is not None else np.linspace(0.01, 0.99, 99)

    y_negative = (y_true == 0).astype(int)
    n = len(y_true)
    scale = a.reviews_per_month / n  # test set -> monthly volume

    profits, precisions, contacted_counts = [], [], []
    for t in thresholds:
        flagged = p_negative >= t
        contacted = flagged.sum() * scale
        true_negatives_caught = (flagged & (y_negative == 1)).sum() * scale

        saved = true_negatives_caught * a.p_at_risk * a.win_back_rate
        profit = saved * a.value_per_save - contacted * a.contact_cost

        profits.append(profit)
        precisions.append(y_negative[flagged].mean() if flagged.any() else 0.0)
        contacted_counts.append(contacted)

    profits = np.array(profits)
    best = int(np.argmax(profits))
    return {
        "thresholds": thresholds,
        "profits": profits,
        "precisions": np.array(precisions),
        "contacted": np.array(contacted_counts),
        "optimal_threshold": float(thresholds[best]),
        "optimal_profit": float(profits[best]),
        "profit_at_05": float(profits[np.argmin(np.abs(thresholds - 0.5))]),
        "breakeven_precision": a.breakeven_precision,
    }
