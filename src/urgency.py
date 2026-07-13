"""Urgency scoring: is this negative review an emergency or calm feedback?

Teacher feedback: negative does not mean urgent. A constructive
"the search filter could be better" review goes to the product backlog;
"I was charged twice, cancelling NOW!!!" needs a human within minutes.

The scorer is deliberately rule-based and transparent: support teams
must be able to see exactly why a review was escalated.
Score is 0..1; each signal group contributes a weighted amount.
"""

import re
from dataclasses import dataclass

CHURN_PHRASES = [
    "cancel", "cancelling", "canceled", "cancellation",
    "refund", "money back", "never again", "never use",
    "switching to", "switch to", "deleting my account", "delete my account",
    "unsubscribe", "last time", "done with",
]

MONEY_PHRASES = [
    "charged", "double charge", "billing", "billed", "overcharged",
    "payment", "credit card", "took my money", "stole",
]

ANGER_WORDS = [
    "furious", "outraged", "unacceptable", "disgusting", "scam",
    "fraud", "worst", "terrible", "horrible", "awful", "rip off", "ripoff",
]

WEIGHTS = {"churn": 0.45, "money": 0.30, "anger": 0.15, "intensity": 0.10}


@dataclass
class UrgencyResult:
    score: float
    signals: dict


def _contains_any(text: str, phrases: list[str]) -> list[str]:
    return [p for p in phrases if p in text]


def _intensity(raw_text: str) -> float:
    """Shouting signals: exclamation density and ALL-CAPS words."""
    exclamations = raw_text.count("!")
    words = re.findall(r"[A-Za-z]{3,}", raw_text)
    caps_words = [w for w in words if w.isupper()]
    caps_ratio = len(caps_words) / len(words) if words else 0.0
    return min(1.0, exclamations / 3 * 0.5 + caps_ratio * 5 * 0.5)


def urgency_score(text: str) -> UrgencyResult:
    lowered = text.lower()

    churn_hits = _contains_any(lowered, CHURN_PHRASES)
    money_hits = _contains_any(lowered, MONEY_PHRASES)
    anger_hits = _contains_any(lowered, ANGER_WORDS)
    intensity = _intensity(text)

    score = (
        WEIGHTS["churn"] * min(1.0, len(churn_hits))
        + WEIGHTS["money"] * min(1.0, len(money_hits))
        + WEIGHTS["anger"] * min(1.0, len(anger_hits) / 2)
        + WEIGHTS["intensity"] * intensity
    )

    return UrgencyResult(
        score=round(min(1.0, score), 3),
        signals={
            "churn_phrases": churn_hits,
            "money_phrases": money_hits,
            "anger_words": anger_hits,
            "intensity": round(intensity, 3),
        },
    )


def route(p_negative: float, urgency: float, negative_threshold: float = 0.5, urgency_threshold: float = 0.4) -> str:
    """Routing matrix — who actually needs intervention.

    negative + urgent -> support team now
    negative + calm   -> product feedback backlog
    positive          -> analytics only
    """
    if p_negative < negative_threshold:
        return "analytics"
    if urgency >= urgency_threshold:
        return "support_urgent"
    return "feedback_backlog"
