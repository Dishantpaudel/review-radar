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

# Fixed terms: a single word or phrase that signals churn on its own.
CHURN_PHRASES = [
    "cancel", "cancelling", "canceled", "cancellation",
    "refund", "money back", "never again", "never use",
    "switching to", "switch to", "deleting my account", "delete my account",
    "unsubscribe", "last time", "done with",
]

# Compositional churn intent: the customer says they are leaving without ever
# using a keyword above ("I dont want my subscription"). A flat word list can't
# express this — "dont want" alone would fire on "I dont want to spoil the
# ending" — so each pattern is anchored to a subscription/account noun.
CHURN_PATTERNS = [
    r"(?:dont|do not|no longer|never) want (?:my|this|the|your) "
    r"(?:subscription|membership|account|service|plan)",
    r"(?:stop|end|drop|ditch|get rid of|dont renew|do not renew) (?:my|the|this|your) "
    r"(?:subscription|membership|account|service|plan)",
    r"(?:not|never|wont|will not|not going to|dont want to) renew",
    r"(?:close|deactivate) (?:my|the) account",
    # "I want out." is churn; "I want out of my seat" is a film review. Only
    # count it as a bare declaration or when it names the subscription.
    r"want out(?:\s*[.!,;]|$|\s+of (?:my|this|the) "
    r"(?:subscription|membership|account|service|plan|contract))",
    r"take my business elsewhere",
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


def _normalize(text: str) -> str:
    """Lowercase and strip apostrophes so "don't" and "dont" match one entry."""
    return text.lower().replace("’", "").replace("'", "")


def _contains_any(text: str, phrases: list[str]) -> list[str]:
    return [p for p in phrases if p in text]


def _match_patterns(text: str, patterns: list[str]) -> list[str]:
    """Return the literal text each churn pattern matched, for the audit trail."""
    hits = []
    for pattern in patterns:
        found = re.search(pattern, text)
        if found:
            hits.append(found.group(0))
    return hits


def _intensity(raw_text: str) -> float:
    """Shouting signals: exclamation density and ALL-CAPS words."""
    exclamations = raw_text.count("!")
    words = re.findall(r"[A-Za-z]{3,}", raw_text)
    caps_words = [w for w in words if w.isupper()]
    caps_ratio = len(caps_words) / len(words) if words else 0.0
    return min(1.0, exclamations / 3 * 0.5 + caps_ratio * 5 * 0.5)


def urgency_score(text: str) -> UrgencyResult:
    lowered = _normalize(text)

    churn_hits = _contains_any(lowered, CHURN_PHRASES) + _match_patterns(lowered, CHURN_PATTERNS)
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


def route(
    p_negative: float,
    urgency: float,
    negative_threshold: float = 0.5,
    urgency_threshold: float = 0.4,
    clearly_positive: float = 0.2,
) -> str:
    """Routing matrix — who actually needs intervention.

    urgent (and not clearly positive) -> support team now
    negative + calm                   -> product feedback backlog
    everything else                   -> analytics only

    Order matters here. Explicit churn intent ("refund", "cancelling today") is
    a support event in its own right, so urgency escalates on its own rather
    than being gated behind the sentiment score.

    The reason is a real weakness of this system: the sentiment model is trained
    on IMDB *movie* reviews, so it is out of domain on billing language and
    scores "Charged twice, cancelling today!!!" at only 0.319. Checking
    sentiment first let that unreliable score veto the urgency engine — which is
    purpose-built for exactly this language — and dropped a churning customer
    into analytics. An unreliable component must not overrule a reliable one.

    The `clearly_positive` floor is the guard in the other direction: it stops a
    churn keyword inside praise ("I love this, I'd never cancel") from paging a
    support agent.
    """
    if urgency >= urgency_threshold and p_negative >= clearly_positive:
        return "support_urgent"
    if p_negative >= negative_threshold:
        return "feedback_backlog"
    return "analytics"
