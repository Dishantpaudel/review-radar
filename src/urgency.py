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

from src.text import is_typo_of, normalize, tokens

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

# Real English words that sit one or two edits from a lexicon term. Mined by
# running the fuzzy matcher over the full IMDB vocabulary (50k reviews) and
# keeping every token that occurs >= 25 times, i.e. every collision a real
# corpus actually produces. The rule they encode is one line:
#
#     if the corpus says it is a word, it is not a typo.
#
# Without this, fuzzy matching is actively dangerous on a *film* site:
# "killed" (1,111 occurrences) is one edit from "billed", so "he killed the
# villain" would raise a billing signal and page a support agent. Likewise
# "changed"/"charged", "style"/"stole", "curious"/"furious", "cancer"/"cancel".
# These words still match through the exact rules below when they genuinely
# appear -- they are barred only from being read as somebody's misspelling.
NEVER_A_TYPO = frozenset([
    "acceptable", "cancelled", "cancer", "changed", "charge", "charges",
    "charmed", "concealed", "curious", "discussing", "disgustingly", "filled",
    "filling", "freud", "horribly", "killed", "killing", "outdated", "outrage",
    "stale", "stolen", "stone", "store", "stowe", "style", "terribly",
    "willed", "willing", "worse",
])

# Churn verbs that a negation flips outright. "I want to cancel" is churn;
# "I would never cancel" is praise wearing the same word. Defect #1 was the
# lexicon being too literal about *meaning*; this is the same literalism
# pointed the other way -- matching the verb while ignoring the "never".
NEGATABLE_CHURN = frozenset([
    "cancel", "cancelling", "canceled", "cancellation", "refund",
    "unsubscribe", "switching to", "switch to",
])
NEGATORS = frozenset(["not", "never", "wont", "cant", "dont", "no", "nor", "neither"])

# A negator only reaches across its own clause. "I would never ask for a refund"
# is five tokens wide, so the window has to be generous -- but a window alone
# would also swallow "I never watch films like this. I want a refund." and
# "i dont like it, want to cancel", where the negation belongs to a different
# thought. Clauses separate them, so scope is clipped to the clause first and the
# window applied inside it.
#
# The comma is deliberately a boundary, and the direction of the error is the
# reason. Not crossing it means "I'd never, ever cancel" (praise) can slip
# through and escalate -- a false *positive*, a wasted support minute. Crossing
# it means "dont like it, want to cancel" gets its churn suppressed -- a false
# *negative*, a dropped customer. This project values a saved customer at ~9x
# the cost of contacting one, so when the two errors trade off, suppression must
# be the timid one: never silence real churn to avoid a wasted minute.
NEGATION_WINDOW = 5
_CLAUSE = re.compile(r"[.!?;,]+")


@dataclass
class UrgencyResult:
    score: float
    signals: dict


def _clauses(text: str) -> list[list[str]]:
    """Normalised text -> one token list per clause (split on . ! ? ; and ,)."""
    return [tokens(c) for c in _CLAUSE.split(text) if c.strip()]


def _negated_everywhere(term: str, clauses: list[list[str]]) -> bool:
    """Is every occurrence of `term` inside the scope of a negator?

    All of them must be negated: "I would never cancel, but now I want to
    cancel" is still churn, and the second clause is what matters.
    """
    head = term.split()[0]
    seen = False
    for toks in clauses:
        for i, t in enumerate(toks):
            if t != head:
                continue
            seen = True
            window = toks[max(0, i - NEGATION_WINDOW):i]
            if not any(w in NEGATORS for w in window):
                return False
    return seen


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving unique. One typo can match several lexicon entries --
    "cancle" is close to both "cancel" and "canceled" -- and the audit trail
    should show the term once, not once per entry it happened to be near."""
    out, seen = [], set()
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _contains_any(text: str, phrases: list[str], toks: list[str] | None = None) -> list[str]:
    """Exact-match phrases, then fall back to fuzzy matching for typos.

    Returns what the *user actually typed*, not the lexicon entry it matched.
    The support team is shown these terms as the reason a review was escalated,
    so reporting "cancel" when the customer wrote "cancle" would make the audit
    trail quietly untrue.
    """
    hits = []
    for p in phrases:
        if p in text:
            hits.append(p)
            continue
        if toks is None or " " in p or len(p) < 5:
            continue  # multi-word and short terms are exact-only
        for tok in toks:
            if tok in NEVER_A_TYPO or tok in phrases:
                continue
            if is_typo_of(tok, p):
                hits.append(tok)  # the misspelling, not the lexicon entry
                break
    return _dedupe(hits)


def _is_negatable(term: str) -> bool:
    """Negation has to survive a typo too: "I would never cancle" is still
    praise, and the hit recorded for it is the misspelling, not the entry."""
    if term in NEGATABLE_CHURN:
        return True
    if " " in term or term in NEVER_A_TYPO:
        return False
    return any(is_typo_of(term, n) for n in NEGATABLE_CHURN if " " not in n)


def _drop_negated_churn(hits: list[str], clauses: list[list[str]]) -> list[str]:
    """Remove churn hits that a negator cancels out."""
    return [
        h for h in hits
        if not _is_negatable(h) or not _negated_everywhere(h, clauses)
    ]


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
    lowered = normalize(text)
    toks = tokens(lowered)
    clauses = _clauses(lowered)

    churn_hits = _contains_any(lowered, CHURN_PHRASES, toks)
    churn_hits = _drop_negated_churn(churn_hits, clauses)
    churn_hits = _dedupe(churn_hits + _match_patterns(lowered, CHURN_PATTERNS))
    money_hits = _contains_any(lowered, MONEY_PHRASES, toks)
    anger_hits = _contains_any(lowered, ANGER_WORDS, toks)
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
