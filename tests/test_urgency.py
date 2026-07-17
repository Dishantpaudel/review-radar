import pytest

from src.urgency import route, urgency_score


def test_angry_churn_review_is_urgent():
    result = urgency_score("I was charged twice and support ignored me. CANCELLING my account NOW!!!")
    assert result.score >= 0.4
    assert "cancelling" in result.signals["churn_phrases"]
    assert "charged" in result.signals["money_phrases"]


def test_calm_feedback_is_not_urgent():
    result = urgency_score("The plot was a bit slow in the middle and the ending felt rushed.")
    assert result.score < 0.4


def test_positive_review_scores_low():
    result = urgency_score("Wonderful film, the cast was brilliant and I enjoyed every minute.")
    assert result.score < 0.4


def test_routing_matrix():
    assert route(p_negative=0.9, urgency=0.8) == "support_urgent"
    assert route(p_negative=0.9, urgency=0.1) == "feedback_backlog"
    assert route(p_negative=0.1, urgency=0.05) == "analytics"


def test_urgency_is_not_vetoed_by_an_unsure_sentiment_score():
    """Regression: a churning customer must never land in analytics.

    The sentiment model is trained on movie reviews, so it is out of domain on
    billing language and scores it low. The original routing checked sentiment
    first, so that unsure score silently overruled a high urgency and dropped a
    leaving customer into analytics.
    """
    assert route(p_negative=0.238, urgency=0.828) == "support_urgent"


def test_flagship_churn_still_escalates_end_to_end():
    """The same defect, asserted against the real model instead of a constant.

    The unit test above pins route(); it cannot notice if the *model* moves.
    That matters more than it looks: this review's p_negative is only 0.238 and
    the clearly_positive floor is 0.20, so the whole escalation survives on
    0.038 of margin. Retraining moved this number once already (0.319 -> 0.238
    when character n-grams were added). If a future retrain pushes it under
    0.20, the floor silently vetoes the flagship case -- exactly the bug this
    project is named for -- and every unit test here would still pass.
    """
    from src.predict import score_review

    result = score_review("Charged twice and nobody replied. I want a REFUND, cancelling today!!!")
    assert result["route"] == "support_urgent"
    assert result["p_negative"] < 0.5, "still out of domain: sentiment alone would not flag this"
    assert result["urgency"] >= 0.4, "urgency is what escalates it"


@pytest.mark.parametrize(
    "text",
    [
        "I love this, Id never cancel.",
        "I love this service, I would never cancel.",
        "Best subscription I have ever had, I will never cancel it.",
        "Amazing film. I would never ask for a refund.",
        "Fantastic, I would never cancle this.",  # negation must survive a typo
    ],
)
def test_churn_word_inside_praise_is_not_churn(text):
    """Regression: praise containing a churn verb must not read as churn.

    The previous version of this test asserted route(p_negative=0.05,
    urgency=0.45) == "analytics" -- a hand-picked score the model never actually
    returns for these sentences. It passed while the real system failed: the
    model scores this praise at 0.15-0.36, and everything from 0.20 up cleared
    the clearly_positive floor and paged a support agent.

    The floor could not be raised to cover it either, because the flagship
    billing case scores 0.319 -- lifting the floor above these would have
    switched off the escalation this project exists to make.

    The fix is negation scope, so this asserts on the urgency engine directly:
    "never cancel" must not produce a churn signal in the first place.
    """
    assert not urgency_score(text).signals["churn_phrases"], f"false churn in: {text}"


def test_negation_does_not_leak_across_sentences():
    """"never" belongs to its own sentence. Two thoughts, not one."""
    result = urgency_score("I never watch films like this. I want a refund.")
    assert result.signals["churn_phrases"] == ["refund"]


def test_negation_does_not_leak_across_a_comma():
    """A negator that belongs to another clause must not suppress churn.

    "i dont like it, want to cancle" -- the "dont" attaches to "like", and
    "cancle" (a churn typo) sits within the raw 5-token window. Only the comma
    tells them apart. Suppressing here would drop a churning customer to avoid a
    wasted support minute, and this project values those the other way round.
    """
    result = urgency_score("i dont like it, want to cancle my subscribtion")
    assert result.signals["churn_phrases"], "comma-separated churn must survive"


def test_negated_churn_followed_by_real_churn_still_escalates():
    """One negated mention does not immunise the rest of the review."""
    result = urgency_score("I would never cancel, but now I want to cancel.")
    assert result.signals["churn_phrases"], "second clause is real churn"


# Churn intent expressed without any keyword from CHURN_PHRASES. These were all
# missed by the original flat word list and are the reason CHURN_PATTERNS exists.
@pytest.mark.parametrize(
    "text",
    [
        "Its horrible, I dont want my subscription",
        "Its horrible, I don't want my subscription",
        "Terrible service. I no longer want my membership.",
        "Awful. Stop my subscription immediately.",
        "Disgusting quality, I am not renewing.",
        "Worst app ever, close my account.",
        "This is awful, I want out.",
    ],
)
def test_implicit_churn_reaches_support(text):
    result = urgency_score(text)
    assert result.signals["churn_phrases"], f"no churn signal detected in: {text}"
    assert route(p_negative=0.9, urgency=result.score) == "support_urgent"


# The mirror image: churn-ish wording that is really just film criticism.
# "I dont want to spoil the ending" must never page a support agent.
@pytest.mark.parametrize(
    "text",
    [
        "I dont want to spoil the ending, but it was boring.",
        "You dont want to miss this film! Wonderful.",
        "The plot was terrible and I dont want to see it again.",
        "I want out of my seat, the pacing dragged.",
    ],
)
def test_no_false_churn_on_movie_talk(text):
    result = urgency_score(text)
    assert not result.signals["churn_phrases"], f"false churn signal in: {text}"
