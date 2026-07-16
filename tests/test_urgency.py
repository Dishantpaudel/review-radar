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

    The sentiment model is trained on movie reviews, so it scores billing
    language ("Charged twice, cancelling today!!!") at only ~0.32. The original
    routing checked sentiment first, so that unsure score silently overruled an
    urgency of 0.83 and dropped a leaving customer into analytics.
    """
    assert route(p_negative=0.319, urgency=0.828) == "support_urgent"


def test_churn_word_inside_praise_does_not_page_support():
    """The guard in the other direction: "I'd never cancel, I love it" is not churn."""
    assert route(p_negative=0.05, urgency=0.45) == "analytics"


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
