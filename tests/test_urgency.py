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
    assert route(p_negative=0.1, urgency=0.9) == "analytics"
