"""Misspelled reviews must still be understood.

Real customers do not spell-check angry reviews. The word-only vectorizer had a
hard failure here: a token it has never seen contributes nothing, so "vad" and
"borst" carried no sentiment at all, and the reviews that matter most were the
ones most likely to be mistyped. These tests pin the behaviour that character
n-grams and fuzzy lexicon matching exist to provide.
"""

import pytest

from src.predict import score_review
from src.text import damerau_levenshtein, is_typo_of, normalize
from src.urgency import NEVER_A_TYPO, urgency_score


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def test_apostrophes_unify():
    """The model and the urgency engine must agree on what a word is."""
    assert normalize("I don't like it") == normalize("I dont like it")


def test_elongation_collapses():
    assert normalize("SOOOO BAAAAD!!!") == "soo baad!!!"


def test_markup_is_stripped():
    """IMDB reviews are full of <br /> and the vectorizer used to learn it."""
    assert "br" not in normalize("Great film.<br /><br />Loved it.")


# --------------------------------------------------------------------------
# edit distance
# --------------------------------------------------------------------------

def test_transposition_is_one_edit():
    """Swapped adjacent letters are the most common typo; plain Levenshtein
    charges 2 for them, which is why this is Damerau."""
    assert damerau_levenshtein("cancle", "cancel") == 1


def test_short_words_are_never_typos():
    """"bad"/"bat"/"mad"/"sad" are one edit apart and all real words."""
    assert not is_typo_of("bat", "bad")


# --------------------------------------------------------------------------
# urgency engine: typos in the lexicon
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("I want to cancle my subscription", "cancle"),
        ("i need a refudn NOW", "refudn"),
        ("unsubscibe me immediately", "unsubscibe"),
    ],
)
def test_misspelled_churn_still_escalates(text, expected):
    result = urgency_score(text)
    assert result.score >= 0.4, f"missed churn in: {text}"
    assert expected in result.signals["churn_phrases"], (
        "the audit trail must show what the customer actually typed, not the "
        "lexicon entry it matched -- support reads these terms as the reason"
    )


# The collision guard. Every one of these is a real word that sits one edit from
# a lexicon term, and every one is ordinary film vocabulary. Without
# NEVER_A_TYPO, "he killed the villain" raises a *billing* signal.
@pytest.mark.parametrize(
    "text",
    [
        "He killed the villain in the final act.",
        "The killing scenes were beautifully shot.",
        "Great style, and the store scene was lovely.",
        "The plot changed halfway through.",
        "A moving film about cancer.",
        "I was curious about the ending.",
        "The pacing was acceptable.",
    ],
)
def test_film_vocabulary_is_not_read_as_a_typo(text):
    result = urgency_score(text)
    assert result.score == 0.0, f"false urgency signal in: {text}"


def test_collision_guard_is_not_empty():
    """A silently-emptied exclusion set would make every test above pass by
    accident only until the lexicon changed."""
    assert {"killed", "cancer", "changed", "style"} <= NEVER_A_TYPO


# --------------------------------------------------------------------------
# sentiment model: typos and negation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "this movie was vad",          # bad, b -> v
        "the film was borst",          # worst
        "absolutely terrable acting",  # terrible
        "i dont like it",              # negation, no misspelling
        "i dont like it at all",
    ],
)
def test_misspelled_negative_reads_as_negative(text):
    """Each of these scored as *positive* under word-only features: the
    misspellings were out of vocabulary and contributed nothing, and in
    "i dont like it" the surviving token "like" is a positive feature."""
    result = score_review(text)
    assert result["p_negative"] > 0.5, f"{text!r} -> p_negative={result['p_negative']}"


def test_clean_spelling_still_works():
    """Character n-grams must not cost anything on correctly spelled text."""
    assert score_review("this movie was bad")["p_negative"] > 0.5
    assert score_review("this movie was wonderful")["p_negative"] < 0.5
