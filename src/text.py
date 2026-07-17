"""Text normalisation shared by the sentiment model and the urgency engine.

Both components must agree on what a word *is*, or they disagree about the same
review. The urgency engine already stripped apostrophes so "don't" and "dont"
were one case; the vectorizer did not, so the model saw them as two unrelated
tokens. This module is the single definition, imported by both.

Kept deliberately dependency-free: it is imported by `src.predict`, which the
service loads at startup, and by the pickled model's preprocessor. Pulling
mlflow/sklearn in here would drag them into the serving path.

WARNING -- editing `normalize` silently changes an already-trained model.
The fitted pipeline stores `preprocessor=normalize` as a *reference*, not as
code, so a saved model picks up whatever this function says at load time. Change
the rules here and the model keeps its old vocabulary while reading text by the
new ones: no error, no version bump, just a quietly worse model. Retrain
(`python -m src.train`) after any change below.
"""

import re

_TAG = re.compile(r"<[^>]+>")            # IMDB reviews are littered with <br />
_APOS = re.compile(r"[’'`´]")
# Letters only. Applying this to punctuation as well would quietly rewrite
# "BAD!!!" to "BAD!!", and the run length of "!" is signal the urgency engine
# scores -- it just happens to read it off the raw text, so the damage would
# have been invisible here and shown up only as a weaker intensity feature.
_RUN = re.compile(r"([a-z])\1{2,}")      # 3+ of the same letter in a row
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z]+")


def normalize(text: str) -> str:
    """Lowercase, drop markup, unify apostrophes, and flatten elongation.

    "SOOOO BAD!!!" and "so bad!" should not be different vocabulary. Runs are
    collapsed to two rather than one so "terrrrible" -> "terrible" keeps its
    real double letter instead of becoming "terible".
    """
    text = text.lower()
    text = _TAG.sub(" ", text)
    text = _APOS.sub("", text)
    text = _RUN.sub(r"\1\1", text)
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    """Alphabetic tokens of already-normalised text."""
    return _WORD.findall(text)


def damerau_levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    """Edit distance with transpositions, bounded for speed.

    Transposition matters more than the extra code costs: "cancle" for "cancel"
    is two plain Levenshtein edits but one typo, and swapped adjacent letters
    are the single most common way people misspell a word they know.

    Returns max_dist + 1 as soon as the true distance is known to exceed it,
    so callers can reject cheaply without computing the exact figure.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1

    prev2: list[int] = []
    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        best = cur[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost,  # substitution
            )
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)  # transposition
            best = min(best, cur[j])
        if best > max_dist:
            return max_dist + 1
        prev2, prev = prev, cur

    return prev[len(b)]


def is_typo_of(token: str, target: str) -> bool:
    """Is `token` a plausible misspelling of `target`?

    The tolerance scales with length because a one-character edit means something
    very different at three letters than at ten: "bad"/"bat"/"mad"/"sad" are all
    one edit apart but are different words, whereas nothing else in English is
    one edit from "cancelling". Short targets therefore demand an exact match.
    """
    if token == target:
        return True
    n = len(target)
    if n < 5:
        return False                       # too short to tell typo from word
    budget = 1 if n < 8 else 2
    return damerau_levenshtein(token, target, budget) <= budget
