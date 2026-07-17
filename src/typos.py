"""Realistic typo injection, for measuring robustness instead of assuming it.

The claim "character n-grams make the model handle misspellings" is easy to
state and easy to get wrong. IMDB is clean, edited prose, so F1 on the IMDB test
set says nothing at all about it -- there are no misspellings there to survive.
This module builds the missing evaluation set: take the same reviews, corrupt
them the way real people actually mistype, and score the same model again. The
gap between the two numbers is the entire claim, measured.

The four corruptions are the four that dominate real typing error corpora:

    transposition   worst -> wrost      adjacent letters swapped
    substitution    bad   -> vad        a physically neighbouring key
    deletion        worst -> wost       a letter dropped
    doubling        worst -> worsst     a letter repeated

Substitution is keyboard-aware rather than random: people hit the key next to
the one they meant, which is why "bad" becomes "vad" (v is left of b) and not
"qad". A uniform random letter would make the task harder than reality and
overstate the benefit.
"""

import random

# QWERTY physical neighbours. Used for substitution so corruptions land where
# real fingers land.
_NEIGHBOURS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}

_MIN_LEN = 4  # shorter words have too little room to stay recognisable


def corrupt_word(word: str, rng: random.Random) -> str:
    """Apply one random corruption to a single word."""
    if len(word) < _MIN_LEN:
        return word

    kind = rng.choice(("transpose", "substitute", "delete", "double"))
    i = rng.randrange(1, len(word) - 1)  # leave the first/last letter alone

    if kind == "transpose":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "substitute":
        opts = _NEIGHBOURS.get(word[i].lower())
        return word[:i] + rng.choice(opts) + word[i + 1:] if opts else word
    if kind == "delete":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]  # double


def corrupt_text(text: str, rate: float = 0.15, seed: int = 0) -> str:
    """Corrupt a share of the words in `text`.

    rate=0.15 means roughly one word in seven is mistyped -- heavy for an edited
    review, ordinary for a furious customer typing on a phone.
    """
    rng = random.Random(seed)
    return " ".join(
        corrupt_word(w, rng) if rng.random() < rate else w
        for w in text.split()
    )


def corrupt_series(texts, rate: float = 0.15, seed: int = 0) -> list[str]:
    """Corrupt an iterable of texts, each with its own derived seed."""
    return [corrupt_text(t, rate=rate, seed=seed + i) for i, t in enumerate(texts)]
