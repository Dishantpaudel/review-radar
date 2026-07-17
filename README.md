# review-radar

Real-time customer review analysis for a movie website: read every review the moment it arrives, spot the angry ones, and route the urgent ones to a human — before the customer leaves.

Built as the final project for Industrial Machine Learning (Harbour.Space, 2026).

## The problem

A movie site gets hundreds of thousands of reviews a month. Nobody can read them all, so angry customers are noticed too late — after they already left. An angry review is an early warning; today it is ignored.

## What this system does

```
new review ──► sentiment model P(negative) ──┐
                                             ├──► router ──► support team, now
          └──► urgency engine (rules) ───────┘              product feedback backlog
                                                            analytics
```

Each review is scored twice, because **"negative" and "urgent" are different questions**:

```python
if urgency >= 0.4 and p_negative >= 0.1:  return "support_urgent"
if p_negative >= 0.5:                     return "feedback_backlog"
return "analytics"
```

Two ideas from review feedback are built in:

1. **Negative ≠ urgent.** "The ending felt rushed" is useful feedback, not an emergency. "Charged twice, cancelling NOW!!!" needs a human in minutes. A transparent rule-based urgency engine (churn intent, billing terms, anger vocabulary, shouting) separates the two, and every escalation returns the exact terms that matched.
2. **The decision threshold is chosen for money, not for F1.** Contacting a flagged customer costs $1; saving one is worth ~$9. Because value is 9× cost, the profit-optimal threshold is *lower* than the F1-optimal one — it pays to contact more people. Break-even precision is only C / $9 ≈ **11%**.

Urgency escalates **independently** of the sentiment score, with sentiment acting only as a floor. That ordering is deliberate — see [Defects found](#defects-found-and-fixed).

Both scorers are built to survive the text real customers actually write — misspellings, missing apostrophes, shouting. `"its borst"` still reads as negative; `"I want to cancle my subscription"` still escalates. See [Handling real-world text](#handling-real-world-text).

## Results

| Model | F1 (negative) | Recall (negative) | Precision (negative) | Notes |
|---|---|---|---|---|
| Constant guess | 0.50 | — | — | the score to beat |
| **TF-IDF (word + char) + Logistic Regression** | **0.903** | 0.901 | 0.905 | trained here, ~3 ms inference, explainable, typo-robust |
| DistilBERT (pretrained SST-2, no fine-tune) | 0.898 | 0.913 | 0.885 | 3,000-review sample, CPU |

Both clear the project gate **F1(negative) ≥ 0.85**. The pretrained transformer ties the baseline; fine-tuning it on IMDB would push it to ~0.93 but needs a GPU. The baseline goes to production: ~100× cheaper to serve, fully explainable, precision@1000 = 1.00 (the top of the support queue is pure signal).

The shipped model unions **word n-grams with character n-grams**. On clean IMDB that is worth almost nothing (+0.003 F1) — but IMDB is long, edited prose, and that number is measuring the wrong thing. The character features exist for two-line reviews with a typo in them, where a word-only model scores `"its borst"` at 0.16 (confidently *positive*, because "borst" is out of vocabulary and contributes nothing). See [Handling real-world text](#handling-real-world-text).

### Business threshold (notebook 04)

- Break-even precision: **0.111**
- Profit-optimal threshold: **0.43** (below the default 0.50, as theory predicts)
- Precision at the optimum: 0.88 — 8× above break-even, so the outreach program is a low-risk bet
- On this well-calibrated model the gain from tuning is modest (~$1.8k/month); with a noisier model or real-world class imbalance (only ~20% of production reviews are negative, vs 50% in IMDB) the gap grows. The method — sweep thresholds, maximize expected profit — is the point.

## Handling real-world text

Real customers do not spell-check angry reviews, and a model trained on edited prose breaks on the text that matters most. Both scorers are hardened against this, and both use one shared definition of "a word" (`src/text.py`) so the model and the urgency engine can never disagree about the same review.

**The sentiment model — character n-grams.** Word-only TF-IDF has a hard failure: a word it never saw contributes *nothing* (not "a little" — a word with no vocabulary slot has nowhere to put a weight). So misspellings drop out of exactly the reviews most likely to contain them.

| Review | word-only | word + char |
|---|---|---|
| `its borst` | 0.16 — positive ✗ | **0.88 — negative ✓** |
| `i dont like it` | 0.46 — positive ✗ | **0.59 — negative ✓** |
| `the film was borst` | 0.64 | **0.96** |
| `this film was awfull` | 0.72 | **0.98** |

`i dont like it` is the sharp one: there is no misspelling in it. Word features chop it to `dont · like · it`, and **`like` is one of the model's strongest *positive* words** — the complaint was read as praise by the only in-vocabulary token left. Character n-grams fix the typos; normalising `don't`→`dont` is what lets the `dont like` bigram exist at all.

> Measured over all of IMDB the gain is only +0.6 F1 on typo'd text — because IMDB reviews average ~230 words and coast on redundancy. Only 233 of 25,000 test reviews are short enough for a typo to cost anything. **IMDB cannot measure this**, which is why `tests/test_typos.py` and `src/robustness.py` exist. `python -m src.robustness` trains word-only vs word+char and scores both on clean and typo'd text.

**The urgency lexicon — bounded edit distance.** `"I want to cancle my subscription"` is unmistakable churn that an exact keyword match scores 0.0. The lexicon now also matches within a small edit distance (1 edit for 5–7 letter terms, 2 for longer, transposition counts as one). The audit trail reports **`cancle`** — what the customer typed — not `cancel`, because support reads those terms as the reason a review escalated.

> This is dangerous on a *film* site, and measuring it is what made it safe. `killed` (1,111 occurrences in IMDB) is one edit from `billed`; `changed` from `charged`; `cancer` from `cancel`. Naively, "he killed the villain" raises a billing alert. The guard is one rule mined from the data: **any token appearing ≥25× in IMDB is barred from fuzzy matching** — if the corpus says it's a word, it isn't a typo.

## Defects found and fixed

All three defects below were found by **manually using the system**, not by the test suite. Each is now pinned by a regression test named after the failure.

### 1. The churn lexicon was too literal

`CHURN_PHRASES` matched fixed keywords only, so two sentences with identical meaning took opposite paths:

| Input | Urgency | Route |
|---|---|---|
| "I want to **cancel** my subscription" | 0.525 | `support_urgent` |
| "I **dont want** my subscription" | 0.075 | `feedback_backlog` ❌ |

**Fix:** added `CHURN_PATTERNS` — regexes for *compositional* churn intent, anchored to subscription/account nouns so `"I dont want to spoil the ending"` cannot false-positive. Apostrophes are normalised so `don't` and `dont` are one case.

### 2. The routing precedence was inverted

The original `route()` evaluated sentiment first:

```python
if p_negative < 0.5:  return "analytics"   # ← vetoed everything below it
```

The sentiment model is trained on IMDB **movie** reviews and is out of domain on billing language. It scored a departing customer at **0.24**:

```
"Charged twice and nobody replied. I want a REFUND, cancelling today!!!"
p_negative = 0.24   urgency = 0.83   →   "analytics"
```

That unreliable score silently overruled an urgency of 0.83 and routed a churning customer to analytics — the exact outcome this project exists to prevent.

**Fix:** urgency escalates independently, with sentiment reduced to a `clearly_positive` floor that stops a churn keyword inside praise from paging an agent.

> **The principle:** an unreliable component must never be able to veto a reliable one.

### 3. The guard for defect 2 didn't actually hold

The floor from defect 2 had a test, and the test passed. The system still failed — because the test asserted on a made-up score:

```python
assert route(p_negative=0.05, urgency=0.45) == "analytics"   # 0.05 is invented
```

Real praise containing a churn word scores far higher, and clears the floor:

| Review | p_negative | Route (floor only) |
|---|---|---|
| "I love this, I'd never cancel." | 0.14 | `analytics` ✓ |
| "I love this service, I would never cancel." | 0.30 | `support_urgent` ❌ |

The floor **cannot** be tuned to fix this: the real churning customer from defect 2 scores **0.24**, and this praise scores **0.30** — the praise is *more negative than the churn*, so no threshold separates them. The floor was asking the out-of-domain model a question it gets backwards.

**Fix:** negation scope. `"never cancel"` is not churn, so a negator now suppresses a churn verb within 5 tokens **inside its own clause** (splitting on commas as well as sentence enders, so `"I never watch films like this. I want a refund."` and `"i dont like it, want to cancel"` both still escalate — the negator belongs to a different clause). The comma boundary is chosen deliberately: not crossing it can let praise like `"I'd never, ever cancel"` slip through and escalate (a wasted support minute), whereas crossing it would silence real churn (a lost customer) — and this project values those the other way round. One known gap remains and is documented rather than papered over: `"never again"` is itself a churn phrase, so `"never again will I switch"` still false-escalates.

> **The lesson:** a unit test that invents its own inputs only proves the function does what the function does. This bug lived in the gap between the component and the pipeline, where a unit test cannot look — so the replacement drives the real model end-to-end.

### 3b. …and the floor it left behind then vetoed real churn

Once negation owned the praise guard, the `clearly_positive` floor (0.2) had only one job left — and it was doing it wrong. Character n-grams pulled billing-language sentiment *down*, and a natural phrasing landed under the floor:

```
"Charged twice, I want a REFUND, cancelling today!!!"
p_negative = 0.17   urgency = 0.84   →   analytics   ❌
```

0.17 < 0.2, so the floor dropped a churning customer — defect 2, resurrected by my own typo fix. **Fix:** lower the floor to **0.1**. It can be that low now precisely because negation took over the praise guard; the floor's only remaining duty is a churn word inside *confident* praise (p_negative < 0.1), the one regime where this out-of-domain model is actually reliable. `test_flagship_churn_still_escalates_end_to_end` now drives the real model on four phrasings so this cannot return silently.

Tests grew **10 → 23 → 56**, covering all defects, the false-positive guards, misspellings, negation, and the collision guard — the flagship-churn test drives the real model, not constants, because that is the only kind of test that would have caught 3b.

## Data

[IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., 2011): 50,000 labeled movie reviews, balanced, 25k train / 25k test. Downloaded automatically on first run. It is a stand-in — in production the model would be retrained monthly on the site's own reviews.

## Project structure

```
src/
  data.py                 load IMDB
  text.py                 shared normalisation + edit distance (model and urgency agree here)
  train.py                TF-IDF (word + char) + LogReg training, tracked in MLflow
  evaluate.py             F1/recall/precision for the negative class, precision@k
  experiments.py          6-config MLflow sweep
  typos.py                keyboard-aware typo injection
  robustness.py           word vs word+char, clean vs typo'd — the evidence for char n-grams
  compare_transformer.py  pretrained DistilBERT comparison
  threshold.py            profit-vs-threshold sweep, break-even math
  urgency.py              urgency scoring + routing matrix (fuzzy lexicon, negation)
  predict.py              load model, score one review
service/app.py            FastAPI: POST /review → {p_negative, urgency, route}; GET /demo
demo/index.html           browser demo — live scoring + decision-space plot
notebooks/01–04           EDA → baseline → transformer → business threshold
tests/                    urgency, typos, threshold math, API (56 tests)
Dockerfile                two-stage, slim, non-root, health-checked
docker-compose.yml        api / dev (reload) / train profiles
requirements-serve.txt    runtime deps only (no torch/mlflow) → 762 MB image
presentation.html         slide deck
how-it-works.html         end-to-end walkthrough
TESTING.md                every query worth trying, and what should happen
```

## How to run

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

python -m src.train                  # trains model, logs to MLflow, saves models/tfidf_logreg.pkl (~7 min)
python -m src.compare_transformer    # optional: DistilBERT comparison (CPU, ~10 min)
python -m src.robustness             # optional: proves char n-grams earn their place
pytest                               # 56 tests

uvicorn service.app:app --port 3000  # live API + demo at /demo
```

Then open **http://127.0.0.1:3000/demo** to score reviews in the browser.

Try it from the command line:

```bash
curl -X POST http://127.0.0.1:3000/review -H "Content-Type: application/json" \
  -d "{\"text\": \"Terrible! Charged twice, I want a REFUND, cancelling today!!!\"}"
# → {"p_negative": 0.50, "urgency": 0.93, "route": "support_urgent", ...}
```

`TESTING.md` has a full set of queries — typos, negation, and the film-vocabulary collisions — each with the result it should produce.

Experiment tracking: `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://127.0.0.1:5000

### With Docker

```bash
python -m src.train                  # models/ is gitignored; the image needs the pickle
docker compose up --build api        # → http://127.0.0.1:3000/demo
docker compose --profile dev up      # live reload against local source
```

The serving image installs `requirements-serve.txt` only — no torch, no mlflow — so it is **762 MB**, runs as a non-root user, and its healthcheck hits `/health` (not just the port, because uvicorn binds before the model finishes loading).

## Roadmap

1. ✅ Baseline model — already passes the goal
2. ✅ Transformer comparison + live service + urgency routing + profit-optimal threshold
3. ✅ Typo/negation robustness, browser demo, containerised service
4. Monthly retraining on own reviews, drift monitoring, threshold re-tuned as costs change
5. Topic mining on negative reviews — find out *why* customers are unhappy, not just that they are
