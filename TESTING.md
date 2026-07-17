# Testing review-radar

Three ways in, cheapest first.

| | What it covers | Command |
|---|---|---|
| Test suite | every rule, both defects, all regressions | `pytest` |
| Browser demo | the whole path, visually | `uvicorn service.app:app --port 3000` → http://127.0.0.1:3000/demo |
| curl | one review, raw JSON | see below |

---

## 1. The test suite

```bash
pytest                       # everything
pytest -v                    # one line per case
pytest tests/test_typos.py   # misspelling + negation handling
pytest tests/test_urgency.py # the urgency engine and both routing defects
pytest -k churn              # anything about churn
```

`tests/test_typos.py` loads the trained model, so run `python -m src.train` first
if `models/tfidf_logreg.pkl` does not exist yet.

## 2. The browser demo

```bash
uvicorn service.app:app --port 3000
```

Then open **http://127.0.0.1:3000/demo** — not the file on disk. Opening
`demo/index.html` directly works too, but only if the API is already running on
port 3000, because the page has to fetch its scores from somewhere.

Deep-link straight to a review (handy mid-presentation):

```
http://127.0.0.1:3000/demo?q=Charged%20twice,%20cancelling%20today!!!
```

Point the page at a different API:

```
http://127.0.0.1:3000/demo?api=http://192.168.1.20:3000
```

## 3. curl

```bash
curl -X POST http://127.0.0.1:3000/review \
  -H "Content-Type: application/json" \
  -d '{"text": "Charged twice, I want a REFUND, cancelling today!!!"}'
```

On Windows `cmd`, escape the inner quotes:

```cmd
curl -X POST http://127.0.0.1:3000/review -H "Content-Type: application/json" -d "{\"text\": \"Charged twice, cancelling today!!!\"}"
```

Score a review without the service at all:

```bash
python -m src.predict "the film was borst"
```

---

# Queries worth trying

Each block is a claim the system makes about itself. The expected result is what
should happen — if it doesn't, that's a bug worth knowing about.

### The point of the whole project: negative ≠ urgent

| Review | p_negative | urgency | Route |
|---|---|---|---|
| `Charged twice and nobody replied. I want a REFUND, cancelling today!!!` | 0.32 | 0.83 | `support_urgent` |
| `The ending felt rushed and the pacing dragged in the second act.` | 0.69 | 0.00 | `feedback_backlog` |

The first is *less* negative than the second and far more urgent. A single
sentiment score cannot tell these apart — that is the argument for scoring twice.

### Urgency escalates without the sentiment model's permission

```bash
-d '{"text": "Charged twice and nobody replied. I want a REFUND, cancelling today!!!"}'
```

`p_negative` is only **0.32** — the model is trained on *movie* reviews and is out
of its depth on billing language. Urgency is **0.83**. It routes to
`support_urgent` anyway. The original code checked sentiment first and dropped
this customer into analytics.

### Churn intent with no churn keyword

```bash
-d '{"text": "I dont want my subscription"}'
```
→ `support_urgent`. No word from the churn list appears; `CHURN_PATTERNS` catches
the meaning. Compare with:

```bash
-d '{"text": "I dont want to spoil the ending, but it was boring."}'
```
→ not urgent. Same opening four words, different meaning, because the patterns
are anchored to subscription nouns.

### Misspellings

```bash
-d '{"text": "this movie was vad"}'        # bad
-d '{"text": "the film was borst"}'        # worst
-d '{"text": "absolutely terrable acting"}' # terrible
-d '{"text": "I want to cancle my subscription"}'
-d '{"text": "i need a refudn NOW"}'
```

All are understood. The first three are the sentiment model (character n-grams);
the last two are the urgency lexicon (bounded edit distance). Check
`urgency_signals` on the churn ones — it reports **`cancle`**, the word actually
typed, not `cancel`. The support team reads those terms as the reason a review
was escalated, so the audit trail has to be literally true.

### Negation

```bash
-d '{"text": "i dont like it"}'                          # -> negative
-d '{"text": "I love this service, I would never cancel."}'  # -> NOT urgent
-d '{"text": "I never watch films like this. I want a refund."}'  # -> urgent
```

The second must not page anyone: "never cancel" is praise wearing a churn word.
The third must, because the negation belongs to a different sentence.

### Film vocabulary that fuzzy matching would ruin

```bash
-d '{"text": "He killed the villain in the final act."}'
-d '{"text": "The plot changed halfway through."}'
-d '{"text": "A moving film about cancer."}'
-d '{"text": "I was curious about the ending."}'
```

All must score **0.00** urgency. Each contains a real word one edit from a lexicon
term — `killed`/`billed`, `changed`/`charged`, `cancer`/`cancel`,
`curious`/`furious`. On a film site these are everywhere, which is why fuzzy
matching is barred from treating any word the corpus knows as a misspelling.

### Edges

```bash
-d '{"text": ""}'          # -> 422, text must not be blank
-d '{"text": "   "}'       # -> 422
-d '{"text": "ok"}'        # -> fine, no signals
```

---

## Reproducing the model results

```bash
python -m src.train         # trains the shipped model
python -m src.experiments   # 6-config MLflow sweep, prints the winner
python -m src.robustness    # word vs word+char, clean vs typo'd
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

`src.robustness` is the one that justifies character n-grams. It trains two
models differing in exactly one variable and scores both on clean IMDB and on
the same reviews with typos injected. F1 on clean IMDB is *not* where the benefit
shows up — IMDB is edited prose with almost nothing to fix.

## Docker

```bash
python -m src.train                 # models/ is gitignored; the image needs the pickle
docker compose up --build api
curl http://127.0.0.1:3000/health   # {"status":"ok"}
```

Then http://127.0.0.1:3000/demo.

```bash
docker compose --profile dev up     # live reload against local source
docker inspect --format='{{.State.Health.Status}}' $(docker compose ps -q api)
```
