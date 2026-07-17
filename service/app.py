"""Real-time sentiment API.

POST /review with {"text": "..."} and get back the negativity score,
urgency score, and where the review should be routed.

Run:
    uvicorn service.app:app --port 3000

Then open http://127.0.0.1:3000/demo to score reviews in the browser.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.predict import load_model, score_review

DEMO_FILE = Path(__file__).resolve().parent.parent / "demo" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # warm up so the first request is already fast
    yield


app = FastAPI(title="review-radar", description="Real-time customer review sentiment + urgency scoring", lifespan=lifespan)

# The demo is served from /demo, which is same-origin and needs none of this.
# It is here so the page also works when opened straight off disk, where the
# browser sends `Origin: null` and blocks the response without an explicit
# allow. The service exposes no auth and no cookies, so a permissive policy
# costs nothing; a deployment that adds either should pin allow_origins to
# the real front-end host.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ReviewIn(BaseModel):
    # Unbounded text is a free denial of service: TF-IDF happily vectorises a
    # 50 MB paste and blocks the worker while it does. 20k characters is ~40x
    # the longest review in IMDB.
    text: str = Field(..., min_length=1, max_length=20_000)


class ReviewOut(BaseModel):
    p_negative: float
    urgency: float
    urgency_signals: dict
    route: str
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/demo", include_in_schema=False)
def demo() -> FileResponse:
    """Serve the browser demo from the API itself, so it is same-origin."""
    return FileResponse(DEMO_FILE)


@app.post("/review", response_model=ReviewOut)
def review(payload: ReviewIn) -> ReviewOut:
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="text must not be blank")
    start = time.perf_counter()
    result = score_review(payload.text)
    latency_ms = (time.perf_counter() - start) * 1000
    return ReviewOut(**result, latency_ms=round(latency_ms, 2))
