"""Real-time sentiment API.

POST /review with {"text": "..."} and get back the negativity score,
urgency score, and where the review should be routed.

Run:
    uvicorn service.app:app --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import load_model, score_review


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # warm up so the first request is already fast
    yield


app = FastAPI(title="review-radar", description="Real-time customer review sentiment + urgency scoring", lifespan=lifespan)


class ReviewIn(BaseModel):
    text: str


class ReviewOut(BaseModel):
    p_negative: float
    urgency: float
    urgency_signals: dict
    route: str
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/review", response_model=ReviewOut)
def review(payload: ReviewIn) -> ReviewOut:
    start = time.perf_counter()
    result = score_review(payload.text)
    latency_ms = (time.perf_counter() - start) * 1000
    return ReviewOut(**result, latency_ms=round(latency_ms, 2))
