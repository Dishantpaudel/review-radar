# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# builder: compile wheels once, so the runtime layer installs without a
# toolchain and without leaving build artefacts in the final image.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-serve.txt

# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=3000

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-serve.txt \
 && rm -rf /wheels

COPY src/ ./src/
COPY service/ ./service/
COPY demo/ ./demo/

# The trained model is a build input, not something the image produces: training
# needs the 80 MB IMDB download and several minutes of CPU, which would make
# every build slow, network-dependent and non-reproducible. models/ is
# gitignored, so a clean checkout has no model -- run `python -m src.train`
# before building, or mount a model at runtime (see docker-compose.yml).
COPY models/ ./models/

# Fail at build rather than at first request. A missing model otherwise surfaces
# as a stack trace inside the lifespan handler, several minutes after the build
# "succeeded".
RUN test -f models/tfidf_logreg.pkl \
 || (echo "ERROR: models/tfidf_logreg.pkl missing. Run: python -m src.train" && exit 1)

# Drop privileges. The service writes nothing; it needs no more than read access
# to its own code.
RUN useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 3000

# Hits the real endpoint, not just the port: uvicorn binds before the model is
# loaded, so a TCP check would report healthy while the first request still 500s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:3000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "3000"]
