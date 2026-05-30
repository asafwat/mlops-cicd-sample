# Multi-stage, non-root, slim Python 3.11. Bakes the trained model.joblib into
# the image at /app/ — same pattern as the other lab serving images.
#
# Build flow:
#   1. CI workflow runs `python -m src.train` to generate model.joblib
#   2. Then `docker build .` COPYs both the code AND the model
#   3. Image is pushed to ghcr.io
#
# Local equivalent:
#   python -m src.train
#   docker build -t mlops-cicd-sample:0.1 .
#   kind load docker-image mlops-cicd-sample:0.1 --name mlops

FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --system --gid 1000 app \
 && useradd  --system --uid 1000 --gid app --home-dir /app --shell /sbin/nologin app \
 && mkdir -p /app && chown -R app:app /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src/ /app/src/
# Bake the trained model. CI generates it before `docker build` runs.
COPY --chown=app:app model.joblib /app/model.joblib

USER app
WORKDIR /app

EXPOSE 8080

ENTRYPOINT ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]
