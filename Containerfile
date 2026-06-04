FROM docker.io/library/python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime libs only (no compiler toolchain)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-container.txt .
# insightface --no-deps: hindari opencv-python (GUI, ~70 MB lebih besar)
RUN pip install -r requirements-container.txt \
    && pip install insightface --no-deps \
    && python -c "from insightface.app import FaceAnalysis; a = FaceAnalysis(providers=['CPUExecutionProvider']); a.prepare(ctx_id=0)"

COPY app.py .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
