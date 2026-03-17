# Phase 1: тонкий CPU-only образ для Telegram ingest.
# Dense embedding — TEI HTTP (внешний). Sparse — fastembed CPU. Store — Qdrant HTTP.
# Контекст сборки: корень репозитория (../../).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY apps/ingest/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY scripts/ /app/scripts/
COPY src/ /app/src/

RUN mkdir -p /app/sessions

ENTRYPOINT ["python3", "-m", "scripts.ingest_telegram"]
