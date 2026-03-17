# Phase 1: тонкий CPU-only образ.
# LLM, embedding, reranker — внешние HTTP-сервисы (llama-server, TEI).
# Контекст сборки: корень репозитория (../../).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY apps/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/ /app/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
