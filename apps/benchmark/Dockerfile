# Benchmark runner — LlamaIndex + naive + custom adapter.
# Контекст сборки: корень репозитория (../../).
# Все GPU-модели доступны по HTTP (gpu_server, llama-server).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY apps/benchmark/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY benchmarks/ /workspace/benchmarks/
COPY datasets/ /workspace/datasets/
COPY scripts/run_benchmark_retrieval.py /workspace/scripts/
COPY scripts/run_benchmark_agent.py /workspace/scripts/
