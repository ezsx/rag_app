"""Benchmark configuration — URLs, paths, parameters from env."""

import os

# Endpoints (Docker → host services via host.docker.internal)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:16333")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8082")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8080")
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8001")

# Qdrant
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_colbert_v2")
DENSE_VECTOR_NAME = "dense_vector"
SPARSE_VECTOR_NAME = "sparse_vector"
COLBERT_VECTOR_NAME = "colbert_vector"

# Datasets
RETRIEVAL_DATASET = "datasets/eval_retrieval_100.json"
AGENT_DATASET = "datasets/eval_golden_v2.json"
RESULTS_DIR = "benchmarks/results"

# Retrieval params
DENSE_TOP_K = 20
SPARSE_TOP_K = 100
FINAL_TOP_K = 20
RERANK_TOP_N = 10

# Agent params
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096
LLM_CONTEXT_WINDOW = 32768

# BM25 sparse model (same as production ingest)
SPARSE_MODEL_NAME = "Qdrant/bm25"

# Agent E2E — exact question IDs for retrieval subset
AGENT_RETRIEVAL_IDS = [
    "golden_q01", "golden_q02", "golden_q03", "golden_q04", "golden_q05",
    "golden_q06", "golden_q07", "golden_q08", "golden_q09", "golden_q10",
    "golden_q11", "golden_q12", "golden_q13", "golden_q14", "golden_q15",
    "golden_q16", "golden_q25",
]
