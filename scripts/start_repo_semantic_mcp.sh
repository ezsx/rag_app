#!/bin/bash
source /home/ezsx/infinity-env/bin/activate

export PYTHONPATH=/mnt/c/cursor_mcp/repo-semantic-mcp:/mnt/c/cursor_mcp/repo-semantic-mcp/libs
export SEMANTIC_MCP_REPO_ROOT=/mnt/c/llms/rag/rag_app
export SEMANTIC_MCP_QDRANT_URL=http://localhost:6333
export SEMANTIC_MCP_COLLECTION_PREFIX=repo_semantic
export SEMANTIC_MCP_INDEX_SCHEMA_VERSION=1
export SEMANTIC_MCP_EMBEDDING_BACKEND=tei_http
export SEMANTIC_MCP_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
export SEMANTIC_MCP_PROFILE_NAME=gpu_qwen3
export SEMANTIC_MCP_TEI_URL=http://localhost:8082
export SEMANTIC_MCP_TRANSPORT=stdio
export SEMANTIC_MCP_AUTO_INDEX_ON_START=0
export SEMANTIC_MCP_WATCH_ENABLED=0
export SEMANTIC_MCP_INCLUDE_GLOBS=auto
export SEMANTIC_MCP_EXCLUDE_GLOBS='.git/**,.venv/**,venv/**,**/__pycache__/**,node_modules/**,**/*.png,**/*.jpg,**/*.jpeg,**/*.gif,**/*.webp,**/*.ico,**/*.svg,**/*.zip,**/*.gz,**/*.tar,**/*.7z,**/*.pdf,**/*.db,**/*.sqlite,**/.env*,**/*.pem,**/*.key,**/*.npy,**/*.npz,datasets/_*'
export SEMANTIC_MCP_MAX_CHUNK_CHARS=1000
export SEMANTIC_MCP_EMBED_BATCH_DOCS=32
export SEMANTIC_MCP_EMBED_BATCH_CHARS=18000
export SEMANTIC_MCP_LOG_LEVEL=warning
export SEMANTIC_MCP_QUERY_TEMPLATE='Instruct: Retrieve relevant code and documentation snippets for repository search.
Query: {query}'

exec python /mnt/c/cursor_mcp/repo-semantic-mcp/apps/repo-semantic-mcp/main.py 2>/tmp/repo_semantic_mcp.log
