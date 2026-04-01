"""
Минимальный GPU-сервер для embedding и reranking.
Без FastAPI/pydantic — только stdlib http.server + torch + transformers.

Модели (март 2026):
  - Embedding: pplx-embed-v1-0.6B (Perplexity, mean pooling, 1024-dim)
  - Reranker: Qwen3-Reranker-0.6B-seq-cls (Tom Aarsen conversion, chat template)
  - ColBERT: jina-colbert-v2 (128-dim per-token MaxSim)

Запуск:
    source /home/ezsx/infinity-env/bin/activate
    CUDA_VISIBLE_DEVICES=0 python /mnt/c/llms/rag/rag_app/scripts/gpu_server.py
"""

import json
import logging
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gpu_server")

# Модели загружаются из /mnt/c/llms/models/ (Windows-accessible) или /home/tei-models/
EMBEDDING_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH", "/mnt/c/llms/models/pplx-embed-v1-0.6B"
)
RERANKER_MODEL_PATH = os.environ.get(
    "RERANKER_MODEL_PATH", "/mnt/c/llms/models/Qwen3-Reranker-0.6B-seq-cls"
)
COLBERT_MODEL_PATH = os.environ.get(
    "COLBERT_MODEL_PATH", "/home/tei-models/jina-colbert-v2"
)

emb_tokenizer = None
emb_model = None
rer_tokenizer = None
rer_model = None
col_tokenizer = None
col_model = None
col_linear = None  # projection 1024→128


def load_models():
    global emb_tokenizer, emb_model, rer_tokenizer, rer_model, col_tokenizer, col_model, col_linear

    logger.info("Загрузка embedding: %s", EMBEDDING_MODEL_PATH)
    t0 = time.time()
    emb_tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_PATH, trust_remote_code=True,
    )
    # pplx-embed: bf16 вместо fp16 — fp16 даёт NaN на длинных текстах (overflow)
    # bf16 имеет больший dynamic range (exp=8 бит vs 5 у fp16), решает overflow
    emb_model = AutoModel.from_pretrained(
        EMBEDDING_MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()
    logger.info("Embedding загружен за %.1f сек", time.time() - t0)

    logger.info("Загрузка reranker: %s", RERANKER_MODEL_PATH)
    t0 = time.time()
    rer_tokenizer = AutoTokenizer.from_pretrained(
        RERANKER_MODEL_PATH, padding_side="left",
    )
    rer_model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL_PATH, torch_dtype=torch.float16,
    ).cuda().eval()
    logger.info("Reranker загружен за %.1f сек", time.time() - t0)

    logger.info("Загрузка ColBERT: %s", COLBERT_MODEL_PATH)
    t0 = time.time()
    col_tokenizer = AutoTokenizer.from_pretrained(COLBERT_MODEL_PATH, trust_remote_code=True)
    col_model = AutoModel.from_pretrained(
        COLBERT_MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True
    ).cuda().eval()
    # Загружаем linear projection 1024→128 отдельно из safetensors
    import safetensors.torch as st
    all_tensors = st.load_file(COLBERT_MODEL_PATH + "/model.safetensors", device="cuda")
    if "linear.weight" in all_tensors:
        col_linear = all_tensors["linear.weight"].half()  # [128, 1024]
        logger.info("ColBERT linear projection: %s", col_linear.shape)
    logger.info("ColBERT загружен за %.1f сек", time.time() - t0)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8", errors="replace"))

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok" if emb_model else "loading"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            data = self._read_body()
            if self.path == "/embed":
                self._send_json(self._embed(data))
            elif self.path == "/v1/embeddings":
                self._send_json(self._embed_openai(data))
            elif self.path == "/rerank":
                self._send_json(self._rerank(data))
            elif self.path == "/colbert-encode":
                self._send_json(self._colbert_encode(data))
            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as e:
            logger.exception("Error in %s", self.path)
            self._send_json({"error": str(e)}, 500)

    def _embed(self, data):
        texts = data.get("inputs", data.get("texts", []))
        if isinstance(texts, str):
            texts = [texts]
        # Guard: пустые строки вызывают IndexError в tokenizer
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return []
        with torch.no_grad():
            enc = emb_tokenizer(
                texts, padding=True, truncation=True, max_length=4096,
                return_tensors="pt"
            ).to("cuda")
            out = emb_model(**enc)
            # Cast to fp32 for mean pooling — fp16 overflows on long sequences → NaN
            hidden = out.last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).float()
            embeddings = (hidden * mask).sum(1) / mask.sum(1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    def _embed_openai(self, data):
        """OpenAI-совместимый /v1/embeddings endpoint (для repo-semantic-mcp)."""
        inp = data.get("input", data.get("inputs", []))
        if isinstance(inp, str):
            inp = [inp]
        vectors = self._embed({"inputs": inp})
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": vec}
                for i, vec in enumerate(vectors)
            ],
            "model": "pplx-embed-v1-0.6b",
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    def _rerank(self, data):
        """Rerank через Qwen3-Reranker-0.6B-seq-cls.

        Требует chat template: system + user (Instruct/Query/Document) + assistant thinking.
        """
        query = data.get("query", "")
        texts = data.get("texts", [])
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements '
            'based on the Query and the Instruct provided. Note that the answer can '
            'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        formatted = [
            f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {t}{suffix}"
            for t in texts
        ]
        with torch.no_grad():
            enc = rer_tokenizer(
                formatted, padding=True, truncation=True, max_length=8192,
                return_tensors="pt"
            ).to("cuda")
            out = rer_model(**enc)
            scores = out.logits.view(-1).float()
        results = [{"index": i, "score": float(s)} for i, s in enumerate(scores)]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def _colbert_encode(self, data):
        """Encode текстов через jina-colbert-v2 → per-token vectors (128-dim).

        Input: {"texts": ["text1", "text2", ...], "is_query": false}
        Output: [[[tok1_128dim], [tok2_128dim], ...], [...], ...]

        is_query=true: добавляет [Q] маркер, truncate до 32 токенов.
        is_query=false: документы, truncate до 8192 токенов.
        """
        texts = data.get("texts", [])
        if isinstance(texts, str):
            texts = [texts]
        is_query = data.get("is_query", False)
        max_len = 32 if is_query else 8192

        with torch.no_grad():
            enc = col_tokenizer(
                texts, padding=True, truncation=True, max_length=max_len,
                return_tensors="pt"
            ).to("cuda")
            out = col_model(**enc)
            # ColBERT: per-token embeddings из last_hidden_state [batch, seq_len, 1024]
            token_embeddings = out.last_hidden_state
            # Проецируем 1024→128 через linear.weight
            if col_linear is not None:
                token_embeddings = token_embeddings @ col_linear.T  # [batch, seq, 1024] @ [1024, 128] → [batch, seq, 128]
            # Нормализуем каждый token vector
            token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)
            # Маскируем padding tokens
            mask = enc["attention_mask"]  # [batch, seq_len]

        results = []
        for i in range(len(texts)):
            seq_len = int(mask[i].sum().item())
            # Берём только реальные токены (без padding)
            vecs = token_embeddings[i, :seq_len, :].cpu().tolist()
            results.append(vecs)
        return results

    def log_message(self, format, *args):
        pass  # тихий лог, чтоб не спамить


if __name__ == "__main__":
    load_models()
    server = HTTPServer(("0.0.0.0", 8082), Handler)
    logger.info("Сервер запущен на 0.0.0.0:8082")
    server.serve_forever()
