"""
Минимальный GPU-сервер для embedding и reranking.
Без FastAPI/pydantic — только stdlib http.server + torch + transformers.

Запуск:
    source /home/ezsx/infinity-env/bin/activate
    CUDA_VISIBLE_DEVICES=0 python /mnt/c/llms/rag/rag_app/scripts/gpu_server.py
"""

import json
import logging
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gpu_server")

EMBEDDING_MODEL_PATH = "/home/tei-models/qwen3-embedding"
RERANKER_MODEL_PATH = "/home/tei-models/reranker"

emb_tokenizer = None
emb_model = None
rer_tokenizer = None
rer_model = None


def load_models():
    global emb_tokenizer, emb_model, rer_tokenizer, rer_model

    logger.info("Загрузка embedding: %s", EMBEDDING_MODEL_PATH)
    t0 = time.time()
    emb_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    emb_model = AutoModel.from_pretrained(
        EMBEDDING_MODEL_PATH, torch_dtype=torch.float16
    ).cuda().eval()
    logger.info("Embedding загружен за %.1f сек", time.time() - t0)

    logger.info("Загрузка reranker: %s", RERANKER_MODEL_PATH)
    t0 = time.time()
    rer_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
    rer_model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL_PATH, torch_dtype=torch.float16
    ).cuda().eval()
    logger.info("Reranker загружен за %.1f сек", time.time() - t0)


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
            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as e:
            logger.exception("Error in %s", self.path)
            self._send_json({"error": str(e)}, 500)

    def _embed(self, data):
        texts = data.get("inputs", [])
        if isinstance(texts, str):
            texts = [texts]
        with torch.no_grad():
            enc = emb_tokenizer(
                texts, padding=True, truncation=True, max_length=4096,
                return_tensors="pt"
            ).to("cuda")
            out = emb_model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1)
            embeddings = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
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
            "model": "qwen3-embedding-0.6b",
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    def _rerank(self, data):
        query = data.get("query", "")
        texts = data.get("texts", [])
        pairs = [[query, t] for t in texts]
        with torch.no_grad():
            enc = rer_tokenizer(
                pairs, padding=True, truncation=True, max_length=512,
                return_tensors="pt"
            ).to("cuda")
            out = rer_model(**enc)
            # AutoModelForSequenceClassification возвращает logits
            # Для cross-encoder: logits shape [batch, num_labels]
            # BGE-M3 reranker: 1 label → logits[:, 0] = relevance score
            scores = out.logits.squeeze(-1)
        results = [{"index": i, "score": float(s)} for i, s in enumerate(scores)]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def log_message(self, format, *args):
        pass  # тихий лог, чтоб не спамить


if __name__ == "__main__":
    load_models()
    server = HTTPServer(("0.0.0.0", 8082), Handler)
    logger.info("Сервер запущен на 0.0.0.0:8082")
    server.serve_forever()
