# RAG Evaluation: от скрипта к production-grade системе

**Рекомендуемый подход — гибридный: custom LLM-judge на Qwen3-8B как основа, DeepEval как CI/CD-фреймворк, RAGAS-метрики для бенчмаркинга.** Чистый custom judge даёт максимальный контроль над русскоязычными промптами и не зависит от breaking changes фреймворков. DeepEval обеспечивает pytest-интеграцию и структуру для CI/CD. RAGAS полезен для разовых аудитов с его формальными метриками (faithfulness через NLI-декомпозицию, answer relevancy через reverse-question embedding). При локальном инференсе на Qwen3-8B все три подхода работоспособны, но custom judge — единственный, где вы полностью контролируете промпты на русском языке.

---

## Обоснованный выбор: почему гибрид, а не один инструмент

Каждый из трёх инструментов решает свою задачу лучше остальных, и ни один не покрывает все потребности целиком.

**RAGAS** (v0.4.3, январь 2026) предлагает формально обоснованные метрики: faithfulness разбивает ответ на атомарные claims и верифицирует каждый через NLI-промпт, answer_relevancy генерирует гипотетические вопросы из ответа и сравнивает embeddings с оригиналом, context_recall проверяет покрытие ground_truth контекстом. Но RAGAS прошёл **два крупных breaking change** за 2024-2025 (v0.1→v0.2: переход на `SingleTurnSample`; v0.3→v0.4: `ground_truths` → `reference`, новый `collections`-based API). Промпты внутри RAGAS — на английском, без встроенной русскоязычной адаптации. При использовании с Qwen3-8B через vLLM часты NaN-скоры из-за невалидного JSON.

**DeepEval** (v3.7.6, декабрь 2025) — инженерный фреймворк с нативной поддержкой vLLM (`deepeval set-local-model`), **50+ метрик**, pytest-интеграцией и GEval для custom-критериев. Архитектурно это «pytest для LLM» — каждый eval-кейс это тест, результаты агрегируются в отчёт. DeepEval также не имеет встроенной мультиязычной поддержки, но позволяет переопределять `evaluation_template` для метрик и создавать полностью кастомные метрики через `BaseMetric`.

**Custom LLM-judge** даёт полный контроль: промпты на русском, structured JSON output, точная настройка критериев под вашу доменную область. Исследования подтверждают жизнеспособность: Prometheus-2 (7B) достигает **0.897 Pearson correlation** с человеческими оценками (GPT-4: 0.882), а панель из нескольких малых моделей (PoLL) превосходит single GPT-4 по Cohen's κ (**0.763 vs 0.627**).

| Критерий | RAGAS | DeepEval | Custom Judge |
|----------|-------|----------|--------------|
| Контроль промптов на русском | ⚠️ Частичный | ⚠️ Через GEval/templates | ✅ Полный |
| CI/CD интеграция | ❌ Скриптовый | ✅ pytest native | ⚠️ Ручная |
| Стабильность API | ❌ 2 breaking changes/год | ✅ Стабильный | ✅ Ваш код |
| Локальный LLM (vLLM) | ✅ `llm_factory` + `instructor.Mode.MD_JSON` | ✅ CLI `set-local-model` | ✅ OpenAI client |
| Метрики без LLM | ⚠️ `NonLLMContextRecall` | ⚠️ `BaseMetric` | ✅ Любые |
| Формальная обоснованность | ✅ NLI-based | ✅ G-Eval (log-prob) | ⚠️ Зависит от промпта |

**Итоговая архитектура**: custom judge промпты (faithfulness, relevance, completeness, citation accuracy) → обёрнуты в DeepEval `BaseMetric` для pytest/CI → RAGAS-метрики запускаются параллельно для бенчмаркинга.

---

## Метрики: расширенный план для evaluate_agent.py

Текущий скрипт считает recall@5, coverage, latency. Расширенный план добавляет **три уровня** метрик.

**Уровень 1 — Retrieval (без LLM-judge):** recall@5 (уже есть), **precision@5** (доля релевантных среди извлечённых), **MRR** (Mean Reciprocal Rank — на какой позиции первый релевантный документ), **NDCG@5** (учитывает порядок). Все считаются по `expected_document_ids` из eval-датасета, не требуют LLM.

**Уровень 2 — Generation quality (LLM-judge):** **faithfulness** (доля claims, подтверждённых контекстом; binary per claim → агрегация), **relevance** (1-5 шкала, насколько ответ соответствует вопросу), **completeness** (1-5 шкала, покрытие всех аспектов вопроса), **citation accuracy** (доля корректных ссылок на источники). Для faithfulness и citation accuracy лучше **binary pass/fail** с QAG-декомпозицией (разбиение на атомарные проверки), для relevance и completeness — **шкала 1-5 с чёткой рубрикой**.

**Уровень 3 — System-level:** latency P50/P95/P99 (уже частично есть), **answer rate** (доля вопросов, на которые система дала ответ vs отказала), **negative rejection rate** (правильно ли система отказывает на unanswerable-вопросы), **cost per eval** (токены × стоимость).

```python
# Расширенная структура метрик в evaluate_agent.py
METRICS_CONFIG = {
    "retrieval": {
        "recall@5": {"requires_llm": False, "threshold": 0.70},
        "precision@5": {"requires_llm": False, "threshold": 0.50},
        "mrr": {"requires_llm": False, "threshold": 0.60},
        "ndcg@5": {"requires_llm": False, "threshold": 0.55},
    },
    "generation": {
        "faithfulness": {"requires_llm": True, "scale": "binary_qag", "threshold": 0.80},
        "relevance": {"requires_llm": True, "scale": "1-5", "threshold": 3.0},
        "completeness": {"requires_llm": True, "scale": "1-5", "threshold": 3.0},
        "citation_accuracy": {"requires_llm": True, "scale": "binary_qag", "threshold": 0.90},
    },
    "system": {
        "latency_p95_ms": {"requires_llm": False, "threshold": 5000},
        "answer_rate": {"requires_llm": False, "threshold": 0.85},
        "negative_rejection_rate": {"requires_llm": False, "threshold": 0.70},
    },
}
```

---

## Промпты судьи для Qwen3-8B на русском

Ключевой принцип из литературы (MT-Bench, Databricks, EvidentlyAI): **один промпт — один критерий**, Chain-of-Thought перед оценкой, structured JSON output. Для 8B-модели **binary + QAG-декомпозиция** надёжнее, чем прямая 5-балльная оценка.

### Faithfulness (верность контексту)

```
Ты — беспристрастный судья, проверяющий верность ответа RAG-системы контексту.

## Входные данные
Вопрос: {question}
Контекст: {context}
Ответ: {answer}

## Инструкция
1. Разбей ответ на отдельные атомарные утверждения (claims).
2. Для каждого утверждения определи: подтверждается ли оно контекстом?
3. Ответ СТРОГО в JSON без дополнительного текста.

## Формат ответа
{
  "claims": [
    {"claim": "текст утверждения", "supported": true, "evidence": "цитата или null"}
  ],
  "supported_claims": <число>,
  "total_claims": <число>,
  "score": <0.0-1.0>,
  "verdict": "pass или fail"
}

verdict = "pass" если score >= 0.8.
```

### Relevance (релевантность ответа)

```
Ты — беспристрастный судья, оценивающий релевантность ответа вопросу.

## Входные данные
Вопрос: {question}
Ответ: {answer}

## Шкала
5 — полностью отвечает на вопрос, без лишнего
4 — в основном отвечает, незначительные отступления
3 — частично отвечает, есть заметные пробелы или лишнее
2 — косвенно связан с вопросом
1 — не имеет отношения к вопросу

## Инструкция
Сначала определи намерение пользователя, затем оцени. Длина ответа НЕ влияет на оценку.
Ответ СТРОГО в JSON:

{
  "intent": "намерение пользователя",
  "reasoning": "анализ",
  "score": <1-5>,
  "verdict": "pass или fail"
}

verdict = "pass" если score >= 3.
```

### Completeness (полнота) и Citation accuracy (точность ссылок)

Промпты для completeness и citation accuracy строятся по аналогичному шаблону. Completeness получает на вход вопрос, контекст и ответ, разбивает вопрос на аспекты и проверяет покрытие каждого (шкала 1-5). Citation accuracy получает ответ с ссылками и документы, проверяет каждую ссылку бинарно (source_exists × claim_supported) и агрегирует в score = accurate/total.

Для **обеспечения детерминированности** при работе с Qwen3-8B через vLLM: `temperature=0`, `top_p=1`, `seed=42`, guided decoding через Outlines (`--guided-decoding-backend outlines`). Thinking mode рекомендуется **отключить** для judging — исследования (JudgeBoard, 2025) показывают inconsistent gains для evaluation-задач.

---

## Генератор eval-датасета из Qdrant

Пайплайн: scroll точек из коллекции → извлечение payload → генерация вопросов с принудительным распределением типов → фильтрация через critique-агента → сохранение в `eval_dataset.json`.

```python
#!/usr/bin/env python3
"""generate_eval_dataset.py — генерация eval-датасета из Qdrant."""

import json, random, asyncio
from qdrant_client import QdrantClient
from openai import OpenAI

COLLECTION = "knowledge_base"
VLLM_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3-8B"
TARGET_SIZE = 250  # генерируем с запасом, ~50% отсеется

qdrant = QdrantClient("localhost", port=6333)
llm = OpenAI(base_url=VLLM_URL, api_key="not-needed")

# Принудительное распределение типов (без этого 95% будут factual)
QUESTION_TYPES = {
    "factual":     {"weight": 0.35, "prompt": "Сформулируй фактологический вопрос, ответ на который — конкретный факт из контекста."},
    "temporal":    {"weight": 0.20, "prompt": "Сформулируй вопрос о датах, сроках или хронологии событий из контекста."},
    "aggregation": {"weight": 0.20, "prompt": "Сформулируй вопрос, требующий объединения информации из разных частей контекста."},
    "negative":    {"weight": 0.15, "prompt": "Сформулируй вопрос по теме контекста, на который НЕЛЬЗЯ ответить из этого контекста."},
    "comparative": {"weight": 0.10, "prompt": "Сформулируй вопрос, требующий сравнения двух сущностей из контекста."},
}

def generate_qa(context: str, q_type: str, type_prompt: str) -> dict | None:
    prompt = f"""Ты генерируешь вопросы для тестирования RAG-системы. Данные на русском.

{type_prompt}

Контекст:
{context[:2000]}

Ответь строго в JSON:
{{"question": "вопрос на русском", "expected_answer": "ожидаемый ответ", "answerable": true/false}}"""
    try:
        resp = llm.chat.completions.create(
            model=MODEL, temperature=0.7, max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None

def critique_filter(question: str, answer: str, context: str) -> bool:
    prompt = f"""Оцени качество QA-пары для eval-датасета (1-5 по каждому критерию):
1. Groundedness: можно ли ответить на вопрос из контекста?
2. Relevance: реалистичен ли вопрос для пользователя?
3. Standalone: понятен ли вопрос без контекста?

Вопрос: {question}
Ответ: {answer}
Контекст: {context[:500]}

JSON: {{"groundedness": N, "relevance": N, "standalone": N}}"""
    try:
        resp = llm.chat.completions.create(
            model=MODEL, temperature=0.1, max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        scores = json.loads(resp.choices[0].message.content)
        return all(v >= 3 for v in scores.values())
    except Exception:
        return False

def main():
    # Шаг 1: выборка из Qdrant
    points, _ = qdrant.scroll(COLLECTION, limit=500, with_payload=True, with_vectors=False)
    sampled = random.sample(points, min(TARGET_SIZE, len(points)))

    # Шаг 2: генерация с распределением типов
    types = list(QUESTION_TYPES.keys())
    weights = [QUESTION_TYPES[t]["weight"] for t in types]
    dataset = []

    for point in sampled:
        context = point.payload.get("text", point.payload.get("content", ""))
        if len(context) < 50:
            continue
        q_type = random.choices(types, weights=weights, k=1)[0]
        qa = generate_qa(context, q_type, QUESTION_TYPES[q_type]["prompt"])
        if not qa:
            continue

        # Шаг 3: critique-фильтрация
        if not critique_filter(qa["question"], qa["expected_answer"], context):
            continue

        dataset.append({
            "id": f"eval_{len(dataset):04d}",
            "question": qa["question"],
            "expected_answer": qa["expected_answer"],
            "contexts": [context],
            "expected_document_ids": [str(point.id)],
            "question_type": q_type,
            "answerable": qa.get("answerable", True),
            "metadata": {
                "qdrant_point_id": str(point.id),
                "source": point.payload.get("source", ""),
                "collection": COLLECTION,
            }
        })

    # Шаг 4: сохранение
    output = {
        "version": "1.0",
        "created_at": "2026-03-16",
        "generation_model": MODEL,
        "statistics": {
            "total": len(dataset),
            "by_type": {t: sum(1 for d in dataset if d["question_type"] == t) for t in types},
        },
        "examples": dataset,
    }
    with open("eval_dataset.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"✅ Сгенерировано {len(dataset)} примеров из {len(sampled)} точек")

if __name__ == "__main__":
    main()
```

**Критический факт**: исследование "Know Your RAG" (arXiv 2411.19710) показало, что без принудительного распределения типов **95% сгенерированных вопросов** — простые factual single-hop, что завышает метрики. Поэтому `QUESTION_TYPES` с весами — обязательный элемент.

**Qwen3-8B для генерации вопросов** достаточен при условии critique-фильтрации (~50% отсеется). Модель поддерживает 119 языков включая русский, превосходит Qwen2.5-14B на >50% бенчмарков. Для production-критичных датасетов можно генерировать через thinking mode (`enable_thinking=True`) для сложных типов (aggregation, comparative).

---

## Qwen3-8B как судья: реалистичная оценка возможностей

**Qwen3-8B достаточна для binary/3-point judgments, но ненадёжна для тонкой 5-point оценки без валидации.** Результаты бенчмарка JudgeBoard (2025) показывают, что judging ability Qwen3-8B выше её problem-solving ability, а Multi-Agent Judging (MAJ) с несколькими инстансами Qwen3-8B работает на уровне Qwen3-30B.

Подход PoLL (Panel of LLM judges, Verga et al. 2024) особенно интересен: панель из **трёх малых моделей превосходит** single GPT-4 при **7-кратно меньшей стоимости** (Cohen's κ: 0.763 vs 0.627). Если доступны несколько моделей (Qwen3-8B + Llama-3.1-8B + Gemma-2-9B), PoLL — рекомендуемый подход.

Для русскоязычных данных Qwen3 — один из лучших вариантов среди open-source моделей. Серия Qwen специально разрабатывалась с мультиязычной поддержкой, и русский входит в число приоритетных языков. Однако **необходима обязательная валидация**: разметить 30-50 примеров вручную, вычислить Cohen's κ между Qwen3-8B и экспертом, и принять решение на основе данных. Целевой κ ≥ 0.6 (substantial agreement).

---

## Минимальный working pipeline: evaluate_agent.py с LLM-judge

```python
#!/usr/bin/env python3
"""evaluate_agent.py — расширенный eval pipeline с LLM-judge."""

import json, time, asyncio, httpx, numpy as np
from dataclasses import dataclass, field
from openai import OpenAI

# ========== КОНФИГУРАЦИЯ ==========
AGENT_URL = "http://localhost:8080/v1/agent/stream"
VLLM_URL = "http://localhost:8000/v1"
JUDGE_MODEL = "Qwen/Qwen3-8B"
EVAL_DATASET = "eval_dataset.json"

judge = OpenAI(base_url=VLLM_URL, api_key="not-needed")

# ========== JUDGE ПРОМПТЫ ==========
FAITHFULNESS_PROMPT = """Ты — беспристрастный судья. Проверь верность ответа контексту.

Вопрос: {question}
Контекст: {context}
Ответ: {answer}

Разбей ответ на утверждения. Для каждого проверь: подтверждается ли контекстом?
JSON: {{"claims": [{{"claim": "...", "supported": true/false}}], "score": 0.0-1.0, "verdict": "pass/fail"}}
verdict = "pass" если score >= 0.8."""

RELEVANCE_PROMPT = """Ты — беспристрастный судья. Оцени релевантность ответа вопросу.

Вопрос: {question}
Ответ: {answer}

Шкала: 5=полностью отвечает, 4=в основном, 3=частично, 2=косвенно, 1=не по теме.
JSON: {{"reasoning": "...", "score": 1-5, "verdict": "pass/fail"}}
verdict = "pass" если score >= 3."""

# ========== ВЫЗОВ АГЕНТА ==========
def call_agent(question: str) -> dict:
    start = time.perf_counter()
    with httpx.Client(timeout=60) as client:
        resp = client.post(AGENT_URL, json={"query": question}, headers={"Accept": "text/event-stream"})
        # Парсинг SSE-ответа (упрощённо)
        answer, docs, doc_ids = "", [], []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "answer" in data: answer = data["answer"]
                if "documents" in data: docs = data["documents"]
                if "document_ids" in data: doc_ids = data["document_ids"]
    latency = (time.perf_counter() - start) * 1000
    return {"answer": answer, "documents": docs, "document_ids": doc_ids, "latency_ms": latency}

# ========== LLM-JUDGE ==========
def judge_call(prompt: str) -> dict:
    resp = judge.chat.completions.create(
        model=JUDGE_MODEL, temperature=0, seed=42, max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {"score": 0, "verdict": "fail", "error": "invalid JSON"}

def evaluate_faithfulness(question, answer, contexts) -> dict:
    context_str = "\n---\n".join(contexts)
    return judge_call(FAITHFULNESS_PROMPT.format(question=question, context=context_str, answer=answer))

def evaluate_relevance(question, answer) -> dict:
    return judge_call(RELEVANCE_PROMPT.format(question=question, answer=answer))

# ========== RETRIEVAL МЕТРИКИ ==========
def retrieval_metrics(retrieved_ids: list, expected_ids: list, k=5) -> dict:
    retrieved_set = set(retrieved_ids[:k])
    expected_set = set(expected_ids)
    hits = retrieved_set & expected_set
    recall = len(hits) / len(expected_set) if expected_set else 0
    precision = len(hits) / k if k > 0 else 0
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in expected_set:
            mrr = 1.0 / rank
            break
    return {"recall@5": recall, "precision@5": precision, "mrr": mrr}

# ========== ОСНОВНОЙ PIPELINE ==========
def run_evaluation():
    with open(EVAL_DATASET) as f:
        dataset = json.load(f)["examples"]

    results = []
    for item in dataset:
        agent_resp = call_agent(item["question"])

        # Retrieval metrics
        r_metrics = retrieval_metrics(agent_resp["document_ids"], item["expected_document_ids"])

        # LLM-judge metrics
        faith = evaluate_faithfulness(item["question"], agent_resp["answer"], agent_resp["documents"])
        relev = evaluate_relevance(item["question"], agent_resp["answer"])

        results.append({
            "id": item["id"],
            "question": item["question"],
            "question_type": item["question_type"],
            "latency_ms": agent_resp["latency_ms"],
            **r_metrics,
            "faithfulness": faith.get("score", 0),
            "faithfulness_verdict": faith.get("verdict", "fail"),
            "relevance": relev.get("score", 0),
            "relevance_verdict": relev.get("verdict", "fail"),
        })

    # Агрегация
    agg = {
        metric: float(np.mean([r[metric] for r in results]))
        for metric in ["recall@5", "precision@5", "mrr", "faithfulness", "relevance", "latency_ms"]
    }
    agg["latency_p95"] = float(np.percentile([r["latency_ms"] for r in results], 95))
    agg["faithfulness_pass_rate"] = np.mean([r["faithfulness_verdict"] == "pass" for r in results])
    agg["relevance_pass_rate"] = np.mean([r["relevance_verdict"] == "pass" for r in results])
    agg["n_examples"] = len(results)

    # Bootstrap 95% CI для ключевых метрик
    for metric in ["faithfulness", "recall@5"]:
        vals = [r[metric] for r in results]
        boots = [float(np.mean(np.random.choice(vals, len(vals)))) for _ in range(1000)]
        agg[f"{metric}_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # Сохранение
    output = {"aggregate": agg, "per_example": results}
    with open("eval_results.json", "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Markdown отчёт
    report = f"""# RAG Evaluation Report

| Metric | Value | CI 95% |
|--------|-------|--------|
| Recall@5 | {agg['recall@5']:.3f} | {agg.get('recall@5_ci95', ['—','—'])} |
| Precision@5 | {agg['precision@5']:.3f} | — |
| MRR | {agg['mrr']:.3f} | — |
| Faithfulness | {agg['faithfulness']:.3f} | {agg.get('faithfulness_ci95', ['—','—'])} |
| Faithfulness Pass Rate | {agg['faithfulness_pass_rate']:.1%} | — |
| Relevance (1-5) | {agg['relevance']:.2f} | — |
| Latency P50 | {np.median([r['latency_ms'] for r in results]):.0f}ms | — |
| Latency P95 | {agg['latency_p95']:.0f}ms | — |
| N examples | {agg['n_examples']} | — |
"""
    with open("eval_report.md", "w") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    run_evaluation()
```

---

## Настройка локального LLM для RAGAS и DeepEval

Оба фреймворка подключаются к Qwen3-8B через vLLM одинаково — через OpenAI-compatible endpoint, но с разными обёртками.

**Запуск vLLM:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --host 0.0.0.0 --port 8000 \
    --dtype auto --max-model-len 8192 \
    --guided-decoding-backend outlines  # для JSON confinement
```

**RAGAS** (v0.4+, рекомендуемый способ):
```python
from openai import OpenAI
from ragas.llms import llm_factory
import instructor

client = OpenAI(api_key="not-needed", base_url="http://localhost:8000/v1")
evaluator_llm = llm_factory(
    "Qwen/Qwen3-8B", provider="openai", client=client,
    mode=instructor.Mode.MD_JSON  # критично для vLLM — стандартный JSON mode часто не поддерживается
)
# Для Answer Relevancy — локальные embeddings:
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
)
```

**DeepEval** (способ 1 — CLI, проще всего):
```bash
deepeval set-local-model --model=Qwen/Qwen3-8B --base-url="http://localhost:8000/v1/"
# Все метрики автоматически используют локальную модель
```

**DeepEval** (способ 2 — программный, для гибкости):
```python
from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI, AsyncOpenAI

class Qwen3Judge(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
        self.async_client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    def load_model(self): return self.client
    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model="Qwen/Qwen3-8B", messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=4096)
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        resp = await self.async_client.chat.completions.create(
            model="Qwen/Qwen3-8B", messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=4096)
        return resp.choices[0].message.content

    def get_model_name(self) -> str: return "Qwen3-8B-vLLM"

# Использование:
from deepeval.metrics import FaithfulnessMetric, GEval
qwen = Qwen3Judge()
faithfulness = FaithfulnessMetric(model=qwen, threshold=0.7)
```

---

## A/B сравнение и CI/CD автоматизация

Для сравнения Baseline QA vs ReAct по одному датасету нужны **парные статистические тесты**: оба агента отвечают на одни и те же вопросы, что позволяет сравнивать попарно.

**McNemar test** — для бинарных метрик (correct/incorrect). Строит таблицу 2×2: сколько примеров оба правильно, оба неправильно, A правильно + B неправильно, и наоборот. Проверяет, значима ли разница в дискордантных парах. **Bootstrap CI** — для любых метрик, особенно при малых N: ресемплируем парные разницы 10000 раз, строим 95% доверительный интервал. Если CI не включает 0 — разница значима. **Paired t-test** — для непрерывных метрик (latency, RAGAS scores).

**Минимальный размер датасета**: при accuracy ≈ 0.80 и N=200, **margin of error ±5.5%** (95% CI). Для обнаружения разницы в 5% между агентами с power=0.8 нужно **N ≈ 200-400**. Рекомендация: **200 примеров для nightly**, **50 для smoke-тестов в PR**.

CI/CD реализуется через два GitHub Actions workflow: smoke-тест на 50 примерах при PR (15 мин, блокирует merge при accuracy < 0.70) и полный eval на 200 примерах nightly (60 мин, шлёт alert в Slack при регрессии > 5%). Quality gate проверяет абсолютные пороги (faithfulness ≥ 0.80, recall@5 ≥ 0.70) и регрессию относительно предыдущего запуска.

---

## Русскоязычные данные: практические workarounds

Основная проблема — все промпты RAGAS и DeepEval написаны на английском. При русскоязычных данных LLM-judge получает инструкции на английском и данные на русском, что снижает качество extraction (claims, aspects) и верификации.

**Три рабочих workaround в порядке приоритета:**

Первый — **GEval с русскоязычными критериями** (DeepEval). Вместо встроенных FaithfulnessMetric/AnswerRelevancyMetric использовать GEval, где criteria полностью на русском. GEval позволяет задать произвольный критерий текстом, и LLM сама сгенерирует evaluation steps. Это самый надёжный подход для русского.

Второй — **custom judge промпты** (описаны выше). Полный контроль над языком, форматом, критериями. Обёртка в DeepEval `BaseMetric` для CI/CD.

Третий — **мультиязычные embeddings** для метрик, использующих cosine similarity (Answer Relevancy в RAGAS). `intfloat/multilingual-e5-large` или `cointegrated/rubert-tiny2` вместо стандартных OpenAI embeddings.

Перевод данных на английский перед оценкой — **не рекомендуется**: теряются нюансы, растёт latency, добавляется источник ошибок. Qwen3-8B достаточно хорошо работает с русским, чтобы этот workaround не стоил усилий.

---

## Заключение: что делать прямо сейчас

Оптимальный path к production-grade eval-системе состоит из четырёх шагов. **Шаг 1** (день 1): запустить `generate_eval_dataset.py`, получить 200 отфильтрованных примеров из Qdrant. **Шаг 2** (день 2-3): добавить faithfulness и relevance judge в `evaluate_agent.py` через прямые вызовы к Qwen3-8B (custom judge), прогнать по датасету, вычислить baseline метрики с bootstrap CI. **Шаг 3** (день 4-5): обернуть judge-промпты в DeepEval `BaseMetric` + GEval, настроить pytest-запуск, добавить GitHub Actions workflow для smoke-тестов. **Шаг 4** (неделя 2): разметить вручную 30-50 примеров, вычислить Cohen's κ между Qwen3-8B-judge и экспертом, калибровать пороги.

Ключевой инсайт исследования: **custom LLM-judge с русскоязычными промптами + DeepEval как test runner — более надёжная комбинация, чем чистый RAGAS или чистый DeepEval**. RAGAS полезен как reference implementation алгоритмов (faithfulness через NLI-декомпозицию), но его нестабильный API и отсутствие мультиязычной поддержки делают его ненадёжным для production. DeepEval стабильнее и лучше для CI/CD, но его встроенные промпты оптимизированы под GPT-4 и английский. Custom judge решает обе проблемы ценой ручной работы над промптами — которую в любом случае придётся делать для русскоязычных данных.