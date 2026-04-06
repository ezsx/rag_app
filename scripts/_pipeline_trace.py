"""Детальный trace одного запроса через весь production pipeline.

Показывает каждый шаг: что вошло, что вышло, почему именно столько документов,
как работают фильтры, merge, CE, compose_context.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
# Force localhost для standalone запуска (не Docker networking)
os.environ["QDRANT_URL"] = "http://localhost:16333"
os.environ["EMBEDDING_TEI_URL"] = "http://localhost:8082"
os.environ["RERANKER_TEI_URL"] = "http://localhost:8082"
os.environ["QDRANT_COLLECTION"] = "news_colbert_v2"
os.environ["LLM_BASE_URL"] = "http://localhost:8080"

from core.deps import get_hybrid_retriever, get_query_planner, get_reranker
from core.settings import get_settings

settings = get_settings()
hybrid = get_hybrid_retriever()
planner = get_query_planner()
reranker = get_reranker()

QUERY = sys.argv[1] if len(sys.argv) > 1 else "Что известно про скандал Anthropic с Пентагоном?"

print(f"{'='*80}")
print("PIPELINE TRACE")
print(f"{'='*80}")
print(f"Query: {QUERY}")
print()

# ══════════════════════════════════════════════════════════════════════
# STEP 1: QUERY PLAN
# ══════════════════════════════════════════════════════════════════════
print(f"{'─'*80}")
print("STEP 1: QUERY PLAN (Qwen3.5-35B)")
print(f"{'─'*80}")
print()
print("Что происходит: LLM получает запрос пользователя и генерирует:")
print("  - 3-6 subqueries (перефразировки для расширения поиска)")
print("  - must_phrases (обязательные термины для BM25)")
print("  - should_phrases (желательные термины)")
print("  - metadata_filters (date/channel если есть)")
print("  - strategy (broad/temporal/channel/entity)")
print()

plan = planner.make_plan(QUERY)

print(f"Strategy: {plan.strategy}")
print(f"k_per_query: {plan.k_per_query}")
print()
print(f"Subqueries ({len(plan.normalized_queries)}):")
for i, sq in enumerate(plan.normalized_queries):
    print(f"  [{i+1}] \"{sq}\"")
print()
print(f"Must phrases: {plan.must_phrases}")
print("  → Эти слова ДОЛЖНЫ быть в документе (BM25 must clause)")
print(f"Should phrases: {plan.should_phrases}")
print("  → Эти слова ЖЕЛАТЕЛЬНЫ, повышают score (BM25 should clause)")
print(f"Metadata filters: {plan.metadata_filters}")
if plan.metadata_filters:
    print("  → Qdrant фильтрует по payload ДО векторного поиска")
else:
    print("  → Нет фильтров — ищем по всей коллекции (13,777 docs)")
print()

# Original query injection
all_queries = [QUERY] + [q for q in plan.normalized_queries if q.lower() != QUERY.lower()]
print("Original query injection: оригинал ставится ПЕРВЫМ")
print(f"Итого queries для поиска: {len(all_queries)}")
for i, q in enumerate(all_queries):
    label = "ORIGINAL" if i == 0 else f"subquery-{i}"
    print(f"  [{label}] \"{q}\"")
print()

# ══════════════════════════════════════════════════════════════════════
# STEP 2: PER-QUERY HYBRID SEARCH
# ══════════════════════════════════════════════════════════════════════
print(f"{'─'*80}")
print("STEP 2: HYBRID SEARCH (per query)")
print(f"{'─'*80}")
print()
print("Каждый query проходит ОДИНАКОВЫЙ pipeline внутри HybridRetriever:")
print(f"  1. Dense embedding (pplx-embed, без prefix) → Qdrant dense search top-{max(plan.k_per_query*4, 40)}")
print(f"  2. BM25 sparse (fastembed + R2 lexicon normalization) → Qdrant sparse search top-{max(plan.k_per_query*10, 100)}")
print(f"  3. RRF fusion [1.0, 3.0] (Dense:BM25=1:3) → top-{max(plan.k_per_query*5, 50)}")
print(f"  4. ColBERT MaxSim rerank (jina-colbert-v2) → top-{plan.k_per_query*2}")
print("  5. Channel dedup (max 2 docs per channel)")
print()
print("Почему именно эти числа:")
print("  - Dense top-40: ablation показал что 40 лучше 20 (+3.4% R@5)")
print("  - BM25 top-100: больше не помогает (saturation)")
print("  - RRF top-50: объединяет dense и BM25 кандидатов, BM25 вес ×3")
print("  - ColBERT top-20: финальный rerank по token-level similarity")
print("  - Dedup: макс 2 поста с одного канала для разнообразия")
print()

per_query_results = []
all_candidates_count = 0

for i, q in enumerate(all_queries):
    label = "ORIGINAL" if i == 0 else f"subquery-{i}"
    candidates = hybrid.search_with_plan(q, plan)
    per_query_results.append(candidates)
    all_candidates_count += len(candidates)

    print(f"  [{label}] \"{q[:60]}\"")
    print(f"    → {len(candidates)} docs после ColBERT + dedup")

    # Показываем top-3 с деталями
    for j, c in enumerate(candidates[:3]):
        ch = c.metadata.get('channel', '?')
        mid = c.metadata.get('message_id', 0)
        text = c.text[:80].replace('\n', ' ')
        print(f"      #{j+1} {ch}:{mid} | {text}...")
    if len(candidates) > 3:
        print(f"      ... и ещё {len(candidates)-3} docs")
    print()

print(f"Итого до merge: {all_candidates_count} docs (с дублями между queries)")
print()

# ══════════════════════════════════════════════════════════════════════
# STEP 3: ROUND-ROBIN MERGE
# ══════════════════════════════════════════════════════════════════════
print(f"{'─'*80}")
print("STEP 3: ROUND-ROBIN MERGE")
print(f"{'─'*80}")
print()
print("Как работает merge:")
print("  Берём rank-1 от каждого query, потом rank-2 от каждого, и т.д.")
print("  Дедупликация: если doc уже добавлен (от другого query), пропускаем.")
print("  Порядок определяется ПОЗИЦИЕЙ в каждом query, не score.")
print()

merged = []
seen = set()
max_len = max(len(r) for r in per_query_results)
merge_log = []

for rank in range(max_len):
    for qi, sub in enumerate(per_query_results):
        if rank < len(sub):
            c = sub[rank]
            label = "ORIGINAL" if qi == 0 else f"sq-{qi}"
            if c.id not in seen:
                merged.append(c)
                seen.add(c.id)
                merge_log.append((len(merged), label, rank+1, c.metadata.get('channel','?'), c.metadata.get('message_id',0)))

k_total = min(plan.k_per_query * len(all_queries), 30)
merged = merged[:k_total]

print(f"k_total cap = min({plan.k_per_query} × {len(all_queries)}, 30) = {k_total}")
print(f"Unique docs после merge: {len(merged)}")
print()
print("Порядок merge (первые 15):")
print(f"  {'#':>3s} {'Source':>10s} {'Rank':>5s} {'Channel':>30s} {'MsgID':>8s}")
for pos, src, rank, ch, mid in merge_log[:15]:
    print(f"  {pos:3d} {src:>10s} rank={rank:<3d} {ch:>30s} {mid:>8d}")
if len(merge_log) > 15:
    print(f"  ... и ещё {len(merge_log)-15}")
print()
print("Проблема round-robin: doc с ColBERT score 0.95 от original")
print("и doc с score 0.30 от subquery-3 чередуются одинаково по позиции.")
print("Score не учитывается — только rank position.")
print()

# ══════════════════════════════════════════════════════════════════════
# STEP 4: CE FILTER
# ══════════════════════════════════════════════════════════════════════
print(f"{'─'*80}")
print("STEP 4: CE FILTER (Qwen3-Reranker-0.6B, threshold=0.0)")
print(f"{'─'*80}")
print()
print("Что происходит: cross-encoder оценивает пару (query, doc_text)")
print("для КАЖДОГО из merged docs. Возвращает raw logit score.")
print("  score > 0 → relevant (positive logit)")
print("  score < 0 → not relevant (negative logit)")
print("  threshold=0.0 = decision boundary")
print()
print("CE НЕ меняет порядок — сохраняет ColBERT ranking.")
print("Только убирает docs с score < threshold.")
print()

docs_text = [c.text for c in merged]
indices, scores = reranker.rerank_with_raw_scores(
    QUERY, docs_text, top_n=min(len(docs_text), 80)
)

# Восстановим score для каждого doc в merged порядке
score_by_idx = {}
for idx, score in zip(indices, scores):
    score_by_idx[idx] = score

passed_indices = set()
filtered_docs = []
kept_docs = []

for idx, score in zip(indices, scores):
    ch = merged[idx].metadata.get('channel', '?')
    mid = merged[idx].metadata.get('message_id', 0)
    if score >= 0.0:
        passed_indices.add(idx)
        kept_docs.append((idx, score, ch, mid))
    else:
        filtered_docs.append((idx, score, ch, mid))

print(f"Input: {len(merged)} docs")
print(f"Kept (score >= 0.0): {len(kept_docs)}")
print(f"Filtered (score < 0.0): {len(filtered_docs)}")
print()

if filtered_docs:
    print("Отфильтрованные (score < 0):")
    for idx, score, ch, mid in filtered_docs:
        text = merged[idx].text[:60].replace('\n', ' ')
        print(f"  score={score:+6.2f}  {ch}:{mid} | {text}...")
    print()

# Результат: ColBERT порядок сохранён, убраны только filtered
final_docs = [c for i, c in enumerate(merged) if i in passed_indices]
print(f"После CE: {len(final_docs)} docs (ColBERT порядок сохранён)")
print()

# ══════════════════════════════════════════════════════════════════════
# STEP 5: COMPOSE_CONTEXT
# ══════════════════════════════════════════════════════════════════════
print(f"{'─'*80}")
print("STEP 5: COMPOSE_CONTEXT (budget=4000 tokens ≈ 16000 chars)")
print(f"{'─'*80}")
print()
print("Что происходит: берёт docs в текущем порядке и добавляет")
print("в контекст пока не кончится бюджет символов.")
print("Каждый doc форматируется как: [N] (channel, date): text...")
print("Этот контекст отправляется LLM для генерации final_answer.")
print()

max_chars = 4000 * 4  # 1 token ≈ 4 chars
used = 0
included = []

for i, c in enumerate(final_docs):
    text_len = len(c.text)
    if used + text_len > max_chars:
        remaining = max_chars - used
        if remaining > 200:  # truncate last
            included.append((i, c, remaining))
            used += remaining
        break
    included.append((i, c, text_len))
    used += text_len

print(f"Доступно: {len(final_docs)} docs")
print(f"В контекст вошло: {len(included)} docs ({used} chars из {max_chars} бюджета)")
print(f"Не вошло: {len(final_docs) - len(included)} docs (бюджет исчерпан)")
print()

print("Документы в контексте LLM:")
for pos, (idx, c, chars_used) in enumerate(included):
    ch = c.metadata.get('channel', '?')
    mid = c.metadata.get('message_id', 0)
    date = str(c.metadata.get('date', ''))[:10]
    ce_score = score_by_idx.get(idx, 0)
    text = c.text[:100].replace('\n', ' ')
    trunc = " (truncated)" if chars_used < len(c.text) else ""
    print(f"  [{pos+1}] {ch}:{mid} ({date}) CE={ce_score:+.1f} | {chars_used} chars{trunc}")
    print(f"      {text}...")
print()

print(f"{'─'*80}")
print("STEP 6: LLM FINAL_ANSWER")
print(f"{'─'*80}")
print()
print("LLM (Qwen3.5-35B) получает:")
print("  - System prompt с инструкциями")
print(f"  - Контекст: {len(included)} docs ({used} chars)")
print(f"  - Оригинальный запрос: \"{QUERY}\"")
print("  - Генерирует ответ с citations [1][2][3]...")
print()
print(f"{'='*80}")
print("END OF TRACE")
print(f"{'='*80}")
