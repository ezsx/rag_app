# Prompt 25: Growth Ceiling Analysis — куда реально расти дальше?

## Как использовать этот prompt

Сначала прочитай и используй как attached context:

- `docs/research/prompts/25-growth-ceiling-context.md`

Но **не доверяй ему слепо**. Проверь ключевые claims по репозиторию и артефактам. Если контекст-документ что-то преувеличивает, устарел или неполон, поправь это в своём ответе.

## Роль

Ты — жёсткий independent reviewer, который смотрит на `rag_app` глазами:

- senior/staff engineer, оценивающего production-readiness системы
- человека, который строил или ревьюил реальные RAG/agent продукты
- эксперта, который знает что делают Perplexity, Glean, Danswer и другие production RAG системы

Нужна не оценка "хватит ли для собеса" — нужен **честный gap analysis**: что отделяет эту систему от genuinely production-grade, и какие из этих gap'ов реально стоит закрывать.

## Задача

Проанализируй текущее состояние проекта `rag_app` и ответь на главный вопрос:

> Что конкретно отделяет эту систему от production-grade quality? Какие практики, паттерны, capabilities есть у реальных production RAG/agent систем, которых здесь нет? Что из этого реально стоит реализовать, а что — overkill для данного контекста?

## Что проверить самостоятельно

Минимальный обязательный набор артефактов:

- `README.md`
- `docs/planning/project_scope.md`
- `docs/specifications/active/SPEC-RAG-16-hot-topics-channel-expertise.md`
- `docs/specifications/active/SPEC-RAG-17-production-hardening.md`
- последний актуальный eval report в `results/reports/`
- `src/services/agent_service.py`
- `src/tests/`

Если возможно, оцени не только docs, но и реальное состояние proof layer:

- что подтверждено метриками
- что утверждается, но ещё не закрыто свежим eval
- насколько тесты и docs синхронизированы с кодом

## На какие вопросы нужно ответить

### 1. Gap analysis vs production RAG/agent systems

- Сравни архитектуру и capabilities `rag_app` с реальными production RAG системами (Perplexity, Glean, Danswer/Onyx, Langdock, enterprise RAG deployments).
- Какие **архитектурные паттерны** они используют, которых здесь нет? (query understanding, intent classification, feedback loops, A/B testing, caching, conversation memory, graceful degradation, etc.)
- Какие **production practices** считаются стандартом в 2025-2026 для RAG/agent систем, но отсутствуют здесь?
- Где система **сильнее** типичных решений? (hybrid retrieval, ColBERT, custom agent без фреймворков, self-hosted)

### 2. Что реально отсутствует (substance, не cosmetics)

- **Evaluation credibility**: 30 вопросов с manual judge — это серьёзная eval story или ниже bar? Что делают production системы для evaluation?
- **Faithfulness / hallucination**: можно ли credibly утверждать что агент не галлюцинирует без NLI или эквивалентного механизма?
- **Robustness**: насколько система хрупкая? Что происходит при edge cases, adversarial inputs, out-of-domain queries?
- **Observability**: ноль structured metrics per component — это приемлемо или critical gap?
- **Multi-turn / conversation memory**: насколько это важно для production RAG системы vs nice-to-have?
- **Error recovery / graceful degradation**: что происходит когда retrieval fails, LLM timeout, Qdrant down?
- **Embedding model choice**: Qwen3-Embedding-0.6B — competitive choice или есть значительно лучшие варианты для русского языка?

### 3. Что из запланированного реально стоит делать

Из текущего backlog:
- NLI citation faithfulness (R19)
- Retrieval robustness NDR/RSR/ROR (R20)
- CRAG-lite quality-gated retrieval
- RAG necessity classifier (R21)
- Multi-turn conversation
- Observability / latency budget
- Eval expansion 30→100 Qs
- Ablation study
- GPT-4o comparison

Для каждого: это **real production gap** или **academic exercise**? Что из этого production systems реально делают?

### 4. Что мы пропустили

- Есть ли **направления, паттерны, или техники** которые не покрыты ни в проекте, ни в 21 research report, ни в scope — но которые production RAG/agent системы считают must-have?
- Есть ли **новые подходы 2025-2026** (Graph RAG, late interaction beyond ColBERT, structured generation, tool-use improvements) которые были бы high-impact?
- Есть ли **non-obvious gaps** которые hiring manager или senior engineer заметит но которые изнутри проекта не видны?

### 5. Конкретный план: что делать дальше

- Приоритизированный список из **5-10 substantive improvements** (не cosmetics, не packaging).
- Для каждого: почему это важно, что это даёт системе, effort estimate.
- Отдельно: что **точно НЕ делать** и почему (overkill, marginal impact, wrong direction).

### 6. Второй проект — нужен ли?

- Если да: какой проект **дополняет** RAG/agent опыт, а не дублирует?
- Post-training / fine-tuning? Multi-agent? Multimodal? Eval infrastructure? Something else?
- Или один глубокий завершённый проект сильнее двух средних?

## Важные рамки

- **НЕ** фокусируйся на cosmetics: UI polish, README, diagram, refactoring — это тривиально решается за полдня с AI
- **НЕ** форсируй ответ “overshoot / хватит для собеса” — это не вопрос. Вопрос: что нужно для production-grade
- **НЕ** предлагай фичи ради фич — каждая рекомендация должна быть обоснована тем, что production системы реально это делают
- **ДА** сравнивай с рыночными инструментами и реальными production deployments
- **ДА** указывай если мы пропустили общепринятые практики
- **ДА** будь жёстким — “вы этого не делаете, а должны, потому что X” лучше чем “всё хорошо, может ещё вот это”

## Формат ответа

```markdown
## Production Gap Analysis
- vs market tools (Perplexity, Glean, Danswer, etc.)
- vs best practices 2025-2026
- vs state of the art techniques

## What's Strong (keep)
- что уже на production level
- unique advantages

## What's Missing (gaps)
- critical gaps
- important but not critical
- nice-to-have

## Backlog Assessment
- для каждого запланированного item: real gap vs academic exercise

## What We Didn't Think Of
- blind spots
- industry patterns we missed

## Concrete Plan
1. ... (with justification from production practice)
2. ...

## What Not To Do
- ... (with justification why it's overkill)

## Second Project
- нужен / не нужен
- если нужен: какой и почему
```

Ключевые claims подтверждай ссылками на repo artifacts, docs, reports, tests или code paths. Если disagree с attached context — это плюс, а не минус.
