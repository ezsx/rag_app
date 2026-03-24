# Manual Judge: оценка 25 ответов агента (eval_golden_v1, прогон 20260324-193000)

## Задача

Ты — reviewer / manual LLM judge. Прочитай 25 пар (вопрос + ответ агента) и оцени каждый по двум критериям:

### Критерий 1: Factual Correctness (0.0 / 0.5 / 1.0)
- **0.0** — фактические ошибки, противоречит expected answer, или должен был отказать но ответил
- **0.5** — частично корректен, упускает важные факты, или нашёл альтернативные но валидные данные
- **1.0** — фактически корректен, соответствует expected answer

### Критерий 2: Usefulness (0 / 1 / 2)
- **0** — бесполезный: пустой, только ссылки без текста, ошибка, или ложный ответ на unanswerable вопрос
- **1** — частично полезный: некоторая информация есть, но неполный или поверхностный
- **2** — полезный: полностью отвечает, конкретен, хорошо структурирован

### Формат ответа
Таблица:
| Q | Factual | Useful | Примечание |

Потом:
- Сводка (mean factual, mean useful)
- Топ-3 лучших ответа
- Топ-3 худших
- Общий вердикт

---

## Контекст

Это RAG-агент с 11 tools, Qwen3-30B-A3B, поиск по 36 Telegram-каналам AI/ML (13088 постов, июль 2025 — март 2026).

Recall@5 = 0.408 (strict document matching, заведомо занижен для compare/summarize).
Key Tool Accuracy = 0.955.

**Важно**: recall@5 считает по exact post_ids, а агент часто находит другие валидные посты. Оценивай quality ответа, не document match.

---

## 25 ответов агента

(ниже — полные данные из прогона)

### q01 (broad_search, easy)
**Q**: Кого Financial Times назвала человеком года в 2025?
**Expected**: Financial Times назвала Дженсена Хуанга (CEO NVIDIA) Человеком года в 2025.
**Tools**: search, related_posts, rerank, compose_context
**Recall**: 1.0
**Answer**: Дженсен Хуанг, основатель и CEO компании NVIDIA, был назван человеком года Financial Times в 2025 году [1]. Во время своей церемонии награждения FT отметила ключевую роль Хуанга в трансформации полупроводниковой индустрии и глобальном распространении ИИ. NVIDIA оказалась в центре беспрецедентной инвестиционной программы частного сектора, что позволило ей первой в мире преодолеть рубеж рыночной капитализации в $5 трлн.

### q02 (broad_search, medium)
**Q**: Какие параметры у open-source моделей GPT от OpenAI?
**Expected**: OpenAI выпустила две open-source модели под Apache 2.0: 120B (117 млрд, 5.1 млрд активных, MoE) и 20B (21 млрд, 3.6 млрд активных, MoE). 4-бит mxfp4.
**Tools**: query_plan, search, rerank, compose_context
**Recall**: 1.0
**Answer**: OpenAI представила две open-source модели: GPT-OSS-120B и GPT-OSS-20B. GPT-OSS-120B: 117B параметров, одна H100 (80GB), MoE с 4-битной квантизацией (MXFP4), до 128K токенов контекста. Заточены на Chain-of-Thought рассуждениях.

### q03 (broad_search, easy)
**Q**: За сколько Meta купила Manus AI?
**Expected**: Meta купила Manus AI за $2 млрд.
**Tools**: search, rerank
**Recall**: 0.0 (нашёл другие посты)
**Answer**: Meta приобрела стартап Manus AI за $2 миллиарда. Подтверждается из ai_newz [1], aioftheday [2], data_secrets [3].

### q04 (broad_search, medium)
**Q**: Что нового у DeepSeek? Какие модели они выпустили?
**Expected**: DeepSeek V3 с mHC (Manifold-Constrained Hyper-Connections).
**Tools**: query_plan, search, rerank, compose_context
**Recall**: 0.0
**Answer**: DeepSeek-V3.2 и DeepSeek-V3.2-Speciale. V3.2 соответствует уровню GPT-5, Speciale — Gemini 3.0. Модели для агентов, долгий reasoning.

### q05 (broad_search, medium)
**Q**: Какие open-source модели для генерации изображений существуют?
**Expected**: FLUX.2 от Black Forest Labs и Kandinsky 5.0 от Сбера.
**Tools**: query_plan, search, rerank, compose_context
**Recall**: 0.0
**Answer**: HunyuanImage 3.0 от Tencent — 80B параметров, 13B активных. MoE + Transfusion. (Нашёл другие валидные модели, но не FLUX/Kandinsky)

### q06 (constrained_search, medium)
**Q**: Что обсуждалось в AI-каналах в январе 2026?
**Expected**: Агенты для ИИ-исследований в gonzo_ml; Gemini 3 Flash Preview any2json в boris_again.
**Tools**: query_plan, temporal_search, rerank, compose_context
**Recall**: 0.5
**Answer**: Источники: [1] [2] [3] [4] [5] (ТОЛЬКО ссылки, без текста ответа!)

### q07 (constrained_search, easy)
**Q**: Что было на GTC 2026 от NVIDIA?
**Expected**: Платформа Vera Rubin с GPU нового поколения.
**Tools**: query_plan, search, rerank, compose_context
**Recall**: 1.0
**Answer**: Платформа Vera Rubin на 3-нм TSMC, 336 млрд транзисторов, 288 ГБ HBM4. NVL72 — 260 ТБ/с NVLink 6. Инференс в 5 раз быстрее Blackwell (NVFP4). CPU Vera собственной разработки.

### q08 (constrained_search, medium)
**Q**: Какие AI-события произошли в феврале 2026?
**Expected**: Opus 4.6, Себрант AI-тренды, Agibot роботы в Шанхае.
**Tools**: query_plan, temporal_search, rerank, compose_context
**Recall**: 0.5
**Answer**: Agibot шоу роботов в Шанхае [1]. Alphabet облигация 100 лет [4]. (Opus 4.6 не упомянут)

### q09 (constrained_search, medium)
**Q**: Что писал llm_under_hood про отключение reasoning у GPT-5?
**Expected**: Developer role инструкция, Juice — интенсивность ризонера.
**Tools**: channel_search, rerank, compose_context
**Recall**: 1.0
**Answer**: developer role инструкция: "Active channels: final / Disabled channels: analysis, commentary / Juice: 0 !important". gpt-5-mini с дефолтовым reasoning — 28 секунд.

### q10 (constrained_search, medium)
**Q**: О чём писал gonzo_ml про трансформеры и рекуррентные сети?
**Expected**: AlphaGenome от DeepMind, ограничения RNN vs трансформеры.
**Tools**: channel_search, rerank, compose_context
**Recall**: 0.5
**Answer**: HRM (Hierarchical Recurrent Memory), deep supervision. Сравнение RNN и трансформеров. (AlphaGenome не упомянут)

### q11 (constrained_search, easy)
**Q**: Что Борис Цейтлин думает о Claude Opus 4.6?
**Expected**: Самый близкий к автономной модели.
**Tools**: query_plan, channel_search, rerank, compose_context
**Recall**: 1.0
**Answer**: Борис оценивает Opus 4.6 как прорыв в длинном контексте. MRCR v2 8-needle. На 17 пунктов лучше Opus 4.5, на 82 лучше Sonnet 4.5. Превзошёл GPT-5.2 и Gemini 3 Pro.

### q12 (constrained_search, easy)
**Q**: Что seeallochnaya писал про GPT-5?
**Expected**: Выход GPT-5.3 и GPT-5.4 с интервалом в два дня.
**Tools**: channel_search, rerank, compose_context, search, compose_context
**Recall**: 0.0
**Answer**: Sama упоминал GPT-5 дважды за 4 дня. Автор тестировал GPT-5, ответила идеально на сложный вопрос. (Про 5.3/5.4 конкретно не написал, но про GPT-5 в целом содержательно)

### q13 (compare_summarize, hard)
**Q**: Как разные каналы обсуждали покупку Manus компанией Meta?
**Expected**: ai_newz (разбор), aioftheday (краткая), data_secrets (хайп).
**Tools**: query_plan, cross_channel_compare, rerank, compose_context
**Recall**: 0.33
**Answer**: data_secrets — связи с Китаем разорваны, из Сингапура, $2 млрд. Три канала обсуждали с разных углов.

### q14 (compare_summarize, hard)
**Q**: Сравни как разные каналы обсуждали DeepSeek — какие мнения?
**Expected**: seeallochnaya — V3/mHC, ai_ml_big_data — безопасность, другие — релизы.
**Tools**: query_plan, cross_channel_compare, rerank, compose_context
**Recall**: 0.0
**Answer**: ai_ml_big_data — DeepSeek-OCR 2, DeepEncoder V2. Технические характеристики. (Не нашёл V3/mHC и безопасность, но про DeepSeek содержательно)

### q15 (compare_summarize, easy)
**Q**: Что нового в канале techsparks за последнюю неделю?
**Expected**: Себрант — AI-тренды, роботакси, маркетинг.
**Tools**: summarize_channel
**Recall**: 0.0
**Answer**: Только 5 ссылок на посты (https://t.me/techsparks/5476-5480) без текста! BUG — summarize flow не генерирует текстовый ответ.

### q16 (compare_summarize, medium)
**Q**: Дай дайджест канала gonzo_ml за последний месяц
**Expected**: Трансформеры, рекуррентные сети, AlphaGenome, ML-темы.
**Tools**: summarize_channel
**Recall**: 0.0
**Answer**: Claude Sonnet 4.6 vs Gemini 3.1 Pro, Categorical Flow Maps. (Частичный дайджест, но не через citation pipeline)

### q17 (navigation, easy)
**Q**: Какие каналы у вас есть?
**Expected**: 36 каналов с количеством постов.
**Tools**: list_channels
**Recall**: N/A
**Answer**: 36 каналов, полный список по именам. (Без количества постов — minor)

### q18 (navigation, easy)
**Q**: Сколько постов в канале llm_under_hood?
**Expected**: 261 пост.
**Tools**: list_channels
**Recall**: N/A
**Answer**: 261 пост [1]. Точно.

### q19 (negative_refusal, easy)
**Q**: Существует ли модель GPT-7?
**Expected**: (должен отказать)
**Tools**: search, rerank, compose_context
**Recall**: N/A
**Failure**: refusal_wrong
**Answer**: Не существует GPT-7, но нашёл пост "GPT-7 может стать Гением по требованию" и цитирует как доказательство. Должен был просто сказать "нет данных".

### q20 (negative_refusal, easy)
**Q**: Выходила ли модель Bard 3 от Google?
**Expected**: (должен отказать)
**Tools**: query_plan, search, rerank, compose_context
**Recall**: N/A
**Failure**: refusal_wrong
**Answer**: "Bard 3 не упоминается, но вот Gemini 3 Flash и Gemma 3". Подменяет вопрос. Должен был отказать.

### q21 (negative_refusal, medium)
**Q**: Что обсуждалось в AI-каналах в апреле 2024?
**Expected**: (должен отказать — вне периода данных)
**Tools**: temporal_search, rerank, compose_context
**Recall**: N/A
**Failure**: refusal_wrong
**Answer**: Детальный ответ про апрель 2024 с цитатами. Но корпус начинается с июля 2025! Галлюцинация через grounding — нашёл посты УПОМИНАЮЩИЕ 2024 год, но не ИЗ 2024. Самый опасный failure.

### q22 (future_baseline, hard)
**Q**: Как менялось обсуждение OpenAI в каналах за последние 3 месяца?
**Expected**: GPT-5.3/5.4, open-source, конкуренция.
**Tools**: query_plan, temporal_search, rerank, compose_context
**Recall**: 0.5
**Answer**: Карпаты про AI-агентов, новые аудиомодели. (Поверхностно, не про 5.3/5.4)

### q23 (future_baseline, hard)
**Q**: Какие arxiv-статьи больше всего обсуждались в каналах?
**Expected**: DeepSeek-V3, Mamba 3, SAM 3.
**Tools**: query_plan, search, rerank, compose_context
**Recall**: 0.0
**Answer**: "Does RL Really Incentivize Reasoning?", "Frontier LLMs Still Struggle". (Нашёл другие статьи — валидные, но не те что expected)

### q24 (future_baseline, medium)
**Q**: Какой канал специализируется на робототехнике и роботакси?
**Expected**: techsparks (Себрант).
**Tools**: search, rerank, compose_context, search, compose_context
**Recall**: 0.5
**Answer**: @papa_robotov (Ваня Калинов, Яндекс Роботикс). Также @techno_yandex. (Нашёл другой валидный канал!)

### q25 (broad_search, hard)
**Q**: Какие подходы к LLM в production обсуждались в llm_under_hood и boris_again?
**Expected**: SGR для PDF в llm_under_hood; any2json в boris_again.
**Tools**: query_plan, channel_search, related_posts, rerank, compose_context
**Recall**: 0.33
**Failure**: tool_selected_wrong
**Answer**: SGR подробно описан. Но boris_again не покрыт — агент ушёл в channel_search только для llm_under_hood.
