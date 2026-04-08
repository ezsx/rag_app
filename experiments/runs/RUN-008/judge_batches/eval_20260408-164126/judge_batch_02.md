# Offline Judge

Questions: 6

## Scoring Instructions

Ты — judge для RAG-системы. Оцени каждый ответ агента по метрикам ниже.
Сравнивай ответ агента с **Expected answer** и **Required claims**.
Учитывай **Documents** — это контекст, который видела LLM при генерации ответа.

### Шкалы

**factual (0.0-1.0, step 0.1)** — полнота и точность фактов:
- 1.0 — все required claims покрыты, нет фактических ошибок
- 0.8-0.9 — основные факты верны, но упущена 1-2 мелкие детали из expected
- 0.5-0.7 — часть ключевых фактов отсутствует или неточна
- 0.1-0.4 — большая часть фактов отсутствует или ошибочна
- 0.0 — ответ полностью неверен или отсутствует

**useful (0.0-2.0, step 0.1)** — полезность для пользователя:
- 2.0 — ответ полностью отвечает на вопрос, можно использовать as-is
- 1.5-1.9 — отвечает хорошо, но есть мелкие недочёты (стиль, порядок, избыточность)
- 1.0-1.4 — частично полезен, но неполный или требует существенного уточнения
- 0.5-0.9 — минимально полезен, есть зерно ответа но в целом не помогает
- 0.0 — бесполезен, не отвечает на вопрос

**evidence_support (0.0-1.0, step 0.1)** — подтверждается ли ответ документами:
- 1.0 — каждое утверждение подкреплено цитатой из documents
- 0.5 — часть утверждений не подтверждена документами
- 0.0 — ответ не опирается на предоставленные документы

**retrieval_sufficiency (0.0-1.0, step 0.1)** — достаточно ли документов:
- 1.0 — documents содержат всю информацию для полного ответа
- 0.5 — documents покрывают тему частично, для полного ответа нужно больше
- 0.0 — documents не релевантны вопросу

### Правила
- Бонусная информация сверх expected answer НЕ снижает factual
- Если факт верен но неточная формулировка — снижение 0.1, не больше
- Если ответ на другом языке чем вопрос — useful максимум 1
- Для refusal: оцени correct_refusal (0/1) вместо factual/evidence/sufficiency

---
## golden_v2_q31 — Какие горячие темы были на неделе 2026-W10?
- Eval mode: `analytics`
- Category: `analytics_hot_topics`
- Answerable: `True`
- Key tools: `hot_topics`
- Forbidden tools: `search, temporal_search`
- Tools invoked: `hot_topics, final_answer`
- Coverage: `0.0`

**Expected answer**

В неделю 2026-W10 среди горячих тем фигурировали Apple (iPhone 17, AirPods), GPT-5.4 и Codex, а также связанные обсуждения вокруг code review и безопасности.

**Required claims**

- Ответ должен использовать hot_topics для периода 2026-W10
- Ответ должен назвать минимум две реальные горячие темы недели

**Agent answer**

В неделю 2026-W10 (с 2 по 9 марта 2026 года) в российских Telegram-каналах было опубликовано 325 постов, посвященных актуальным событиям в сфере ИИ.

Наиболее горячие темы недели:
1. **GPT-5 и его новые версии**: Активно обсуждались возможности GPT-54, режима GPT-5 thinking и GPT-Plus. Тема получила 11 постов и охватила 9 каналов, включая 'addmeto', 'ai_newz' и 'techno_yandex' [1].
2. **Новые продукты Apple**: Дискуссия о MacBook, iPhone (включая iPhone 17) и AirPods продолжала набирать обороты, собрав 16 постов в 6 каналах [1].
3. **Технологические архитектуры**: Тема «hrm, kv, mlp» с фокусом на reward и selfattention заняла первое место по хит-скорингу (0.476) с 22 постами [1].
4. **Нейроинтерфейсы (BCI)**: Обсуждение Neuralink и BCI, хотя и в меньшем объеме (3 поста), оставалось актуальным для таких каналов, как 'ai_newz' и 'denissexy' [1].
5. **Безопасность кода**: Растущий интерес к прозрачности, PR, code review и security, с 6 постами в 5 каналах [1].

Топ упоминаемых компаний: OpenAI (48 упоминаний), Claude (33), Anthropic (28) и GPT-5 (22). Также внимание уделялось Google, Gemini, Qwen и Яндексу [1].

**Agent thoughts**

- _none_

**Tool observations**

- `hot_topics`: period: 2026-W10; posts: 325; summary: В этой неделе активно обсуждались темы, связанные с GPT-5 и его возможностями. Посты о GPT-54, GPT-5 thinking и GPT-Plus появились в 11 каналах, включая 'addmeto', 'ai_newz' и 'techno_yandex'. Также п
- hrm, kv, mlp (score=0.476, 22 posts, channels: ai_machinelearning_big_data,data_secrets,gonzo_ml)
- gpt54, gpt5 thinking, plus (score=0.467, 11 posts, channels: addmeto,ai_machinelearning_big_data,ai_newz)
- macbook, iphone, airpods (score=0.463, 16 posts, c
- `final_answer`: Final answer prepared (1125 chars)

**Citations (документы использованные в ответе)**

- _none_

**Offline Judge Scoring**

| Metric | Score | Reasoning |
|--------|-------|-----------|
| factual (0.0-1.0, step 0.1): полнота и точность фактов vs expected | ___ | |
| useful (0.0-2.0, step 0.1): 0=бесполезен, 1=частично, 2=полностью отвечает | ___ | |

---

## golden_v2_q32 — Что обсуждали на неделе 2026-W11? Какие были тренды?
- Eval mode: `analytics`
- Category: `analytics_hot_topics`
- Answerable: `True`
- Key tools: `hot_topics`
- Forbidden tools: `search, temporal_search`
- Tools invoked: `hot_topics, final_answer`
- Coverage: `0.0`

**Expected answer**

На неделе 2026-W11 среди заметных трендов были Codex и RFC, Eval/ground truth, Amazon AI/AWS AI, а также связанные обсуждения вокруг NeurIPS и legal topics.

**Required claims**

- Ответ должен использовать hot_topics для периода 2026-W11
- Ответ должен перечислить минимум две реальные темы из weekly digest за эту неделю

**Agent answer**

На неделе 2026-W11 (с 9 по 16 марта 2026 года) в российских Telegram-каналах, посвященных AI/ML, было опубликовано 317 постов. Основные тренды и темы обсуждений включали:

1. **Технические аспекты моделирования**: Самой горячей темой стали метрики и подходы к моделированию, связанные с HRM (Human-Robot Management?), KV (Key-Value) и MLP (Multi-Layer Perceptron). Это обсуждали в каналах MLunderhood, ai_machinelearning_big_data, cryptovalerii и других [1].

2. **Оценка моделей**: Активно обсуждались вопросы оценки моделей, правды (ground truth) и достоверности данных, хотя этот тренд был менее выражен по сравнению с техническими аспектами [1].

3. **Научные конференции**: В каналах rybolos_channel и scientific_opensource шли дискуссии о принятии статей в CS и научные конференции, такие как NEURIPS [1].

4. **Инструменты разработки**: Обсуждались темы UI, Figma и HTML, особенно в контексте интеграции с AI-инструментами [1].

5. **Крупнейшие компании и модели**: В топ упоминаний вошли:
   - Claude (32 упоминания)
   - OpenAI (26 упоминаний)
   - Anthropic (21 упоминание)
   - Google (21 упоминание)
   - NVIDIA (20 упоминаний)
   - Gemini (18 упоминаний)
   - Microsoft (10 упоминаний)
   - GPT-5 (7 упоминаний) [1].

Также обсуждались протоколы, редакции и проект aliceyandexru, а также практические аспекты применения AI в различных сферах. Наиболее активные обсуждения начались 9 марта 2026 года в 21:05 UTC [1].

**Agent thoughts**

- _none_

**Tool observations**

- `hot_topics`: period: 2026-W11; posts: 317; summary: В этой неделе в Telegram-каналах активно обсуждались темы, связанные с HRM, KV и MLP. Наибольший интерес вызвало обсуждение метрик и подходов к моделированию, особенно в каналах MLunderhood и ai_machi
- hrm, kv, mlp (score=0.456, 21 posts, channels: MLunderhood,ai_machinelearning_big_data,cryptovalerii)
- eval, ground truth, truth (score=0.427, 4 posts, channels: gonzo_ml)
- accept, cs, neurips (score=0.387, 7 posts, channels: rybolos_channel,scientific_ope
- `final_answer`: Final answer prepared (1428 chars)

**Citations (документы использованные в ответе)**

- _none_

**Offline Judge Scoring**

| Metric | Score | Reasoning |
|--------|-------|-----------|
| factual (0.0-1.0, step 0.1): полнота и точность фактов vs expected | ___ | |
| useful (0.0-2.0, step 0.1): 0=бесполезен, 1=частично, 2=полностью отвечает | ___ | |

---

## golden_v2_q33 — Какие горячие темы были в марте 2026?
- Eval mode: `analytics`
- Category: `analytics_hot_topics`
- Answerable: `True`
- Key tools: `hot_topics`
- Forbidden tools: `search, temporal_search`
- Tools invoked: `hot_topics, final_answer`
- Coverage: `0.0`

**Expected answer**

В марте 2026 среди горячих тем выделялись GPT-5.4 и Codex, Apple-related announcements, Amazon AI/AWS AI и обсуждения eval/ground truth.

**Required claims**

- Ответ должен использовать hot_topics для month aggregation
- Ответ должен назвать минимум две темы, реально попавшие в мартовский digest

**Agent answer**

В марте 2026 года русскоязычное сообщество AI/ML в Telegram активно обсуждало следующие горячие темы:

1. **Архитектура моделей (HRM, KV, MLP):** Самая популярная тема (90 постов), касающаяся методов оптимизации и самопроверки нейросетей [1].
2. **Футуристические концепции (XXI, ASI):** Обсуждение искусственного сверхразума и этики ИИ, получившее 44 упоминания [1].
3. **Экосистема GPT-5:** Детальное обсуждение возможностей моделей GPT-5, включая версии «thinking» и «Plus» [1].
4. **Продукты Apple:** Интерес к новым MacBook, iPhone и AirPods с ИИ-функциями (16 постов) [1].
5. **Модели MoE и Qwen:** Фокус на архитектуре Mixture of Experts и моделях семейства Qwen [1].
6. **Нейроинтерфейсы:** Обсуждение разработок Neuralink и BCI (9 постов) [1].

Лидерами по упоминаниям в марте 2026 года стали: OpenAI (152), Claude/Anthropic (139+101), Google (84) и NVIDIA (70) [2].

**Agent thoughts**

- В марте 2026 года в русскоязычном сегменте Telegram-каналов, посвященных AI/ML, наблюдался высокий интерес к ряду ключевых тем. Основное внимание уделялось техническим аспектам архитектуры моделей и н

**Tool observations**

- `hot_topics`: period: month:2026-03; posts: 1364; summary: В этой неделе основное внимание в сообществе AI/ML было сосредоточено на темах, связанных с HRM, KV и MLP, где обсуждались методы оптимизации и эффективности моделей. В каналах, таких как MLunderhood 
- xxi, asi, what is (score=0.544, 44 posts, channels: techsparks,techno_yandex,theworldisnoteasy)
- hrm, kv, mlp (score=0.521, 90 posts, channels: cryptovalerii,neurohive,singularityfm)
- gpt54, gpt5 thinking, plus (score=0.467, 11 posts, channels: addme
- `final_answer`: Final answer prepared (875 chars)

**Citations (документы использованные в ответе)**

- _none_

**Offline Judge Scoring**

| Metric | Score | Reasoning |
|--------|-------|-----------|
| factual (0.0-1.0, step 0.1): полнота и точность фактов vs expected | ___ | |
| useful (0.0-2.0, step 0.1): 0=бесполезен, 1=частично, 2=полностью отвечает | ___ | |

---

## golden_v2_q34 — Какие каналы эксперты по NLP? Кто лучше разбирается в NLP?
- Eval mode: `analytics`
- Category: `analytics_channel_expertise`
- Answerable: `True`
- Key tools: `channel_expertise`
- Forbidden tools: `search, temporal_search`
- Tools invoked: `channel_expertise, final_answer`
- Coverage: `0.0`

**Expected answer**

Среди сильных каналов по NLP должны появляться xor_journal, ai_newz и gonzo_ml; ответ должен опираться на channel_expertise ranking, а не на обычный search.

**Required claims**

- Ответ должен использовать topic mode инструмента channel_expertise
- Ответ должен назвать минимум два релевантных канала из топа по NLP

**Agent answer**

Согласно анализу экспертности каналов по теме NLP (Natural Language Processing), лучшие каналы, разбирающиеся в этой области, определяются по показателю авторитетности:

1. **@xor_journal** — лидер с коэффициентом авторитетности 0.77. Канал активно освещает темы OpenAI, Claude, Яндекс и Google [1].
2. **@ai_newz** — второй по авторитетности (0.742), фокусируется на новостях и разработках от Google, OpenAI, Claude и Gemini [1].
3. **@gonzo_ml** (0.735) — специализируется на технических деталях моделей, Transformer-архитектурах, Qwen и DeepSeek [1].
4. **@seeallochnaya** (0.733) — публикует аналитику и обзоры по OpenAI, GPT-5, Anthropic и Claude [1].

Также в топ входят каналы:
*   **@protechietich** (0.654) — упоминает NVIDIA, OpenAI, Google.
*   **@llm_under_hood** (0.63) — глубоко анализирует архитектуру LLM, фокус на OpenAI и Claude [1].
*   **@stuffynlp** (0.569) — узкоспециализированный канал про SFT, Qwen, Llama и DeepSeek [1].

Наиболее экспертными каналами по NLP являются @xor_journal, @ai_newz и @gonzo_ml, так как они имеют наивысшие баллы авторитетности и наиболее релевантные технические темы (Transformer, SFT, архитектуры моделей) в своих публикациях.

**Agent thoughts**

- _none_

**Tool observations**

- `channel_expertise`: Found 10 channels for topic='NLP', metric=authority: xor_journal(0.77), ai_newz(0.742), gonzo_ml(0.735), seeallochnaya(0.733), protechietich(0.654)
- `final_answer`: Final answer prepared (1179 chars)

**Citations (документы использованные в ответе)**

- _none_

**Offline Judge Scoring**

| Metric | Score | Reasoning |
|--------|-------|-----------|
| factual (0.0-1.0, step 0.1): полнота и точность фактов vs expected | ___ | |
| useful (0.0-2.0, step 0.1): 0=бесполезен, 1=частично, 2=полностью отвечает | ___ | |

---

## golden_v2_q35 — В чём экспертиза канала gonzo_ml?
- Eval mode: `analytics`
- Category: `analytics_channel_expertise`
- Answerable: `True`
- Key tools: `channel_expertise`
- Forbidden tools: `search, temporal_search, channel_search`
- Tools invoked: `channel_expertise, final_answer`
- Coverage: `0.0`

**Expected answer**

Ответ должен вернуть профиль gonzo_ml: темы канала, top_topics и общий профиль экспертизы без обычного retrieval path.

**Required claims**

- Ответ должен использовать profile lookup режима channel_expertise
- Ответ должен описать экспертизу конкретного канала gonzo_ml

**Agent answer**

Канал @gonzo_ml специализируется на освещении последних разработок в области искусственного интеллекта и машинного обучения, с особым акцентом на технические аспекты и сравнения моделей. 

Среди ключевых тем и обсуждаемых сущностей канала:
- Архитектуры и модели: Transformer, Claude, Qwen, DeepSeek, GPT-5, Gemini, MoE (Mixture of Experts).
- Платформы и инструменты: HuggingFace, NVIDIA.
- Технические детали: упоминаются такие термины как "hrm, kv, mlp", что указывает на глубокое погружение в архитектуру нейросетей [1].

Канал обладает высоким авторитетом (0.735) и публикует около 22 постов в неделю, охватывая более 39 активных недель. Его профиль характеризуется как структурированный и информативный, подходящий как для специалистов, так и для любителей ИИ [1].

**Agent thoughts**

- _none_

**Tool observations**

- `channel_expertise`: Channel gonzo_ml: authority=0.735, summary: Канал @gonzo_ml публикует 856 постов о AI/ML. Среди тем — Transformer, Claude, Qwen, DeepSeek, HuggingFace. Авторитетность 0.735, скорость публикаций 0.346. Охватывает последние разработки и модели. А
- `final_answer`: Final answer prepared (770 chars)

**Citations (документы использованные в ответе)**

- _none_

**Offline Judge Scoring**

| Metric | Score | Reasoning |
|--------|-------|-----------|
| factual (0.0-1.0, step 0.1): полнота и точность фактов vs expected | ___ | |
| useful (0.0-2.0, step 0.1): 0=бесполезен, 1=частично, 2=полностью отвечает | ___ | |

---

## golden_v2_q36 — Какие каналы лучше пишут про робототехнику и роботакси?
- Eval mode: `analytics`
- Category: `analytics_channel_expertise`
- Answerable: `True`
- Key tools: `channel_expertise`
- Forbidden tools: `search, temporal_search`
- Tools invoked: `channel_expertise, final_answer`
- Coverage: `0.0`

**Expected answer**

Среди релевантных каналов по робототехнике и роботакси должен фигурировать techsparks; ответ должен использовать channel_expertise как ranking/profile tool.

**Required claims**

- Ответ должен использовать topic mode инструмента channel_expertise
- Ответ должен назвать минимум один канал, релевантный теме робототехники/роботакси

**Agent answer**

Согласно анализу авторитетности каналов в базе данных за период с июля 2025 по март 2026 года, лучше всего про робототехнику и роботакси пишут следующие каналы:

1. **@xor_journal** — лидер по авторитетности (0.77). В топ его тематических сущностей входят Яндекс (60 публикаций), Google (59) и OpenAI (86), что коррелирует с новостями о роботакси Яндекса и глобальных разработках в сфере автономного вождения [1].
2. **@techno_yandex** — второй по авторитетности канал (0.675). Он специализируется на технологиях Яндекса, где Яндекс является ключевой сущностью (59 публикаций), что делает его основным источником новостей о роботакси и робототехнике в экосистеме компании [1].
3. **@protechietech** — третий по значимости канал (0.654). Высокая релевантность достигается за счет частых упоминаний NVIDIA (34 публикации), которая является ключевым поставщиком вычислительных мощностей для робототехники и беспилотников, а также Google и Microsoft [1].

Также стоит отметить каналы **@llm_under_hood** (авторитетность 0.63) и **@singularityfm** (0.59), которые регулярно обсуждают связанные с робототехникой технологии и компании [1].

**Agent thoughts**

- _none_

**Tool observations**

- `channel_expertise`: Found 10 channels for topic='робототехника роботакси', metric=authority: xor_journal(0.77), techno_yandex(0.675), protechietich(0.654), llm_under_hood(0.63), singularityfm(0.59)
- `final_answer`: Final answer prepared (1132 chars)

**Citations (документы использованные в ответе)**

- _none_

**Offline Judge Scoring**

| Metric | Score | Reasoning |
|--------|-------|-----------|
| factual (0.0-1.0, step 0.1): полнота и точность фактов vs expected | ___ | |
| useful (0.0-2.0, step 0.1): 0=бесполезен, 1=частично, 2=полностью отвечает | ___ | |

---
