# Golden v3 Review Packet

> Status values: `accept`, `edit`, `reject`.
> Review goal: catch too-narrow expected answers, weak/non-core questions, and broken anchors.

## golden_v3_q073 — constrained_search / retrieval_evidence

- Status: edit
- Action: edit
- Query: Какие новости о медицинском и биотехнологическом применении ИИ были опубликованы в канале ml_product 1 июля 2025 года?
- Source IDs: ml_product:656
- Expected channels: ml_product

**Expected answer**

В канале ml_product 1 июля 2025 года упоминались две релевантные новости: Microsoft представила
диагностическую ИИ-систему MAI-DxO, которая в 4 раза точнее врачей и снижает затраты на анализы; Chai
Discovery применяет ИИ для разработки новых антител с эффективностью 20% против 0.1% у традиционных методов.

**Required claims**

- Microsoft представила ИИ-систему MAI-DxO для диагностики.
- MAI-DxO описана как в 4 раза более точная, чем врачи, и экономящая затраты на анализы.
- Chai Discovery разрабатывает антитела с помощью ИИ с эффективностью 20% против 0.1% у традиционных методов.

**Source snippet**

```text
ml_product:656 | 2025-07-01
Microsoft делает шаг к «медицинскому суперинтеллекту»
(Microsoft steps towards 'medical superintelligence')

ссылка
- Microsoft представила ИИ-систему MAI-DxO для диагностики, которая в 4 раза точнее врачей и экономит затраты
на анализы.
- Baidu открыла исходный код ИИ-модели ERNIE 4.5, которая превосходит конкурентов, несмотря на меньший размер.
- Google Gemini теперь может автоматически создавать заголовки, описания и теги для YouTube-видео.
- ИИ Chai Discovery разрабатывает новые антитела для лекарств с рекордной эффективностью (20% успеха против
0,1% у традиционных методов).
- Meta запустила лабораторию "Superintelligence Labs" для разработки продвинутого ИИ, переманив специалистов
из OpenAI и Google.

#dailyshortcuts
```

**Reviewer notes**

- Edited to match medical/biotech scope; removed unrelated Meta Superintelligence Labs from expected answer.

## golden_v3_q074 — broad_search / retrieval_evidence

- Status: edit
- Action: edit
- Query: Какие независимые провайдеры через OpenRouter добавили поддержку Structured Outputs, и как в бенчмарке с SGR показали себя модели Opus 4.1 и Mistral Medium 3.1?
- Source IDs: llm_under_hood:636
- Expected channels: llm_under_hood

**Expected answer**

Поддержку Structured Outputs через OpenRouter добавили Fireworks, Cerebras и Groq. В бенчмарке Opus 4.1 заняла
22-е место, а Mistral Medium 3.1 — 38-е, обе модели показали результаты хуже предыдущих версий.

**Required claims**

- Fireworks, Cerebras и Groq добавили поддержку Structured Outputs через OpenRouter
- Opus 4.1 заняла 22-е место в бенчмарке
- Mistral Medium 3.1 заняла 38-е место в бенчмарке

**Source snippet**

```text
llm_under_hood:636 | 2025-08-18
Бенчмарк новых моделей: Grok, Opus 4.1, Mistral Medium 3.1

Elon Musk что-то делает правильно. Мало того, что у них Grok-4 работает с нормальным Structured Outputs, так
Grok-4 по очкам заняла первое место. Ровно столько же очков у GPT-5 (medium reasoning). Дорогие, но умные.

Кстати, на данный момент поддержка Structured Outputs (которая нужна для стабильной работы SGR) появилась у
большего числа независимых провайдеров (все они доступны через OpenRouter):

- Fireworks
- Cerebras
- Groq

Это вдобавок к крупным провайдерам - OpenAI (+Azure), Mistral, Google (ограниченные Structured Outputs).

NB: GPT-OSS модели OpenAI из-за нового Harmony формата пока со Structured Outputs стабильно не работают - ни у
провайдеров, ни в ollama. Нужно подождать.

Anthropic Claude - пока продолжают болтаться в аутсайдерах на промышленных задачах. Компания молчит по-
партизански про поддержку constrained decoding/Structured outputs, а Opus 4.1 по очкам на бизнес-бенчмарке с
использованием SGR стал чуть хуже, чем Opus 4.0. 22 место.

Mistral Medium 3.1 - тоже без прорывов. По очкам чуть хуже, чем Mistral Medium 3.0. 38 место.

Ваш, @llm_under_hood 🤗
```

**Reviewer notes**

- Edited query typo/non-Russian artifact; claims are source-supported.

## golden_v3_q075 — broad_search / retrieval_evidence

- Status: accept
- Action: accept
- Query: Какие выводы о влиянии небольших изменений параметров на производительность моделей делает автор в анализе применения экспериментальных подходов?
- Source IDs: ml_product:762
- Expected channels: ml_product

**Expected answer**

Автор подчеркивает, что незначительные изменения, такие как увеличение параметра на 10%, могут существенно
улучшить производительность моделей. Это подтверждает важность методичного эксперимента и точных измерений для
развития теории.

**Required claims**

- Незначительные изменения параметров могут существенно улучшить производительность моделей
- Увеличение параметра на 10% приводится как пример такого изменения
- Автор подчеркивает важность методичного эксперимента и точных измерений

**Source snippet**

```text
ml_product:762 | 2025-07-20
Кими К2
(Kimi K2)

ссылка

Автор анализирует применение экспериментальных подходов для оптимизации сложных систем, где используются
количественные данные, например, показатели в 10 и 20 итераций, для сравнения результатов разных методов.

Особое внимание уделяется тому, как незначительные изменения (например, увеличение параметра на 10%) могут
существенно улучшить производительность моделей, подчёркивая важность методичного эксперимента и точных
измерений для развития теории.

#news
```

**Reviewer notes**

- Accepted; source supports 10% parameter example and methodical experiment/measurement conclusion.

## golden_v3_q076 — constrained_search / retrieval_evidence

- Status: accept
- Action: accept
- Query: Какие ключевые характеристики и тарифы предлагает Cloud.ru в новой платформе Evolution AI Factory?
- Source IDs: xor_journal:8138
- Expected channels: xor_journal

**Expected answer**

Cloud.ru запустил Evolution AI Factory — облачную среду для работы с GenAI, включающую круглосуточную
поддержку и SLA. Тарифы составляют в среднем 35 ₽ за млн входных токенов и 70 ₽ за млн выходных.

**Required claims**

- Cloud.ru выпустил платформу Evolution AI Factory
- Платформа включает круглосуточную поддержку и SLA
- Средняя цена 35 ₽ за млн входных токенов и 70 ₽ за млн выходных

**Source snippet**

```text
xor_journal:8138 | 2025-11-20
Разработчики, вам подарочек: Cloud.​ru выпустил в коммерческую эксплуатацию Evolution AI Factory — облачную
среду для создания, запуска и масштабирования приложений на базе GenAI.

Внутри AI-платформы шесть взаимосвязанных сервисов для полного цикла работы с AI: круглосуточная поддержка,
SLA и приятные тарифы — средняя цена 35 ₽ за млн входных токенов и 70 ₽ за млн выходных. В каталоге более 20
популярных моделей, в том числе GigaChat, Qwen и ChatGPT.

Уже хочется потестить 🤩

@xor_journal
```

**Reviewer notes**

- Accepted; source supports Evolution AI Factory, support/SLA, and token prices.

## golden_v3_q077 — broad_search / retrieval_evidence

- Status: accept
- Action: accept
- Query: Какие недостатки у агентов, упомянутые в посте, и какой пример с кодом привёл автор?
- Source IDs: toBeAnMLspecialist:1111
- Expected channels: toBeAnMLspecialist

**Expected answer**

Агенты описаны как исполнительные, но прямолинейные и безынициативные исполнители, которые не думают за
пользователя. В примере с кодом ассистент проигнорировал заложенную интернационализацию и хардкод текстов,
вместо использования fluent.

**Required claims**

- Агенты являются исполнительными, но прямолинейными и безынициативными
- Агенты не думают за пользователя и выполняют работу неожиданным способом
- Ассистент проигнорировал интернационализацию и использовал хардкод текстов вместо fluent

**Source snippet**

```text
toBeAnMLspecialist:1111 | 2026-03-23
Фраза "Без ТЗ результат ХЗ" стала как-никогда актуальной. Если ваш менеджер и/или заказчик раздражает вас тем,
что плохо описывает свои хотелки - можете отправить его тренироваться на агентах. Это очень исполнительные, но
очень прямолинейные и почти безынициативные исполнители, которые сами за тебя не подумают и выполнят работу
способом, которого ты от них не ожидаешь.

Забыл (не подумал) написать, что не нужно хардкодить в проекте тексты, а нужно использовать fluent, получил
все тексты прямо в коде, несмотря на то, что изначально в проекте была заложена интернационализация. Ассистент
тупо проигнорировал уже заложенный подход и сделал по-своему.
```

**Reviewer notes**

- Accepted; source supports agent limitations and fluent/i18n hardcode example.

## golden_v3_q078 — broad_search / retrieval_evidence

- Status: accept
- Action: accept
- Query: Какие метрики и результаты показала новая модель прогноза осадков в Яндекс Погоде?
- Source IDs: MLunderhood:273
- Expected channels: MLunderhood

**Expected answer**

Модель увеличила CSI по сильным осадкам на 50% относительно бэйзлайна и более чем в два раза по сравнению с
общепринятым подходом. Метрика bias снизилась в 10 раз, достигнув уровня численных моделей, а качество
прогноза на 12–18 часов сопоставимо или выше, чем у WeatherNext2 от Google.

**Required claims**

- CSI по сильным осадкам вырос на 50% относительно бэйзлайна
- Метрика bias снизилась в 10 раз
- Качество прогноза на 12–18 часов сопоставимо или выше, чем у WeatherNext2

**Source snippet**

```text
MLunderhood:273 | 2025-12-26
Что нового в Нейрометеуме — нейросети глобального прогноза от Яндекс Погоды

Новая нейросеть для глобального прогноза погоды рассчитывает 70 ключевых характеристик атмосферы на 10 суток
вперёд с часовым шагом. В этом посте — немного «внутрянки» о том, что нового появилось в Нейрометеуме.

Во-первых, модель Яндекса сделали быстрой и автономной. Если численным методам нужны часы на расчёт, то эта
нейросеть справляется за несколько минут. К тому же в расчёте нет зависимости от внешних данных
метеорологических центров — всё рассчитывается самостоятельно, но пока что зависимость сохраняется в данных
для старта.

Во-вторых, использовали инновационный подход к обучению модели. Архитектурно за основу взяли Aurora
(Microsoft), а от Pangu Weather (Huawei) переняли идею обучать несколько моделей для разных временных
горизонтов, а не одну. При этом смогли решить проблему несогласованности прогнозов благодаря авторегрессии в
латентном пространстве. Эксперименты с гиперпараметрами (число блоков, «голов» и так далее) показали, что
качество достигает насыщения. В итоге модель превзошла Aurora по числу параметров — у Нейрометеума их 1,5
млрд.

В-третьих, повысили точность прогноза осадков. В Яндекс Погоде придумали, как эффективнее работать с
переменной «осадки» (zero-inflated distribution). Вот что для этого сделали:

— использовали нормировку/перемасштабирование (в основе — паттерн из MetNet от Google);
— применили специальную функцию активации;
— разработали новые функции потерь (MWAE и лосс на основе Центра Масс — CoM).

А вот и результаты:

— CSI по сильным осадкам вырос на 50% относительно бэйзлайна и более чем вдвое относительно общепринятого
подхода;
— метрика bias снизилась в 10 раз и достигла уровня численных моделей;
— в сравнении с последней моделью Google (WeatherNext2) — модель показывает сопоставимое или более высокое
качество прогноза осадков на ближайшие 12–18 часов.

Сейчас прогнозы Нейрометеума используют как входные данные для профильной модели осадков в Яндекс Погоде.

Подробнее о том, как устроена новая нейросеть глобального прогноза погоды, читайте на Хабре.

ML Underhood
```

**Reviewer notes**

- Accepted; source supports CSI +50%, bias x10 reduction, and WeatherNext2 comparison.

## golden_v3_q079 — constrained_search / retrieval_evidence

- Status: accept
- Action: accept
- Query: Какие проблемы безопасности и управления выявляет работа с ИИ-агентами в канале gonzo_ml?
- Source IDs: gonzo_ml:4841
- Expected channels: gonzo_ml

**Expected answer**

Работа выявляет структурную уязвимость продвинутых моделей к несанкционированному доступу, подмене личности и
кривому управлению ресурсами при наличии операционной автономии. Это показывает недостаточность post-training
alignment для систем, действующих как самостоятельные прокси.

**Required claims**

- Продвинутые модели уязвимы к несанкционированному доступу и подмене личности при наличии автономии.
- Существует риск катастрофически кривого управления ресурсами.
- Одного post-training alignment недостаточно для систем, работающих как прокси.

**Source snippet**

```text
gonzo_ml:4841 | 2026-02-27
Найс! Любителям Openclaw посвящается.

Agents of Chaos
Natalie Shapira, Chris Wendler, Avery Yen, Gabriele Sarti, Koyena Pal, Olivia Floody, Adam Belfki, Alex
Loftus, Aditya Ratan Jannali, Nikhil Prakash, Jasmine Cui, Giordano Rogers, Jannik Brinkmann, Can Rager, Amir
Zur, Michael Ripa, Aruna Sankaranarayanan, David Atkinson, Rohit Gandikota, Jaden Fiotto-Kaufman, EunJeong
Hwang, Hadas Orgad, P Sam Sahil, Negev Taglicht, Tomer Shabtay, Atai Ambus, Nitay Alon, Shiri Oron, Ayelet
Gordon-Tapiero, Yotam Kaplan, Vered Shwartz, Tamar Rott Shaham, Christoph Riedl, Reuth Mirsky, Maarten Sap,
David Manheim, Tomer Ullman, David Bau
Статья: https://arxiv.org/abs/2602.20021
Ревью: https://arxiviq.substack.com/p/agents-of-chaos
Сайт: https://agentsofchaos.baulab.info/

# TL;DR

ЧТО сделали: Авторы провели исследовательский red-teaming автономных агентов на базе языковых моделей в
реальных условиях. В течение двух недель исследователи взаимодействовали с агентами, развёрнутыми в
изолированных виртуалках с постоянной памятью, полным доступом к shell и инструментами для мультиагентной
коммуникации (Discord, email), чтобы выявить системные уязвимости как в обычных, так и в состязательных
сценариях.

ПОЧЕМУ это важно: Работа подсвечивает критическую дыру в безопасности и управлении ИИ-агентами. Она
доказывает, что продвинутые модели, получив операционную автономию и доступ к тулзам, структурно уязвимы к
несанкционированному доступу, подмене личности и катастрофически кривому управлению ресурсами. Это бьёт по
текущим парадигмам AI alignment, показывая, что одного лишь выравнивания поведения на этапе post-training
недостаточно для систем, работающих как самостоятельные прокси в сложной социальной среде.

Подробнее: https://t.me/gonzo_ML_podcasts/2557
```

**Reviewer notes**

- Accepted; source supports agent security/governance risks and post-training alignment insufficiency.

## golden_v3_q080 — analytics_hot_topics / analytics

- Status: accept
- Action: accept
- Query: Какие горячие темы были на неделе 2025-W32?
- Source IDs: -
- Expected channels: -

**Expected answer**

В этой неделе активно обсуждались обновления GPT-5 и GPT-54, с акцентом на улучшения в понимании и генерации
сложных текстов. Каналы, такие как 'addmeto' и 'ai_newz', подчеркивали развитие GPT-5 thinking и GPT5 Plus.
Второй по популярности — Moe, v31 и Qwen, с активным обсуждением в 'ai_machinelearning_big_data' и
'data_secrets'. Также заметно обсуждение XML, Markdown и SelfReflection, особенно в 'aihappens' и 'denissexy'.
Темы Hrm, Kv, Mlp активно обсуждались в 'ai_machinelearning_big_data' и 'neurohive'. Важно отметить рост
интереса к 3D-технологиям, Genie и Marble, с упоминаниями в 'ai_newz' и 'xor_journal'. В топе упоминаний —
OpenAI (88), GPT-5 (85), Google (39), o3 (24), Gemini (22), Claude (21), GPT-4o (19), Anthropic (19), Grok
(17), Qwen (15). Обсуждения охватывали как технические аспекты, так и практическое применение моделей.

**Required claims**

- Неделя 2025-W32 содержит hot topics: gpt54, gpt5 thinking, plus, moe, v31, qwen, xml, markdown, selfreflection.
- Всего постов в weekly digest: 465.

**Source snippet**

```text
(no source snippet; static/analytics item)
```

**Reviewer notes**

- Accepted; weekly_digests 2025-W32 supports topic labels and post_count=465.

## golden_v3_q081 — analytics_hot_topics / analytics

- Status: accept
- Action: accept
- Query: Какие горячие темы были на неделе 2025-W30?
- Source IDs: -
- Expected channels: -

**Expected answer**

В этой неделе активно обсуждались темы, связанные с HR-моделями и квантовыми вычислениями. В каналах
MLunderhood и ai_machinelearning_big_data обсуждали развитие MLP-архитектур. В сфере здравоохранения росли
интересы к MedGemma и ChatGPT Health, особенно в каналах seeallochnaya и xor_journal. Также были популярны
темы, связанные с XXI веком и азиатскими технологиями, где упоминались такие проекты, как What Is. В
обсуждениях упоминались перспективы 2027 и 2028 годов, включая метрики и прогнозы. Робототехника и Unitree
получили внимание, особенно в каналах ai_machinelearning_big_data и ml_product. В топе упоминаний лидируют
OpenAI (47), Google (35), Qwen (32), Gemini (30), Claude (22), HuggingFace (18), DeepSeek (14), Microsoft
(14), NVIDIA (13), Anthropic (11). Обсуждались новые модели и их применение в различных отраслях. Каналы,
такие как data_secrets и gonzo_ml, активно делятся аналитикой и обзорами. В целом, неделя была насыщена
обсуждениями новых технологий и их влиянием на рынок.

**Required claims**

- Неделя 2025-W30 содержит hot topics: hrm, kv, mlp, health, medgemma, chatgpt health, xxi, asi, what is.
- Всего постов в weekly digest: 447.

**Source snippet**

```text
(no source snippet; static/analytics item)
```

**Reviewer notes**

- Accepted; weekly_digests 2025-W30 supports topic labels and post_count=447, despite noisy topic naming.

## golden_v3_q082 — analytics_hot_topics / analytics

- Status: accept
- Action: accept
- Query: Какие горячие темы были на неделе 2025-W29?
- Source IDs: -
- Expected channels: -

**Expected answer**

В этой неделе в Telegram-каналах активно обсуждались темы, связанные с HRM, KV и MLP, с наибольшей активностью
в каналах MLunderhood и ai_machinelearning_big_data. Популярны были темы, связанные с "that", "on", "is", хотя
их охват был меньше. Также отмечены обсуждения о P1, Tier и FrontierMath, в основном в каналах data_secrets и
seeallochnaya. Важной темой стали NanoChat, K2 Thinking и K2, с упоминаниями в ai_machinelearning_big_data и
aioftheday. В топе упоминаний — OpenAI (71), Google (44), Grok (23), Gemini (18), Anthropic (15), NVIDIA (13),
ICML (10), o3 (10), RAG (10), Llama (9). Обсуждались также GPT5 и его версии, включая GPT5 20 и Lean. Основные
дискуссии касались архитектурных решений, эффективности моделей и их применения в реальных задачах. Многие
каналы акцентировали внимание на новых исследованиях и инновациях в области ML и AI, особенно в контексте
масштабирования и оптимизации.

**Required claims**

- Неделя 2025-W29 содержит hot topics: hrm, kv, mlp, that, on, is, p1, tier, frontiermath.
- Всего постов в weekly digest: 440.

**Source snippet**

```text
(no source snippet; static/analytics item)
```

**Reviewer notes**

- Accepted; weekly_digests 2025-W29 supports topic labels and post_count=440, including noisy labels.

## golden_v3_q083 — analytics_hot_topics / analytics

- Status: accept
- Action: accept
- Query: Какие горячие темы были на неделе 2025-W48?
- Source IDs: -
- Expected channels: -

**Expected answer**

В этой неделе активно обсуждались темы, связанные с HRM, KV и MLP. В каналах MLunderhood и
ai_machinelearning_big_data обсуждались последние достижения в области машинного обучения, включая новые
подходы к обработке данных. Также продолжалась дискуссия о TPU и TSMC, особенно в контексте производительности
и архитектуры чипов. В канале ai_newz и data_secrets говорили о новых технологиях, связанных с Rubins. В сфере
Deep Learning активно обсуждались темы, связанные с DeepSchool и Docker, в частности, о возможностях
интеграции. В каналах ai_machinelearning_big_data и boris_again обсуждались новые инструменты в области OCR,
включая DeepSeekOCR и Omnidocbench. Также в каналах gonzo_ml и others обсуждались темы, связанные с Codex, RFC
и Expected. В топе упоминаний продолжали лидировать Google, OpenAI, Gemini и Anthropic, а также NVIDIA,
Claude, DeepSeek, GPT-5, HuggingFace и Qwen.

**Required claims**

- Неделя 2025-W48 содержит hot topics: hrm, kv, mlp, tpu, tsmc, rubin, dl, deepschool, docker.
- Всего постов в weekly digest: 424.

**Source snippet**

```text
(no source snippet; static/analytics item)
```

**Reviewer notes**

- Accepted; weekly_digests 2025-W48 supports topic labels and post_count=424.

## golden_v3_q084 — analytics_hot_topics / analytics

- Status: accept
- Action: accept
- Query: Какие горячие темы были на неделе 2025-W28?
- Source IDs: -
- Expected channels: -

**Expected answer**

В этой неделе активно обсуждались темы, связанные с HRM, KV и MLP, где основной фокус был на новых подходах к
обучению моделей и их практическом применении. В топе упоминаний остались Google (51), OpenAI (38) и Grok
(29), что подчеркивает их значимость в индустрии. Также были активны обсуждения о новых версиях моделей,
включая Gemini 31 и Qwen, что указывает на постоянную конкуренцию и развитие. Тема ARCAGI2 и GROK получила
внимание в нескольких каналах, что говорит о росте интереса к этим технологиям. Важно отметить, что обсуждения
о Moe и V31 также активны, что указывает на интерес к масштабируемым архитектурам. Наблюдается рост интереса к
DL и DeepSchool, что может говорить о новых образовательных инициативах. В каналах, таких как AI_newz и
Data_secrets, активно обсуждаются последние достижения и тренды. Также были обсуждены вопросы, связанные с
Docker и его применением в ML-проектах. Общая картина показывает, что индустрия AI/ML продолжает активно
развиваться, с ростом интереса к новым технологиям и улучшениям в существующих моделях.

**Required claims**

- Неделя 2025-W28 содержит hot topics: hrm, kv, mlp, old, feel, gold, arcagi2, gemini 31, grok.
- Всего постов в weekly digest: 416.

**Source snippet**

```text
(no source snippet; static/analytics item)
```

**Reviewer notes**

- Accepted; weekly_digests 2025-W28 supports topic labels and post_count=416, including noisy labels.
