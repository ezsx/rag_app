Ниже — готовые файлы правил и минимальные пояснения. Формат — `.cursor/rules/*.mdc`. Это валидный фронт-маттер и структура для Cursor Rules; `alwaysApply: true` делает правило глобальным, globs — для авто-подключения по файлам. Поддержка правил/auto-attach/auto-run MCP — официально в Cursor. ([Cursor][1])

---

# 1) Глобальное правило с учётом Ripgrep MCP

`.cursor/rules/global.mdc`

```md
---
description: "RAG App — авто-контекст (docs/ai) + дисциплина извлечения (MCP: tree_sitter, workspace-code-search, ripgrep)"
globs:
  - "**/*"
alwaysApply: true
---

# Политика контекста
- Всегда сначала используй статическую доку из `docs/ai/**` (brief, architecture, pipelines, glossary, modules, ADR).
- Не подтягивай целые файлы. Ссылайся на (path, line-range), минимально достаточный фрагмент.
- Для «по смыслу» — `workspace-code-search`. Для точных строк/регексов/флагов — `ripgrep`.

# Auto-Context Refresh (выполняй при старте чата и перед крупными правками; авто-run MCP включён)
1) Зарегистрировать проект (если не зарегистрирован):
```

tool: tree\_sitter.register\_project\_tool(path="C:/llms/rag/rag\_app", name="rag\_app")

```
2) Обновить AST/символы:
```

tool: tree\_sitter.analyze\_project(project="rag\_app")

```
3) Перегенерировать обзор проекта:
```

prompt: tree\_sitter.project\_overview(project="rag\_app")
→ перезапиши docs/ai/project\_brief.md (сохрани блок «Инварианты», если есть)

```
4) Инкремент по изменённым файлам:
```

terminal: git diff --name-only HEAD\~1..HEAD
для каждого изменённого src/\*\*/\*.py:
tool: tree\_sitter.explain\_code(project="rag\_app", path="<file>")
tool: tree\_sitter.get\_symbols(project="rag\_app", path="<file>")
→ обнови docs/ai/modules/<relative-path>.md

```
5) Если менялись эндпоинты (src/api/v1/**): обнови краткий список в `docs/ai/contracts/openapi.md`:
- `workspace-code-search` по ключам `APIRouter`, `@router.(get|post|put|delete)`
- при необходимости точная верификация:
```

tool: ripgrep.search(pattern="@router\\.(get|post|put|delete)", path="src/api/v1")

```

# Дисциплина извлечения
- Сначала docs/ai/**, затем:
  - `workspace-code-search` — семантические кандидаты (быстро найти «куда смотреть»).
  - `ripgrep.search|advanced-search` — точные строки, флаги, regex, типы файлов.
- Перед ответом проверяй противоречия между докой и кодом; при расхождении приоритет у кода → обнови доку.

# When to ADR
- Меняется пайплайн (ingest/index/retrieve/rerank/cache), SLA, протоколы, границы сервисов.
- Создай `docs/ai/adr/ADR-YYYYMMDD-<slug>.md` (контекст → решение → последствия) и сослись в PR.

# Качество
- План → diff → тест-план → риски → ссылки на строки. Никаких лишних файлов в контексте.
```

Примечания: авто-run MCP настраивается в Cursor (чтобы инструменты вызывались без подтверждения). ([Cursor][2])

---

# 2) Правила для «планирования» и «реализации» (автоприменение по контексту файлов)

Идея: эти правила подхватываются автоматически, когда в контекст подтягиваются соответствующие файлы (глоб — «семантическое» подключение по текущим файлам проекта). Это стандартная механика Cursor: правило подхватывается при матчинге glob в активном контексте/редактируемых файлах. ([cursor101.com][3])

## 2.1 Планирование

`.cursor/rules/chat-planning.mdc`

```md
---
description: "Planning — составление плана по docs/ai + AST, без правок кода"
globs:
  - "docs/ai/**"
  - "src/**"
alwaysApply: false
---

# Поведение
- Цель: из запроса получить Task Brief, Context Bundle (список файлов с причинами), Acceptance Checklist и риски — без изменений кода.

# Шаги
1) Обнови контекст (см. Auto-Context Refresh в global.mdc).
2) Сформируй Task Brief (цель, ограничения, затронутые модули).
3) Построй Context Bundle:
   - `workspace-code-search` для семантических кандидатов
   - `ripgrep.search` для точного подтверждения имён/флагов/эндпоинтов
   - укажи (path, line-range, why)
4) Сформируй Acceptance Checklist и риски.
5) При значимых изменениях архитектуры — предложи ADR драфт (не коммить сам).

# Ограничения
- Не вноси правок. Только план/чек-лист/риски/список файлов+строк.
```

## 2.2 Реализация

`.cursor/rules/chat-implementation.mdc`

```md
---
description: "Implementation — минимальные правки по Context Bundle, обновление docs/ai/modules"
globs:
  - "src/**"
  - "docs/ai/modules/**"
alwaysApply: false
---

# Поведение
- Цель: внести точечные изменения по согласованному плану/Bundle, обновить модульные обзоры и тесты.

# Шаги
1) Обнови контекст (Auto-Context Refresh).
2) На каждую правку ссылаться на (path, line-range) из Bundle.
3) После правок:
   - перегенерируй docs/ai/modules/<file>.md через `tree_sitter.explain_code/get_symbols`
   - если менялись роуты — освежи `docs/ai/contracts/openapi.md`
4) Прогон тестов/линтера (если включён авто-run terminal).

# Ограничения
- Не трогай несвязанные файлы. Не увеличивай контекст сверх необходимого.
```

---

# 3) Ripgrep MCP — что это и как звать

Рекомендуемый сервер: `mcollina/mcp-ripgrep`. Доступные инструменты: `search`, `advanced-search`, `count-matches`, `list-files`, `list-file-types`. Мы используем `search`/`advanced-search` в правилах выше. ([GitHub][4])

Подключение (напоминание) в `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "ripgrep": {
      "command": "npx",
      "args": ["-y", "mcp-ripgrep@latest"]
    }
  }
}
```

Убедись, что `rg` в PATH. Сам ripgrep — кроссплатформенный, уважает .gitignore. ([GitHub][5])

Когда он нужен:

* Проверка точных сигнатур декораторов роутов, флагов, сообщений ошибок, ключей конфига.
* Быстрый «срез» по маске/типу файла (через `advanced-search`).
* Семантический слой это не заменяет; они дополняют друг друга. ([npm][6], [GitHub][7])

---

# 4) Нужен ли memory-bank «архив»

Для твоей модели работы (docs-as-code в репозитории) — не обязателен. Если требуется кросс-чатовая личная память вне git, можно потом добавить Extended Memory MCP. Но сначала достаточно `docs/ai/**` + эти правила. ([Cursor - Community Forum][8])

---

## Что сделать сейчас

1. Скопируй три файла правил в `.cursor/rules/`.
2. Проверь в Cursor, что правила видны; включи Auto-run MCP Tools. ([Cursor][2])
3. В новом чате начни с «планирования» — правило `chat-planning.mdc` подхватится как только будут в контексте файлы из `docs/ai/**`/`src/**` (или прикрепи через “@Cursor Rules”). ([Cursor - Community Forum][9])

Если нужно — добавлю узкоспециализированные правила по директориям (`src/api/v1/**`, `src/services/**`, `adapters/search/**`), чтобы Cursor «семантически» подхватывал нужные инструкции при работе с конкретной частью кода.

[1]: https://docs.cursor.com/context/rules-for-ai?utm_source=chatgpt.com "Cursor – Rules"
[2]: https://docs.cursor.com/context/model-context-protocol?utm_source=chatgpt.com "Cursor – Model Context Protocol (MCP)"
[3]: https://cursor101.com/cursor/rules?utm_source=chatgpt.com "Understanding Cursor Rules"
[4]: https://github.com/mcollina/mcp-ripgrep "GitHub - mcollina/mcp-ripgrep: An MCP server to wrap ripgrep"
[5]: https://github.com/BurntSushi/ripgrep?utm_source=chatgpt.com "ripgrep recursively searches directories for a regex pattern ..."
[6]: https://www.npmjs.com/package/%40mseep/mcp-ripgrep?utm_source=chatgpt.com "mseep/mcp-ripgrep"
[7]: https://github.com/mcollina/mcp-ripgrep?utm_source=chatgpt.com "An MCP server to wrap ripgrep"
[8]: https://forum.cursor.com/t/my-best-practices-for-mdc-rules-and-troubleshooting/50526?utm_source=chatgpt.com "My Best Practices for MDC rules and troubleshooting"
[9]: https://forum.cursor.com/t/rules-not-automatically-picked-up/47304?page=2&utm_source=chatgpt.com "Rules not automatically picked up - Page 2 - Discussions"
