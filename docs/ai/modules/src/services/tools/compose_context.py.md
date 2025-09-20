### Модуль `src/services/tools/compose_context.py`

MVP сбор контекста по документам `{id,text,metadata}` с ограничением ~1800 токенов (по символам).
Нумерует источники `[1..N]`, формирует `citations`.

Вход: `{docs:[...], max_tokens_ctx:int, citation_format:"footnotes"}`.
Выход: `{prompt:str, citations:[{id,index}]}`.


