## src/utils/gbnf.py — GBNF грамматики и билдеры

### Назначение
- Предоставляет полную GBNF‑грамматику для `SearchPlan` и микро‑грамматики для массивов строк фиксированной длины.
- Совместим с `llama.cpp` через `llama_cpp.LlamaGrammar`.

### Экспортируемые объекты
- `SEARCH_PLAN_GBNF: str` — грамматика полного JSON‑объекта плана (3–6 `normalized_queries`, фильтры, `k_per_query`, `fusion`).
- `gbnf_selfcheck()` — самопроверка корректности грамматик (создание LlamaGrammar из строк).
- `build_searchplan_grammar()` / `get_searchplan_grammar()` — собрать и вернуть `LlamaGrammar` для плана.
- `build_micro_grammar(n: int)` / `get_string_array_grammar(n)` — микро‑грамматика массива из `n` JSON‑строк.

### Использование (пример)
```python
from llama_cpp import Llama
from utils.gbnf import get_searchplan_grammar

llm = Llama(model_path=..., n_ctx=2048)
grammar = get_searchplan_grammar()
res = llm.create_chat_completion(messages=[...], grammar=grammar, temperature=0.2)
```

### Особенности реализации
- Строковые литералы поддерживают escape/unicode; запрещены лишние ключи и trailing‑запятые.
- `normalized_queries` — массив из 3–6 элементов (жёстко в грамматике).
- Числовые поля ограничены: `k_per_query` ∈ [1..50].


