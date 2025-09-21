### Модуль `src/services/tools/semantic_similarity.py`

Назначение: вычисление косинусной близости между текстами по эмбеддингам `Retriever`.

API:
- `semantic_similarity(texts: List[str], retriever: Retriever, pairs?: List[List[int]]) -> Dict`
  - Если `pairs` задан — возвращается список `{a,b,score}`; иначе — полная матрица N×N.

Детали:
- Эмбеддинги через `Retriever.embed_texts`
- Косинусная мера, диагональ = 1.0
