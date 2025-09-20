### Модуль `src/services/tools/dedup_diversify.py`

Функция `dedup_diversify` выполняет:
- логическую дедупликацию кандидатов (по `id`, URL, хешу текста),
- упрощённый MMR‑отбор top‑k по имеющимся эмбеддингам.

Вход: `hits[{id,text,score?,metadata,embedding?}], lambda:float, k:int`.
Выход: `{hits:[...]}` — исходный формат, упорядоченный.


