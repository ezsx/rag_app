### Модуль `src/schemas/agent.py`

Добавляет единый контракт Tool API:
- `ToolRequest{ tool:str, input:dict }`
- `ToolMeta{ took_ms:int, error?:str }`
- `ToolResponse{ ok:bool, data:dict, meta:ToolMeta }`
- `AgentAction{ step:int, tool:str, input:dict, output:ToolResponse }`

Используется `ToolRunner` для трассировки шагов «короткого пути».


