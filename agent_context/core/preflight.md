# Preflight — обязательно перед каждой задачей

> Этот файл загружается ВСЕГДА. Короткий. Не пропускать.

## Перед началом работы — определи и озвучь:

1. **Task type**: debug / implementation / review / docs / research / eval
2. **Modules to load**:
   - Любой поиск по репо → `agent_context/core/tool_policy.md`
   - Debug / failure / unexpected behavior → `agent_context/modules/debugging_protocol.md`
   - Review / handoff / multi-agent → `agent_context/modules/parallel_agents.md`
   - ReAct agent code → `agent_context/modules/agent.md`
   - Retrieval pipeline → `agent_context/modules/retrieval.md`
   - Docs changes → `docs/architecture/00-meta/02-documentation-governance.md`
3. **First step**: что конкретно делаешь первым

## Operational rules (всегда активны)

- **Tool policy**: MCP-first. Не Grep/Glob при живом MCP.
- **Debugging**: no fix before root cause (для runtime bugs). Static bugs — прямой fix с объяснением.
- **Parallel work**: owner + sidecar, no shared-file parallel edits.
- **Commits**: только когда пользователь просит. Не амендить, не force push.
- **Docs**: следовать governance. Research → docs/research/, specs → docs/specifications/, architecture → docs/architecture/.
