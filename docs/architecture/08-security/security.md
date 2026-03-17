## Security

### Auth Model

- JWT токены, HS256 (`JWT_SECRET_KEY` из `.env`)
- Dev endpoint: `POST /v1/auth/admin` с `ADMIN_KEY` → выдаёт JWT со всеми ролями
- `require_read` dependency на `/v1/agent/stream`
- `.env` файл не коммитится (`.gitignore`)

### Logging Policy

- JWT токены — **не логируются нигде**
- API ключи, ADMIN_KEY — **не логируются**
- Весь external input (query пользователя, параметры инструментов) — через `sanitize_for_logging()`
- `SecurityManager` — центральный guard для входящих данных
- Промпты с возможным PII не логируются полностью (truncation)

### Telegram Session

- Telethon session хранится в `./sessions/telegram.session`
- В Docker: volume mount `./sessions:/app/sessions`
- `TG_SESSION=/app/sessions/telegram.session` — путь в `.env`
- Файл сессии — не коммитится

### Потенциальные риски

| Риск | Статус |
|------|--------|
| ADMIN_KEY в .env → admin JWT | Приемлемо для single-user |
| LLM prompt injection через query | Базовый sanitize, нет полной защиты |
| Qdrant без auth (localhost / Docker network only) | Приемлемо для Docker-only deployment |
| llama-server без auth на 0.0.0.0:8080 | Приемлемо для локальной сети; не выставлять в интернет |
