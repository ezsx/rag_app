"""
Dev auth endpoint: POST /v1/auth/admin — выдает JWT с admin/write/read по ADMIN_KEY
"""

import os
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.auth import JWT_SECRET, JWT_ALGORITHM
import jwt


class AdminLoginRequest(BaseModel):
    key: str = Field(..., description="Значение ADMIN_KEY из окружения")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = Field("bearer", description="Тип токена")
    expires_in_hours: int = Field(24, ge=1, description="Срок жизни токена в часах")


router = APIRouter()


@router.post("/auth/admin", tags=["auth"], response_model=TokenResponse)
def admin_login(body: AdminLoginRequest) -> TokenResponse:
    """
    Временный dev-логин: принимает {"key": "..."} (ADMIN_KEY) и выдает JWT со скоупами
    [admin, write, read] на 24 часа.

    Пример запроса:
    {
      "key": "my_admin_key"
    }
    """
    admin_key_env = os.getenv("ADMIN_KEY")
    if not admin_key_env:
        raise HTTPException(status_code=503, detail="ADMIN_KEY не сконфигурирован")

    if not body.key or body.key != admin_key_env:
        raise HTTPException(status_code=401, detail="Неверный ключ")

    now = datetime.utcnow()
    payload: dict[str, Any] = {
        "sub": "admin",
        "iat": now,
        "exp": now + timedelta(hours=24),
        "scopes": ["admin", "write", "read"],
        "metadata": {"auth_method": "admin_key"},
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return TokenResponse(access_token=token)
