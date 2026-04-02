"""
Модуль аутентификации и авторизации для API
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# FIX-05: hard fail без explicit dev mode
_DEV_MODE = os.getenv("RAG_ENV", "production").lower() in ("dev", "development", "local")
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    if _DEV_MODE:
        JWT_SECRET = "dev-only-insecure-secret"
        logger.warning("JWT_SECRET not set — using insecure dev default (RAG_ENV=%s)",
                       os.getenv("RAG_ENV"))
    else:
        raise RuntimeError(
            "JWT_SECRET environment variable is required in production. "
            "Set RAG_ENV=dev to use insecure default for local development."
        )
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# API Key authentication as alternative
API_KEY_HEADER = "X-API-Key"
VALID_API_KEYS = set(filter(None, os.getenv("VALID_API_KEYS", "").split(",")))

security = HTTPBearer(auto_error=False)


class TokenData(BaseModel):
    """JWT токен данные"""

    sub: str  # Subject (user_id или api_key_id)
    exp: datetime
    iat: datetime
    scopes: list[str] = []
    metadata: dict[str, Any] = {}


def create_access_token(
    subject: str, scopes: list[str] | None = None, metadata: dict[str, Any] | None = None
) -> str:
    """Создает JWT токен"""
    now = datetime.utcnow()
    expire = now + timedelta(hours=JWT_EXPIRATION_HOURS)

    payload = {
        "sub": subject,
        "exp": expire,
        "iat": now,
        "scopes": scopes or [],
        "metadata": metadata or {},
    }

    assert JWT_SECRET is not None, "JWT_SECRET must be set"
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> TokenData:
    """Проверяет и декодирует JWT токен"""
    try:
        assert JWT_SECRET is not None, "JWT_SECRET must be set"
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenData(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> TokenData:
    """Dependency для проверки аутентификации"""

    # Проверка API Key в заголовке
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key and api_key in VALID_API_KEYS:
        # Создаем фиктивный токен для API key
        return TokenData(
            sub=f"api_key:{api_key[:8]}",
            exp=datetime.utcnow() + timedelta(days=365),
            iat=datetime.utcnow(),
            scopes=["api_key"],
            metadata={"auth_method": "api_key"},
        )

    # Проверка JWT токена
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return verify_token(credentials.credentials)


def require_scopes(*required_scopes: str):
    """Dependency factory для проверки scopes"""

    async def check_scopes(user: TokenData = Depends(get_current_user)) -> TokenData:
        if not required_scopes:
            return user

        user_scopes = set(user.scopes)
        required = set(required_scopes)

        if not required.intersection(user_scopes):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {required_scopes}",
            )

        return user

    return check_scopes


# Shortcuts для общих разрешений
require_read = require_scopes("read")
require_write = require_scopes("write", "admin")
require_admin = require_scopes("admin")
