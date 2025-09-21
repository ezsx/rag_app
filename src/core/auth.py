"""
Модуль аутентификации и авторизации для API
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Конфигурация
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
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
    metadata: Dict[str, Any] = {}


def create_access_token(
    subject: str, scopes: list[str] = None, metadata: Dict[str, Any] = None
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

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> TokenData:
    """Проверяет и декодирует JWT токен"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenData(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
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
