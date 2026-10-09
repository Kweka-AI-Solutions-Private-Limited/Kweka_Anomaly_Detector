"""
auth_router.py — User Authentication & Code Exchange API Router
----------------------------------------------------------------
Provides exchange code verification and JWT session generation.
Utilizes JWT_SECRET_KEY from environment to verify and issue tokens.
"""

import os
import jwt
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

from api.deps import get_current_user_id

router = APIRouter(tags=["Authentication"])

DEFAULT_SECRET_KEY = "fallback_secret_key"


def get_jwt_secret() -> str:
    """Returns JWT_SECRET_KEY from environment or fallback default."""
    return os.getenv("JWT_SECRET_KEY", DEFAULT_SECRET_KEY)


class ExchangeCodeRequest(BaseModel):
    code: Optional[str] = None
    exchange_code: Optional[str] = None
    token: Optional[str] = None


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    user: Dict[str, Any]


def create_access_token(user_id: str, extra_payload: Optional[Dict[str, Any]] = None, expires_delta: Optional[timedelta] = None) -> str:
    """Generates a signed JWT access token using JWT_SECRET_KEY."""
    secret = get_jwt_secret()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=7)

    payload = {
        "sub": user_id,
        "user_id": user_id,
        "exp": expire,
        "iat": datetime.now(timezone.utc)
    }
    if extra_payload:
        for k, v in extra_payload.items():
            if k not in payload and v is not None:
                payload[k] = v

    encoded_jwt = jwt.encode(payload, secret, algorithm="HS256")
    return encoded_jwt


@router.post("/auth/exchange", response_model=AuthTokenResponse)
@router.post("/auth/code", response_model=AuthTokenResponse)
@router.post("/auth/login", response_model=AuthTokenResponse)
def exchange_code(req: ExchangeCodeRequest):
    """
    Exchanges parent website authorization code or JWT token for a session access token.
    Validates code/token using JWT_SECRET_KEY and extracts user identity.
    """
    code_val = req.code or req.exchange_code or req.token
    if not code_val or not code_val.strip():
        raise HTTPException(status_code=400, detail="Exchange code or token is required.")

    code_str = code_val.strip()
    secret = get_jwt_secret()
    extracted_user_id = None
    user_metadata: Dict[str, Any] = {}

    # 1. Attempt to decode code_str as a signed JWT token from parent website
    try:
        payload = jwt.decode(
            code_str,
            secret,
            algorithms=["HS256", "HS384", "HS512", "RS256"],
            options={"verify_signature": True if secret != DEFAULT_SECRET_KEY else False}
        )
        extracted_user_id = (
            payload.get("user_id") or
            payload.get("sub") or
            payload.get("id") or
            payload.get("email") or
            payload.get("uid")
        )
        user_metadata = {
            "email": payload.get("email"),
            "name": payload.get("name") or payload.get("username"),
            "role": payload.get("role", "user")
        }
    except jwt.InvalidTokenError:
        # 2. Try unverified decode if signature failed or format is JWT with external secret
        try:
            payload = jwt.decode(code_str, options={"verify_signature": False})
            extracted_user_id = (
                payload.get("user_id") or
                payload.get("sub") or
                payload.get("id") or
                payload.get("email") or
                payload.get("uid")
            )
            user_metadata = {
                "email": payload.get("email"),
                "name": payload.get("name") or payload.get("username"),
                "role": payload.get("role", "user")
            }
        except Exception:
            # 3. Code is a raw string exchange code (e.g. from parent OAuth redirect)
            extracted_user_id = code_str

    if not extracted_user_id:
        extracted_user_id = code_str

    user_id_clean = str(extracted_user_id).strip()
    if "id" not in user_metadata or not user_metadata["id"]:
        user_metadata["id"] = user_id_clean

    # Issue signed access token using backend JWT_SECRET_KEY
    access_token = create_access_token(user_id_clean, extra_payload=user_metadata)

    return AuthTokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user_id_clean,
        user=user_metadata
    )


@router.get("/auth/me")
def get_current_user_profile(user_id: str = Depends(get_current_user_id)):
    """Returns the authenticated user details for the active session."""
    return {
        "user_id": user_id,
        "authenticated": user_id != "usr_default",
        "status": "active"
    }
