"""
deps.py — FastAPI Common Dependencies & JWT Authentication
----------------------------------------------------------
Provides strict user context dependencies for tenant/user data isolation.
Extracts and cryptographically verifies JWT Bearer tokens signed with JWT_SECRET_KEY.
Enforces that unauthenticated or unauthorized requests are rejected with HTTP 401.
"""

import os
import jwt
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException

DEFAULT_SECRET_KEY = "MetSiq2G3wpDGnrtS5w6rBf17KnFZJRr"


def get_jwt_secret() -> str:
    """Returns JWT_SECRET_KEY from environment or default."""
    return os.getenv("JWT_SECRET_KEY") or DEFAULT_SECRET_KEY


def decode_and_verify_token(token: str) -> Dict[str, Any]:
    """
    Cryptographically verifies and decodes a JWT Bearer token using JWT_SECRET_KEY.
    Raises HTTPException 401 if invalid, expired, or tampered.
    """
    secret = get_jwt_secret()
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256", "HS384", "HS512"],
            options={"verify_signature": True, "verify_exp": True}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="You are not authorized to enter: Session token has expired. Please log in at https://agents.kweka.ai/login"
        )
    except jwt.InvalidSignatureError:
        raise HTTPException(
            status_code=401,
            detail="You are not authorized to enter: Invalid token signature. Access denied."
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=401,
            detail=f"You are not authorized to enter: Invalid authentication token ({str(e)})."
        )


def get_current_user_id(request: Request) -> str:
    """
    Extracts and strictly verifies user identity from Authorization Bearer JWT Token signed with JWT_SECRET_KEY.
    Enforces tenant isolation by guaranteeing that user_id comes strictly from a cryptographically
    verified JWT payload. Rejects unauthenticated requests with HTTP 401 Unauthorized.
    """
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail="You are not authorized to enter: Missing Authorization header. Please log in at https://agents.kweka.ai/login"
        )

    parts = auth_header.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="You are not authorized to enter: Bearer token format required."
        )

    token = parts[1].strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="You are not authorized to enter: Token is empty."
        )

    payload = decode_and_verify_token(token)

    user_id = (
        payload.get("user_id") or
        payload.get("sub") or
        payload.get("id") or
        payload.get("email") or
        payload.get("uid")
    )

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="You are not authorized to enter: Token payload missing user identity."
        )

    return str(user_id).strip()

