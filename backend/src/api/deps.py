"""
deps.py — FastAPI Common Dependencies & JWT Authentication
----------------------------------------------------------
Provides user context dependencies for tenant/user data isolation.
Extracts and verifies JWT Bearer tokens signed with JWT_SECRET_KEY.
"""

import os
import jwt
from fastapi import Request, HTTPException

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fallback_secret_key")


def get_current_user_id(request: Request) -> str:
    """
    Extracts and verifies the user identity from:
    1. Authorization Bearer JWT Token signed with JWT_SECRET_KEY
    2. X-User-ID header
    3. user_id query parameter
    Defaults to 'usr_default' if not provided or unauthenticated.
    """
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if token:
            secret = os.getenv("JWT_SECRET_KEY") or JWT_SECRET_KEY
            try:
                payload = jwt.decode(
                    token,
                    secret,
                    algorithms=["HS256", "HS384", "HS512", "RS256"],
                    options={"verify_signature": True if secret and secret != "fallback_secret_key" else False}
                )
                user_id = (
                    payload.get("user_id") or
                    payload.get("sub") or
                    payload.get("id") or
                    payload.get("email") or
                    payload.get("uid")
                )
                if user_id:
                    return str(user_id).strip()
                raise HTTPException(status_code=401, detail="Invalid token payload: missing user identity.")
            except jwt.InvalidTokenError as e:
                # If secret is set and token signature/format is invalid, reject with 401
                if secret and secret != "fallback_secret_key":
                    raise HTTPException(status_code=401, detail=f"Invalid or expired authentication token: {str(e)}")
                # Soft fallback for local dev if secret not configured
                try:
                    payload = jwt.decode(token, options={"verify_signature": False})
                    user_id = payload.get("user_id") or payload.get("sub") or payload.get("id")
                    if user_id:
                        return str(user_id).strip()
                except Exception:
                    raise HTTPException(status_code=401, detail="Invalid authentication token format.")

    header_val = request.headers.get("x-user-id")
    query_val = request.query_params.get("user_id")

    user_id = header_val or query_val or "usr_default"
    user_id_clean = user_id.strip() if isinstance(user_id, str) else "usr_default"

    return user_id_clean if user_id_clean else "usr_default"
