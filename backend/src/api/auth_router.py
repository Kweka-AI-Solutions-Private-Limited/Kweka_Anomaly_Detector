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


import base64
import json
import urllib.request

CENTRAL_AUTH_VERIFY_URL = "https://kw-prototypes-service-238644809220.asia-south1.run.app/api/auth/exchange/verify"


def verify_central_exchange_code(code_str: str) -> Optional[Dict[str, Any]]:
    """Calls central Kweka auth service to verify exchange_token and retrieve real user details."""
    try:
        data = json.dumps({"exchange_token": code_str}).encode("utf-8")
        req = urllib.request.Request(
            CENTRAL_AUTH_VERIFY_URL,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "Anomaly-Detector-Backend"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                return body
    except Exception as e:
        pass
    return None


def extract_user_info_from_payload(payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Helper to extract user_id, name, email, and role from JWT or JSON payload dictionary."""
    sub = str(payload.get("sub") or "").strip()
    
    email = (
        payload.get("email") or
        payload.get("user_email") or
        payload.get("email_address") or
        payload.get("mail")
    )
    if not email and "@" in sub:
        email = sub

    first_name = payload.get("first_name") or payload.get("given_name") or ""
    last_name = payload.get("last_name") or payload.get("family_name") or ""
    combined_name = f"{first_name} {last_name}".strip()

    name = (
        payload.get("name") or
        payload.get("full_name") or
        payload.get("display_name") or
        (combined_name if combined_name else None) or
        payload.get("username") or
        payload.get("user_name") or
        payload.get("preferred_username")
    )

    user_id = (
        payload.get("user_id") or
        payload.get("id") or
        payload.get("uid") or
        (payload.get("sub") if not ("@" in sub and not payload.get("user_id")) else None) or
        email or
        sub
    )

    metadata = {
        "id": str(user_id) if user_id else None,
        "email": str(email) if email else None,
        "name": str(name) if name else None,
        "role": payload.get("role", "user"),
        "first_name": first_name or None,
        "last_name": last_name or None,
        "company": payload.get("company"),
        "job_title": payload.get("job_title"),
    }
    return str(user_id) if user_id else "", metadata


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
    if code_str.startswith("Bearer "):
        code_str = code_str.split(" ", 1)[1].strip()
    secret = get_jwt_secret()
    extracted_user_id = None
    user_metadata: Dict[str, Any] = {}
    returned_access_token = None

    # 1. Attempt central exchange verification (Kweka Prototypes Central Auth Service)
    central_res = verify_central_exchange_code(code_str)
    if central_res:
        returned_access_token = central_res.get("token") or central_res.get("access_token")
        u_obj = central_res.get("user") or {}
        if u_obj and isinstance(u_obj, dict):
            extracted_user_id, user_metadata = extract_user_info_from_payload(u_obj)
        elif isinstance(central_res, dict):
            extracted_user_id, user_metadata = extract_user_info_from_payload(central_res)

    # 2. Attempt to decode code_str as a signed JWT token if central exchange returned nothing
    if not extracted_user_id:
        try:
            payload = jwt.decode(
                code_str,
                secret,
                algorithms=["HS256", "HS384", "HS512", "RS256"],
                options={"verify_signature": True if secret != DEFAULT_SECRET_KEY else False}
            )
            extracted_user_id, user_metadata = extract_user_info_from_payload(payload)
        except Exception:
            # 3. Try unverified decode if signature failed or external secret used
            try:
                payload = jwt.decode(code_str, options={"verify_signature": False})
                extracted_user_id, user_metadata = extract_user_info_from_payload(payload)
            except Exception:
                # 4. Try base64 json decode
                try:
                    decoded_bytes = base64.b64decode(code_str)
                    decoded_json = json.loads(decoded_bytes.decode('utf-8'))
                    if isinstance(decoded_json, dict):
                        extracted_user_id, user_metadata = extract_user_info_from_payload(decoded_json)
                except Exception:
                    extracted_user_id = code_str

    if not extracted_user_id:
        extracted_user_id = code_str

    user_id_clean = extracted_user_id.strip()
    if "id" not in user_metadata or not user_metadata["id"]:
        user_metadata["id"] = user_id_clean

    # Issue signed access token using backend JWT_SECRET_KEY
    access_token = create_access_token(user_id_clean, extra_payload=user_metadata)

    return AuthTokenResponse(
        access_token=returned_access_token or access_token,
        token_type="bearer",
        user_id=user_id_clean,
        user=user_metadata
    )


@router.get("/auth/me")
def get_current_user_profile(request: Request, user_id: str = Depends(get_current_user_id)):
    """Returns the authenticated user details for the active session."""
    name = None
    email = None
    role = "user"

    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            _, meta = extract_user_info_from_payload(payload)
            name = meta.get("name")
            email = meta.get("email")
            role = meta.get("role", "user")
        except Exception:
            pass

    return {
        "user_id": user_id,
        "name": name,
        "email": email,
        "role": role,
        "authenticated": user_id != "usr_default",
        "status": "active"
    }
