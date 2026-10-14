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

DEFAULT_SECRET_KEY = "MetSiq2G3wpDGnrtS5w6rBf17KnFZJRr"


def get_jwt_secret() -> str:
    """Returns JWT_SECRET_KEY from environment or fallback default."""
    return os.getenv("JWT_SECRET_KEY") or DEFAULT_SECRET_KEY


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
    except Exception:
        pass
    return None


def lookup_user_profile(user_id: Optional[str] = None, email: Optional[str] = None) -> Dict[str, Any]:
    """Looks up user details from kp_users collection in MongoDB or derives friendly name from email."""
    result: Dict[str, Any] = {}
    
    # 1. Query MongoDB kp_users collection if database is reachable
    try:
        from db.connection import get_db
        from bson import ObjectId
        db = get_db()
        queries = []
        if user_id:
            uid_str = str(user_id).strip()
            if ObjectId.is_valid(uid_str):
                queries.append({"_id": ObjectId(uid_str)})
            queries.append({"_id": uid_str})
            queries.append({"user_id": uid_str})
            queries.append({"id": uid_str})
        if email:
            clean_email = email.strip()
            queries.append({"email": clean_email.lower()})
            queries.append({"email": clean_email})

        for q in queries:
            doc = db.kp_users.find_one(q)
            if doc:
                first = (doc.get("first_name") or "").strip()
                last = (doc.get("last_name") or "").strip()
                combined = f"{first} {last}".strip()
                full = combined or doc.get("name") or doc.get("full_name") or doc.get("display_name")
                result = {
                    "id": str(doc.get("_id", user_id)),
                    "first_name": first or None,
                    "last_name": last or None,
                    "name": full or None,
                    "email": doc.get("email") or email,
                    "company": doc.get("company"),
                    "job_title": doc.get("job_title"),
                    "role": doc.get("role", "user")
                }
                break
    except Exception:
        pass

    # 2. If name is still missing or matches raw ID, derive a friendly human name from email
    target_email = result.get("email") or email
    current_name = result.get("name")
    if target_email and (not current_name or current_name == user_id or (user_id and current_name.startswith(f"Operator ({user_id[:8]}")) or (len(str(current_name)) >= 16 and all(c in "0123456789abcdefABCDEF" for c in str(current_name)))):
        local_part = target_email.split("@")[0].strip()
        import re
        alpha_match = re.match(r"^([a-zA-Z]+)", local_part)
        if alpha_match and len(alpha_match.group(1)) >= 2:
            derived = alpha_match.group(1).capitalize()
            result["name"] = derived
            if not result.get("first_name"):
                result["first_name"] = derived
        else:
            clean_local = re.sub(r"[._-]+", " ", local_part).strip().title()
            if clean_local:
                result["name"] = clean_local

    return result


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

    # Enrich from database kp_users collection
    db_profile = lookup_user_profile(user_id=str(user_id) if user_id else None, email=str(email) if email else None)
    if db_profile:
        if not name or name == user_id or (user_id and str(name).startswith(f"Operator ({str(user_id)[:8]}")):
            name = db_profile.get("name") or name
        if not first_name:
            first_name = db_profile.get("first_name") or first_name
        if not last_name:
            last_name = db_profile.get("last_name") or last_name
        if not email:
            email = db_profile.get("email") or email

    metadata = {
        "id": str(user_id) if user_id else None,
        "email": str(email) if email else None,
        "name": str(name) if name else None,
        "role": payload.get("role") or db_profile.get("role", "user"),
        "first_name": first_name or db_profile.get("first_name"),
        "last_name": last_name or db_profile.get("last_name"),
        "company": payload.get("company") or db_profile.get("company"),
        "job_title": payload.get("job_title") or db_profile.get("job_title"),
    }
    return str(user_id) if user_id else "", metadata


@router.post("/auth/exchange", response_model=AuthTokenResponse)
@router.post("/auth/code", response_model=AuthTokenResponse)
@router.post("/auth/login", response_model=AuthTokenResponse)
def exchange_code(req: ExchangeCodeRequest):
    """
    Exchanges authorization code or JWT token for a session access token.
    Validates code/token using JWT_SECRET_KEY and extracts user identity.
    Rejects unauthorized tokens with 401.
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

    # 2. Attempt to decode and cryptographically verify code_str as a signed JWT token
    if not extracted_user_id:
        try:
            payload = jwt.decode(
                code_str,
                secret,
                algorithms=["HS256", "HS384", "HS512"],
                options={"verify_signature": True, "verify_exp": True}
            )
            extracted_user_id, user_metadata = extract_user_info_from_payload(payload)
        except Exception:
            pass

    # If neither verification succeeds, the user is NOT authorized
    if not extracted_user_id:
        raise HTTPException(
            status_code=401,
            detail="You are not authorized to enter: Invalid authentication credentials or exchange code. Please log in at https://agents.kweka.ai/login"
        )

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
            secret = get_jwt_secret()
            payload = jwt.decode(token, secret, algorithms=["HS256", "HS384", "HS512"], options={"verify_signature": True})
            _, meta = extract_user_info_from_payload(payload)
            name = meta.get("name")
            email = meta.get("email")
            role = meta.get("role", "user")
        except Exception:
            pass

    return {
        "user_id": user_id,
        "name": name or user_id,
        "first_name": meta.get("first_name"),
        "last_name": meta.get("last_name"),
        "company": meta.get("company"),
        "job_title": meta.get("job_title"),
        "email": email or "",
        "role": role,
        "authenticated": True,
        "status": "active"
    }

