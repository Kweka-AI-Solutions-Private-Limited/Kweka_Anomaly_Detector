"""
deps.py — FastAPI Common Dependencies & User Identity Extraction
----------------------------------------------------------------
Provides user context dependencies for tenant/user data isolation.
"""

from fastapi import Request

def get_current_user_id(request: Request) -> str:
    """
    Extracts the user identity from X-User-ID header or user_id query parameter.
    Defaults to 'usr_default' if not provided by client.
    """
    header_val = request.headers.get("x-user-id")
    query_val = request.query_params.get("user_id")
    
    user_id = header_val or query_val or "usr_default"
    user_id_clean = user_id.strip() if isinstance(user_id, str) else "usr_default"
    
    return user_id_clean if user_id_clean else "usr_default"
