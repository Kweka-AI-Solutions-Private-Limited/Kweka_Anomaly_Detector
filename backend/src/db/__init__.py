"""
InspectAI Database Module
"""

from .connection import get_db, get_client, close_connection
from .indexes import ensure_indexes

__all__ = ["get_db", "get_client", "close_connection", "ensure_indexes"]
