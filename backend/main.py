"""
Main FastAPI entrypoint wrapper for InspectAI backend.
Allows running:
  uvicorn main:app --reload
  python main.py
"""

import sys
from pathlib import Path

# Add src/ to sys.path so all imports resolve seamlessly
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from api_server import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
