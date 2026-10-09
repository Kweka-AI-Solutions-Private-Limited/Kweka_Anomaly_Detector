import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from google import genai
from google.genai import types
from services.vlm_service import get_gemini_api_key
from services.gemini_pipeline_b_service import GEMINI_PIPELINE_B_PROMPT

api_key = get_gemini_api_key()
client = genai.Client(api_key=api_key)

image_path = Path("c:/dev/Anomaly_Detector/backend/storage/test_verification_artifacts/grid_2x2.png")
with open(image_path, "rb") as f:
    image_bytes = f.read()

contents = [
    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
    GEMINI_PIPELINE_B_PROMPT
]

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_budget=1024)
    )
)

print("=== CANDIDATES COUNT ===", len(response.candidates or []))
if response.candidates:
    cand = response.candidates[0]
    print("=== PARTS COUNT ===", len(cand.content.parts or []))
    for i, p in enumerate(cand.content.parts or []):
        is_thought = getattr(p, "thought", False)
        text_content = getattr(p, "text", "")
        print(f"Part {i}: thought={is_thought}, text_len={len(text_content or '')}")
        if text_content and not is_thought:
            print(f"TEXT SNIPPET: {text_content[:200]}")
