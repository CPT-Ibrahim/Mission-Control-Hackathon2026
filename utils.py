"""
utils.py — shared helpers used across multiple modules.

Keeping these here means if you ever need to fix or change one,
you only do it in one place instead of hunting across files.
"""

import re
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load .env as early as possible — utils is always the first import,
# so this guarantees env vars are available before any client is created.
load_dotenv()


def get_deepseek_client() -> OpenAI:
    """
    Return a DeepSeek client.
    Uses the openai package because DeepSeek's API is OpenAI-compatible —
    we just point it at DeepSeek's URL instead of OpenAI's.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing in .env")
    return OpenAI(api_key=api_key, base_url=base_url)


def extract_json(text: str) -> dict:
    """
    Parse JSON from a model response.
    Models sometimes wrap JSON in extra text or markdown fences,
    so we first try a clean parse, then fall back to regex extraction.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON found in model output.")
    return json.loads(m.group(0))


def norm_topic(t: str) -> str:
    """
    Normalise a topic label to a clean, short string.
    Strips special characters, collapses whitespace, caps at 28 chars.
    """
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s&\-]", "", t)
    return t[:28] if t else "Unsorted"