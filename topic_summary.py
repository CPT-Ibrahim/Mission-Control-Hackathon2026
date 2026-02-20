# topic_summary.py
import json
import os
from typing import List, Dict
from openai import OpenAI

def summarize_topic(topic: str, items: List[Dict], model: str = "deepseek-chat") -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing in .env")

    client = OpenAI(api_key=api_key, base_url=base_url)

    payload = []
    for e in items[:12]:
        tri = e.get("triage") or {}
        payload.append(
            {
                "date": e.get("date", ""),
                "from": e.get("from", ""),
                "subject": e.get("subject", ""),
                "summary": tri.get("summary", ""),
                "action": tri.get("action", ""),
                "urgent": tri.get("is_urgent", False),
            }
        )

    system = (
        "Summarize this topic for a dashboard. "
        "Return plain text bullet points only (max 6). "
        "Include: current situation, latest update, next actions, deadlines/risks, who to contact."
    )

    user = json.dumps({"topic": topic, "emails": payload}, ensure_ascii=False)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=False,
    )

    return (resp.choices[0].message.content or "").strip()
