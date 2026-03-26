"""
topic_summary.py — summarises a single email thread into bullet points.

Used in the detail view to give the user a quick overview of a thread
without having to read every email in it.
"""

from typing import List, Dict
from utils import get_deepseek_client

# Module-level client — created once, reused on every call
_client = get_deepseek_client()


def summarize_topic(topic: str, items: List[Dict], model: str = "deepseek-chat") -> str:
    """
    Given a topic name and its emails, return a plain-text bullet summary.
    Covers: current situation, latest update, next actions, deadlines, who to contact.
    """
    payload = [
        {
            "date": e.get("date", ""),
            "from": e.get("from", ""),
            "subject": e.get("subject", ""),
            "summary": (e.get("triage") or {}).get("summary", ""),
            "action": (e.get("triage") or {}).get("action", ""),
            "urgent": (e.get("triage") or {}).get("is_urgent", False),
        }
        for e in items[:12]
    ]

    system = (
        "Summarise this email thread for a dashboard. "
        "Return plain text bullet points only (max 6). "
        "Cover: current situation, latest update, next actions, deadlines/risks, who to contact."
    )

    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str({"topic": topic, "emails": payload})},
        ],
        stream=False,
    )

    return (resp.choices[0].message.content or "").strip()
