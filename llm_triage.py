# llm_triage.py
import json
import re
import os
from typing import List, Dict, Any
from openai import OpenAI

TOPICS = [
    "Project 1",
    "University",
    "DVSA",
    "Tax",
    "Banking",
    "Bills",
    "Work",
    "Shopping",
    "Health",
    "Other",
]

ACTIONS = ["Reply", "Read", "Pay", "Book", "Follow-up", "Ignore"]

def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON found in model output.")
    return json.loads(m.group(0))

def triage_emails(emails: List[dict], model: str = "deepseek-chat") -> List[dict]:
    if not emails:
        return []

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing in .env")

    client = OpenAI(api_key=api_key, base_url=base_url)

    system = (
        "You are an email triage classifier. "
        f"Allowed topics: {', '.join(TOPICS)}. "
        f"Allowed actions: {', '.join(ACTIONS)}. "
        "Return ONLY valid JSON, no extra text. "
        "Schema: {\"items\": ["
        "{\"id\": \"...\", \"topic\": \"...\", \"is_urgent\": true/false, \"is_spam\": true/false, "
        "\"summary\": \"one sentence\", \"action\": \"Reply|Read|Pay|Book|Follow-up|Ignore\", \"confidence\": 0.0-1.0}"
        "]}. "
        "Urgent: deadlines, payments, account issues, cancellations, time-sensitive tasks. "
        "Spam: scams, promos, phishing, irrelevant marketing."
    )

    payload = []
    for e in emails:
        payload.append(
            {
                "id": e.get("id", ""),
                "from": e.get("from", ""),
                "subject": e.get("subject", ""),
                "date": e.get("date", ""),
                "snippet": e.get("snippet", ""),
            }
        )

    user = json.dumps({"emails": payload}, ensure_ascii=False)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=False,
    )

    out_text = resp.choices[0].message.content or ""
    data = _extract_json(out_text)

    items = data.get("items", [])
    cleaned = []
    for it in items:
        mid = it.get("id", "")
        if not mid:
            continue

        topic = it.get("topic", "Other")
        if topic not in TOPICS:
            topic = "Other"

        action = it.get("action", "Read")
        if action not in ACTIONS:
            action = "Read"

        try:
            conf = float(it.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        cleaned.append(
            {
                "id": mid,
                "topic": topic,
                "is_urgent": bool(it.get("is_urgent", False)),
                "is_spam": bool(it.get("is_spam", False)),
                "summary": (it.get("summary", "") or "")[:200],
                "action": action,
                "confidence": conf,
            }
        )

    return cleaned
