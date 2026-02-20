# llm_triage.py
import json
import re
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
    # Try direct JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: find first JSON object in the text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON found in model output.")
    return json.loads(m.group(0))

def triage_emails(emails: List[dict], model: str = "gpt-5-mini") -> List[dict]:
    if not emails:
        return []

    client = OpenAI()

    system = (
        "You are an email triage classifier. "
        f"Allowed topics: {', '.join(TOPICS)}. "
        f"Allowed actions: {', '.join(ACTIONS)}. "
        "Return ONLY valid JSON, no extra text. "
        "JSON schema: {\"items\": ["
        "{\"id\": \"...\", \"topic\": \"...\", \"is_urgent\": true/false, \"is_spam\": true/false, "
        "\"summary\": \"one sentence\", \"action\": \"Reply|Read|Pay|Book|Follow-up|Ignore\", \"confidence\": 0.0-1.0}"
        "]} "
        "Rules: urgent for deadlines/payments/account issues/time-sensitive tasks. spam for scams/promotions/phishing/irrelevant marketing."
    )

    # Keep payload small: metadata + snippet only
    lines = []
    for e in emails:
        lines.append(
            {
                "id": e.get("id", ""),
                "from": e.get("from", ""),
                "subject": e.get("subject", ""),
                "date": e.get("date", ""),
                "snippet": e.get("snippet", ""),
            }
        )

    user = json.dumps({"emails": lines}, ensure_ascii=False)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    # Get text output robustly
    out_text = getattr(resp, "output_text", None)
    if not out_text:
        # fallback for dict-like responses
        try:
            out_text = resp["output_text"]
        except Exception:
            out_text = str(resp)

    data = _extract_json(out_text)
    items = data.get("items", [])
    cleaned = []

    for it in items:
        topic = it.get("topic", "Other")
        if topic not in TOPICS:
            topic = "Other"

        action = it.get("action", "Read")
        if action not in ACTIONS:
            action = "Read"

        conf = it.get("confidence", 0.5)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        cleaned.append(
            {
                "id": it.get("id", ""),
                "topic": topic,
                "is_urgent": bool(it.get("is_urgent", False)),
                "is_spam": bool(it.get("is_spam", False)),
                "summary": (it.get("summary", "") or "")[:200],
                "action": action,
                "confidence": conf,
            }
        )

    # Ensure each returned item has an id
    cleaned = [c for c in cleaned if c.get("id")]
    return cleaned
