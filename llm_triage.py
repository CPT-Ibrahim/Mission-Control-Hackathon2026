# llm_triage.py
import json
import re
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI

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

def _norm_topic(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s&\-]", "", t)
    return t[:28] if t else "Unsorted"

def triage_emails(
    emails: List[dict],
    model: str = "deepseek-chat",
    existing_topics: Optional[List[str]] = None,
) -> List[dict]:
    if not emails:
        return []

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing in .env")

    client = OpenAI(api_key=api_key, base_url=base_url)

    existing_topics = [_norm_topic(x) for x in (existing_topics or []) if _norm_topic(x)]
    existing_topics = sorted(list(dict.fromkeys(existing_topics)))[:60]

    system = (
        "You classify emails into a stable mailbox label (mail type) that groups related emails.\n"
        "Return ONLY valid JSON. No extra text.\n\n"
        "Schema: {\"items\": ["
        "{\"id\":\"...\","
        "\"topic\":\"...\","
        "\"is_urgent\":true/false,"
        "\"is_spam\":true/false,"
        "\"summary\":\"one short sentence\","
        "\"action\":\"Reply|Read|Pay|Book|Follow-up|Ignore\","
        "\"confidence\":0.0-1.0"
        "}]}.\n\n"
        "Topic rules:\n"
        "- 1–3 words, <=28 chars.\n"
        "- Use organization/system name when possible (DVSA, Devpost, NHS, HMRC, Bank, University, Amazon, Inditex).\n"
        "- Reuse existing topics when appropriate to avoid duplicates.\n"
        f"Existing topics to reuse when relevant: {existing_topics}\n\n"
        "Urgent: deadlines, payments due, account/security issues, cancellations, time-sensitive actions.\n"
        "Spam: scams, phishing, irrelevant promotions.\n"
        f"Allowed actions: {', '.join(ACTIONS)}.\n"
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
        model="deepseek-chat",  # forced
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        stream=False,
    )

    out_text = resp.choices[0].message.content or ""
    data = _extract_json(out_text)
    items = data.get("items", [])

    cleaned = []
    for it in items:
        mid = (it.get("id") or "").strip()
        if not mid:
            continue

        topic = _norm_topic(it.get("topic", ""))

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
                "summary": (it.get("summary", "") or "")[:220],
                "action": action,
                "confidence": conf,
            }
        )

    return cleaned