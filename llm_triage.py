"""
llm_triage.py — classifies a batch of emails using DeepSeek.

For each email it returns:
  - topic     : short label grouping related emails (e.g. "DVSA", "NHS")
  - is_urgent : bool — needs attention now
  - is_spam   : bool — promotional / irrelevant
  - summary   : one-sentence description
  - action    : what the user should do (Reply / Read / Pay / Book / Follow-up / Ignore)
  - confidence: 0.0–1.0 how sure the model is
"""

from typing import List, Optional
from utils import get_deepseek_client, extract_json, norm_topic

# Module-level client — created once, reused on every call
_client = get_deepseek_client()

ACTIONS = ["Reply", "Read", "Pay", "Book", "Follow-up", "Ignore"]


def triage_emails(
    emails: List[dict],
    model: str = "deepseek-chat",
    existing_topics: Optional[List[str]] = None,
) -> List[dict]:
    """
    Send a batch of emails to DeepSeek and get back classification data.
    Returns a cleaned list of dicts, one per email.
    """
    if not emails:
        return []

    # Normalise existing topics so the model can reuse them
    existing_topics = [norm_topic(x) for x in (existing_topics or []) if norm_topic(x)]
    existing_topics = sorted(set(existing_topics))[:60]

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
        "]}.\n\n"
        "Topic rules:\n"
        "- 1–3 words, <=28 chars.\n"
        "- Prefer organisation/system label: DVSA, Devpost, NHS, HMRC, Bank, University, Amazon.\n"
        f"- Reuse existing topics when relevant: {existing_topics}\n\n"
        "CRITICAL — promotions:\n"
        "- If promotional/marketing/discount/offer/newsletter: is_spam=true, is_urgent=false, action=Ignore.\n\n"
        "Urgent only for: deadlines, payments due, account/security issues, time-sensitive actions.\n"
        f"Allowed actions: {', '.join(ACTIONS)}.\n"
    )

    payload = [
        {
            "id": e.get("id", ""),
            "from": e.get("from", ""),
            "subject": e.get("subject", ""),
            "date": e.get("date", ""),
            "snippet": e.get("snippet", ""),
        }
        for e in emails
    ]

    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str(payload)},
        ],
        stream=False,
    )

    items = extract_json(resp.choices[0].message.content or "").get("items", [])

    cleaned = []
    for it in items:
        mid = (it.get("id") or "").strip()
        if not mid:
            continue

        action = it.get("action", "Read")
        if action not in ACTIONS:
            action = "Read"

        try:
            conf = max(0.0, min(1.0, float(it.get("confidence", 0.5))))
        except Exception:
            conf = 0.5

        cleaned.append({
            "id": mid,
            "topic": norm_topic(it.get("topic", "")),
            "is_urgent": bool(it.get("is_urgent", False)),
            "is_spam": bool(it.get("is_spam", False)),
            "summary": (it.get("summary", "") or "")[:220],
            "action": action,
            "confidence": conf,
        })

    return cleaned
