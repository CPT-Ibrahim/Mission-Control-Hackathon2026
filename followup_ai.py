"""
followup_ai.py — builds follow-up dashboard cards from grouped email topics.

For each topic it returns:
  - topic        : the topic name
  - include      : False if it's mostly promotional (hide from dashboard)
  - status       : "Needs action" | "Waiting" | "Completed"
  - status_detail: 3–7 word description of the status
  - last_update  : YYYY-MM-DD date of most recent relevant email
  - next_step    : short action for the user, or "None"
  - priority     : 1–5 (5 = highest)
  - accuracy     : 0–100 model confidence
"""

from typing import List, Dict, Any
from utils import get_deepseek_client, extract_json

# Module-level client — created once, reused on every call
_client = get_deepseek_client()

VALID_STATUSES = {"Needs action", "Waiting", "Completed"}


def summarize_followups(topic_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Send grouped email topics to DeepSeek and get back dashboard card data.
    Returns a cleaned list of dicts, one per topic.
    """
    system = (
        "You produce a FOLLOW-UP dashboard from grouped emails.\n"
        "Return ONLY valid JSON.\n"
        "Schema: {\"items\": ["
        "{\"topic\":\"DVSA\","
        "\"include\":true/false,"
        "\"status\":\"Needs action|Waiting|Completed\","
        "\"status_detail\":\"3-7 words\","
        "\"last_update\":\"YYYY-MM-DD\","
        "\"next_step\":\"short action or None\","
        "\"priority\":1-5,"
        "\"accuracy\":0-100"
        "]}.\n\n"
        "Rules:\n"
        "- If mostly promotional/marketing/newsletter: include=false.\n"
        "- Completion inferred from the LATEST email: payment confirmed, order complete, ticket confirmed.\n"
        "- Needs action: user must do something now.\n"
        "- Waiting: awaiting a response or processing.\n"
        "- priority 5 = highest.\n"
    )

    resp = _client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str({"topics": topic_payloads})},
        ],
        stream=False,
    )

    items = extract_json(resp.choices[0].message.content or "").get("items", [])

    cleaned = []
    for it in items:
        topic = (it.get("topic") or "").strip()[:28]
        if not topic:
            continue

        status = it.get("status", "Waiting")
        if status not in VALID_STATUSES:
            status = "Waiting"

        try:
            priority = max(1, min(5, int(it.get("priority", 3))))
        except Exception:
            priority = 3

        try:
            accuracy = max(0.0, min(100.0, float(it.get("accuracy", 70))))
        except Exception:
            accuracy = 70.0

        cleaned.append({
            "topic": topic,
            "include": bool(it.get("include", True)),
            "status": status,
            "status_detail": (it.get("status_detail") or "").strip()[:60],
            "last_update": (it.get("last_update") or "").strip()[:10],
            "next_step": (it.get("next_step") or "").strip()[:90] or "None",
            "priority": priority,
            "accuracy": accuracy,
        })

    return cleaned
