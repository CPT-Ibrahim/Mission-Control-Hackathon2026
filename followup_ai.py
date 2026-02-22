# followup_ai.py
import json
import re
import os
from typing import List, Dict, Any
from openai import OpenAI


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON found in model output.")
    return json.loads(m.group(0))


def summarize_followups(topic_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing in .env")

    client = OpenAI(api_key=api_key, base_url=base_url)

    system = (
        "You produce a FOLLOW-UP dashboard from grouped emails.\n"
        "Return ONLY valid JSON. No extra text.\n"
        "Schema: {\"items\": ["
        "{\"topic\":\"DVSA\","
        "\"status\":\"Needs action|Waiting|Completed\","
        "\"status_detail\":\"3-7 words\","
        "\"last_update\":\"YYYY-MM-DD\","
        "\"next_step\":\"short action or None\","
        "\"priority\":1-5,"
        "\"accuracy\":0-100"
        "}]}.\n"
        "Rules:\n"
        "- 'Needs action' only if user must do something.\n"
        "- 'Waiting' if awaiting reply/processing.\n"
        "- 'Completed' only if latest indicates done.\n"
        "- priority 5 = most important.\n"
        "- accuracy is your confidence 0-100.\n"
    )

    user = json.dumps({"topics": topic_payloads}, ensure_ascii=False)

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        stream=False,
    )

    data = _extract_json(resp.choices[0].message.content or "")
    items = data.get("items", [])

    cleaned: List[Dict[str, Any]] = []
    for it in items:
        topic = (it.get("topic") or "").strip()[:28]
        if not topic:
            continue

        status = it.get("status", "Waiting")
        if status not in ["Needs action", "Waiting", "Completed"]:
            status = "Waiting"

        detail = (it.get("status_detail") or "").strip()[:60]
        last_update = (it.get("last_update") or "").strip()[:10]
        next_step = (it.get("next_step") or "").strip()[:90]

        pr = it.get("priority", 3)
        try:
            pr = int(pr)
        except Exception:
            pr = 3
        pr = max(1, min(5, pr))

        acc = it.get("accuracy", 70)
        try:
            acc = float(acc)
        except Exception:
            acc = 70.0
        acc = max(0.0, min(100.0, acc))

        cleaned.append(
            {
                "topic": topic,
                "status": status,
                "status_detail": detail,
                "last_update": last_update,
                "next_step": next_step if next_step else "None",
                "priority": pr,
                "accuracy": acc,
            }
        )

    return cleaned