import os
import base64
import re
from typing import Tuple, Dict, Any

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def list_message_ids(service, max_results=50, query=None, label_ids=None, include_spam_trash=False):
    res = service.users().messages().list(
        userId="me",
        maxResults=max_results,
        q=query,
        labelIds=label_ids,
        includeSpamTrash=include_spam_trash,
    ).execute()
    return [m["id"] for m in res.get("messages", [])]


def get_message_metadata(service, msg_id):
    msg = service.users().messages().get(
        userId="me",
        id=msg_id,
        format="metadata",
        metadataHeaders=["From", "Subject", "Date"],
    ).execute()

    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
    return {
        "id": msg_id,
        "threadId": msg.get("threadId"),
        "from": headers.get("From", ""),
        "subject": headers.get("Subject", ""),
        "date": headers.get("Date", ""),
        "snippet": msg.get("snippet", ""),
    }


def _b64url_decode(data: str) -> str:
    if not data:
        return ""
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding).decode("utf-8", errors="replace")


def _strip_html(html_text: str) -> str:
    # Basic stripping for display (no extra deps)
    text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html_text)
    text = re.sub(r"(?s)<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_best_body(payload: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Returns (mime_preferred, plain_text, html_text).
    Prefers text/plain; falls back to text/html.
    """
    plain = ""
    html = ""

    stack = [payload]
    while stack:
        part = stack.pop()
        mime = part.get("mimeType", "")
        body_data = (part.get("body") or {}).get("data")

        if body_data and mime in ("text/plain", "text/html"):
            decoded = _b64url_decode(body_data)
            if mime == "text/plain" and not plain and decoded.strip():
                plain = decoded
            if mime == "text/html" and not html and decoded.strip():
                html = decoded

        # traverse
        for p in (part.get("parts") or []):
            stack.append(p)

    if plain:
        return "text/plain", plain, html
    if html:
        return "text/html", _strip_html(html), html

    return "", "", ""


def get_message_full(service, msg_id: str) -> Dict[str, Any]:
    """
    Fetch full message and extract body text.
    """
    msg = service.users().messages().get(
        userId="me",
        id=msg_id,
        format="full",
    ).execute()

    payload = msg.get("payload", {}) or {}
    mime, plain, html = _find_best_body(payload)

    return {
        "id": msg_id,
        "mimeType": mime,
        "plain": plain,
        "html": html,
    }