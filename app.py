import os
import re
import json
import time
import html
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from gmail_client import get_service, list_message_ids, get_message_metadata, get_message_full
from storage import (
    init_db,
    get_triage,
    upsert_triage,
    get_followup,
    upsert_followup,
    set_followup_status,
    get_followup_all,
)
from llm_triage import triage_emails
from followup_ai import summarize_followups

load_dotenv()
init_db()

st.set_page_config(page_title="AI Gmail", layout="wide")

# ----------------------------
# State
# ----------------------------
if "inbox_view" not in st.session_state:
    st.session_state.inbox_view = "list"  # list | detail
if "opened_id" not in st.session_state:
    st.session_state.opened_id = None
if "full_cache" not in st.session_state:
    st.session_state.full_cache = {}
if "reply_cache" not in st.session_state:
    st.session_state.reply_cache = {}

if "emails" not in st.session_state:
    st.session_state.emails = []
if "max_emails" not in st.session_state:
    st.session_state.max_emails = 200
if "pending_max_emails" not in st.session_state:
    st.session_state.pending_max_emails = st.session_state.max_emails

# Inbox filters (ONLY inbox)
if "type_filter" not in st.session_state:
    st.session_state.type_filter = []
if "active_topic" not in st.session_state:
    st.session_state.active_topic = None
if "inbox_status_filter" not in st.session_state:
    st.session_state.inbox_status_filter = "All"  # All/Urgent/Spam/No action

# Dashboard filter (ONLY dashboard)
if "show_completed_cards" not in st.session_state:
    st.session_state.show_completed_cards = False

# Sync + gating
if "last_sync_ts" not in st.session_state:
    st.session_state.last_sync_ts = 0.0
if "emails_digest" not in st.session_state:
    st.session_state.emails_digest = ""
if "dashboard_dirty" not in st.session_state:
    st.session_state.dashboard_dirty = True  # first run builds dashboard

api_ok = bool(os.getenv("DEEPSEEK_API_KEY"))
service = get_service()

# Poll ONLY in list view
if st.session_state.inbox_view == "list":
    try:
        st.autorefresh(interval=90_000, key="poll")
    except Exception:
        pass

# ----------------------------
# Styling (keep your current look)
# ----------------------------
st.markdown(
    """
<style>
h1 { display:none; }
.header { font-size: 18px; font-weight: 800; margin: 4px 0 6px 0; }
.stTextInput input { font-size: 18px; padding: 10px 14px; border-radius: 999px; }

div.stButton > button {
  border-radius: 10px;
  justify-content:flex-start !important;
  text-align:left !important;
  align-items: center !important;
  padding: 3px 8px !important;
  line-height: 1.05 !important;
  width: 100% !important;
}
div.stButton > button p { text-align:left !important; width: 100% !important; margin: 0 !important; }

.row-summary { font-size: 0.82rem; opacity: 0.85; margin-top: 1px; }
.divline { border-bottom: 1px solid rgba(255,255,255,0.06); margin: 5px 0; }
.mail-date { opacity:0.75; font-size:0.78rem; text-align:right; white-space:nowrap; }

.badge {
  display:inline-block; padding:1px 8px; border-radius:999px;
  font-size:11px; border:1px solid rgba(255,255,255,0.16);
  background:rgba(255,255,255,0.02);
  white-space: nowrap;
}
.badge-urgent { border-color: rgba(255,165,0,0.55); }
.badge-spam   { border-color: rgba(255,70,70,0.55); }
.badge-ok     { border-color: rgba(120,255,170,0.35); }
.badge-acc    { border-color: rgba(120,255,170,0.35); }

.card { border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; padding: 12px;
  background: rgba(255,255,255,0.02); margin-bottom: 10px; }
.card-title { font-size: 16px; font-weight: 800; }
.card-sub { font-size: 13px; opacity: 0.85; margin-top: 4px; }
.card-meta { font-size: 12px; opacity: 0.75; margin-top: 8px; }
.pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid rgba(255,255,255,0.14); margin-right:6px; }
.pill-action { border-color: rgba(255,165,0,0.55); }
.pill-wait { border-color: rgba(120,170,255,0.45); }
.pill-done { border-color: rgba(120,255,170,0.40); }

.detail-top { font-size: 0.95rem; font-weight: 750; margin-top: 4px; }
.detail-from { opacity: 0.8; font-size: 0.82rem; margin-top: 2px; }
.detail-mini { opacity: 0.85; font-size: 0.82rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<div class='header'>AI Gmail</div>", unsafe_allow_html=True)

# ----------------------------
# Heuristics
# ----------------------------
PROMO_WORDS = [
    "discount", "off", "%", "sale", "deal", "offer", "last chance", "save", "promo",
    "newsletter", "limited time", "exclusive", "ends soon", "clearance", "buy 1", "bogo",
    "free shipping", "voucher", "coupon"
]

def promo_like(subject: str, snippet: str) -> bool:
    text = f"{subject} {snippet}".lower()
    if re.search(r"\b\d{1,3}%\s*off\b", text):
        return True
    return any(w in text for w in PROMO_WORDS)

# ----------------------------
# Helpers
# ----------------------------
def norm_topic(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s&\-]", "", t)
    return t[:28] if t else "Unsorted"

def parse_date_safe(date_str: str):
    try:
        dt = parsedate_to_datetime(date_str)
        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def format_date(date_str: str) -> str:
    dt = parse_date_safe(date_str)
    if not dt:
        return ""
    now = datetime.now(dt.tzinfo or timezone.utc)
    if dt.date() == now.date():
        return dt.strftime("%H:%M")
    return dt.strftime("%d %b")

def attach_triage():
    ids = [e["id"] for e in st.session_state.emails]
    cached = get_triage(ids)
    for e in st.session_state.emails:
        e["triage"] = cached.get(e["id"])

def mail_type(e) -> str:
    tri = e.get("triage") or {}
    return norm_topic(tri.get("topic", ""))

def is_spam(e) -> bool:
    tri = e.get("triage") or {}
    if promo_like(e.get("subject",""), e.get("snippet","")):
        return True
    return tri.get("is_spam") is True

def is_urgent(e) -> bool:
    tri = e.get("triage") or {}
    if promo_like(e.get("subject",""), e.get("snippet","")):
        return False
    return tri.get("is_urgent") is True

def no_action_needed(e) -> bool:
    if is_spam(e):
        return False
    tri = e.get("triage") or {}
    action = tri.get("action", "Read")
    return (not is_urgent(e)) and (action in ["Read", "Ignore"])

def needs_followup(e) -> bool:
    tri = e.get("triage") or {}
    if is_spam(e):
        return False
    if is_urgent(e):
        return True
    return tri.get("action", "Read") in ["Reply", "Pay", "Book", "Follow-up"]

def conf_pct(e):
    tri = e.get("triage") or {}
    c = tri.get("confidence")
    if isinstance(c, (int, float)):
        return max(0.0, min(100.0, float(c) * 100.0))
    return None

def overall_accuracy(emails):
    vals = [conf_pct(e) for e in emails if conf_pct(e) is not None]
    return (sum(vals) / len(vals)) if vals else None

def status_badge_for_row(e) -> str:
    if is_urgent(e):
        return "<span class='badge badge-urgent'>Urgent</span>"
    if is_spam(e):
        return "<span class='badge badge-spam'>Spam</span>"
    if no_action_needed(e):
        return "<span class='badge badge-ok'>No action</span>"
    return ""

def get_full_body(eid: str) -> str:
    if eid in st.session_state.full_cache:
        return st.session_state.full_cache[eid]
    full = get_message_full(service, eid)
    body = (full.get("plain") or "").strip()
    body = html.unescape(body).replace("\xa0", " ")
    body = re.sub(r"[ \t]{5,}", "    ", body)
    st.session_state.full_cache[eid] = body
    return body

def cb_open_email(eid: str):
    st.session_state.opened_id = eid
    st.session_state.inbox_view = "detail"

def cb_close_email():
    st.session_state.opened_id = None
    st.session_state.inbox_view = "list"

def auto_fetch_new_emails() -> bool:
    if st.session_state.inbox_view == "detail":
        return False

    now = time.time()
    if now - st.session_state.last_sync_ts < 25:
        return False
    st.session_state.last_sync_ts = now

    ids = list_message_ids(service, max_results=st.session_state.max_emails)
    if not ids:
        return False

    current_ids = [e["id"] for e in st.session_state.emails]
    if ids == current_ids:
        return False

    existing_map = {e["id"]: e for e in st.session_state.emails}
    new_list = []
    for mid in ids:
        if mid in existing_map:
            new_list.append(existing_map[mid])
        else:
            new_list.append(get_message_metadata(service, mid))

    st.session_state.emails = new_list
    return True

def auto_ai_run_new_only() -> bool:
    if st.session_state.inbox_view == "detail":
        return False
    if not api_ok or not st.session_state.emails:
        return False

    attach_triage()
    target = [e for e in st.session_state.emails if e.get("triage") is None]
    if not target:
        return False

    existing = []
    for e in st.session_state.emails:
        tri = e.get("triage") or {}
        t = norm_topic(tri.get("topic", ""))
        if t and t != "Unsorted":
            existing.append(t)
    existing = list(dict.fromkeys(existing))[:60]

    BATCH = 25
    results = []
    for i in range(0, len(target), BATCH):
        results.extend(triage_emails(target[i:i+BATCH], model="deepseek-chat", existing_topics=existing))

    upsert_triage(results)
    attach_triage()
    return True

def build_dynamic_types(emails):
    counts = {}
    for e in emails:
        t = mail_type(e)
        counts[t] = counts.get(t, 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in items]

def apply_inbox_filters(emails, q: str):
    out = emails

    # Type filter / active topic
    if st.session_state.active_topic:
        out = [e for e in out if mail_type(e) == st.session_state.active_topic]
    elif st.session_state.type_filter:
        want = set(st.session_state.type_filter)
        out = [e for e in out if mail_type(e) in want]

    # Status filter (INBOX ONLY)
    sf = st.session_state.inbox_status_filter
    if sf == "Urgent":
        out = [e for e in out if is_urgent(e)]
    elif sf == "Spam":
        out = [e for e in out if is_spam(e)]
    elif sf == "No action":
        out = [e for e in out if no_action_needed(e)]

    # Search
    qq = (q or "").strip().lower()
    if qq:
        def match(e):
            tri = e.get("triage") or {}
            blob = " ".join([
                (e.get("from") or ""),
                (e.get("subject") or ""),
                (e.get("snippet") or ""),
                (tri.get("topic") or ""),
                (tri.get("summary") or ""),
            ]).lower()
            return qq in blob
        out = [e for e in out if match(e)]

    out = sorted(out, key=lambda e: (parse_date_safe(e.get("date","")) or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    return out

def topic_signature(topic: str, topic_emails):
    ids = [e["id"] for e in topic_emails]
    newest = ids[0] if ids else ""
    return f"{topic}:{newest}:{len(ids)}"

def extract_email_addr(from_field: str) -> str:
    m = re.search(r"<([^>]+)>", from_field or "")
    if m: return m.group(1).strip()
    m2 = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", from_field or "")
    return m2.group(0).strip() if m2 else ""

def gmail_compose_link(to: str, subject: str, body: str) -> str:
    return (
        "https://mail.google.com/mail/?view=cm&fs=1"
        f"&to={quote_plus(to)}"
        f"&su={quote_plus(subject)}"
        f"&body={quote_plus(body)}"
    )

# ----------------------------
# Initial load
# ----------------------------
if not st.session_state.emails:
    ids = list_message_ids(service, max_results=st.session_state.max_emails)
    st.session_state.emails = [get_message_metadata(service, mid) for mid in ids]
    st.session_state.dashboard_dirty = True

# ----------------------------
# Sync + triage (ONLY on poll / email change)
# ----------------------------
changed = auto_fetch_new_emails()
triaged = auto_ai_run_new_only()

ids_now = [e["id"] for e in st.session_state.emails]
digest = f"{len(ids_now)}:{ids_now[0] if ids_now else ''}:{ids_now[-1] if ids_now else ''}"
if digest != st.session_state.emails_digest:
    st.session_state.emails_digest = digest
    st.session_state.dashboard_dirty = True

if changed or triaged:
    st.session_state.dashboard_dirty = True

# ----------------------------
# Header search
# ----------------------------
search_text = st.text_input("", value=st.session_state.get("search_text",""), placeholder="Search mail (live)…", key="search_text")

# ----------------------------
# Sidebar widgets FIRST (fix one-click lag)
# ----------------------------
with st.sidebar:
    st.markdown("### Load")
    st.session_state.pending_max_emails = st.slider("Emails to load", 20, 800, st.session_state.pending_max_emails, 10)
    if st.button("Apply"):
        st.session_state.max_emails = st.session_state.pending_max_emails
        st.session_state.last_sync_ts = 0.0
        st.session_state.emails = []
        st.session_state.full_cache = {}
        st.session_state.reply_cache = {}
        st.session_state.inbox_view = "list"
        st.session_state.opened_id = None
        st.session_state.dashboard_dirty = True
        st.rerun()

    st.markdown("### Filters (Inbox)")
    types = build_dynamic_types(st.session_state.emails)

    if st.session_state.active_topic:
        st.info(f"Viewing: {st.session_state.active_topic}")
        if st.button("Clear topic view"):
            st.session_state.active_topic = None

    st.session_state.type_filter = st.multiselect(
        "Mail types",
        options=types,
        default=st.session_state.type_filter,
        disabled=bool(st.session_state.active_topic),
    )

    st.markdown("### Inbox status")
    st.session_state.inbox_status_filter = st.radio(
        " ",
        ["All", "Urgent", "Spam", "No action"],
        index=["All", "Urgent", "Spam", "No action"].index(st.session_state.inbox_status_filter),
        label_visibility="collapsed",
    )

    st.markdown("### Dashboard")
    st.session_state.show_completed_cards = st.checkbox("Show completed tasks", value=st.session_state.show_completed_cards)

# ----------------------------
# Now compute filtered (after sidebar updates) => 1-click filters
# ----------------------------
filtered = apply_inbox_filters(st.session_state.emails, search_text)

acc = overall_accuracy(st.session_state.emails)
acc_text = f"AI accuracy: {acc:.2f}%" if acc is not None else "AI accuracy: N/A"
st.caption(f"Showing {len(filtered)} of {len(st.session_state.emails)}  |  {acc_text}")

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([2.2, 3.1], gap="large")

with left:
    inbox_box = st.container(height=740, border=True)
    with inbox_box:
        if st.session_state.inbox_view == "detail" and st.session_state.opened_id:
            eid = st.session_state.opened_id
            st.button("← Back", on_click=cb_close_email)

            selected = next((e for e in st.session_state.emails if e["id"] == eid), None)
            if not selected:
                st.info("Email not found.")
            else:
                tri = selected.get("triage") or {}
                st.markdown(f"<div class='detail-top'>{mail_type(selected)} | {selected.get('subject','')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='detail-from'>{selected.get('from','')}</div>", unsafe_allow_html=True)

                # Detail indicators (includes accuracy)
                badges = []
                if is_urgent(selected): badges.append("<span class='badge badge-urgent'>Urgent</span>")
                if is_spam(selected): badges.append("<span class='badge badge-spam'>Spam</span>")
                if no_action_needed(selected): badges.append("<span class='badge badge-ok'>No action</span>")
                c = conf_pct(selected)
                if c is not None: badges.append(f"<span class='badge badge-acc'>Accuracy {c:.1f}%</span>")
                if badges:
                    st.markdown(" ".join(badges), unsafe_allow_html=True)

                st.markdown(f"<div class='detail-mini'><b>AI:</b> {tri.get('summary','(no summary yet)')}</div>", unsafe_allow_html=True)

                # --- Reply + Generate AI response (side-by-side) ---
                import re
                from urllib.parse import quote

                def _extract_email_addr(text: str) -> str:
                    if not text:
                        return ""
                    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
                    return m.group(0) if m else ""

                def _gmail_compose_link(to: str, subject: str, body: str) -> str:
                    base = "https://mail.google.com/mail/?view=cm&fs=1"
                    return f"{base}&to={quote(to)}&su={quote(subject)}&body={quote(body)}"

                sender = selected.get("from", "") or ""
                subject = selected.get("subject", "") or ""
                to_addr = _extract_email_addr(sender)
                subj = f"Re: {subject}" if subject else "Re:"
                reply_link = _gmail_compose_link(to_addr, subj, "")

                b1, b2 = st.columns([1, 1], gap="small")

                # Reply button (opens Gmail compose)
                if hasattr(b1, "link_button"):
                    b1.link_button("Reply", reply_link, use_container_width=True)
                else:
                    b1.markdown(f"[Reply]({reply_link})")

                # AI response cache (only shows between info + full body when generated)
                if "ai_response_cache" not in st.session_state:
                    st.session_state.ai_response_cache = {}
                if "reply_cache" not in st.session_state:
                    st.session_state.reply_cache = {}

                def cb_generate_ai_response(mid: str):
                    try:
                        cb_generate_reply(mid)  # expected to fill st.session_state.reply_cache[mid]
                        st.session_state.ai_response_cache[mid] = st.session_state.reply_cache.get(mid, "")
                    except Exception as ex:
                        st.session_state.ai_response_cache[mid] = f"(Could not generate AI response: {ex})"

                btn_label = "Generate AI response" if eid not in st.session_state.ai_response_cache else "Regenerate AI response"
                b2.button(btn_label, on_click=cb_generate_ai_response, args=(eid,), use_container_width=True)

                # Show AI response ONLY if generated (between info and full mail)
                ai_text = (st.session_state.ai_response_cache.get(eid, "") or "").strip()
                if ai_text:
                    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)
                    st.markdown("**AI generated response**")
                    st.text_area("", ai_text, height=140, label_visibility="collapsed")

                    # Optional: open Gmail compose prefilled with the AI text
                    link2 = _gmail_compose_link(to_addr, subj, ai_text)
                    if hasattr(st, "link_button"):
                        st.link_button("Open AI response in Gmail compose", link2, use_container_width=True)
                    else:
                        st.markdown(f"[Open AI response in Gmail compose]({link2})")

                st.markdown("<div class='divline'></div>", unsafe_allow_html=True)
                st.text_area("Full email", get_full_body(eid), height=600)

        else:
            if not filtered:
                st.info("No emails match.")
            else:
                for e in filtered:
                    tri = e.get("triage") or {}
                    mt = mail_type(e) or "Unsorted"
                    subj = (e.get("subject") or "(no subject)").strip()
                    summary = (tri.get("summary") or e.get("snippet") or "").strip()

                    title = f"{mt} | {subj[:92]}"

                    # Subject | status badge | date
                    c1, c2, c3 = st.columns([6.2, 1.2, 0.9])
                    with c1:
                        st.button(
                            title,
                            key=f"row_{e['id']}",
                            use_container_width=True,
                            on_click=cb_open_email,
                            args=(e["id"],),
                        )
                        st.markdown(f"<div class='row-summary'>{summary[:140]}</div>", unsafe_allow_html=True)
                    with c2:
                        badge = status_badge_for_row(e)
                        if badge:
                            st.markdown(badge, unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"<div class='mail-date'>{format_date(e.get('date',''))}</div>", unsafe_allow_html=True)

                    st.markdown("<div class='divline'></div>", unsafe_allow_html=True)

with right:
    dash_box = st.container(height=740, border=True)
    with dash_box:
        st.subheader("Follow-up Dashboard")

        # Build follow-up topics (independent of inbox filters)
        by_topic = {}
        for e in st.session_state.emails:
            tri = e.get("triage") or {}
            if not tri:
                continue
            if is_spam(e):
                continue
            t = mail_type(e)
            if not t or t == "Unsorted":
                continue
            if needs_followup(e):
                by_topic.setdefault(t, []).append(e)

        # Require back-and-forth
        by_topic = {t: v for t, v in by_topic.items() if len(v) >= 2}

        topics = list(by_topic.keys())
        for t in topics:
            by_topic[t] = sorted(
                by_topic[t],
                key=lambda e: (parse_date_safe(e.get("date","")) or datetime.min.replace(tzinfo=timezone.utc)),
                reverse=True,
            )

        # Pull cached cards (fast)
        cached_cards = get_followup(topics) if topics else {}

        # NEW RULE: dashboard AI updates only when dashboard_dirty AND list view
        if (st.session_state.inbox_view == "list") and api_ok and st.session_state.dashboard_dirty and topics:
            to_build = []
            sig_map = {}
            for t in topics:
                sig = topic_signature(t, by_topic[t][:10])
                sig_map[t] = sig
                cached = cached_cards.get(t)
                if not cached or cached.get("_signature") != sig:
                    to_build.append(t)
            to_build = to_build[:6]

            if to_build:
                payload = []
                for t in to_build:
                    items = by_topic[t][:10]
                    payload.append(
                        {
                            "topic": t,
                            "emails": [
                                {
                                    "date": (i.get("date") or ""),
                                    "from": (i.get("from") or ""),
                                    "subject": (i.get("subject") or ""),
                                    "summary": ((i.get("triage") or {}).get("summary") or (i.get("snippet") or "")),
                                    "action": ((i.get("triage") or {}).get("action") or ""),
                                    "urgent": is_urgent(i),
                                }
                                for i in items
                            ],
                        }
                    )
                cards = summarize_followups(payload)
                cards = [c for c in cards if c.get("include", True) is True]
                upsert_followup(cards, signature_map=sig_map)
                cached_cards = get_followup(topics)

            # Done: mark clean so filter clicks never trigger this
            st.session_state.dashboard_dirty = False

        # Completed cards (from DB) only shown when toggle is on
        completed_extra = {}
        if st.session_state.show_completed_cards:
            completed_extra = get_followup_all("complete")

        # Render list
        cards_list = []
        for t in topics:
            c = cached_cards.get(t)
            if not c:
                continue
            if c.get("include", True) is False:
                continue
            done = (c.get("_status") == "complete")
            if done and (not st.session_state.show_completed_cards):
                continue
            cards_list.append(c)

        # Add completed cards even if not currently in by_topic (optional)
        if st.session_state.show_completed_cards:
            for t, c in completed_extra.items():
                if t in topics:
                    continue
                cards_list.append(c)

        if not cards_list:
            st.info("No real follow-up threads detected.")
        else:
            def rank(c):
                if c.get("_status") == "complete":
                    return (9, 0)
                stt = c.get("status", "Waiting")
                s_rank = 0 if stt == "Needs action" else (1 if stt == "Waiting" else 2)
                pr = int(c.get("priority", 3))
                return (s_rank, -pr)

            cards_list.sort(key=rank)
            cols = st.columns(2)

            def cb_show_topic(topic: str):
                st.session_state.active_topic = topic
                cb_close_email()

            for idx, c in enumerate(cards_list[:14]):
                topic = c.get("topic", "")
                last_up = c.get("last_update", "")
                next_step = c.get("next_step", "")

                done = (c.get("_status") == "complete")
                status = "Completed" if done else c.get("status", "Waiting")
                detail = "Marked complete" if done else c.get("status_detail", "")

                pill = "pill-wait"
                if status == "Needs action":
                    pill = "pill-action"
                elif status == "Completed":
                    pill = "pill-done"

                boxc = cols[idx % 2]
                boxc.markdown(
                    f"""
<div class="card">
  <div class="card-title">{topic}</div>
  <div class="card-sub"><span class="pill {pill}">{status}</span> {detail}</div>
  <div class="card-meta">Next: {next_step}</div>
  <div class="card-meta">Last update: {last_up}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                a1, a2 = boxc.columns(2)
                a1.button("Show emails", key=f"show_{topic}", on_click=cb_show_topic, args=(topic,))
                a2.button(
                    "Reopen" if done else "Complete",
                    key=f"done_{topic}",
                    on_click=set_followup_status,
                    args=(topic, "open" if done else "complete"),
                )