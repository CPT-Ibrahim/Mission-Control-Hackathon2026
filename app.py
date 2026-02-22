import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import streamlit as st
from dotenv import load_dotenv

from gmail_client import get_service, list_message_ids, get_message_metadata
from storage import (
    init_db,
    get_triage,
    upsert_triage,
    get_followup,
    upsert_followup,
    set_followup_status,
)
from llm_triage import triage_emails
from followup_ai import summarize_followups

load_dotenv()
init_db()

st.set_page_config(page_title="AI Gmail", layout="wide")

st.markdown(
    """
<style>
h1 { display:none; }
.header { font-size: 18px; font-weight: 800; margin: 4px 0 6px 0; }

.stTextInput input { font-size: 18px; padding: 10px 14px; border-radius: 999px; }

/* Make rows compact and left-aligned */
div.stButton > button {
  border-radius: 10px;
  justify-content:flex-start !important;
  text-align:left !important;
  padding: 4px 8px !important;
  line-height: 1.1 !important;
}

/* Compact row */
.row-summary { font-size: 0.84rem; opacity: 0.85; margin-top: 1px; }
.row-badges { margin-top: 2px; }
.badge {
  display:inline-block; padding:1px 7px; border-radius:999px;
  font-size:11px; margin-right:6px; margin-bottom:2px;
  border:1px solid rgba(255,255,255,0.16);
  background:rgba(255,255,255,0.02);
}
.badge-urgent { border-color: rgba(255,165,0,0.50); }
.badge-spam   { border-color: rgba(255,70,70,0.50); }
.badge-acc    { border-color: rgba(120,255,170,0.35); }

.mail-date { opacity:0.75; font-size:0.80rem; text-align:right; white-space:nowrap; }
.divline { border-bottom: 1px solid rgba(255,255,255,0.06); margin: 6px 0; }

/* Follow-up cards */
.card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px;
  background: rgba(255,255,255,0.02);
  margin-bottom: 10px;
}
.card-title { font-size: 16px; font-weight: 800; }
.card-sub { font-size: 13px; opacity: 0.85; margin-top: 4px; }
.card-meta { font-size: 12px; opacity: 0.75; margin-top: 8px; }

.pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid rgba(255,255,255,0.14); margin-right:6px; }
.pill-action { border-color: rgba(255,165,0,0.55); }
.pill-wait { border-color: rgba(120,170,255,0.45); }
.pill-done { border-color: rgba(120,255,170,0.40); }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<div class='header'>AI Gmail</div>", unsafe_allow_html=True)

# ----------------------------
# State
# ----------------------------
if "emails" not in st.session_state:
    st.session_state.emails = []
if "max_emails" not in st.session_state:
    st.session_state.max_emails = 200
if "type_filter" not in st.session_state:
    st.session_state.type_filter = []
if "urgent_only" not in st.session_state:
    st.session_state.urgent_only = False
if "spam_only" not in st.session_state:
    st.session_state.spam_only = False
if "show_completed_cards" not in st.session_state:
    st.session_state.show_completed_cards = False
if "active_topic" not in st.session_state:
    st.session_state.active_topic = None  # set by dashboard "Show emails"

api_ok = bool(os.getenv("DEEPSEEK_API_KEY"))
service = get_service()

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

def refresh_emails():
    ids = list_message_ids(service, max_results=st.session_state.max_emails)
    st.session_state.emails = [get_message_metadata(service, mid) for mid in ids]

def attach_triage():
    ids = [e["id"] for e in st.session_state.emails]
    cached = get_triage(ids)
    for e in st.session_state.emails:
        e["triage"] = cached.get(e["id"])

def mail_type(e) -> str:
    tri = e.get("triage") or {}
    return norm_topic(tri.get("topic", ""))

def conf_pct(e):
    tri = e.get("triage") or {}
    c = tri.get("confidence")
    if isinstance(c, (int, float)):
        return max(0.0, min(100.0, float(c) * 100.0))
    return None

def overall_accuracy(emails):
    vals = [conf_pct(e) for e in emails if conf_pct(e) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)

def is_urgent(e) -> bool:
    return (e.get("triage") or {}).get("is_urgent") is True

def is_spam(e) -> bool:
    return (e.get("triage") or {}).get("is_spam") is True

def needs_followup(e) -> bool:
    tri = e.get("triage") or {}
    if tri.get("is_spam") is True:
        return False
    if tri.get("is_urgent") is True:
        return True
    action = tri.get("action", "Read")
    return action in ["Reply", "Pay", "Book", "Follow-up"]

def extract_email_addr(from_field: str) -> str:
    m = re.search(r"<([^>]+)>", from_field or "")
    if m:
        return m.group(1).strip()
    # fallback: find something that looks like an email
    m2 = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", from_field or "")
    return m2.group(0).strip() if m2 else ""

def gmail_compose_link(to: str, subject: str, body: str) -> str:
    # Opens Gmail compose window in browser
    return (
        "https://mail.google.com/mail/?view=cm&fs=1"
        f"&to={quote_plus(to)}"
        f"&su={quote_plus(subject)}"
        f"&body={quote_plus(body)}"
    )

def auto_ai_run_new_only():
    """
    Runs AI ONLY for emails missing triage in cache.
    This means: no email is processed twice.
    """
    if not api_ok:
        return

    attach_triage()
    target = [e for e in st.session_state.emails if e.get("triage") is None]
    if not target:
        return

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

def build_dynamic_types(emails):
    counts = {}
    for e in emails:
        t = mail_type(e)
        counts[t] = counts.get(t, 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in items]

def apply_filters(emails, q: str):
    out = emails

    # if dashboard selected a topic, override type filter
    if st.session_state.active_topic:
        out = [e for e in out if mail_type(e) == st.session_state.active_topic]
    else:
        if st.session_state.type_filter:
            want = set(st.session_state.type_filter)
            out = [e for e in out if mail_type(e) in want]

    if st.session_state.urgent_only:
        out = [e for e in out if is_urgent(e)]

    if st.session_state.spam_only:
        out = [e for e in out if is_spam(e)]

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

def badges_html(e):
    parts = []
    if is_urgent(e):
        parts.append("<span class='badge badge-urgent'>🟠 Urgent</span>")
    if is_spam(e):
        parts.append("<span class='badge badge-spam'>🚫 Spam</span>")
    c = conf_pct(e)
    if c is not None:
        parts.append(f"<span class='badge badge-acc'>Accuracy {c:.1f}%</span>")
    return " ".join(parts)

def topic_signature(topic: str, topic_emails):
    # signature changes when new follow-up emails appear
    # use newest message id + count
    ids = [e["id"] for e in topic_emails]
    newest = ids[0] if ids else ""
    return f"{topic}:{newest}:{len(ids)}"

# ----------------------------
# Load + auto AI run (new only)
# ----------------------------
if not st.session_state.emails:
    refresh_emails()
attach_triage()
auto_ai_run_new_only()

# ----------------------------
# Header search
# ----------------------------
search_text = st.text_input(
    "",
    value=st.session_state.get("search_text",""),
    placeholder="Search mail (live)…",
    key="search_text",
)

acc = overall_accuracy(st.session_state.emails)
acc_text = f"AI accuracy: {acc:.2f}%" if acc is not None else "AI accuracy: N/A"

# ----------------------------
# Sidebar (NO run AI buttons)
# ----------------------------
with st.sidebar:
    st.markdown("### Inbox")
    if st.button("⟳ Refresh inbox"):
        refresh_emails()
        attach_triage()
        auto_ai_run_new_only()

    st.session_state.max_emails = st.slider("Emails to load", 20, 800, st.session_state.max_emails, 10)

    st.markdown("### Filters (AI-generated)")
    types = build_dynamic_types(st.session_state.emails)

    if st.session_state.active_topic:
        st.info(f"Viewing: {st.session_state.active_topic}")
        if st.button("Clear topic view"):
            st.session_state.active_topic = None

    st.session_state.type_filter = st.multiselect("Mail types", options=types, default=st.session_state.type_filter, disabled=bool(st.session_state.active_topic))

    st.session_state.urgent_only = st.checkbox("Urgent only", value=st.session_state.urgent_only)
    st.session_state.spam_only = st.checkbox("Spam only", value=st.session_state.spam_only)

    st.session_state.show_completed_cards = st.checkbox("Show completed cards", value=st.session_state.show_completed_cards)

# ----------------------------
# Main layout: compact inbox + follow-up dashboard
# ----------------------------
left, right = st.columns([2.2, 3.1], gap="large")

filtered = apply_filters(st.session_state.emails, search_text)

with left:
    st.caption(f"Showing {len(filtered)} of {len(st.session_state.emails)}  |  {acc_text}")

    list_box = st.container(height=740, border=True)
    with list_box:
        if not filtered:
            st.info("No emails match.")
        else:
            for e in filtered:
                tri = e.get("triage") or {}
                mt = mail_type(e) or "Unsorted"
                subj = (e.get("subject") or "(no subject)").strip()
                summary = (tri.get("summary") or e.get("snippet") or "").strip()

                title = f"{mt} | {subj[:92]}"

                c1, c2 = st.columns([4.8, 1.2])
                with c1:
                    st.button(title, key=f"row_{e['id']}", use_container_width=True)
                    st.markdown(f"<div class='row-summary'>{summary[:160]}</div>", unsafe_allow_html=True)
                    bh = badges_html(e)
                    if bh:
                        st.markdown(f"<div class='row-badges'>{bh}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='mail-date'>{format_date(e.get('date',''))}</div>", unsafe_allow_html=True)

                st.markdown("<div class='divline'></div>", unsafe_allow_html=True)

with right:
    dash = st.container(height=740, border=True)
    with dash:
        st.subheader("Follow-up Dashboard")

        # group follow-up emails by topic
        by_topic = {}
        for e in st.session_state.emails:
            tri = e.get("triage") or {}
            if not tri or is_spam(e):
                continue
            t = mail_type(e)
            if not t or t == "Unsorted":
                continue
            if needs_followup(e):
                by_topic.setdefault(t, []).append(e)

        if not by_topic:
            st.info("No follow-up topics detected.")
        else:
            # newest first per topic
            for t in list(by_topic.keys()):
                by_topic[t] = sorted(
                    by_topic[t],
                    key=lambda e: (parse_date_safe(e.get("date","")) or datetime.min.replace(tzinfo=timezone.utc)),
                    reverse=True,
                )

            topics = list(by_topic.keys())
            cached_cards = get_followup(topics)

            # Auto-generate missing or stale cards (NO BUTTONS)
            to_build = []
            sig_map = {}
            for t in topics:
                sig = topic_signature(t, by_topic[t][:10])
                sig_map[t] = sig
                cached = cached_cards.get(t)
                if not cached or cached.get("_signature") != sig:
                    to_build.append(t)

            # limit rebuild per rerun to keep UI fast
            to_build = to_build[:8]

            if api_ok and to_build:
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
                with st.spinner("Updating follow-up dashboard..."):
                    cards = summarize_followups(payload)
                upsert_followup(cards, signature_map=sig_map)
                cached_cards = get_followup(topics)

            # build render list
            cards_list = []
            for t in topics:
                c = cached_cards.get(t) or {"topic": t, "status": "Waiting", "status_detail": "No status yet", "last_update": "", "next_step": "None", "priority": 2, "accuracy": 50, "_status": "open"}
                if (c.get("_status") == "complete") and (not st.session_state.show_completed_cards):
                    continue
                cards_list.append(c)

            def rank(c):
                stt = c.get("status", "Waiting")
                s_rank = 0 if stt == "Needs action" else (1 if stt == "Waiting" else 2)
                pr = int(c.get("priority", 3))
                return (s_rank, -pr)

            cards_list.sort(key=rank)

            cols = st.columns(2)
            for idx, c in enumerate(cards_list[:14]):
                topic = c.get("topic", "")
                status = c.get("status", "Waiting")
                detail = c.get("status_detail", "")
                last_up = c.get("last_update", "")
                next_step = c.get("next_step", "")
                acc2 = float(c.get("accuracy", 0))

                pill = "pill-wait"
                if status == "Needs action":
                    pill = "pill-action"
                elif status == "Completed":
                    pill = "pill-done"

                box = cols[idx % 2]

                # Card UI
                box.markdown(
                    f"""
<div class="card">
  <div class="card-title">{topic}</div>
  <div class="card-sub"><span class="pill {pill}">{status}</span> {detail}</div>
  <div class="card-meta">Next: {next_step}</div>
  <div class="card-meta">Last update: {last_up} &nbsp; | &nbsp; Accuracy: {acc2:.1f}%</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                # Card actions (requested)
                a1, a2, a3 = box.columns(3)

                # 1) Show related emails (filters left inbox)
                if a1.button("Show emails", key=f"show_{topic}"):
                    st.session_state.active_topic = topic

                # 2) Mark complete (persist)
                is_done = (c.get("_status") == "complete")
                if a2.button("Complete" if not is_done else "Reopen", key=f"done_{topic}"):
                    set_followup_status(topic, "complete" if not is_done else "open")

                # 3) Reply (opens Gmail compose with prefilled follow-up)
                # Use latest email sender + Re: latest subject
                latest = by_topic.get(topic, [None])[0]
                if latest:
                    to_addr = extract_email_addr(latest.get("from", ""))
                    su = latest.get("subject", "")
                    subj = f"Re: {su}" if su else f"Follow-up: {topic}"
                    body = f"Hi,\n\nFollowing up regarding: {topic}.\n\nNext step: {next_step}\n\nThanks,\n"
                    link = gmail_compose_link(to_addr, subj, body)
                    a3.markdown(f"[Reply]({link})")
                else:
                    a3.caption("Reply N/A")