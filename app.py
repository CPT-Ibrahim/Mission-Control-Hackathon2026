import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import streamlit as st
from dotenv import load_dotenv

from gmail_client import get_service, list_message_ids, get_message_metadata, get_message_full
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

# Auto refresh (poll) every 60s
try:
    st.autorefresh(interval=60_000, key="poll")
except Exception:
    pass

st.markdown(
    """
<style>
h1 { display:none; }
.header { font-size: 18px; font-weight: 800; margin: 4px 0 6px 0; }
.stTextInput input { font-size: 18px; padding: 10px 14px; border-radius: 999px; }

/* compact clickable rows */
div.stButton > button {
  border-radius: 10px;
  justify-content:flex-start !important;
  text-align:left !important;
  padding: 3px 8px !important;
  line-height: 1.05 !important;
}

.row-summary { font-size: 0.82rem; opacity: 0.85; margin-top: 1px; }
.row-badges { margin-top: 2px; }

.badge {
  display:inline-block; padding:1px 7px; border-radius:999px;
  font-size:11px; margin-right:6px; margin-bottom:2px;
  border:1px solid rgba(255,255,255,0.16);
  background:rgba(255,255,255,0.02);
}
.badge-urgent { border-color: rgba(255,165,0,0.55); }
.badge-spam   { border-color: rgba(255,70,70,0.55); }
.badge-acc    { border-color: rgba(120,255,170,0.35); }

.mail-date { opacity:0.75; font-size:0.78rem; text-align:right; white-space:nowrap; }
.divline { border-bottom: 1px solid rgba(255,255,255,0.06); margin: 5px 0; }

/* follow-up cards */
.card { border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; padding: 12px;
  background: rgba(255,255,255,0.02); margin-bottom: 10px; }
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
    st.session_state.active_topic = None

# inbox internal view
if "inbox_view" not in st.session_state:
    st.session_state.inbox_view = "list"  # list | detail
if "opened_id" not in st.session_state:
    st.session_state.opened_id = None
if "full_cache" not in st.session_state:
    st.session_state.full_cache = {}

api_ok = bool(os.getenv("DEEPSEEK_API_KEY"))
service = get_service()

# ----------------------------
# Heuristics to enforce promo->spam
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

def is_spam(e) -> bool:
    tri = e.get("triage") or {}
    subj = e.get("subject") or ""
    snip = e.get("snippet") or ""
    if promo_like(subj, snip):
        return True
    return tri.get("is_spam") is True

def is_urgent(e) -> bool:
    # promo can NEVER be urgent
    tri = e.get("triage") or {}
    subj = e.get("subject") or ""
    snip = e.get("snippet") or ""
    if promo_like(subj, snip):
        return False
    return tri.get("is_urgent") is True

def needs_followup(e) -> bool:
    tri = e.get("triage") or {}
    if is_spam(e):
        return False
    # only real follow-up actions
    action = tri.get("action", "Read")
    if is_urgent(e):
        return True
    return action in ["Reply", "Pay", "Book", "Follow-up"]

def extract_email_addr(from_field: str) -> str:
    m = re.search(r"<([^>]+)>", from_field or "")
    if m:
        return m.group(1).strip()
    m2 = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", from_field or "")
    return m2.group(0).strip() if m2 else ""

def gmail_compose_link(to: str, subject: str, body: str) -> str:
    return (
        "https://mail.google.com/mail/?view=cm&fs=1"
        f"&to={quote_plus(to)}"
        f"&su={quote_plus(subject)}"
        f"&body={quote_plus(body)}"
    )

def auto_fetch_new_emails():
    """
    Poll Gmail. If new message IDs appear, fetch metadata only for the new ones and prepend.
    """
    ids = list_message_ids(service, max_results=st.session_state.max_emails)
    if not ids:
        return

    existing_map = {e["id"]: e for e in st.session_state.emails}
    if st.session_state.emails:
        current_ids = [e["id"] for e in st.session_state.emails]
        if ids == current_ids:
            return

    new_emails = []
    for mid in ids:
        if mid in existing_map:
            new_emails.append(existing_map[mid])
        else:
            new_emails.append(get_message_metadata(service, mid))

    st.session_state.emails = new_emails

def auto_ai_run_new_only():
    """
    Runs AI ONLY for emails missing triage in cache.
    Never processes the same email twice.
    """
    if not api_ok or not st.session_state.emails:
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
    ids = [e["id"] for e in topic_emails]
    newest = ids[0] if ids else ""
    return f"{topic}:{newest}:{len(ids)}"

def open_email_in_inbox(eid: str):
    st.session_state.opened_id = eid
    st.session_state.inbox_view = "detail"

def get_full_body(eid: str) -> str:
    if eid in st.session_state.full_cache:
        return st.session_state.full_cache[eid]
    full = get_message_full(service, eid)
    body = (full.get("plain") or "").strip()
    st.session_state.full_cache[eid] = body
    return body

# ----------------------------
# Poll + triage
# ----------------------------
auto_fetch_new_emails()
attach_triage()
auto_ai_run_new_only()

# ----------------------------
# Header search + accuracy
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
# Sidebar (no refresh button)
# ----------------------------
with st.sidebar:
    st.markdown("### Auto sync")
    st.caption("Checks Gmail periodically.")

    st.session_state.max_emails = st.slider("Emails to load", 20, 800, st.session_state.max_emails, 10)

    st.markdown("### Filters (AI-generated)")
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

    st.session_state.urgent_only = st.checkbox("Urgent only", value=st.session_state.urgent_only)
    st.session_state.spam_only = st.checkbox("Spam only", value=st.session_state.spam_only)
    st.session_state.show_completed_cards = st.checkbox("Show completed cards", value=st.session_state.show_completed_cards)

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([2.2, 3.1], gap="large")
filtered = apply_filters(st.session_state.emails, search_text)

with left:
    st.caption(f"Showing {len(filtered)} of {len(st.session_state.emails)}  |  {acc_text}")

    box = st.container(height=740, border=True)
    with box:
        if st.session_state.inbox_view == "detail" and st.session_state.opened_id:
            eid = st.session_state.opened_id
            back = st.button("← Back", use_container_width=False)
            if back:
                st.session_state.inbox_view = "list"
                st.session_state.opened_id = None
            else:
                selected = next((e for e in st.session_state.emails if e["id"] == eid), None)
                if selected:
                    tri = selected.get("triage") or {}
                    st.write(f"**{mail_type(selected)} | {selected.get('subject','')}**")
                    st.caption(selected.get("from",""))
                    bh = badges_html(selected)
                    if bh:
                        st.markdown(bh, unsafe_allow_html=True)
                    st.markdown("---")
                    st.text_area("Full email", get_full_body(eid), height=560)
                else:
                    st.info("Email not found.")
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
                    c1, c2 = st.columns([4.8, 1.2])
                    with c1:
                        if st.button(title, key=f"row_{e['id']}", use_container_width=True):
                            open_email_in_inbox(e["id"])
                        st.markdown(f"<div class='row-summary'>{summary[:150]}</div>", unsafe_allow_html=True)
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

        # group ONLY real follow-up topics (exclude spam/promos)
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

        # Enforce "continuous back and forth": require at least 2 emails in the topic
        by_topic = {t: v for t, v in by_topic.items() if len(v) >= 2}

        if not by_topic:
            st.info("No real follow-up threads detected.")
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

            # Auto-generate missing/stale cards
            to_build = []
            sig_map = {}
            for t in topics:
                sig = topic_signature(t, by_topic[t][:10])
                sig_map[t] = sig
                cached = cached_cards.get(t)
                if not cached or cached.get("_signature") != sig:
                    to_build.append(t)
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
                with st.spinner("Updating dashboard..."):
                    cards = summarize_followups(payload)
                # Keep only include=true
                cards = [c for c in cards if c.get("include", True) is True]
                upsert_followup(cards, signature_map=sig_map)
                cached_cards = get_followup(topics)

            # Render
            cards_list = []
            for t in topics:
                c = cached_cards.get(t)
                if not c:
                    continue
                if c.get("include", True) is False:
                    continue
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

                boxc = cols[idx % 2]
                boxc.markdown(
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

                a1, a2, a3 = boxc.columns(3)

                if a1.button("Show emails", key=f"show_{topic}"):
                    st.session_state.active_topic = topic
                    st.session_state.inbox_view = "list"
                    st.session_state.opened_id = None

                is_done = (c.get("_status") == "complete")
                if a2.button("Complete" if not is_done else "Reopen", key=f"done_{topic}"):
                    set_followup_status(topic, "complete" if not is_done else "open")

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