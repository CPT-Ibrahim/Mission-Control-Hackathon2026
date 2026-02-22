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
)
from llm_triage import triage_emails
from followup_ai import summarize_followups

load_dotenv()
init_db()

st.set_page_config(page_title="AI Gmail", layout="wide")

# ----------------------------
# Stable polling
# - Only poll in LIST view
# - Throttle Gmail calls
# ----------------------------
if "inbox_view" not in st.session_state:
    st.session_state.inbox_view = "list"  # list | detail

if st.session_state.inbox_view == "list":
    try:
        st.autorefresh(interval=90_000, key="poll")
    except Exception:
        pass

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
<style>
h1 { display:none; }
.header { font-size: 18px; font-weight: 800; margin: 4px 0 6px 0; }
.stTextInput input { font-size: 18px; padding: 10px 14px; border-radius: 999px; }

/* compact clickable subject button, force LEFT alignment */
div.stButton > button {
  border-radius: 10px;
  justify-content:flex-start !important;
  text-align:left !important;
  align-items: center !important;
  padding: 3px 8px !important;
  line-height: 1.05 !important;
  width: 100% !important;
}
div.stButton > button p { 
  text-align:left !important;
  width: 100% !important;
  margin: 0 !important;
}

/* compact row */
.row-summary { font-size: 0.82rem; opacity: 0.85; margin-top: 1px; }
.divline { border-bottom: 1px solid rgba(255,255,255,0.06); margin: 5px 0; }

.mail-date { opacity:0.75; font-size:0.78rem; text-align:right; white-space:nowrap; }

/* status badges */
.badge {
  display:inline-block; padding:1px 8px; border-radius:999px;
  font-size:11px; border:1px solid rgba(255,255,255,0.16);
  background:rgba(255,255,255,0.02);
  white-space: nowrap;
}
.badge-urgent { border-color: rgba(255,165,0,0.55); }
.badge-spam   { border-color: rgba(255,70,70,0.55); }
.badge-ok     { border-color: rgba(120,255,170,0.35); }

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

/* detail view: ultra-compact header rows */
.detail-top { font-size: 0.95rem; font-weight: 750; margin-top: 4px; }
.detail-from { opacity: 0.8; font-size: 0.82rem; margin-top: 2px; }
.detail-mini { opacity: 0.85; font-size: 0.82rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<div class='header'>AI Gmail</div>", unsafe_allow_html=True)

# ----------------------------
# State
# ----------------------------
if "emails" not in st.session_state: st.session_state.emails = []
if "max_emails" not in st.session_state: st.session_state.max_emails = 200
if "pending_max_emails" not in st.session_state: st.session_state.pending_max_emails = st.session_state.max_emails

# Filters (INBOX ONLY)
if "type_filter" not in st.session_state: st.session_state.type_filter = []
if "inbox_status_filter" not in st.session_state: st.session_state.inbox_status_filter = "All"  # All/Urgent/Spam/No action

# Dashboard-only filter
if "show_completed_cards" not in st.session_state: st.session_state.show_completed_cards = False

if "active_topic" not in st.session_state: st.session_state.active_topic = None
if "opened_id" not in st.session_state: st.session_state.opened_id = None
if "full_cache" not in st.session_state: st.session_state.full_cache = {}
if "reply_cache" not in st.session_state: st.session_state.reply_cache = {}
if "last_sync_ts" not in st.session_state: st.session_state.last_sync_ts = 0.0

api_ok = bool(os.getenv("DEEPSEEK_API_KEY"))
service = get_service()

def ds_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

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
    return (sum(vals) / len(vals)) if vals else None

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
    """
    Inbox-only convenience label.
    Spam is excluded (spam has its own filter).
    No-action means: not urgent AND action in {Read, Ignore} OR missing triage.
    """
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

def auto_fetch_new_emails():
    # No polling while reading an email
    if st.session_state.inbox_view == "detail":
        return

    now = time.time()
    if now - st.session_state.last_sync_ts < 25:
        return
    st.session_state.last_sync_ts = now

    ids = list_message_ids(service, max_results=st.session_state.max_emails)
    if not ids:
        return

    existing_map = {e["id"]: e for e in st.session_state.emails}
    current_ids = [e["id"] for e in st.session_state.emails]
    if ids == current_ids:
        return

    new_list = []
    for mid in ids:
        if mid in existing_map:
            new_list.append(existing_map[mid])
        else:
            new_list.append(get_message_metadata(service, mid))
    st.session_state.emails = new_list

def auto_ai_run_new_only():
    # No triage while reading an email
    if st.session_state.inbox_view == "detail":
        return
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

def apply_inbox_filters(emails, q: str):
    """
    Inbox-only filters: mail types + status filter + search.
    """
    out = emails
    if st.session_state.active_topic:
        out = [e for e in out if mail_type(e) == st.session_state.active_topic]
    elif st.session_state.type_filter:
        want = set(st.session_state.type_filter)
        out = [e for e in out if mail_type(e) in want]

    # status filter (inbox only)
    sf = st.session_state.inbox_status_filter
    if sf == "Urgent":
        out = [e for e in out if is_urgent(e)]
    elif sf == "Spam":
        out = [e for e in out if is_spam(e)]
    elif sf == "No action":
        out = [e for e in out if no_action_needed(e)]

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

def status_badge_for_row(e) -> str:
    """
    Single status badge placed between subject and date.
    Priority: Urgent > Spam > No action > none
    """
    if is_urgent(e):
        return "<span class='badge badge-urgent'>Urgent</span>"
    if is_spam(e):
        return "<span class='badge badge-spam'>Spam</span>"
    if no_action_needed(e):
        return "<span class='badge badge-ok'>No action</span>"
    return ""

def cb_open_email(eid: str):
    st.session_state.opened_id = eid
    st.session_state.inbox_view = "detail"

def cb_close_email():
    st.session_state.opened_id = None
    st.session_state.inbox_view = "list"

def get_full_body(eid: str) -> str:
    if eid in st.session_state.full_cache:
        return st.session_state.full_cache[eid]
    full = get_message_full(service, eid)
    body = (full.get("plain") or "").strip()
    body = html.unescape(body).replace("\xa0", " ")
    body = re.sub(r"[ \t]{5,}", "    ", body)
    st.session_state.full_cache[eid] = body
    return body

def cb_generate_reply(eid: str):
    selected = next((e for e in st.session_state.emails if e["id"] == eid), None)
    if not selected:
        return
    if is_spam(selected):
        st.session_state.reply_cache[eid] = "No reply suggested (spam/promotion)."
        return

    tri = selected.get("triage") or {}
    body_clip = get_full_body(eid)[:3000]

    sys = (
        "Draft a concise professional email reply.\n"
        "Return ONLY the reply body text.\n"
        "If no reply is needed, return: No reply needed.\n"
    )
    user = json.dumps(
        {
            "from": selected.get("from",""),
            "subject": selected.get("subject",""),
            "ai_summary": tri.get("summary",""),
            "ai_action": tri.get("action",""),
            "email_body_excerpt": body_clip,
        },
        ensure_ascii=False,
    )
    try:
        client = ds_client()
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            stream=False,
        )
        txt = (resp.choices[0].message.content or "").strip()
        st.session_state.reply_cache[eid] = txt if txt else "No reply needed."
    except Exception:
        st.session_state.reply_cache[eid] = "Reply generation failed."

# ----------------------------
# Load + sync
# ----------------------------
if not st.session_state.emails:
    ids = list_message_ids(service, max_results=st.session_state.max_emails)
    st.session_state.emails = [get_message_metadata(service, mid) for mid in ids]

auto_fetch_new_emails()
attach_triage()
auto_ai_run_new_only()

# ----------------------------
# Header search + global accuracy
# ----------------------------
search_text = st.text_input("", value=st.session_state.get("search_text",""), placeholder="Search mail (live)…", key="search_text")

acc = overall_accuracy(st.session_state.emails)
acc_text = f"AI accuracy: {acc:.2f}%" if acc is not None else "AI accuracy: N/A"

filtered = apply_inbox_filters(st.session_state.emails, search_text)
st.caption(f"Showing {len(filtered)} of {len(st.session_state.emails)}  |  {acc_text}")

# ----------------------------
# Sidebar (apply button for emails to load)
# ----------------------------
with st.sidebar:
    st.markdown("### Auto sync")
    st.caption("Checks Gmail periodically. No manual refresh.")

    st.markdown("### Load")
    st.session_state.pending_max_emails = st.slider("Emails to load", 20, 800, st.session_state.pending_max_emails, 10)
    if st.button("Apply"):
        # apply change and force next poll to fetch immediately
        st.session_state.max_emails = st.session_state.pending_max_emails
        st.session_state.last_sync_ts = 0.0
        st.session_state.emails = []
        st.session_state.full_cache = {}
        st.session_state.reply_cache = {}
        st.session_state.inbox_view = "list"
        st.session_state.opened_id = None
        st.rerun()

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

                # In detail view, keep accuracy available
                badges = []
                if is_urgent(selected): badges.append("<span class='badge badge-urgent'>Urgent</span>")
                if is_spam(selected):   badges.append("<span class='badge badge-spam'>Spam</span>")
                if no_action_needed(selected): badges.append("<span class='badge badge-ok'>No action</span>")
                c = conf_pct(selected)
                if c is not None:
                    badges.append(f"<span class='badge badge-acc'>Accuracy {c:.1f}%</span>")
                if badges:
                    st.markdown(" ".join(badges), unsafe_allow_html=True)

                c1, c2 = st.columns([5, 1.4])
                with c1:
                    st.markdown(f"<div class='detail-mini'><b>AI:</b> {tri.get('summary','(no summary yet)')}</div>", unsafe_allow_html=True)
                with c2:
                    if eid not in st.session_state.reply_cache:
                        st.button("Reply", on_click=cb_generate_reply, args=(eid,), use_container_width=True)
                    else:
                        st.caption("Reply ready")

                if eid in st.session_state.reply_cache:
                    with st.expander("Suggested reply", expanded=False):
                        st.text_area("", st.session_state.reply_cache[eid], height=120, label_visibility="collapsed")
                        to_addr = extract_email_addr(selected.get("from",""))
                        subj = selected.get("subject","")
                        link = gmail_compose_link(to_addr, f"Re: {subj}" if subj else "Re:", st.session_state.reply_cache[eid])
                        st.markdown(f"[Open in Gmail compose]({link})")

                st.text_area("Full email", get_full_body(eid), height=560)

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

                    # Place status badge BETWEEN subject and date
                    c1, c2, c3 = st.columns([6.0, 1.2, 1.0])
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

        by_topic = {t: v for t, v in by_topic.items() if len(v) >= 2}

        if not by_topic:
            st.info("No real follow-up threads detected.")
        else:
            for t in list(by_topic.keys()):
                by_topic[t] = sorted(
                    by_topic[t],
                    key=lambda e: (parse_date_safe(e.get("date","")) or datetime.min.replace(tzinfo=timezone.utc)),
                    reverse=True,
                )

            topics = list(by_topic.keys())
            cached_cards = get_followup(topics)

            allow_dashboard_update = (st.session_state.inbox_view == "list")

            if allow_dashboard_update and api_ok:
                to_build = []
                sig_map = {}
                for t in topics:
                    sig = f"{t}:{by_topic[t][0]['id']}:{len(by_topic[t])}"
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
                    with st.spinner("Updating dashboard..."):
                        cards = summarize_followups(payload)
                    cards = [c for c in cards if c.get("include", True) is True]
                    upsert_followup(cards, signature_map=sig_map)
                    cached_cards = get_followup(topics)

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
                acc2 = float(c.get("accuracy", 0))

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

                a1, a2, a3 = boxc.columns(3)

                a1.button("Show emails", key=f"show_{topic}", on_click=cb_show_topic, args=(topic,))
                a2.button(
                    "Reopen" if done else "Complete",
                    key=f"done_{topic}",
                    on_click=set_followup_status,
                    args=(topic, "open" if done else "complete"),
                )

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