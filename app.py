import os
import re
import streamlit as st
from dotenv import load_dotenv

from gmail_client import (
    get_service,
    list_message_ids,
    get_message_metadata,
    get_message_full,
)
from storage import init_db, get_triage, upsert_triage
from llm_triage import triage_emails

load_dotenv()
init_db()

st.set_page_config(page_title="Mission Control Inbox", layout="wide")

# ---- Simple CSS to make "flashcards" bigger
st.markdown(
    """
<style>
.mc-card {
  border: 1px solid rgba(49, 51, 63, 0.2);
  border-radius: 14px;
  padding: 14px 16px;
  margin: 10px 0;
  background: rgba(255,255,255,0.02);
}
.mc-card-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 8px;
}
.mc-item {
  border-top: 1px solid rgba(49, 51, 63, 0.15);
  padding-top: 10px;
  margin-top: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Mission Control Inbox")

api_ok = bool(os.getenv("DEEPSEEK_API_KEY"))  # using DeepSeek in your current setup
if not api_ok:
    st.warning("DEEPSEEK_API_KEY missing. AI Filtered tab will not work until you add it to .env")

# ---- Controls (shared)
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    max_emails = st.slider("Emails to load", 20, 500, 150, 10)
with col2:
    model = st.selectbox("AI Model", ["deepseek-chat", "deepseek-reasoner"], index=0)
with col3:
    query = st.text_input("Gmail search (optional)", value="")

# ---- Session state
if "emails" not in st.session_state:
    st.session_state.emails = []
if "full_cache" not in st.session_state:
    st.session_state.full_cache = {}  # message_id -> {"plain":..., "html":..., "mimeType":...}
if "selected_inbox_id" not in st.session_state:
    st.session_state.selected_inbox_id = None
if "selected_ai_id" not in st.session_state:
    st.session_state.selected_ai_id = None

service = get_service()

# ---- Load emails
def refresh_emails():
    ids = list_message_ids(service, max_results=max_emails, query=query if query.strip() else None)
    st.session_state.emails = [get_message_metadata(service, mid) for mid in ids]

if st.button("Refresh emails"):
    refresh_emails()

if not st.session_state.emails:
    refresh_emails()

emails = st.session_state.emails
email_ids = [e["id"] for e in emails]

# ---- Attach cached AI triage if exists
cached = get_triage(email_ids)
for e in emails:
    e["triage"] = cached.get(e["id"])

def norm_topic(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s&\-]", "", t)
    return t[:28] if t else "Unsorted"

def show_email_details(selected, right_col):
    if not selected:
        right_col.info("Select an email.")
        return

    right_col.write("**From:** " + (selected.get("from") or ""))
    right_col.write("**Subject:** " + (selected.get("subject") or ""))
    right_col.write("**Date:** " + (selected.get("date") or ""))
    right_col.write("**Snippet:** " + (selected.get("snippet") or ""))

    msg_id = selected["id"]
    full_cached = st.session_state.full_cache.get(msg_id)

    if right_col.button("Load full email", key=f"load_full_{msg_id}"):
        full_cached = get_message_full(service, msg_id)
        st.session_state.full_cache[msg_id] = full_cached

    if full_cached and full_cached.get("plain"):
        right_col.markdown("---")
        right_col.write("**Full email:**")
        right_col.text_area("Body", full_cached.get("plain", ""), height=360, key=f"body_{msg_id}")

# ---- Tabs
tab_inbox, tab_ai = st.tabs(["Inbox", "AI Filtered"])

# =========================
# TAB 1: Inbox (Gmail-like)
# =========================
with tab_inbox:
    st.caption("Inbox view (no AI). Select an email to preview, then load full message.")

    left, right = st.columns([2, 3])

    with left:
        st.subheader("All emails")
        options = [
            f"{(e.get('subject') or '(no subject)')[:80]}  |  {(e.get('from') or '')[:40]}"
            for e in emails
        ]

        # keep selection stable
        default_index = 0
        if st.session_state.selected_inbox_id:
            for i, e in enumerate(emails):
                if e["id"] == st.session_state.selected_inbox_id:
                    default_index = i
                    break

        idx = st.selectbox("Select", range(len(emails)), index=default_index, format_func=lambda i: options[i])
        st.session_state.selected_inbox_id = emails[idx]["id"]

    selected = next((e for e in emails if e["id"] == st.session_state.selected_inbox_id), None)
    show_email_details(selected, right)

# ===========================================
# TAB 2: AI Filtered (flashcards + summaries)
# ===========================================
with tab_ai:
    st.caption("Press AI Filter to generate folders + clickable AI summaries.")

    # Run AI triage only when requested
    to_triage = [e for e in emails if e.get("triage") is None]
    btn_label = f"AI Filter ({len(to_triage)} new)"

    if st.button(btn_label, disabled=(not api_ok or len(to_triage) == 0)):
        # reuse existing topics to keep labels stable
        existing_topics = []
        for e in emails:
            tri = e.get("triage") or {}
            t = (tri.get("topic") or "").strip()
            if t:
                existing_topics.append(t)
        existing_topics = sorted(list(dict.fromkeys(existing_topics)))

        BATCH = 25
        new_results = []
        for i in range(0, len(to_triage), BATCH):
            batch = to_triage[i : i + BATCH]
            new_results.extend(triage_emails(batch, model=model, existing_topics=existing_topics))

        upsert_triage(new_results)

        # reload triage cache
        cached2 = get_triage(email_ids)
        for e in emails:
            e["triage"] = cached2.get(e["id"])

    # Build topic map
    topic_map = {}
    for e in emails:
        tri = e.get("triage") or {}
        t = norm_topic(tri.get("topic", "")) if tri else "Unsorted"
        topic_map.setdefault(t, []).append(e)

    # Sort folders by size (Unsorted last)
    topics_sorted = sorted(topic_map.items(), key=lambda kv: (kv[0] == "Unsorted", -len(kv[1]), kv[0]))

    left, right = st.columns([2, 3])

    with left:
        # If AI never ran, this will be mostly "Unsorted"
        for topic, items in topics_sorted[:20]:
            # Big flashcard
            st.markdown(
                f"""
<div class="mc-card">
  <div class="mc-card-title">{topic} ({len(items)})</div>
</div>
""",
                unsafe_allow_html=True,
            )

            # Show up to 6 emails inside each card
            shown = items[:6]
            for e in shown:
                tri = e.get("triage") or {}
                summary = tri.get("summary") or e.get("snippet") or ""
                action = tri.get("action") or ""
                urgent = tri.get("is_urgent") is True

                label = summary.strip()
                if len(label) > 90:
                    label = label[:90] + "…"

                # clickable summary
                if st.button(
                    f"{'⚠️ ' if urgent else ''}{label}",
                    key=f"ai_pick_{topic}_{e['id']}",
                ):
                    st.session_state.selected_ai_id = e["id"]

                if action:
                    st.caption(f"Action: {action}")

            if len(items) > 6:
                st.caption(f"+ {len(items) - 6} more")

    selected_ai = None
    if st.session_state.selected_ai_id:
        selected_ai = next((e for e in emails if e["id"] == st.session_state.selected_ai_id), None)

    show_email_details(selected_ai, right)