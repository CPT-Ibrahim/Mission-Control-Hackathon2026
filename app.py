import os
import streamlit as st
from dotenv import load_dotenv

from gmail_client import get_service, list_message_ids, get_message_metadata
from storage import init_db, get_triage, upsert_triage
from llm_triage import triage_emails, TOPICS

load_dotenv()
init_db()

st.set_page_config(page_title="Mission Control Inbox", layout="wide")
st.title("Mission Control Inbox")

api_ok = bool(os.getenv("DEEPSEEK_API_KEY"))
if not api_ok:
    st.warning("DEEPSEEK_API_KEY missing. AI triage disabled until you add it to .env")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    max_emails = st.slider("Emails to load", 20, 200, 80, 10)
with col2:
    model = st.selectbox("Model", ["deepseek-chat", "deepseek-reasoner"], index=0)
with col3:
    query = st.text_input("Gmail search (optional)", value="")

if "emails" not in st.session_state:
    st.session_state.emails = []
if "bucket" not in st.session_state:
    st.session_state.bucket = "All"

service = get_service()

cA, cB = st.columns([1, 5])
with cA:
    if st.button("Refresh emails"):
        st.session_state.emails = []

if not st.session_state.emails:
    ids = list_message_ids(service, max_results=max_emails, query=query if query.strip() else None)
    st.session_state.emails = [get_message_metadata(service, mid) for mid in ids]

emails = st.session_state.emails
email_ids = [e["id"] for e in emails]

cached = get_triage(email_ids)
for e in emails:
    e["triage"] = cached.get(e["id"])

to_triage = [e for e in emails if e.get("triage") is None]
btn_label = f"Run AI triage ({len(to_triage)} new)"

if st.button(btn_label, disabled=(not api_ok or len(to_triage) == 0)):
    BATCH = 25
    new_results = []
    for i in range(0, len(to_triage), BATCH):
        batch = to_triage[i:i+BATCH]
        new_results.extend(triage_emails(batch, model=model))
    upsert_triage(new_results)

    cached = get_triage(email_ids)
    for e in emails:
        e["triage"] = cached.get(e["id"])

def topic_of(e):
    t = (e.get("triage") or {}).get("topic")
    return t if t in TOPICS else "Other"

urgent = [e for e in emails if (e.get("triage") or {}).get("is_urgent") is True]
spam = [e for e in emails if (e.get("triage") or {}).get("is_spam") is True]

topic_map = {t: [] for t in TOPICS}
topic_map["Other"] = []
for e in emails:
    topic_map[topic_of(e)].append(e)

st.subheader("Dashboard")

row = st.columns(6)
if row[0].button(f"All ({len(emails)})"): st.session_state.bucket = "All"
if row[1].button(f"Urgent ({len(urgent)})"): st.session_state.bucket = "Urgent"
if row[2].button(f"Spam ({len(spam)})"): st.session_state.bucket = "Spam"
if row[3].button(f"Project 1 ({len(topic_map.get('Project 1', []))})"): st.session_state.bucket = "Project 1"
if row[4].button(f"University ({len(topic_map.get('University', []))})"): st.session_state.bucket = "University"
if row[5].button(f"Other ({len(topic_map.get('Other', []))})"): st.session_state.bucket = "Other"

bucket = st.session_state.bucket
if bucket == "Urgent":
    filtered = urgent
elif bucket == "Spam":
    filtered = spam
elif bucket == "All":
    filtered = emails
else:
    filtered = topic_map.get(bucket, [])

left, right = st.columns([2, 3])

with left:
    st.subheader(f"Emails: {bucket}")
    if not filtered:
        st.info("No emails in this bucket.")
        selected = None
    else:
        options = [f"{e.get('subject','(no subject)')[:80]} | {e.get('from','')[:40]}" for e in filtered]
        idx = st.selectbox("Select", range(len(filtered)), format_func=lambda i: options[i])
        selected = filtered[idx]

with right:
    st.subheader("Details")
    if not selected:
        st.info("Select an email from the left.")
    else:
        st.write("**From:**", selected.get("from", ""))
        st.write("**Subject:**", selected.get("subject", ""))
        st.write("**Date:**", selected.get("date", ""))
        st.write("**Snippet:**", selected.get("snippet", ""))

        tri = selected.get("triage") or {}
        st.markdown("---")
        if tri:
            st.write("**Topic:**", tri.get("topic"))
            st.write("**Urgent:**", tri.get("is_urgent"))
            st.write("**Spam:**", tri.get("is_spam"))
            st.write("**Action:**", tri.get("action"))
            st.write("**Summary:**", tri.get("summary"))
            st.write("**Confidence:**", tri.get("confidence"))
        else:
            st.warning("No AI triage for this email yet. Click 'Run AI triage'.")
