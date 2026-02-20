import streamlit as st
from gmail_client import get_service, list_message_ids, get_message_metadata

st.set_page_config(page_title="Mission Control Inbox", layout="wide")
st.title("Mission Control Inbox")

service = get_service()
ids = list_message_ids(service, max_results=20)

emails = [get_message_metadata(service, mid) for mid in ids]

st.success(f"Fetched {len(emails)} emails.")
st.dataframe(emails)
