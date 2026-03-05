# AI Gmail (Mission Control)
Overview - https://youtu.be/-0G-lnpHp18
A Gmail-like inbox that automatically:
- Groups emails into **Mail Types** (DVSA, Devpost, Bank, NHS, etc.)
- Flags **Urgent**, **Spam/Promotion**, and **No action needed**
- Builds a **Follow-up Dashboard** for real back-and-forth threads that need action
- Lets you open an email inside the inbox pane, see an **AI summary**, and generate an **AI response**
- “Reply” opens a prefilled Gmail compose window (safe: app uses Gmail readonly access)

---

## Features

### Inbox
- Stacked Gmail-style rows: `MAIL TYPE | SUBJECT`
- Filters:
  - Mail Type (AI-generated)
  - Status: **Urgent / Spam / No action / All**
- Live search (filters instantly)
- Open email in-place (with Back arrow)
- Shows:
  - AI summary
  - “Reply” (opens Gmail compose)
  - “Generate AI response” (creates suggested reply text)

### Follow-up Dashboard
- Shows only topics that:
  - Are **not spam/promotions**
  - Have **back-and-forth** (>= 2 emails)
  - Need follow-up (urgent or action=Reply/Pay/Book/Follow-up)
- Actions per card:
  - Show emails (filters inbox)
  - Complete / Reopen (persisted in local cache)

### Performance & Caching
- Uses `cache.sqlite` to cache AI results + dashboard state
- Auto-sync checks Gmail periodically
- AI runs only on **new** emails (never reprocesses the same email unless code changes)

---

## Tech Stack
- Python + Streamlit UI
- Gmail API (readonly)
- DeepSeek (OpenAI-compatible API) for AI classification + follow-up summaries + suggested replies
- SQLite cache (`cache.sqlite`)

---

## Requirements
- Python 3.10+ (you have 3.12.10 ✅)
- A Google Cloud OAuth Client (Desktop) with Gmail API enabled
- A DeepSeek API key

---

## Setup (New PC / New Account)

### 1) Clone and install
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd mission-control
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
