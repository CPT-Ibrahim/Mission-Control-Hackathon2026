# storage.py
import json
import sqlite3
from typing import Dict, Any, Iterable, List, Optional, Tuple

DB_PATH = "cache.sqlite"


def init_db(db_path: str = DB_PATH) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS triage (
            message_id TEXT PRIMARY KEY,
            result_json TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS followup (
            topic TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            signature TEXT
        )
        """
    )

    # Backward compatible upgrades (if table existed earlier without columns)
    for col, ddl in [
        ("status", "ALTER TABLE followup ADD COLUMN status TEXT NOT NULL DEFAULT 'open'"),
        ("signature", "ALTER TABLE followup ADD COLUMN signature TEXT"),
    ]:
        try:
            cur.execute(ddl)
        except Exception:
            pass

    con.commit()
    con.close()


def get_triage(message_ids: Iterable[str], db_path: str = DB_PATH) -> Dict[str, Dict[str, Any]]:
    ids = list(dict.fromkeys(message_ids))
    if not ids:
        return {}

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    placeholders = ",".join(["?"] * len(ids))
    cur.execute(f"SELECT message_id, result_json FROM triage WHERE message_id IN ({placeholders})", ids)

    out: Dict[str, Dict[str, Any]] = {}
    for mid, rjson in cur.fetchall():
        try:
            out[mid] = json.loads(rjson)
        except Exception:
            pass

    con.close()
    return out


def upsert_triage(results: List[Dict[str, Any]], db_path: str = DB_PATH) -> None:
    if not results:
        return

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    rows = []
    for r in results:
        mid = r.get("id")
        if not mid:
            continue
        rows.append((mid, json.dumps(r, ensure_ascii=False)))

    cur.executemany(
        "INSERT OR REPLACE INTO triage (message_id, result_json) VALUES (?, ?)",
        rows,
    )
    con.commit()
    con.close()


def get_followup(topics: Iterable[str], db_path: str = DB_PATH) -> Dict[str, Dict[str, Any]]:
    tps = list(dict.fromkeys([t for t in topics if t]))
    if not tps:
        return {}

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    placeholders = ",".join(["?"] * len(tps))
    cur.execute(
        f"SELECT topic, result_json, status, signature FROM followup WHERE topic IN ({placeholders})",
        tps,
    )

    out: Dict[str, Dict[str, Any]] = {}
    for topic, rjson, status, signature in cur.fetchall():
        try:
            data = json.loads(rjson)
        except Exception:
            data = {}
        data["_status"] = status
        data["_signature"] = signature
        out[topic] = data

    con.close()
    return out


def upsert_followup(results: List[Dict[str, Any]], signature_map: Optional[Dict[str, str]] = None, db_path: str = DB_PATH) -> None:
    if not results:
        return

    signature_map = signature_map or {}

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    rows: List[Tuple[str, str, str]] = []
    for r in results:
        topic = r.get("topic")
        if not topic:
            continue
        sig = signature_map.get(topic)
        rows.append((topic, json.dumps(r, ensure_ascii=False), sig))

    cur.executemany(
        """
        INSERT INTO followup (topic, result_json, signature)
        VALUES (?, ?, ?)
        ON CONFLICT(topic) DO UPDATE SET
          result_json=excluded.result_json,
          signature=excluded.signature
        """,
        rows,
    )

    con.commit()
    con.close()


def set_followup_status(topic: str, status: str, db_path: str = DB_PATH) -> None:
    if status not in ("open", "complete"):
        status = "open"

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("UPDATE followup SET status=? WHERE topic=?", (status, topic))
    con.commit()
    con.close()