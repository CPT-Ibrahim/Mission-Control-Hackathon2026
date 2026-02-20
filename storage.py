# storage.py
import json
import sqlite3
from typing import Dict, Any, Iterable, List

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
