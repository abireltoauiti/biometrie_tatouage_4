"""
modules/database/db.py — SQLite schema and all CRUD operations.

GDPR: surveillance.db contains personal data. Use delete_person() for erasure.
"""

import sqlite3
import os
from typing import Optional
from config import DB_PATH

_DDL = """
CREATE TABLE IF NOT EXISTS persons (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name   TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('authorized','unauthorized'))
);
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name  TEXT NOT NULL,
    category     TEXT NOT NULL,
    camera_id    TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    image_path   TEXT NOT NULL,
    hash_sha256  TEXT NOT NULL,
    alert_level  TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript(_DDL)
        conn.commit()
    print(f"[Database] Ready: {DB_PATH}")


def insert_person_if_missing(name: str, status: str) -> None:
    with _connect() as conn:
        conn.execute("INSERT OR IGNORE INTO persons (name,status) VALUES (?,?)",
                     (name, status))
        conn.commit()


def get_person_status(name: str) -> Optional[str]:
    """Returns 'authorized', 'unauthorized', or None."""
    with _connect() as conn:
        row = conn.execute("SELECT status FROM persons WHERE name=?", (name,)).fetchone()
    return row["status"] if row else None


def list_persons() -> list:
    with _connect() as conn:
        return conn.execute("SELECT * FROM persons ORDER BY name").fetchall()


def insert_event(person_name, category, camera_id,
                 timestamp, image_path, hash_sha256, alert_level) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO events
               (person_name,category,camera_id,timestamp,image_path,hash_sha256,alert_level)
               VALUES (?,?,?,?,?,?,?)""",
            (person_name, category, camera_id, timestamp,
             image_path, hash_sha256, alert_level))
        conn.commit()
        return cur.lastrowid


def get_event_by_image_path(image_path: str):
    norm = os.path.normpath(image_path)
    with _connect() as conn:
        row = conn.execute("SELECT * FROM events WHERE image_path=?", (norm,)).fetchone()
        if row:
            return row
        return conn.execute("SELECT * FROM events WHERE image_path=?",
                            (os.path.abspath(image_path),)).fetchone()


def get_recent_events(limit: int = 100) -> list:
    with _connect() as conn:
        return conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?",
                            (limit,)).fetchall()


def delete_person(name: str) -> None:
    """GDPR erasure: removes person + all their events from the database."""
    with _connect() as conn:
        conn.execute("DELETE FROM events  WHERE person_name=?", (name,))
        conn.execute("DELETE FROM persons WHERE name=?",        (name,))
        conn.commit()
    print(f"[Database] '{name}' erased. Delete their image files manually.")
