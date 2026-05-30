import json
import os
import sqlite3
import zlib
from datetime import datetime
from typing import Optional


DB_PATH = os.path.join(os.path.dirname(__file__), "ecochem_vault.db")


def _compress(data: dict | list) -> bytes:
    return zlib.compress(json.dumps(data, default=str).encode("utf-8"))


def _decompress(blob: bytes) -> dict | list:
    return json.loads(zlib.decompress(blob).decode("utf-8"))


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS target_compounds (
            compound_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            smiles TEXT NOT NULL,
            compound_class TEXT NOT NULL,
            functional_groups BLOB,
            is_chiral INTEGER DEFAULT 0,
            natural_enantiomer TEXT
        );

        CREATE TABLE IF NOT EXISTS extraction_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            compound_id INTEGER,
            hba_name TEXT NOT NULL,
            hbd_name TEXT NOT NULL,
            molar_ratio REAL NOT NULL,
            method TEXT NOT NULL,
            temperature_c REAL,
            time_minutes INTEGER,
            yield_percent REAL,
            purity_percent REAL,
            isolation_steps BLOB,
            source_type TEXT DEFAULT 'lab',
            citation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (compound_id) REFERENCES target_compounds(compound_id)
        );

        CREATE INDEX IF NOT EXISTS idx_compound_class ON target_compounds(compound_class);
        CREATE INDEX IF NOT EXISTS idx_extraction_link ON extraction_records(compound_id);
        """)
        conn.commit()
    finally:
        conn.close()


def insert_target_compound(
    name: str,
    smiles: str,
    compound_class: str,
    functional_groups: Optional[dict] = None,
    is_chiral: int = 0,
    natural_enantiomer: Optional[str] = None,
) -> int:
    conn = get_connection()
    try:
        fg_blob = _compress(functional_groups) if functional_groups else None
        cur = conn.execute(
            """INSERT OR IGNORE INTO target_compounds
               (name, smiles, compound_class, functional_groups, is_chiral, natural_enantiomer)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name.strip(), smiles, compound_class, fg_blob, is_chiral, natural_enantiomer),
        )
        conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute("SELECT compound_id FROM target_compounds WHERE name = ?", (name.strip(),)).fetchone()
        return row["compound_id"] if row else -1
    finally:
        conn.close()


def get_target_compound_by_name(name: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM target_compounds WHERE name = ?", (name.strip(),)).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d["functional_groups"]:
            d["functional_groups"] = _decompress(d["functional_groups"])
        return d
    finally:
        conn.close()


def get_target_compound_by_class(compound_class: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM target_compounds WHERE compound_class = ?", (compound_class,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d["functional_groups"]:
                d["functional_groups"] = _decompress(d["functional_groups"])
            result.append(d)
        return result
    finally:
        conn.close()


def insert_extraction_record(
    compound_id: int,
    hba_name: str,
    hbd_name: str,
    molar_ratio: float,
    method: str,
    temperature_c: Optional[float] = None,
    time_minutes: Optional[int] = None,
    yield_percent: Optional[float] = None,
    purity_percent: Optional[float] = None,
    isolation_steps: Optional[list] = None,
    source_type: str = "lab",
    citation: Optional[str] = None,
) -> int:
    conn = get_connection()
    try:
        steps_blob = _compress(isolation_steps) if isolation_steps else None
        cur = conn.execute(
            """INSERT INTO extraction_records
               (compound_id, hba_name, hbd_name, molar_ratio, method,
                temperature_c, time_minutes, yield_percent, purity_percent,
                isolation_steps, source_type, citation)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (compound_id, hba_name, hbd_name, molar_ratio, method,
             temperature_c, time_minutes, yield_percent, purity_percent,
             steps_blob, source_type, citation),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_extraction_records(
    compound_id: Optional[int] = None,
    source_type: Optional[str] = None,
) -> list[dict]:
    conn = get_connection()
    try:
        query = "SELECT * FROM extraction_records WHERE 1=1"
        params = []
        if compound_id is not None:
            query += " AND compound_id = ?"
            params.append(compound_id)
        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query, params).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d["isolation_steps"]:
                d["isolation_steps"] = _decompress(d["isolation_steps"])
            result.append(d)
        return result
    finally:
        conn.close()


def get_all_valid_records(min_records: int = 3) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT r.*, t.name as compound_name, t.smiles
               FROM extraction_records r
               JOIN target_compounds t ON r.compound_id = t.compound_id
               WHERE r.yield_percent IS NOT NULL
                 AND r.temperature_c IS NOT NULL
                 AND r.time_minutes IS NOT NULL
               ORDER BY r.created_at DESC"""
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d["isolation_steps"]:
                d["isolation_steps"] = _decompress(d["isolation_steps"])
            result.append(d)
        return result
    finally:
        conn.close()


def get_compound_classes_with_data() -> list[str]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT t.compound_class FROM target_compounds t"
        ).fetchall()
        return [r["compound_class"] for r in rows if r["compound_class"]]
    finally:
        conn.close()


def get_compound_names() -> list[str]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT name FROM target_compounds ORDER BY name").fetchall()
        return [r["name"] for r in rows]
    finally:
        conn.close()


def extraction_record_count() -> int:
    conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) as cnt FROM extraction_records").fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()
