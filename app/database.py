"""
SQLite database layer for the Edge Attendance system.

Tables: employees, face_embeddings, attendance_logs, users.
Embeddings stored as binary BLOB for fast I/O.
Passwords hashed with bcrypt.
"""
import sqlite3
import struct
import numpy as np
import bcrypt
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any, Optional

from .config import DB_PATH

# ── Embedding serialization (BLOB) ─────────────────────

def embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Serialize a numpy embedding to binary BLOB."""
    flat = embedding.flatten().astype(np.float32)
    return struct.pack(f'{len(flat)}f', *flat)


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialize binary BLOB back to numpy array."""
    n = len(blob) // 4
    return np.array(struct.unpack(f'{n}f', blob), dtype=np.float32).reshape(1, -1)


# ── Connection ──────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-8000")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


# ── Schema ──────────────────────────────────────────────

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_code TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        department TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL,
        image_path TEXT,
        embedding BLOB NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(employee_id) REFERENCES employees(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL,
        check_time TEXT DEFAULT CURRENT_TIMESTAMP,
        check_type TEXT DEFAULT 'CHECK_IN',
        confidence REAL,
        image_path TEXT,
        FOREIGN KEY(employee_id) REFERENCES employees(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'employee',
        employee_id INTEGER,
        FOREIGN KEY(employee_id) REFERENCES employees(id)
    )
    """)

    # Default admin account (password: admin123)
    admin_hash = bcrypt.hashpw(b"admin123", bcrypt.gensalt()).decode("utf-8")
    cur.execute("""
    INSERT OR IGNORE INTO users(username, password_hash, role)
    VALUES('admin', ?, 'admin')
    """, (admin_hash,))

    conn.commit()
    conn.close()


# ── Employee CRUD ───────────────────────────────────────

def create_employee(employee_code: str, full_name: str, department: str = "") -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO employees(employee_code, full_name, department) VALUES (?, ?, ?)",
        (employee_code, full_name, department),
    )
    conn.commit()
    cur.execute("SELECT id FROM employees WHERE employee_code = ?", (employee_code,))
    employee_id = cur.fetchone()["id"]
    conn.close()
    return employee_id


def list_employees() -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT id, employee_code, full_name, department, created_at
        FROM employees ORDER BY id DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_employee_by_code(employee_code: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM employees WHERE employee_code = ?", (employee_code,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Face Embedding CRUD ────────────────────────────────

def save_embedding(employee_id: int, embedding: np.ndarray, image_path: str = ""):
    """Save a face embedding as binary BLOB."""
    blob = embedding_to_blob(embedding)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO face_embeddings(employee_id, image_path, embedding) VALUES (?, ?, ?)",
        (employee_id, image_path, blob),
    )
    conn.commit()
    conn.close()


def load_embeddings() -> List[Dict[str, Any]]:
    """Load all face embeddings with employee info. Embeddings returned as numpy arrays."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT fe.id, fe.employee_id, fe.embedding, e.employee_code, e.full_name
        FROM face_embeddings fe
        JOIN employees e ON e.id = fe.employee_id
    """).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "employee_id": r["employee_id"],
            "employee_code": r["employee_code"],
            "full_name": r["full_name"],
            "embedding": blob_to_embedding(r["embedding"]),
        }
        for r in rows
    ]


# ── Attendance CRUD ─────────────────────────────────────

def add_attendance(
    employee_id: int,
    confidence: float,
    check_type: str = "CHECK_IN",
    image_path: str = "",
):
    # Use explicit local time instead of SQLite CURRENT_TIMESTAMP (which is UTC)
    local_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO attendance_logs(employee_id, check_time, confidence, check_type, image_path) VALUES (?, ?, ?, ?, ?)",
        (employee_id, local_now, confidence, check_type, image_path),
    )
    conn.commit()
    conn.close()


def get_last_attendance_today(employee_id: int, check_type: str) -> Optional[Dict[str, Any]]:
    """Get the most recent attendance log of the given type for an employee today."""
    today = date.today().isoformat()
    conn = get_conn()
    row = conn.execute("""
        SELECT * FROM attendance_logs
        WHERE employee_id = ?
          AND check_type = ?
          AND date(check_time) = ?
        ORDER BY check_time DESC
        LIMIT 1
    """, (employee_id, check_type, today)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_attendance(
    limit: int = 100,
    employee_id: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    department: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List attendance logs with optional filters."""
    query = """
        SELECT al.id, e.employee_code, e.full_name, e.department,
               al.check_time, al.check_type, al.confidence, al.image_path
        FROM attendance_logs al
        JOIN employees e ON e.id = al.employee_id
        WHERE 1=1
    """
    params = []

    if employee_id is not None:
        query += " AND al.employee_id = ?"
        params.append(employee_id)

    if date_from:
        query += " AND date(al.check_time) >= ?"
        params.append(date_from)

    if date_to:
        query += " AND date(al.check_time) <= ?"
        params.append(date_to)

    if department:
        query += " AND e.department = ?"
        params.append(department)

    query += " ORDER BY al.check_time DESC LIMIT ?"
    params.append(limit)

    conn = get_conn()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_attendance_summary(date_str: Optional[str] = None) -> Dict[str, Any]:
    """Get attendance summary stats for a given date (default: today)."""
    if date_str is None:
        date_str = date.today().isoformat()

    conn = get_conn()

    total_employees = conn.execute("SELECT COUNT(*) as cnt FROM employees").fetchone()["cnt"]

    checked_in = conn.execute("""
        SELECT COUNT(DISTINCT employee_id) as cnt
        FROM attendance_logs
        WHERE check_type = 'CHECK_IN' AND date(check_time) = ?
    """, (date_str,)).fetchone()["cnt"]

    checked_out = conn.execute("""
        SELECT COUNT(DISTINCT employee_id) as cnt
        FROM attendance_logs
        WHERE check_type = 'CHECK_OUT' AND date(check_time) = ?
    """, (date_str,)).fetchone()["cnt"]

    total_logs = conn.execute("""
        SELECT COUNT(*) as cnt
        FROM attendance_logs
        WHERE date(check_time) = ?
    """, (date_str,)).fetchone()["cnt"]

    conn.close()

    return {
        "date": date_str,
        "total_employees": total_employees,
        "checked_in": checked_in,
        "checked_out": checked_out,
        "not_checked_in": total_employees - checked_in,
        "total_logs": total_logs,
    }


# ── User / Auth ─────────────────────────────────────────

def verify_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify username/password. Returns user dict or None."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if row is None:
        return None

    stored_hash = row["password_hash"].encode("utf-8")
    if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "employee_id": row["employee_id"],
        }
    return None


def create_user(username: str, password: str, role: str = "employee", employee_id: Optional[int] = None):
    """Create a new user with bcrypt-hashed password."""
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users(username, password_hash, role, employee_id) VALUES (?, ?, ?, ?)",
        (username, pw_hash, role, employee_id),
    )
    conn.commit()
    conn.close()


def get_departments() -> List[str]:
    """Get list of distinct departments."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT department FROM employees WHERE department IS NOT NULL AND department != '' ORDER BY department"
    ).fetchall()
    conn.close()
    return [r["department"] for r in rows]