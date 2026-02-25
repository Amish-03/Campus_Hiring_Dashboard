import sqlite3
import os
from typing import List, Optional
from ..models import EmailRecord, HiringDetail, StructuredHiringData


class DatabaseManager:
    def __init__(self, db_path: str = "data/campus_hiring.db"):
        self.db_path = db_path
        self._initialize_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Table for raw emails
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emails (
                    id TEXT PRIMARY KEY,
                    subject TEXT,
                    sender TEXT,
                    date TEXT,
                    body TEXT,
                    raw_body TEXT
                )
            """)
            # Legacy table for regex-based extraction
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hiring_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id TEXT,
                    company TEXT,
                    role TEXT,
                    cgpa_cutoff REAL,
                    ctc REAL,
                    registration_deadline TEXT,
                    test_date TEXT,
                    interview_date TEXT,
                    extracted_at TEXT,
                    FOREIGN KEY (email_id) REFERENCES emails (id)
                )
            """)
            # New table for LLM-based structured extraction
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS structured_hiring_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id TEXT UNIQUE,
                    company_name TEXT,
                    role TEXT,
                    ctc_lpa REAL,
                    cgpa_cutoff REAL,
                    eligibility_branches TEXT,
                    registration_deadline TEXT,
                    test_date TEXT,
                    interview_date TEXT,
                    selection_count INTEGER,
                    total_openings INTEGER,
                    extracted_at TEXT,
                    model_used TEXT,
                    FOREIGN KEY (email_id) REFERENCES emails (id)
                )
            """)
            conn.commit()

    # ── Email operations ──────────────────────────────────────

    def save_email(self, email: EmailRecord):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO emails (id, subject, sender, date, body, raw_body)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (email.id, email.subject, email.sender, email.date, email.body, email.raw_body))
            conn.commit()

    def get_all_emails(self) -> List[EmailRecord]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM emails")
            return [EmailRecord(**row) for row in cursor.fetchall()]

    def get_hiring_emails(self, email_ids: List[str]) -> List[EmailRecord]:
        """Get specific emails by their IDs."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(email_ids))
            cursor.execute(f"SELECT * FROM emails WHERE id IN ({placeholders})", email_ids)
            return [EmailRecord(**row) for row in cursor.fetchall()]

    # ── Legacy hiring details (regex-based) ───────────────────

    def save_hiring_detail(self, detail: HiringDetail):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO hiring_details (
                    email_id, company, role, cgpa_cutoff, ctc,
                    registration_deadline, test_date, interview_date, extracted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                detail.email_id, detail.company, detail.role, detail.cgpa_cutoff,
                detail.ctc, detail.registration_deadline, detail.test_date,
                detail.interview_date, detail.extracted_at
            ))
            conn.commit()

    def get_hiring_details(self) -> List[dict]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM hiring_details")
            return [dict(row) for row in cursor.fetchall()]

    # ── Structured hiring data (LLM-based) ────────────────────

    def save_structured_data(self, data: StructuredHiringData):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO structured_hiring_data (
                    email_id, company_name, role, ctc_lpa, cgpa_cutoff,
                    eligibility_branches, registration_deadline, test_date,
                    interview_date, selection_count, total_openings,
                    extracted_at, model_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.email_id, data.company_name, data.role, data.ctc_lpa,
                data.cgpa_cutoff, data.eligibility_branches,
                data.registration_deadline, data.test_date, data.interview_date,
                data.selection_count, data.total_openings,
                data.extracted_at, data.model_used
            ))
            conn.commit()

    def get_structured_data(self) -> List[dict]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, e.subject, e.date as email_date
                FROM structured_hiring_data s
                JOIN emails e ON s.email_id = e.id
                ORDER BY e.date DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def clear_structured_data(self):
        """Clear all LLM-extracted data (for re-extraction)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM structured_hiring_data")
            conn.commit()

    def get_extracted_email_ids(self) -> set:
        """Get email IDs that already have structured data (for resumability)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT email_id FROM structured_hiring_data")
            return {row[0] for row in cursor.fetchall()}

    def get_email_count(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM emails")
            return cursor.fetchone()[0]
