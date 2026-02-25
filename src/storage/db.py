import sqlite3
import os
from typing import List, Optional, Set
from ..models import EmailRecord, HiringDetail, StructuredHiringData, DriveRecord, AuditEntry


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

            # Table 1: Raw emails
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

            # Table 2: Legacy regex-based extraction
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

            # Table 3: LLM-based structured extraction (per email)
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

            # Table 4: Deduplicated drives (one per company+role)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drives (
                    drive_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    role TEXT,
                    ctc_lpa REAL,
                    cgpa_cutoff REAL,
                    eligibility_branches TEXT,
                    registration_deadline TEXT,
                    test_date TEXT,
                    interview_date TEXT,
                    selection_count INTEGER,
                    total_openings INTEGER,
                    email_count INTEGER DEFAULT 1,
                    first_seen TEXT,
                    last_updated TEXT,
                    UNIQUE(company_name, role)
                )
            """)

            # Table 5: Drive ↔ Email mapping
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drive_email_mapping (
                    drive_id INTEGER,
                    email_id TEXT,
                    PRIMARY KEY (drive_id, email_id),
                    FOREIGN KEY (drive_id) REFERENCES drives(drive_id),
                    FOREIGN KEY (email_id) REFERENCES emails(id)
                )
            """)

            # Table 6: Data audit trail
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id TEXT,
                    field_name TEXT,
                    original TEXT,
                    corrected TEXT,
                    rule TEXT,
                    timestamp TEXT
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

    def get_email_count(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM emails")
            return cursor.fetchone()[0]

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

    # ── Structured hiring data (LLM-based, per email) ─────────

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
        with self._get_connection() as conn:
            conn.cursor().execute("DELETE FROM structured_hiring_data")
            conn.commit()

    def get_extracted_email_ids(self) -> Set[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT email_id FROM structured_hiring_data")
            return {row[0] for row in cursor.fetchall()}

    # ── Drives (deduplicated) ─────────────────────────────────

    def save_drive(self, drive: DriveRecord) -> int:
        """Insert or update a drive. Returns drive_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO drives (
                    company_name, role, ctc_lpa, cgpa_cutoff,
                    eligibility_branches, registration_deadline, test_date,
                    interview_date, selection_count, total_openings,
                    email_count, first_seen, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(company_name, role) DO UPDATE SET
                    ctc_lpa = COALESCE(excluded.ctc_lpa, drives.ctc_lpa),
                    cgpa_cutoff = COALESCE(excluded.cgpa_cutoff, drives.cgpa_cutoff),
                    eligibility_branches = COALESCE(excluded.eligibility_branches, drives.eligibility_branches),
                    registration_deadline = COALESCE(excluded.registration_deadline, drives.registration_deadline),
                    test_date = COALESCE(excluded.test_date, drives.test_date),
                    interview_date = COALESCE(excluded.interview_date, drives.interview_date),
                    selection_count = COALESCE(excluded.selection_count, drives.selection_count),
                    total_openings = COALESCE(excluded.total_openings, drives.total_openings),
                    email_count = excluded.email_count,
                    last_updated = excluded.last_updated
            """, (
                drive.company_name, drive.role, drive.ctc_lpa, drive.cgpa_cutoff,
                drive.eligibility_branches, drive.registration_deadline,
                drive.test_date, drive.interview_date,
                drive.selection_count, drive.total_openings,
                drive.email_count, drive.first_seen, drive.last_updated
            ))
            conn.commit()
            # Get the drive_id
            cursor.execute(
                "SELECT drive_id FROM drives WHERE company_name = ? AND role IS ?",
                (drive.company_name, drive.role)
            )
            return cursor.fetchone()[0]

    def add_drive_email_mapping(self, drive_id: int, email_id: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO drive_email_mapping (drive_id, email_id) VALUES (?, ?)",
                (drive_id, email_id)
            )
            conn.commit()

    def get_drives(self) -> List[dict]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM drives ORDER BY first_seen DESC")
            return [dict(row) for row in cursor.fetchall()]

    def clear_drives(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM drive_email_mapping")
            cursor.execute("DELETE FROM drives")
            conn.commit()

    # ── Data audit ────────────────────────────────────────────

    def log_audit(self, entry: AuditEntry):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO data_audit (email_id, field_name, original, corrected, rule, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (entry.email_id, entry.field_name, entry.original, entry.corrected,
                  entry.rule, entry.timestamp))
            conn.commit()

    def get_audit_log(self) -> List[dict]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM data_audit ORDER BY timestamp DESC")
            return [dict(row) for row in cursor.fetchall()]

    def clear_audit_log(self):
        with self._get_connection() as conn:
            conn.cursor().execute("DELETE FROM data_audit")
            conn.commit()
