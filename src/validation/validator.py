"""
Post-processing validation with audit trail.

Applies sanity checks, normalizes values, standardizes dates, and logs
every correction to the data_audit table for transparency.
"""

import re
import logging
from datetime import datetime
from typing import Optional, List

from ..models import StructuredHiringData, AuditEntry
from ..storage.db import DatabaseManager

logger = logging.getLogger(__name__)

# Branch name standardization mapping
BRANCH_ALIASES = {
    "computer science": "CSE",
    "computer science and engineering": "CSE",
    "cs": "CSE",
    "cse": "CSE",
    "information technology": "IT",
    "it": "IT",
    "information science": "ISE",
    "ise": "ISE",
    "electronics and communication": "ECE",
    "electronics & communication": "ECE",
    "ec": "ECE",
    "ece": "ECE",
    "electrical and electronics": "EEE",
    "electrical & electronics": "EEE",
    "ee": "EEE",
    "eee": "EEE",
    "mechanical": "ME",
    "mechanical engineering": "ME",
    "me": "ME",
    "civil": "CE",
    "civil engineering": "CE",
    "ce": "CE",
    "chemical": "CH",
    "chemical engineering": "CH",
    "aeronautical": "AE",
    "automobile": "AU",
    "biotechnology": "BT",
    "mining": "MN",
    "electronics": "ECE",
    "artificial intelligence": "AI",
    "ai": "AI",
    "data science": "DS",
    "ds": "DS",
    "robotics": "ROB",
    "mca": "MCA",
    "mba": "MBA",
}


class Validator:
    """Validates and cleans LLM-extracted hiring data with audit logging."""

    MAX_CTC_LPA = 100.0
    MIN_CGPA = 0.0
    MAX_CGPA = 10.0

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db
        self._audit_buffer: List[AuditEntry] = []

    def validate(self, data: StructuredHiringData) -> StructuredHiringData:
        """Apply all validation rules. Returns cleaned data."""
        self._audit_buffer.clear()

        data.company_name = self._clean_company_name(data.email_id, data.company_name)
        data.ctc_lpa = self._validate_ctc(data.email_id, data.ctc_lpa)
        data.cgpa_cutoff = self._validate_cgpa(data.email_id, data.cgpa_cutoff)
        data.registration_deadline = self._validate_date(data.email_id, "registration_deadline", data.registration_deadline)
        data.test_date = self._validate_date(data.email_id, "test_date", data.test_date)
        data.interview_date = self._validate_date(data.email_id, "interview_date", data.interview_date)
        data.selection_count = self._validate_non_negative_int(data.email_id, "selection_count", data.selection_count)
        data.total_openings = self._validate_non_negative_int(data.email_id, "total_openings", data.total_openings)
        data.eligibility_branches = self._standardize_branches(data.email_id, data.eligibility_branches)
        data.role = self._clean_role(data.email_id, data.role)

        # Flush audit entries to DB
        if self.db and self._audit_buffer:
            for entry in self._audit_buffer:
                self.db.log_audit(entry)

        return data

    def _log(self, email_id: str, field: str, original: str, corrected: str, rule: str):
        """Buffer an audit entry."""
        self._audit_buffer.append(AuditEntry(
            email_id=email_id,
            field_name=field,
            original=str(original),
            corrected=str(corrected),
            rule=rule
        ))

    # ── CTC ───────────────────────────────────────────────────

    def _validate_ctc(self, email_id: str, ctc: Optional[float]) -> Optional[float]:
        if ctc is None:
            return None
        if ctc <= 0:
            self._log(email_id, "ctc_lpa", str(ctc), "null", "CTC <= 0")
            return None
        if ctc > self.MAX_CTC_LPA:
            self._log(email_id, "ctc_lpa", str(ctc), "null", f"CTC > {self.MAX_CTC_LPA} LPA (unrealistic)")
            return None
        return round(ctc, 2)

    # ── CGPA ──────────────────────────────────────────────────

    def _validate_cgpa(self, email_id: str, cgpa: Optional[float]) -> Optional[float]:
        if cgpa is None:
            return None
        if cgpa < self.MIN_CGPA or cgpa > self.MAX_CGPA:
            self._log(email_id, "cgpa_cutoff", str(cgpa), "null", f"CGPA outside {self.MIN_CGPA}-{self.MAX_CGPA}")
            return None
        return round(cgpa, 1)

    # ── Dates ─────────────────────────────────────────────────

    def _validate_date(self, email_id: str, field: str, date_str: Optional[str]) -> Optional[str]:
        if date_str is None or str(date_str).lower() in ("null", "none", ""):
            return None

        date_str = date_str.strip()
        formats = [
            "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
            "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y",
            "%d-%b-%Y", "%d-%B-%Y",
        ]

        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

        self._log(email_id, field, date_str, "null", "Unparseable date format")
        return None

    # ── Integers ──────────────────────────────────────────────

    def _validate_non_negative_int(self, email_id: str, field: str, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if value < 0:
            self._log(email_id, field, str(value), "null", "Negative integer")
            return None
        return value

    # ── Company Name ──────────────────────────────────────────

    def _clean_company_name(self, email_id: str, name: Optional[str]) -> str:
        if not name or str(name).lower() in ("unknown", "unknown company", "null", "none", "n/a", ""):
            return "Unknown"
        original = name
        name = name.strip().strip('"').strip("'").strip()
        # Remove common prefixes
        for prefix in ["M/s ", "M/s. ", "Messrs. ", "Messrs "]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        name = name.strip()
        if name != original:
            self._log(email_id, "company_name", original, name, "Company name cleanup")
        return name if name else "Unknown"

    # ── Branches ──────────────────────────────────────────────

    def _standardize_branches(self, email_id: str, branches: Optional[str]) -> Optional[str]:
        if not branches or str(branches).lower() in ("null", "none", "n/a", ""):
            return None

        original = branches
        branches = branches.replace(";", ",").replace("/", ",").replace("&", ",")
        parts = []
        for b in branches.split(","):
            b = b.strip()
            if not b:
                continue
            key = b.lower().strip()
            standardized = BRANCH_ALIASES.get(key, b.upper())
            parts.append(standardized)

        result = ", ".join(sorted(set(parts))) if parts else None
        if result != original:
            self._log(email_id, "eligibility_branches", original, str(result), "Branch standardization")
        return result

    # ── Role ──────────────────────────────────────────────────

    def _clean_role(self, email_id: str, role: Optional[str]) -> Optional[str]:
        if not role or str(role).lower() in ("null", "none", "n/a", "unknown", ""):
            return None
        role = role.strip()
        # Capitalize properly
        if role == role.upper() and len(role) > 5:
            role = role.title()
        return role
