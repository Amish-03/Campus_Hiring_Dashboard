"""
Post-processing validation for LLM-extracted hiring data.

Applies sanity checks, normalizes values, and standardizes date formats
to ensure data quality before storing in the database.
"""

import re
from datetime import datetime
from typing import Optional

from ..models import StructuredHiringData


class Validator:
    """Validates and cleans LLM-extracted hiring data."""

    MAX_CTC_LPA = 100.0
    MIN_CGPA = 0.0
    MAX_CGPA = 10.0

    def validate(self, data: StructuredHiringData) -> StructuredHiringData:
        """Apply all validation rules and return cleaned data."""
        data.company_name = self._clean_company_name(data.company_name)
        data.ctc_lpa = self._validate_ctc(data.ctc_lpa)
        data.cgpa_cutoff = self._validate_cgpa(data.cgpa_cutoff)
        data.registration_deadline = self._validate_date(data.registration_deadline)
        data.test_date = self._validate_date(data.test_date)
        data.interview_date = self._validate_date(data.interview_date)
        data.selection_count = self._validate_positive_int(data.selection_count)
        data.total_openings = self._validate_positive_int(data.total_openings)
        data.eligibility_branches = self._clean_branches(data.eligibility_branches)
        data.role = self._clean_role(data.role)
        return data

    def _validate_ctc(self, ctc: Optional[float]) -> Optional[float]:
        """Reject unrealistic CTC values."""
        if ctc is None:
            return None
        if ctc <= 0 or ctc > self.MAX_CTC_LPA:
            return None
        return round(ctc, 2)

    def _validate_cgpa(self, cgpa: Optional[float]) -> Optional[float]:
        """Normalize CGPA to valid range."""
        if cgpa is None:
            return None
        if cgpa < self.MIN_CGPA or cgpa > self.MAX_CGPA:
            return None
        return round(cgpa, 1)

    def _validate_date(self, date_str: Optional[str]) -> Optional[str]:
        """Standardize dates to ISO format (YYYY-MM-DD)."""
        if date_str is None or date_str == "null":
            return None

        # Try ISO format directly
        for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%B %d, %Y", "%d %B %Y"]:
            try:
                parsed = datetime.strptime(date_str.strip(), fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return None

    def _validate_positive_int(self, value: Optional[int]) -> Optional[int]:
        """Ensure integer is positive."""
        if value is None:
            return None
        if value < 0:
            return None
        return value

    def _clean_company_name(self, name: Optional[str]) -> str:
        """Clean company name — remove extra quotes, whitespace."""
        if not name or name.lower() in ("unknown", "unknown company", "null", "none", "n/a"):
            return "Unknown"
        # Strip quotes and extra whitespace
        name = name.strip().strip('"').strip("'").strip()
        return name if name else "Unknown"

    def _clean_branches(self, branches: Optional[str]) -> Optional[str]:
        """Normalize branch names."""
        if not branches or branches.lower() in ("null", "none", "n/a"):
            return None
        # Standardize separators
        branches = branches.replace(";", ",").replace("/", ",")
        parts = [b.strip().upper() for b in branches.split(",") if b.strip()]
        return ", ".join(sorted(set(parts))) if parts else None

    def _clean_role(self, role: Optional[str]) -> Optional[str]:
        """Clean role string."""
        if not role or role.lower() in ("null", "none", "n/a", "unknown"):
            return None
        return role.strip()
