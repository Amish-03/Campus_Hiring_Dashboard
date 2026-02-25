"""
Deduplication module for campus hiring drives.

Groups multiple emails about the same company drive into a single
consolidated DriveRecord, merging fields from later emails.
"""

import datetime
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from ..models import DriveRecord
from ..storage.db import DatabaseManager

logger = logging.getLogger(__name__)


def _normalize_key(company_name: str, role: Optional[str]) -> Tuple[str, Optional[str]]:
    """Create a normalized deduplication key."""
    company = (company_name or "").strip().lower()
    # Remove common suffixes
    for suffix in [" pvt ltd", " pvt. ltd.", " private limited", " limited", " ltd", " ltd.", " inc", " inc.", " llp"]:
        if company.endswith(suffix):
            company = company[:-len(suffix)].strip()
    normalized_role = (role or "").strip().lower() or None
    return company, normalized_role


def _merge_field(existing, new):
    """Keep existing if non-null, otherwise take new."""
    return existing if existing is not None else new


class Deduplicator:
    """Groups extracted hiring data into unique drives."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def run(self) -> int:
        """
        Deduplicate structured_hiring_data into drives.
        Returns number of drives created/updated.
        """
        print("\n  Running deduplication...")

        # Clear old drives
        self.db.clear_drives()

        # Fetch all structured data with email dates
        all_data = self.db.get_structured_data()
        if not all_data:
            print("  No structured data to deduplicate.")
            return 0

        # Group by normalized (company, role) key
        groups: Dict[Tuple, List[dict]] = defaultdict(list)
        for record in all_data:
            company = record.get("company_name", "Unknown")
            role = record.get("role")
            key = _normalize_key(company, role)
            groups[key].append(record)

        drive_count = 0
        for (norm_company, norm_role), records in groups.items():
            # Sort by email date (earliest first)
            records.sort(key=lambda r: r.get("email_date", "") or "")

            primary = records[0]  # earliest email is the primary

            # Start with primary record, merge from later emails
            merged = DriveRecord(
                company_name=primary.get("company_name", "Unknown"),
                role=primary.get("role"),
                ctc_lpa=primary.get("ctc_lpa"),
                cgpa_cutoff=primary.get("cgpa_cutoff"),
                eligibility_branches=primary.get("eligibility_branches"),
                registration_deadline=primary.get("registration_deadline"),
                test_date=primary.get("test_date"),
                interview_date=primary.get("interview_date"),
                selection_count=primary.get("selection_count"),
                total_openings=primary.get("total_openings"),
                email_count=len(records),
                first_seen=primary.get("email_date", ""),
                last_updated=records[-1].get("email_date", ""),
            )

            # Merge fields from later emails (fill nulls)
            for later in records[1:]:
                merged.ctc_lpa = _merge_field(merged.ctc_lpa, later.get("ctc_lpa"))
                merged.cgpa_cutoff = _merge_field(merged.cgpa_cutoff, later.get("cgpa_cutoff"))
                merged.eligibility_branches = _merge_field(merged.eligibility_branches, later.get("eligibility_branches"))
                merged.registration_deadline = _merge_field(merged.registration_deadline, later.get("registration_deadline"))
                merged.test_date = _merge_field(merged.test_date, later.get("test_date"))
                merged.interview_date = _merge_field(merged.interview_date, later.get("interview_date"))
                merged.selection_count = _merge_field(merged.selection_count, later.get("selection_count"))
                merged.total_openings = _merge_field(merged.total_openings, later.get("total_openings"))

            # Save drive and map all emails
            drive_id = self.db.save_drive(merged)
            for record in records:
                self.db.add_drive_email_mapping(drive_id, record["email_id"])

            drive_count += 1

        print(f"  Deduplicated {len(all_data)} email extractions → {drive_count} unique drives")
        return drive_count
