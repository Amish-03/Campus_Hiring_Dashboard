from dataclasses import dataclass, asdict, field
from typing import Optional, List
import datetime


@dataclass
class EmailRecord:
    """Raw email from Gmail API."""
    id: str
    subject: str
    sender: str
    date: str
    body: str
    raw_body: str

    def to_dict(self):
        return asdict(self)


@dataclass
class HiringDetail:
    """Legacy regex-based extraction result (kept for backward compatibility)."""
    email_id: str
    company: str
    role: str
    ctc: Optional[float] = None
    cgpa_cutoff: Optional[float] = None
    registration_deadline: Optional[str] = None
    test_date: Optional[str] = None
    interview_date: Optional[str] = None
    extracted_at: str = ""

    def __post_init__(self):
        if not self.extracted_at:
            self.extracted_at = datetime.datetime.now().isoformat()

    def to_dict(self):
        return asdict(self)


@dataclass
class StructuredHiringData:
    """LLM-based extraction result per email."""
    email_id: str
    company_name: str
    role: Optional[str] = None
    ctc_lpa: Optional[float] = None
    cgpa_cutoff: Optional[float] = None
    eligibility_branches: Optional[str] = None
    registration_deadline: Optional[str] = None
    test_date: Optional[str] = None
    interview_date: Optional[str] = None
    selection_count: Optional[int] = None
    total_openings: Optional[int] = None
    extracted_at: str = ""
    model_used: str = "mistralai/Mistral-7B-Instruct-v0.3"

    def __post_init__(self):
        if not self.extracted_at:
            self.extracted_at = datetime.datetime.now().isoformat()

    def to_dict(self):
        return asdict(self)


@dataclass
class DriveRecord:
    """Deduplicated campus drive — one per (company, role)."""
    drive_id: Optional[int] = None
    company_name: str = ""
    role: Optional[str] = None
    ctc_lpa: Optional[float] = None
    cgpa_cutoff: Optional[float] = None
    eligibility_branches: Optional[str] = None
    registration_deadline: Optional[str] = None
    test_date: Optional[str] = None
    interview_date: Optional[str] = None
    selection_count: Optional[int] = None
    total_openings: Optional[int] = None
    email_count: int = 1
    first_seen: str = ""
    last_updated: str = ""

    def __post_init__(self):
        now = datetime.datetime.now().isoformat()
        if not self.first_seen:
            self.first_seen = now
        if not self.last_updated:
            self.last_updated = now

    def to_dict(self):
        return asdict(self)


@dataclass
class AuditEntry:
    """Record of a validation correction."""
    email_id: str
    field_name: str
    original: str
    corrected: str
    rule: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()
