from dataclasses import dataclass, asdict
from typing import Optional
import datetime


@dataclass
class EmailRecord:
    id: str  # Gmail Message ID
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
    """LLM-based extraction result with full schema."""
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
    model_used: str = "mistral-7b-instruct-v0.3"

    def __post_init__(self):
        if not self.extracted_at:
            self.extracted_at = datetime.datetime.now().isoformat()

    def to_dict(self):
        return asdict(self)
