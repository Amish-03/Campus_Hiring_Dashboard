import re
from typing import Optional, List
from ..models import HiringDetail

class InfoExtractor:
    def __init__(self):
        # Regex patterns
        self.ctc_patterns = [
            r"(\d+(\.\d+)?)\s*(LPA|Lacs|Lakhs|L)",
            r"CTC[:\s]*(\d+(\.\d+)?)",
            r"Package[:\s]*(\d+(\.\d+)?)",
        ]
        self.cgpa_patterns = [
            r"CGPA[:\s]*(\d+(\.\d+)?)",
            r"(\d+(\.\d+)?)\s*CGPA",
            r"cutoff[:\s]*(\d+(\.\d+)?)",
        ]
        self.date_keywords = {
            "registration": ["registration", "apply by", "deadline", "last date"],
            "test": ["test date", "assessment", "exam"],
            "interview": ["interview", "shortlisted candidates"]
        }

    def extract_field(self, text: str, patterns: List[str]) -> Optional[float]:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    def extract_company(self, subject: str, body: str) -> str:
        # Heuristic: Often in subject or first lines
        # "Hiring at [Company]" or "Invitation from [Company]"
        patterns = [
            r"Hiring at ([\w\s\&]+)",
            r"Invitation from ([\w\s\&]+)",
            r"Opportunities? at ([\w\s\&]+)",
            r"([\w\s\&]+) \| Campus Recruitment"
        ]
        for p in patterns:
            match = re.search(p, subject, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Second attempt: check first few lines of body
        first_lines = "\n".join(body.split("\n")[:5])
        for p in patterns:
            match = re.search(p, first_lines, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Unknown Company"

    def extract_dates(self, body: str) -> dict:
        dates = {"registration": None, "test": None, "interview": None}
        # Simple heuristic: look for dates near keywords
        # Date pattern for DD-MM-YYYY or DD/MM/YYYY or MMM DD
        date_pattern = r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w{3,9}\s\d{1,2}(st|nd|rd|th)?)"
        
        lines = body.split("\n")
        for line in lines:
            line_lower = line.lower()
            for category, keywords in self.date_keywords.items():
                if any(kw in line_lower for kw in keywords):
                    match = re.search(date_pattern, line, re.IGNORECASE)
                    if match and not dates[category]:
                        dates[category] = match.group(1)
        return dates

    def extract_all(self, email_id: str, subject: str, body: str) -> HiringDetail:
        company = self.extract_company(subject, body)
        ctc = self.extract_field(body, self.ctc_patterns)
        cgpa = self.extract_field(body, self.cgpa_patterns)
        dates = self.extract_dates(body)
        
        # Role heuristic: often in subject or near 'role'
        role_match = re.search(r"Role[:\s]*([\w\s]+)", body, re.IGNORECASE)
        role = role_match.group(1).strip() if role_match else "Software Engineer" # Default or placeholder

        return HiringDetail(
            email_id=email_id,
            company=company,
            role=role,
            ctc=ctc,
            cgpa_cutoff=cgpa,
            registration_deadline=dates["registration"],
            test_date=dates["test"],
            interview_date=dates["interview"]
        )
