from src.models import EmailRecord
from src.storage.db import DatabaseManager
from src.classifier.rule_classifier import HiringClassifier
from src.extractor.info_extractor import InfoExtractor
from src.analytics.metrics import AnalyticsEngine
import os

def run_mock_demo():
    print("--- Starting Mock Pipeline Demo ---")
    
    # Ensure fresh DB for demo
    if os.path.exists("data/campus_hiring.db"):
        os.remove("data/campus_hiring.db")
    
    db = DatabaseManager()
    classifier = HiringClassifier()
    extractor = InfoExtractor()
    analytics = AnalyticsEngine(db)

    # Sample Emails
    mock_emails = [
        EmailRecord(
            id="msg_001",
            subject="Hiring at Google | Campus Recruitment 2025",
            sender="placement_officer@kletech.ac.in",
            date="Wed, 26 Feb 2025 10:00:00",
            body="We are excited to announce our placement drive. Role: Software Engineer. CTC: 35 LPA. Eligibility: 7.0 CGPA and above. Deadline: 15-03-2025.",
            raw_body="{}"
        ),
        EmailRecord(
            id="msg_002",
            subject="Job Opportunity: SDE at Amazon",
            sender="placement_officer@kletech.ac.in",
            date="Tue, 25 Feb 2025 14:00:00",
            body="Apply for SDE Role. Package: 42 Lacs. Test date: 20/03/2025. Last date to register is March 10th.",
            raw_body="{}"
        ),
        EmailRecord(
            id="msg_003",
            subject="Lunch tomorrow?",
            sender="friend@example.com",
            date="Wed, 26 Feb 2025 11:00:00",
            body="Hey, want to grab lunch tomorrow at the cafeteria?",
            raw_body="{}"
        )
    ]

    print("\n1. Ingesting Mock Emails...")
    for email in mock_emails:
        db.save_email(email)
        print(f"   Saved: {email.subject}")

    print("\n2. Processing & Extracting...")
    all_emails = db.get_all_emails()
    for email in all_emails:
        if classifier.is_hiring_email(email.sender, email.subject, email.body):
            print(f"   [MATCH] {email.subject}")
            detail = extractor.extract_all(email.id, email.subject, email.body)
            db.save_hiring_detail(detail)
            print(f"     -> Extracted Company: {detail.company}")
            print(f"     -> Extracted CTC: {detail.ctc} LPA")
            print(f"     -> Extracted CGPA: {detail.cgpa_cutoff}")
        else:
            print(f"   [SKIP]  {email.subject}")

    print("\n3. Generating Analytics Report...")
    analytics.print_report()

if __name__ == "__main__":
    run_mock_demo()
