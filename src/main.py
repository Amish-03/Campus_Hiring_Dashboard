"""
Campus Hiring NLP Pipeline — Main Orchestrator.

Usage:
    python -m src.main                  # Full pipeline: fetch + classify + extract
    python -m src.main --extract-only   # Re-run LLM extraction on existing emails
    python -m src.main --no-fetch       # Classify + extract without fetching
    streamlit run src/dashboard/dashboard.py  # Launch analytics dashboard
"""

import sys
import os
import argparse
from tqdm import tqdm

from .ingestion.gmail_api import GmailFetcher
from .storage.db import DatabaseManager
from .classifier.rule_classifier import HiringClassifier
from .extraction.llm_extractor import LLMExtractor
from .validation.validator import Validator
from .models import EmailRecord

DB_PATH = "data/campus_hiring.db"


def fetch_emails(db: DatabaseManager):
    """Step 1: Fetch emails from Gmail API."""
    print("\n" + "=" * 60)
    print("  STEP 1: Fetching Emails from Gmail")
    print("=" * 60)

    fetcher = GmailFetcher()
    query = "from:placement_officer@kletech.ac.in after:2025/03/01"
    print(f"  Query: {query}")

    emails = fetcher.fetch_emails(query=query)
    print(f"\n  Fetched {len(emails)} emails.")

    for email in emails:
        db.save_email(email)

    print(f"  All emails saved to database.")
    return len(emails)


def classify_emails(db: DatabaseManager) -> list:
    """Step 2: Classify emails as hiring / non-hiring."""
    print("\n" + "=" * 60)
    print("  STEP 2: Classifying Emails")
    print("=" * 60)

    classifier = HiringClassifier()
    all_emails = db.get_all_emails()
    hiring_emails = []
    non_hiring_count = 0

    for email in all_emails:
        if classifier.is_hiring_email(email.sender, email.subject, email.body):
            hiring_emails.append(email)
        else:
            non_hiring_count += 1

    print(f"  Total emails:      {len(all_emails)}")
    print(f"  Hiring emails:     {len(hiring_emails)}")
    print(f"  Non-hiring emails: {non_hiring_count}")

    return hiring_emails


def extract_with_llm(db: DatabaseManager, hiring_emails: list, force: bool = False):
    """Step 3: Run LLM extraction on hiring emails."""
    print("\n" + "=" * 60)
    print("  STEP 3: LLM-Based Structured Extraction")
    print("=" * 60)

    if force:
        db.clear_structured_data()
        print("  Cleared existing structured data (force mode).")

    # Check which emails already have extractions (resumability)
    already_extracted = db.get_extracted_email_ids()
    pending_emails = [e for e in hiring_emails if e.id not in already_extracted]

    if not pending_emails:
        print("  All hiring emails already extracted. Use --force to re-extract.")
        return

    print(f"  Emails to extract: {len(pending_emails)}")
    print(f"  Already extracted:  {len(already_extracted)}")

    # Load model
    extractor = LLMExtractor()
    extractor.load_model()

    validator = Validator()

    success = 0
    failed = 0

    for email in tqdm(pending_emails, desc="  Extracting", unit="email"):
        try:
            result = extractor.extract(email.id, email.subject, email.body)
            if result:
                result = validator.validate(result)
                db.save_structured_data(result)
                success += 1
            else:
                failed += 1
        except Exception as e:
            safe_msg = str(e).encode('ascii', errors='replace').decode('ascii')
            tqdm.write(f"  [ERROR] {email.subject[:40]}: {safe_msg}")
            failed += 1

    # Free GPU memory
    extractor.unload_model()

    print(f"\n  Extraction complete.")
    print(f"  Successful: {success}")
    print(f"  Failed:     {failed}")


def print_summary(db: DatabaseManager):
    """Step 4: Print a quick summary of extracted data."""
    print("\n" + "=" * 60)
    print("  STEP 4: Quick Analytics Summary")
    print("=" * 60)

    data = db.get_structured_data()
    if not data:
        print("  No structured data available.")
        return

    import pandas as pd
    df = pd.DataFrame(data)

    valid_ctc = df["ctc_lpa"].dropna()
    valid_cgpa = df["cgpa_cutoff"].dropna()
    valid_sel = df["selection_count"].dropna()

    unique_companies = df["company_name"].nunique()

    print(f"  Total records:      {len(df)}")
    print(f"  Unique companies:   {unique_companies}")
    print(f"  Avg CTC:            {valid_ctc.mean():.1f} LPA" if not valid_ctc.empty else "  Avg CTC: N/A")
    print(f"  Highest CTC:        {valid_ctc.max():.1f} LPA" if not valid_ctc.empty else "  Highest CTC: N/A")
    print(f"  Avg CGPA cutoff:    {valid_cgpa.mean():.1f}" if not valid_cgpa.empty else "  Avg CGPA: N/A")
    print(f"  Total selections:   {int(valid_sel.sum())}" if not valid_sel.empty else "  Total selections: N/A")

    print(f"\n  Launch dashboard:")
    print(f"    streamlit run src/dashboard/dashboard.py")


def main():
    parser = argparse.ArgumentParser(description="Campus Hiring NLP Pipeline")
    parser.add_argument("--extract-only", action="store_true",
                        help="Re-run LLM extraction on existing classified emails (no fetch)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip Gmail fetch, classify + extract from existing DB")
    parser.add_argument("--force", action="store_true",
                        help="Force re-extraction (clear previous results)")
    args = parser.parse_args()

    # Credential check
    if not args.extract_only and not args.no_fetch:
        if not os.path.exists("credentials.json") and not os.path.exists("token.json"):
            print("Error: 'credentials.json' not found.")
            print("1. Go to https://console.cloud.google.com/")
            print("2. Create a project and enable Gmail API.")
            print("3. Create OAuth 2.0 Client ID (Desktop App) and download as 'credentials.json'.")
            sys.exit(1)

    db = DatabaseManager(db_path=DB_PATH)

    # Step 1: Fetch (unless skipped)
    if not args.extract_only and not args.no_fetch:
        fetch_emails(db)

    # Step 2: Classify
    hiring_emails = classify_emails(db)

    # Step 3: LLM Extract
    extract_with_llm(db, hiring_emails, force=args.force)

    # Step 4: Summary
    print_summary(db)


if __name__ == "__main__":
    main()
