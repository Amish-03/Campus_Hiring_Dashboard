"""
Campus Hiring NLP Pipeline — Main Orchestrator.

Runs a 6-stage pipeline:
  1. Fetch emails from Gmail
  2. Classify hiring vs non-hiring
  3. LLM extraction (structured JSON)
  4. Validation (sanity checks + audit)
  5. Deduplication (merge into drives)
  6. Summary report

Usage:
    python -m src.main                       # Full pipeline
    python -m src.main --extract-only        # Re-extract + deduplicate (no fetch)
    python -m src.main --no-fetch            # Classify + extract (no fetch)
    python -m src.main --force               # Clear and re-extract
    python -m src.main --sample-eval         # Generate evaluation template
    python -m src.main --evaluate            # Run evaluation against ground truth
    streamlit run src/dashboard/dashboard.py # Launch dashboard
"""

import sys
import os
import argparse
import logging
from tqdm import tqdm

from .ingestion.gmail_api import GmailFetcher
from .storage.db import DatabaseManager
from .classifier.rule_classifier import HiringClassifier
from .extraction.llm_extractor import LLMExtractor
from .validation.validator import Validator
from .deduplication.deduplicator import Deduplicator
from .evaluation.evaluator import Evaluator
from .models import EmailRecord

DB_PATH = "data/campus_hiring.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("data/pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def step1_fetch(db: DatabaseManager) -> int:
    """Step 1: Fetch emails from Gmail API."""
    print("\n" + "=" * 60)
    print("  STEP 1: Fetching Emails from Gmail")
    print("=" * 60)

    fetcher = GmailFetcher()
    query = "from:placement_officer@kletech.ac.in after:2025/03/01"
    print(f"  Query: {query}")

    emails = fetcher.fetch_emails(query=query)
    print(f"  Fetched {len(emails)} emails.")

    for email in emails:
        db.save_email(email)

    print(f"  Saved to database.")
    return len(emails)


def step2_classify(db: DatabaseManager) -> list:
    """Step 2: Classify emails as hiring / non-hiring."""
    print("\n" + "=" * 60)
    print("  STEP 2: Classifying Emails")
    print("=" * 60)

    classifier = HiringClassifier()
    all_emails = db.get_all_emails()
    hiring = []
    non_hiring = 0

    for email in all_emails:
        if classifier.is_hiring_email(email.sender, email.subject, email.body):
            hiring.append(email)
        else:
            non_hiring += 1

    print(f"  Total:       {len(all_emails)}")
    print(f"  Hiring:      {len(hiring)}")
    print(f"  Non-hiring:  {non_hiring}")
    return hiring


def step3_extract(db: DatabaseManager, hiring_emails: list, force: bool = False) -> int:
    """Step 3: LLM-based structured extraction."""
    print("\n" + "=" * 60)
    print("  STEP 3: LLM Extraction (Mistral-7B-Instruct)")
    print("=" * 60)

    if force:
        db.clear_structured_data()
        db.clear_audit_log()
        print("  Cleared previous extraction data.")

    already = db.get_extracted_email_ids()
    pending = [e for e in hiring_emails if e.id not in already]

    if not pending:
        print("  All emails already extracted. Use --force to re-extract.")
        return 0

    print(f"  Pending:     {len(pending)}")
    print(f"  Already done: {len(already)}")

    extractor = LLMExtractor()
    extractor.load_model()

    validator = Validator(db=db)

    success = 0
    failed = 0

    for email in tqdm(pending, desc="  Extracting", unit="email"):
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
            tqdm.write(f"  [ERROR] {safe_msg[:80]}")
            failed += 1

    extractor.unload_model()

    stats = extractor.stats
    print(f"\n  Extraction complete.")
    print(f"  Success:  {success}")
    print(f"  Failed:   {failed}")
    print(f"  Retries:  {stats.get('retries', 0)}")
    return success


def step4_deduplicate(db: DatabaseManager) -> int:
    """Step 4: Deduplicate into drives."""
    print("\n" + "=" * 60)
    print("  STEP 4: Deduplication")
    print("=" * 60)

    deduplicator = Deduplicator(db)
    return deduplicator.run()


def step5_summary(db: DatabaseManager):
    """Step 5: Quick analytics summary."""
    print("\n" + "=" * 60)
    print("  STEP 5: Analytics Summary")
    print("=" * 60)

    import pandas as pd

    drives = db.get_drives()
    if not drives:
        print("  No drive data available.")
        return

    df = pd.DataFrame(drives)
    ctc = df["ctc_lpa"].dropna()
    cgpa = df["cgpa_cutoff"].dropna()
    sel = df["selection_count"].dropna()

    print(f"  Total drives:        {len(df)}")
    print(f"  Unique companies:    {df['company_name'].nunique()}")
    if not ctc.empty:
        print(f"  Avg CTC:             {ctc.mean():.1f} LPA")
        print(f"  Median CTC:          {ctc.median():.1f} LPA")
        print(f"  Highest CTC:         {ctc.max():.1f} LPA")
    if not cgpa.empty:
        print(f"  Avg CGPA cutoff:     {cgpa.mean():.1f}")
    if not sel.empty:
        print(f"  Total selections:    {int(sel.sum())}")

    audit = db.get_audit_log()
    print(f"  Audit corrections:   {len(audit)}")

    print(f"\n  Launch dashboard:")
    print(f"    streamlit run src/dashboard/dashboard.py")


def main():
    parser = argparse.ArgumentParser(description="Campus Hiring NLP Pipeline")
    parser.add_argument("--extract-only", action="store_true",
                        help="Re-run LLM extraction + deduplication (no Gmail fetch)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Classify + extract from existing DB")
    parser.add_argument("--force", action="store_true",
                        help="Clear previous extractions and re-do")
    parser.add_argument("--sample-eval", action="store_true",
                        help="Generate ground truth template for evaluation")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation against annotated ground truth")
    args = parser.parse_args()

    db = DatabaseManager(db_path=DB_PATH)

    # Evaluation modes
    if args.sample_eval:
        evaluator = Evaluator(db)
        evaluator.sample_emails(n=20)
        return

    if args.evaluate:
        evaluator = Evaluator(db)
        results = evaluator.evaluate()
        evaluator.print_report(results)
        return

    # Credential check for fetch mode
    if not args.extract_only and not args.no_fetch:
        if not os.path.exists("credentials.json") and not os.path.exists("token.json"):
            print("Error: 'credentials.json' not found.")
            print("Get it from https://console.cloud.google.com/")
            sys.exit(1)

    # Step 1: Fetch
    if not args.extract_only and not args.no_fetch:
        step1_fetch(db)

    # Step 2: Classify
    hiring_emails = step2_classify(db)

    # Step 3: Extract
    step3_extract(db, hiring_emails, force=args.force)

    # Step 4: Deduplicate
    step4_deduplicate(db)

    # Step 5: Summary
    step5_summary(db)


if __name__ == "__main__":
    main()
