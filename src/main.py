"""
Campus Hiring NLP Pipeline — Main Orchestrator.

Runs a 5-stage pipeline:
  1. Fetch emails from Gmail
  2. LLM classification + extraction (structured JSON)
  3. Validation (sanity checks + audit)
  4. Deduplication (merge into drives)
  5. Summary report

The LLM itself decides if an email is hiring-related (no hard-coded classifier).
ALL emails are sent through the LLM — it classifies and extracts in one pass.

Usage:
    python -m src.main                       # Full pipeline
    python -m src.main --extract-only        # Re-extract + deduplicate (no fetch)
    python -m src.main --no-fetch            # Extract from existing DB (no fetch)
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


def step2_extract(db: DatabaseManager, force: bool = False) -> int:
    """
    Step 2: LLM classification + extraction.
    Sends ALL emails to the LLM. The LLM decides if each is hiring-related.
    Only hiring emails get saved to structured_hiring_data.
    """
    print("\n" + "=" * 60)
    print("  STEP 2: LLM Classification + Extraction (Mistral-7B)")
    print("=" * 60)

    if force:
        db.clear_structured_data()
        db.clear_audit_log()
        print("  Cleared previous extraction data.")

    already = db.get_extracted_email_ids()
    all_emails = db.get_all_emails()
    pending = [e for e in all_emails if e.id not in already]

    if not pending:
        print("  All emails already processed. Use --force to re-extract.")
        return 0

    print(f"  Total emails:    {len(all_emails)}")
    print(f"  Already done:    {len(already)}")
    print(f"  Pending:         {len(pending)}")
    print(f"\n  The LLM will classify each email as hiring/non-hiring")
    print(f"  and extract structured data from hiring emails.\n")

    extractor = LLMExtractor()
    extractor.load_model()

    validator = Validator(db=db)

    hiring_count = 0
    non_hiring_count = 0
    failed = 0

    for email in tqdm(pending, desc="  Processing", unit="email"):
        try:
            result = extractor.extract(email.id, email.subject, email.body)
            if result:
                # LLM classified as hiring — validate and save
                result = validator.validate(result)
                db.save_structured_data(result)
                hiring_count += 1
            else:
                # LLM classified as non-hiring OR JSON parse failed
                non_hiring_count += 1
        except Exception as e:
            safe_msg = str(e).encode('ascii', errors='replace').decode('ascii')
            tqdm.write(f"  [ERROR] {safe_msg[:80]}")
            failed += 1

    extractor.unload_model()

    stats = extractor.stats
    print(f"\n  Processing complete.")
    print(f"  Hiring emails:     {hiring_count}")
    print(f"  Non-hiring:        {non_hiring_count}")
    print(f"  LLM-classified:    {stats.get('skipped_non_hiring', 0)} (non-hiring)")
    print(f"  JSON failures:     {failed}")
    print(f"  Retries used:      {stats.get('retries', 0)}")
    return hiring_count


def step3_deduplicate(db: DatabaseManager) -> int:
    """Step 3: Deduplicate into drives."""
    print("\n" + "=" * 60)
    print("  STEP 3: Deduplication")
    print("=" * 60)

    deduplicator = Deduplicator(db)
    return deduplicator.run()


def step4_summary(db: DatabaseManager):
    """Step 4: Quick analytics summary."""
    print("\n" + "=" * 60)
    print("  STEP 4: Analytics Summary")
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

    total_emails = db.get_email_count()
    extracted = len(db.get_extracted_email_ids())

    print(f"  Total emails:        {total_emails}")
    print(f"  Hiring emails:       {extracted}")
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
                        help="LLM extraction + deduplication only (no Gmail fetch)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Extract from existing DB (no Gmail fetch)")
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

    # Step 2: LLM classify + extract (ALL emails)
    step2_extract(db, force=args.force)

    # Step 3: Deduplicate
    step3_deduplicate(db)

    # Step 4: Summary
    step4_summary(db)


if __name__ == "__main__":
    main()
