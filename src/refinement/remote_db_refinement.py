"""
Remote DB Refinement — Company Normalization & Drive Deduplication.

FULLY STANDALONE — no relative imports, no package dependencies.
Run directly: python remote_db_refinement.py

Steps:
  1. Filter structured_hiring_data to emails after June 1, 2025
  2. Batch distinct company names -> LLM semantic normalization
  3. Store mapping in company_normalization_map table (NEW)
  4. Group by (canonical_company, role) -> merge -> refined_drives table (NEW)

Does NOT modify existing tables.
"""

import sqlite3
import json
import re
import os
import sys
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ── LLM Configuration ──────────────────────────────────────

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 10
MAX_RETRIES = 2

NORMALIZATION_PROMPT = """<s>[INST] You are given a list of company names extracted from university placement officer emails.
Some may refer to the same company but written differently (e.g. abbreviations, suffixes like Pvt Ltd, typos, accent marks).

Return a strict JSON mapping where:
- Keys = original names exactly as given
- Values = canonical (standardized) company name

Rules:
- Only merge if clearly the same organization.
- Remove suffixes like "Pvt Ltd", "Private Limited", "Inc", "LLC", "LLP" from canonical names.
- Fix obvious typos and accent inconsistencies.
- "Unknown" should remain "Unknown".
- Do NOT hallucinate or invent new companies.
- Do NOT merge companies that are merely similar but different (e.g., "Tata Motors" and "Tata Technologies" are different).

Company names:
{companies}

Output ONLY valid JSON, nothing else:
[/INST]"""


# ── Database Helpers ─────────────────────────────────────────

def find_db_path():
    """Locate campus_hiring.db relative to CWD."""
    candidates = [
        os.path.join(os.getcwd(), "data", "campus_hiring.db"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "campus_hiring.db"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "campus_hiring.db"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "campus_hiring.db"),
    ]
    for path in candidates:
        normalized = os.path.normpath(path)
        if os.path.exists(normalized):
            return normalized
    raise FileNotFoundError(
        f"Cannot find campus_hiring.db. Tried: {[os.path.normpath(c) for c in candidates]}"
    )


def create_tables(conn):
    """Create new tables (idempotent). Does NOT touch existing tables."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS company_normalization_map (
            original_name TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS refined_drives (
            drive_id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_company_name TEXT,
            role TEXT,
            ctc_lpa REAL,
            cgpa_cutoff REAL,
            eligibility_branches TEXT,
            registration_deadline TEXT,
            test_date TEXT,
            interview_date TEXT,
            selection_count INTEGER,
            source_email_ids TEXT
        )
    """)

    conn.commit()
    logger.info("Tables ready: company_normalization_map, refined_drives")


def get_filtered_data(conn):
    """Get structured data joined with emails, filtered to >= June 1, 2025.
    
    Handles multiple date formats: RFC 2822 (e.g., 'Fri, 11 Jul 2025 10:28:54 +0530')
    and ISO format.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.email_id, s.company_name, s.role, s.ctc_lpa, s.cgpa_cutoff,
               s.eligibility_branches, s.registration_deadline, s.test_date,
               s.interview_date, s.selection_count, s.total_openings,
               e.date as email_date, e.subject
        FROM structured_hiring_data s
        JOIN emails e ON s.email_id = e.id
    """)
    columns = [desc[0] for desc in cursor.description]
    all_rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    # Parse month names for filtering
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    filtered = []
    for row in all_rows:
        date_str = row.get("email_date", "") or ""
        keep = False

        # Try RFC 2822: "Fri, 11 Jul 2025 10:28:54 +0530"
        parts = date_str.split()
        if len(parts) >= 4:
            try:
                day = int(parts[1])
                month_str = parts[2]
                year = int(parts[3])
                month = month_map.get(month_str, 0)
                if year > 2025 or (year == 2025 and month >= 6):
                    keep = True
            except (ValueError, IndexError):
                pass

        # Try ISO: "2025-06-01..."
        if not keep and date_str[:10] >= "2025-06-01":
            try:
                if int(date_str[:4]) >= 2025:
                    keep = True
            except (ValueError, IndexError):
                pass

        if keep:
            filtered.append(row)

    return filtered


def get_distinct_companies(data):
    """Get sorted unique company names."""
    companies = set()
    for record in data:
        name = (record.get("company_name") or "").strip()
        if name:
            companies.add(name)
    return sorted(companies)


# ── LLM Normalization ───────────────────────────────────────

def load_llm():
    """Load Mistral-7B with 4-bit quantization."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    logger.info("Loading Mistral-7B (4-bit)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info("Model loaded successfully.")
    return tokenizer, model


def generate_response(tokenizer, model, prompt):
    """Generate LLM response deterministically."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.1,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_json_from_llm(raw):
    """Parse JSON from LLM output using multiple strategies."""
    # Strategy 1: Brace depth — find the outermost complete JSON object
    brace_depth = 0
    start_idx = None
    for i, ch in enumerate(raw):
        if ch == '{':
            if start_idx is None:
                start_idx = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start_idx is not None:
                try:
                    return json.loads(raw[start_idx:i+1])
                except json.JSONDecodeError:
                    break

    # Strategy 2: If JSON was truncated (incomplete braces), try to complete it
    if start_idx is not None:
        fragment = raw[start_idx:]
        # Try adding closing brace
        for suffix in ['}', '"\n}', '"}']:
            try:
                return json.loads(fragment + suffix)
            except json.JSONDecodeError:
                pass
        # Try extracting key-value pairs manually
        pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', fragment)
        if pairs:
            return {k: v for k, v in pairs}

    # Strategy 3: Regex for simple JSON
    match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: Extract key-value pairs from non-JSON text
    pairs = re.findall(r'"([^"]+)"\s*[:=→]\s*"([^"]+)"', raw)
    if pairs:
        return {k: v for k, v in pairs}

    # Strategy 5: Direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    return None


def normalize_companies_with_llm(tokenizer, model, companies):
    """Batch company names and use LLM to create canonical mapping."""
    full_mapping = {}

    batches = [companies[i:i+BATCH_SIZE] for i in range(0, len(companies), BATCH_SIZE)]
    logger.info(f"Processing {len(companies)} companies in {len(batches)} batches...")

    for batch_idx, batch in enumerate(batches):
        companies_str = "\n".join(f"- {c}" for c in batch)
        prompt = NORMALIZATION_PROMPT.format(companies=companies_str)

        for attempt in range(MAX_RETRIES):
            raw_output = generate_response(tokenizer, model, prompt)
            logger.info(f"  Batch {batch_idx+1} raw output ({len(raw_output)} chars):")
            # Log first 500 chars of raw output for debugging
            for line in raw_output[:500].split('\n'):
                logger.info(f"    > {line}")
            if len(raw_output) > 500:
                logger.info(f"    > ... ({len(raw_output)-500} more chars)")
            parsed = parse_json_from_llm(raw_output)

            if parsed and isinstance(parsed, dict):
                for key in batch:
                    if key in parsed:
                        full_mapping[key] = parsed[key]
                    else:
                        full_mapping[key] = key  # identity fallback
                mapped_count = sum(1 for k in batch if k in parsed)
                logger.info(f"  Batch {batch_idx+1}/{len(batches)}: {mapped_count}/{len(batch)} mapped")
                break
            else:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"  Batch {batch_idx+1}: JSON parse failed, retrying...")
                else:
                    logger.warning(f"  Batch {batch_idx+1}: Failed after retries, identity mapping")
                    for c in batch:
                        full_mapping[c] = c

    # Safety: ensure every company has a mapping
    for c in companies:
        if c not in full_mapping:
            full_mapping[c] = c

    return full_mapping


def save_normalization_map(conn, mapping):
    """Save company normalization mapping to DB."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM company_normalization_map")
    for original, canonical in mapping.items():
        cursor.execute(
            "INSERT OR REPLACE INTO company_normalization_map (original_name, canonical_name) VALUES (?, ?)",
            (original, canonical)
        )
    conn.commit()
    logger.info(f"Saved {len(mapping)} normalization mappings to DB")


# ── Drive Deduplication ──────────────────────────────────────

def _earliest(a, b):
    """Return the earliest non-null date string."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _union_branches(a, b):
    """Union unique branches from two comma-separated strings."""
    parts = set()
    for val in [a, b]:
        if val:
            for branch in val.split(","):
                branch = branch.strip()
                if branch:
                    parts.add(branch)
    return ", ".join(sorted(parts)) if parts else None


def deduplicate_drives(data, mapping):
    """Group by (canonical_company, role) and merge with specified rules."""
    groups = defaultdict(list)

    for record in data:
        original_company = (record.get("company_name") or "Unknown").strip()
        canonical = mapping.get(original_company, original_company)
        role = (record.get("role") or "").strip() or None

        key = (canonical.lower(), (role or "").lower() or None)
        record["_canonical"] = canonical
        groups[key].append(record)

    drives = []
    for (norm_company, norm_role), records in groups.items():
        canonical_name = records[0]["_canonical"]

        merged = {
            "canonical_company_name": canonical_name,
            "role": records[0].get("role"),
            "ctc_lpa": None,
            "cgpa_cutoff": None,
            "eligibility_branches": None,
            "registration_deadline": None,
            "test_date": None,
            "interview_date": None,
            "selection_count": None,
            "source_email_ids": ",".join(r["email_id"] for r in records),
        }

        for r in records:
            # ctc_lpa: maximum (but <=100)
            ctc = r.get("ctc_lpa")
            if ctc is not None and ctc <= 100:
                if merged["ctc_lpa"] is None or ctc > merged["ctc_lpa"]:
                    merged["ctc_lpa"] = ctc

            # cgpa_cutoff: minimum
            cgpa = r.get("cgpa_cutoff")
            if cgpa is not None:
                if merged["cgpa_cutoff"] is None or cgpa < merged["cgpa_cutoff"]:
                    merged["cgpa_cutoff"] = cgpa

            # selection_count: prefer non-null
            sel = r.get("selection_count")
            if sel is not None and merged["selection_count"] is None:
                merged["selection_count"] = sel

            # eligibility_branches: union
            merged["eligibility_branches"] = _union_branches(
                merged["eligibility_branches"], r.get("eligibility_branches")
            )

            # Dates: earliest non-null
            merged["registration_deadline"] = _earliest(
                merged["registration_deadline"], r.get("registration_deadline")
            )
            merged["test_date"] = _earliest(merged["test_date"], r.get("test_date"))
            merged["interview_date"] = _earliest(
                merged["interview_date"], r.get("interview_date")
            )

        drives.append(merged)

    return drives


def save_refined_drives(conn, drives):
    """Save deduplicated drives to refined_drives table."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM refined_drives")

    for drive in drives:
        cursor.execute("""
            INSERT INTO refined_drives (
                canonical_company_name, role, ctc_lpa, cgpa_cutoff,
                eligibility_branches, registration_deadline, test_date,
                interview_date, selection_count, source_email_ids
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            drive["canonical_company_name"],
            drive["role"],
            drive["ctc_lpa"],
            drive["cgpa_cutoff"],
            drive["eligibility_branches"],
            drive["registration_deadline"],
            drive["test_date"],
            drive["interview_date"],
            drive["selection_count"],
            drive["source_email_ids"],
        ))

    conn.commit()
    logger.info(f"Saved {len(drives)} refined drives to DB")


# ── Main Pipeline ────────────────────────────────────────────

def main():
    print()
    print("=" * 65)
    print("  CAMPUS HIRING — COMPANY NORMALIZATION & DRIVE DEDUPLICATION")
    print("=" * 65)

    # Step 1: Locate DB
    db_path = find_db_path()
    logger.info(f"Database: {db_path}")
    conn = sqlite3.connect(db_path)

    # Step 2: Create new tables
    create_tables(conn)

    # Step 3: Filter data
    print()
    print("-" * 50)
    print("  STEP 1: Filtering records (>= June 1, 2025)")
    print("-" * 50)

    data = get_filtered_data(conn)
    logger.info(f"Filtered records: {len(data)}")

    if not data:
        logger.error("No data found after filtering. Exiting.")
        conn.close()
        return

    # Step 4: Get companies
    companies = get_distinct_companies(data)
    logger.info(f"Distinct companies BEFORE normalization: {len(companies)}")
    for c in companies:
        print(f"    - {c}")

    # Step 5: LLM normalization
    print()
    print("-" * 50)
    print("  STEP 2: LLM Company Name Normalization")
    print("-" * 50)

    import torch
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer, model = load_llm()
    mapping = normalize_companies_with_llm(tokenizer, model, companies)

    # Unload model
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded, CUDA cache cleared.")

    # Count canonical names
    canonical_names = set(mapping.values())
    logger.info(f"Distinct companies AFTER normalization: {len(canonical_names)}")
    logger.info(f"Companies merged: {len(companies) - len(canonical_names)}")

    # Print merges
    merges = defaultdict(list)
    for orig, canon in mapping.items():
        merges[canon].append(orig)
    for canon, originals in sorted(merges.items()):
        if len(originals) > 1:
            logger.info(f"  MERGED: {originals} -> '{canon}'")

    # Save mapping
    save_normalization_map(conn, mapping)

    # Step 6: Deduplicate
    print()
    print("-" * 50)
    print("  STEP 3: Drive Deduplication")
    print("-" * 50)

    drives_before = len(set(
        ((r.get("company_name") or "").lower(), (r.get("role") or "").lower())
        for r in data
    ))
    logger.info(f"Drives BEFORE deduplication: {drives_before}")

    refined = deduplicate_drives(data, mapping)
    logger.info(f"Drives AFTER deduplication: {len(refined)}")
    logger.info(f"Drives reduced by: {drives_before - len(refined)}")

    save_refined_drives(conn, refined)

    # Step 7: Summary
    print()
    print("=" * 65)
    print("  REFINEMENT COMPLETE")
    print("=" * 65)
    print(f"  Records processed:    {len(data)}")
    print(f"  Companies:            {len(companies)} -> {len(canonical_names)} ({len(companies) - len(canonical_names)} merged)")
    print(f"  Drives:               {drives_before} -> {len(refined)} ({drives_before - len(refined)} reduced)")
    print(f"  Tables created:       company_normalization_map ({len(mapping)} rows)")
    print(f"                        refined_drives ({len(refined)} rows)")
    print()
    print("  Original tables are UNTOUCHED.")

    conn.close()


if __name__ == "__main__":
    main()
