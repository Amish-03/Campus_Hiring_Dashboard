# Campus Hiring NLP Analytics System (v2)

> Production-quality LLM-powered semantic extraction from university placement emails, with deduplication, validation audit trail, and an interactive Streamlit dashboard.

## Current State

- **497 emails** fetched from `placement_officer@kletech.ac.in` and stored in `data/campus_hiring.db`
- **~203 emails** classified as hiring-related
- **LLM extraction not yet run** — this is the next step on the CUDA device

## Architecture

```
Gmail API → SQLite (emails)
         → Rule Classifier
         → Mistral-7B LLM (4-bit, CUDA) → JSON retry (3 attempts)
         → Validator (audit trail)
         → SQLite (structured_hiring_data)
         → Deduplicator (group by company+role)
         → SQLite (drives)
         → Streamlit Dashboard
```

## Setup on CUDA Device

### Prerequisites
- Python 3.10+
- NVIDIA GPU ≥ 8 GB VRAM (12 GB recommended)
- CUDA toolkit

### Step 1: Clone & Install

```bash
git clone https://github.com/Amish-03/Campus_Hiring_Dashboard.git
cd Campus_Hiring_Dashboard
pip install -r requirements.txt
```

### Step 2: Run Full Pipeline

The 497 emails are already in the database. No Gmail credentials needed:

```bash
python -m src.main --extract-only
```

This runs **4 stages** automatically:
1. **Classify** → identifies ~203 hiring emails
2. **Extract** → Mistral-7B with 4-bit quantization, JSON retry, ~10 min
3. **Validate** → CTC cap, CGPA range, date normalization, branch standardization
4. **Deduplicate** → merges multiple emails per company into one drive record

### Step 3: Launch Dashboard

```bash
streamlit run src/dashboard/dashboard.py
```

## CLI Reference

| Command | What it does |
|:---|:---|
| `python -m src.main` | Full pipeline (fetch + classify + extract + deduplicate) |
| `python -m src.main --extract-only` | **Recommended** — extract from existing DB |
| `python -m src.main --extract-only --force` | Re-extract (clears previous results) |
| `python -m src.main --no-fetch` | Skip Gmail fetch |
| `python -m src.main --sample-eval` | Generate ground truth template (20 emails) |
| `python -m src.main --evaluate` | Run evaluation against annotated ground truth |
| `streamlit run src/dashboard/dashboard.py` | Launch dashboard |

## Project Structure

```
campus_hiring_nlp/
├── data/
│   └── campus_hiring.db              # 497 emails + 6 tables
├── src/
│   ├── main.py                        # 6-stage pipeline orchestrator
│   ├── models.py                      # 5 dataclasses (Email, Hiring, Drive, Audit)
│   ├── ingestion/gmail_api.py         # Gmail API fetcher
│   ├── classifier/rule_classifier.py  # Weighted keyword classifier
│   ├── extraction/llm_extractor.py    # Mistral-7B + prompt + retry
│   ├── validation/validator.py        # Sanity checks + audit logging
│   ├── deduplication/deduplicator.py  # company+role grouping + merge
│   ├── analytics/metrics.py           # Metrics computation
│   ├── dashboard/dashboard.py         # Streamlit (7 charts, CSV export)
│   └── evaluation/evaluator.py        # Precision/recall framework
├── requirements.txt
└── .gitignore
```

## Database Schema (6 Tables)

| Table | Purpose |
|:---|:---|
| `emails` | Raw emails (497 rows) |
| `hiring_details` | Legacy regex extraction |
| `structured_hiring_data` | LLM extraction (per email, UNIQUE email_id) |
| `drives` | Deduplicated drives (UNIQUE company_name+role) |
| `drive_email_mapping` | N:N drive ↔ email relationship |
| `data_audit` | Validation corrections log |

## LLM Details

- **Model**: `mistralai/Mistral-7B-Instruct-v0.3` (4-bit NF4)
- **VRAM**: ~4 GB
- **Retry**: 3 attempts with JSON hint on failure
- **Prompt**: Placement-officer-aware (subject-line quotes, date-context alignment)
- **Deterministic**: `temperature=0.1`, `do_sample=False`, `repetition_penalty=1.1`

## Dashboard Features

| Section | Details |
|:---|:---|
| **Metrics** | Total drives, unique companies, avg/median/max CTC, avg CGPA, total offers |
| **Drive Table** | One row per drive, searchable, sortable |
| **Charts** | Hiring by month, CTC histogram, CGPA histogram, Top 10 CTC, Role pie, Branch bar, CGPA vs CTC scatter |
| **Export** | CSV download button |
| **Audit** | Expandable validation audit log |

## Evaluation Framework

1. `--sample-eval` → generates `ground_truth.json` with 20 random emails
2. Manually annotate the `ground_truth` fields
3. `--evaluate` → computes field-level precision, recall, hallucination rate
