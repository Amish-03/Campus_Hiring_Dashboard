# Campus Hiring NLP Analytics System

> LLM-powered semantic extraction from university placement emails with a Streamlit analytics dashboard.

## Current State

- **497 emails** already fetched from `placement_officer@kletech.ac.in` (since March 2025) and stored in `data/campus_hiring.db`
- **203 emails** classified as hiring-related by the rule-based classifier
- **LLM extraction has NOT been run yet** — this is the next step

## Architecture

```
Gmail API → SQLite (emails) → Rule Classifier → Mistral-7B LLM → Validator → SQLite (structured) → Streamlit
```

## Setup on CUDA Device (Step-by-Step)

### Prerequisites
- Python 3.10+
- NVIDIA GPU with **≥8 GB VRAM** (12 GB confirmed available)
- CUDA toolkit installed

### Step 1: Clone & Install

```bash
git clone https://github.com/Amish-03/Campus_Hiring_Dashboard.git
cd Campus_Hiring_Dashboard
pip install -r requirements.txt
```

> **Note**: `bitsandbytes` requires a Linux environment for GPU support. If on Windows, use WSL2.

### Step 2: Run LLM Extraction

The 497 emails are already in `data/campus_hiring.db`. **No Gmail credentials needed** — just run extraction:

```bash
python -m src.main --extract-only
```

This will:
1. Load **Mistral-7B-Instruct-v0.3** (4-bit quantized, ~4 GB VRAM)
2. Classify all 497 emails → identify ~203 hiring emails
3. Run LLM extraction on each hiring email → structured JSON
4. Validate output (CTC cap at 100 LPA, CGPA 0–10, ISO dates)
5. Store results in `structured_hiring_data` table
6. Takes ~10 minutes on a mid-range GPU

### Step 3: Launch Dashboard

```bash
streamlit run src/dashboard/dashboard.py
```

Dashboard sections:
- **Summary Metrics**: Total companies, Avg CTC, Highest CTC, Avg CGPA, Total offers
- **Company Table**: Searchable/sortable with all extracted fields
- **Charts**: Hiring by month, CTC distribution, CGPA distribution, Top 10 CTC, Role distribution

## CLI Reference

| Command | What it does |
|:---|:---|
| `python -m src.main` | Full pipeline: fetch Gmail + classify + LLM extract |
| `python -m src.main --extract-only` | **Use this** — LLM extract from existing DB (no fetch) |
| `python -m src.main --extract-only --force` | Re-extract (clears previous LLM results) |
| `python -m src.main --no-fetch` | Classify + extract without re-fetching Gmail |
| `streamlit run src/dashboard/dashboard.py` | Launch analytics dashboard |

## Project Structure

```
Campus_Hiring_Dashboard/
├── data/campus_hiring.db              # 497 emails already fetched
├── src/
│   ├── main.py                        # Pipeline orchestrator (CLI flags)
│   ├── models.py                      # EmailRecord, StructuredHiringData
│   ├── ingestion/gmail_api.py         # Gmail API fetcher
│   ├── classifier/rule_classifier.py  # Weighted keyword classifier
│   ├── extraction/llm_extractor.py    # Mistral-7B structured JSON extractor
│   ├── validation/validator.py        # Post-processing sanity checks
│   ├── storage/db.py                  # SQLite manager (3 tables)
│   ├── analytics/metrics.py           # Metrics computation
│   └── dashboard/dashboard.py         # Streamlit app (5 charts)
├── requirements.txt
└── .gitignore                         # Excludes credentials.json, token.json
```

## LLM Details

- **Model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Quantization**: 4-bit NF4 via `bitsandbytes` (~4 GB VRAM)
- **Decoding**: Deterministic (`temperature=0.1`, `do_sample=False`)
- **Output**: Strict JSON with 10 fields (company, role, CTC, CGPA, branches, dates, selections)
- **Resumable**: Skips already-extracted emails (UNIQUE constraint on email_id)

## Extracted JSON Schema

```json
{
  "company_name": "NVIDIA",
  "role": "ASIC Design Engineer",
  "ctc_lpa": 17.0,
  "cgpa_cutoff": 7.5,
  "eligibility_branches": "CSE, ECE",
  "registration_deadline": "2025-09-10",
  "test_date": "2025-09-15",
  "interview_date": null,
  "selection_count": 4,
  "total_openings": null
}
```
