# Campus Hiring NLP Analytics System (v2)

> Production-quality LLM-powered semantic extraction from university placement emails, with deduplication, validation audit trail, and an interactive dashboard.

## 🔗 Live Dashboard

**[View Interactive Dashboard](https://amish-03.github.io/Campus_Hiring_Dashboard/)** — deployed via GitHub Pages

## Pipeline Results

| Metric | Value |
|:---|:---|
| Total emails processed | 497 |
| Hiring emails classified | 475 |
| Unique campus drives | **129** |
| Unique companies | **102** |
| Average CTC | **8.5 LPA** |
| Median CTC | **6.5 LPA** |
| Highest CTC | **33.0 LPA** |
| Average CGPA cutoff | **7.2** |
| Total selections | **3,023** |
| Audit corrections | 123 |

## Architecture

```
Gmail API → SQLite (emails)
         → Mistral-7B LLM (4-bit, CUDA) → classify + extract → JSON retry (3 attempts)
         → Validator (audit trail)
         → SQLite (structured_hiring_data)
         → Deduplicator (group by company+role)
         → SQLite (drives)
         → Streamlit Dashboard (local)
         → Static HTML Dashboard (GitHub Pages)
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
pip install sentencepiece hf_xet
```

### Step 2: Run Full Pipeline

The 497 emails are already in the database. No Gmail credentials needed:

```bash
python -m src.main --extract-only
```

This runs **3 stages** automatically:
1. **LLM Classify + Extract** → ALL 497 emails sent to Mistral-7B, which decides hiring/non-hiring and extracts data in one pass (~15 min on RTX 3060)
2. **Validate** → CTC cap, CGPA range, date normalization, branch standardization
3. **Deduplicate** → merges multiple emails per company into one drive record

### Step 3: Launch Dashboard (Local)

```bash
python -m streamlit run src/dashboard/dashboard.py
```

### Step 4: Deploy to GitHub Pages

The static dashboard is already built in the `docs/` folder. To deploy:

1. Go to **Settings** → **Pages** in your GitHub repository
2. Under **Source**, select **Deploy from a branch**
3. Set the branch to `main` and folder to `/docs`
4. Click **Save**

The dashboard will be live at `https://<username>.github.io/Campus_Hiring_Dashboard/`

## CLI Reference

| Command | What it does |
|:---|:---|
| `python -m src.main` | Full pipeline (fetch + LLM classify/extract + deduplicate) |
| `python -m src.main --extract-only` | **Recommended** — extract from existing DB |
| `python -m src.main --extract-only --force` | Re-extract (clears previous results) |
| `python -m src.main --no-fetch` | Skip Gmail fetch |
| `python -m src.main --sample-eval` | Generate ground truth template (20 emails) |
| `python -m src.main --evaluate` | Run evaluation against annotated ground truth |
| `python -m streamlit run src/dashboard/dashboard.py` | Launch Streamlit dashboard |

## Project Structure

```
campus_hiring_nlp/
├── data/
│   └── campus_hiring.db              # 497 emails + 6 tables
├── docs/
│   ├── index.html                     # Static dashboard (GitHub Pages)
│   └── data.json                      # Exported drive data
├── src/
│   ├── main.py                        # 4-stage pipeline orchestrator
│   ├── models.py                      # 5 dataclasses (Email, Hiring, Drive, Audit)
│   ├── ingestion/gmail_api.py         # Gmail API fetcher
│   ├── extraction/llm_extractor.py    # Mistral-7B: classify + extract + retry
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
- **Classification**: LLM decides `is_hiring_email` (true/false) — no hard-coded rules
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
| **Audit** | Expandable validation audit log (Streamlit only) |

## Evaluation Framework

1. `--sample-eval` → generates `ground_truth.json` with 20 random emails
2. Manually annotate the `ground_truth` fields
3. `--evaluate` → computes field-level precision, recall, hallucination rate
