# Campus Hiring NLP Analytics System

> LLM-powered semantic extraction from university placement emails with a Streamlit analytics dashboard.

## Architecture

```
Gmail API → SQLite → Rule Classifier → LLM Extractor → Validator → Structured DB → Streamlit Dashboard
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Gmail credentials
- Go to [Google Cloud Console](https://console.cloud.google.com/)
- Create project → Enable Gmail API → Create OAuth 2.0 Client ID (Desktop App)
- Download as `credentials.json` and place in project root

### 3. Run the full pipeline
```bash
python -m src.main                  # Full: fetch + classify + LLM extract
python -m src.main --extract-only   # Re-extract without re-fetching
python -m src.main --no-fetch       # Classify + extract from existing DB
python -m src.main --force          # Force re-extraction (clear previous)
```

### 4. Launch dashboard
```bash
streamlit run src/dashboard/dashboard.py
```

## Project Structure

```
campus_hiring_nlp/
├── data/campus_hiring.db          # SQLite database
├── src/
│   ├── main.py                    # Pipeline orchestrator
│   ├── models.py                  # Data models
│   ├── ingestion/                 # Gmail API fetcher
│   ├── classifier/                # Rule-based hiring classifier
│   ├── extraction/                # LLM-based JSON extractor (Mistral-7B)
│   ├── validation/                # Post-processing & sanity checks
│   ├── storage/                   # SQLite database manager
│   ├── analytics/                 # Metrics computation
│   └── dashboard/                 # Streamlit analytics dashboard
├── requirements.txt
└── README.md
```

## LLM Model

- **Model**: Mistral-7B-Instruct-v0.3
- **Quantization**: 4-bit NF4 via bitsandbytes (~4 GB VRAM)
- **Decoding**: Deterministic (temperature=0.1)
- **Output**: Strict JSON with 10 fields

## Technical Requirements

- Python 3.10+
- CUDA-capable GPU with ≥8 GB VRAM
- Gmail API credentials (OAuth 2.0)
